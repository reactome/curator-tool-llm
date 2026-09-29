"""Bridge from the retrieval manifest to the partner's full-text extraction + merge.

Phase 1 (deterministic) hands this a FullTextResolver manifest {pmid: {source, path}}.
This drives the partner's tested CLIs -- run_extraction.py then run_merge.py -- and
returns the merged reaction dicts for the gene.

Why subprocess and not import: run_merge.py (and run_review.py) execute their driver on
import (argparse at module top level), so they are not importable. Shelling out to the
scripts as written touches none of the partner's files, so re-pulling her branch stays a
clean 3-way merge even as she keeps developing the merge/scoring logic.

Route:
  1. one run_extraction call PER non-miss spec, run sequentially (one paper at a time) --
     each writes its own results/<stem>_extraction.json. xml -> PMID (PubMedFetcher hits
     data/fulltext_cache with no refetch); pdf -> absolute path.
  2. once all papers are extracted, concatenate the per-paper reaction arrays into
     results/<gene>_extraction.json -- the single-file-per-gene input run_merge expects. Merge
     is inherently cross-paper (it dedupes reactions reported in more than one paper), so it
     runs after every paper is done.
  3. one run_merge call -> results/<gene>_merged.json.
  4. read that back as the list of merged reactions.
  5. (optional, review=True) one run_review call -> results/<gene>_review.md. This is the
     OpenAI second-opinion judge (gpt-5.6-luna, needs OPENAI_API_KEY): a different model
     than the Claude extractor/merger reads the same material and scores whether the run
     was done well. Purely a side-effect report for the curator -- the returned reactions
     are unchanged whether or not review runs.

Extraction/merge are Claude judging its own work; the review step is the cross-model check.

Every failure degrades to "fewer/no full-text reactions" (or "no review") and is logged;
the caller proceeds on whatever else it assembled rather than crashing the annotation.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

logger = logging.getLogger(__name__)

# Per-paper mode runs a FULL independent pipeline (extract -> merge -> review) for each paper,
# fanned out across papers. Capped low on purpose: run_merge already fans out to 12 concurrent
# LLM calls internally (and run_review's sweep to 8), so N papers in flight = up to N*12
# concurrent requests. Cap 3 -> ~36 peak, which stays under typical Anthropic rate limits.
# Raise it on a high API tier; lower it if you see 429s. (extract_and_merge's cross-paper mode
# still runs papers one at a time.)
_MAX_PAPER_WORKERS = 3
_SCORE_RE = re.compile(r"SCORE:\s*([\d.]+)")

# ---- Full-text token accounting -------------------------------------------------------
# The partner's scripts (run_extraction / run_merge / run_review) each track their own token
# usage and print it to stdout. Because we drive them as subprocesses, that usage is only in
# their captured stdout -- so we parse it here and accumulate into one process-global tally
# that run_analysis.py reads to combine with the retrieval-side token_profiler totals. Formats:
#   run_extraction / run_merge:  "N call(s), tokens in X / out Y[, cache rZ/wW]"
#   run_review:                  "[usage] tokens in X / out Y"
_FT_USAGE = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
# Match a number with optional thousands separators STRICTLY (\d{1,3}(,\d{3})*) rather than
# [\d,]+ -- the latter greedily swallows the trailing comma of "out 900," and then the optional
# ", cache ..." group can never match.
_NUM = r"\d{1,3}(?:,\d{3})*"
_USAGE_RE = re.compile(
    rf"({_NUM}) call\(s\), tokens in ({_NUM}) / out ({_NUM})"
    rf"(?:, cache r({_NUM})/w({_NUM}))?")
_REVIEW_USAGE_RE = re.compile(rf"\[usage\] tokens in ({_NUM}) / out ({_NUM})")


def reset_usage() -> None:
    """Zero the full-text token tally at the start of a run."""
    for k in _FT_USAGE:
        _FT_USAGE[k] = 0


def get_usage() -> dict:
    """Accumulated full-text (partner-extractor) token usage across all subprocesses."""
    return dict(_FT_USAGE)


def _int(s):
    return int(s.replace(",", "")) if s else 0


def _accumulate_usage(stdout: str) -> None:
    """Parse a subprocess's stdout and add its FINAL token counts to _FT_USAGE.

    run_extraction/run_merge reprint a cumulative usage line per chunk -> take the LAST match so
    running totals aren't double-counted. run_review uses its own one-line format."""
    if not stdout:
        return
    m = _USAGE_RE.findall(stdout)
    if m:
        calls, tin, tout, cr, cw = m[-1]
        _FT_USAGE["calls"] += _int(calls)
        _FT_USAGE["input"] += _int(tin)
        _FT_USAGE["output"] += _int(tout)
        _FT_USAGE["cache_read"] += _int(cr)
        _FT_USAGE["cache_write"] += _int(cw)
    rev = _REVIEW_USAGE_RE.findall(stdout)
    if rev:
        tin, tout = rev[-1]
        _FT_USAGE["calls"] += 1
        _FT_USAGE["input"] += _int(tin)
        _FT_USAGE["output"] += _int(tout)

# This module sits at the repo root beside run_extraction.py / run_merge.py.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
_REACTOME_LLM = os.path.join(PROJECT_ROOT, "reactome_llm")
if _REACTOME_LLM not in sys.path:
    sys.path.insert(0, _REACTOME_LLM)
# output_stem is the partner's own filename rule; reusing it means we predict exactly the
# files run_extraction will write instead of globbing and risking stale matches. PAPERS_DIR is
# the folder load_source resolves a bare PDF filename against (data/fulltext_pdf).
from PubMedFetcher import output_stem, PAPERS_DIR  # noqa: E402


def _cache_local_pdf(spec: dict) -> None:
    """Copy a matched local PDF into PAPERS_DIR (data/fulltext_pdf) so run_review can find it.

    Extraction uses the PDF's absolute path directly and works fine. But run_review re-derives
    the source from the merged file as a BARE FILENAME (e.g. "PINK1.pdf") and load_source
    resolves that against PAPERS_DIR -- so without a copy there, the local-PDF paper's review
    fails ('no such PDF: data/fulltext_pdf/PINK1.pdf') and its score comes back n/a. Caching the
    file here fixes that. Best-effort; a copy failure just means that paper stays unreviewed."""
    src = spec.get("spec")
    if spec.get("tag") != "localpdf" or not src or not os.path.isfile(src):
        return
    try:
        os.makedirs(PAPERS_DIR, exist_ok=True)
        dest = os.path.join(PAPERS_DIR, os.path.basename(src))
        if not os.path.exists(dest):
            shutil.copy2(src, dest)
            logger.info("full-text: cached local PDF -> %s", dest)
    except Exception as e:
        logger.warning("full-text: could not cache local PDF %s: %s", src, e)


def _manifest_specs(manifest: Dict[str, dict], gene: str) -> List[dict]:
    """One spec per non-miss manifest entry, tagged with the file run_extraction will write.

    xml -> the PMID (load_source then reads data/fulltext_cache, no NCBI round-trip);
    pdf -> the absolute PDF path (load_source accepts an absolute spec);
    miss -> dropped (the manifest is authoritative on misses).
    """
    specs = []
    for pmid, entry in (manifest or {}).items():
        source = (entry or {}).get("source")
        if source == "xml":
            spec = str(pmid)
        elif source == "pdf":
            spec = entry.get("path")
        else:
            continue
        if not spec:
            continue
        stem = output_stem(spec, gene=gene)
        # Local PDFs get a "localpdf" tag on their output filename. This both labels them
        # clearly AND avoids a collision: output_stem('<GENE>.pdf', <GENE>) collapses to just
        # "<gene>" (e.g. PINK1.pdf -> "pink1"), which would otherwise equal the combined-file
        # name results/<gene>_extraction.json and get overwritten in step 2/3. run_extraction's
        # --tag appends the suffix, so the file becomes results/<stem>_localpdf_extraction.json.
        tag = "localpdf" if source == "pdf" else None
        label = f"{stem}_{tag}" if tag else stem
        specs.append({
            "pmid": str(pmid),
            "spec": spec,
            "tag": tag,
            "extraction_path": os.path.join(RESULTS_DIR, f"{label}_extraction.json"),
        })
    return specs


def _run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    logger.info("full-text: running %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=timeout,
                          capture_output=True, text=True)
    _accumulate_usage(proc.stdout)  # harvest the partner script's token usage from its stdout
    return proc


def _run_review(gene: str, timeout: int) -> None:
    """Step 5: OpenAI cross-model second opinion on results/<gene>_merged.json.

    Writes results/<gene>_review.md. Pure side effect (a curator-facing report) -- never
    affects the reactions extract_and_merge returns. Degrades on any failure, including a
    missing OPENAI_API_KEY (run_review.py exits non-zero with a clear message).
    """
    rev = _run([sys.executable, "run_review.py", gene.lower()], timeout)
    if rev.returncode != 0:
        logger.warning("run_review exited %s for %s: %s -- no review written",
                       rev.returncode, gene, (rev.stderr or "")[-800:])
        return
    logger.info("full-text review for %s -> results/%s_review.md", gene, gene.lower())


def extract_and_merge(manifest: Dict[str, dict], gene: str, timeout: int = 1800,
                      review: bool = True) -> List[dict]:
    """Manifest -> partner extraction + merge (+ optional OpenAI review) -> merged reaction
    dicts for `gene`.

    Returns [] when there is nothing to extract; returns the unmerged reactions if merge
    fails but extraction succeeded. With review=True, also runs the OpenAI second-opinion
    judge over the merged file, writing results/<gene>_review.md (best-effort; the returned
    reactions are identical whether or not review runs). Never raises -- failures are logged
    and degraded.
    """
    specs = _manifest_specs(manifest, gene)
    if not specs:
        logger.info("full-text: no non-miss papers to extract for %s", gene)
        return []

    py = sys.executable
    gene_l = gene.lower()

    # 1) Extract each paper in its own subprocess, ONE AT A TIME (sequential). Each writes its
    #    own results/<stem>_extraction.json; --overwrite makes a re-run idempotent. A failing
    #    paper just leaves no file; step 2 skips it.
    logger.info("full-text: extracting %d paper(s) for %s (sequential, one at a time)",
                len(specs), gene)
    for i, spec in enumerate(specs, 1):
        logger.info("full-text: paper %d/%d (%s)", i, len(specs), spec["pmid"])
        cmd = [py, "run_extraction.py", spec["spec"], "--gene", gene, "--overwrite"]
        if spec.get("tag"):
            cmd += ["--tag", spec["tag"]]
        r = _run(cmd, timeout)
        if r.returncode != 0:
            logger.warning("run_extraction exited %s for %s (%s): %s",
                           r.returncode, gene, spec["pmid"], (r.stderr or "")[-800:])

    # 2) Concatenate the per-paper reaction arrays (a paper that failed simply has no file).
    combined: List[dict] = []
    for s in specs:
        path = s["extraction_path"]
        if not os.path.isfile(path):
            continue
        try:
            combined.extend(json.load(open(path)))
        except Exception as e:
            logger.warning("full-text: could not read %s: %s", path, e)
    if not combined:
        logger.info("full-text: extraction produced no reactions for %s", gene)
        return []

    # 3) Write the single-file-per-gene input run_merge expects.
    gene_extraction = os.path.join(RESULTS_DIR, f"{gene_l}_extraction.json")
    with open(gene_extraction, "w") as f:
        json.dump(combined, f, indent=2)

    # 4) Merge. On failure, fall back to the unmerged reactions rather than losing them.
    mrg = _run([py, "run_merge.py", gene_extraction], timeout)
    if mrg.returncode != 0:
        logger.warning("run_merge exited %s for %s: %s -- returning unmerged reactions",
                       mrg.returncode, gene, (mrg.stderr or "")[-800:])
        return combined

    merged_path = gene_extraction.replace("_extraction", "_merged")
    if not os.path.isfile(merged_path):
        logger.warning("full-text: run_merge wrote no %s -- returning unmerged", merged_path)
        return combined
    try:
        merged = json.load(open(merged_path))
        logger.info("full-text for %s: %d paper(s) -> %d extracted -> %d merged reaction(s)",
                    gene, len(specs), len(combined), len(merged))
    except Exception as e:
        logger.warning("full-text: could not read merged %s: %s -- returning unmerged",
                       merged_path, e)
        return combined

    # 5) Cross-model OpenAI review of the merged run (side-effect report only).
    if review:
        _run_review(gene, timeout)
    return merged


def _read_json(path):
    try:
        return json.load(open(path)) if os.path.isfile(path) else []
    except Exception as e:
        logger.warning("full-text: could not read %s: %s", path, e)
        return []


def _review_score(review_path):
    """Pull the OpenAI reviewer's SCORE (0-10) out of a *_review.md, or None."""
    if not os.path.isfile(review_path):
        return None
    m = _SCORE_RE.search(open(review_path).read())
    return float(m.group(1)) if m else None


def _pipeline_one_paper(spec: dict, gene: str, timeout: int, review: bool) -> dict:
    """Full INDEPENDENT pipeline for ONE paper: extract -> merge -> (optional) OpenAI review.

    No cross-paper merge -- each paper is judged and merged on its own, so its review score is
    fair (the reviewer sees only that paper's text). Returns a per-paper result dict; never raises.
    """
    py = sys.executable
    result = {"pmid": spec["pmid"], "source": ("pdf" if spec.get("tag") == "localpdf" else "xml"),
              "spec": spec["spec"], "n_extracted": 0, "n_merged": 0, "reactions": [],
              "review_score": None, "review_path": None,
              "extraction_path": None, "merged_path": None, "ok": False}

    # cache a matched local PDF into PAPERS_DIR so the review step can find it by filename
    _cache_local_pdf(spec)

    # extract
    cmd = [py, "run_extraction.py", spec["spec"], "--gene", gene, "--overwrite"]
    if spec.get("tag"):
        cmd += ["--tag", spec["tag"]]
    if _run(cmd, timeout).returncode != 0:
        logger.warning("full-text: extraction failed for %s (%s)", gene, spec["pmid"])
        return result
    extraction_path = spec["extraction_path"]
    result["extraction_path"] = extraction_path
    extracted = _read_json(extraction_path)
    result["n_extracted"] = len(extracted)
    if not extracted:
        return result
    result["reactions"] = extracted  # fallback if merge fails

    # merge (this ONE paper's reactions; run_merge writes <stem>_merged.json)
    if _run([py, "run_merge.py", extraction_path], timeout).returncode == 0:
        merged_path = extraction_path.replace("_extraction", "_merged")
        merged = _read_json(merged_path)
        if merged:
            result["reactions"] = merged
            result["n_merged"] = len(merged)
            result["merged_path"] = merged_path
            # review THIS paper's merged file (reviewer re-derives this paper's own chunks)
            if review:
                if _run([py, "run_review.py", merged_path], timeout).returncode == 0:
                    review_path = os.path.splitext(merged_path)[0].replace("_merged", "") + "_review.md"
                    result["review_path"] = review_path if os.path.isfile(review_path) else None
                    result["review_score"] = _review_score(review_path)
    result["ok"] = True
    return result


def extract_abstracts_for_misses(manifest: Dict[str, dict], gene: str,
                                 abstracts_by_pmid: Dict[str, str] = None,
                                 max_workers: int = 4, review: bool = True) -> List[dict]:
    """Abstract fallback: for every manifest entry marked ``miss`` (no full text), extract
    reactions from its ABSTRACT via abstract_extractor (one bounded LLM call each, run in
    parallel and capped). Returns one per-paper result dict per miss paper, shaped like
    extract_review_per_paper's results but with source/provenance "abstract" -- so the caller
    can concatenate full-text and abstract reactions into one stream.

    ``abstracts_by_pmid`` supplies the abstract text already in hand from retrieval (keyed by
    PMID); abstract_extractor falls back to its own cache / a PubMed efetch when it's absent.
    Token usage from these in-process calls is folded into the shared full-text tally so
    run_analysis reports one combined figure. Never raises -- a paper that fails yields no
    reactions.
    """
    miss = [str(pmid) for pmid, entry in (manifest or {}).items()
            if (entry or {}).get("source") == "miss"]
    if not miss:
        return []

    import abstract_extractor
    abstract_extractor.reset_usage()
    abstracts_by_pmid = abstracts_by_pmid or {}

    workers = max(1, min(max_workers, len(miss)))
    logger.info("abstract fallback for %s -- %d miss paper(s), %d worker(s)",
                gene, len(miss), workers)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(abstract_extractor.extract_abstract, p, gene,
                            abstracts_by_pmid.get(p), review): p for p in miss}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            logger.info("abstract [%d/%d] %s: %d reaction(s) from abstract",
                        done, len(miss), r["pmid"], r["n_extracted"])
            results.append(r)

    # Fold the abstract-extractor's in-process token usage into the shared full-text tally.
    u = abstract_extractor.get_usage()
    _FT_USAGE["calls"] += u.get("calls", 0)
    _FT_USAGE["input"] += u.get("input", 0)
    _FT_USAGE["output"] += u.get("output", 0)
    _FT_USAGE["cache_read"] += u.get("cache_read", 0)
    _FT_USAGE["cache_write"] += u.get("cache_write", 0)

    order = {p: i for i, p in enumerate(miss)}
    results.sort(key=lambda r: order.get(r["pmid"], 0))
    return results


def extract_review_per_paper(manifest: Dict[str, dict], gene: str, timeout: int = 1800,
                             review: bool = True, max_workers: int = _MAX_PAPER_WORKERS) -> List[dict]:
    """Per-paper mode: run extract -> merge -> review INDEPENDENTLY for each paper, in parallel
    (capped). Prints a one-line update as each paper finishes. Returns one result dict per paper
    (pmid, source, n_extracted, n_merged, reactions, review_score, review_path).

    NOTE: no cross-paper dedup -- the same reaction reported by two papers appears twice across
    results. Good for per-paper trust scoring; the final annotation still needs a dedup pass.
    """
    specs = _manifest_specs(manifest, gene)
    if not specs:
        logger.info("full-text: no non-miss papers to extract for %s", gene)
        return []

    workers = max(1, min(max_workers, len(specs)))
    logger.info("full-text: per-paper pipeline for %s -- %d paper(s), %d worker(s) "
                "(each merge is 12-way concurrent internally)", gene, len(specs), workers)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_pipeline_one_paper, s, gene, timeout, review): s for s in specs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            score = r["review_score"]
            logger.info("full-text [%d/%d] %s (%s): %d extracted -> %d merged | review %s",
                        done, len(specs), r["pmid"], r["source"], r["n_extracted"], r["n_merged"],
                        f"{score}/10" if score is not None else "n/a")
            results.append(r)
    # Stable order by original selection for a tidy summary.
    order = {s["pmid"]: i for i, s in enumerate(specs)}
    results.sort(key=lambda r: order.get(r["pmid"], 0))
    return results
