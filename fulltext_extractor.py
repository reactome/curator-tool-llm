"""Bridge from the retrieval manifest to the full-text extraction + merge (+ review) of each paper.

The gene-first pipeline hands this a FullTextResolver manifest {pmid: {source, path}} and gets back one
result dict per paper: its merged reactions, review score and token usage. The per-paper work itself is
curator_llm.services.extraction.SubprocessExtractor (run_extraction.py -> run_merge.py -> run_review.py,
each in its own process: run_merge/run_review run their driver at import and keep state in module
globals, so they cannot be imported or run concurrently in one process). This module only maps manifest
entries to that call, fans papers out in parallel, caches a local PDF where run_review expects to find it,
and keeps the process-wide token tally that run_curator.py reads.

Every failure degrades to "fewer/no full-text reactions" (or "no review") and is logged; the caller
proceeds on whatever else it assembled rather than crashing the annotation.
"""
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

logger = logging.getLogger(__name__)

# Per-paper mode runs a FULL independent pipeline (extract -> merge -> review) for each paper,
# fanned out across papers. Capped low on purpose: run_merge already fans out to 12 concurrent
# LLM calls internally (and run_review's sweep to 8), so N papers in flight = up to N*12
# concurrent requests. Cap 3 -> ~36 peak, which stays under typical Anthropic rate limits.
# Raise it on a high API tier; lower it if you see 429s.
_MAX_PAPER_WORKERS = 3

# ---- Full-text token accounting -------------------------------------------------------
# Each step reports its token usage on stdout; SubprocessExtractor parses it into the result, and
# _pipeline_one_paper adds it to this process-global tally, which run_curator.py combines with the
# retrieval-side token_profiler totals.
_FT_USAGE = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0}


def reset_usage() -> None:
    """Zero the full-text token tally at the start of a run."""
    for k in _FT_USAGE:
        _FT_USAGE[k] = 0


def get_usage() -> dict:
    """Accumulated full-text token usage across all per-paper pipelines."""
    return dict(_FT_USAGE)


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


def _extractor(timeout: int, review: bool):
    """The per-paper extractor (a seam, so tests can substitute one without subprocesses)."""
    from curator_llm.services.extraction import SubprocessExtractor
    return SubprocessExtractor(root=PROJECT_ROOT, results_dir=RESULTS_DIR, timeout=timeout, review=review)


def _pipeline_one_paper(spec: dict, gene: str, timeout: int, review: bool) -> dict:
    """Full INDEPENDENT pipeline for ONE paper: extract -> merge -> (optional) OpenAI review.

    No cross-paper merge -- each paper is judged and merged on its own, so its review score is
    fair (the reviewer sees only that paper's text). Returns a per-paper result dict; never raises.
    """
    is_pdf = spec.get("tag") == "localpdf"
    result = {"pmid": spec["pmid"], "source": "pdf" if is_pdf else "xml", "spec": spec["spec"],
              "n_extracted": 0, "n_merged": 0, "reactions": [], "review_score": None, "review_path": None,
              "extraction_path": None, "merged_path": None, "ok": False}
    # cache a matched local PDF into PAPERS_DIR so the review step can find it by filename
    _cache_local_pdf(spec)
    res = _extractor(timeout, review).extract(None if is_pdf else spec["spec"],
                                              spec["spec"] if is_pdf else None, gene)
    for k, v in (("calls", "calls"), ("input", "input"), ("output", "output"),
                 ("cache_read", "cache_read"), ("cache_write", "cache_write")):
        _FT_USAGE[k] += res.usage.get(v, 0)
    for w in res.warnings:
        logger.warning("full-text %s (%s): %s", gene, spec["pmid"], w)
    if res.error:
        logger.warning("full-text: %s for %s (%s)", res.error, gene, spec["pmid"])
    result.update(n_extracted=res.n_extracted, n_merged=res.n_merged, reactions=res.reactions,
                  review_score=res.review_score, review_path=res.review_path,
                  extraction_path=res.extraction_path, merged_path=res.merged_path, ok=res.ok)
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
