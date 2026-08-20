"""Interactive runner for LITERATURE RETRIEVAL + FULL-TEXT ANALYSIS on a single gene.

Runs the front half of the pipeline only -- retrieve + cross-encoder + LLM curator-judge,
then full-text resolve + extract + merge (+ optional OpenAI review) -- and STOPS before the
CrewAI annotation phases. Reports RUNTIME and TOKEN USAGE for each half separately and combined,
merging the retrieval-side token_profiler totals with the partner extractor's subprocess usage.

Just run it and answer the prompts:

    conda run -n paperqa python run_analysis.py
    conda run -n paperqa python run_analysis.py GENE                    # skip the gene prompt
    conda run -n paperqa python run_analysis.py GENE --papers-dir /path/to/pdf_folder
    conda run -n paperqa python run_analysis.py GENE --no-full-text     # retrieval only

Prereqs: Neo4j (bolt 7687) and MongoDB (27017) running; ANTHROPIC_API_KEY (and OPENAI_API_KEY
for the review) in .env. The PDF-folder answer is remembered in data/user_config.json.
"""
import argparse
import asyncio
import os
import shutil
import socket
import sys
import time
from datetime import datetime

# Token profiling must be ON before ModelConfig builds any ChatAnthropic (the callback is
# attached at construction), so set it before importing anything that pulls the model in.
os.environ.setdefault("TOKEN_PROFILE", "1")

sys.path.append("reactome_llm")
sys.path.append("examples")

from dotenv import load_dotenv
load_dotenv()

import token_profiler
import fulltext_extractor
import FullTextResolver
from crewai_annotation_examples import build_annotators
from CrewAILiteratureAnnotator import AnnotationRequest, CrewAILiteratureAnnotator

_FT_LABEL = {"pdf": "PDF", "xml": "XML", "miss": "miss"}


class _StopAfterFrontHalf(Exception):
    """Sentinel: retrieval + full-text done; do not enter the CrewAI phases."""


# ---------------------------------------------------------------- prompts / validation
def prompt_gene(cli_gene):
    if cli_gene:
        return cli_gene.strip().upper()
    g = ""
    while not g:
        g = input("Gene symbol (e.g. SHANK3): ").strip()
    return g.upper()


def prompt_papers_dir(cli_dir, no_full_text):
    """Resolve the full-text PDF folder: CLI flag > saved config > interactive prompt.
    Validates the path and re-prompts until it's a real directory (or 'none' for PMC-only)."""
    if no_full_text:
        return None
    cfg = FullTextResolver.load_config()
    saved = cfg.get("papers_dir")

    if cli_dir is not None:
        chosen = cli_dir.strip() or None
    elif saved:
        ans = input(f"Full-text PDF folder [{saved}]\n"
                    f"  (Enter = keep, a new path, or 'none' for PMC-only): ").strip()
        chosen = saved if ans == "" else (None if ans.lower() == "none" else ans)
    else:
        ans = input("Full-text PDF folder (path, or 'none' for PMC-only): ").strip()
        chosen = None if ans.lower() in ("", "none") else ans

    # Re-prompt until valid. An interactive session can retry; a bad --papers-dir flag aborts.
    while chosen is not None and not os.path.isdir(os.path.expanduser(chosen)):
        if cli_dir is not None and not sys.stdin.isatty():
            sys.exit(f"[error] --papers-dir '{chosen}' is not a directory.")
        ans = input(f"  '{chosen}' is not a valid directory. Enter a valid path "
                    f"(or 'none' for PMC-only): ").strip()
        chosen = None if ans.lower() in ("", "none") else ans

    chosen = os.path.expanduser(chosen) if chosen else None
    cfg["papers_dir"] = chosen
    FullTextResolver.save_config(cfg)
    return chosen


def check_services():
    """Fail early with a clear message if Neo4j / Mongo aren't up (retrieval needs both)."""
    def _up(port):
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            return False
    down = [name for name, port in (("Neo4j (7687)", 7687), ("MongoDB (27017)", 27017)) if not _up(port)]
    if down:
        sys.exit(f"[error] required service(s) not reachable: {', '.join(down)}.\n"
                 f"        Start them, then re-run. (Neo4j: bin/neo4j start · Mongo: brew services start mongodb-community)")


# ---------------------------------------------------------------- reporting helpers
def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


def _fmt_secs(s):
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s" if m else f"{sec}s"


def main():
    ap = argparse.ArgumentParser(description="Literature retrieval + full-text analysis for one gene.")
    ap.add_argument("gene", nargs="?", help="Gene symbol (prompted if omitted).")
    ap.add_argument("--papers-dir", default=None, help="Full-text PDF folder (prompted if omitted).")
    ap.add_argument("--no-full-text", action="store_true", help="Retrieval only; skip full-text.")
    ap.add_argument("--max-papers", type=int, default=5, help="Papers the judge selects (default 5).")
    args = ap.parse_args()

    gene = prompt_gene(args.gene)
    papers_dir = prompt_papers_dir(args.papers_dir, args.no_full_text)
    check_services()

    token_profiler.reset()
    fulltext_extractor.reset_usage()
    _run_stamp = datetime.now().strftime("%Y%m%d_%H%M")  # per-run output folder suffix

    index = FullTextResolver.build_index(papers_dir) if papers_dir else {}
    print(f"\n{'=' * 74}\nRUN — {gene}   (full-text {'OFF' if args.no_full_text else 'ON'}, "
          f"max_papers={args.max_papers})\n{'=' * 74}")
    if papers_dir:
        print(f"PDF folder : {papers_dir}  ->  {len(index)} PDF(s) indexed")
    else:
        print("PDF folder : (none — full text via PMC only)" if not args.no_full_text
              else "PDF folder : (n/a — full text disabled)")

    # Timing hooks: stamp the boundary between retrieval and full-text, and the stop point.
    timing = {}
    orig_resolve = CrewAILiteratureAnnotator._resolve_and_extract_fulltext

    def timed_resolve(self, request):
        timing["ft_start"] = time.perf_counter()
        timing["retrieval_tokens"] = token_profiler.totals()  # snapshot before extraction
        return orig_resolve(self, request)

    async def stop_front_half(self, *a, **k):
        timing["end"] = time.perf_counter()
        raise _StopAfterFrontHalf("STOP_AFTER_FRONT_HALF")

    CrewAILiteratureAnnotator._resolve_and_extract_fulltext = timed_resolve
    CrewAILiteratureAnnotator._phase_1_literature_extraction = stop_front_half

    _, crewai = build_annotators(verbose=False)
    request = AnnotationRequest(
        gene=gene, papers=[], max_papers=args.max_papers,
        enable_full_text=not args.no_full_text, enable_literature_search=True,
        fulltext_index=index)

    timing["t0"] = time.perf_counter()
    try:
        asyncio.run(crewai.annotate_literature(request))
        print("\n!! Ran to completion without hitting the stop point.")
    except Exception as e:
        if "STOP_AFTER_FRONT_HALF" not in str(e) and not isinstance(e, _StopAfterFrontHalf):
            raise

    # ---- runtimes ----
    t0 = timing["t0"]
    end = timing.get("end", time.perf_counter())
    if "ft_start" in timing:
        retrieval_s = timing["ft_start"] - t0
        fulltext_s = end - timing["ft_start"]
    else:  # full text disabled -> everything up to the stop point is retrieval
        retrieval_s = end - t0
        fulltext_s = 0.0

    # ---- tokens ----
    retrieval_tok = timing.get("retrieval_tokens") or token_profiler.totals()
    fulltext_tok = fulltext_extractor.get_usage()

    ga = crewai.gene_annotator

    # ---- 1) retrieval ----
    print(f"\n{'-' * 74}\n1) RETRIEVAL\n{'-' * 74}")
    jp = (getattr(ga, "judged_papers", {}) or {}).get(gene, {})
    scored = jp.get("scored", {}) or {}
    selected = jp.get("papers", [])
    sel_mean = _mean([pp.get("score") for pp in scored.get("per_paper", [])
                      if str(pp.get("pmid")) in {str(p.get("pmid")) for p in selected}])
    print(f"candidates scored : {scored.get('n', 0)}   pool mean : {scored.get('mean_score')}")
    print(f"selected          : {len(selected)}   selected mean : {sel_mean}")
    print(f"PMIDs             : {', '.join(str(p.get('pmid')) for p in selected) or '(none)'}")

    # ---- 2) full text ----
    print(f"\n{'-' * 74}\n2) FULL-TEXT ANALYSIS\n{'-' * 74}")
    if args.no_full_text:
        print("(disabled)")
    else:
        manifest = (getattr(ga, "fulltext_manifest", {}) or {}).get(gene, {})
        n = {k: sum(1 for v in manifest.values() if v.get("source") == k) for k in ("pdf", "xml", "miss")}
        print(f"resolution        : {n['pdf']} PDF · {n['xml']} XML · {n['miss']} miss  (of {len(manifest)})")
        reactions = (getattr(ga, "fulltext_reactions", {}) or {}).get(gene, [])
        per_paper = (getattr(ga, "fulltext_per_paper", {}) or {}).get(gene, [])
        if reactions:
            ars = [r.get("annotation_result", r) for r in reactions]
            mc = [ar.get("merge_confidence") for ar in ars if isinstance(ar.get("merge_confidence"), (int, float))]
            print(f"reactions (all papers): {len(reactions)}   (per-paper merge; NOT cross-paper deduped)")
            print(f"extraction conf.      : mean {_mean([ar.get('confidence') for ar in ars])}")
            print(f"merge confidence      : mean {_mean(mc)}  min {round(min(mc), 3) if mc else None}")
        elif manifest and n["miss"] == len(manifest):
            # Every selected paper resolved to a miss -> there was simply no full text to read.
            # Expected, not a failure: full text needs a local PDF matching a selected paper, or
            # an open-access PMC copy of one.
            print("reactions             : 0")
            print(f"  -> no full text available: none of the {len(manifest)} selected PMIDs had a")
            print(f"     local PDF match or an open-access PMC copy, so nothing was extracted.")
            print(f"     Add PDFs for this gene to your folder, or test a gene with OA coverage.")
        else:
            print("reactions             : 0  (extraction produced nothing, or extractor module absent)")

        # Per-paper breakdown -- each paper independently extracted, merged, and OpenAI-reviewed,
        # so its score is a fair, own-text judgment of that paper's reactions.
        if per_paper:
            print(f"\n  per-paper (each independently reviewed against its own text):")
            print(f"    {'paper':<12}{'source':<7}{'extracted':>10}{'merged':>8}{'review':>9}")
            print("    " + "-" * 46)
            for p in per_paper:
                sc = p.get("review_score")
                print(f"    {str(p.get('pmid')):<12}{str(p.get('source')):<7}"
                      f"{p.get('n_extracted', 0):>10}{p.get('n_merged', 0):>8}"
                      f"{(str(sc) + '/10' if sc is not None else 'n/a'):>9}")

            # Collect this run's per-paper artifacts (merged JSON + OpenAI review) into one folder.
            run_dir = os.path.join("results", f"{gene.lower()}_run_{_run_stamp}")
            os.makedirs(run_dir, exist_ok=True)
            copied = 0
            for p in per_paper:
                for key in ("merged_path", "review_path", "extraction_path"):
                    src = p.get(key)
                    if src and os.path.isfile(src):
                        shutil.copy2(src, os.path.join(run_dir, os.path.basename(src)))
                        copied += 1
            print(f"\n  saved this run's per-paper merged JSON + OpenAI review files -> {run_dir}/"
                  f"  ({copied} files)")

    # ---- 3) runtime + tokens + cost (the combined tracker) ----
    # Rough cost. Retrieval runs on sonnet-4-6, full-text on sonnet-5 (+ OpenAI review); exact
    # sonnet-5 / OpenAI rates aren't published in this repo, so both halves are estimated at
    # sonnet-tier $3 in / $15 out per MTok (same constants token_profiler uses). Adjust if needed.
    _USD_IN, _USD_OUT = 3.0, 15.0
    def _cost(tin, tout):
        return (tin / 1e6) * _USD_IN + (tout / 1e6) * _USD_OUT

    print(f"\n{'=' * 78}\nRUNTIME + TOKENS + COST\n{'=' * 78}")
    print(f"  {'phase':<12}{'runtime':<10}{'in':>12}{'out':>11}{'total':>12}{'calls':>7}{'$cost':>10}")
    print("  " + "-" * 74)
    r_in, r_out = retrieval_tok["input"], retrieval_tok["output"]
    f_in, f_out = fulltext_tok["input"], fulltext_tok["output"]
    print(f"  {'retrieval':<12}{_fmt_secs(retrieval_s):<10}{r_in:>12,}{r_out:>11,}{r_in + r_out:>12,}"
          f"{retrieval_tok['calls']:>7}{'$' + format(_cost(r_in, r_out), '.2f'):>10}")
    if not args.no_full_text:
        print(f"  {'full-text':<12}{_fmt_secs(fulltext_s):<10}{f_in:>12,}{f_out:>11,}{f_in + f_out:>12,}"
              f"{fulltext_tok['calls']:>7}{'$' + format(_cost(f_in, f_out), '.2f'):>10}")
    tot_in, tot_out = r_in + f_in, r_out + f_out
    print("  " + "-" * 74)
    print(f"  {'TOTAL':<12}{_fmt_secs(retrieval_s + fulltext_s):<10}{tot_in:>12,}{tot_out:>11,}{tot_in + tot_out:>12,}"
          f"{retrieval_tok['calls'] + fulltext_tok['calls']:>7}{'$' + format(_cost(tot_in, tot_out), '.2f'):>10}")
    if fulltext_tok["cache_read"] or fulltext_tok["cache_write"]:
        print(f"  (full-text cache: read {fulltext_tok['cache_read']:,} / write {fulltext_tok['cache_write']:,})")
    print(f"\nNote: retrieval tokens = your token_profiler (in-process ChatAnthropic); full-text "
          f"tokens = partner extractor subprocesses. Cost is estimated at sonnet-tier $3/$15 per MTok.")

    # Detailed per-call-site retrieval breakdown + CSV. emit_report() both logs AND prints the
    # same block, so raise the token_profiler logger to WARNING around it to drop the duplicate
    # (logged) copy and keep only the clean printed one.
    import logging
    _tp_log = logging.getLogger("token_profiler")
    _prev_level = _tp_log.level
    _tp_log.setLevel(logging.WARNING)
    try:
        csv_path = token_profiler.emit_report(gene)
    finally:
        _tp_log.setLevel(_prev_level)
    if csv_path:
        print(f"Retrieval token breakdown CSV: {csv_path}")


if __name__ == "__main__":
    main()
