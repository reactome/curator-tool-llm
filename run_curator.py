"""run_curator.py — kickoff for the 3-agent Curator pipeline (Curator -> Reviewer -> QA).

Runs all three phases for ONE OR MORE genes in a single invocation:

    conda run -n paperqa python run_curator.py SHANK3
    conda run -n paperqa python run_curator.py SHANK3 TANC1 CTTNBP2
    conda run -n paperqa python run_curator.py SHANK3 --papers-dir /path/to/pdfs
    conda run -n paperqa python run_curator.py SHANK3 --no-full-text --no-llm-review   # cheap smoke test

Per gene (all synchronous — no CrewAI, no event loop):

    ┌────────────── adjustment (Reviewer's proposed knobs) ──────────────┐
    ▼                                                                     │
  Curator.run(gene, adjustment)  ──▶  Reviewer.review(result)  ──sufficient?──┐
   (deterministic: retrieve ->        (LLM: decide + adjust)          no       │
    place -> full text)                                     ─────────────────-─┘
                                                    │ yes / give_up / cap hit
                                                    ▼
                     QA (agent 3): build_instances -> schema/consistency + LLM review -> verdict

The retry cap lives HERE (the `for attempt in range(max_attempts)` loop), so the Reviewer can
only PROPOSE "retry with X" — the loop decides when to stop.

Prereqs: Neo4j (7687) + MongoDB (27017) up; ANTHROPIC_API_KEY (and OPENAI_API_KEY for the
full-text review) in .env.
"""

import argparse
import json
import os
import socket
import sys
import time

os.environ.setdefault("TOKEN_PROFILE", "1")
sys.path.append("reactome_llm")

from dotenv import load_dotenv
load_dotenv()

import token_profiler
import fulltext_extractor
import FullTextResolver
import ReactomeNeo4jUtils as neo4j_utils
from ReactomeCurator import ReactomeCurator
from ReactomeReviewer import ReactomeReviewer
from ReactomeQA import ReactomeQA


def _usage_snapshot(start_time):
    """Combined token spend + wall-clock for one gene: the LangChain-side calls (retrieval judge,
    convert, QA) from token_profiler PLUS the full-text extraction subprocesses from
    fulltext_extractor — otherwise the cost undercounts (extraction is most of it). Cost is a
    list-price estimate (claude-sonnet-4-6 $3/$15 per Mtok); it ignores cache discounts, so it's
    an upper-ish bound."""
    retr = token_profiler.totals()
    ft = fulltext_extractor.get_usage()
    tin = retr.get("input", 0) + ft.get("input", 0)
    tout = retr.get("output", 0) + ft.get("output", 0)
    return {"in": tin, "out": tout,
            "cost": tin / 1e6 * 3.0 + tout / 1e6 * 15.0,
            "min": (time.time() - start_time) / 60.0}


def classify_gene(gene, result):
    """Human-facing gene classification, from live graph state + placement:
      - has_data: does the gene ALREADY have curated Reactome pathways? (annotated vs cold-start)
      - gate:     did FI-partner placement clear the confidence bar? (pass vs fail)
    These two axes are what makes a QA 'FAIL' interpretable — see _print_result_explanation."""
    n_existing = None
    try:
        n_existing = len(neo4j_utils.query_pathways_for_gene(gene) or [])
    except Exception:
        pass  # Neo4j unreachable -> has_data stays unknown
    pl = (result.placement if result else {}) or {}
    return {
        "has_data": None if n_existing is None else n_existing > 0,
        "n_existing_pathways": n_existing,
        "gate": "pass" if pl.get("confident") else "fail",
        "predicted_pathway": pl.get("predicted_pathway"),
    }


def _print_result_explanation(gene, cls, n_reactions, qa_verdict, qa_score, usage=None):
    """One compact block: what kind of gene this was, and the one-line status a curator needs."""
    print(f"\n{'=' * 70}\nRESULT — {gene}\n{'=' * 70}")
    if cls["has_data"] is True:
        print(f"  Gene type : ANNOTATED (has-data) — already has {cls['n_existing_pathways']} "
              f"curated Reactome pathway(s).")
    elif cls["has_data"] is False:
        print("  Gene type : UNANNOTATED (cold-start) — no curated Reactome pathways yet.")
    else:
        print("  Gene type : UNKNOWN — could not query Reactome for existing pathways.")
    if cls["gate"] == "pass":
        print(f'  Placement : gate PASS — confident placement under "{cls["predicted_pathway"]}".')
    else:
        print("  Placement : gate FAIL — no confident FI-partner placement; any reactions are unplaced.")
    print(f"  Outcome   : {n_reactions} reaction(s) drafted"
          + (f"; QA verdict {qa_verdict.upper()}"
             + (f" (qa_score {qa_score})" if qa_score is not None else "") if qa_verdict else "") + ".")
    if n_reactions == 0:
        print("  Status    : no draft produced -> needs manual curation.")
    else:
        print("  Status    : DRAFT -> needs curator review before acceptance (see the QA flags above).")
    if usage:
        print(f"  Cost/time : ~${usage['cost']:.2f} est (list price) · {usage['in'] // 1000}k in + "
              f"{usage['out'] // 1000}k out tokens · {usage['min']:.1f} min")


def check_services():
    def _up(port):
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            return False
    down = [n for n, p in (("Neo4j (7687)", 7687), ("MongoDB (27017)", 27017)) if not _up(p)]
    if down:
        sys.exit(f"[error] required service(s) not reachable: {', '.join(down)}. Start them and re-run.")


def _print_attempt(attempt, result):
    """Show the CURATOR agent and the return of each of its 4 tools for this attempt."""
    r, ft, pl = result.retrieval, result.fulltext, result.placement
    c = ft.get("counts", {})
    print(f"\n{'─' * 70}\nCURATOR AGENT — attempt {attempt + 1}   "
          f"(adjustment applied: {result.adjustment_applied or 'none'})\n{'─' * 70}")
    print("  called 4 tools:")
    print(f"    [Tool 1] pathway placement    -> {pl.get('predicted_pathway')}  "
          f"(confident={pl.get('confident')}, status={pl.get('status')})")
    if r.get("papers_only"):
        print(f"    [Tool 2] literature retrieval -> SKIPPED (papers-only) — using "
              f"{len(r.get('papers', []))} curator-supplied paper(s)")
    else:
        print(f"    [Tool 2] literature retrieval -> pool {r.get('pool_size')} -> {r.get('candidate_pool')} "
              f"candidates -> {len(r.get('papers', []))} selected  (mean {r.get('mean_score')}, "
              f"dropped {r.get('dropped_below_threshold')})")
    print(f"               PMIDs: {', '.join(str(p.get('pmid')) for p in r.get('papers', [])) or '(none)'}")
    print(f"    [Tool 3] full-text resolver   -> {c.get('pdf', 0)} PDF · {c.get('xml', 0)} XML · "
          f"{c.get('miss', 0)} miss")
    print(f"    [Tool 4] full-text analysis   -> {len(ft.get('reactions', []))} reaction(s) extracted")


def _print_verdict(verdict):
    print(f"\n  REVIEWER AGENT -> {verdict.decision.upper()}: {verdict.reason}")
    if verdict.adjustment:
        print(f"                   next adjustment -> {verdict.adjustment}")


def _n_reactions(result):
    return len((result.fulltext or {}).get("reactions") or [])


def resolve_papers_dir(cli_dir, no_full_text):
    """Full-text PDF folder: --papers-dir wins; else PROMPT interactively (defaulting to the saved
    folder); else, non-interactive (e.g. background), reuse the saved folder. 'none' = PMC-only.
    Remembers the choice in data/user_config.json (shared with run_analysis)."""
    if no_full_text:
        return None
    cfg = FullTextResolver.load_config()
    saved = cfg.get("papers_dir")
    if cli_dir is not None:
        chosen = cli_dir.strip() or None
    elif sys.stdin.isatty():
        if saved:
            ans = input(f"Full-text PDF folder [{saved}]\n"
                        f"  (Enter = keep · a new path · 'none' for PMC-only): ").strip()
            chosen = saved if ans == "" else (None if ans.lower() == "none" else ans)
        else:
            ans = input("Full-text PDF folder (path, or 'none' for PMC-only): ").strip()
            chosen = None if ans.lower() in ("", "none") else ans
    else:
        chosen = saved  # non-interactive -> reuse the saved folder
    if chosen:
        chosen = os.path.expanduser(chosen)
        if not os.path.isdir(chosen):
            print(f"[warn] '{chosen}' is not a directory — full text via PMC only.")
            chosen = None
    cfg["papers_dir"] = chosen
    FullTextResolver.save_config(cfg)
    return chosen


def run_one(gene, curator, reviewer, qa, args):
    """Run the full 3-phase pipeline for one gene. Returns a one-line summary dict for the batch table."""
    token_profiler.reset()
    fulltext_extractor.reset_usage()   # per-gene full-text subprocess token counter
    _t0 = time.time()
    # papers-only has nothing to retry (no retrieval), so it's always a single attempt.
    max_attempts = 1 if args.papers_only else args.max_attempts
    mode = "papers-only" if args.papers_only else ("full-text ON" if not args.no_full_text else "full-text OFF")
    print(f"\n{'=' * 70}\nGENE: {gene}   ({mode}, "
          f"reviewer={'rules' if args.no_llm_review else 'LLM'}, max_attempts={max_attempts})\n{'=' * 70}")

    # ---- Curator <-> Reviewer feedback loop (retry cap owned here) ----
    adjustment, approved, final_verdict, best = None, None, None, None
    history = []
    for attempt in range(max_attempts):
        result = curator.run(gene, adjustment=adjustment, max_papers=args.max_papers,
                             papers_only=args.papers_only)
        _print_attempt(attempt, result)
        # Keep the best attempt (most reactions) so a regressing retry can't destroy earlier work.
        if best is None or _n_reactions(result) > _n_reactions(best):
            best = result
        final_verdict = reviewer.review(result, attempt, args.max_attempts, history=history)
        _print_verdict(final_verdict)
        history.append({"attempt": attempt + 1, "adjustment_applied": result.adjustment_applied,
                        "decision": final_verdict.decision, "reason": final_verdict.reason})
        if final_verdict.decision == "sufficient":
            approved = result
            break
        if final_verdict.decision == "give_up":
            break
        adjustment = final_verdict.adjustment

    summary = {"gene": gene, "attempts": len(history),
               "decision": final_verdict.decision if final_verdict else "n/a",
               "reactions": 0, "qa_verdict": "—", "qa_score": None, "schema_valid": None,
               "gene_type": "—"}

    # ---- QA on the approved result, else the BEST attempt (most reactions), so a regressing
    #      retry can't throw away earlier work ----
    final_result = approved if approved is not None else best
    provenance = "approved" if approved is not None else f"best_attempt ({final_verdict.decision})"
    reactions = (final_result.fulltext.get("reactions") or []) if final_result else []
    summary["reactions"] = len(reactions)

    # Classify the gene once (has-data vs cold-start · gate pass/fail) for the explanation + batch table.
    cls = classify_gene(gene, final_result)
    summary["gene_type"] = (("has-data" if cls["has_data"] else "cold-start")
                            if cls["has_data"] is not None else "unknown") + f"/{cls['gate']}"

    print(f"\n{'-' * 70}\nQA AGENT — {gene}\n{'-' * 70}")
    if not reactions:
        print(f"No reactions in any attempt ({final_verdict.decision}: {final_verdict.reason}) "
              f"-> manual review. QA skipped.")
        summary["qa_verdict"] = "manual_review"
        _print_result_explanation(gene, cls, 0, "manual_review", None, usage=_usage_snapshot(_t0))
        return summary
    if approved is None:
        print(f"Not Reviewer-approved ({final_verdict.decision}); running QA on the best attempt "
              f"({len(reactions)} reaction(s)) so the work isn't lost.")

    placement_arg, placement_status = curator.build_instances_args(final_result)
    qa_result = qa.check(gene, reactions, accession=final_result.accession,
                         placement=placement_arg, placement_status=placement_status,
                         provenance=provenance)
    inst = qa_result.instances
    kinds = ("entities", "complexes", "reactions", "pathways")
    print(f"instances : {len(inst['entities'])} entities · {len(inst['complexes'])} complexes · "
          f"{len(inst['reactions'])} reactions · {len(inst['pathways'])} pathways")
    if len(qa_result.repair_history) > 1:
        trail = " -> ".join(f"{h['qa_score']}({h['n_issues']}i)" for h in qa_result.repair_history)
        print(f"QA repair : {len(qa_result.repair_history)} iterations  score(issues): {trail}")

    # Per-instance QA breakdown (anti-masking): the LLM names only the instances that are NOT
    # curator-ready; everything else is good. So a low overall qa_score no longer hides that most
    # instances are fine. (LLM-QA only — the rule-based checker doesn't judge per instance.)
    if not args.no_llm_qa:
        flagged = qa_result.report.get("flagged_instances", []) or []
        defined = {i.get("displayName") for k in kinds for i in inst[k]}
        total_inst = sum(len(inst[k]) for k in kinds)
        seen, n_bad, n_rev = set(), 0, 0
        for f in flagged:
            name = f.get("instance", "")
            if name in seen:
                continue
            seen.add(name)
            n_bad += f.get("verdict") == "bad"
            n_rev += f.get("verdict") == "needs_revision"
        n_good = max(total_inst - len(seen), 0)
        print(f"per-inst  : {n_good} good · {n_rev} needs-revision · {n_bad} bad  (of {total_inst})")
        for f in flagged:
            unknown = "" if f.get("instance") in defined else "  (name not matched)"
            print(f"    [{f.get('verdict')}] {f.get('instance_class')} {f.get('instance')}{unknown}"
                  f": {f.get('reason')}")

    print(f"QA verdict: {qa_result.verdict.upper()}  "
          f"(qa_score={qa_result.report.get('qa_score')}, schema_valid={qa_result.schema_check['valid']})")
    for issue in qa_result.report.get("technical_issues", []):
        print(f"    [{issue.get('severity')}] {issue.get('category')}: {issue.get('description')}")
    for err in qa_result.schema_check.get("errors", []):
        print(f"    schema-error: {err}")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{gene.lower()}_curator_instances.json"), "w") as f:
        json.dump(inst, f, indent=2)
    with open(os.path.join("results", f"{gene.lower()}_qa_report.json"), "w") as f:
        json.dump(qa_result.report, f, indent=2)
    print(f"  saved -> results/{gene.lower()}_curator_instances.json + _qa_report.json")

    summary.update({"qa_verdict": qa_result.verdict, "qa_score": qa_result.report.get("qa_score"),
                    "schema_valid": qa_result.schema_check["valid"]})
    _print_result_explanation(gene, cls, len(reactions), qa_result.verdict,
                              qa_result.report.get("qa_score"), usage=_usage_snapshot(_t0))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run the 3-agent Curator pipeline on one or more genes.")
    ap.add_argument("genes", nargs="+", help="Gene symbol(s), e.g. SHANK3 TANC1")
    ap.add_argument("--papers-dir", default=None, help="Local full-text PDF folder.")
    ap.add_argument("--max-attempts", type=int, default=3, help="Curator attempts incl. retries (default 3).")
    ap.add_argument("--max-papers", type=int, default=5)
    ap.add_argument("--no-full-text", action="store_true", help="Skip full-text; retrieval + placement only.")
    ap.add_argument("--papers-only", action="store_true",
                    help="Skip retrieval entirely and annotate ONLY the PDFs in --papers-dir "
                         "(for when the curator already has the papers they want). Implies full text on.")
    ap.add_argument("--no-llm-qa", action="store_true",
                    help="Run QA's deterministic checks only (skip the LLM expert review).")
    ap.add_argument("--no-llm-review", action="store_true",
                    help="Use the rule-based Reviewer instead of the LLM (free, deterministic).")
    args = ap.parse_args()

    if args.papers_only and args.no_full_text:
        sys.exit("[error] --papers-only annotates uploaded PDFs, so it can't be combined with --no-full-text.")

    genes = [g.strip().upper() for g in args.genes]
    check_services()

    # papers-only always needs a PDF folder (there's no retrieval to fall back on); full text is forced on.
    papers_dir = resolve_papers_dir(args.papers_dir, no_full_text=False if args.papers_only else args.no_full_text)
    if args.papers_only and not papers_dir:
        sys.exit("[error] --papers-only needs a PDF folder. Pass --papers-dir /path/to/pdfs (or type a path when prompted).")
    index = FullTextResolver.build_index(papers_dir) if papers_dir else {}
    print(f"PDF folder: {papers_dir} -> {len(index)} PDF(s) indexed" if papers_dir
          else "PDF folder: (none — full text via PMC only)")
    if args.papers_only and not index:
        sys.exit(f"[error] --papers-only: no usable PDFs found in {papers_dir} (need a DOI on page 1 to identify each).")

    # Build the three agents once; reuse across genes. papers-only forces full text on.
    curator = ReactomeCurator(enable_full_text=args.papers_only or not args.no_full_text, fulltext_index=index)
    reviewer = ReactomeReviewer(use_llm=not args.no_llm_review)
    qa = ReactomeQA(use_llm=not args.no_llm_qa)

    summaries = []
    for gene in genes:
        try:
            summaries.append(run_one(gene, curator, reviewer, qa, args))
        except Exception as e:
            import traceback
            traceback.print_exc()
            summaries.append({"gene": gene, "attempts": "-", "decision": "ERROR", "reactions": "-",
                              "qa_verdict": str(e)[:40], "qa_score": None, "schema_valid": None,
                              "gene_type": "-"})

    # ---- batch summary ----
    print(f"\n{'=' * 70}\nBATCH SUMMARY\n{'=' * 70}")
    print(f"  {'gene':<12}{'type':>18}{'reactions':>11}{'QA':>16}{'score':>7}")
    print("  " + "-" * 62)
    for s in summaries:
        print(f"  {s['gene']:<12}{str(s.get('gene_type', '-')):>18}{str(s['reactions']):>11}"
              f"{str(s['qa_verdict']):>16}{str(s['qa_score']) if s['qa_score'] is not None else '-':>7}")


if __name__ == "__main__":
    main()
