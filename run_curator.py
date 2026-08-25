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

os.environ.setdefault("TOKEN_PROFILE", "1")
sys.path.append("reactome_llm")

from dotenv import load_dotenv
load_dotenv()

import token_profiler
import FullTextResolver
from ReactomeCurator import ReactomeCurator
from ReactomeReviewer import ReactomeReviewer
from ReactomeQA import ReactomeQA


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
    print(f"\n{'=' * 70}\nGENE: {gene}   (full-text {'OFF' if args.no_full_text else 'ON'}, "
          f"reviewer={'rules' if args.no_llm_review else 'LLM'}, max_attempts={args.max_attempts})\n{'=' * 70}")

    # ---- Curator <-> Reviewer feedback loop (retry cap owned here) ----
    adjustment, approved, final_verdict, best = None, None, None, None
    history = []
    for attempt in range(args.max_attempts):
        result = curator.run(gene, adjustment=adjustment, max_papers=args.max_papers)
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
               "reactions": 0, "qa_verdict": "—", "qa_score": None, "schema_valid": None}

    # ---- QA on the approved result, else the BEST attempt (most reactions), so a regressing
    #      retry can't throw away earlier work ----
    final_result = approved if approved is not None else best
    provenance = "approved" if approved is not None else f"best_attempt ({final_verdict.decision})"
    reactions = (final_result.fulltext.get("reactions") or []) if final_result else []
    summary["reactions"] = len(reactions)

    print(f"\n{'-' * 70}\nQA AGENT — {gene}\n{'-' * 70}")
    if not reactions:
        print(f"No reactions in any attempt ({final_verdict.decision}: {final_verdict.reason}) "
              f"-> manual review. QA skipped.")
        summary["qa_verdict"] = "manual_review"
        return summary
    if approved is None:
        print(f"Not Reviewer-approved ({final_verdict.decision}); running QA on the best attempt "
              f"({len(reactions)} reaction(s)) so the work isn't lost.")

    placement_arg, placement_status = curator.build_instances_args(final_result)
    qa_result = qa.check(gene, reactions, accession=final_result.accession,
                         placement=placement_arg, placement_status=placement_status,
                         provenance=provenance)
    inst = qa_result.instances
    print(f"instances : {len(inst['entities'])} entities · {len(inst['complexes'])} complexes · "
          f"{len(inst['reactions'])} reactions · {len(inst['pathways'])} pathways")
    if len(qa_result.repair_history) > 1:
        trail = " -> ".join(f"{h['qa_score']}({h['n_issues']}i)" for h in qa_result.repair_history)
        print(f"QA repair : {len(qa_result.repair_history)} iterations  score(issues): {trail}")
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
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run the 3-agent Curator pipeline on one or more genes.")
    ap.add_argument("genes", nargs="+", help="Gene symbol(s), e.g. SHANK3 TANC1")
    ap.add_argument("--papers-dir", default=None, help="Local full-text PDF folder.")
    ap.add_argument("--max-attempts", type=int, default=3, help="Curator attempts incl. retries (default 3).")
    ap.add_argument("--max-papers", type=int, default=5)
    ap.add_argument("--no-full-text", action="store_true", help="Skip full-text; retrieval + placement only.")
    ap.add_argument("--no-llm-qa", action="store_true",
                    help="Run QA's deterministic checks only (skip the LLM expert review).")
    ap.add_argument("--no-llm-review", action="store_true",
                    help="Use the rule-based Reviewer instead of the LLM (free, deterministic).")
    args = ap.parse_args()

    genes = [g.strip().upper() for g in args.genes]
    check_services()

    papers_dir = resolve_papers_dir(args.papers_dir, args.no_full_text)
    index = FullTextResolver.build_index(papers_dir) if papers_dir else {}
    print(f"PDF folder: {papers_dir} -> {len(index)} PDF(s) indexed" if papers_dir
          else "PDF folder: (none — full text via PMC only)")

    # Build the three agents once; reuse across genes.
    curator = ReactomeCurator(enable_full_text=not args.no_full_text, fulltext_index=index)
    reviewer = ReactomeReviewer(use_llm=not args.no_llm_review)
    qa = ReactomeQA(use_llm=not args.no_llm_qa)

    summaries = []
    for gene in genes:
        try:
            summaries.append(run_one(gene, curator, reviewer, qa, args))
        except Exception as e:
            import traceback
            traceback.print_exc()
            summaries.append({"gene": gene, "attempts": "-", "decision": "ERROR",
                              "reactions": "-", "qa_verdict": str(e)[:40], "qa_score": None, "schema_valid": None})

    # ---- batch summary ----
    print(f"\n{'=' * 70}\nBATCH SUMMARY\n{'=' * 70}")
    print(f"  {'gene':<12}{'attempts':>9}{'decision':>16}{'reactions':>11}{'QA':>16}{'score':>7}")
    print("  " + "-" * 68)
    for s in summaries:
        print(f"  {s['gene']:<12}{str(s['attempts']):>9}{s['decision']:>16}{str(s['reactions']):>11}"
              f"{str(s['qa_verdict']):>16}{str(s['qa_score']) if s['qa_score'] is not None else '-':>7}")


if __name__ == "__main__":
    main()
