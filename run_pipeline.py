"""Single end-to-end driver for the CrewAI Reactome annotation pipeline.

Runs the FULL 5-phase pipeline on one or more genes and emits, per gene:

  * Retrieval    -- both Stage-1 queries, pool sizes, and EVERY candidate PMID with its
                    cross-encoder rank, LLM curator-judge score, evidence type,
                    annotatability, selection status, and the judge's justification.
  * Phase 1-5    -- extraction counts + cited PMIDs, curated Reactome instances, reviewer
                    criterion scores, QA score + technical issues, and the per-agent
                    consensus votes behind the final decision.
  * Runtime      -- wall-clock for the Stage-0 precompute and each of the five phases.
  * Token usage  -- input/output tokens and estimated cost, per call-site and per phase.

Outputs land in results/ :
    results/<GENE>_<tag>.md      human-readable report
    results/<GENE>_<tag>.json    complete machine-readable dump
    results/summary_<tag>.csv    one row per gene

Run (from repo root):
    python run_pipeline.py SHANK3
    python run_pipeline.py SHANK3 CTTNBP2 FAM120C --max-papers 5 --tag smoke
    caffeinate -d conda run -n paperqa python run_pipeline.py SHANK3

Token profiling is switched on automatically -- there is no separate TOKEN_PROFILE step.
Requires MongoDB up (brew services), Neo4j reachable, and ANTHROPIC_API_KEY in .env.
"""
import os
import sys

# Must precede the token_profiler / ModelConfig imports: profiling is read at model-construction
# time, and the LangChain callback is only attached when this is already set.
os.environ.setdefault("TOKEN_PROFILE", "1")

import argparse
import asyncio
import csv
import json
import logging
import re
import time
import urllib.request
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.append("reactome_llm")
sys.path.append("examples")

from dotenv import load_dotenv
load_dotenv()

import token_profiler
import ReactomeNeo4jUtils as neo4jutils
import FullTextResolver
from crewai_annotation_examples import build_annotators
from CrewAILiteratureAnnotator import AnnotationRequest, CrewAILiteratureAnnotator

# The five phase coroutines on the annotator, in execution order, paired with report labels.
PHASES = [
    ("_phase_1_literature_extraction", "Phase 1 · extraction"),
    ("_phase_2_data_model_creation", "Phase 2 · curation"),
    ("_phase_3_expert_review", "Phase 3 · review"),
    ("_phase_4_quality_assurance", "Phase 4 · QA"),
    ("_phase_5_final_consensus_meeting", "Phase 5 · consensus"),
]

# Filled by the instrumentation below, cleared per gene.
_TIMES: dict = {}
_CONTEXTS: dict = {}


# ------------------------------------------------------------------------------------------
# Instrumentation — capture each phase's wall-clock AND its returned context without touching
# the pipeline. The phase contexts carry strictly more than AnnotationResult exposes (notably
# Phase 5's per-agent votes and Phase 1's raw extraction), which is what this report needs.
# ------------------------------------------------------------------------------------------
def _instrument():
    for name, label in PHASES:
        orig = getattr(CrewAILiteratureAnnotator, name)

        def make(name, orig, label):
            async def timed(self, *a, **k):
                print(f"  {label} …", flush=True)
                t0 = time.perf_counter()
                try:
                    ctx = await orig(self, *a, **k)
                    _CONTEXTS[name] = ctx
                    return ctx
                finally:
                    _TIMES[name] = round(time.perf_counter() - t0, 1)
            return timed

        setattr(CrewAILiteratureAnnotator, name, make(name, orig, label))

    # Stage-0 retrieval + LLM curator-judge, timed separately from the rest of the precompute
    # (accession resolution, placement gate, rerank-target generation).
    orig_rj = CrewAILiteratureAnnotator._retrieve_and_judge

    def timed_rj(self, *a, **k):
        print("  Retrieving candidates → cross-encoder rerank → LLM curator judge …", flush=True)
        t0 = time.perf_counter()
        try:
            return orig_rj(self, *a, **k)
        finally:
            _TIMES["_retrieve_and_judge"] = round(time.perf_counter() - t0, 1)

    CrewAILiteratureAnnotator._retrieve_and_judge = timed_rj

    # annotate_literature calls emit_report(), which writes a data/token_usage_<gene>_<date>.csv
    # on every run. This driver aggregates the same records itself, so suppress that side file
    # and keep data/ clean.
    token_profiler.emit_report = lambda gene, out_dir="data": None


class WarnCollector(logging.Handler):
    """Buffer WARNING+ log records so anything odd surfaces in the report."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        try:
            self.records.append((record.levelname, self.format(record)))
        except Exception:
            pass


# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------
def determine_branch(crewai: CrewAILiteratureAnnotator, gene: str) -> str:
    """Which Stage-1/Stage-2 branch the pipeline took for this gene.

    Read from state the run already computed rather than recomputing the placement: the gene
    has released Reactome pathways (has-data), or it is cold-start and the confidence gate
    either passed (a directive was injected) or failed.
    """
    try:
        if neo4jutils.query_pathways_for_gene(gene):
            return "has-data"
    except Exception:
        return "unknown"
    return "cold-start/gate-pass" if crewai.resolved_placement else "cold-start/gate-fail"


def fetch_titles(pmids) -> dict:
    """Best-effort NCBI esummary title lookup — the Mongo cache stores only pmid + abstract,
    so retrieved papers usually arrive title-less. Returns {pmid: title}; {} on any failure."""
    pmids = [str(p) for p in pmids if p]
    if not pmids:
        return {}
    titles = {}
    for i in range(0, len(pmids), 100):
        chunk = pmids[i:i + 100]
        try:
            api_key = os.getenv("PUBMED_API_KEY", "")
            url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                   "?db=pubmed&retmode=json&id=" + ",".join(chunk)
                   + (f"&api_key={api_key}" if api_key else ""))
            time.sleep(0.4)
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            res = data.get("result", {})
            for u in res.get("uids", []):
                titles[u] = res.get(u, {}).get("title", "")
        except Exception:
            continue
    return titles


def token_breakdown():
    """Aggregate the current gene's records -> (totals, per-phase rows, per-call-site rows)."""
    recs = token_profiler._RECORDS
    tot = {"calls": 0, "in": 0, "out": 0}
    by_phase = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0})
    by_label = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "phase": ""})
    for r in recs:
        for bucket in (tot, by_phase[r.phase], by_label[r.label]):
            bucket["calls"] += r.calls
            bucket["in"] += r.input_tokens
            bucket["out"] += r.output_tokens
        by_label[r.label]["phase"] = r.phase
    for d in list(by_phase.values()) + list(by_label.values()):
        d["total"] = d["in"] + d["out"]
        d["cost"] = round(token_profiler._cost_usd(d["in"], d["out"]), 4)
    tot["total"] = tot["in"] + tot["out"]
    tot["cost"] = round(token_profiler._cost_usd(tot["in"], tot["out"]), 4)
    return tot, dict(by_phase), dict(by_label)


def retrieval_block(crewai: CrewAILiteratureAnnotator, gene: str):
    """Every judged candidate with its cross-encoder rank, rubric score and selection status.

    `scored["per_paper"]` is aligned 1:1 with the cross-encoder-ranked candidate pool, so its
    index IS the Stage-2 rank. `papers` holds only the judge's final selection, which is where
    cross_score lives — join on PMID to attach it where available.
    """
    cache = (getattr(crewai.gene_annotator, "judged_papers", {}) or {}).get(gene)
    if not cache:
        return None

    selected = cache.get("papers", [])
    sel_by_pmid = {str(p.get("pmid")): p for p in selected}
    per_paper = (cache.get("scored") or {}).get("per_paper", [])

    missing = [pp.get("pmid") for pp in per_paper if not (pp.get("title") or "").strip()]
    titles = fetch_titles(missing)

    candidates = []
    for rank, pp in enumerate(per_paper, 1):
        pmid = str(pp.get("pmid", ""))
        hit = sel_by_pmid.get(pmid)
        candidates.append({
            "ce_rank": rank,
            "pmid": pmid,
            "title": (pp.get("title") or "").strip() or titles.get(pmid, ""),
            "cross_score": (hit or {}).get("cross_score"),
            "judge_score": pp.get("score"),
            "evidence_type": pp.get("evidence_type"),
            "specific_interaction": pp.get("specific_interaction"),
            "annotatable": pp.get("annotatable"),
            "justification": pp.get("justification", ""),
            "selected": hit is not None,
        })

    sel_rows = [c for c in candidates if c["selected"]]
    vals = [c["judge_score"] for c in sel_rows if isinstance(c["judge_score"], (int, float))]
    all_vals = [c["judge_score"] for c in candidates if isinstance(c["judge_score"], (int, float))]
    return {
        "name_query": cache.get("name_query", ""),
        "context_query": cache.get("context_query", ""),
        "pool_size": cache.get("pool_size", 0),
        "candidate_pool": cache.get("candidate_pool", len(candidates)),
        "dropped_below_threshold": cache.get("dropped_below_threshold", 0),
        "n_selected": len(sel_rows),
        "selected_pmids": [c["pmid"] for c in sel_rows],
        "mean_selected": round(sum(vals) / len(vals), 2) if vals else None,
        "mean_pool": round(sum(all_vals) / len(all_vals), 2) if all_vals else None,
        "annotatable_selected": sum(1 for c in sel_rows if c["annotatable"]),
        "evidence_types_selected": dict(Counter(c["evidence_type"] for c in sel_rows
                                                if c["evidence_type"])),
        "score_distribution_selected": dict(sorted(Counter(vals).items())),
        "candidates": candidates,
    }


def _n(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def _trunc(s, n):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def instance_counts(instances: dict) -> dict:
    return {k: len(instances.get(k) or []) for k in
            ("entities", "complexes", "reactions", "pathways")}


# ------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------
_FT_LABEL = {"pdf": "PDF", "xml": "XML", "miss": "miss"}


def ft_summary(manifest: dict) -> tuple:
    """(n_pdf, n_xml, n_miss) over a resolution manifest."""
    src = [v.get("source") for v in (manifest or {}).values()]
    return src.count("pdf"), src.count("xml"), src.count("miss")


def build_markdown(gene, branch, accession, args, elapsed, times, tot, by_phase, by_label,
                   retr, ctx, flags, warns, manifest=None) -> str:
    manifest = manifest or {}
    L = []
    ext = ctx.get("_phase_1_literature_extraction") or {}
    cur = ctx.get("_phase_2_data_model_creation") or {}
    rev = ctx.get("_phase_3_expert_review") or {}
    qa = ctx.get("_phase_4_quality_assurance") or {}
    con = ctx.get("_phase_5_final_consensus_meeting") or {}

    extraction = ext.get("structured_information") or {}
    instances = cur.get("reactome_instances") or {}
    review = rev.get("validation_report") or {}
    qa_report = qa.get("consistency_check") or {}
    consensus = con.get("final_consensus") or {}
    votes = con.get("individual_votes") or {}

    L.append(f"# {gene} — {branch}\n")
    L.append(f"_UniProt {accession or '—'} · max_papers={args.max_papers} · "
             f"full_text={'on' if args.full_text else 'off'} · "
             f"quality_threshold={args.quality_threshold} · {datetime.now():%Y-%m-%d %H:%M}_\n")

    if flags:
        L.append("## ⚠️ Flags\n")
        L += [f"- {f}" for f in flags]
        L.append("")

    # ---- runtime + tokens -------------------------------------------------------------
    L.append("## Run\n")
    phase_sum = sum(v for k, v in times.items() if k.startswith("_phase_"))
    stage0 = round(max(0.0, elapsed - phase_sum), 1)
    L.append(f"- **Wall-clock:** {elapsed / 60:.1f} min ({elapsed}s)")
    L.append(f"- **Tokens:** {tot['total']:,} (in {tot['in']:,} / out {tot['out']:,}) "
             f"across {tot['calls']} API calls")
    L.append(f"- **Estimated cost:** ${tot['cost']}")
    L.append("")
    L.append("| stage | wall-clock | tokens | cost |")
    L.append("|---|--:|--:|--:|")
    rj = times.get("_retrieve_and_judge")
    L.append(f"| Stage 0 · precompute + retrieval | {stage0}s"
             f"{f' (retrieve+judge {rj}s)' if rj else ''} | "
             f"{by_phase.get('precompute', {}).get('total', 0):,} | "
             f"${by_phase.get('precompute', {}).get('cost', 0)} |")
    for i, (method, label) in enumerate(PHASES, 1):
        p = by_phase.get(f"phase_{i}", {})
        L.append(f"| {label} | {times.get(method, '?')}s | {p.get('total', 0):,} | "
                 f"${p.get('cost', 0)} |")
    L.append("")

    # ---- retrieval --------------------------------------------------------------------
    L.append("## Retrieval — Stage 1 search → cross-encoder → LLM curator judge\n")
    if not retr:
        L.append("- _No judged cache: the upfront retrieve+judge failed or returned nothing, "
                 "so Phase 1 fell back to live cross-encoder retrieval with no judge._\n")
    else:
        L.append(f"- **name_query:** `{_trunc(retr['name_query'], 300) or '(empty)'}`")
        L.append(f"- **context_query:** `{_trunc(retr['context_query'], 300) or '(empty — gate-fail branch)'}`")
        L.append(f"- Stage-1 union pool: **{retr['pool_size']}** papers → cross-encoder "
                 f"candidate pool: **{retr['candidate_pool']}** → judge selected: "
                 f"**{retr['n_selected']}** ({retr['dropped_below_threshold']} dropped below "
                 f"the rubric floor)")
        L.append(f"- **Mean judge score:** {retr['mean_selected']} over selected "
                 f"(pool mean {retr['mean_pool']})  |  **annotatable:** "
                 f"{retr['annotatable_selected']}/{retr['n_selected']}")
        L.append(f"- Evidence types (selected): {retr['evidence_types_selected'] or '—'}")
        L.append(f"- Score distribution (selected): {retr['score_distribution_selected'] or '—'}")
        L.append(f"- **Selected PMIDs:** {', '.join(retr['selected_pmids']) or '—'}")
        L.append("")

        L.append("### Selected papers\n")
        for c in [c for c in retr["candidates"] if c["selected"]]:
            cs = c["cross_score"]
            ce = f" (score {cs:.2f})" if isinstance(cs, (int, float)) else ""
            src = (manifest.get(c["pmid"]) or {}).get("source")
            ft = f" · full text: {_FT_LABEL[src]}" if src else ""
            L.append(f"- **PMID {c['pmid']}** — judge {c['judge_score']} · "
                     f"{c['evidence_type'] or '?'}"
                     f"{' · annotatable' if c['annotatable'] else ''} · "
                     f"CE rank {c['ce_rank']}{ce}{ft}")
            L.append(f"    - {c['title'] or '(no title)'}")
            L.append(f"    - _{c['justification']}_")
        L.append("")

        # ---- full-text resolution -----------------------------------------------------
        if args.full_text:
            L.append("### Full-text resolution\n")
            if not manifest:
                L.append("- _Full-text enabled but no papers were resolved "
                         "(no selected PMIDs, or resolution failed)._\n")
            else:
                n_pdf, n_xml, n_miss = ft_summary(manifest)
                L.append(f"- **{n_pdf}** local PDF · **{n_xml}** PMC XML · **{n_miss}** miss "
                         f"(of {len(manifest)} selected)")
                L.append("")
                L.append("| PMID | source | path |")
                L.append("|---|:-:|---|")
                for pmid, entry in manifest.items():
                    src = entry.get("source")
                    L.append(f"| {pmid} | {_FT_LABEL.get(src, src or '—')} "
                             f"| {entry.get('path', '—')} |")
                L.append("")

        L.append(f"### Full candidate pool ({len(retr['candidates'])})\n")
        L.append("| CE rank | PMID | judge | evidence | annot. | sel. | title |")
        L.append("|--:|---|--:|---|:-:|:-:|---|")
        for c in retr["candidates"]:
            L.append(f"| {c['ce_rank']} | {c['pmid']} | {c['judge_score'] if c['judge_score'] is not None else '—'} "
                     f"| {c['evidence_type'] or '—'} | {'✓' if c['annotatable'] else ''} "
                     f"| {'**✓**' if c['selected'] else ''} | {_trunc(c['title'], 90) or '—'} |")
        L.append("")

    # ---- phase 1 ----------------------------------------------------------------------
    L.append("## Phase 1 — Literature extraction\n")
    if extraction:
        inter = extraction.get("interactions") or []
        paths = extraction.get("pathways") or []
        funcs = extraction.get("functions") or []
        cited = sorted({i.get("pmid", "").strip() for i in inter + paths + funcs
                        if (i.get("pmid") or "").strip()})
        L.append(f"- Evidence extracted: **{len(inter)}** interactions, **{len(paths)}** pathway "
                 f"roles, **{len(funcs)}** functions")
        L.append(f"- Distinct PMIDs cited: **{ext.get('papers_processed', 0)}** "
                 f"({', '.join(cited) or '—'})")
        if extraction.get("summary"):
            L.append(f"- Summary: {extraction['summary'].strip()}")
        if inter:
            L.append("\n| partner | type | confidence | strength | PMID | evidence |")
            L.append("|---|---|---|--:|---|---|")
            for i in inter:
                L.append(f"| {i.get('partner', '?')} | {i.get('interaction_type', '')} "
                         f"| {i.get('confidence', '')} | {_n(i.get('evidence_strength_score'))} "
                         f"| {i.get('pmid', '')} | {_trunc(i.get('evidence'), 80)} |")
        if paths:
            L.append("\n| pathway | role | confidence | PMID |")
            L.append("|---|---|---|---|")
            for p in paths:
                L.append(f"| {_trunc(p.get('pathway_name'), 60)} | {p.get('role', '')} "
                         f"| {p.get('confidence', '')} | {p.get('pmid', '')} |")
        if funcs:
            L.append("\n**Functions**")
            for f in funcs:
                L.append(f"- {_trunc(f.get('function'), 160)} "
                         f"({f.get('confidence', '')}, PMID {f.get('pmid', '') or '—'})")
    else:
        L.append("- _(phase produced no output)_")
    L.append("")

    # ---- phase 2 ----------------------------------------------------------------------
    L.append("## Phase 2 — Reactome data model\n")
    counts = instance_counts(instances)
    L.append(f"- Instances created: **{sum(counts.values())}** — "
             + ", ".join(f"{v} {k}" for k, v in counts.items()))
    for ent in instances.get("entities") or []:
        L.append(f"    - _Entity_ **{ent.get('name', '?')}** "
                 f"[{ent.get('class', '?')}] {ent.get('identifier', '')} "
                 f"({ent.get('compartment', '') or 'no compartment'})")
    for cx in instances.get("complexes") or []:
        L.append(f"    - _Complex_ **{cx.get('name', '?')}** ← "
                 f"{', '.join(cx.get('components') or []) or '(no components)'}")
    for rx in instances.get("reactions") or []:
        L.append(f"    - _Reaction_ **{rx.get('name', '?')}**: "
                 f"{', '.join(rx.get('input') or []) or '∅'} → "
                 f"{', '.join(rx.get('output') or []) or '∅'}")
    for pw in instances.get("pathways") or []:
        L.append(f"    - _Pathway_ **{pw.get('name', '?')}** — {_trunc(pw.get('summation'), 120)}")
    L.append("")

    # ---- phase 3 ----------------------------------------------------------------------
    L.append("## Phase 3 — Expert review\n")
    if review:
        cs = review.get("criterion_scores") or {}
        L.append(f"- **Overall:** {_n(review.get('overall_score'))}  |  "
                 f"**Approval:** {review.get('approval_status', '?')}")
        L.append(f"- Criteria — biological accuracy {_n(cs.get('biological_accuracy'))}, "
                 f"evidence support {_n(cs.get('evidence_support'))}, "
                 f"mechanistic consistency {_n(cs.get('mechanistic_consistency'))}, "
                 f"integration quality {_n(cs.get('integration_quality'))}")
        if review.get("summary"):
            L.append(f"- Summary: {review['summary'].strip()}")
        issues = [(r.get("instance_id", "?"), i)
                  for r in review.get("instance_reviews") or [] for i in r.get("issues") or []]
        if issues:
            L.append(f"- Issues ({len(issues)}):")
            L += [f"    - `{inst}` {txt}" for inst, txt in issues]
        recs = review.get("recommendations") or []
        if recs:
            L.append("- Recommendations:")
            L += [f"    - {r}" for r in recs]
    else:
        L.append("- _(phase produced no output)_")
    L.append("")

    # ---- phase 4 ----------------------------------------------------------------------
    L.append("## Phase 4 — Quality assurance\n")
    if qa_report:
        ia = qa_report.get("integration_assessment") or {}
        L.append(f"- **qa_score:** {_n(qa_report.get('qa_score'))}  |  "
                 f"conflicts detected: {ia.get('conflicts_detected')}  |  "
                 f"performance impact: {ia.get('performance_impact', '?')}  |  "
                 f"compatibility: {_n(ia.get('compatibility_score'))}")
        ti = qa_report.get("technical_issues") or []
        if ti:
            by_sev = defaultdict(list)
            for t in ti:
                by_sev[t.get("severity", "?")].append(t)
            L.append(f"- Technical issues ({len(ti)}):")
            for sev in ("high", "medium", "low", "?"):
                for t in by_sev.get(sev, []):
                    loc = t.get("location")
                    where = f" — _{loc}_" if loc else ""
                    L.append(f"    - **[{sev}/{t.get('category', '?')}]** "
                             f"{t.get('description', '')}{where}")
                    if t.get("resolution"):
                        L.append(f"        - fix: {t['resolution']}")
        else:
            L.append("- No technical issues reported.")
    else:
        L.append("- _(phase produced no output)_")
    L.append("")

    # ---- phase 5 ----------------------------------------------------------------------
    L.append("## Phase 5 — Consensus\n")
    if consensus:
        vt = consensus.get("vote_tally") or {}
        L.append(f"- **Decision:** {consensus.get('decision', '?').upper()}  "
                 f"(confidence {_n(consensus.get('confidence'))})")
        L.append(f"- Tally — approve {vt.get('approve', 0)}, "
                 f"requires_revision {vt.get('requires_revision', 0)}, "
                 f"reject {vt.get('reject', 0)}")
        if consensus.get("summary"):
            L.append(f"- Summary: {consensus['summary'].strip()}")
        bi = consensus.get("blocking_issues") or []
        if bi:
            L.append("- Blocking issues:")
            L += [f"    - {b}" for b in bi]
        rr = consensus.get("required_revisions") or []
        if rr:
            L.append("- Required revisions:")
            L += [f"    - {r}" for r in rr]
    else:
        L.append("- _(phase produced no output)_")
    if votes:
        L.append("\n### Individual votes\n")
        L.append("| agent | vote | confidence | rationale |")
        L.append("|---|---|--:|---|")
        for role, v in votes.items():
            L.append(f"| {role} | {v.get('decision', '?')} | {_n(v.get('confidence'))} "
                     f"| {_trunc(v.get('summary'), 100)} |")
        for role, v in votes.items():
            for b in v.get("blocking_issues") or []:
                L.append(f"- _{role} blocking:_ {b}")
    L.append("")

    # ---- token detail ------------------------------------------------------------------
    L.append("## Token usage by call site\n")
    L.append("| call site | phase | calls | in | out | total | cost |")
    L.append("|---|---|--:|--:|--:|--:|--:|")
    for lbl, d in sorted(by_label.items(), key=lambda kv: -kv[1]["total"]):
        L.append(f"| `{lbl}` | {d['phase']} | {d['calls']} | {d['in']:,} | {d['out']:,} "
                 f"| {d['total']:,} | ${d['cost']} |")
    L.append(f"| **TOTAL** | | **{tot['calls']}** | **{tot['in']:,}** | **{tot['out']:,}** "
             f"| **{tot['total']:,}** | **${tot['cost']}** |")
    L.append("")

    if warns:
        L.append("## Warnings (deduped)\n")
        for (lvl, msg), c in warns.most_common():
            L.append(f"- [{lvl} ×{c}] {msg}")
        L.append("")

    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------
async def run_gene(crewai, gene, args, collector) -> dict:
    _TIMES.clear()
    _CONTEXTS.clear()
    collector.records.clear()

    flags, err, result = [], None, None
    t0 = time.perf_counter()
    try:
        request = AnnotationRequest(
            gene=gene,
            papers=[],
            max_papers=args.max_papers,
            quality_threshold=args.quality_threshold,
            enable_full_text=args.full_text,
            enable_literature_search=True,
            schema_path=args.schema_path,
            fulltext_index=getattr(args, "fulltext_index", None),
        )
        result = await crewai.annotate_literature(request)
    except Exception as e:
        import traceback
        err = f"{type(e).__name__}: {e}"
        flags.append(f"❌ **HARD ERROR** — {err}")
        print(f"  !! {gene} FAILED: {err}\n{traceback.format_exc()}", flush=True)
    elapsed = round(time.perf_counter() - t0, 1)

    # Snapshot BEFORE the next gene's token_profiler.reset().
    tot, by_phase, by_label = token_breakdown()
    branch = determine_branch(crewai, gene)
    accession = crewai.resolved_accession
    retr = retrieval_block(crewai, gene)
    # Full-text resolution manifest {pmid: {source, path}}, published by the annotator's
    # pre-Phase-1 resolve step ({} when full-text is off or nothing resolved).
    manifest = (getattr(crewai.gene_annotator, "fulltext_manifest", {}) or {}).get(gene, {})

    if retr is None:
        flags.append("retrieval judge produced no cache — Phase 1 used the unjudged fallback")
    elif retr["mean_selected"] is None or retr["mean_selected"] < 3:
        flags.append(f"weak retrieval evidence (mean judge score {retr['mean_selected']})")
    elif retr["n_selected"] < args.max_papers:
        flags.append(f"judge returned only {retr['n_selected']}/{args.max_papers} papers "
                     f"above the rubric floor")

    cur = _CONTEXTS.get("_phase_2_data_model_creation") or {}
    counts = instance_counts(cur.get("reactome_instances") or {})
    n_inst = sum(counts.values())
    consensus = ((_CONTEXTS.get("_phase_5_final_consensus_meeting") or {})
                 .get("final_consensus") or {})
    decision = consensus.get("decision", "—")
    if result:
        if n_inst == 0:
            flags.append("no Reactome instances were created")
        if decision == "reject":
            flags.append("Phase 5 decision = REJECT")

    warns = Counter((lvl, msg[:160]) for lvl, msg in collector.records)
    if warns:
        flags.append(f"{sum(warns.values())} warning(s) logged")

    md = build_markdown(gene, branch, accession, args, elapsed, dict(_TIMES),
                        tot, by_phase, by_label, retr, dict(_CONTEXTS), flags, warns, manifest)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"{gene}_{args.tag}.md"
    md_path.write_text(md)

    review = (_CONTEXTS.get("_phase_3_expert_review") or {}).get("validation_report") or {}
    qa_report = (_CONTEXTS.get("_phase_4_quality_assurance") or {}).get("consistency_check") or {}

    dump = {
        "gene": gene, "branch": branch, "accession": accession, "error": err,
        "params": {"max_papers": args.max_papers, "quality_threshold": args.quality_threshold,
                   "full_text": args.full_text, "schema_path": args.schema_path},
        "runtime": {"total_s": elapsed, "phases": dict(_TIMES)},
        "tokens": {"total": tot, "by_phase": by_phase, "by_call_site": by_label},
        "retrieval": retr,
        "fulltext_manifest": manifest,
        "phases": dict(_CONTEXTS),
    }
    json_path = out / f"{gene}_{args.tag}.json"
    json_path.write_text(json.dumps(dump, indent=2, default=str))

    print(f"  -> {elapsed / 60:.1f} min · {tot['total']:,} tok · ${tot['cost']} · "
          f"{n_inst} instances · judge {(retr or {}).get('mean_selected')} · "
          f"{decision} · {len(flags)} flag(s)\n     {md_path}", flush=True)

    return {
        "gene": gene, "branch": branch, "status": "FAIL" if err else "ok",
        "time_min": round(elapsed / 60, 1), "tokens": tot["total"], "cost_usd": tot["cost"],
        "calls": tot["calls"],
        "pool_size": (retr or {}).get("pool_size"),
        "candidates": (retr or {}).get("candidate_pool"),
        "papers_selected": (retr or {}).get("n_selected"),
        "selected_pmids": ";".join((retr or {}).get("selected_pmids") or []),
        "judge_mean": (retr or {}).get("mean_selected"),
        "annotatable": (retr or {}).get("annotatable_selected"),
        "ft_pdf": ft_summary(manifest)[0],
        "ft_xml": ft_summary(manifest)[1],
        "ft_miss": ft_summary(manifest)[2],
        "instances": n_inst,
        "p3_score": review.get("overall_score"), "p3_approval": review.get("approval_status"),
        "p4_score": qa_report.get("qa_score"),
        "p4_conflicts": (qa_report.get("integration_assessment") or {}).get("conflicts_detected"),
        "p5_decision": decision, "p5_confidence": consensus.get("confidence"),
        "flags": len(flags), "error": err,
    }


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def resolve_papers_dir_interactive() -> str | None:
    """Interactive: confirm/update the persisted full-text folder. Returns the path (or None).

    Asks about the folder only ONCE across runs — the choice is saved to data/user_config.json.
    On later runs it just offers to update (or, if none is set, to add) the folder.
    """
    cfg = FullTextResolver.load_config()
    papers_dir = cfg.get("papers_dir")

    if papers_dir:
        print(f"\nFull-text paper folder: {papers_dir}")
        if _ask("Update it? [y/N] ").lower().startswith("y"):
            new = _ask("New path (blank to remove the folder): ")
            papers_dir = new or None
            cfg["papers_dir"] = papers_dir
            FullTextResolver.save_config(cfg)
    else:
        if _ask("\nDo you have a folder of full-text papers to use? [y/N] ").lower().startswith("y"):
            new = _ask("Path to your full-text paper folder: ")
            papers_dir = new or None
            if papers_dir:
                cfg["papers_dir"] = papers_dir
                FullTextResolver.save_config(cfg)
    return papers_dir


async def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("genes", nargs="*", help="Gene symbol(s). Omit to run the interactive wizard.")
    ap.add_argument("--max-papers", type=int, default=5)
    ap.add_argument("--quality-threshold", type=float, default=0.7)
    # Full-text is ON by default (curators always want it). Resolves the selected PMIDs to local
    # curator PDFs or downloaded PMC XML (see FullTextResolver). --no-full-text restores the
    # abstract-only path for batch/eval runs that need the old behavior.
    ap.add_argument("--no-full-text", action="store_false", dest="full_text", default=True,
                    help="Disable full-text; annotate from abstracts only.")
    ap.add_argument("--papers-dir", default=None,
                    help="Folder of full-text PDFs (batch mode). Persisted to data/user_config.json.")
    ap.add_argument("--schema-path", default="resources/reactome_domain_model.json")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--verbose", action="store_true", help="Stream CrewAI agent chatter")
    args = ap.parse_args()

    # Resolve the gene list and full-text folder. Genes on the CLI => batch/non-interactive (keeps
    # unattended eval scripts working); no genes => interactive wizard.
    if args.papers_dir:
        cfg = FullTextResolver.load_config()
        cfg["papers_dir"] = args.papers_dir
        FullTextResolver.save_config(cfg)

    if args.genes:
        genes = args.genes
        papers_dir = args.papers_dir or FullTextResolver.load_config().get("papers_dir")
    else:
        papers_dir = resolve_papers_dir_interactive()
        genes = [g for g in re.split(r"[\s,]+", _ask("\nWhich gene(s) do you want to run? ")) if g]
        if not genes:
            print("No genes provided — nothing to run.")
            return

    # Build the IndexDoc once (curator PDFs -> {pmid: path}); only relevant with full-text on.
    args.fulltext_index = (
        FullTextResolver.build_index(papers_dir) if (args.full_text and papers_dir) else {})

    _instrument()
    logging.captureWarnings(True)
    warnings.simplefilter("once")
    # CrewAI/pymongo emit ~70 DeprecationWarnings and a shutdown ResourceWarning per run. Left on
    # they trip the warning flag on every gene and bury anything actually worth reading.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    collector = WarnCollector()
    logging.getLogger().addHandler(collector)

    _, crewai = build_annotators(verbose=args.verbose)

    rows = []
    for i, gene in enumerate(genes, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(genes)}] {gene}\n{'=' * 78}", flush=True)
        rows.append(await run_gene(crewai, gene, args, collector))

    out = Path(args.out_dir)
    csv_path = out / f"summary_{args.tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    print("\n" + "=" * 78)
    print(f"{'gene':<12}{'branch':<22}{'min':>6}{'tokens':>10}{'cost$':>8}"
          f"{'judge':>7}{'inst':>6}  decision")
    for r in rows:
        print(f"{r['gene']:<12}{r['branch']:<22}{r['time_min']:>6}{r['tokens']:>10,}"
              f"{r['cost_usd']:>8}{str(r['judge_mean']):>7}{r['instances']:>6}  {r['p5_decision']}")
    if ok:
        print("-" * 78)
        print(f"{'TOTAL':<12}{f'({len(ok)} ok)':<22}"
              f"{sum(r['time_min'] for r in ok):>6.1f}"
              f"{sum(r['tokens'] for r in ok):>10,}"
              f"{round(sum(r['cost_usd'] for r in ok), 2):>8}")
    print("=" * 78)
    print(f"\nPer-gene reports : {out}/<GENE>_{args.tag}.md  (+ .json)")
    print(f"Summary          : {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
