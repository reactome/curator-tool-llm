"""A/B the retrieval curator-judge model: claude-sonnet-5 vs claude-sonnet-4-6.

For each gene, runs the REAL retrieve + cross-encoder + LLM curator-judge under each
model (full-text OFF, stops the instant Phase 1 would begin) and diffs the scores. The
candidate pool is esearch(relevance) + cross-encoder, both deterministic, so both models
score the SAME papers -- the harness verifies pool identity and only then compares.

Reports per gene: candidate-pool mean, selected-5 mean, selection overlap, and the
per-PMID score deltas so you can see whether sonnet-5 is a meaningful change or noise.

    conda run -n paperqa python mock_test_judge_ab.py TANC1 CTTNBP2 CBLN3
"""
import argparse
import asyncio
import sys

sys.path.append("reactome_llm")
sys.path.append("examples")

from dotenv import load_dotenv
load_dotenv()

import ModelConfig
from crewai_annotation_examples import build_annotators
from CrewAILiteratureAnnotator import AnnotationRequest, CrewAILiteratureAnnotator

# A = current/baseline (4.6), B = candidate (5.0). Order matters: A runs first per gene.
A_MODEL = "claude-sonnet-4-6"
B_MODEL = "claude-sonnet-5"
MODELS = [A_MODEL, B_MODEL]
LIVE_FILE = "/tmp/judge_ab_live.txt"  # appended per-gene; bypasses conda stdout buffering


class _StopAfterJudge(Exception):
    pass


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


async def _run_once(crewai, gene, model, max_papers):
    """Run retrieve+judge for one gene under one model; return the scored dict + selection."""
    ModelConfig.REACTOME_MODEL_NAME = model  # read fresh by create_reactome_chat_model()
    request = AnnotationRequest(
        gene=gene, papers=[], max_papers=max_papers,
        enable_full_text=False, enable_literature_search=True, fulltext_index={})
    try:
        await crewai.annotate_literature(request)
    except Exception as e:
        if "STOP_AFTER_JUDGE" not in str(e) and not isinstance(e, _StopAfterJudge):
            raise
    jp = (getattr(crewai.gene_annotator, "judged_papers", {}) or {}).get(gene, {})
    scored = jp.get("scored", {}) or {}
    selected = jp.get("papers", [])
    per = {str(p.get("pmid")): p.get("score") for p in scored.get("per_paper", [])}
    return {
        "mean_pool": scored.get("mean_score"),
        "distribution": scored.get("distribution", {}),
        "annotatable": scored.get("annotatable_count"),
        "per_pmid": per,
        "selected": [str(p.get("pmid")) for p in selected],
        "selected_mean": _mean([per.get(str(p.get("pmid"))) for p in selected]),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("genes", nargs="*", default=["TANC1", "CTTNBP2", "CBLN3"])
    ap.add_argument("--max-papers", type=int, default=5)
    args = ap.parse_args()
    genes = args.genes or ["TANC1", "CTTNBP2", "CBLN3"]

    async def _stop(self, *a, **k):
        raise _StopAfterJudge("STOP_AFTER_JUDGE")
    CrewAILiteratureAnnotator._phase_1_literature_extraction = _stop

    _, crewai = build_annotators(verbose=False)

    print(f"\n{'=' * 78}\nJUDGE MODEL A/B — {' vs '.join(MODELS)}\n"
          f"genes: {', '.join(genes)}  (max_papers={args.max_papers})\n{'=' * 78}")

    summary = []
    for gene in genes:
        res = {}
        for model in MODELS:
            res[model] = await _run_once(crewai, gene, model, args.max_papers)
            print(f"[{gene} / {model}] pool_mean={res[model]['mean_pool']} "
                  f"sel_mean={res[model]['selected_mean']} "
                  f"selected={res[model]['selected']}")

        a, b = res[A_MODEL], res[B_MODEL]   # a = 4.6 (baseline), b = 5.0 (candidate)
        pool_a, pool_b = set(a["per_pmid"]), set(b["per_pmid"])
        same_pool = pool_a == pool_b
        shared = pool_a & pool_b
        # Per-PMID score deltas over the shared pool: B(5.0) minus A(4.6).
        deltas = {p: (b["per_pmid"][p] - a["per_pmid"][p]) for p in shared
                  if isinstance(a["per_pmid"].get(p), (int, float))
                  and isinstance(b["per_pmid"].get(p), (int, float))}
        sel_overlap = set(a["selected"]) & set(b["selected"])
        dpool = round((b["mean_pool"] or 0) - (a["mean_pool"] or 0), 3)

        # Live per-gene line -> file (bypasses conda stdout buffering) + flushed stdout.
        live = (f"{gene:<9} A(4.6): pool={a['mean_pool']} sel={a['selected_mean']}   "
                f"B(5.0): pool={b['mean_pool']} sel={b['selected_mean']}   "
                f"Δpool(5-4.6)={dpool}  overlap={len(sel_overlap)}/{args.max_papers}  "
                f"same_pool={same_pool}")
        with open(LIVE_FILE, "a") as f:
            f.write(live + "\n")
        print(live, flush=True)
        if deltas:
            avg_abs = round(sum(abs(d) for d in deltas.values()) / len(deltas), 3)
            biggest = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            print(f"   per-PMID Δ(5-4.6): mean|Δ|={avg_abs}  biggest={biggest}", flush=True)
        summary.append((gene, a, b, same_pool, len(sel_overlap), dpool))

    print(f"\n{'=' * 78}\nSUMMARY  (A = sonnet-4-6 baseline, B = sonnet-5 candidate)\n{'=' * 78}")
    print(f"  {'gene':<10}{'A pool':<9}{'B pool':<9}{'Δpool':<9}{'A sel':<8}{'B sel':<8}{'ovlp':<7}pool=")
    for gene, a, b, same_pool, ovlp, dpool in summary:
        print(f"  {gene:<10}{str(a['mean_pool']):<9}{str(b['mean_pool']):<9}{str(dpool):<9}"
              f"{str(a['selected_mean']):<8}{str(b['selected_mean']):<8}"
              f"{str(ovlp) + '/' + str(args.max_papers):<7}{same_pool}")
    print(f"\nInterpretation: small Δpool + high selection overlap => sonnet-5 ~ sonnet-4-6 "
          f"(no meaningful judge change). Large positive Δpool or different selections => 5 helps.")


if __name__ == "__main__":
    asyncio.run(main())
