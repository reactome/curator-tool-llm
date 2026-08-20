"""Run the REAL pipeline for one gene up through full-text extraction+merge+review, then STOP
before Phase 1. This exercises the whole connected front half in one shot:

    retrieve (name+context union) -> cross-encoder rerank -> LLM curator-judge
      -> full-text resolve (local PDF / PMC XML / miss)
      -> extract_and_merge (parallel per-paper extraction -> single merge -> OpenAI review)

and then reports, without touching Phases 1-5:
  * retrieval LLM judge scores (mean, distribution, per-paper table, final selection)
  * full-text resolution manifest (how many PDF vs XML vs miss)
  * mean extraction confidence + merge confidence over the merged reactions
  * the OpenAI (gpt-5.6-luna) review score for the run

Unlike pause_after_resolve.py this builds the REAL local index from data/user_config.json's
papers_dir, so a curator PDF whose PMID the judge selected resolves as source="pdf".

    conda run -n paperqa python mock_test_fulltext.py SHANK3
"""
import argparse
import asyncio
import json
import os
import re
import sys

os.environ.setdefault("TOKEN_PROFILE", "1")

sys.path.append("reactome_llm")
sys.path.append("examples")

from dotenv import load_dotenv
load_dotenv()

import FullTextResolver
from crewai_annotation_examples import build_annotators
from CrewAILiteratureAnnotator import AnnotationRequest, CrewAILiteratureAnnotator

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_FT_LABEL = {"pdf": "PDF", "xml": "XML", "miss": "miss"}


class _StopAfterFullText(Exception):
    """Sentinel: full-text extraction+merge is done, do not enter Phase 1."""


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


def _rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gene", nargs="?", default="SHANK3")
    ap.add_argument("--max-papers", type=int, default=5)
    args = ap.parse_args()
    gene = args.gene

    # Stop the moment Phase 1 would begin: resolve + extract_and_merge already ran by then.
    async def _stop(self, *a, **k):
        raise _StopAfterFullText("STOP_AFTER_FULLTEXT")
    CrewAILiteratureAnnotator._phase_1_literature_extraction = _stop

    papers_dir = FullTextResolver.load_config().get("papers_dir")
    index = FullTextResolver.build_index(papers_dir) if papers_dir else {}
    _rule(f"MOCK TEST — {gene}  (max_papers={args.max_papers})")
    print(f"papers_dir : {papers_dir}")
    print(f"local index: {len(index)} PDF(s) resolved to PMIDs -> "
          f"{', '.join(sorted(index)) or '(none)'}")

    _, crewai = build_annotators(verbose=False)
    request = AnnotationRequest(
        gene=gene, papers=[], max_papers=args.max_papers,
        enable_full_text=True, enable_literature_search=True, fulltext_index=index)

    try:
        await crewai.annotate_literature(request)
        print("\n!! Ran to completion without hitting the pause point.")
    except Exception as e:
        if "STOP_AFTER_FULLTEXT" not in str(e) and not isinstance(e, _StopAfterFullText):
            raise
        print("\n-- Paused after full-text extraction+merge (Phase 1 not started). --")

    ga = crewai.gene_annotator

    # ---------- 1) Retrieval LLM judge ----------
    _rule("1) RETRIEVAL — LLM curator-judge scores")
    jp = (getattr(ga, "judged_papers", {}) or {}).get(gene, {})
    scored = jp.get("scored", {}) or {}
    selected = jp.get("papers", [])
    print(f"candidates scored : {scored.get('n', 0)}")
    print(f"mean score        : {scored.get('mean_score')}")
    print(f"score distribution: {scored.get('distribution', {})}")
    print(f"annotatable       : {scored.get('annotatable_count')}")
    print(f"evidence types    : {scored.get('evidence_type_counts', {})}")
    per = scored.get("per_paper", [])
    if per:
        print(f"\n  {'PMID':<12}{'score':<7}{'annot':<7}{'evidence_type':<20}justification")
        print("  " + "-" * 92)
        for pp in sorted(per, key=lambda x: (x.get('score') or 0), reverse=True):
            just = (pp.get('justification') or '')[:46]
            print(f"  {str(pp.get('pmid','')):<12}{str(pp.get('score')):<7}"
                  f"{str(pp.get('annotatable')):<7}{str(pp.get('evidence_type') or ''):<20}{just}")
    sel_pmids = [str(p.get('pmid')) for p in selected]
    print(f"\nSELECTED {len(selected)} paper(s) for full text: {', '.join(sel_pmids) or '(none)'}")

    # ---------- 2) Full-text resolution ----------
    _rule("2) FULL-TEXT RESOLUTION — where each selected PMID's text came from")
    manifest = (getattr(ga, "fulltext_manifest", {}) or {}).get(gene, {})
    if not manifest:
        print("No manifest (no selected PMIDs or resolution failed).")
    else:
        n = {k: sum(1 for v in manifest.values() if v.get("source") == k)
             for k in ("pdf", "xml", "miss")}
        print(f"{n['pdf']} local PDF · {n['xml']} PMC XML · {n['miss']} miss  (of {len(manifest)})\n")
        print(f"  {'PMID':<12}{'source':<8}path")
        print("  " + "-" * 66)
        for pmid, entry in manifest.items():
            print(f"  {pmid:<12}{_FT_LABEL.get(entry.get('source'), '?'):<8}{entry.get('path', '')}")

    # ---------- 3) Extraction + merge confidences ----------
    _rule("3) FULL-TEXT EXTRACTION + MERGE — reactions & confidences")
    reactions = (getattr(ga, "fulltext_reactions", {}) or {}).get(gene, [])
    if not reactions:
        merged_path = os.path.join(RESULTS_DIR, f"{gene.lower()}_merged.json")
        if os.path.isfile(merged_path):
            try:
                reactions = json.load(open(merged_path))
                print(f"(read {merged_path})")
            except Exception as e:
                print(f"(could not read {merged_path}: {e})")
    if not reactions:
        print("No merged reactions produced (all misses, extraction failed, or module absent).")
    else:
        ars = [r.get("annotation_result", r) for r in reactions]
        ext_conf = _mean([ar.get("confidence") for ar in ars])
        mrg_vals = [ar.get("merge_confidence") for ar in ars
                    if isinstance(ar.get("merge_confidence"), (int, float))]
        print(f"merged reactions        : {len(reactions)}")
        print(f"mean extraction conf.   : {ext_conf}")
        print(f"mean merge confidence   : {_mean(mrg_vals)}")
        print(f"min merge confidence    : {round(min(mrg_vals), 3) if mrg_vals else None}")
        print(f"\n  {'#':<4}{'ext_conf':<10}{'merge_conf':<12}reaction")
        print("  " + "-" * 74)
        for i, ar in enumerate(ars, 1):
            name = (ar.get("name") or ar.get("reaction") or ar.get("summation") or "")
            if isinstance(name, list):
                name = " / ".join(map(str, name))
            print(f"  {i:<4}{str(ar.get('confidence')):<10}{str(ar.get('merge_confidence')):<12}{str(name)[:52]}")

    # ---------- 4) OpenAI review ----------
    _rule("4) OPENAI REVIEW (gpt-5.6-luna) — cross-model second opinion")
    review_path = os.path.join(RESULTS_DIR, f"{gene.lower()}_review.md")
    if not os.path.isfile(review_path):
        print(f"No review file at {review_path} (review skipped, or no merged run to review).")
    else:
        txt = open(review_path).read()
        m = re.search(r'SCORE:\s*([\d.]+)', txt)
        print(f"review file : {review_path}")
        print(f"OpenAI SCORE: {m.group(1) + '/10' if m else 'not reported'}")
        print(f"\n----- review.md (first 60 lines) -----")
        print("\n".join(txt.splitlines()[:60]))

    _rule("DONE — front half only; Phases 1-5 not run.")


if __name__ == "__main__":
    asyncio.run(main())
