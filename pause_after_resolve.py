"""Run the REAL pipeline up through full-text resolution, then STOP before Phase 1.

Drives run_pipeline's own annotator (accession resolve -> placement gate -> rerank
targets -> Stage-1 retrieve + cross-encoder + LLM judge -> full-text resolution),
then raises a sentinel the instant Phase 1 would start. No full-text analysis, no
Phases 1-5 -- just the manifest showing, for each judge-selected PMID, whether a PMC
XML was found/downloaded (source "xml") or marked a miss.

Local index is forced empty so every PMID takes the PMC path (cached XML -> idconv
/efetch -> miss), matching "no local full-text papers for this gene".

    conda run -n paperqa python pause_after_resolve.py TANC1
"""
import os
import sys

os.environ.setdefault("TOKEN_PROFILE", "1")

import asyncio
import argparse

sys.path.append("reactome_llm")
sys.path.append("examples")

from dotenv import load_dotenv
load_dotenv()

from crewai_annotation_examples import build_annotators
from CrewAILiteratureAnnotator import AnnotationRequest, CrewAILiteratureAnnotator


class _StopAfterResolve(Exception):
    """Sentinel: full-text resolution is done, do not enter Phase 1."""


_FT_LABEL = {"pdf": "PDF", "xml": "XML", "miss": "miss"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gene", nargs="?", default="TANC1")
    ap.add_argument("--max-papers", type=int, default=5)
    args = ap.parse_args()
    gene = args.gene

    # Halt the pipeline the moment Phase 1 would begin -- resolution already ran and
    # published the manifest on the shared gene_annotator by this point.
    async def _stop(self, *a, **k):
        raise _StopAfterResolve("STOP_AFTER_RESOLVE")

    CrewAILiteratureAnnotator._phase_1_literature_extraction = _stop

    _, crewai = build_annotators(verbose=False)

    request = AnnotationRequest(
        gene=gene,
        papers=[],
        max_papers=args.max_papers,
        enable_full_text=True,
        enable_literature_search=True,
        fulltext_index={},  # no local PDFs -> force the PMC download path for every PMID
    )

    print(f"\n{'=' * 72}\nRunning pipeline for {gene} up to full-text resolution "
          f"(max_papers={args.max_papers})\n{'=' * 72}", flush=True)

    try:
        await crewai.annotate_literature(request)
        print("\n!! Pipeline ran to completion without hitting the pause point.")
    except Exception as e:
        # annotate_literature wraps everything as CrewAIAnnotationError; the sentinel's
        # name survives in the message.
        if "STOP_AFTER_RESOLVE" not in str(e) and not isinstance(e, _StopAfterResolve):
            raise
        print("\n-- Paused after full-text resolution (Phase 1 not started). --")

    manifest = (getattr(crewai.gene_annotator, "fulltext_manifest", {}) or {}).get(gene, {}) \
        or (getattr(crewai, "fulltext_manifest", {}) or {})
    selected = (getattr(crewai.gene_annotator, "judged_papers", {}) or {}).get(gene, {}) \
        .get("papers", [])

    print(f"\nJudge selected {len(selected)} PMIDs for {gene}: "
          f"{', '.join(str(p.get('pmid')) for p in selected) or '(none)'}")

    if not manifest:
        print("\nNo manifest produced (no selected PMIDs, or resolution failed).")
        return

    n_pdf = sum(1 for v in manifest.values() if v.get("source") == "pdf")
    n_xml = sum(1 for v in manifest.values() if v.get("source") == "xml")
    n_miss = sum(1 for v in manifest.values() if v.get("source") == "miss")
    print(f"\nFull-text resolution: {n_pdf} local PDF · {n_xml} PMC XML · "
          f"{n_miss} miss  (of {len(manifest)})\n")
    print(f"  {'PMID':<12}{'source':<8}path")
    print("  " + "-" * 68)
    for pmid, entry in manifest.items():
        print(f"  {pmid:<12}{_FT_LABEL.get(entry.get('source'), '?'):<8}"
              f"{entry.get('path', '')}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
