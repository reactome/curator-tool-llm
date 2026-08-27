"""Deterministic convert step: extracted reactions -> Reactome data-model instances.

This is the second half of the pipeline (what used to be the Phase-2 curator AGENT). It is a
single bounded, structured LLM call -- NO agent, no tool loop:

    build_instances(gene, reactions, accession=..., placement=..., placement_status=...,
                    target_pathways=...) -> ReactomeDataModel

Inputs are the reactions the extractor already produced (full-text OR abstract, each carrying
its own participants/roles/evidence/confidence/provenance), plus the deterministically-resolved
context the front half already computed: the verified UniProt accession and the pathway-placement
outcome. The LLM's job is the residual JUDGMENT that can't be grounded up front -- creating the
entity/complex instances, mapping each reaction onto the Reactome schema, and grouping the
reactions under a pathway -- NOT re-deriving reactions or re-deciding placement.

Placement modes:
  * target_pathways given      -> anchor the reactions under those pathways.
  * confident placement        -> anchor under the suggested primary Reactome pathway (verify).
  * gate fail / no placement    -> emit a PROPOSED de-novo pathway, clearly flagged "no confident
                                   placement", for a curator to place. The reactions are still
                                   returned; only their pathway home is left provisional.

`with_structured_output(ReactomeDataModel)` forces schema-valid output, so the caller gets a
typed object back and no JSON scraping is needed. Any failure degrades to an empty-but-valid
model so the pipeline continues.
"""
import os
import re
import sys
import json
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_REACTOME_LLM = os.path.join(PROJECT_ROOT, "reactome_llm")
if _REACTOME_LLM not in sys.path:
    sys.path.insert(0, _REACTOME_LLM)

from langchain_anthropic import ChatAnthropic
import token_profiler
from ModelConfig import REACTOME_MODEL_NAME
from ReactomeModels import ReactomeDataModel

logger = logging.getLogger(__name__)


# A gene with many reactions produces a large structured output (entities + complexes + reactions
# + pathways). 8000 was too low -- SHANK3 (30 reactions) truncated after entities/complexes with
# ZERO reactions emitted. 32000 still truncated a rich multi-paper gene: SHANK3 (51 reactions)
# came out at 46 with an entity cut mid-object (QA flagged "JSON truncated mid-entity"). 64000 is
# claude-sonnet-4-6's max output ceiling and gives ~2x headroom. NOTE: this is the ceiling, not a
# fix for arbitrarily large genes -- a gene whose annotation exceeds 64k output tokens still needs
# the conversion BATCHED (convert N reactions at a time and stitch), which removes the cap entirely.
_CONVERT_MAX_TOKENS = 64000


def _model():
    """A token-profiled sonnet model at low temperature for deterministic schema conversion."""
    return ChatAnthropic(model=REACTOME_MODEL_NAME, temperature=0.1, max_tokens=_CONVERT_MAX_TOKENS,
                         callbacks=token_profiler.langchain_callbacks())


def _pmid_of(source: str) -> str:
    """Best-effort PMID from a reaction's source tag ('12345', '12345 (abstract)', a filename)."""
    m = re.match(r"\s*(\d{5,9})", str(source or ""))
    return m.group(1) if m else ""


def _reaction_items(reactions):
    """Flatten the extractor's wrapped reactions into the payload the prompt shows the model:
    the reaction's own fields plus its provenance/source/PMID hint."""
    items = []
    for r in reactions or []:
        ar = r.get("annotation_result", r) if isinstance(r, dict) else r
        src = r.get("source", "") if isinstance(r, dict) else ""
        prov = r.get("provenance", "") if isinstance(r, dict) else ""
        items.append({
            "provenance": prov or "fulltext",
            "pmid": _pmid_of(src),
            "reaction": ar,
        })
    return items


def _placement_directive(gene, placement, placement_status, target_pathways):
    """Render the pathway-grouping instruction from the front half's placement outcome."""
    if target_pathways:
        pw = ", ".join(target_pathways)
        return (f"PATHWAY PLACEMENT: group the reactions under the caller-specified pathway(s): "
                f"{pw}. Create a Pathway instance for each and list the relevant reactions in "
                f"its hasEvent.")
    if placement and placement.get("primary"):
        primary = placement["primary"]
        name = primary.get("pathway_name", "")
        partners = ", ".join(primary.get("mapped_genes", []) or []) or "n/a"
        return (f"PATHWAY PLACEMENT: interaction-partner enrichment confidently places {gene} in "
                f"the existing Reactome pathway **{name}** (supported by partners: {partners}). "
                f"Create a Pathway instance named '{name}' and list the reactions in its hasEvent. "
                f"Verify this is consistent with the reactions; if it clearly is not, fall back to "
                f"a proposed pathway (next rule).")
    # Gate fail / no confident placement.
    msg = ""
    if placement_status and placement_status.get("message"):
        msg = " " + placement_status["message"]
    return (f"PATHWAY PLACEMENT: there is NO confident pathway placement for {gene}.{msg} Do NOT "
            f"invent an existing Reactome pathway. Instead create ONE proposed de-novo Pathway "
            f"named '{gene} candidate pathway (proposed — no confident placement)', list all the "
            f"reactions in its hasEvent, and state in its summation that a curator must decide the "
            f"final placement.")


def build_prompt(gene, reactions, accession, placement, placement_status, target_pathways,
                 fix_notes=None):
    items = _reaction_items(reactions)
    reactions_json = json.dumps(items, indent=2, default=str)
    accession_line = (
        f"VERIFIED IDENTIFIER: the canonical UniProt accession for {gene} is {accession}. Use it "
        f"exactly as the identifier/referenceEntity of the {gene} protein entity; do not invent "
        f"another.\n" if accession else "")
    placement_line = _placement_directive(gene, placement, placement_status, target_pathways)
    # QA repair loop: when a prior conversion of these SAME reactions was rejected by QA, its
    # corrections are injected here so the re-conversion fixes them instead of repeating them.
    fix_block = "" if not fix_notes else f"""

QA CORRECTIONS — a previous conversion of THESE SAME reactions was reviewed by a Reactome QA
expert and REJECTED. Fix every issue below in this re-conversion and do NOT reintroduce them
(correct UniProt identifiers; model multi-protein complexes as Complex, not EWAS; attach
literatureReference PMIDs; set compartments where the evidence states one; drop off-target /
positive-control reactions the reviewer flagged as out of scope):
{fix_notes}
"""

    return f"""You are a Reactome biocurator. You are given biochemical reactions that have ALREADY
been extracted from the literature for the gene {gene} (each with its participants, roles,
evidence excerpts, self-assessed confidence, and a provenance tag of 'fulltext' or 'abstract').
Your job is to TRANSLATE these reactions into valid Reactome data-model instances -- NOT to
invent new reactions, not to re-extract, and not to add reactions that are not in the input.

{accession_line}
INPUT REACTIONS (JSON):
```json
{reactions_json}
```

Do ALL of the following:
1. ENTITIES — create one EntityWithAccessionedSequence for {gene} (using the verified accession
   above) and one for every DISTINCT participant (input, output, catalyst, regulator) named
   across the reactions. Normalize obvious name variants of the same molecule to a single entity.
   Only set `compartment` when a reaction explicitly states one; otherwise leave it "".
2. REACTIONS — create one Reaction per input reaction, carrying its fields over faithfully:
   reactionType, input, output, catalystActivity (the catalyst entity, if any), regulatedBy
   (format each as 'regulationType: regulator (note)'), compartment, summation, evidence (copy the
   verbatim excerpts), confidence, and provenance ('fulltext' or 'abstract' from the input).
   Put the reaction's PMID in literatureReference. Every entity you name in a reaction's
   input/output/catalystActivity MUST also exist as an entity (rule 1) — no dangling references.
3. COMPLEXES — when a binding reaction forms a named complex (or a dissociation breaks one up),
   create a Complex whose components are the member entities, and reference it by the SAME name in
   the relevant reaction's input/output.
4. {placement_line}

Rules:
- Trust the extracted reactions; do not upgrade a weak/abstract-provenance reaction into a
  stronger claim than its evidence supports. Preserve each reaction's provenance and confidence.
- Do not fabricate participants, modifications, compartments, or catalysts that the input
  reactions do not contain.
- Reuse a single entity instance for a molecule that appears in several reactions.
- ONLY create entities for molecular species (proteins, complexes, small molecules, modified
  proteins). Do NOT create entities for phenotypes, measurements, or outcomes such as
  "decreased EPSC amplitude", "increased synapse density", or "blocked transmission" -- those are
  observations, not Reactome PhysicalEntities. Drop reactions whose only output is such an
  observation.
- Keep the output compact so nothing is truncated: for each reaction's `evidence`, include at most
  the 2 most relevant verbatim excerpts, and keep `summation` to one sentence. Every input
  reaction that survives the rule above MUST appear as a Reaction — do not stop early.
{fix_block}
Return the Reactome data model for {gene}."""


def build_instances(gene, reactions, accession=None, placement=None,
                    placement_status=None, target_pathways=None,
                    fix_notes=None) -> ReactomeDataModel:
    """Convert extracted reactions into a ReactomeDataModel via one structured LLM call.

    Never raises: on an empty input or any failure it returns an empty-but-valid model so the
    pipeline continues. Blocking (.invoke) -- call it from a worker thread on the async path.
    """
    reactions = reactions or []
    if not reactions:
        logger.info("convert: no reactions for %s -- empty data model", gene)
        return ReactomeDataModel(gene=gene)

    prompt = build_prompt(gene, reactions, accession, placement, placement_status,
                          target_pathways, fix_notes=fix_notes)
    try:
        model = _model().with_structured_output(ReactomeDataModel)
        result = model.invoke(prompt)
    except Exception as e:
        logger.warning("convert: reaction->instance conversion failed for %s: %s", gene, e)
        return ReactomeDataModel(gene=gene)

    # with_structured_output returns a validated ReactomeDataModel; guard the gene field.
    if isinstance(result, ReactomeDataModel):
        if not result.gene:
            result.gene = gene
        logger.info("convert %s: %d entities, %d complexes, %d reactions, %d pathways",
                    gene, len(result.entities), len(result.complexes),
                    len(result.reactions), len(result.pathways))
        return result
    logger.warning("convert: unexpected result type %s for %s", type(result), gene)
    return ReactomeDataModel(gene=gene)
