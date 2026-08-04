"""
FullTextPDFPrompts.py
---------------------
Claude prompt templates for the FullTextPDF pipeline.

    from FullTextPDFPrompts import build_extraction_prompt
"""


def build_extraction_prompt(current_chunk, prev_contexts=None, next_context=None):
    """
    Prompt for extracting biochemical reactions from the CURRENT chunk of a paper.

    The model extracts reactions from the CURRENT CHUNK only. Context is reference
    material to resolve entities/references that appear in the current chunk but are
    described in a neighbouring chunk — never a source of new reactions. The model
    decides which DIRECTION to consult based on the nature of the missing reference:
    backward for antecedents (nearest previous chunk first, then two-back), forward for
    continuations.

    Args:
      current_chunk : text of the chunk to extract reactions from.
      prev_contexts : list of previously-extracted reaction NOTES, ordered MOST RECENT
                      FIRST — index 0 = immediately previous chunk (CONTEXT A),
                      index 1 = two chunks back (CONTEXT B). Up to 2 used.
      next_context  : RAW text of the next chunk (CONTEXT C, look-ahead).

    Output format follows the Reactome ReactionlikeEvent data model:
      https://curator.reactome.org/cgi-bin/classbrowser?DB=gk_central&CLASS=ReactionlikeEvent

    ReactionlikeEvent fields per reaction:
      - name             : display name of the reaction
      - reactionType     : transition / omitted / binding / dissociation / blackBoxEvent
      - input            : PhysicalEntity list — reactants/substrates consumed
      - output           : PhysicalEntity list — products produced
      - catalystActivity : enzyme + GO molecular function
      - regulatedBy      : positiveRegulation / negativeRegulation / requirement
      - compartment      : cellular location
      - condition        : general biological condition/state (coarse; groups reactions)
      - summation        : Summation instance {text, literatureReference}
      - relationships    : "EntityA - relationship_type -> EntityB" lines
      - evidence         : list of verbatim source-text excerpts the reaction was drawn from (provenance)

    NOTE: confidence is Claude's SELF-ASSESSED rating (subjective, 0-1).
    Objective accuracy is measured separately via cosine similarity vs Neo4j.
    """
    prev_contexts = prev_contexts or []
    prev1 = prev_contexts[0] if len(prev_contexts) > 0 else ""
    prev2 = prev_contexts[1] if len(prev_contexts) > 1 else ""

    ctx = ""
    if prev1:
        ctx += ("=== CONTEXT A — previous chunk (most recent), reactions already extracted "
                "[reference only, do NOT re-extract] ===\n" + prev1 + "\n\n")
    if prev2:
        ctx += ("=== CONTEXT B — two chunks back, reactions already extracted "
                "[reference only, do NOT re-extract] ===\n" + prev2 + "\n\n")
    if next_context:
        ctx += ("=== CONTEXT C — next chunk (RAW text), look-ahead [reference only] ===\n"
                + next_context + "\n\n")

    return f"""You are an expert Reactome biocurator. Extract the biochemical reactions
described in the CURRENT CHUNK below.

A biochemical reaction requires:
  - At least two named biological entities (genes, proteins, metabolites)
  - A directional relationship between them (e.g. phosphorylates, binds, cleaves,
    activates, inhibits, ubiquitinates, translocates, recruits)
  - Evidence this is a real molecular event

EXTRACT ONLY GENUINE BIOCHEMICAL REACTIONS — actual molecular events that occur in the biology
(an enzyme modifies a substrate; molecules bind; a complex forms or dissociates; a molecule is
transported/translocated; a molecule is synthesized or degraded). Describe each reaction by its
underlying molecular event and STRIP the experimental framing — do not fold the assay,
technique, instrument, tag, or detection method used to observe it into the reaction.

NORMALIZE each reaction to its underlying biological molecular event, so the SAME event is
captured ONCE no matter how many experiments demonstrate it:
  - Use plain biological entity names; strip affinity/epitope/fluorescent tags and fusion
    partners (e.g. HA-, His-, MBP-, GST-, FLAG-, GFP-, YFP-).
  - Normalize reagents and ortholog/species stand-ins to the entity they represent (a
    recombinant protein or an ortholog used as an in-vitro proxy is the SAME entity as the
    native one).
  - Do NOT let the experimental SYSTEM (in vitro / cell-free / recombinant / in cells), the
    inducing TREATMENT, the DETECTION method, or a tool MUTANT (catalytic-dead trap,
    phosphomimetic, phospho-null) create separate reactions — the same enzyme + substrate +
    molecular action is ONE reaction regardless of how it was shown. The raw experimental
    detail is preserved verbatim in the "evidence" excerpts (below), not in the reaction identity.

DO NOT extract experimental observations that are not themselves molecular reactions:
  - Negative or failed results (something "does not…", "fails to…", "no change", "no effect",
    "is unaffected", "is not required", "remained unchanged").
  - Pure detection / measurement / assay readouts that assert no new molecular event (band or
    gel shifts, signal intensities, peak detection, quantitation, colocalization, imaging).
  - Comparative measurements between variants or conditions ("A does/binds X more than B").
  - Controls, reagent treatments, or mutant/knockout/inhibitor manipulations performed only to
    test or interpret a result.

When a mutant, knockout, inhibitor, or negative result establishes that a factor is REQUIRED for
(or regulates) a real reaction, record it as a `regulatedBy` entry ON that reaction — NOT as its
own separate reaction.

FOCUS — this is the most important instruction:
  - Extract reactions ONLY from the CURRENT CHUNK section (the last section below).
  - Do NOT extract, repeat, or re-describe any reaction that appears only in a CONTEXT
    section. Context is reference material, NOT a source of reactions.
  - Work from the current chunk alone by default. ONLY IF a reaction in the current chunk
    refers to an entity or state you cannot resolve from the current chunk itself, decide
    which DIRECTION to look based on what is missing:
      * BACKWARD — if the reference is to something introduced or produced EARLIER (an
        antecedent, e.g. "the phosphorylated protein" with no prior mention in this chunk):
        consult CONTEXT A (previous chunk) FIRST; only if it is still unresolved, then also
        consult CONTEXT B (two chunks back).
      * FORWARD — if the current chunk's reaction appears cut off or CONTINUES into what
        follows (e.g. a product or outcome named only afterwards): consult CONTEXT C
        (next chunk raw text).
    Stop as soon as the reference is resolved; do not read further context than you need.
  - Use context only to complete/resolve a reaction stated in the current chunk — never to
    add a reaction that belongs to a neighbouring chunk.
  - For each reaction, record in its "context_used" field which context (if any) you had to
    consult to complete it: "none" if the CURRENT CHUNK alone sufficed, else "previous_chunk"
    (CONTEXT A), "two_chunks_back" (CONTEXT B), or "next_chunk" (CONTEXT C).

For each reaction, express it using the Reactome ReactionlikeEvent data model fields, and
also as relationship lines in this exact format:
EntityA - relationship_type -> EntityB

Use precise relationship types (e.g. phosphorylates, binds, ubiquitinates, cleaves,
activates, inhibits, localizes_to, translocates_to).

For EACH reaction, identify the biological CONDITION under which it occurs and put ONLY the
GENERAL condition in the "condition" field — a coarse, reusable category that many reactions in
the same context would share (so downstream steps can group reactions by it), with no
experiment-specific detail. If no condition is stated or implied, set "condition" to null.

The "evidence" field is your CITATION — a LIST of the VERBATIM source-text excerpt(s) from the
CURRENT CHUNK that you drew this reaction from (quote the actual sentence(s), copied exactly).
These excerpts are the proof of provenance and will naturally contain the experimental system,
treatment, dose, mutant, and method — that is where all such specific detail lives. Add one
excerpt per place the reaction is supported in this chunk; if the reaction recurs in other
chunks, its excerpts are pooled together later.

EVERY CLAIM MUST BE CITABLE — no field may assert more than its excerpts state:
  - Each excerpt must independently support THIS reaction: it must name the same
    participants AND state the same molecular action. A sentence that merely mentions one
    participant, or states a different action, is NOT evidence for this reaction.
  - Do NOT attach the same excerpt to several reactions as shared background. Quote a
    sentence for two reactions only if it genuinely asserts both.
  - HEDGED or UNRESOLVED statements are NOT evidence: "is suggested to be", "may",
    "might", "could", "has remained elusive", "is unclear", "remains unknown". If the only
    sentence you can quote for a reaction is hedged, do NOT emit the reaction.

Return ONLY a JSON object, no markdown:
{{
  "reactions": [
    {{
      "name": "<display name, e.g. 'PINK1 phosphorylates Parkin at Ser65'>",
      "reactionType": "<transition | omitted | binding | dissociation | blackBoxEvent>",
      "input": ["<PhysicalEntity consumed — reactant or substrate>"],
      "output": ["<PhysicalEntity produced — product>"],
      "catalystActivity": {{
        "catalyst": "<enzyme or catalyst, or null>",
        "molecularFunction": "<GO molecular function term if known, or null>"
      }},
      "regulatedBy": [
        {{
          "regulationType": "<positiveRegulation | negativeRegulation | requirement>",
          "regulator": "<gene, protein, or small molecule>"
        }}
      ],
      "compartment": "<cellular compartment, or null>",
      "condition": "<general biological state/context only — coarse and reusable, no specific detail, or null>",
      "summation": {{
        "text": "<factual description of the molecular event; state the general condition here. Keep it about the biology — the raw experimental specifics live in the evidence excerpts.>",
        "literatureReference": ["<PMID or citation string if mentioned, else empty list>"]
      }},
      "relationships": ["EntityA - relationship_type -> EntityB"],
      "evidence": ["<verbatim source-text excerpt supporting this reaction>", "<every additional excerpt where this reaction is stated — include all of them>"],
      "context_used": "<none | previous_chunk | two_chunks_back | next_chunk — which context you consulted, if any>",
      "confidence": <float 0-1, confidence this reaction is correct and well-supported>
    }}
  ]
}}

If no biochemical reaction is present in the CURRENT CHUNK, return {{"reactions": []}}.
Don't speculate or just list interacting genes if there is no information.

{ctx}=== CURRENT CHUNK — extract reactions from THIS text only ===
{current_chunk}"""
