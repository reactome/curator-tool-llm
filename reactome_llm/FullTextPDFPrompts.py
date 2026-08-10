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
      - regulatedBy      : positiveRegulation / negativeRegulation / requirement, each entry
                           carrying an optional `note` — a FLAG FOR A CURATOR naming the
                           experimental condition that direction was observed under, set only
                           when the regulator's effect is condition-dependent
      - compartment      : cellular location
      - condition        : general biological condition/state (coarse; groups reactions)
      - summation        : Summation instance {text, literatureReference}
      - relationships    : "EntityA - relationship_type -> EntityB" lines
      - evidence         : list of verbatim source-text excerpts the reaction was drawn from
                           (provenance). Single citation pool for the WHOLE reaction — every
                           subsection filled in (regulatedBy, catalystActivity, compartment,
                           condition, input/output) must have a supporting excerpt here. May
                           include text quoted from an adjacent chunk when the supporting
                           passage spans a chunk boundary, and the same excerpt may be cited
                           by more than one reaction.
      - context_used     : list of the contexts consulted for this reaction — to resolve a
                           reference OR to quote a boundary-spanning excerpt (["none"] if the
                           current chunk sufficed)

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

CATALYST vs REGULATOR — assign an entity as the "catalyst" ONLY when the evidence shows it
DIRECTLY PERFORMS the reaction: it is the enzyme carrying out the chemistry on the substrate.
An entity that FACILITATES, PROMOTES, ENHANCES, ENABLES, or IS REQUIRED FOR a reaction is a
REGULATOR, not automatically a catalyst — put it in "regulatedBy" and leave "catalyst" null.
  - "X was identified to facilitate Y" makes X a positiveRegulation regulator OF Y, not the
    catalyst of Y, and not a reaction of its own.
  - Never infer a catalyst from association, correlation, or a screen hit. If no entity in the
    evidence is explicitly shown performing the reaction, "catalyst" is null.
  - Many events have no catalyst at all — transport, translocation, binding, dissociation, and
    conformational change are usually uncatalyzed. Null is the correct answer there, not a
    slot to fill with whichever protein is nearby.

CONDITION-DEPENDENT REGULATION — the "note" field on a regulatedBy entry is a FLAG FOR A HUMAN
CURATOR. The same regulator can act in OPPOSITE directions depending on the experimental
condition: a factor that activates a reaction in depolarized mitochondria may be inhibitory or
dispensable in the basal state. That is real biology, not a contradiction — but a curator must
see it, because once records from different experiments are pooled the two directions look like
conflicting data.
  - Set "note" when the paper shows, or the excerpt you are citing makes clear, that THIS
    regulatory direction holds only under a particular condition. Name that condition in the
    note, in the paper's own terms (e.g. "positive only after CCCP-induced depolarization").
  - Leave "note" null when the regulation is unconditional, or when the paper gives you no
    basis for calling it condition-dependent. Do NOT speculate about a condition, and do NOT
    use the note for anything other than this flag.
  - The note does not replace the "condition" field: "condition" is the coarse reusable
    category for the WHOLE reaction, the note is specific to ONE regulator's direction.

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
  - For each reaction, record in its "context_used" field EVERY context you actually consulted
    for it — a LIST, since resolving one reaction can take more than one direction. Use
    "previous_chunk" (CONTEXT A), "two_chunks_back" (CONTEXT B), "next_chunk" (CONTEXT C), or
    the single entry "none" if the CURRENT CHUNK alone sufficed. This field must be accurate
    in BOTH of the cases where you look outside the current chunk:
      * you looked BACKWARD or FORWARD to resolve an entity, antecedent, or outcome, AND
      * you quoted an evidence excerpt from that context (boundary-spanning support above).
    If any excerpt in "evidence" was quoted from CONTEXT C, "context_used" must contain
    "next_chunk". Never report "none" while citing text you took from a context section.

For each reaction, express it using the Reactome ReactionlikeEvent data model fields, and
also as relationship lines in this exact format:
EntityA - relationship_type -> EntityB

Use precise relationship types (e.g. phosphorylates, binds, ubiquitinates, cleaves,
activates, inhibits, localizes_to, translocates_to).

For EACH reaction, identify the biological CONDITION under which it occurs and put ONLY the
GENERAL condition in the "condition" field — a coarse, reusable category that many reactions in
the same context would share (so downstream steps can group reactions by it), with no
experiment-specific detail. If no condition is stated or implied, set "condition" to null.

The "evidence" field is your CITATION — a LIST of the VERBATIM source-text excerpt(s) you drew
this reaction from (quote the actual sentence(s), copied exactly). These excerpts are the proof
of provenance and will naturally contain the experimental system, treatment, dose, mutant, and
method — that is where all such specific detail lives. Add one excerpt per place the reaction is
supported; if the reaction recurs in other chunks, its excerpts are pooled together later.

EVIDENCE THAT BLEEDS ACROSS THE CHUNK BOUNDARY — chunks cut the paper mid-sentence and
mid-paragraph, so the passage supporting a reaction is often split across two chunks. You can
always see the whole passage from the EARLIER side of a cut, because CONTEXT C gives you the
next chunk's raw text. So a split passage is handled from the earlier chunk, never patched
together from the later one:
  - NEVER DROP AN EXCERPT BECAUSE IT IS CUT OFF. LOOK
    FORWARD into CONTEXT C, find where the sentence continues, and cite the completed
    passage. Then record "next_chunk" in "context_used". Silently omitting a cut excerpt
    loses the citation for that reaction, which is worse than any imperfection in the quote.
  - A reaction whose supporting passage RUNS ON into the next chunk belongs to THIS chunk.
    Extract it here and cite the WHOLE passage: the text in the CURRENT CHUNK together with
    its continuation from CONTEXT C, joined into the complete sentence(s) they form.
  - If the CURRENT CHUNK instead OPENS mid-sentence, the missing words are BEHIND you and
    CONTEXT C cannot supply them. Prefer the record already made from the previous chunk: if
    that reaction is listed in CONTEXT A, it is captured, so do not re-extract it. If it is
    NOT in CONTEXT A, extract it here and quote the partial sentence the current chunk does
    contain — an incomplete quote is still a citation; omitting it leaves the reaction uncited.
  - Reproduce the author's WORDING exactly — do not paraphrase, summarize, or add words. You
    MAY tidy mechanical text artefacts so the excerpt reads as the running prose it is:
    rejoin a word split across a line ("mitochon- dria" -> "mitochondria"), close up a line
    break inside a sentence, and join a sentence split at the chunk boundary.
  - This is for EVIDENCE ONLY. Quoting the next chunk never licenses extracting a reaction
    that is wholly described there — the reaction must still be stated, at least in part, in
    the CURRENT CHUNK.
  - Whenever an excerpt comes from a context section, you MUST record that context in
    "context_used" (see below).

EVERY SUBSECTION MUST BE CITED IN "evidence" — the one "evidence" list is the citation pool for
the WHOLE reaction, not just for its name and participants. It must contain a verbatim excerpt
supporting EVERY subsection you fill in:
  - regulatedBy — the regulation is nearly always shown by a DIFFERENT sentence than the
    reaction itself (the knockout, mutant, inhibitor, or requirement experiment). Quote THAT
    sentence, for each regulator you list.
  - catalystActivity — quote the sentence establishing that this enzyme carries out the event.
  - compartment — quote the sentence stating where it happens.
  - condition — quote the sentence stating the condition or treatment under which it occurs.
  - input / output — quote the sentence naming the participants and products.
  A subsection you cannot quote for must be left null or empty rather than asserted uncited.
  Do not add facts to "summation" that no excerpt in the list supports.

Excerpt rules:
  - Every excerpt must support the reaction itself OR one of its subsections above. A sentence
    that merely mentions one participant while asserting nothing about this reaction or any of
    its subsections is NOT evidence.
  - The SAME excerpt MAY be cited by several reactions when it genuinely supports each of
    them — a sentence describing two molecular events is real evidence for both, so quote it
    on both. Do not withhold an excerpt because another reaction already uses it.
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
        "catalyst": "<the enzyme that DIRECTLY performs this reaction, or null — an entity that merely facilitates, promotes, enables or is required for it is a regulator, not a catalyst>",
        "molecularFunction": "<GO molecular function term if known, or null>"
      }},
      "regulatedBy": [
        {{
          "regulationType": "<positiveRegulation | negativeRegulation | requirement>",
          "regulator": "<gene, protein, or small molecule>",
          "note": "<null, OR a curator flag when this regulator's effect is condition-dependent — name the experimental condition THIS direction was observed under>"
        }}
      ],
      "compartment": "<cellular compartment, or null>",
      "condition": "<general biological state/context only — coarse and reusable, no specific detail, or null>",
      "summation": {{
        "text": "<factual description of the molecular event; state the general condition here. Keep it about the biology — the raw experimental specifics live in the evidence excerpts.>",
        "literatureReference": ["<PMID or citation string if mentioned, else empty list>"]
      }},
      "relationships": ["EntityA - relationship_type -> EntityB"],
      "evidence": ["<verbatim excerpt supporting this reaction>", "<every additional excerpt where this reaction is stated — include all of them>", "<an excerpt for EACH subsection you filled in: regulator, catalyst, compartment, condition>", "<the continuing text from CONTEXT C if the passage ran past the chunk boundary>"],
      "context_used": ["<none | previous_chunk | two_chunks_back | next_chunk — list EVERY context you consulted, whether to resolve the reaction or to quote an excerpt>"],
      "confidence": <float 0-1, confidence this reaction is correct and well-supported>
    }}
  ]
}}

If no biochemical reaction is present in the CURRENT CHUNK, return {{"reactions": []}}.
Don't speculate or just list interacting genes if there is no information.

{ctx}=== CURRENT CHUNK — extract reactions from THIS text only ===
{current_chunk}"""
