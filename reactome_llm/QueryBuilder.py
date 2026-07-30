"""
Query builder for smarter literature retrieval.

Expands a bare gene name into a short biological description suitable for
embedding-based re-ranking, optionally grounded with existing Reactome
pathway context.
"""

import json
import os
import re
import urllib.error
import urllib.request

import ReactomeNeo4jUtils as neo4jutils
from ModelConfig import create_reactome_chat_model

UNIPROT_REST_URL = "https://rest.uniprot.org/uniprotkb/{}.json"

# Resolve the pathway-gene flat file relative to this module (repo_root/resources/...),
# so enrichment works regardless of the caller's cwd (e.g. running from notebooks/).
_PATHWAY_GENE_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "resources",
                 "ReactomePathwayGenes_Ver_91.txt")
)

# Lazily-constructed MongoFILoader singleton — its __init__ hits Mongo to load the gene
# index, so we build it once and reuse it across genes rather than per call.
_fi_loader = None


def _get_fi_loader():
    global _fi_loader
    if _fi_loader is None:
        from ProteinProteinInteractionsLoader import MongoFILoader
        _fi_loader = MongoFILoader()
    return _fi_loader

# Reactome top-level / mega-generic pathways whose huge result pools drown out relevance
# ranking. Filtered out of pathway-based queries so specific leaf pathways dominate.
GENERIC_PATHWAYS = {
    "Signal Transduction", "Developmental Biology", "Disease", "Immune System",
    "Adaptive Immune System", "Innate Immune System", "Metabolism", "Metabolism of proteins",
    "Metabolism of RNA", "Metabolism of lipids", "Gene expression (Transcription)", "Cell Cycle",
    "Hemostasis", "Cellular responses to stimuli", "Cellular responses to stress", "Cancer Hallmarks",
    "Programmed Cell Death", "Vesicle-mediated transport", "Membrane Trafficking",
    "Transport of small molecules", "Neuronal System", "Extracellular matrix organization",
    "Cytokine Signaling in Immune system", "Post-translational protein modification",
    "Cellular Senescence", "Muscle contraction", "Reproduction", "Circadian Clock",
    "Organelle biogenesis and maintenance", "Chromatin organization", "DNA Repair", "DNA Replication",
    "Autophagy", "Signaling by Interleukins", "Homeostasis", "Sensory Perception",
    "Activating Invasion and Metastasis", "Invasive Phase", "Breakthrough Phase",
}

def build_query_and_search_terms(gene: str) -> tuple[str, str]:
    """Generate both a short PubMed-friendly search string and a longer 
    biological description for a gene.

    Args:
        gene (str): Gene symbol (e.g. 'SHANK3')

    Returns:
        tuple[str, str]: (search_terms, full_description)
            search_terms: short keyword-style string suitable for PubMed E-Search
            full_description: longer biological description suitable for embedding
    """
    pathways = neo4jutils.query_pathways_for_gene(gene)

    if pathways:
        pathway_names = [p["pathway"] for p in pathways]
        pathway_list_str = ", ".join(pathway_names)
        prompt = (
            f"The gene {gene} is known to participate in the following Reactome "
            f"pathways: {pathway_list_str}.\n\n"
            f"Provide two things, clearly labeled exactly as shown:\n\n"
            f"SEARCH TERMS: 4-6 concise keywords or short phrases (specific protein "
            f"names, interaction partners, or pathway terms) as a COMMA-SEPARATED "
            f"list, e.g. 'term one, term two, term three'. Do NOT number them, do "
            f"NOT write 'OR', and do NOT write a full sentence.\n\n"
            f"DESCRIPTION: {gene}'s known biological role, interaction partners, "
            f"and pathway context in 2-3 sentences."
        )
    else:
        prompt = (
            f"Provide two things, clearly labeled exactly as shown, for the gene {gene}:\n\n"
            f"SEARCH TERMS: 4-6 concise keywords or short phrases as a COMMA-SEPARATED "
            f"list, e.g. 'term one, term two, term three'. Do NOT number them, do NOT "
            f"write 'OR', and do NOT write a full sentence.\n\n"
            f"DESCRIPTION: {gene}'s known biological role, interaction partners, "
            f"and pathway context in 2-3 sentences."
        )

    model = create_reactome_chat_model()
    response = model.invoke(prompt)
    text = response.content

    # Split the response into the two labeled sections
    search_terms = ""
    description = ""

    if "SEARCH TERMS:" in text and "DESCRIPTION:" in text:
        search_part = text.split("SEARCH TERMS:")[1].split("DESCRIPTION:")[0]
        description_part = text.split("DESCRIPTION:")[1]
        search_terms = _to_pubmed_or_query(gene, search_part)
        description = description_part.strip()
    else:
        # fallback: if the model didn't follow the format, treat whole thing as description
        description = text.strip()
        search_terms = gene  # fallback to bare gene name for search

    return search_terms, description


def _or_join_terms(terms: list[str]) -> str:
    """Join a list of raw terms into a valid PubMed OR query.

    PubMed's automatic term mapping treats a space between two distinct concepts
    as AND, so a bare space-separated list gets AND-chained and returns 0 hits.
    We quote multi-word phrases (so each stays a single phrase search), drop list
    bullet/numbering artifacts and surrounding quotes, de-duplicate case-insensitively
    (preserving order), and join everything with OR.
    """
    phrases = []
    seen = set()
    for p in terms:
        p = p.strip().strip("\"'").lstrip("-*0123456789.) ").strip()
        if not p or p.lower() in seen:
            continue
        seen.add(p.lower())
        phrases.append(f'"{p}"' if " " in p else p)
    return " OR ".join(phrases)


def _to_pubmed_or_query(gene: str, raw_terms: str) -> str:
    """Turn a comma-separated list of LLM keywords into a valid PubMed boolean,
    always including the gene symbol itself."""
    parts = re.split(r"[,\n]", raw_terms)
    return _or_join_terms([gene] + parts)


def get_uniprot_synonyms(accession: str) -> list[str]:
    """Fetch every protein/gene name synonym UniProt records for an accession.

    Collects recommended, alternative, and CD-antigen protein names (both full and
    short forms) plus the gene name and gene synonyms. Returns a de-duplicated list
    (order preserved). Returns [] on a missing accession or any lookup/parse failure,
    so callers can fall back gracefully to the bare gene symbol.

    Args:
        accession (str): UniProt accession (e.g. 'P16671' for CD36).

    Returns:
        list[str]: Synonyms, e.g. ['Platelet glycoprotein 4', 'Fatty acid translocase',
            'FAT', 'Platelet glycoprotein IV', 'GPIV', 'CD36', 'GP3B', ...].
    """
    if not accession:
        return []
    try:
        with urllib.request.urlopen(UNIPROT_REST_URL.format(accession)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError):
        return []

    names: list[str] = []

    def _add_name_block(block: dict) -> None:
        # a UniProt name block has one fullName and zero or more shortNames
        if not block:
            return
        full = block.get("fullName", {}).get("value")
        if full:
            names.append(full)
        for short in block.get("shortNames", []):
            if short.get("value"):
                names.append(short["value"])

    protein = data.get("proteinDescription", {})
    _add_name_block(protein.get("recommendedName"))
    for alt in protein.get("alternativeNames", []):
        _add_name_block(alt)
    for cd in protein.get("cdAntigenNames", []):
        if cd.get("value"):
            names.append(cd["value"])

    for gene_entry in data.get("genes", []):
        gene_name = gene_entry.get("geneName", {}).get("value")
        if gene_name:
            names.append(gene_name)
        for syn in gene_entry.get("synonyms", []):
            if syn.get("value"):
                names.append(syn["value"])

    # de-duplicate case-insensitively, preserving first-seen order
    seen = set()
    unique = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique.append(n)
    return unique


def build_synonym_search_query(gene: str) -> str:
    """Build a PubMed E-Search query OR-ing the gene symbol with all UniProt protein
    synonyms for that gene's canonical protein.

    Resolves the gene to its canonical UniProt accession via the Reactome graph, pulls
    the full synonym list from UniProt, and OR-joins them (multi-word phrases quoted).
    Falls back to the bare gene symbol when the accession or synonyms can't be resolved.

    Args:
        gene (str): Gene symbol (e.g. 'CD36').

    Returns:
        str: A PubMed boolean query, e.g. 'CD36 OR "Platelet glycoprotein 4" OR ...'.
    """
    accession = neo4jutils.query_accession_for_gene(gene)
    synonyms = get_uniprot_synonyms(accession)
    return _or_join_terms([gene] + synonyms)


def select_pathway_names(gene: str, drop_generic: bool = True, max_pathways: int = 15) -> list[str]:
    """Return the Reactome pathway names used to build a pathway query for a gene.

    Drops mega-generic top-level pathways (see GENERIC_PATHWAYS) and caps the count, so
    specific leaf pathways dominate rather than broad ones that swamp relevance ranking.
    """
    names = [p["pathway"] for p in neo4jutils.query_pathways_for_gene(gene)]
    if drop_generic:
        names = [n for n in names if n not in GENERIC_PATHWAYS]
    return names[:max_pathways]


def _pathway_fragment(names: list[str]) -> str:
    """OR-join pathway names as parenthesised (unquoted) groups.

    Pathway names are left unquoted so PubMed's automatic term mapping expands the words
    within each; quoting them as exact phrases roughly halves their hits (they rarely
    appear verbatim in an abstract). Parentheses group each pathway's terms.
    """
    return " OR ".join(f"({n})" for n in names)


def build_pathway_query(gene: str, drop_generic: bool = True, max_pathways: int = 15) -> str:
    """Build a PubMed query from a gene's (generic-filtered) Reactome pathway names.

    Falls back to the bare gene symbol when the gene has no usable pathways.
    """
    names = select_pathway_names(gene, drop_generic, max_pathways)
    return _pathway_fragment(names) if names else gene


def build_union_query(gene: str, drop_generic: bool = True, max_pathways: int = 15,
                      fi_cutoff: float = 0.8, partner_top_n: int = 10) -> str:
    """Combine all Stage-1 signals into one OR'd PubMed query.

    - gene symbol + UniProt protein synonyms -> _or_join_terms (short names, quoted/deduped)
    - top-N FI interaction-partner names      -> _or_join_terms (same quoting/dedup as names)
    - Reactome pathway names (direct lookup)  -> parenthesised unquoted groups (proven pathway
      construction)

    The fragments are OR'd together. Reuses the same generic-pathway filtering and cap as the
    standalone pathway-name strategy, so the pathway contribution is identical. Partner names
    come from select_top_partner_names (fetch_fis top-N by FI score); the partner-name source
    is what differs from build_partner_enrichment_query (direct pathways here vs. enriched there).

    Note: with a fixed fetch cap, a very broad union (many pathways + noisy short synonyms +
    partner names) can dilute Best Match relevance and is not guaranteed to beat the best single
    strategy per gene; lower max_pathways / partner_top_n if that happens rather than raising the
    fetch cap.
    """
    synonyms = get_uniprot_synonyms(neo4jutils.query_accession_for_gene(gene))
    name_fragment = _or_join_terms([gene] + synonyms)
    partner_fragment = _or_join_terms(select_top_partner_names(gene, fi_cutoff, partner_top_n))
    pathway_fragment = _pathway_fragment(select_pathway_names(gene, drop_generic, max_pathways))
    return " OR ".join(f for f in (name_fragment, partner_fragment, pathway_fragment) if f)


def select_top_partner_names(gene: str, fi_cutoff: float = 0.8, top_n: int = 10) -> list[str]:
    """Top-N functional-interaction partner gene names for a gene, ranked by FI score.

    Ranks the gene's partners by FI score (fetch_fis) and returns the top-N partner gene
    symbols. Shared by the query builders (partner names as a search fragment) and the
    re-rank entity-mention set (build_entity_mention_set), and reused inside
    select_partner_enriched_pathway_names so partner selection is defined once.

    Returns [] when the gene has no FI partners clearing fi_cutoff.
    """
    fi_df = _get_fi_loader().fetch_fis(gene, fi_cutoff=fi_cutoff)
    if fi_df is None or fi_df.empty:
        return []
    return list(fi_df.sort_values("score", ascending=False)["gene"].head(top_n))


def build_judge_context(gene: str, description: str | None = None,
                        max_pathways: int = 15, partner_top_n: int = 10) -> str:
    """Assemble a RICH biological context string for the curator-judge (CuratorRubric.judge_select)
    so it can correctly score gene-ABSENT-but-relevant papers -- ones whose mechanism sits in
    {gene}'s pathway/partner context without ever naming {gene} (the majority of real curator-cited
    evidence). Combines a prose role description with the gene's EXPLICIT Reactome pathway names and
    top functional-interaction partner symbols (Neo4j + FI lookups, no LLM), giving the judge the
    concrete pathway/partner vocabulary it needs to recognize those papers.

    `description`: prose lead (e.g. a precomputed gene background or the gene-specific pathway
    descriptions). If falsy, one is generated via build_query_and_search_terms. Pathway/partner lines
    are appended when available and silently skipped otherwise (e.g. cold-start genes with no
    released pathways)."""
    parts = []
    prose = description
    if not prose:
        try:
            _, prose = build_query_and_search_terms(gene)
        except Exception:
            prose = None
    if prose:
        parts.append(prose.strip())
    try:
        pathways = [p["pathway"]
                    for p in (neo4jutils.query_pathways_for_gene(gene) or [])][:max_pathways]
        if pathways:
            parts.append(f"{gene}'s Reactome pathways: " + "; ".join(pathways) + ".")
    except Exception:
        pass
    try:
        partners = select_top_partner_names(gene, top_n=partner_top_n)
        if partners:
            parts.append(
                f"{gene}'s top functional-interaction partners (a paper about the mechanism of these "
                f"partners within this pathway context is relevant even if {gene} is never named): "
                + ", ".join(partners) + ".")
    except Exception:
        pass
    return "\n\n".join(parts)


def select_partner_enriched_pathway_names(gene: str, fi_cutoff: float = 0.8,
                                          top_n: int = 10, fdr_cutoff: float = 0.05,
                                          drop_generic: bool = True,
                                          max_pathways: int = 15) -> list[str]:
    """Pathway names for a gene derived from FI-partner enrichment (not direct lookup).

    Ranks the gene's functional-interaction partners by FI score (fetch_fis), takes the
    top N, and runs Reactome pathway enrichment (binomial + BH-FDR) on them. Returns the
    enriched, FDR-sorted leaf pathway names, generic-filtered and capped exactly like
    select_pathway_names -- so this is a drop-in pathway-name source that also works for
    genes with no direct Reactome annotation.

    No PMIDs are involved: partners map into pathways as an unweighted gene set. Returns []
    when the gene has no FI partners, none map to a pathway, or nothing clears fdr_cutoff.
    """
    # Lazy import: ReactomeUtils pulls in scanpy/faiss at module load, so only the
    # partner-enrichment path (not every QueryBuilder import) pays that cost.
    import ReactomeUtils as utils

    top_partners = select_top_partner_names(gene, fi_cutoff=fi_cutoff, top_n=top_n)
    if not top_partners:
        return []

    interaction_dict = {partner: set() for partner in top_partners}  # no PMIDs needed
    map_df = utils.map_interactions_in_pathways(interaction_dict,
                                                pathway_file=_PATHWAY_GENE_FILE)
    if map_df is None or map_df.empty:
        return []
    enriched = utils.pathway_binomial_enrichment_df(map_df, top_partners,
                                                    pathway_file=_PATHWAY_GENE_FILE,
                                                    fdr_cutoff=fdr_cutoff)
    if enriched is None or enriched.empty:
        return []

    names = list(enriched["pathway_name"])  # already sorted ascending by FDR
    if drop_generic:
        names = [n for n in names if n not in GENERIC_PATHWAYS]
    return names[:max_pathways]


def build_partner_enrichment_query(gene: str, fi_cutoff: float = 0.8, top_n: int = 10,
                                   fdr_cutoff: float = 0.05, drop_generic: bool = True,
                                   max_pathways: int = 15) -> str:
    """Build a PubMed query for a cold-start gene from FI-partner-enrichment pathway names,
    OR'd with the gene name, UniProt synonyms, and top-N interaction-partner names.

    Sources pathway names from interaction-partner enrichment instead of the direct Reactome
    graph lookup, so it works for genes with no existing Reactome annotation. This is the only
    difference from build_union_query's pathway source; both branches OR in the same gene +
    synonyms + partner-name ingredients.

    The gene/synonyms/partners are OR'd in (NOT AND'd): an `[gene] AND (pathways)` form was
    validated and collapsed citation recall (36 -> 4 pool_hits over the 10 locked genes) because
    curator-cited papers often never name the gene (the section-3 recall ceiling). OR-ing only
    ADDS gene/partner-mentioning papers to the pool, so it does not reintroduce that failure.
    Falls back to the bare gene symbol when neither enriched pathways nor name fragments exist.
    """
    synonyms = get_uniprot_synonyms(neo4jutils.query_accession_for_gene(gene))
    partners = select_top_partner_names(gene, fi_cutoff, top_n)
    name_fragment = _or_join_terms([gene] + synonyms + partners)
    pathway_fragment = _pathway_fragment(
        select_partner_enriched_pathway_names(gene, fi_cutoff, top_n, fdr_cutoff,
                                              drop_generic, max_pathways))
    joined = " OR ".join(f for f in (name_fragment, pathway_fragment) if f)
    return joined if joined else gene


def build_retrieval_query(gene: str, drop_generic: bool = True, max_pathways: int = 15) -> str:
    """Choose the Stage-1 retrieval query based on the gene's Reactome annotation status.

    Has existing Reactome data  -> build_union_query (pathway names + UniProt synonyms), the
        best-performing strategy on the validation set (76 pool_hits).
    No data (cold-start gene)    -> build_partner_enrichment_query IF the partner-enrichment
        placement is confident; otherwise fall back to build_synonym_search_query. A weak
        placement (e.g. TANC1: FDR 9.8e-3, single partner) anchors retrieval on the wrong
        pathways and degrades the annotation, so we only trust it when it clears the gate.

    NOTE (RAB6C dilution): for a gate-pass gene whose enriched pathways are broad, well-studied
    fields (RAB6C's partners are 8 other RABs -> RAB/vesicle-trafficking pathways), this single
    combined query returns a large but gene-agnostic pool (0% mentioning RAB6C) because PubMed
    relevance ranks the gene papers below the fetch cap. This function still builds ONE combined
    query (used by the notebook dispatcher/harnesses); the production retriever instead calls
    build_retrieval_query_pair + LiteratureSearchTool._merge_search to run the name and pathway
    searches SEPARATELY and union the result sets, which is what actually fixes the dilution
    (RAB6C on-target 0 -> 22 in the pool). Keep this dispatch in sync with that of the pair.
    """
    if neo4jutils.query_pathways_for_gene(gene):
        return build_union_query(gene, drop_generic, max_pathways)

    # Cold-start: gate partner-enrichment on placement confidence.
    import ReactomeUtils as utils
    if utils.is_confident_placement(utils.suggest_pathway_placement(gene)):
        return build_partner_enrichment_query(gene)
    return build_synonym_search_query(gene)


def build_retrieval_query_pair(gene: str, drop_generic: bool = True, max_pathways: int = 15,
                               fi_cutoff: float = 0.8, partner_top_n: int = 10,
                               include_partners: bool = True) -> tuple[str, str]:
    """Two INDEPENDENT Stage-1 queries for the result-set merge, as (name_query, context_query).

    The retriever (LiteratureSearchTool) runs each as its own PubMed E-Search and UNIONs the PMID
    sets, rather than OR-ing everything into one query. This is the fix for the RAB6C dilution
    failure: with a single combined query, PubMed's relevance ranking lets the broad pathway/partner
    terms dominate and buries the gene-specific papers below the fetch cap (RAB6C's 25 gene papers
    ranked ~1348/1449). Two searches each get their OWN ranking, so the gene-specific set is
    guaranteed into the pool. Verified on RAB6C: 0/5 -> 3/5 on-target in the final Stage-2 output.

    - name_query    : gene symbol + UniProt synonyms (the specific signal).
    - context_query : the broad signal -- pathway names + top-N partner names. Pathway source
                      mirrors build_retrieval_query's dispatch EXACTLY (keep the two in sync):
                        has Reactome data      -> direct pathway names + partners
                        cold-start & gate-pass -> partner-enrichment pathway names + partners
                        cold-start & gate-fail -> "" (name search only == the synonym-only fallback)

    Partner names live in context_query, NOT name_query: co-locating them with the gene would
    re-create the same in-query ranking competition the merge exists to avoid. This is distinct
    from Change 1 (partner names OR'd into the single combined query, in build_union_query /
    build_partner_enrichment_query), which stays in place for the single-query dispatcher.
    """
    synonyms = get_uniprot_synonyms(neo4jutils.query_accession_for_gene(gene))
    name_query = _or_join_terms([gene] + synonyms)
    # include_partners=False drops top-N FI partner names from context_query entirely, so the broad
    # search is gene + direct pathways only (mentor-directed has-data simplification / A/B knob).
    partner_fragment = (_or_join_terms(select_top_partner_names(gene, fi_cutoff, partner_top_n))
                        if include_partners else "")

    def _context(pathway_fragment: str) -> str:
        return " OR ".join(f for f in (pathway_fragment, partner_fragment) if f)

    if neo4jutils.query_pathways_for_gene(gene):
        context_query = _context(_pathway_fragment(
            select_pathway_names(gene, drop_generic, max_pathways)))
    else:
        import ReactomeUtils as utils
        if utils.is_confident_placement(utils.suggest_pathway_placement(gene)):
            context_query = _context(_pathway_fragment(
                select_partner_enriched_pathway_names(gene, fi_cutoff=fi_cutoff,
                                                      top_n=partner_top_n,
                                                      drop_generic=drop_generic,
                                                      max_pathways=max_pathways)))
        else:
            context_query = ""  # gate-fail: name search only (== the synonym-only fallback)

    return name_query, context_query


def build_entity_mention_set(gene: str, fi_cutoff: float = 0.8,
                             partner_top_n: int = 10) -> list[str]:
    """Entities whose presence in an abstract earns the Stage-2 re-rank mention bonus.

    Gene symbol + UniProt protein synonyms + top-N FI interaction-partner names -- the same
    ingredients OR'd into the retrieval query, so retrieval and re-ranking agree on what counts
    as an on-topic entity. De-duplicated case-insensitively, order preserved.

    LLM-free (REST + Neo4j + Mongo only): callers use it inside re-ranking, which must not make
    nested LLM calls (see get_reranking_target's event-loop note). The mention bonus is additive,
    never a filter -- papers mentioning none of these still rank by cosine similarity.
    """
    synonyms = get_uniprot_synonyms(neo4jutils.query_accession_for_gene(gene))
    partners = select_top_partner_names(gene, fi_cutoff, partner_top_n)
    seen, entities = set(), []
    for e in [gene] + synonyms + partners:
        if e and e.lower() not in seen:
            seen.add(e.lower())
            entities.append(e)
    return entities


def build_gene_specific_pathway_descriptions(gene: str, drop_generic: bool = True,
                                             max_pathways: int = 15,
                                             summary_snippet: int = 400,
                                             reference_interactions: bool = True) -> dict:
    """Gene-SPECIFIC-within-pathway descriptions for the has-data Stage-2 re-rank target.

    For each of the gene's (generic-filtered, capped) released-human pathways, an LLM writes 2-3
    sentences on how THIS gene functions within THAT pathway -- its role, mechanism, key
    interactions -- grounded by the pathway name + summary. This replaces the raw pathway summary,
    which is generic and embeds close to broad review articles (validated A/B: SHANK3 curator
    usefulness 1.0 -> 7.2, annotatable 0/5 -> 4/5; BRCA1 2.0 -> 3.6).

    ONE batched call. max_tokens is raised and scaled with the pathway count so the JSON doesn't
    truncate (the default cap truncated BRCA1's 15 descriptions); the parse is truncation-tolerant
    -- it salvages each complete object individually, so an overflow just yields a SUBSET and the
    caller (get_reranking_target) falls back to the raw summary for any missing pathway.

    Returns {pathway_name: description} (possibly a subset). Returns {} -- with NO LLM call -- for
    cold-start genes (no released human pathways).

    Makes a BLOCKING LLM .invoke(): run it ONCE upfront in a worker thread (asyncio.to_thread),
    NEVER inside the async re-rank path -- same event-loop constraint as build_query_and_search_terms
    and get_reranking_target.
    """
    names = select_pathway_names(gene, drop_generic, max_pathways)
    pairs = [(n, neo4jutils.query_pathway_summary(n)) for n in names]
    pairs = [(n, s) for n, s in pairs if s]
    if not pairs:
        return {}

    listing = "\n".join(
        f"[{i}] {name}: {' '.join(summ.split())[:summary_snippet]}"
        for i, (name, summ) in enumerate(pairs, 1))
    # reference_interactions=False = mentor-directed simplification: describe the gene's role in the
    # pathway only, with no interaction/partner content (A/B knob vs. the shipped default).
    focus_clause = ("its role, mechanism, and key molecular interactions --"
                    if reference_interactions else
                    "its role and mechanism within the pathway --")
    no_partner_clause = ("" if reference_interactions
                         else " Do NOT reference specific interaction partners.")
    prompt = (
        f"For the human gene {gene}, below are Reactome pathways it participates in, each with the "
        f"pathway's summary. For EACH pathway, write a 2-3 sentence description of how {gene} "
        f"SPECIFICALLY functions within THAT pathway -- {focus_clause} grounded in the pathway "
        f"context but focused on {gene}, NOT a general description of the pathway "
        f"itself.{no_partner_clause}\n\n"
        f"Pathways:\n{listing}\n\n"
        f"Return ONLY a JSON array, one object per pathway:\n"
        f'[{{"index": <int matching [n]>, "description": "<2-3 sentences>"}}]')

    model = create_reactome_chat_model()
    # Scale the output cap with pathway count (>= the validated 4096 for 15), capped at 8192, so
    # even 30+-pathway genes don't truncate; the tolerant parse below is the final backstop.
    model.max_tokens = min(8192, max(4096, 256 * len(pairs)))
    content = model.invoke(prompt).content

    # Truncation-tolerant parse: pull each index+description pair individually (order-independent;
    # a cut-off trailing object simply doesn't match), rather than requiring a whole valid array.
    out = {}
    for mo in re.finditer(r'"index"\s*:\s*(\d+)\s*,\s*"description"\s*:\s*"((?:[^"\\]|\\.)*)"',
                          content):
        idx = int(mo.group(1))
        if 1 <= idx <= len(pairs):
            try:
                desc = json.loads(f'"{mo.group(2)}"').strip()
            except Exception:
                desc = mo.group(2).strip()
            if desc:
                out[pairs[idx - 1][0]] = desc
    return out


def build_gene_specific_enriched_pathway_description(gene: str,
                                                     summary_snippet: int = 400) -> dict:
    """Gene-SPECIFIC description of a COLD-START gate-pass gene's PREDICTED role in its primary
    partner-enriched pathway -- the cold-start analogue of build_gene_specific_pathway_descriptions
    for the Stage-2 re-rank target.

    build_gene_specific_pathway_descriptions serves has-data genes (directly-known pathways) and
    returns {} for cold-start genes. This one serves cold-start GATE-PASS genes: those with a
    CONFIDENT suggest_pathway_placement but no released human pathway of their own. It asks an LLM
    for 2-3 sentences on how {gene} is PREDICTED to function within its primary enriched pathway,
    grounded in the interaction partners that placed it there. The framing is partner-grounded /
    PREDICTED -- the gene does NOT directly participate, so the prompt does not assert an
    established role. This replaces the raw pathway summary (generic, pulls broad reviews); validated
    A/B on gate-pass genes: CTTNBP2 curator usefulness 4.8 -> 9.0 (annotatable 2/5 -> 5/5),
    RAB6C 5.8 -> 6.2, GDF6 unchanged (already at ceiling).

    Returns {primary_pathway_name: description}, keyed EXACTLY as get_reranking_target's cold-start
    branch names the pathway (placement["primary"]["pathway_name"]), so it is picked up there via the
    same prefer-description-else-raw-summary handling used for has-data pathways. Returns {} -- with
    NO LLM call -- for any gene that is NOT cold-start gate-pass (has direct pathways, or placement
    not confident), so has-data / gate-fail genes are unaffected.

    Kept PRIMARY-only for now, matching the validated A/B scope (secondary close-tie pathways from
    suggest_pathway_placement are not used).

    Makes a BLOCKING LLM .invoke(): run ONCE upfront in a worker thread (asyncio.to_thread), NEVER
    inside the async re-rank path -- same event-loop constraint as build_gene_specific_pathway_descriptions.
    """
    import ReactomeUtils as utils  # lazy: pulls in scanpy/faiss at module load
    # Cold-start only: use the SAME has-data test as get_reranking_target's branch (select_pathway_names
    # non-empty). Has-data genes re-rank against their DIRECT pathways, so an enriched-placement
    # description would never be read there.
    if select_pathway_names(gene):
        return {}
    placement = utils.suggest_pathway_placement(gene)
    if not utils.is_confident_placement(placement):
        return {}

    primary = placement["primary"]
    name = primary["pathway_name"]
    summary = neo4jutils.query_pathway_summary(name) or ""
    partners = primary.get("mapped_genes") or [p["gene"] for p in placement.get("partners_used", [])]
    partner_str = ", ".join(partners) if partners else "(none identified)"

    prompt = (
        f"The human gene {gene} has NO direct Reactome pathway annotation. Based on its "
        f"functional-interaction partners that participate in the Reactome pathway '{name}' "
        f"({partner_str}), {gene} is computationally PREDICTED to function in this pathway.\n\n"
        f"Pathway summary: {' '.join(summary.split())[:summary_snippet]}\n\n"
        f"Write a 2-3 sentence description of how {gene} is PREDICTED to function within THIS "
        f"pathway -- its likely role, mechanism, and key molecular interactions with the partners "
        f"above -- grounded in the pathway context but focused on {gene}, NOT a general description "
        f"of the pathway itself. Frame it as a prediction based on interaction partners; do NOT "
        f"assert established or documented roles for {gene}. Return ONLY the description text.")

    model = create_reactome_chat_model()
    model.max_tokens = 512
    desc = (model.invoke(prompt).content or "").strip()
    return {name: desc} if desc else {}


def get_reranking_target(gene: str, drop_generic: bool = True,
                         max_pathways: int = 15,
                         description_override: str | None = None,
                         pathway_descriptions: dict | None = None) -> list[str]:
    """Pathway-level text(s) to re-rank retrieved papers against, replacing the gene-specific
    description that mis-ranked pathway/mechanism-level ground-truth papers (final recall
    collapsed to ~1% vs 2.8% pool recall).

    Returns a LIST of texts (one per pathway); the re-ranker scores each paper by its MAX
    cosine similarity across them, so a paper relevant to ANY of the gene's pathways ranks
    high (no concatenation, no embedder truncation).

    Has data   -> per pathway (generic-filtered, capped): the precomputed gene-SPECIFIC pathway
                  description (pathway_descriptions[name]) when supplied, else the raw pathway
                  summary as a safety fallback. Gene-specific descriptions rerank specific,
                  annotatable primary papers above broad reviews (validated A/B).
    Cold-start -> the primary suggested pathway (suggest_pathway_placement): its precomputed
                  gene-SPECIFIC enriched-pathway description (pathway_descriptions[primary], from
                  build_gene_specific_enriched_pathway_description) when supplied for a gate-pass
                  gene, else the raw pathway summary as a safety fallback -- same
                  prefer-description-else-summary handling as the has-data branch.
    Gate-fail  -> [description_override]: a precomputed LLM biological description of the gene
                  (resolved_description), when supplied. Validated to rerank markedly better than
                  the bare identity string on cold-start genes.
    Fallback   -> [gene symbol + synonyms] identity string when no pathway/placement/summary and
                  no override is available, so re-ranking never breaks.

    pathway_descriptions is passed IN (built once upfront by CrewAILiteratureAnnotator via
    build_gene_specific_pathway_descriptions in a worker thread) so this function stays LLM-FREE
    inside the async re-rank path -- same constraint as description_override.
    """
    names = select_pathway_names(gene, drop_generic, max_pathways)
    if not names:
        # Cold-start: use the placement's primary pathway ONLY if the placement is confident;
        # a weak placement would anchor re-ranking on the wrong pathway. Otherwise leave names
        # empty so we fall through to the gene-description target below.
        import ReactomeUtils as utils  # lazy: pulls in scanpy/faiss at module load
        placement = utils.suggest_pathway_placement(gene)
        if utils.is_confident_placement(placement):
            names = [placement["primary"]["pathway_name"]]

    # Per pathway: prefer the precomputed gene-specific description; fall back to the raw summary
    # for any pathway missing from the cache (or when no cache was supplied at all).
    pathway_descriptions = pathway_descriptions or {}
    summaries = [t for t in
                 (pathway_descriptions.get(n) or neo4jutils.query_pathway_summary(n) for n in names)
                 if t]
    if summaries:
        return summaries

    # Cold-start gate-fail: prefer the precomputed LLM biological description when the caller
    # supplies it. It is passed IN (not generated here) because get_reranking_target must stay
    # LLM-FREE: it runs inside LiteratureSearchTool within CrewAI's async flow, and a nested LLM
    # call here (e.g. build_query_and_search_terms) collides with the event loop ("no running
    # event loop" / empty-response -> pipeline abort). CrewAILiteratureAnnotator generates it once
    # upfront in a worker thread and stashes it on the shared gene_annotator.
    if description_override:
        return [description_override]

    # Last resort (LLM-FREE): gene symbol + UniProt synonyms as one identity text. Used when no
    # override is available (e.g. the upfront generation failed and degraded to None). Synonyms
    # are REST + Neo4j, so this stays event-loop-safe.
    synonyms = get_uniprot_synonyms(neo4jutils.query_accession_for_gene(gene))
    return [" ".join([gene] + synonyms)] if synonyms else [gene]