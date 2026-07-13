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


def build_union_query(gene: str, drop_generic: bool = True, max_pathways: int = 15) -> str:
    """Combine all three Stage-1 signals into one OR'd PubMed query.

    - gene symbol + UniProt protein synonyms -> _or_join_terms (short names, quoted/deduped)
    - Reactome pathway names -> parenthesised unquoted groups (proven pathway construction)

    The two fragments are OR'd together. Reuses the same generic-pathway filtering and cap
    as the standalone pathway-name strategy, so the pathway contribution is identical.

    Note: with a fixed fetch cap, a very broad union (many pathways + noisy short synonyms)
    can dilute Best Match relevance and is not guaranteed to beat the best single strategy
    per gene; lower max_pathways if that happens rather than raising the fetch cap.
    """
    synonyms = get_uniprot_synonyms(neo4jutils.query_accession_for_gene(gene))
    name_fragment = _or_join_terms([gene] + synonyms)
    pathway_fragment = _pathway_fragment(select_pathway_names(gene, drop_generic, max_pathways))
    return " OR ".join(frag for frag in (name_fragment, pathway_fragment) if frag)


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

    fi_df = _get_fi_loader().fetch_fis(gene, fi_cutoff=fi_cutoff)
    if fi_df is None or fi_df.empty:
        return []
    top_partners = list(fi_df.sort_values("score", ascending=False)["gene"].head(top_n))

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
    """Build a PubMed query from FI-partner-enrichment pathway names.

    Mirrors build_pathway_query but sources pathway names from interaction-partner
    enrichment instead of the direct Reactome graph lookup, so it works for genes with no
    existing Reactome annotation. The gene is deliberately NOT AND'd in: an `[gene] AND
    (pathways)` form was validated and collapsed citation recall (36 -> 4 pool_hits over the
    10 locked genes) because curator-cited papers often never name the gene (the section-3
    recall ceiling). Falls back to the bare gene symbol when no enriched pathways are found.
    """
    names = select_partner_enriched_pathway_names(gene, fi_cutoff, top_n, fdr_cutoff,
                                                   drop_generic, max_pathways)
    return _pathway_fragment(names) if names else gene


def build_retrieval_query(gene: str, drop_generic: bool = True, max_pathways: int = 15) -> str:
    """Choose the Stage-1 retrieval query based on the gene's Reactome annotation status.

    Has existing Reactome data  -> build_union_query (pathway names + UniProt synonyms), the
        best-performing strategy on the validation set (76 pool_hits).
    No data (cold-start gene)    -> build_partner_enrichment_query IF the partner-enrichment
        placement is confident; otherwise fall back to build_synonym_search_query. A weak
        placement (e.g. TANC1: FDR 9.8e-3, single partner) anchors retrieval on the wrong
        pathways and degrades the annotation, so we only trust it when it clears the gate.
    """
    if neo4jutils.query_pathways_for_gene(gene):
        return build_union_query(gene, drop_generic, max_pathways)

    # Cold-start: gate partner-enrichment on placement confidence.
    import ReactomeUtils as utils
    if utils.is_confident_placement(utils.suggest_pathway_placement(gene)):
        return build_partner_enrichment_query(gene)
    return build_synonym_search_query(gene)


def get_reranking_target(gene: str, drop_generic: bool = True,
                         max_pathways: int = 15) -> list[str]:
    """Pathway-level text(s) to re-rank retrieved papers against, replacing the gene-specific
    description that mis-ranked pathway/mechanism-level ground-truth papers (final recall
    collapsed to ~1% vs 2.8% pool recall).

    Returns a LIST of texts (one per pathway); the re-ranker scores each paper by its MAX
    cosine similarity across them, so a paper relevant to ANY of the gene's pathways ranks
    high (no concatenation, no embedder truncation).

    Has data   -> summation text of each of the gene's (generic-filtered, capped) pathways.
    Cold-start -> summation text of the primary suggested pathway (suggest_pathway_placement).
    Fallback   -> [gene-specific description] when no pathway/placement/summary is available,
                  so re-ranking never breaks.
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

    summaries = [t for t in (neo4jutils.query_pathway_summary(n) for n in names) if t]
    if summaries:
        return summaries

    # Last resort: gene symbol + UniProt synonyms as one identity text. Must be LLM-FREE:
    # get_reranking_target runs inside LiteratureSearchTool within CrewAI's async flow, and a
    # nested LLM call here (e.g. build_query_and_search_terms) collides with the event loop
    # ("no running event loop" / empty-response -> pipeline abort). Synonyms are REST + Neo4j.
    synonyms = get_uniprot_synonyms(neo4jutils.query_accession_for_gene(gene))
    return [" ".join([gene] + synonyms)] if synonyms else [gene]