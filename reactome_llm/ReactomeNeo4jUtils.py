import os
import dotenv
from neo4j import GraphDatabase
import neo4j

import pandas as pd

# Handle configuration in .env
dotenv.load_dotenv()
URI = os.getenv('REACTOME_NEO4J_URI')
AUTH = (os.getenv('REACTOME_NEO4J_USER'), os.getenv('REACTOME_NEO4J_PWD'))
DB = os.getenv('REACTOME_NEO4J_DATABASE')


def load_event_to_topic_map() -> pd.DataFrame:
    """Load the map from the pathways to their TopLevelPathways.
    Note: Some of pathways may not have any top level pathways.
    Returns:
        pd.DataFrame: _description_
    """
    query = '''
        MATCH (p:Event) <- [:hasEvent*] - (t:TopLevelPathway) 
        RETURN p.dbId as pathway_id, p.displayName as pathway_name,
               t.dbId as topic_id, t.displayName as topic_name
    '''
    top_pathway_df = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        top_pathway_df = driver.execute_query(query,
                                              db=DB,
                                              result_transformer_=neo4j.Result.to_df)
    return top_pathway_df


def load_pathway_dbIds() -> list:
    """Load the list of dbIds for pathways. This may be used to filter events to pathways only.

    Returns:
        list: _description_
    """
    query = 'MATCH (p:Pathway) RETURN p.dbId as dbId'
    pathway_ids = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        pathway_ids = driver.execute_query(query,
                                           db=DB,
                                           result_transformer_=neo4j.Result.to_df)['dbId'].to_list()
    return pathway_ids


def load_event_summary(limit: int = None) -> pd.DataFrame:
    """Load text summary for all pathways and reactions.

    Returns:
        pd.DataFrame: the dataframe contains columns, dbId, displayName, text summary.
    """
    event_summary_pd = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=DB) as session:
            event_summary_pd = session.execute_read(
                _run_event_summary_query, limit)
    # Set dbId as the index
    event_summary_pd.set_index('dbId', drop=False, inplace=True)
    return event_summary_pd


def _run_event_summary_query(tx, limit: int = None) -> pd.DataFrame:
    """Run the actual cyphery query.

    Args:
        tx (_type_): _description_

    Returns:
        pd.DataFrame: the query results in a DataFrame object.
    """
    query = """
        Match (e:Event) 
        OPTIONAL Match (e)-[:summation]->(s:Summation)
        return e.dbId as dbId, e.displayName as displayName, s.text as summary 
    """
    if limit:
        query = (query + ' limit {}').format(limit)
    result = tx.run(query)
    return result.to_df()


def query_pathways_for_gene(gene: str) -> list[dict]:
    """Get all Reactome pathways that the given gene participates in.

    Args:
        gene (str): Gene symbol (e.g. 'NTN1')

    Returns:
        list[dict]: List of dicts with keys 'pathway' and 'pathway_id'
    """
    # Restrict to RELEASED HUMAN pathways: without this, the match returns (a) unreleased
    # draft/proposal curations -- e.g. "(NEW)InfectiousDiseaseProposal", "(draft)Viral Infection
    # Pathways", "...Proposal" (doRelease = false/null) -- and (b) non-human orthologs, since the
    # geneName match isn't species-scoped (BRCA1 pulled in Gallus gallus R-GGA- pathways). Both
    # polluted the Stage-1 context query and the Stage-2 re-rank target. doRelease is Reactome's
    # canonical "included in the release" flag (releaseStatus is null even for released pathways,
    # so it can't be used).
    query = """
        MATCH (ewas:EntityWithAccessionedSequence)-[:referenceEntity]->(g:ReferenceSequence)
        WHERE g.geneName[0] = $gene_name
        MATCH (p:Pathway)-[:hasEvent*]->(r:ReactionLikeEvent)
              -[:input|catalystActivity|regulatedBy|physicalEntity|hasComponent|hasMember|hasCandidate*]->(ewas)
        WHERE p.speciesName = 'Homo sapiens' AND p.doRelease = true
        RETURN DISTINCT p.displayName AS pathway, p.dbId AS pathway_id
        ORDER BY p.displayName
    """
    result_df = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        result_df = driver.execute_query(query,
                                         db=DB,
                                         gene_name=gene,
                                         result_transformer_=neo4j.Result.to_df)
    if result_df is None or result_df.empty:
        return []
    return result_df.to_dict(orient='records')


def query_accession_for_gene(gene: str) -> str | None:
    """Look up the canonical UniProt accession for a gene from the Reactome graph.

    Resolves a gene symbol to the reviewed UniProt accession Reactome already stores,
    preferring the entry where the symbol is the primary gene name and the canonical
    (non-isoform) identifier. Returns None if the gene is not present in the graph, in
    which case callers should fall back to an external lookup (e.g. the UniProt REST API).

    Args:
        gene (str): Gene symbol (e.g. 'TANC1').

    Returns:
        str | None: The canonical UniProt accession (e.g. 'Q9C0D5'), or None if not found.
    """
    query = """
        MATCH (rgp:ReferenceGeneProduct)
        WHERE $gene IN rgp.geneName AND rgp.databaseName = 'UniProt'
        RETURN rgp.identifier AS accession
        ORDER BY (rgp.geneName[0] = $gene) DESC, (rgp.variantIdentifier IS NULL) DESC
        LIMIT 1
    """
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records = driver.execute_query(query, db=DB, gene=gene).records
    if not records:
        return None
    return records[0]["accession"]


def query_reaction_roles_of_pathway(pathway: str,
                                    genes: list[str]) -> pd.DataFrame:
    """Get the reactions and roles for a list of genes in a specific pathway.

    Args:
        pathway (str): _description_
        genes (list[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    query = """
        MATCH (p:Pathway {displayName: $pathway_name})
        MATCH (p) - [:hasEvent*] -> (r:ReactionLikeEvent)
        MATCH (r) - [r_role:input|catalystActivity|regulatedBy|physicalEntity|hasComponent|hasMember|hasCandidate*] -> (ewas:EntityWithAccessionedSequence)
        MATCH (ewas) - [:referenceEntity] -> (g:ReferenceSequence) WHERE g.geneName[0] in $gene_names
        RETURN DISTINCT p.displayName AS pathway, r.displayName AS reaction, type(r_role[0]) AS role, g.geneName[0] AS gene
    """
    result_df = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        result_df = driver.execute_query(query,
                                         db=DB,
                                         pathway_name=pathway,
                                         gene_names=genes,
                                         result_transformer_=neo4j.Result.to_df)
    # Somehow "input" is not understood by llm. Change it to reactant as more popular term.
    def map_fun(role):
        if role == 'input': 
            return 'reactant'
        elif role == 'catalystActivity':
            return 'catalyst'
        elif role == 'regulatedBy':
            return 'regulator'
        return role
    result_df['role'] = result_df['role'].map(map_fun)
    return result_df


def query_pathway_summary(pathway: str) -> str:
    """Query the pathway summary from the database.

    Args:
        pathway (_type_, optional): _description_.

    Returns:
        str: _description_
    """
    query = """
        MATCH (p:Pathway {displayName: $pathway})
        OPTIONAL MATCH (p)-[:summation]->(summation:Summation)
        RETURN summation.text AS text
    """
    text = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=DB) as session:
            result = session.run(query, pathway=pathway).single()
            if result is not None:
                text = result['text']
    return text


def map_pathway_name_to_dbId(pathway_names: list[str]) -> dict[int, str]:
    """Map display names to dbIds into a dict.

    Args:
        pathway_names (list[str]): _description_

    Returns:
        dict[int, str]: _description_
    """
    query = """
        MATCH (m:Pathway) WHERE m.displayName IN $pathway_names
        RETURN m.displayName as displayName, m.dbId as dbId
    """
    result_df = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        result_df = driver.execute_query(query,
                                         db=DB,
                                         pathway_names=pathway_names,
                                         result_transformer_=neo4j.Result.to_df)
    if result_df is not None:
        name2id = {}
        for _, row in result_df.iterrows():
            name2id[row['displayName']] = row['dbId']
        return name2id
    return None

def query_literature_references_for_gene(gene: str) -> list[int]:
    """Get all PMIDs cited (at pathway or reaction level) for events the gene participates in.

    Args:
        gene (str): Gene symbol (e.g. 'SHANK3')

    Returns:
        list[int]: List of PMIDs
    """
    query = """
        MATCH (ewas:EntityWithAccessionedSequence)-[:referenceEntity]->(g:ReferenceSequence)
        WHERE g.geneName[0] = $gene_name AND ewas.speciesName = "Homo sapiens"
        MATCH (p:Pathway)-[:hasEvent*]->(r:ReactionLikeEvent)
              -[:input|catalystActivity|regulatedBy|physicalEntity|hasComponent|hasMember|hasCandidate*]->(ewas)
        MATCH (p)-[:literatureReference]->(plit:LiteratureReference)
        RETURN DISTINCT plit.pubMedIdentifier AS pmid

        UNION

        MATCH (ewas:EntityWithAccessionedSequence)-[:referenceEntity]->(g:ReferenceSequence)
        WHERE g.geneName[0] = $gene_name
        MATCH (p:Pathway)-[:hasEvent*]->(r:ReactionLikeEvent)
              -[:input|catalystActivity|regulatedBy|physicalEntity|hasComponent|hasMember|hasCandidate*]->(ewas)
        MATCH (r)-[:literatureReference]->(rlit:LiteratureReference)
        RETURN DISTINCT rlit.pubMedIdentifier AS pmid
    """
    result_df = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        result_df = driver.execute_query(query,
                                         db=DB,
                                         gene_name=gene,
                                         result_transformer_=neo4j.Result.to_df)
    if result_df is None or result_df.empty:
        return []
    return result_df['pmid'].to_list()



def get_random_annotated_genes(n: int = 10, min_pathways: int = 3) -> list[str]:
    """Get a random sample of gene names with at least min_pathways pathway annotations.

    Args:
        n (int): Number of genes to sample.
        min_pathways (int): Minimum number of pathways a gene must appear in to qualify.

    Returns:
        list[str]: List of gene names.
    """
    query = """
        MATCH (ewas:EntityWithAccessionedSequence)-[:referenceEntity]->(g:ReferenceSequence)
        WHERE g.geneName IS NOT NULL AND ewas.speciesName = "Homo sapiens"
        MATCH (p:Pathway)-[:hasEvent*]->(r:ReactionLikeEvent)
            -[:input|catalystActivity|regulatedBy|physicalEntity|hasComponent|hasMember|hasCandidate*]->(ewas)
        WITH g.geneName[0] AS gene, count(DISTINCT p) AS pathway_count
        WHERE pathway_count >= $min_pathways
        RETURN gene
        ORDER BY rand()
        LIMIT $n
    """
    result_df = None
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        result_df = driver.execute_query(query,
                                        db=DB,
                                        min_pathways=min_pathways,
                                        n=n,
                                        result_transformer_=neo4j.Result.to_df)
    if result_df is None or result_df.empty:
        return []
    return result_df['gene'].to_list()
