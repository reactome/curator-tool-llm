import os
from pathlib import Path
from typing import Collection
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

import pandas as pd
import scanpy as sc
from scanpy import AnnData
import numpy as np
import plotly.express as px
import requests

from sympy import false

import ReactomeNeo4jUtils as neo4jutils
import statsmodels.stats.multitest as smm
import scipy.stats as stats

import logging as log

logger = log.getLogger(__name__)

# Used to query interacting pathways
REACTOME_IDG_FI_API_URL = 'https://idg.reactome.org/idgpairwise/relationships/combinedScoreGenesForTerm/'
# Resolve relative to this module (repo_root/resources/...) so callers work regardless of
# cwd (e.g. running from notebooks/). Points at the same file the cwd-relative path used to.
REACTOME_PATHWAY_GENE_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'resources',
                 'ReactomePathwayGenes_Ver_91.txt'))

RANDOM_STATE = 123456

class Pathway:
    def __init__(self, id, name, fdr, pVal, bottomLevel) -> None:
        self.id = id
        self.name = name
        self.fdr = fdr
        self.pVal = pVal
        self.bottomLevel = bottomLevel
        self.annotated = false

# The following several functions are used for data analysis. They will be refactored to other places later on.


def embed_event_summary(pathway_summary_pd: pd.DataFrame,
                        model_name: str = 'pritamdeka/S-PubMedBert-MS-MARCO',
                        db_path: str = '../data/faiss/reactome_pathway_index'):
    # Create a list of Document objects from the pathway dataframe
    # Iterate the pathway dataframe
    documents = []
    for _, row in pathway_summary_pd.iterrows():
        content = row['summary']
        if not content:
            log.info('no summary {}'.format(row['displayName']))
            content = row['displayName']
        metadata = {'dbId': row['dbId'],
                    'displayName': row['displayName']}
        document = Document(page_content=content, metadata=metadata)
        documents.append(document)
    log.info('Total documents: {}'.format(len(documents)))
    # Use the default settings in paperqa
    # This is something we'd like to use for embedding
    # The maximum of the default model we used, S-PubMedBert-MS_MARCH, is 350 tokens.
    # chunk_overlap is calcualted for tokens, not characters
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10,
                                                          model_name=model_name,
                                                          add_start_index=True)()
    # text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    log.info('After splitting: {}'.format(len(docs)))
    # for doc in docs:
    #     print(doc.metadata['dbId'], doc.metadata['displayName'])
    # Use the sentence transformers embedding, which was used in the IDG project
    # The embedding model is: https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO
    # Which is used for performance analysis in the Biomedical Knoweldge graph RAG paper (https://arxiv.org/abs/2311.17330)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # The whole embedding took about 15 minutes at the 14'' MacBook Pro (no gpu was used)
    db = FAISS.from_documents(docs, embeddings)
    log.info('Done embedding and saving into {}...'.format(db_path))
    db.save_local(db_path)
    log.info('Done saving.')


def load_event_summary_embed_db(db_path: str = '../data/faiss/reactome_pathway_index',
                                model_name: str = 'pritamdeka/S-PubMedBert-MS-MARCO') -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db: FAISS = FAISS.load_local(db_path, embeddings=embeddings)
    return db


def export_event_summary_embedding_to_anndata(db: FAISS) -> sc.AnnData:
    """Export the text embedding into an AnnData object so that we can use scanpy to do analysis and visualization.

    Returns:
        sc.AnnData: _description_
    """

    # Peek the first embedding to get the dimension of the embedding
    first_vector = db.index.reconstruct(0)
    id_map = db.index_to_docstore_id
    text_matrix = np.zeros((len(id_map), len(first_vector)))
    dbIds = []
    displayNames = []
    has_startIndex = False
    startIndices = []

    for index, doc_id in id_map.items():
        text_matrix[index] = db.index.reconstruct(index)
        doc = db.docstore.search(doc_id)
        dbIds.append(doc.metadata['dbId'])
        displayNames.append(doc.metadata['displayName'])
        if 'start_index' in doc.metadata.keys():
            startIndices.append(doc.metadata['start_index'])
            has_startIndex = True
        else:
            startIndices.append('-1')  # mark as unknown

    adata = AnnData(text_matrix)
    adata.obs['dbId'] = dbIds
    adata.obs['displayName'] = displayNames
    if has_startIndex:
        adata.obs['start_index'] = startIndices

    return adata


def plotly_umap(adata: sc.AnnData):
    umap_df = pd.DataFrame(
        columns=['UMAP1', 'UMAP2', 'leiden', 'dbId', 'displayName'])
    umap_df['UMAP1'] = adata.obsm['X_umap'][:, 0]
    umap_df['UMAP2'] = adata.obsm['X_umap'][:, 1]
    hover_cols = []
    umap_df['leiden'] = pd.Categorical(adata.obs['leiden'].to_list())
    hover_cols.append('leiden')
    umap_df['dbId'] = adata.obs['dbId'].to_list()
    hover_cols.append('dbId')
    umap_df['displayName'] = adata.obs['displayName'].to_list()
    if 'topic' in adata.obs.keys():
        umap_df['topic'] = adata.obs['topic'].to_list()
        hover_cols.append('topic')
    umap_df.sort_values(by='leiden', inplace=True)

    fig = px.scatter(umap_df,
                     x='UMAP1',
                     y='UMAP2',
                     color='leiden',
                     hover_data=umap_df[hover_cols],
                     width=850,
                     height=800)
    fig.show()


def add_topic_event(adata: sc.AnnData):
    event_topic_df = neo4jutils.load_event_to_topic_map()
    removed_topics = ['GOCAM test events', 'Cancer Hallmarks']

    def map_event_topic(dbId):
        topics = event_topic_df[event_topic_df['pathway_id']
                                == dbId]['topic_name'].to_list()
        topics = set(topics)
        for removed_topic in removed_topics:
            if removed_topic in topics:
                topics.remove(removed_topic)
        if len(topics) == 0:  # Assign unknown
            return 'unknown'
        topics = list(topics)
        topics.sort()
        # If there is more than one topic, pick the first one after soring.
        return topics[0]
        # return ','.join(topics)
    adata.obs['topic'] = adata.obs['dbId'].map(
        lambda dbId: map_event_topic(dbId))


def get_annotated_pathways(gene: str, pathway_file: str = REACTOME_PATHWAY_GENE_FILE) -> list[Pathway]:
    """Get the annotated pathways for a gene based on a pre-generated file.

    Args:
        gene (str): _description_
        fi_cutoff (float, optional): _description_. Defaults to 0.8.
    """
    pathway_gene_df = pd.read_csv(pathway_file, sep='\t')
    pathways = []
    for _, row in pathway_gene_df.iterrows():
        pathway_id = row['pathway_id']
        pathway_name = row['pathway_name']
        # Convert to set for faster intersection
        pathway_genes = set(row['genes'].split(','))
        if gene in pathway_genes:
            # Put the results in the Pathway data structure for other code
            # All pathways in this file are bottom level
            pathway = Pathway(pathway_id, pathway_name, None, None, True)
            pathway.annotated = True
            pathways.append(pathway)
    return pathways


def create_interacting_pathway_text(pathway: str,
                                    interacting_genes: list[str]):
    # Fetch the roles of interacting genes in pathways according to reactions
    reaction_roles_df = neo4jutils.query_reaction_roles_of_pathway(
        pathway, interacting_genes)
    # print(reaction_roles_df)
    reaction_gene_role_text = ''
    genes_in_pathway = []
    for _, row in reaction_roles_df.iterrows():
        reaction = row['reaction']
        gene = row['gene']
        genes_in_pathway.append(gene)
        role = row['role']
        if len(reaction_gene_role_text) > 0:
            reaction_gene_role_text = reaction_gene_role_text + "; "
        reaction_gene_role_text = '{}{} in "{}" as {}'.format(
            reaction_gene_role_text, gene, reaction, role)

    # Because of the version issue, some pathways may not have genes. Escape them
    if len(genes_in_pathway) == 0:
        return None, None, None

    summation = neo4jutils.query_pathway_summary(pathway)

#     pathway_text_template = """
# Pathway title: {}\n\n
# Pathway summary: {}\n\n
# Genes annotated in the pathway and interacting with the query gene: {}\n\n
# Roles of interacting genes in reactions annotated in the pathway: {}
#     """
    # As of March 19, 2025, split the genes and their roles so that they can be
    # highlighted in the LLM generated text
    pathway_text_template = """
Pathway title: {}\n\n
Pathway summary: {}
    """
    pathway_text = pathway_text_template.format(
        pathway,
        summation
    )
    log.debug('text for {}\n'.format(pathway, pathway_text))
    # Make sure genes_in_pathway is not repeated since a gene may be invovled in multiple reactions
    genes_in_pathway = list(set(genes_in_pathway))
    genes_in_pathway.sort()
    return pathway_text, ','.join(genes_in_pathway), reaction_gene_role_text


def create_annotated_pathway_text(pathway: str,
                                  annotated_gene: str):
    """
    This function is similar to create_interacting_pathway_text(), but for an annotated pathway only.
    """
    # Fetch the roles of interacting genes in pathways according to reactions
    reaction_roles_df = neo4jutils.query_reaction_roles_of_pathway(
        pathway, [annotated_gene])
    reaction_gene_role_text = ''
    genes_in_pathway = []
    for _, row in reaction_roles_df.iterrows():
        reaction = row['reaction']
        gene = row['gene']
        genes_in_pathway.append(gene)
        role = row['role']
        if len(reaction_gene_role_text) > 0:
            reaction_gene_role_text = reaction_gene_role_text + "; "
        reaction_gene_role_text = '{}{} in "{}" as {}'.format(
            reaction_gene_role_text, gene, reaction, role)

    # Because of the version issue, some pathways may not have genes. Escape them
    if len(genes_in_pathway) == 0:
        return None

    summation = neo4jutils.query_pathway_summary(pathway)

    pathway_text_template = """
Pathway title: {}\n\n
Pathway summary: {}\n\n
Roles of the query gene in reactions annotated in the pathway: {}
    """
    pathway_text = pathway_text_template.format(
        pathway,
        summation,
        reaction_gene_role_text
    )
    log.debug('text for {}\n'.format(pathway, pathway_text))
    return pathway_text


def download_pdf_paper(url: str,
                       pmid: str | int,
                       paper_dir: str):
    """Download a PDF file directly from the provided URL.
    Note: The downloading of using a script from the PMC web site is blocked!

    Args:
        url (str): _description_
        pmid (str | int): _description_
        paper_dir (str): _description_
    """
    response = requests.get(url)
    file_name = Path(paper_dir, '{}.pdf'.format(pmid))
    with open(file_name, 'wb') as file:
        file.write(response.content)


def map_interactions_in_pathways(interaction_dict, pathway_file: str = REACTOME_PATHWAY_GENE_FILE) -> pd.DataFrame:
    """Map the interactions to pathways.

    Args:
        interaction_dict (_type_): interaction partners vs pubmed_ids
        pathway_file (str, optional): _description_. Defaults to REACTOME_PATHWAY_GENE_FILE.
    return: pd.DataFrame
    """
    pathway_data = pd.read_csv(pathway_file, sep='\t')
    map_df = pd.DataFrame(columns=[
        "pathway_id",
        "pathway_name",
        "mapped_genes",
        "pmids"
    ])
    row_index = 0
    for _, row in pathway_data.iterrows():
        pathway_id = row['pathway_id']
        pathway_name = row['pathway_name']
        # Convert to set for faster intersection
        pathway_genes = set(row['genes'].split(','))
        hits = []
        pmids = []
        for partner, pmids_1 in interaction_dict.items():
            if partner in pathway_genes:
                hits.append(partner)
                pmids.append('|'.join(pmids_1))
        if len(hits) > 0:
            map_df.loc[row_index] = [pathway_id, pathway_name, hits, pmids]
            row_index += 1
    return map_df


def pathway_binomial_enrichment_df(map_df,
                                   total_interacting_genes: Collection[str],
                                   pathway_file: str = REACTOME_PATHWAY_GENE_FILE,
                                   fdr_cutoff: float = 0.05) -> pd.DataFrame:
    """Perform pathway enrichment analysis for the mapped dataframe object. The DF object
    should be generated by function map_interactions_in_pathways().

    Args:
        map_df (_type_): _description_
        pathway_file (str, optional): _description_. Defaults to REACTOME_PATHWAY_GENE_FILE.
    """
    pathway_data = pd.read_csv(pathway_file, sep='\t')
    pathway_2_size = {
        row['pathway_id']: len(row['genes'].split(','))
        for _, row in pd.read_csv(pathway_file, sep='\t').iterrows()
    }

    # Compute background size as the total number of unique genes across all pathways
    all_genes_in_pathways = set(
        gene for genes in pathway_data['genes'] for gene in genes.split(','))
    background_size = len(all_genes_in_pathways)
    # Need to get the genes that can be mapped to pathways
    total_interacting_genes = [gene for gene in total_interacting_genes if gene in all_genes_in_pathways]
    test_gene_count = len(total_interacting_genes)  # Total genes in the test set
    
    # Store pathway p-values
    p_values = []
    pathway_ids = []
    pathway_names = []
    overlap_counts = []
    mapped_genes_all = []
    pmids_all = []

    for _, row in map_df.iterrows():
        pathway_id = row['pathway_id']
        pathway_name = row['pathway_name']
        mapped_genes = row['mapped_genes']
        # Find the overlap between the pathway genes and the gene set
        overlap_count = len(mapped_genes)

        if overlap_count > 0:
            # Probability of a gene being in this pathway by random chance
            pathway_size = pathway_2_size[pathway_id]
            expected_prob = pathway_size / background_size

            # Perform binomial test
            p_value = stats.binomtest(
                overlap_count, test_gene_count, expected_prob, alternative='greater').pvalue

            # Store results for FDR correction
            pathway_ids.append(pathway_id)
            pathway_names.append(pathway_name)
            overlap_counts.append(overlap_count)
            p_values.append(p_value)
            mapped_genes_all.append(mapped_genes)
            pmids_all.append(row['pmids'])

    # Apply FDR correction using Benjamini-Hochberg method
    q_values = smm.multipletests(p_values, method='fdr_bh')[1]

    # Create a DataFrame
    df = pd.DataFrame({
        "pathway_id": pathway_ids,
        "pathway_name": pathway_names,
        "overlap_count": overlap_counts,
        'mapped_genes': mapped_genes_all,
        'pmids_all': pmids_all,
        "pVal": p_values,
        "FDR": q_values
    })

    # Sort by FDR-adjusted q-values (ascending order)
    df = df.sort_values(by="FDR").reset_index(drop=True)
    df = df[df['FDR'] < fdr_cutoff]
    return df


def pathway_binomial_enrichment(gene_list, pathway_file: str = REACTOME_PATHWAY_GENE_FILE):
    """
    Perform enrichment analysis using the binomial test with FDR correction.

    :param pathway_data: DataFrame with columns ['pathway_id', 'pathway_name', 'genes']
    :param gene_list: List of genes to test for enrichment.
    :return: Pandas DataFrame.
    """
    pathway_data = pd.read_csv(pathway_file, sep='\t')

    # Convert gene_list to set for faster lookup
    gene_set = set(gene_list)
    test_gene_count = len(gene_set)  # Total genes in the test set

    # Compute background size as the total number of unique genes across all pathways
    all_genes_in_pathways = set(
        gene for genes in pathway_data['genes'] for gene in genes.split(','))
    background_size = len(all_genes_in_pathways)

    # Store pathway p-values
    p_values = []
    pathway_ids = []
    pathway_names = []
    overlap_counts = []

    for _, row in pathway_data.iterrows():
        pathway_id = row['pathway_id']
        pathway_name = row['pathway_name']
        # Convert to set for faster intersection
        pathway_genes = set(row['genes'].split(','))

        # Find the overlap between the pathway genes and the gene set
        overlap_count = len(gene_set & pathway_genes)

        if overlap_count > 0:
            # Probability of a gene being in this pathway by random chance
            pathway_size = len(pathway_genes)
            expected_prob = pathway_size / background_size

            # Perform binomial test
            p_value = stats.binomtest(
                overlap_count, test_gene_count, expected_prob, alternative='greater').pvalue

            # Store results for FDR correction
            pathway_ids.append(pathway_id)
            pathway_names.append(pathway_name)
            overlap_counts.append(overlap_count)
            p_values.append(p_value)

    # Apply FDR correction using Benjamini-Hochberg method
    q_values = smm.multipletests(p_values, method='fdr_bh')[1]

    # Create a DataFrame
    df = pd.DataFrame({
        "pathway_id": pathway_ids,
        "pathway_name": pathway_names,
        "overlap_count": overlap_counts,
        "pVal": p_values,
        "FDR": q_values
    })

    # Sort by FDR-adjusted q-values (ascending order)
    df = df.sort_values(by="FDR").reset_index(drop=True)

    return df


# --- Deterministic pathway placement (Part 3) --------------------------------------------
# Lazily-constructed MongoFILoader singleton; its __init__ hits Mongo to load the gene
# index, so build it once and reuse across genes.
_fi_loader = None


def _get_fi_loader():
    global _fi_loader
    if _fi_loader is None:
        from ProteinProteinInteractionsLoader import MongoFILoader
        _fi_loader = MongoFILoader()
    return _fi_loader


def _placement_candidate(row) -> dict:
    """One enriched-pathway row -> a JSON-serializable candidate dict."""
    return {
        "pathway_id": int(row["pathway_id"]),
        "pathway_name": row["pathway_name"],
        "fdr": float(row["FDR"]),
        "overlap_count": int(row["overlap_count"]),
        "mapped_genes": list(row["mapped_genes"]),  # top-N partners hitting this pathway
    }


def suggest_pathway_placement(gene: str, fi_cutoff: float = 0.8, top_n: int = 10,
                              fdr_cutoff: float = 0.05,
                              tie_margin_orders: float = 1.0) -> dict:
    """Suggest Reactome pathway placement for a gene from its top interaction partners.

    Deterministic and LLM-free: ranks the gene's functional-interaction partners by FI score
    (top ``top_n``), maps them into Reactome pathways, and runs binomial + BH-FDR enrichment.
    The #1 FDR-ranked pathway is the primary suggestion; any pathway whose FDR is within
    ``tie_margin_orders`` decades of the primary's FDR is surfaced as a close secondary
    candidate (a genuine statistical tie, not a re-decision). This produces a ranked,
    statistically-grounded suggestion for Phase 2's reactome_curator agent to *verify*; it
    does not itself decide the answer beyond what the FDR ranking determines.

    Returns a JSON-serializable dict:
        gene, status, primary, secondary, partners_used, n_significant_total, params
    status is one of:
        "ok"                     -> primary populated
        "no_partners"            -> gene has no FI partners above fi_cutoff
        "no_significant_pathway" -> partners found but none clear fdr_cutoff
    """
    result = {
        "gene": gene,
        "status": "ok",
        "primary": None,
        "secondary": [],
        "partners_used": [],
        "n_significant_total": 0,
        "params": {"fi_cutoff": fi_cutoff, "top_n": top_n,
                   "fdr_cutoff": fdr_cutoff, "tie_margin_orders": tie_margin_orders},
    }

    fi_df = _get_fi_loader().fetch_fis(gene, fi_cutoff=fi_cutoff)
    if fi_df is None or fi_df.empty:
        result["status"] = "no_partners"
        return result

    top = fi_df.sort_values("score", ascending=False).head(top_n)
    partners = list(top["gene"])
    result["partners_used"] = [{"gene": g, "score": float(s)}
                               for g, s in zip(top["gene"], top["score"])]

    map_df = map_interactions_in_pathways({p: set() for p in partners},
                                          pathway_file=REACTOME_PATHWAY_GENE_FILE)
    if map_df is None or map_df.empty:
        result["status"] = "no_significant_pathway"
        return result

    enriched = pathway_binomial_enrichment_df(map_df, partners,
                                              pathway_file=REACTOME_PATHWAY_GENE_FILE,
                                              fdr_cutoff=fdr_cutoff)
    if enriched is None or enriched.empty:
        result["status"] = "no_significant_pathway"
        return result

    result["n_significant_total"] = len(enriched)
    primary_fdr = float(enriched.iloc[0]["FDR"])
    result["primary"] = _placement_candidate(enriched.iloc[0])

    # Close ties: within tie_margin_orders decades of the primary FDR. enriched is sorted by
    # FDR ascending, so once a row exceeds the threshold every later row does too -> break.
    # Floor the primary FDR to avoid a zero threshold when it underflows to 0.0.
    threshold = max(primary_fdr, 1e-300) * (10 ** tie_margin_orders)
    for _, row in enriched.iloc[1:].iterrows():
        if float(row["FDR"]) <= threshold:
            result["secondary"].append(_placement_candidate(row))
        else:
            break
    return result


def is_confident_placement(placement: dict, max_fdr: float = 1e-3,
                           min_partners: int = 2) -> bool:
    """Whether a suggest_pathway_placement result is confident enough to anchor retrieval /
    the curation directive on (vs falling back to gene-name/synonym retrieval).

    Gate tuned on a cold-start vs have-data sweep (2026-07-08): every have-data reference
    placement clears FDR<=1.9e-5 with >=3 partners, while the TANC1-class failures (which
    tanked the end-to-end annotation: 0.88 -> 0.48, approved -> rejected) had FDR>1e-3 and a
    single supporting partner. Cost is asymmetric — a wrong anchor propagates through Stage-1
    retrieval + rerank + the curator prompt, whereas a false negative just reverts to the safe
    gene-name baseline — so the gate errs toward gating out.
    """
    if not placement or placement.get("status") != "ok" or not placement.get("primary"):
        return False
    primary = placement["primary"]
    return (primary["fdr"] < max_fdr
            and len(primary.get("mapped_genes", [])) >= min_partners)
