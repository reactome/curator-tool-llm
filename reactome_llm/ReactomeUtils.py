from operator import itemgetter
from pathlib import Path

from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

import pandas as pd
import scanpy as sc
from scanpy import AnnData
import numpy as np
import plotly.express as px
import requests

from paperqa.types import Text
from paperqa import EmbeddingModel
from paperqa.llms import NumpyVectorStore
from paperqa.docs import Docs, Doc
from sympy import false
from ReactomeLLMErrors import NoAbstractFoundError, NoAbstractSupportingInteractingPathwayError, NoInteractingPathwayFoundError

import ReactomePrompts as prompts
import ReactomeNeo4jUtils as neo4jutils
import statsmodels.stats.multitest as smm
import scipy.stats as stats

import logging as log

from ReactomePubMed import ReactomePubMedRetriever
import ProteinProteinInteractionsLoader as ppi_loader

logger = log.getLogger()
logger.setLevel(log.INFO)
# logger.setLevel(log.DEBUG)
log.basicConfig(
    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s', filename=None)

# Used to embedding
EMBEDDING_MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'

# Used to query interacting pathways
REACTOME_IDG_INTERACTING_PATHWAY_API_URL = 'https://idg.reactome.org/idgpairwise/relationships/enrichedSecondaryPathwaysForTerm1'
REACTOME_IDG_FI_API_URL = 'https://idg.reactome.org/idgpairwise/relationships/combinedScoreGenesForTerm/'
REACTOME_IDG_PATHWAY_API_URL = 'https://idg.reactome.org/idgpairwise/relationships/hierarchyForTerm/'
REACTOME_PATHWAY_GENE_FILE = 'resources/ReactomePathwayGenes_Ver_91.txt'

RANDOM_STATE = 123456

class Pathway:
    def __init__(self, id, name, fdr, pVal, bottomLevel) -> None:
        self.id = id
        self.name = name
        self.fdr = fdr
        self.pVal = pVal
        self.bottomLevel = bottomLevel
        self.annotated = false

# Probably use a class in the future
ppi_loader = ppi_loader.PPILoader()

def _get_text_splitter() -> TextSplitter:
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10,
                                                          model_name=EMBEDDING_MODEL_NAME,
                                                          add_start_index=True)
    return text_splitter


def _get_embedding(model_name: str = EMBEDDING_MODEL_NAME) -> EmbeddingModel:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def get_default_llm():
    model = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
    return model


def embed_event_summary(pathway_summary_pd: pd.DataFrame,
                        model_name: str = EMBEDDING_MODEL_NAME,
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
    text_splitter = _get_text_splitter()
    # text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    log.info('After splitting: {}'.format(len(docs)))
    # for doc in docs:
    #     print(doc.metadata['dbId'], doc.metadata['displayName'])
    # Use the sentence transformers embedding, which was used in the IDG project
    # The embedding model is: https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO
    # Which is used for performance analysis in the Biomedical Knoweldge graph RAG paper (https://arxiv.org/abs/2311.17330)
    embeddings = _get_embedding()
    # The whole embedding took about 15 minutes at the 14'' MacBook Pro (no gpu was used)
    db = FAISS.from_documents(docs, embeddings)
    log.info('Done embedding and saving into {}...'.format(db_path))
    db.save_local(db_path)
    log.info('Done saving.')


def load_event_summary_embed_db(db_path: str = '../data/faiss/reactome_pathway_index',
                                model_name: str = 'pritamdeka/S-PubMedBert-MS-MARCO') -> FAISS:
    embeddings = _get_embedding()
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


def query_fis(gene: str,
              fi_cutoff: float = 0.8) -> pd.DataFrame:
    """Query the reactome's idg RESTful API to get the list of functional interactions.

    Args:
        gene (str): _description_
        fi_cutoff (float, optional): _description_. Defaults to 0.8.
    """
    return ppi_loader.query_fis(gene, fi_cutoff)


def query_reactome_pathways_in_hierarchy(gene: str) -> any:
    """Query the hierarchy of the Reactome pathways for the passed gene.

    Args:
        gene (str): _description_

    Returns:
        any: May return None if the gene has not been annotated yet.
    """
    url = '{}{}'.format(REACTOME_IDG_PATHWAY_API_URL, gene)
    result = requests.get(url)
    json_objs = result.json()
    if len(json_objs['stIds']) == 0:
        return None
    # TODO: To build the pathway hierarchy later on.
    return json_objs['stIds']


def query_reactome_interacting_pathways(gene: str,
                                        fi_cutoff: float = 0.8,
                                        fdr_cutoff: float = 0.01,
                                        bottomLevel_only: bool = True,
                                        pathway_count: int = 10,
                                        exclude_annotated: bool = True) -> list[Pathway]:
    """Query the reactome's idg RESTful API to get the list of interacting pathways.

    Args:
        gene (str): _description_
        fi_cutoff (float, optional): _description_. Defaults to 0.8.
    """
    query = {
        'term': gene,
        'dataDescKeys': [0],
        'prd': fi_cutoff  # The FI prediction score cutoff
    }
    result = requests.post(REACTOME_IDG_INTERACTING_PATHWAY_API_URL, json=query)
    annotated_stIds = query_reactome_pathways_in_hierarchy(gene)
    json_objs = result.json()
    pathways = []
    for obj_data in json_objs:
        if pathway_count and len(pathways) >= pathway_count:
            continue
        if (obj_data['fdr'] > fdr_cutoff):
            continue
        if (bottomLevel_only and not obj_data['bottomLevel']):
            continue
        pathway = Pathway(obj_data['stId'],
                          obj_data['name'],
                          obj_data['fdr'],
                          obj_data['pVal'],
                          obj_data['bottomLevel'])
        if annotated_stIds is not None and pathway.id in annotated_stIds:
            pathway.annotated = True
        if exclude_annotated and pathway.annotated:
            continue # Escape it
        pathways.append(pathway)
    return pathways


def get_annotated_pathways(gene: str, pathway_file: str=REACTOME_PATHWAY_GENE_FILE) -> list[Pathway]:
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
        pathway_genes = set(row['genes'].split(','))  # Convert to set for faster intersection
        if gene in pathway_genes:
            # Put the results in the Pathway data structure for other code
            # All pathways in this file are bottom level
            pathway = Pathway(pathway_id, pathway_name, None, None, True)
            pathway.annotated = True    
            pathways.append(pathway)
    return pathways


def collect_summaries_for_interacting_pathways(gene: str,
                                               pathway_summary_pd: pd.DataFrame,
                                               fi_cutoff: float = 0.8,
                                               fdr_cutoff: float = 0.01,
                                               bottomLevel_only: bool = True) -> list[str]:
    pathways = query_reactome_interacting_pathways(gene,
                                                   fi_cutoff=fi_cutoff,
                                                   fdr_cutoff=fdr_cutoff,
                                                   bottomLevel_only=bottomLevel_only,
                                                   pathway_count=None)
    collected_reactome_summary = []

    for pathway in pathways:
        # print('{}: {}'.format(pathway.name, pathway.fdr))
        dbId = int(pathway.id.split('-')[2])
        if dbId in pathway_summary_pd.index:
            text = pathway_summary_pd.loc[int(dbId)]['summary']
            # Don't attach id here for the time being.
            collected_reactome_summary.append(
                '{}: {}'.format(pathway.name, text))
            # print('{}:\n{}\n\n'.format(pathway.name, text))
    return collected_reactome_summary


def write_summary_of_interacting_pathways_for_annotated_gene(gene: str,
                                                             collected_reactome_summary: list[str],
                                                             model,
                                                             total_words: int = 300) -> any:
    prompt = prompts.summary_prompt
    parameters = {'gene': gene,
                  'total_words': total_words,
                  'text_for_important_reactome_pathways': '\n\n'.join(collected_reactome_summary),
                  'docs': collected_reactome_summary}
    return invoke_llm(parameters, prompt, model)


async def write_summary_of_interacting_pathway_for_unannotated_gene(query_gene: str,
                                                                    interacting_genes: list[str],
                                                                    pathway: str,
                                                                    model: any,
                                                                    total_words: int = 150) -> any:
    """Write a summary for one single interacting pathway for an unannotated gene.

    Args:
        gene (str): _description_
        pathway_text (str): _description_
        total_words (int, optional): _description_. Defaults to 150.

    Returns:
        any: _description_
    """
    pathway_text = create_interacting_pathway_text(pathway,
                                                   interacting_genes)
    if pathway_text is None:
        return None

    prompt = prompts.interacting_pathway_summary_prompt
    parameters = {'gene': query_gene,
                  'total_words': total_words,
                  'interacting_pathway_text': pathway_text,
                  'docs': pathway_text}
    return invoke_llm(parameters, prompt, model)


async def write_summary_of_annotated_pathway(query_gene: str,
                                             pathway: str,
                                             model: any,
                                             total_words: int = 150) -> any:
    """Write a summary for one single interacting pathway for a gene annotated in Reactome.

    Args:
        gene (str): _description_
        pathway_text (str): _description_
        total_words (int, optional): _description_. Defaults to 150.

    Returns:
        any: _description_
    """
    pathway_text = create_annotated_pathway_text(pathway, query_gene)
    if pathway_text is None:
        return None

    prompt = prompts.annotated_pathway_summary_prompt
    parameters = {'gene': query_gene,
                  'total_words': total_words,
                  'annotated_pathway_text': pathway_text,
                  'docs': pathway_text}
    return invoke_llm(parameters, prompt, model)


async def write_summary_of_annotated_pathways(query_gene: str,
                                              model: any,
                                              total_words: int = 300) -> any:
    """Write a summary for multiple pathways for a gene annotated in Reactome

    Args:
        query_gene (str): _description_
        model (any): _description_
        total_words (int, optional): _description_. Defaults to 300.

    Returns:
        any: _description_
    """
    # Try to get all annotated pathways
    # TODO: To be optimized using Neo4j directly.
    # pathways = query_reactome_interacting_pathways(query_gene,
    #                                                fdr_cutoff=1.0, # No cutoff to get all pathways
    #                                                bottomLevel_only=True,
    #                                                pathway_count=None,
    #                                                exclude_annotated=False) # Should be all
    # Change to use a local pre-processed file as of March 12, 2025
    pathways = get_annotated_pathways(query_gene)
    annotated_pathways = [pathway for pathway in pathways if pathway.annotated]
    if len(annotated_pathways) == 0:
        return None
    if len(annotated_pathways) == 1: # No need to write any summary
        return await write_summary_of_annotated_pathway(query_gene, 
                                                        annotated_pathways[0].name,
                                                        model,
                                                        total_words=300)
    pathway_text_list = []
    for pathway in annotated_pathways:
        pathway_result = await write_summary_of_annotated_pathway(query_gene, 
                                                                  pathway.name, 
                                                                  model)
        if pathway_result is None:
            continue
        pathway_text = pathway_result['answer'].content
        pathway_text_list.append('{}: {}'.format(pathway.name, pathway_text))
    pathway_text_all = '\n\n'.join(pathway_text_list)

    prompt = prompts.annotated_pathways_summary_prompt
    parameters = {'gene': query_gene,
                  'total_words': total_words,
                  'annotated_pathways_text': pathway_text_all,
                  'docs': pathway_text_all}
    result = invoke_llm(parameters, prompt, model)

    return result


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
        return None

    summation = neo4jutils.query_pathway_summary(pathway)

    pathway_text_template = """
Pathway title: {}\n\n
Pathway summary: {}\n\n
Genes annotated in the pathway and interacting with the query gene: {}\n\n
Roles of interacting genes in reactions annotated in the pathway: {}
    """
    pathway_text = pathway_text_template.format(
        pathway,
        summation,
        ', '.join(genes_in_pathway),
        reaction_gene_role_text
    )
    log.debug('text for {}\n'.format(pathway, pathway_text))
    return pathway_text


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


async def write_summary_of_interacting_pathways_for_unannotated_gene(gene: str,
                                                                     model,
                                                                     fi_cutoff: float = 0.8,
                                                                     fdr_cutoff: float = 1.0E-2,
                                                                     pathway_count: int = 8,
                                                                     total_words: int = 300) -> any:
    """Write a summary for a set of interacting pathways for a gene that has not been annotated in Reactome.

    Args:
        gene (str): _description_
        model (_type_): _description_
        fi_cutoff (float, optional): _description_. Defaults to 0.8.
        fdr_cutoff (float, optional): _description_. Defaults to 1.0E-2.
        pathway_count (int, optional): _description_. Defaults to 8.
        total_words (int, optional): _description_. Defaults to 300.

    Returns:
        any: _description_
    """
    # TODO: Make sure we don't pick up too many genes and pathways
    pathways = query_reactome_interacting_pathways(gene,
                                                   fi_cutoff=fi_cutoff,
                                                   fdr_cutoff=fdr_cutoff,
                                                   bottomLevel_only=True,
                                                   pathway_count=pathway_count)
    pathways_with_fdr = ''
    selected_pathways = []
    for pathway in pathways:
        pathways_with_fdr = '{}{}:{}\n'.format(pathways_with_fdr,
                                               pathway.name,
                                               pathway.fdr)
        selected_pathways.append(pathway.name)
    # print(pathways_with_fdr)

    fi_df = query_fis(gene=gene, fi_cutoff=fi_cutoff)
    interacting_genes = fi_df['gene'].to_list()

    pathway_text_list = []
    for pathway in selected_pathways:
        pathway_result = await write_summary_of_interacting_pathway_for_unannotated_gene(query_gene=gene,
                                                                                         interacting_genes=interacting_genes,
                                                                                         pathway=pathway,
                                                                                         model=model)
        if pathway_result is None:
            continue
        pathway_text = pathway_result['answer'].content
        pathway_text_list.append('{}: {}'.format(pathway, pathway_text))
    pathway_text_all = '\n\n'.join(pathway_text_list)

    prompt = prompts.interacting_pathways_summary_prompt
    parameters = {'gene': gene,
                  'total_words': total_words,
                  'pathways_with_fdr': pathways_with_fdr,
                  'interacting_partners': ','.join(interacting_genes),
                  'text_for_interacting_pathways': pathway_text_all,
                  'docs': pathway_text_all}
    result = invoke_llm(parameters, prompt, model)

    return result


async def write_summary_of_interacting_pathways_for_unannotated_gene_via_ppis(gene: str,
                                                                     model,
                                                                     fi_cutoff: float=0.8,
                                                                     fdr_cutoff: float = 0.05,
                                                                     pathway_count: int = 8,
                                                                     total_words: int = 300) -> any:
    """Write a summary for a set of interacting pathways for a gene that has not been annotated in Reactome.
    The interacting pathways are fetched based on PPIs collected from IntAct and BioGrid.

    Args:
        gene (str): _description_
        model (_type_): _description_
        fi_cutoff (float, optional): Default=0.8. Used to filter the pulled PPIs.
        fdr_cutoff (float, optional): _description_. Defaults to 1.0E-2.
        pathway_count (int, optional): _description_. Defaults to 8.
        total_words (int, optional): _description_. Defaults to 300.

    Returns:
        any: _description_
    """
    ppi = ppi_loader.get_interactions(gene, filter_ppis_with_fi=True, fi_cutoff=fi_cutoff)
    if ppi is None:
        raise NoInteractingPathwayFoundError('No interacting pathway found for {}'.format(gene))
    
    # Make sure this is a list.
    interacting_genes = list(ppi.keys())

    ppis_in_pathways_df = map_interactions_in_pathways(ppi)
    # Here we'd like to have all pathways that are mapped. Therefore, we use fdr cutoff = 1.0
    pathway_enrichment_results = pathway_binomial_enrichment_df(ppis_in_pathways_df, 
                                                                len(interacting_genes),
                                                                fdr_cutoff=fdr_cutoff)
    if pathway_enrichment_results is None or pathway_enrichment_results.empty:
        raise NoInteractingPathwayFoundError('No interacting pathway found for {}'.format(gene))
    
    pathways_with_fdr = ''
    selected_pathways = []
    for _, row in pathway_enrichment_results.iterrows():
        if (len(selected_pathways)) >= pathway_count:
            break
        pathways_with_fdr = '{}{}:{}\n'.format(pathways_with_fdr,
                                               row['pathway_name'],
                                               row['FDR'])
        selected_pathways.append(row['pathway_name'])
    
    if len(selected_pathways) == 0:
        raise NoInteractingPathwayFoundError('No interacting pathway found for {}'.format(gene))
    
    pathway_text_list = []
    used_pathways = []
    for pathway in selected_pathways:
        pathway_result = await write_summary_of_interacting_pathway_for_unannotated_gene(query_gene=gene,
                                                                                         interacting_genes=interacting_genes,
                                                                                         pathway=pathway,
                                                                                         model=model)
        if pathway_result is None:
            continue
        pathway_text = pathway_result['answer'].content
        pathway_text_list.append('{}: {}'.format(pathway, pathway_text))
        used_pathways.append(pathway)
    pathway_text_all = '\n\n'.join(pathway_text_list)

    prompt = prompts.interacting_pathways_summary_prompt
    parameters = {'gene': gene,
                  'total_words': total_words,
                  'pathways_with_fdr': pathways_with_fdr,
                  'interacting_partners': ','.join(interacting_genes),
                  'text_for_interacting_pathways': pathway_text_all,
                  'docs': pathway_text_all}
    result = invoke_llm(parameters, prompt, model)

    # Returns only whatever is used
    enrichment_df = pathway_enrichment_results[pathway_enrichment_results['pathway_name'].isin(used_pathways)]

    return result, enrichment_df


async def write_summary_for_known_gene_via_paperqa(gene: str,
                                                   pathway_summary_df: pd.DataFrame,
                                                   fi_cutoff: float = 0.8,
                                                   fdr_cutoff: float = 0.01,
                                                   total_words: int = 300,
                                                   max_sources: int = 10) -> any:
    # Use sentence transformer as we did before
    # The code here is based on test_sentence_transformer_embedding in test_paperqa.py in the paper-qa GitHub repo
    # TODO: See how to use mps under mac
    # embedding_model = SentenceTransformerEmbeddingModel(name=EMBEDDING_MODEL_NAME)
    embedding_model = _get_embedding()

    text_splitter = _get_text_splitter()

    docs = Docs(
        texts_index=NumpyVectorStore(embedding_model=embedding_model),
        doc_index=NumpyVectorStore(embedding_model=embedding_model),
        embedding_client=None,
        llm='gpt-3.5-turbo',  # Maybe fine tuned (how?)
        # llm='gpt-3.5-turbo-0125', # Optimized for dialog. Support 16K! cheaper! But it is more dialog style. Better to use gpt-3.5-turbo.
        index_path='../data/paperqa'
    )

    interacting_pathway_summary = collect_summaries_for_interacting_pathways(gene,
                                                                             pathway_summary_df,
                                                                             fi_cutoff=fi_cutoff,
                                                                             fdr_cutoff=fdr_cutoff)
    for pathway_summary in interacting_pathway_summary:
        tokens = pathway_summary.split(':')
        pathway_name = tokens[0].strip()
        pathway_text = tokens[1].strip()
        # Create a Doc based on the above information
        doc = Doc(citation=pathway_name,
                  docname=pathway_name,
                  dockey=pathway_text)
        # Used for splitted
        text_doc = Document(page_content=pathway_text)
        text_docs = text_splitter.split_documents([text_doc])
        # Convert to Text in paperqa
        texts = [Text(text=text_doc_tmp.page_content, name=doc.docname, doc=doc)
                 for text_doc_tmp in text_docs]
        await docs.aadd_texts(texts, doc)

    result = await docs.aquery('Can you describe gene {} according to the provided context?'.format(gene),
                               max_sources=max_sources,
                               length_prompt='about {} words'.format(total_words))
    return result


async def build_abstract_vector_db_for_gene(query_gene: str,
                                            top_k_results: int = 8,
                                            max_query_length: int = 1000) -> VectorStore:
    """Query pubmed about interactions, reactions, and pathways for a gene and return
    a vector store for collected abstracts from PubMed.

    Args:
        query_gene (str): _description_
        top_k_results (int, optional): _description_. Defaults to 8.
        max_query_length (int, optional): _description_. Defaults to 1000.

    Returns:
        any: _description_
    """
    pubmed_retriever = ReactomePubMedRetriever()
    # Make sure it doesn't exceed the quote: 3 per second. use 0.5.
    pubmed_retriever.sleep_time = 0.5
    pubmed_retriever.top_k_results = top_k_results
    pubmed_retriever.MAX_QUERY_LENGTH = max_query_length
    pubmed_retriever.doc_content_chars_max = max_query_length * top_k_results

    pubmed_query = '{} interactions or {} reactions or {} pathways'.format(
        query_gene, query_gene, query_gene)
    log.debug('pubmed_query: {}'.format(pubmed_query))
    pubmed_result = pubmed_retriever.get_relevant_documents(pubmed_query)
    log.debug('pubmed_result: {}'.format(pubmed_result))
    # In case nothing is returned
    if len(pubmed_result) == 0:
        raise NoAbstractFoundError(query_gene)

    text_splitter = _get_text_splitter()
    docs = text_splitter.split_documents(pubmed_result)
    embeddings = _get_embedding()
    pubmed_db = FAISS.from_documents(docs, embeddings)

    return pubmed_db


async def write_summary_of_abstract_pathway_text(query_gene: str,
                                                 interacting_genes: list[str],
                                                 pathway: str,
                                                 pathway_text: str,
                                                 abstract_text: str,
                                                 model: any,
                                                 total_words: int = 150) -> any:
    """Create a summary for collected abstracts that are related to pathways.

    Args:
        query_gene (str): _description_
        interacting_genes (list[str]): _description_
        pathway (str): _description_
        pathway_text (str): _description_
        abstract_text (str): _description_
        model (any): _description_
        total_words (int, optional): _description_. Defaults to 150.

    Returns:
        any: _description_
    """
    prompt = prompts.abstract_summary_prompt
    parameters = {'query_gene': query_gene,
                  'interacting_genes': ','.join(interacting_genes),
                  'total_words': total_words,
                  'pathway_text': pathway_text,
                  'abstract_text': abstract_text,
                  'pathway': pathway,
                  'docs': '{}\n\n{}'.format(pathway_text, abstract_text)}
    result = invoke_llm(parameters, prompt, model)
    return result


def invoke_llm(parameters: dict,
               prompt: ChatPromptTemplate,
               model: any) -> any:
    # Don't include docs
    final_input = {key: itemgetter(key)
                   for key in parameters.keys() if key != 'docs'}
    answer = {
        'answer': final_input | prompt | model,
    }
    if 'docs' in parameters.keys():
        answer['docs'] = itemgetter('docs')

    # Pass a dummy runnable passthrough to make the chain work.
    dummy = RunnablePassthrough()
    llm_chain = dummy | answer
    result = llm_chain.invoke(parameters)
    return result


async def summarize_abstract_results_for_multiple_pathways(query_gene: str,
                                                           interacting_genes: list[str],
                                                           pathway_abstract_summary_df: pd.DataFrame,
                                                           model: any,
                                                           total_words: int = 300):
    """Summarize multiple abstracts for multiple pathways for a gene.

    Args:
        query_gene (str): _description_
        interacting_genes (list[str]): _description_
        pathway_abstract_summary_df (pd.DataFrame): _description_
        model (any): _description_
        total_words (int, optional): _description_. Defaults to 300.

    Returns:
        _type_: _description_
    """
    context = ''
    for _, row in pathway_abstract_summary_df.iterrows():
        pathway = row['pathway']
        pmid = row['pmid']
        abstract_summary = row['summary']
        if len(context) > 0:
            context = '{}\n\n'.format(context)
        context = '{}PMID:{};PATHWAY_NAME:"{}": {}'.format(
            context, pmid, pathway, abstract_summary)

    parameters = {'query_gene': query_gene,
                  'interacting_genes': ','.join(interacting_genes),
                  'total_words': total_words,
                  'context': context,
                  'docs': context}
    result = invoke_llm(parameters,
                        prompts.multiple_abstracts_summary_prompt,
                        model)

    return result


async def write_summary_of_abstracts_for_gene(query_gene: str,
                                              pubmed_db: VectorStore,
                                              model: any):
    """Write a summary for a query gene by collecting Reactome pathway related abstracts from PubMed.
    This function basically is a wrap of multiple calls of LLMs, as well as PubMed retrieval.

    Args:
        query_gene (str): _description_
        pubmed_db (VectorStore): _description_
        model (any): _description_

    Returns:
        _type_: _description_
    """
    # Get pathways and genes
    # TODO: need to check how many pathways should be used.
    fi_cutoff = 0.6
    pathways = query_reactome_interacting_pathways(query_gene, pathway_count=8, fi_cutoff=fi_cutoff)
    log.debug('Total pathways for {}: {}'.format(query_gene, len(pathways)))
    if len(pathways) == 0:
        raise NoInteractingPathwayFoundError(query_gene)
    fi_df = query_fis(gene=query_gene, fi_cutoff=fi_cutoff)
    interacting_genes = fi_df['gene'].to_list()
    log.debug('Total interacting genes: {}'.format(len(interacting_genes)))

    pathway_abstract_pd = await build_pathway_abstract_df(query_gene,
                                                          interacting_genes,
                                                          pathways,
                                                          pubmed_db,
                                                          model)
    if pathway_abstract_pd.empty:
        raise NoAbstractSupportingInteractingPathwayError(query_gene)

    abstract_result_for_multiple_pathways = await summarize_abstract_results_for_multiple_pathways(query_gene,
                                                                                                   interacting_genes,
                                                                                                   pathway_abstract_pd,
                                                                                                   model)
    return abstract_result_for_multiple_pathways

async def build_pathway_abstract_df(query_gene: str,
                                    interacting_genes: list[str],
                                    pathways: list[Pathway],
                                    pubmed_db: VectorStore,
                                    model: BaseChatModel) -> pd.DataFrame:
    pathway_abstract_pd = pd.DataFrame(
        columns=['pathway', 'pmid', 'title', 'abstract', 'score', 'summary'])
    row = 0
    text_splitter = _get_text_splitter()
    for pathway in pathways:
        pathway_text = create_interacting_pathway_text(
            pathway.name, interacting_genes)
        if pathway_text is None:
            continue
        splitted_texts = text_splitter.split_text(pathway_text)
        best_matched_abstract = None
        for splitted_text in splitted_texts:
            # Just need the top scored text
            # This is a two element tuple: document and score
            matched_abstract_score = pubmed_db.similarity_search_with_score(splitted_text)[
                0]
            # print('{}:\n{}'.format(splitted_text, matched_abstract_score))
            if not best_matched_abstract:
                best_matched_abstract = matched_abstract_score
            else:
                if matched_abstract_score[1] < best_matched_abstract[1]:
                    best_matched_abstract = matched_abstract_score
        log.debug('\n\nBest matched abstract: {}'.format(
            best_matched_abstract))
        abstract_result = await write_summary_of_abstract_pathway_text(query_gene,
                                                                       interacting_genes,
                                                                       pathway.name,
                                                                       splitted_text,
                                                                       best_matched_abstract[0].page_content,
                                                                       model)
        # In case there is no title
        title = '' if best_matched_abstract[0].metadata['Title'] is None else best_matched_abstract[0].metadata['Title']
        pathway_abstract_pd.loc[row] = [pathway.name,
                                        best_matched_abstract[0].metadata['uid'],
                                        title,
                                        best_matched_abstract[0].page_content,
                                        best_matched_abstract[1],
                                        abstract_result['answer'].content]
        row += 1
    log.debug('pathway_abstract_pd:\n{}'.format(pathway_abstract_pd.head()))
    return pathway_abstract_pd


def output_llm_result(result):
    formatted_result = '\nResult: {}\n\nDoc: {}'.format(result['answer'].content, result['docs'])
    return formatted_result


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


def analyze_full_paper(paper_file_name: str,
                       query_gene: str,
                       model: any,
                       top_pages: int = 12,
                       max_score: float = 50.0) -> list:
    """Analyze a full text pdf paper provided by paper_file_name.

    Args:
        paper_file_name (str): the file location of the full text pdf
        query_gene (str): the query gene the analysis should focus on
        model (any): an LLM model
        top_pages (int): only select these many top matched text chunks
        max_score (float): the max score should be used to filter out text chunks
    """
    # Load the paper first
    loader = PyPDFLoader(paper_file_name)
    token_splitter = _get_text_splitter()
    pages = loader.load_and_split(token_splitter)

    # Embedding the paper
    embeddings = _get_embedding()
    paper_db = FAISS.from_documents(pages, embeddings)

    # Fetch the best matched text
    query = f'({query_gene} interactions) or ({query_gene} reactions) or ({query_gene} pathways)'
    matched_pages = paper_db.similarity_search_with_score(query, k=top_pages)

    # Prepare to call llm
    parameters = {
        'query_gene': query_gene
    }
    prompt = prompts.relationship_extraction_prompt
    results = []
    for doc, score in matched_pages:
        if score > max_score:
            break # The returned results are sorted. If we see this, we can break the loop.
        parameters['docs'] = doc.page_content
        parameters['document'] = doc.page_content
        result = invoke_llm(model=model, parameters=parameters, prompt=prompt)
        results.append(result)
    return results


def map_interactions_in_pathways(interaction_dict, pathway_file: str=REACTOME_PATHWAY_GENE_FILE) -> pd.DataFrame:
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
        pathway_genes = set(row['genes'].split(','))  # Convert to set for faster intersection
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
                                   test_gene_count, 
                                   pathway_file: str=REACTOME_PATHWAY_GENE_FILE,
                                   fdr_cutoff: float=0.05) -> pd.DataFrame:
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
    all_genes_in_pathways = set(gene for genes in pathway_data['genes'] for gene in genes.split(','))
    background_size = len(all_genes_in_pathways)

    # Store pathway p-values
    p_values = []
    pathway_ids = []
    pathway_names = []
    overlap_counts = []
    mapped_genes_all = []
    pubmids_all = []

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
            p_value = stats.binomtest(overlap_count, test_gene_count, expected_prob, alternative='greater').pvalue

            # Store results for FDR correction
            pathway_ids.append(pathway_id)
            pathway_names.append(pathway_name)
            overlap_counts.append(overlap_count)
            p_values.append(p_value)
            mapped_genes_all.append(mapped_genes)
            pubmids_all.append(row['pmids'])

    # Apply FDR correction using Benjamini-Hochberg method
    q_values = smm.multipletests(p_values, method='fdr_bh')[1]

    # Create a DataFrame
    df = pd.DataFrame({
        "pathway_id": pathway_ids,
        "pathway_name": pathway_names,
        "overlap_count": overlap_counts,
        'mapped_genes': mapped_genes_all,
        'pubmids': pubmids_all,
        "pVal": p_values,
        "FDR": q_values
    })

    # Sort by FDR-adjusted q-values (ascending order)
    df = df.sort_values(by="FDR").reset_index(drop=True)
    df = df[df['FDR'] < fdr_cutoff]
    return df


def pathway_binomial_enrichment(gene_list, pathway_file: str=REACTOME_PATHWAY_GENE_FILE):
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
    all_genes_in_pathways = set(gene for genes in pathway_data['genes'] for gene in genes.split(','))
    background_size = len(all_genes_in_pathways)

    # Store pathway p-values
    p_values = []
    pathway_ids = []
    pathway_names = []
    overlap_counts = []

    for _, row in pathway_data.iterrows():
        pathway_id = row['pathway_id']
        pathway_name = row['pathway_name']
        pathway_genes = set(row['genes'].split(','))  # Convert to set for faster intersection

        # Find the overlap between the pathway genes and the gene set
        overlap_count = len(gene_set & pathway_genes)

        if overlap_count > 0:
            # Probability of a gene being in this pathway by random chance
            pathway_size = len(pathway_genes)
            expected_prob = pathway_size / background_size

            # Perform binomial test
            p_value = stats.binomtest(overlap_count, test_gene_count, expected_prob, alternative='greater').pvalue

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


# # Simple text in script for fast performance at VSCode
# async def test_api():
#     gene = 'DUX4L2'
#     pubmed_db = await build_abstract_vector_db_for_gene(gene, top_k_results=10)
#     model = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
#     query_result = await write_summary_of_abstracts_for_gene(gene,
#                                                              pubmed_db,
#                                                              model)
#     print(query_result)


# if __name__ == '__main__':
#     print('Running test_api...')
#     asyncio.run(test_api())


