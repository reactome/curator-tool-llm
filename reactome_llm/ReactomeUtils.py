import os
import dotenv
from operator import itemgetter
from neo4j import GraphDatabase
import neo4j


from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

import pandas as pd
import scanpy as sc
from scanpy import AnnData
import numpy as np
import plotly.express as px
import requests

from paperqa.types import Text
from paperqa import SentenceTransformerEmbeddingModel
from paperqa.llms import NumpyVectorStore
from paperqa.docs import Docs, Doc

import ReactomePrompts as prompts
import ReactomeNeo4jUtils as neo4jutils

import logging as log
logger = log.getLogger()
logger.setLevel(log.INFO)
log.basicConfig(
    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s', filename=None)

# Used to embedding
EMBEDDING_MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'

# Used to query interacting pathways
REACTOME_IDG_INTERACTING_PATHWAY_API_URL = 'https://idg.reactome.org/idgpairwise/relationships/enrichedSecondaryPathwaysForTerm1'
REACTOME_IDG_FI_API_URL = 'https://idg.reactome.org/idgpairwise/relationships/combinedScoreGenesForTerm/'


class Pathway:
    def __init__(self, id, name, fdr, pVal, bottomLevel) -> None:
        self.id = id
        self.name = name
        self.fdr = fdr
        self.pVal = pVal
        self.bottomLevel = bottomLevel


def load_event_to_topic_map() -> pd.DataFrame:
    return neo4jutils.load_event_to_topic_map()


def load_pathway_dbIds() -> list:
    return neo4jutils.load_pathway_dbIds()


def load_event_summary(limit: int = None) -> pd.DataFrame:
    return neo4jutils.load_event_summary(limit=limit)


def embed_event_summary(pathway_summary_pd: pd.DataFrame,
                        model_name: str = EMBEDDING_MODEL_NAME,
                        db_path: str = '../data/faiss/reactome_pathway_index'):
    # Create a list of Document objects from the pathway dataframe
    # Iterate the pathway dataframe
    documents = []
    for index, row in pathway_summary_pd.iterrows():
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
                                                          add_start_index=True)
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
    event_topic_df = load_event_to_topic_map()
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
    result = requests.get(url='{}{}'.format(REACTOME_IDG_FI_API_URL, gene))
    json_obj = result.json()
    fi_df = pd.DataFrame({'gene': json_obj.keys(),
                          'score': json_obj.values()})
    fi_df = fi_df[fi_df['score'] > fi_cutoff]
    return fi_df


def query_reactome_interacting_pathways(gene: str,
                                        fi_cutoff: float = 0.8) -> list[Pathway]:
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
    result = requests.post(
        REACTOME_IDG_INTERACTING_PATHWAY_API_URL, json=query)
    json_objs = result.json()
    pathways = []
    for obj_data in json_objs:
        pathway = Pathway(obj_data['stId'],
                          obj_data['name'],
                          obj_data['fdr'],
                          obj_data['pVal'],
                          obj_data['bottomLevel'])
        pathways.append(pathway)
    return pathways


def collect_pathway_summary_for_interacting_pathways(gene: str,
                                                     pathway_summary_pd: pd.DataFrame,
                                                     fi_cutoff: float = 0.8,
                                                     fdr_cutoff: float = 0.01,
                                                     bottomLevel_only: bool = False) -> list[str]:
    pathways = query_reactome_interacting_pathways(gene, fi_cutoff=fi_cutoff)
    collected_reactome_summary = []

    for pathway in pathways:
        # print('{}: {}'.format(pathway.name, pathway.fdr))
        if pathway.fdr > fdr_cutoff:
            continue
        if bottomLevel_only and not pathway.bottomLevel:
            continue
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
    final_input = {
        'gene': itemgetter('gene'),
        'total_words': itemgetter('total_words'),
        'text_for_important_reactome_pathways': itemgetter('text_for_important_reactome_pathways')
    }
    prompt = prompts.summary_prompt
    answer = {
        'answer': final_input | prompt | model,
        'docs': itemgetter('docs')
    }

    # Pass a dummy runnable passthrough to make the chain work.
    dummy = RunnablePassthrough()

    doc_chain = dummy | answer
    result = doc_chain.invoke({'gene': gene,
                               'total_words': total_words,
                               'text_for_important_reactome_pathways': '\n\n'.join(collected_reactome_summary),
                               'docs': collected_reactome_summary})
    return result


def write_summary_of_interacting_pathways_for_unannotated_gene(gene: str,
                                                               collected_reactome_summary: list[str],
                                                               model,
                                                               fi_cutoff: float = 0.8,
                                                               fdr_cutoff: float = 1.0E-2,
                                                               total_words: int = 300) -> any:
    final_input = {
        'gene': itemgetter('gene'),
        'total_words': itemgetter('total_words'),
        'pathways_with_fdr': itemgetter('pathways_with_fdr'),
        'interacting_partners': itemgetter('interacting_partners'),
        'text_for_interacting_pathways': itemgetter('text_for_interacting_pathways')
    }
    prompt = prompts.unannotated_gene_prompt
    answer = {
        'answer': final_input | prompt | model,
        'docs': itemgetter('docs')
    }

    pathways = query_reactome_interacting_pathways(gene, fi_cutoff=fi_cutoff)
    pathways_with_fdr = ''
    for pathway in pathways:
        if not pathway.bottomLevel:
            continue
        if pathway.fdr < fdr_cutoff:
            pathways_with_fdr = '{}{}:{}\n'.format(
                pathways_with_fdr, pathway.name, pathway.fdr)
    # print(pathways_with_fdr)

    fi_df = query_fis(gene, fi_cutoff=fi_cutoff)
    interacting_partners = ''
    for _, fi_row in fi_df.iterrows():
        interacting_partners = '{}{}:{}\n'.format(
            interacting_partners, fi_row['gene'], fi_row['score'])

    # Pass a dummy runnable passthrough to make the chain work.
    dummy = RunnablePassthrough()

    doc_chain = dummy | answer
    result = doc_chain.invoke({'gene': gene,
                               'total_words': total_words,
                               'pathways_with_fdr': pathways_with_fdr,
                               'interacting_partners': interacting_partners,
                               'text_for_interacting_pathways': '\n'.join(collected_reactome_summary),
                               'docs': collected_reactome_summary})
    return result


async def write_summary_for_known_gene_via_paperqa(gene: str,
                                                   pathway_summary_df: pd.DataFrame,
                                                   fi_cutoff: float = 0.8,
                                                   fdr_cutoff: float = 0.01,
                                                   total_words: int = 300,
                                                   max_sources: int = 10) -> any:
    # Use sentence transformer as we did before
    # The code here is based on test_sentence_transformer_embedding in test_paperqa.py in the paper-qa GitHub repo
    # TODO: See how to use mps under mac
    embedding_model = SentenceTransformerEmbeddingModel(name=EMBEDDING_MODEL_NAME)

    text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10,
                                                          model_name=EMBEDDING_MODEL_NAME,
                                                          add_start_index=True)

    docs = Docs(
        texts_index=NumpyVectorStore(embedding_model=embedding_model),
        doc_index=NumpyVectorStore(embedding_model=embedding_model),
        embedding_client=None,
        llm='gpt-3.5-turbo',  # Maybe fine tuned (how?)
        # llm='gpt-3.5-turbo-0125', # Optimized for dialog. Support 16K! cheaper! But it is more dialog style. Better to use gpt-3.5-turbo.
        index_path='../data/paperqa'
    )

    interacting_pathway_summary = collect_pathway_summary_for_interacting_pathways(gene,
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
