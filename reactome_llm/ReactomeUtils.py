from operator import itemgetter


from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import PubMedRetriever

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


def _get_text_splitter() -> TextSplitter:
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10,
                                                        model_name=EMBEDDING_MODEL_NAME,
                                                        add_start_index=True)
    return text_splitter

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
    fi_df.sort_values(by=['score'], ascending=False, inplace=True)
    return fi_df


def query_reactome_interacting_pathways(gene: str,
                                        fi_cutoff: float = 0.8,
                                        fdr_cutoff: float = 0.01,
                                        bottomLevel_only: bool = True,
                                        pathway_count: int = 10) -> list[Pathway]:
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
        pathways.append(pathway)
    return pathways


def collect_pathway_summary_for_interacting_pathways(gene: str,
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
    pathway_text = await create_pathway_text(pathway, 
                                              interacting_genes)

    final_input = {
        'gene': itemgetter('gene'),
        'total_words': itemgetter('total_words'),
        'interacting_pathway_text': itemgetter('interacting_pathway_text')
    }
    prompt = prompts.interacting_pathway_summary_prompt
    answer = {
        'answer': final_input | prompt | model,
        'docs': itemgetter('docs')
    }

    # Pass a dummy runnable passthrough to make the chain work.
    dummy = RunnablePassthrough()

    doc_chain = dummy | answer
    result = doc_chain.invoke({'gene': query_gene,
                               'total_words': total_words,
                               'interacting_pathway_text': pathway_text,
                               'docs': pathway_text})
    return result


async def create_pathway_text(pathway: str,
                             interacting_genes: list[str]):
    # Fetch the roles of interacting genes in pathways according to reactions
    reaction_roles_df = neo4jutils.query_reaction_roles_of_pathway(
        pathway, interacting_genes)
    # print(reaction_roles_df)
    reaction_gene_role_text = ''
    for _, row in reaction_roles_df.iterrows():
        reaction = row['reaction']
        gene = row['gene']
        role = row['role']
        if len(reaction_gene_role_text) > 0:
            reaction_gene_role_text = reaction_gene_role_text + "; "
        reaction_gene_role_text = '{}{} in "{}" as {}'.format(
            reaction_gene_role_text, gene, reaction, role)

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
        ', '.join(interacting_genes),
        reaction_gene_role_text
    )
    return pathway_text


async def write_summary_of_interacting_pathways_for_unannotated_gene(gene: str,
                                                                     model,
                                                                     fi_cutoff: float = 0.8,
                                                                     fdr_cutoff: float = 1.0E-2,
                                                                     pathway_count: int = 8,
                                                                     total_words: int = 300) -> any:
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
        pathway_text = pathway_result['answer'].content
        pathway_text_list.append('{}: {}'.format(pathway, pathway_text))
    pathway_text_all = '\n\n'.join(pathway_text_list)

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

    # Pass a dummy runnable passthrough to make the chain work.
    dummy = RunnablePassthrough()

    doc_chain = dummy | answer
    result = doc_chain.invoke({'gene': gene,
                               'total_words': total_words,
                               'pathways_with_fdr': pathways_with_fdr,
                               'interacting_partners': ','.join(interacting_genes),
                               'text_for_interacting_pathways': pathway_text_all,
                               'docs': pathway_text_all})
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
    embedding_model = SentenceTransformerEmbeddingModel(
        name=EMBEDDING_MODEL_NAME)

    text_splitter = _get_text_splitter()

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


def build_abstract_vector_db_for_gene(query_gene: str,
                                      top_k_results: int = 8,
                                      max_query_length: int = 1000) -> any:
    pubmed_retriever = PubMedRetriever()
    pubmed_retriever.top_k_results = top_k_results
    pubmed_retriever.MAX_QUERY_LENGTH = max_query_length
    pubmed_retriever.doc_content_chars_max = max_query_length * top_k_results

    pubmed_query = '{} interactions or {} reactions or {} pathways'.format(query_gene, query_gene, query_gene)
    pubmed_result = pubmed_retriever.get_relevant_documents(pubmed_query)
    
    text_splitter = _get_text_splitter()
    docs = text_splitter.split_documents(pubmed_result)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    pubmed_db = FAISS.from_documents(docs, embeddings)

    return pubmed_db


async def write_summary_of_abstract_pathway_text(query_gene: str,
                                                 interacting_genes: list[str],
                                                 pathway: str,
                                                 pathway_text: str,
                                                 abstract_text: str,
                                                 model: any,
                                                 total_words: int = 150) -> any:
    final_input = {
        'query_gene': itemgetter('query_gene'),
        'interacting_genes': itemgetter('interacting_genes'),
        'total_words': itemgetter('total_words'),
        'pathway_text': itemgetter('pathway_text'),
        'abstract_text': itemgetter('abstract_text'),
        'pathway': itemgetter('pathway')
    }
    prompt = prompts.abstract_summary_prompt
    answer = {
        'answer': final_input | prompt | model,
        'docs': itemgetter('docs')
    }

    # Pass a dummy runnable passthrough to make the chain work.
    dummy = RunnablePassthrough()

    doc_chain = dummy | answer
    result = doc_chain.invoke({'query_gene': query_gene,
                               'interacting_genes': ','.join(interacting_genes),
                               'total_words': total_words,
                               'pathway_text': pathway_text,
                               'abstract_text': abstract_text,
                               'pathway': pathway,
                               'docs': '{}\n\n{}'.format(pathway_text, abstract_text)})
    return result


async def summarize_abstract_results_for_multiple_pathways(query_gene: str,
                                                           interacting_genes: list[str],
                                                           pathway_abstract_summary_df: pd.DataFrame,
                                                           model: any,
                                                           total_words: int = 300):
    context = ''
    for _, row in pathway_abstract_summary_df.iterrows():
        pathway = row['pathway']
        pmid = row['pmid']
        abstract_summary = row['summary']
        if len(context) > 0:
            context = '{}\n'.format(context)
        context = '{}PMID:{};PATHWAY_NAME:"{}": {}'.format(context, pmid, pathway, abstract_summary)

    final_input = {
        'query_gene': itemgetter('query_gene'),
        'interacting_genes': itemgetter('interacting_genes'),
        'total_words': itemgetter('total_words'),
        'context': itemgetter('context'),
    }
    prompt = prompts.multiple_abstracts_summary_prompt
    answer = {
        'answer': final_input | prompt | model,
        'docs': itemgetter('docs')
    }

    # Pass a dummy runnable passthrough to make the chain work.
    dummy = RunnablePassthrough()

    doc_chain = dummy | answer
    result = doc_chain.invoke({'query_gene': query_gene,
                               'interacting_genes': ','.join(interacting_genes),
                               'total_words': total_words,
                               'context': context,
                               'docs': context})
    return result


async def write_summary_of_abstracts_for_gene(query_gene: str,
                                              pubmed_db: VectorStore,
                                              model: any):
    # Get pathways and genes
    pathways = query_reactome_interacting_pathways(
        query_gene, 
        pathway_count=8)
    log.debug('Total pathways for {}: {}'.format(query_gene, len(pathways)))
    fi_df = query_fis(gene=query_gene)
    interacting_genes = fi_df['gene'].to_list()
    log.debug('Total interacting genes: {}'.format(len(interacting_genes)))

    pathway_abstract_pd = pd.DataFrame(
        columns=['pathway', 'pmid', 'title', 'summary'])
    row = 0
    text_splitter = _get_text_splitter()
    for pathway in pathways:
        pathway_text = await create_pathway_text(pathway.name, interacting_genes)
        splitted_texts = text_splitter.split_text(pathway_text)
        best_matched_abstract = None
        for splitted_text in splitted_texts:
            # Just need the top scored text
            matched_abstract_score = pubmed_db.similarity_search_with_score(splitted_text)[0]
            # print('{}:\n{}'.format(splitted_text, matched_abstract_score))
            if not best_matched_abstract:
                best_matched_abstract = matched_abstract_score
            else:
                if matched_abstract_score[1] < best_matched_abstract[1]:
                    best_matched_abstract = matched_abstract_score
        log.debug('\n\nBest matched abstract: {}'.format(best_matched_abstract))
        abstract_result = await write_summary_of_abstract_pathway_text(query_gene,
                                                                    interacting_genes,
                                                                    pathway.name,
                                                                    splitted_text,
                                                                    best_matched_abstract[0].page_content,
                                                                    model)
        pathway_abstract_pd.loc[row] = [pathway.name,
                                        best_matched_abstract[0].metadata['uid'],
                                        best_matched_abstract[0].metadata['Title'],
                                        abstract_result['answer'].content]
        row += 1
    log.debug('pathway_abstract_pd:\n{}'.format(pathway_abstract_pd.head()))
    abstract_result_for_multiple_pathways = await summarize_abstract_results_for_multiple_pathways(query_gene,
                                                                                               interacting_genes,
                                                                                               pathway_abstract_pd,
                                                                                               model)
    return abstract_result_for_multiple_pathways, interacting_genes, pathway_abstract_pd