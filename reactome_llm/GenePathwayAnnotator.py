import logging
from operator import itemgetter
import re
from typing import List
from xml.dom.minidom import Document

from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd

from paperqa import EmbeddingModel
from ReactomeLLMErrors import NoAbstractFoundError, NoAbstractSupportingInteractingPathwayError, NoAbstractSupportingProteinInteractions, NoInteractingPathwayFoundError

import ReactomePrompts as prompts

from ReactomePubMed import ReactomePubMedRetriever
import ReactomeUtils as utils
import ProteinProteinInteractionsLoader as ppi_loader

# This script should be the main entry.
import logging_config
logging_config.setup_logging()

logger = logging.getLogger(__name__)

# Disable INFO and lower level logs from the sentence_transformers library
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

# Used to embedding
# NB: Apparent this model cannot generate a good cosine similarity score.
# e.g. the following two sentences are not similar at all, but the cosine similarity is 0.8
# sentence1 = "The cat sat on the windowsill, basking in the warm afternoon sun."
# sentence2 = "Quantum mechanics explores the behavior of particles at the subatomic level."
# EMBEDDING_MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'
# Use this generic model gives us a better result
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'


class GenePathwayAnnotator:
    """Annotation of genes and pathways in Reactome using LLMs.
    """

    def __init__(self) -> None:
        self.ppi_loader = None
        self.model = None

    def _get_text_splitter(self) -> TextSplitter:
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10,
                                                              model_name=EMBEDDING_MODEL_NAME,
                                                              add_start_index=True)
        return text_splitter

    def _get_embedding(self, model_name: str = EMBEDDING_MODEL_NAME) -> EmbeddingModel:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        embeddings.model_kwargs['show_progress_bar'] = False
        return embeddings
    
    def _get_pubmed_retriver(self, 
                             top_k_results: int = 8,
                             max_query_length: int = 1000) -> ReactomePubMedRetriever:
        pubmed_retriever = ReactomePubMedRetriever()
        # Make sure it doesn't exceed the quote: 3 per second. use 0.5.
        pubmed_retriever.sleep_time = 0.5
        pubmed_retriever.top_k_results = top_k_results
        pubmed_retriever.MAX_QUERY_LENGTH = max_query_length
        pubmed_retriever.doc_content_chars_max = max_query_length * top_k_results
        return pubmed_retriever
    
    def get_ppi_loader(self):
        if self.ppi_loader is None:
            self.ppi_loader = ppi_loader.PPILoader()
        return self.ppi_loader
    
    def set_ppi_loader(self, ppi_loader):
        self.ppi_loader = ppi_loader

    def set_model(self, model: any):
        """Set the model to be used.

        Args:
            model (any): _description_
        """
        self.model = model

    def get_default_llm(self):
        if self.model is not None:
            return self.model
        # model = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
        model = ChatOpenAI(temperature=0, model='gpt-4o-mini')
        return model

    async def write_summary_of_annotated_pathway(self,
                                                 query_gene: str,
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
        pathway_text = utils.create_annotated_pathway_text(pathway, query_gene)
        if pathway_text is None:
            return None

        prompt = prompts.annotated_pathway_summary_prompt
        parameters = {'gene': query_gene,
                      'total_words': total_words,
                      'annotated_pathway_text': pathway_text,
                      'docs': pathway_text}
        return self.invoke_llm(parameters, prompt, model)

    async def write_summary_of_annotated_pathways(self,
                                                  query_gene: str,
                                                  model: any = None,
                                                  total_words: int = 300) -> any:
        """Write a summary for multiple pathways for a gene annotated in Reactome

        Args:
            query_gene (str): _description_
            model (any): _description_
            total_words (int, optional): _description_. Defaults to 300.

        Returns:
            any: _description_
        """
        pathways = utils.get_annotated_pathways(query_gene)
        annotated_pathways = [
            pathway for pathway in pathways if pathway.annotated]
        if len(annotated_pathways) == 0:
            return None
        if len(annotated_pathways) == 1:  # No need to write any summary
            return await self.write_summary_of_annotated_pathway(query_gene,
                                                                 annotated_pathways[0].name,
                                                                 model,
                                                                 total_words=300)
        pathway_text_list = []
        for pathway in annotated_pathways:
            pathway_result = await self.write_summary_of_annotated_pathway(query_gene,
                                                                           pathway.name,
                                                                           model)
            if pathway_result is None:
                continue
            pathway_text = pathway_result['answer'].content
            pathway_text_list.append(
                '{}: {}'.format(pathway.name, pathway_text))
        pathway_text_all = '\n\n'.join(pathway_text_list)

        prompt = prompts.annotated_pathways_summary_prompt
        parameters = {'gene': query_gene,
                      'total_words': total_words,
                      'annotated_pathways_text': pathway_text_all,
                      'docs': pathway_text_all}
        result = self.invoke_llm(parameters, prompt, model)

        return result

    async def write_summary_of_interacting_pathway_for_unannotated_gene(self,
                                                                        query_gene: str,
                                                                        interacting_genes: list[str],
                                                                        pathway: str,
                                                                        model: any = None,
                                                                        total_words: int = 150) -> any:
        """Write a summary for one single interacting pathway for an unannotated gene.

        Args:
            gene (str): _description_
            pathway_text (str): _description_
            total_words (int, optional): _description_. Defaults to 150.

        Returns:
            any: _description_
        """
        pathway_text, interacting_genes_text, gene_roles_text = utils.create_interacting_pathway_text(pathway,
                                                                                                      interacting_genes)
        if pathway_text is None:
            return None

        prompt = prompts.interacting_pathway_summary_prompt
        parameters = {'gene': query_gene,
                      'total_words': total_words,
                      'interacting_genes': interacting_genes_text,
                      'roles_of_genes': gene_roles_text,
                      'interacting_pathway_text': pathway_text,
                      'docs': '{}\n\n{}\n\n{}'.format(pathway_text, interacting_genes_text, gene_roles_text)}
        return self.invoke_llm(parameters, prompt, model)


    async def write_summary_of_interacting_pathways_for_unannotated_gene(self,
                                                                         gene: str,
                                                                         model: any=None,
                                                                         fi_cutoff: float = 0.8,
                                                                         fdr_cutoff: float = 0.05,
                                                                         pathway_count: int = 8,
                                                                         total_words: int = 300) -> any:
        """Write a summary for a set of interacting pathways for a gene that has not been annotated in Reactome.
        The interacting pathways are fetched based on PPIs collected from IntAct and BioGrid. The summary has not
        been supported by any abstract.

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
        ppi = self.get_ppi_loader().get_interactions(gene,
                                               filter_ppis_with_fi=True,
                                               fi_cutoff=fi_cutoff)
        if ppi is None:
            raise NoInteractingPathwayFoundError(
                'No interacting pathway found for {}'.format(gene))

        # Make sure this is a list.
        interacting_genes = list(ppi.keys())

        ppis_in_pathways_df = utils.map_interactions_in_pathways(ppi)
        # Here we'd like to have all pathways that are mapped. Therefore, we use fdr cutoff = 1.0
        pathway_enrichment_results = utils.pathway_binomial_enrichment_df(ppis_in_pathways_df,
                                                                          interacting_genes,
                                                                          fdr_cutoff=fdr_cutoff)
        if pathway_enrichment_results is None or pathway_enrichment_results.empty:
            raise NoInteractingPathwayFoundError(
                'No interacting pathway found for {}'.format(gene))

        pathways_with_fdr = ''
        selected_pathways = []
        pathway_text_list = []
        for _, row in pathway_enrichment_results.iterrows():
            if (len(selected_pathways)) >= pathway_count:
                break
            pathway_result = await self.write_summary_of_interacting_pathway_for_unannotated_gene(query_gene=gene,
                                                                                                  interacting_genes=row[
                                                                                                      'mapped_genes'],
                                                                                                  pathway=row['pathway_name'],
                                                                                                  model=model)
            if pathway_result is None:
                continue
            pathway_text = pathway_result['answer'].content
            pathway_text_list.append('{}: {}'.format(
                row['pathway_name'], pathway_text))
            pathways_with_fdr = '{}{}:{}\n'.format(pathways_with_fdr,
                                                   row['pathway_name'],
                                                   row['FDR'])
            selected_pathways.append(row['pathway_name'])

        if len(selected_pathways) == 0:
            raise NoInteractingPathwayFoundError(
                'No interacting pathway found for {}'.format(gene))

        pathway_text_all = '\n\n'.join(pathway_text_list)

        prompt = prompts.interacting_pathways_summary_prompt
        parameters = {'gene': gene,
                      'total_words': total_words,
                      'pathways_with_fdr': pathways_with_fdr,
                      'interacting_partners': ','.join(interacting_genes),
                      'text_for_interacting_pathways': pathway_text_all,
                      'docs': pathway_text_all}
        result = self.invoke_llm(parameters, prompt, model)

        # Returns only whatever is used
        enrichment_df = pathway_enrichment_results[pathway_enrichment_results['pathway_name'].isin(
            selected_pathways)]

        return result, enrichment_df
    
    async def summarize_pubmed_abstracts_for_interactions(self,
                                                          query_gene: str,
                                                          pathway: str,
                                                          interactors: list,
                                                          pmids: set,
                                                          model: any = None,
                                                          total_words: int=300,
                                                          top_abstracts: int = 8) -> any:
        """Write a summary for a list of abstracts that are collected from PPIs.

        Args:
            pmids (set): _description_
            model (any): _description_
            top_abstracts (int, optional): _description_. Defaults to 8.

        Returns:
            any: _description_
        """
        # Expected to see abstracts
        abstract_df = self._select_top_abstracts_for_interactions(pathway, pmids, top_abstracts)
        if abstract_df is None or abstract_df.empty:
            return NoAbstractSupportingProteinInteractions(query_gene)
        abstract_text = ''
        for _, row in abstract_df.iterrows():
            pmid = row['pmid']
            abstract = row['abstract']
            abstract_text = '{}PMID {}:{}\n\n'.format(abstract_text, pmid, abstract)
        prompt = prompts.protein_interaction_abstracts_summary_prompt
        parameters = {'query_gene': query_gene,
                      'interactors': ','.join(interactors),
                      'pathway': pathway,
                      'context': abstract_text,
                      'total_words': total_words,
                      'docs': abstract_text}
        result = self.invoke_llm(parameters, prompt, model)
        return result, abstract_df

    def _select_top_abstracts_for_interactions(self, pathway, pmids, top_abstract) -> pd.DataFrame:
        # Need to do filtering
        pubmed_retriever = self._get_pubmed_retriver()
        # Use pd.DataFrame as the data structure so that we can do sorting
        abstract_df = pd.DataFrame(
            columns=['pathway', 'pmid', 'abstract', 'cos_score']
        )
        row = 0
        # Pull abstracts
        for pmid in pmids:
            abstract = pubmed_retriever.get_abstract_from_mongodb(pmid)
            if abstract is None:
                continue
            abstract_df.loc[row] = [pathway, pmid, abstract['Summary'], None]
            row += 1
        if abstract_df.shape[0] > top_abstract:
            # Sort based on pathway information
            pathway_embedding = self._embed_text(pathway)[0] # Pathway is just a name and there should be just one embedding
            for _, row in abstract_df.iterrows():
                abstract: HuggingFaceEmbeddings = row['abstract']
                # This is a quick way. We basically just do a simple
                abstract_embeddings = self._embed_text(abstract)
                cos_score = np.max([cosine_similarity([pathway_embedding], [abstract_embedding])[0][0] 
                             for abstract_embedding in abstract_embeddings])
                row['cos_score'] = cos_score
            abstract_df.sort_values(by=['cos_score'], ascending=False, inplace=True)
            abstract_df = abstract_df.head(top_abstract)
        return abstract_df
    

    async def query_pubmed_abstracts_for_gene(self,
                                            query_gene: str,
                                            top_k_results: int = 8,
                                            max_query_length: int = 1000) -> List[Document]:
        """Query pubmed about interactions, reactions, and pathways for a gene and return
        a vector store for collected abstracts from PubMed.

        Args:
            query_gene (str): _description_
            top_k_results (int, optional): _description_. Defaults to 8.
            max_query_length (int, optional): _description_. Defaults to 1000.

        Returns:
            any: _description_
        """
        pubmed_retriever = self._get_pubmed_retriver(top_k_results=top_k_results,
                                                     max_query_length=max_query_length)
        
        pubmed_query = '{} interactions or {} reactions or {} pathways'.format(
            query_gene, query_gene, query_gene)
        logger.debug('pubmed_query: {}'.format(pubmed_query))
        pubmed_result = pubmed_retriever.get_relevant_documents(pubmed_query)
        logger.debug('pubmed_result: {}'.format(pubmed_result))
        # In case nothing is returned
        if len(pubmed_result) == 0:
            raise NoAbstractFoundError(query_gene)
        
        return pubmed_result
    

    async def build_abstract_vector_db_for_gene(self,
                                                pubmed_result: List[Document]) -> VectorStore:
        """Query pubmed about interactions, reactions, and pathways for a gene and return
        a vector store for collected abstracts from PubMed.

        Args:
            query_gene (str): _description_
            top_k_results (int, optional): _description_. Defaults to 8.
            max_query_length (int, optional): _description_. Defaults to 1000.

        Returns:
            any: _description_
        """
        text_splitter = self._get_text_splitter()
        docs = text_splitter.split_documents(pubmed_result)
        embeddings = self._get_embedding()
        pubmed_db = FAISS.from_documents(docs, embeddings)

        return pubmed_db

    async def _write_summary_of_abstract_pathway_text(self,
                                                     query_gene: str,
                                                     interacting_genes: list[str],
                                                     pathway: str,
                                                     pathway_text: str,
                                                     abstract_text: str,
                                                     model: any,
                                                     total_words: int = 150) -> any:
        """Create a summary for an abstract that is collected for a pathway.

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
        result = self.invoke_llm(parameters, prompt, model)
        return result

    def invoke_llm(self,
                   parameters: dict,
                   prompt: ChatPromptTemplate,
                   model: any) -> any:
        if model is None:
            model = self.get_default_llm()
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

    async def _summarize_abstract_results_for_multiple_pathways(self,
                                                               query_gene: str,
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
        interacting_genes = []
        for _, row in pathway_abstract_summary_df.iterrows():
            pathway = row['pathway']
            pmid = row['pmid']
            abstract_summary = row['summary']
            if len(context) > 0:
                context = '{}\n\n'.format(context)
            context = '{}PMID:{};PATHWAY_NAME:"{}": {}'.format(
                context, pmid, pathway, abstract_summary)
            interacting_genes.extend(row['ppi_genes'])
        interacting_genes = list(set(interacting_genes))

        parameters = {'query_gene': query_gene,
                      'interacting_genes': ','.join(interacting_genes),
                      'total_words': total_words,
                      'context': context,
                      'docs': context}
        result = self.invoke_llm(parameters,
                                 prompts.multiple_abstracts_summary_prompt,
                                 model)

        return result

    def query_fis(self, gene, fi_cutoff):
        return self.get_ppi_loader().get_interactions(gene, fi_cutoff=fi_cutoff)

    async def write_summary_for_gene_annotation(self,
                                                query_gene: str,
                                                pubmed_results: List[Document],
                                                fi_cutoff: float = 0.8,
                                                fdr_cutoff: float = 0.05,
                                                pathway_count: int = 8,
                                                pathway_abstract_similiary: float = 0.4,
                                                llm_score: int = 3,
                                                model: any = None):
        """Write a summary for a query gene by collecting Reactome pathway related abstracts from PubMed.
        This function basically is a wrap of multiple calls of LLMs, as well as PubMed retrieval.

        Args:
            query_gene (str): _description_
            pubmed_db (VectorStore): _description_
            model (any): _description_

        Returns:
            _type_: _description_
        """
        interaction_dict = self.get_ppi_loader().get_interactions(query_gene, fi_cutoff=fi_cutoff, filter_ppis_with_fi=True)
        interaction_map_df = utils.map_interactions_in_pathways(interaction_dict)
        pathway_enrichment_df = utils.pathway_binomial_enrichment_df(interaction_map_df,
                                                                     interaction_dict.keys(),
                                                                     fdr_cutoff=fdr_cutoff)
        if pathway_enrichment_df is None or pathway_enrichment_df.empty:
            raise NoInteractingPathwayFoundError(query_gene)
        logger.debug('Total pathways for {}: {}'.format(
            query_gene, pathway_enrichment_df.shape[0]))
        if pathway_enrichment_df.shape[0] > pathway_count: # Pick the top pathways
            pathway_enrichment_df = pathway_enrichment_df.head(pathway_count)
        
        pathway_abstract_pd = await self.build_pathway_abstract_df_from_docs(query_gene,
                                                                   pathway_enrichment_df,
                                                                   pubmed_results,
                                                                   model,
                                                                   similarity=pathway_abstract_similiary,
                                                                   llm_score=llm_score)
        if pathway_abstract_pd is None or pathway_abstract_pd.empty:
            raise NoAbstractSupportingInteractingPathwayError(query_gene)

        abstract_result_for_multiple_pathways = await self._summarize_abstract_results_for_multiple_pathways(query_gene,
                                                                                                            pathway_abstract_pd,
                                                                                                            model)
        return abstract_result_for_multiple_pathways, pathway_abstract_pd


    async def validate_similarity_of_abstract_pathway_text(self,
                                                           pathway: str,
                                                           pathway_text: str,
                                                           abstract_text: str,
                                                           model: any) -> any:
        prompt = prompts.abstract_pathway_match_prompt
        parameters = {'pathway': pathway,
                      'pathway_text': pathway_text,
                      'abstract_text': abstract_text,
                      'docs': '{}\n\n{}'.format(pathway_text, abstract_text)}
        result = self.invoke_llm(parameters, prompt, model)
        return result
    

    async def build_pathway_abstract_df_from_docs(self,
                                                query_gene: str,
                                                pathway_enrichment_results_df: pd.DataFrame,
                                                pubmed_results: List[Document],
                                                model: BaseChatModel,
                                                total_abstracts: int=3,
                                                similarity: float=0.4,
                                                llm_score: int=3) -> pd.DataFrame:
        """Build the pathway abstract dataframe from the abstracts directly without building the index
        using vector store. We use this method so that we can have a fine control on retrieving the abstracts.

        Args:
            query_gene (str): _description_
            pathway_enrichment_results_df (pd.DataFrame): _description_
            pubmed_db (VectorStore): _description_
            model (BaseChatModel): _description_
            total_abstracts: the number of the abstracts collected for a pathway.
            similarity: the cosine similarity used to filter out abstracts that have less similarity.
            llm_score: the similarity score determined by the llm model.
        Returns:
            pd.DataFrame: _description_
        """
        # To increase the embedding efficiency, we will cache the results
        pmid2embedding = {}
        for doc in pubmed_results:
            pmid2embedding[doc.metadata['uid']] = self._embed_text(doc.page_content)
        pathway_abstract_pd = pd.DataFrame(
            columns=['pathway', 'pathway_text', 'ppi_genes', 'ppi_genes_pmids', 
                     'pmid', 'title', 'abstract', 'cos_score', 'summary', 'llm_score'])
        row_index = 0
        for _, row in pathway_enrichment_results_df.iterrows():
            pathway = row['pathway_name']
            interacting_genes = row['mapped_genes']
            # Need the text only
            pathway_text, _, _ = utils.create_interacting_pathway_text(pathway, interacting_genes)
            if pathway_text is None:
                continue
            # Use the pd.DataFrame so that we can sort if needed
            pathway_abstract_match_pd = pd.DataFrame(
                columns=['pmid', 'abstract', 'cos_score', 'llm_score']
            )
            pathway_abstract_match_pd_row = 0
            for doc in pubmed_results:
                pmid = doc.metadata['uid']
                abstract = doc.page_content
                cos_similarity = self._average_cos_similiarity(
                    pathway_text, pmid2embedding[pmid])
                if cos_similarity < similarity:
                    continue
                llm_similarity_result = await self.validate_similarity_of_abstract_pathway_text(
                    pathway, pathway_text,
                    abstract,
                    model)
                # Cannot call directly in the above statement.
                llm_similarity_result = llm_similarity_result['answer'].content
                # extract the score using RE
                match = re.search(r"score:\s*(\d+)", llm_similarity_result)
                llm_score_result = int(match.group(1)) if match else None
                if llm_score_result is not None and llm_score_result < llm_score:
                    continue
                pathway_abstract_match_pd.loc[pathway_abstract_match_pd_row] = [
                    pmid,
                    abstract,
                    cos_similarity,
                    llm_score_result
                ]
                pathway_abstract_match_pd_row += 1
            # Nothing for this interacting pathway. Try next.
            if pathway_abstract_match_pd.empty:
                continue
            # Sort and take the top. Sort is based on cos_score. But we may sore based on the sum of the two scores.
            pathway_abstract_match_pd.sort_values(by=['cos_score', 'llm_score'], inplace=True, ascending=False)
            pathway_abstract_match_pd = pathway_abstract_match_pd.head(total_abstracts)
            for _, match_row in pathway_abstract_match_pd.iterrows():
                abstract_result = await self._write_summary_of_abstract_pathway_text(query_gene,
                                                                                    interacting_genes,
                                                                                    pathway,
                                                                                    pathway_text,
                                                                                    match_row['abstract'],
                                                                                    model)

                pathway_abstract_pd.loc[row_index] = [pathway,
                                                    pathway_text,
                                                    interacting_genes,
                                                    row['pmids_all'],
                                                    match_row['pmid'],
                                                    '',
                                                    match_row['abstract'],
                                                    match_row['cos_score'],
                                                    abstract_result['answer'].content,
                                                    match_row['llm_score']]
                row_index += 1
        logger.debug('pathway_abstract_pd:\n{}'.format(pathway_abstract_pd.head()))
        return pathway_abstract_pd
    
    def _embed_text(self, text1: str) -> np.array:
        # Split the texts into chunks
        chunks1 = self._get_text_splitter().split_text(text1)
        
        # Get embeddings for each chunk
        # embedder: HuggingFaceEmbeddings = self._get_embedding()
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings1 = np.array([embedder.encode(chunk, show_progress_bar=False) for chunk in chunks1])
        return embeddings1
    
    def _average_cos_similiarity(self, text1: str, abstract_embeddings: np.array) -> float:
        embeddings1 = self._embed_text(text1)
        
        # Calculate pairwise cosine similarities between the chunks of text1 and text2
        similarity_scores = []
        for emb1 in embeddings1:
            for emb2 in abstract_embeddings:
                similarity = cosine_similarity([emb1], [emb2])[0][0]  # Cosine similarity score
                similarity_scores.append(similarity)
        
        # Return the average of all pairwise similarities
        avg_similarity = np.mean(similarity_scores)
        return avg_similarity
        

    def analyze_full_paper(self,
                           paper_file_name: str,
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
        token_splitter = self._get_text_splitter()
        pages = loader.load_and_split(token_splitter)

        # Embedding the paper
        embeddings = self._get_embedding()
        paper_db = FAISS.from_documents(pages, embeddings)

        # Fetch the best matched text
        query = f'({query_gene} interactions) or ({query_gene} reactions) or ({query_gene} pathways)'
        matched_pages = paper_db.similarity_search_with_score(
            query, k=top_pages)

        # Prepare to call llm
        parameters = {
            'query_gene': query_gene
        }
        prompt = prompts.relationship_extraction_prompt
        results = []
        for doc, score in matched_pages:
            if score > max_score:
                # The returned results are sorted. If we see this, we can break the loop.
                break
            parameters['docs'] = doc.page_content
            parameters['document'] = doc.page_content
            result = self.invoke_llm(
                model=model, parameters=parameters, prompt=prompt)
            results.append(result)
        return results

    def output_llm_result(self, result):
        formatted_result = '\nResult: {}\n\nDoc: {}'.format(
            result['answer'].content, result['docs'])
        return formatted_result
