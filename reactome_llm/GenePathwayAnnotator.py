from code import interact
import logging
from operator import itemgetter
from turtle import fd

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

import pandas as pd

from paperqa import EmbeddingModel
from ReactomeLLMErrors import NoAbstractFoundError, NoAbstractSupportingInteractingPathwayError, NoInteractingPathwayFoundError

import ReactomePrompts as prompts

from ReactomePubMed import ReactomePubMedRetriever
import ReactomeUtils as utils
from ReactomeUtils import Pathway
import ProteinProteinInteractionsLoader as ppi_loader

# This script should be the main entry.
import logging_config
logging_config.setup_logging()

logger = logging.getLogger(__name__)
# Used to embedding
EMBEDDING_MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'


class GenePathwayAnnotator:
    """Annotation of genes and pathways in Reactome using LLMs.
    """

    def __init__(self) -> None:
        self.ppi_loader = ppi_loader.PPILoader()

    def _get_text_splitter(self) -> TextSplitter:
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10,
                                                              model_name=EMBEDDING_MODEL_NAME,
                                                              add_start_index=True)
        return text_splitter

    def _get_embedding(self, model_name: str = EMBEDDING_MODEL_NAME) -> EmbeddingModel:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings

    def get_default_llm(self):
        # model = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
        model = ChatOpenAI(temperature=0, model='gpt-4o-mini')
        return model

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

    async def write_summary_of_interacting_pathways_for_unannotated_gene(self,
                                                                         gene: str,
                                                                         model: any=None,
                                                                         fi_cutoff: float = 0.8,
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
        ppi = self.ppi_loader.get_interactions(gene,
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

    async def build_abstract_vector_db_for_gene(self,
                                                query_gene: str,
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
        logger.debug('pubmed_query: {}'.format(pubmed_query))
        pubmed_result = pubmed_retriever.get_relevant_documents(pubmed_query)
        logger.debug('pubmed_result: {}'.format(pubmed_result))
        # In case nothing is returned
        if len(pubmed_result) == 0:
            raise NoAbstractFoundError(query_gene)

        text_splitter = self._get_text_splitter()
        docs = text_splitter.split_documents(pubmed_result)
        embeddings = self._get_embedding()
        pubmed_db = FAISS.from_documents(docs, embeddings)

        return pubmed_db

    async def write_summary_of_abstract_pathway_text(self,
                                                     query_gene: str,
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

    async def summarize_abstract_results_for_multiple_pathways(self,
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
        return self.ppi_loader.get_interactions(gene, fi_cutoff=fi_cutoff)

    async def write_summary_of_abstracts_for_gene_annotation(self,
                                                             query_gene: str,
                                                             pubmed_db: VectorStore,
                                                             fi_cutoff: float = 0.8,
                                                             fdr_cutoff: float = 0.05,
                                                             pathway_count: int = 8,
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
        interaction_dict = self.ppi_loader.get_interactions(query_gene, fi_cutoff=fi_cutoff, filter_ppis_with_fi=True)
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
        
        pathway_abstract_pd = await self.build_pathway_abstract_df(query_gene,
                                                                   pathway_enrichment_df,
                                                                   pubmed_db,
                                                                   model)
        if pathway_abstract_pd is None or pathway_abstract_pd.empty:
            raise NoAbstractSupportingInteractingPathwayError(query_gene)

        abstract_result_for_multiple_pathways = await self.summarize_abstract_results_for_multiple_pathways(query_gene,
                                                                                                            pathway_abstract_pd,
                                                                                                            model)
        return abstract_result_for_multiple_pathways

    async def build_pathway_abstract_df(self,
                                        query_gene: str,
                                        pathway_enrichment_results_df: pd.DataFrame,
                                        pubmed_db: VectorStore,
                                        model: BaseChatModel) -> pd.DataFrame:
        pathway_abstract_pd = pd.DataFrame(
            columns=['pathway', 'ppi_genes', 'ppi_genes_pmids', 'pmid', 'title', 'abstract', 'score', 'summary'])
        row_index = 0
        text_splitter = self._get_text_splitter()
        for _, row in pathway_enrichment_results_df.iterrows():
            pathway = row['pathway_name']
            interacting_genes = row['mapped_genes']
            # Need the text only
            pathway_text, _, _ = utils.create_interacting_pathway_text(pathway, interacting_genes)
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
            logger.debug('\n\nBest matched abstract: {}'.format(
                best_matched_abstract))
            # TODO: 1). Make sure interacting_genes are for the genes in the specific pathway only, not the query genes
            # 2). Make sure the pathway text and the abstract text are the full, not just the splitted ones.
            abstract_result = await self.write_summary_of_abstract_pathway_text(query_gene,
                                                                                interacting_genes,
                                                                                pathway,
                                                                                splitted_text,
                                                                                best_matched_abstract[0].page_content,
                                                                                model)
            # In case there is no title
            title = '' if 'Title' not in best_matched_abstract[0].metadata.keys() else best_matched_abstract[0].metadata['Title']
            pathway_abstract_pd.loc[row_index] = [pathway,
                                            interacting_genes,
                                            row['pmids_all'],
                                            best_matched_abstract[0].metadata['uid'],
                                            title,
                                            best_matched_abstract[0].page_content,
                                            best_matched_abstract[1],
                                            abstract_result['answer'].content]
            row_index += 1
        logger.debug('pathway_abstract_pd:\n{}'.format(pathway_abstract_pd.head()))
        return pathway_abstract_pd

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
