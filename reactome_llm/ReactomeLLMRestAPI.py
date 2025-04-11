import logging as log
import os
from pathlib import Path

from sympy import public
from torch import mode

import ReactomeNeo4jUtils as neo4jutils
import ReactomeUtils as utils
import GenePathwayAnnotator as gpa

import re
from langchain_openai import ChatOpenAI
from flask_cors import CORS
from flask import Flask, request
import warnings

from ReactomeLLMErrors import NoAbstractFoundError, NoAbstractSupportingInteractingPathwayError, NoInteractingPathwayFoundError
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()
PDF_PAPERS_FOLDER = os.getenv('PDF_PAPERS_FOLDER')

# Shared variables for all
api = Flask(__name__)
CORS(api)
# model = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
# As of Dec 18, 2024, switch to 4o-mini.
model = ChatOpenAI(temperature=0, model='gpt-4o-mini')

# This is for test
return_json = None
full_text_return_json = None

# The following code is used to set up an instance of GenePathwayAnnotator.
annotator = gpa.GenePathwayAnnotator()
annotator.set_model(model)

# Test genes: TANC1 (not annotated), DUX4L2 (not annotated, limited abstracts), NTN1 (annotated)
# TODO: Add cellular locations and cell types reuqest into the prompts!
# TODO: This is important. Add GUIs to configure the cutoff of FIs and pathway FDRs and also make sure they are 
# consistent across the whole application.

# Most likely this is temporary. Need to think how to encript it.
@api.route('/openai_key')
async def get_openai_key():
    return os.getenv('OPENAI_API_KEY')


@api.route('/fulltext/<pmid>/<gene>')
async def analyze_full_text(pmid, gene):
    # global full_text_return_json
    # if full_text_return_json:
    #     return full_text_return_json
    # For test
    # pdf_file = '/Users/wug/git/reactome/curator-tool-llm/data/papers/zns15102.pdf'
    file_path = Path(PDF_PAPERS_FOLDER, '{}.pdf'.format(pmid))
    # print(str(file_path))
    full_text_return_json = []
    if not file_path.exists():
        # Make sure the same data structure is returned
        result_json = {
            'failure': 'The PDF full text paper cannot be found at the server'
        }
        full_text_return_json.append(result_json)
        return full_text_return_json
    results = utils.analyze_full_paper(str(file_path), gene, model, top_pages=4) # Use 4 for test
    
    for result in results:
        result_json = {
            'content': result['answer'].content,
            'docs':  result['docs'],
        }
        full_text_return_json.append(result_json)
    return full_text_return_json


@api.route('/fulltext/check_pdf/<pmid>')
async def pdfExists(pmid: str) -> bool:
    file_path = Path(PDF_PAPERS_FOLDER, '{}.pdf'.format(pmid))
    return True if file_path.exists else False


@api.route('/fulltext/uploadPDF', methods=['POST'])
async def uploadPDF():
    if 'pdf' not in request.files and 'pmid' not in request.form:
        return {'status': 'no file provided or no pmid provided'}
    try:
        pdf = request.files['pdf']
        pmid = request.form['pmid']
        # utils.save_pdf_paper(pdf, pmid, PDF_PAPERS_FOLDER)
        file_name = '{}'.format(Path(PDF_PAPERS_FOLDER, '{}.pdf'.format(pmid)))
        # print(file_name)
        pdf.save(file_name)
        return {'status': 'success'}
    except Exception as e:
        return {'failure': 'An error occurred: {}'.format(e)}


@api.route('/fulltext/download/<pmid>')
async def downloadPdf(pmid: str):
    try: 
        # Expect the URL from the query parameters
        url = request.args.get('pdfUrl')
        utils.download_pdf_paper(url, pmid, PDF_PAPERS_FOLDER)
        return {'status': 'success'}
    except Exception as e:
        return {'failure': 'An error occurred: {}'.format(e)}

@api.route('/annotate', methods=['POST'])
async def annotate_gene():
    data = request.get_json()

    # Extract gene and parameters from the JSON request
    gene = data.get('queryGene')
    top_pubmed_results = data.get('numberOfPubmed', 8)
    max_query_length = data.get('max_query_length', 1000)  # Don't change this for now
    pathway_count = data.get('numberOfPathways', 8)
    pathway_abstract_similiary_cutoff = data.get('cosineSimilarityCutoff', 0.38)
    llm_score_cutoff = data.get('llmScoreCutoff', 3)
    fi_cutoff = data.get('fiScoreCutoff', 0.8)
    fdr_cutoff = data.get('fdrCutoff', 0.05)
    # This is not set by the front-end and determined by the local pubmed abstracts latest date
    pubmed_maxdate = '2024/12/31'

    try:
        annotated_pathway_summary = await annotator.write_summary_of_annotated_pathways(gene)
        pubmed_results = await annotator.query_pubmed_abstracts_for_gene(gene, 
                                                                         pubmed_maxdate=pubmed_maxdate,
                                                                         top_k_results=top_pubmed_results,
                                                                         max_query_length=max_query_length)
        abstract_result_for_pathways, pathway_abstract_df = await annotator.write_summary_for_gene_annotation(gene,
                                                                                                              pubmed_results=pubmed_results,
                                                                                                              fi_cutoff=fi_cutoff,
                                                                                                              fdr_cutoff=fdr_cutoff,
                                                                                                              pathway_count=pathway_count,
                                                                                                              pathway_abstract_similiary=pathway_abstract_similiary_cutoff,
                                                                                                              llm_score=llm_score_cutoff,
                                                                                                              model=model)
        # Get summary for literature supporting PPIs
        pathway2ppi_summary = dict()
        for _, row in pathway_abstract_df.iterrows():
            pathway = row['pathway']
            # Escape if it has been checked since the interactions are
            # collected for genes
            if pathway in pathway2ppi_summary.keys():
                continue
            pmids = row['ppi_genes_pmids']
            ppi_genes = row['ppi_genes']
            abstract_result = {}
            abstract_result['ppi_genes'] = ppi_genes
            abstract_result['pmids'] = pmids
            # Parse the pmids
            # Make sure it is not duplicated
            pmid_set = set([pmid for pmids_1 in pmids for pmid in pmids_1.split('|')])
            # If the total abstracts are less than 8, no score is provided since all abstracts will be analyzed
            llm_result, abstract_df = await annotator.summarize_pubmed_abstracts_for_interactions(annotate_gene,
                                                                                                pathway,
                                                                                                ppi_genes,
                                                                                                pmid_set,
                                                                                                # Use the same top PPIs abstracts as pubmed for the time being.
                                                                                                top_abstracts=top_pubmed_results)
            abstract_result['summary'] = llm_result['answer'].content
            pathway2ppi_summary[pathway] = abstract_result
    except (NoAbstractFoundError, NoInteractingPathwayFoundError, NoAbstractSupportingInteractingPathwayError) as e:
        log.error('error: {}'.format(e.message))
        return {'failure': e.message}
    # Add mapping from pathway to dbId for the front so that a link can be created
    pattern = re.compile(r'PATHWAY_NAME:"([^"]+)"')
    pathway_names = re.findall(pattern, abstract_result_for_pathways['docs'])
    name2id = neo4jutils.map_pathway_name_to_dbId(pathway_names)

    return_json = {
        'content': abstract_result_for_pathways['answer'].content,
        'docs':  abstract_result_for_pathways['docs'],
        'pathway_2_ppi_abstracts_summary': pathway2ppi_summary,
        'pathway_name_2_id': name2id
    }
    if annotated_pathway_summary:
        return_json['annotated_pathways_content'] = annotated_pathway_summary['answer'].content
        return_json['annotated_pathways_docs'] = annotated_pathway_summary['docs']
        # Get the names
        annotated_pathway_names = _collect_pathway_names(annotated_pathway_summary['docs'])
        annoated_name2id = neo4jutils.map_pathway_name_to_dbId(annotated_pathway_names)
        name2id.update(annoated_name2id)
    return return_json


def _collect_pathway_names(docs: str) -> list[str]:
    names = []
    for paragraph in docs.split('\n'):
        names.append(paragraph.split(':')[0].strip())
    return names


# Run the api at the terminal with this command: flask --app reactome_llm/ReactomeLLMRestAPI run (--debug for debug)
if __name__ == '__main__':
    api.run()
