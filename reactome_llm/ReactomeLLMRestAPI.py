from crypt import methods
import logging as log
import os
from pathlib import Path

import dotenv
from exceptiongroup import catch
import ReactomeNeo4jUtils as neo4jutils
import ReactomeUtils as utils
import re
from langchain_openai import ChatOpenAI
from flask_cors import CORS
from flask import Flask, request
import warnings

from ReactomeLLMErrors import NoAbstractFoundError, NoInteractingPathwayFoundError
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()
PDF_PAPERS_FOLDER = os.getenv('PDF_PAPERS_FOLDER')

# Shared variables for all
api = Flask(__name__)
CORS(api)
model = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

# This is for test
return_json = None
full_text_return_json = None

# Test genes: TANC1 (not annotated), DUX4L2 (not annotated, limited abstracts), NTN1 (annotated)
# TODO: Add cellular locations and cell types reuqest into the prompts!

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


@api.route('/query/<gene>')
async def query_gene(gene):
    # global return_json
    # if return_json:
    #     return return_json
    try:
        annotated_pathway_summary = await utils.write_summary_of_annotated_pathways(gene, 
                                                                                    model)
        pubmed_db = await utils.build_abstract_vector_db_for_gene(gene, top_k_results=12)
        query_result = await utils.write_summary_of_abstracts_for_gene(gene,
                                                                       pubmed_db,
                                                                       model)
    # TODO: Make sure anotated_pathway_summary can be returned if it is there.
    except (NoAbstractFoundError, NoInteractingPathwayFoundError) as e:
        log.error('error: {}'.format(e.message))
        return {'failure': e.message}
    # Add mapping from pathway to dbId for the front so that a link can be created
    pattern = re.compile(r'PATHWAY_NAME:"([^"]+)"')
    pathway_names = re.findall(pattern, query_result['docs'])
    name2id = neo4jutils.map_pathway_name_to_dbId(pathway_names)

    return_json = {
        'content': query_result['answer'].content,
        'docs':  query_result['docs'],
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


# Run the api at the terminal with this command: flask --app ReactomeLLMRestAPI run (--debug for debug)
if __name__ == '__main__':
    api.run()
