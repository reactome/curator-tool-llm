import warnings
warnings.filterwarnings('ignore')

from flask import Flask
from flask_cors import CORS
from langchain_openai import ChatOpenAI

import re

import ReactomeUtils as utils
import ReactomeNeo4jUtils as neo4jutils

# Shared variables for all
api = Flask(__name__)
CORS(api)
model = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

# This is for test
# return_json = None


# TODO: Add cellular locations and cell types reuqest into the prompts!
@api.route('/query/<gene>')
async def query_gene(gene):
    # global return_json
    # if return_json:
    #     return return_json
    pubmed_db = await utils.build_abstract_vector_db_for_gene(gene, top_k_results=10)
    query_result = await utils.write_summary_of_abstracts_for_gene(gene,
                                                                   pubmed_db,
                                                                   model)
    # Add mapping from pathway to dbId for the front so that a link can be created
    pattern = re.compile(r'PATHWAY_NAME:"([^"]+)"')
    pathway_names = re.findall(pattern, query_result['docs'])
    name2id = neo4jutils.map_pathway_name_to_dbId(pathway_names)

    return_json = {
        'content': query_result['answer'].content,
        'docs':  query_result['docs'],
        'pathway_name_2_id': name2id
    }
    return return_json


if __name__ == '__main__':
    api.run()

