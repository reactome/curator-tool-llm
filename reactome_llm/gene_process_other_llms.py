'''
    gene_process_other_llms.py works similar to gene_process.py, but tries to use various LLM models.
    It is designed to support both OpenAI, Hugging Face models.
    The current implementation for Hugging Face models is not fully operational.
'''

import os
from pathlib import Path
import re
import logging as log
import dotenv
import asyncio
import requests

from transformers import AutoModelForCausalLM, AutoTokenizer
import ReactomeNeo4jUtils as neo4jutils
import ReactomeUtils as utils
from ReactomeLLMErrors import NoAbstractFoundError, NoInteractingPathwayFoundError

# Load environment variables 
dotenv.load_dotenv()
PDF_PAPERS_FOLDER = os.getenv('PDF_PAPERS_FOLDER')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN') 

# Configure the AI model
async def get_model(model_name, temperature=0):
    # Hugging Face models
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
    return model, tokenizer


# Analyze full-text paper
async def analyze_full_text(pmid, gene, model):
    file_path = Path(PDF_PAPERS_FOLDER, f'{pmid}.pdf')
    if not file_path.exists():
        return {'failure': 'The PDF full text paper cannot be found at the server'}

    results = await utils.analyze_full_paper(str(file_path), gene, model, top_pages=4)
    return [
        {'content': result['answer'].content, 'docs': result['docs']}
        for result in results
    ]


# Check if PDF exists
async def pdf_exists(pmid):
    file_path = Path(PDF_PAPERS_FOLDER, f'{pmid}.pdf')
    return file_path.exists()


# Upload PDF
async def upload_pdf(pdf, pmid):
    file_name = Path(PDF_PAPERS_FOLDER, f'{pmid}.pdf')
    try:
        pdf.save(file_name)
        return {'status': 'success'}
    except Exception as e:
        return {'failure': f'An error occurred: {e}'}


# Download PDF
async def download_pdf(url, pmid):
    try:
        await utils.download_pdf_paper(url, pmid, PDF_PAPERS_FOLDER)
        return {'status': 'success'}
    except Exception as e:
        return {'failure': f'An error occurred: {e}'}


# Query gene information
async def query_gene(gene, model):
    try:
        annotated_pathway_summary = await utils.write_summary_of_annotated_pathways(gene, model)
        pubmed_db = await utils.build_abstract_vector_db_for_gene(gene, top_k_results=12)
        query_result = await utils.write_summary_of_abstracts_for_gene(gene, pubmed_db, model)
    except (NoAbstractFoundError, NoInteractingPathwayFoundError) as e:
        log.error(f'error: {e.message}')
        return {'failure': e.message}

    pattern = re.compile(r'PATHWAY_NAME:"([^"]+)"')
    pathway_names = re.findall(pattern, query_result['docs'])
    name2id = neo4jutils.map_pathway_name_to_dbId(pathway_names)

    return_json = {
        'content': query_result['answer'].content,
        'docs': query_result['docs'],
        'pathway_name_2_id': name2id
    }
    if annotated_pathway_summary:
        return_json['annotated_pathways_content'] = annotated_pathway_summary['answer'].content
        return_json['annotated_pathways_docs'] = annotated_pathway_summary['docs']
        annotated_pathway_names = await _collect_pathway_names(annotated_pathway_summary['docs'])
        annotated_name2id = neo4jutils.map_pathway_name_to_dbId(annotated_pathway_names)
        name2id.update(annotated_name2id)

    return return_json


# Collect pathway names from the document
async def _collect_pathway_names(docs):
    names = []
    for paragraph in docs.split('\n'):
        names.append(paragraph.split(':')[0].strip())
    return names


# Save result to a file
async def save_result(gene, result, model_name):
    file_path = 'gene_result_3b.txt'
    with open(file_path, 'a') as f:
        f.write(f"LLM: {model_name}\n")
        f.write(f"Gene: {gene}\n")
        f.write("Result:\n")
        f.write(str(result) + "\n\n")
    print(f'Result saved to {file_path}')


# Main function 
async def main():
    model_name = "aaditya/Llama3-OpenBioLLM-70B"  
    model, tokenizer = await get_model(model_name, temperature=0)

    # List of genes to process
    genes = ['TANC1', 'FADD', 'NTN1', 'ICAM1', 'RACK1', 'ABCC1', 'SPIDR', 'XRCC6', 'MPDZ', 'TNFRSF14']

    # Process each gene
    for gene in genes:
        print(f'Processing gene: {gene}')
        result = await query_gene(gene, model)
        print(result)
        await save_result(gene, result, model_name)

if __name__ == '__main__':
    asyncio.run(main())
