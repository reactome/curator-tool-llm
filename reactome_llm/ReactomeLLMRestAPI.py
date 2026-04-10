import asyncio
import logging as log
import os
import threading
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import ReactomeNeo4jUtils as neo4jutils
import ReactomeUtils as utils
import GenePathwayAnnotator as gpa

import re
from flask_cors import CORS
from flask import Flask, request, jsonify
import warnings

from ReactomeLLMErrors import NoAbstractFoundError, NoAbstractSupportingInteractingPathwayError, NoInteractingPathwayFoundError, NoProteinInteractionFoundError
from CrewAIEventLogger import clear_event_sink, set_event_sink
from CrewAILiteratureAnnotator import CrewAILiteratureAnnotator, AnnotationRequest
from ModelConfig import create_reactome_chat_model, get_crewai_model_settings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()
PDF_PAPERS_FOLDER = os.getenv('PDF_PAPERS_FOLDER')

# Shared variables for all
api = Flask(__name__)
CORS(api)
# Configure base model using environment (REACTOME_LLM_MODEL / REACTOME_LLM_TEMPERATURE).
model = create_reactome_chat_model()

# This is for test
return_json = None
full_text_return_json = None

# The following code is used to set up an instance of GenePathwayAnnotator.
annotator = gpa.GenePathwayAnnotator()
annotator.set_model(model)

# Initialize CrewAI annotator
crewai_model, crewai_temperature = get_crewai_model_settings()
crewai_annotator = CrewAILiteratureAnnotator(
    annotator,
    model=crewai_model,
    temperature=crewai_temperature,
    verbose=True,
)

# Test genes: TANC1 (not annotated), DUX4L2 (not annotated, limited abstracts), NTN1 (annotated)
# TODO: Add cellular locations and cell types reuqest into the prompts!
# TODO: This is important. Add GUIs to configure the cutoff of FIs and pathway FDRs and also make sure they are 
# consistent across the whole application.

# ---------------------------------------------------------------------------
# In-memory job store for long-running CrewAI annotation requests.
# Maps job_id (str) -> dict with keys: status, result, error
# ---------------------------------------------------------------------------
_crewai_jobs: dict = {}
_crewai_jobs_lock = threading.Lock()
_crewai_job_logs: dict = {}
_crewai_job_logs_lock = threading.Lock()

_CREWAI_DASHBOARD_CONFIG = {
    'phases': [
        {'id': 'phase_1_literature_extraction', 'label': 'Phase 1: Literature Extraction'},
        {'id': 'phase_2_data_model_creation', 'label': 'Phase 2: Data Model Creation'},
        {'id': 'phase_3_expert_review', 'label': 'Phase 3: Expert Review'},
        {'id': 'phase_4_quality_assurance', 'label': 'Phase 4: Quality Assurance'},
        {'id': 'phase_5_final_consensus', 'label': 'Phase 5: Final Consensus'},
    ],
    'agents': [
        {'id': 'extractor', 'label': 'LiteratureExtractor', 'phases': ['phase_1_literature_extraction']},
        {'id': 'curator', 'label': 'ReactomeCurator', 'phases': ['phase_2_data_model_creation']},
        {'id': 'reviewer', 'label': 'Reviewer', 'phases': ['phase_3_expert_review', 'phase_5_final_consensus']},
        {'id': 'qa_checker', 'label': 'QualityChecker', 'phases': ['phase_4_quality_assurance']},
    ],
    'toolsByAgent': {
        'extractor': [
            {'id': 'literature_search', 'label': 'literature_search'},
            {'id': 'fulltext_analysis', 'label': 'fulltext_analysis'},
            {'id': 'protein_interactions', 'label': 'protein_interactions'},
            {'id': 'evidence_evaluation', 'label': 'evidence_evaluation'},
        ],
        'curator': [
            {'id': 'reactome_query', 'label': 'reactome_query'},
            {'id': 'schema_validation', 'label': 'schema_validation'},
            {'id': 'protein_interactions', 'label': 'protein_interactions'},
            {'id': 'evidence_evaluation', 'label': 'evidence_evaluation'},
        ],
        'reviewer': [
            {'id': 'literature_search', 'label': 'literature_search'},
            {'id': 'reactome_query', 'label': 'reactome_query'},
            {'id': 'evidence_evaluation', 'label': 'evidence_evaluation'},
            {'id': 'quality_metrics', 'label': 'quality_metrics'},
            {'id': 'consistency_check', 'label': 'consistency_check'},
        ],
        'qa_checker': [
            {'id': 'schema_validation', 'label': 'schema_validation'},
            {'id': 'consistency_check', 'label': 'consistency_check'},
            {'id': 'quality_metrics', 'label': 'quality_metrics'},
            {'id': 'reactome_query', 'label': 'reactome_query'},
        ],
    }
}


def _append_job_event(job_id: str, event: dict):
    if not job_id:
        return
    with _crewai_job_logs_lock:
        logs = _crewai_job_logs.setdefault(job_id, [])
        logs.append({
            'seq': len(logs),
            'ts': datetime.now(timezone.utc).isoformat(),
            **event,
        })


def _make_job_event_sink(job_id: str):
    def _sink(event: dict):
        _append_job_event(job_id, event)
    return _sink

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
    results = annotator.analyze_full_paper(str(file_path), gene, model, top_pages=4) # Use 4 for test
    
    for result in results:
        result_json = {
            'content': result['answer'].content,
            'docs':  result['docs'],
        }
        full_text_return_json.append(result_json)
    return full_text_return_json


@api.route('/fulltext/list')
def listPdfs():
    """Return a list of all PMIDs that have a PDF available in PDF_PAPERS_FOLDER."""
    if not PDF_PAPERS_FOLDER:
        return jsonify({'papers': [], 'error': 'PDF_PAPERS_FOLDER not configured'}), 500
    folder = Path(PDF_PAPERS_FOLDER)
    if not folder.is_dir():
        return jsonify({'papers': [], 'error': f'Folder not found: {PDF_PAPERS_FOLDER}'}), 500
    papers = [
        {
            'pmid': p.stem,
            'exists': True,
            'size_bytes': p.stat().st_size,
        }
        for p in sorted(folder.glob('*.pdf'))
        if p.stem.isdigit()
    ]
    return jsonify({'papers': papers})


@api.route('/fulltext/check_pdf/<pmid>')
async def pdfExists(pmid: str) -> bool:
    file_path = Path(PDF_PAPERS_FOLDER, '{}.pdf'.format(pmid))
    return True if file_path.exists() else False


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
    # This is for test
    # load the test json without querying LLMs etc
    # with open('test/NTN1.json', 'r') as f:
    #     data = f.read()
    #     return data

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
    interactionSource = data.get('interactionSource', 'intact_biogrid')
    filterPPIs = data.get('filterPPIs', True)
    # This is not set by the front-end and determined by the local pubmed abstracts latest date
    pubmed_maxdate = '2024/12/31'

    try:
        annotated_pathway_summary = await annotator.write_summary_of_annotated_pathways(gene)
        pubmed_results = await annotator.query_pubmed_abstracts_for_gene(gene, 
                                                                         pubmed_maxdate=pubmed_maxdate,
                                                                         top_k_results=top_pubmed_results,
                                                                         max_query_length=max_query_length)
        abstract_result_for_pathways, pathway_abstract_df = await annotator.write_summary_for_gene_annotation(gene,
                                                                                                              interactionSource=interactionSource,
                                                                                                              filterPPIs=filterPPIs,
                                                                                                              pubmed_results=pubmed_results,
                                                                                                              fi_cutoff=fi_cutoff,
                                                                                                              fdr_cutoff=fdr_cutoff,
                                                                                                              pathway_count=pathway_count,
                                                                                                              pathway_abstract_similiary=pathway_abstract_similiary_cutoff,
                                                                                                              llm_score=llm_score_cutoff,
                                                                                                              model=model)
        pathway_abstract_scores = pathway_abstract_df[['pathway', 'pmid', 'cos_score', 'llm_score', 'enrichment_fdr']]
        # Get summary for literature supporting PPIs
        pathway2ppi_summary = dict()
        # reactome_fis doesn't have pmids
        if interactionSource == 'intact_biogrid':
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
                llm_result, abstract_df = await annotator.summarize_pubmed_abstracts_for_interactions(gene,
                                                                                                    pathway,
                                                                                                    ppi_genes,
                                                                                                    pmid_set,
                                                                                                    # Use the same top PPIs abstracts as pubmed for the time being.
                                                                                                    top_abstracts=top_pubmed_results)
                abstract_result['summary'] = llm_result['answer'].content
                pathway2ppi_summary[pathway] = abstract_result
    except (NoAbstractFoundError, NoProteinInteractionFoundError, NoInteractingPathwayFoundError, NoAbstractSupportingInteractingPathwayError) as e:
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
        'pathway_name_2_id': name2id,
        'pathway_abstract_scores': pathway_abstract_scores.to_dict(orient='records'),
    }
    if annotated_pathway_summary:
        return_json['annotated_pathways_content'] = annotated_pathway_summary['answer'].content
        return_json['annotated_pathways_docs'] = annotated_pathway_summary['docs']
        # Get the names
        annotated_pathway_names = _collect_pathway_names(annotated_pathway_summary['docs'])
        annotated_name2id = neo4jutils.map_pathway_name_to_dbId(annotated_pathway_names)
        name2id.update(annotated_name2id)
    return return_json


def _collect_pathway_names(docs: str) -> list[str]:
    names = []
    for paragraph in docs.split('\n'):
        names.append(paragraph.split(':')[0].strip())
    return names


@api.route('/crewai/annotate', methods=['POST'])
def crewai_annotate_gene():
    """
    Submit a multi-agent CrewAI annotation job.

    Request body (JSON):
        queryGene          (str)        – optional; if omitted, framework runs in gene-agnostic mode
        pmids              (list[str])  – optional list of PMIDs whose PDFs live in data/papers/<pmid>.pdf
        papers             (list[str])  – optional alias for pmids
        numberOfPubmed     (int)        – max papers, default 8
        targetPathways     (list[str])  – optional pathway focus
        qualityThreshold   (float)      – default 0.7
        enableFullText     (bool)       – default true
        enableLiteratureSearch (bool)   – default false; set true to search PubMed instead of using pmids
        schemaPath         (str)        – default 'resources/reactome_domain_model.json'

    Returns:
        {"job_id": "<uuid>", "status": "running"}
    """
    data = request.get_json(force=True)
    gene = (data.get('queryGene') or '').strip() or None
    papers = data.get('pmids') or data.get('papers') or []
    if not isinstance(papers, list):
        return jsonify({'error': 'pmids/papers must be a list'}), 400
    if len(papers) == 0:
        return jsonify({'error': 'pmids or papers is required and must contain at least one item'}), 400
    if data.get('enableLiteratureSearch', False) and not gene:
        return jsonify({'error': 'queryGene is required when enableLiteratureSearch is true'}), 400

    job_id = str(uuid.uuid4())
    with _crewai_jobs_lock:
        _crewai_jobs[job_id] = {'status': 'running', 'gene': gene or 'UNSPECIFIED_GENE'}
    _append_job_event(job_id, {
        'event_type': 'job',
        'status': 'queued',
        'gene': gene or 'UNSPECIFIED_GENE',
        'papers_count': len(papers),
    })

    def _run():
        set_event_sink(_make_job_event_sink(job_id), job_id=job_id)
        try:
            _append_job_event(job_id, {
                'event_type': 'job',
                'status': 'start',
                'gene': gene or 'UNSPECIFIED_GENE',
                'papers_count': len(papers),
            })
            req = AnnotationRequest(
                gene=gene,
                papers=papers,
                pathways=data.get('targetPathways'),
                max_papers=data.get('numberOfPubmed', 8),
                quality_threshold=data.get('qualityThreshold', 0.7),
                enable_full_text=data.get('enableFullText', True),
                enable_literature_search=data.get('enableLiteratureSearch', False),
                enabled_phases=data.get('enabledPhases'),
                enabled_agents=data.get('enabledAgents'),
                enabled_tools=data.get('enabledTools'),
                schema_path=data.get('schemaPath', 'resources/reactome_domain_model.json'),
            )
            result = asyncio.run(crewai_annotator.annotate_literature(req))
            with _crewai_jobs_lock:
                _crewai_jobs[job_id] = {
                    'status': 'done',
                    'gene': result.gene,
                    'result': asdict(result),
                }
            _append_job_event(job_id, {
                'event_type': 'job',
                'status': 'end',
                'gene': result.gene,
                'outcome': 'success',
            })
        except Exception as exc:
            log.error(f'CrewAI job {job_id} failed: {exc}')
            with _crewai_jobs_lock:
                _crewai_jobs[job_id] = {
                    'status': 'error',
                    'gene': gene or 'UNSPECIFIED_GENE',
                    'error': str(exc),
                }
            _append_job_event(job_id, {
                'event_type': 'job',
                'status': 'end',
                'gene': gene or 'UNSPECIFIED_GENE',
                'outcome': 'error',
                'error': str(exc),
            })
        finally:
            clear_event_sink()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'job_id': job_id, 'status': 'running'})


@api.route('/crewai/dashboard')
def crewai_dashboard():
    """Get configurable phases/agents/tools for the Paper2Path agent dashboard."""
    return jsonify(_CREWAI_DASHBOARD_CONFIG)


@api.route('/crewai/result/<job_id>')
def crewai_result(job_id: str):
    """
    Poll for the result of a submitted CrewAI annotation job.

    Returns one of:
        {"status": "running", "gene": "..."}
        {"status": "done",    "gene": "...", "result": {...AnnotationResult fields...}}
        {"status": "error",   "gene": "...", "error": "..."}
        {"status": "not_found"}
    """
    with _crewai_jobs_lock:
        job = _crewai_jobs.get(job_id)
    if job is None:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(job)


@api.route('/crewai/logs/<job_id>')
def crewai_logs(job_id: str):
    """
        Fetch incremental structured runtime events for a submitted CrewAI job.

    Query params:
      since (int, default 0): Return logs with seq >= since.
      limit (int, default 200): Max logs returned in one response.

    Returns:
      {
        "status": "running|done|error",
        "gene": "...",
                "logs": [
                    {"seq": 0, "ts": "...", "event_type": "job", "status": "start", "gene": "..."},
                    {"seq": 1, "ts": "...", "event_type": "agent", "status": "start", "agent": "LiteratureExtractor", "phase": "phase_1_literature_extraction", "gene": "..."},
                    {"seq": 2, "ts": "...", "event_type": "tool", "status": "start", "tool": "fulltext_analysis"},
                    {"seq": 3, "ts": "...", "event_type": "tool", "status": "end", "tool": "fulltext_analysis"}
                ],
        "next_since": 12
      }
    """
    since_raw = request.args.get('since', '0')
    limit_raw = request.args.get('limit', '200')
    try:
        since = max(0, int(since_raw))
    except ValueError:
        since = 0
    try:
        limit = min(1000, max(1, int(limit_raw)))
    except ValueError:
        limit = 200

    with _crewai_jobs_lock:
        job = _crewai_jobs.get(job_id)
    if job is None:
        return jsonify({'status': 'not_found', 'logs': [], 'next_since': since}), 404

    with _crewai_job_logs_lock:
        logs = _crewai_job_logs.get(job_id, [])
        selected_logs = logs[since:since + limit]
        next_since = since + len(selected_logs)

    return jsonify({
        'status': job.get('status', 'running'),
        'gene': job.get('gene'),
        'logs': selected_logs,
        'next_since': next_since,
    })


@api.route('/crewai/status')
def crewai_status():
    """Get status of CrewAI annotation system and queued jobs."""
    with _crewai_jobs_lock:
        running = sum(1 for j in _crewai_jobs.values() if j['status'] == 'running')
        done = sum(1 for j in _crewai_jobs.values() if j['status'] == 'done')
        errors = sum(1 for j in _crewai_jobs.values() if j['status'] == 'error')
    return jsonify({
        'status': 'active',
        'model': crewai_annotator.model,
        'temperature': crewai_annotator.temperature,
        'max_iterations': crewai_annotator.max_iter,
        'agents': ['ReactomeCurator', 'LiteratureExtractor', 'Reviewer', 'QualityChecker'],
        'jobs': {'running': running, 'done': done, 'error': errors},
    })


# Run the api at the terminal with this command: flask --app reactome_llm/ReactomeLLMRestAPI run (--debug for debug)
# The conda environment should be paperqa (G.W.)
if __name__ == '__main__':
    api.run()
