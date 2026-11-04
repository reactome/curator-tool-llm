"""Production wiring shared by the REST service (main.py) and the command-line runner (cli.py): the settings in .env, and
the pipeline built from the real adapters (Neo4j, UniProt, OLS, the models) with saved results (snapshots) around it."""
import os
from typing import Tuple

from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_env() -> None:
    """Read .env; a value in the file wins over the shell's."""
    load_dotenv(os.path.join(ROOT, '.env'), override=True)


def build_pipeline() -> Tuple[object, object]:
    """(pipeline to run, the DefaultPipeline inside it). The first decides whether each paper is a review (PubMed for a
    PMID), records each successful run and, with
    SNAPSHOT_MODE=replay, serves a saved result instead of calling the models; the second holds the lookup, UniProt and
    ontology clients, which the service reuses to resolve what an edit introduces."""
    from curator_llm.adapters.neo4j_lookup import Neo4jInstanceLookup
    from curator_llm.adapters.ols import OlsClient
    from curator_llm.adapters.pubmed import PubMedClient
    from curator_llm.adapters.uniprot import RestUniProtClient
    from curator_llm.services.paper_kind import KindPipeline
    from curator_llm.services.pipeline_default import DefaultPipeline
    from curator_llm.services.snapshots import SnapshotPipeline, SnapshotStore

    lookup = Neo4jInstanceLookup.from_env()
    inner = DefaultPipeline(lookup, RestUniProtClient(), OlsClient(), review=os.getenv('LLM_REVIEW') == '1', events=lookup)
    store = SnapshotStore(os.getenv('SNAPSHOT_DIR', os.path.join(ROOT, 'data', 'snapshots')))
    # review or research is decided first: the saved-result layer keys on it
    return KindPipeline(SnapshotPipeline(inner, store, os.getenv('SNAPSHOT_MODE', 'record')), PubMedClient()), inner
