"""Production wiring. Run:  uvicorn curator_llm.main:app --port 8000

Settings (.env is loaded):
  WS_BASE_URL      curator-tool-ws, which checks every token (default http://localhost:9090)
  SESSION_STORE    memory (default) or mongo
  SNAPSHOT_MODE    record (default: save each successful run), replay (reuse a saved result instead of calling the
                   models; for interface development and demos) or off. SNAPSHOT_DIR defaults to data/snapshots.
  LLM_MODEL_<ROLE> which model a step uses (all defaults are in curator_llm/model_settings.py; CHAT_MODEL still works)
  CORS_ORIGINS     default http://localhost:4200
  UPLOAD_DIR       default data/uploads
  MAX_PAPERS       most papers in one session (default 10)
  PAPER_WORKERS    papers extracted at the same time within a session (default 2)
  LLM_REVIEW=1     also run the OpenAI cross-model review
plus the existing ANTHROPIC_API_KEY, REACTOME_NEO4J_* and PUBMED_MONGO_URI.
"""
import os

from curator_llm.wiring import ROOT, build_pipeline, load_env

load_env()


def build_app():
    from curator_llm.adapters.chat_anthropic import AnthropicChatModel
    from curator_llm.adapters.jobs_inprocess import InProcessJobRunner
    from curator_llm.adapters.sessions_memory import InMemorySessionStore
    from curator_llm.adapters.ws_auth import WsAuthProvider
    from curator_llm.api.app import create_app
    from curator_llm.services.resolvers import Resolver

    if os.getenv('SESSION_STORE', 'memory') == 'mongo':
        from curator_llm.adapters.sessions_mongo import MongoSessionStore
        store = MongoSessionStore.from_env()
    else:
        store = InMemorySessionStore()
    pipeline, inner = build_pipeline()
    lookup, uniprot, ols = inner.lookup, inner.uniprot, inner.ontology
    origins = [o.strip() for o in os.getenv('CORS_ORIGINS', 'http://localhost:4200').split(',') if o.strip()]
    return create_app(WsAuthProvider(os.getenv('WS_BASE_URL', 'http://localhost:9090')),
                      store, InProcessJobRunner(), pipeline,
                      resolver_factory=lambda: Resolver(lookup, uniprot, ols), events=lookup,
                      upload_dir=os.getenv('UPLOAD_DIR', os.path.join(ROOT, 'data', 'uploads')),
                      cors_origins=origins, chat_model=AnthropicChatModel(),
                      qa_model=AnthropicChatModel(role='qa'), max_papers=int(os.getenv('MAX_PAPERS', '10')))


app = build_app() if os.getenv('CURATOR_LLM_AUTOSTART', '1') == '1' else None
