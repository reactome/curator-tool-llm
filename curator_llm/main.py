"""Production wiring. Run:  uvicorn curator_llm.main:app --port 8000

Environment (.env is loaded): WS_BASE_URL (default http://localhost:9090), SESSION_STORE=memory|mongo
(default memory), CHAT_MODEL (default claude-sonnet-5-5), CORS_ORIGINS (default http://localhost:4200), UPLOAD_DIR (default data/uploads), LLM_REVIEW=1 to also run the OpenAI cross-model review, plus the existing
ANTHROPIC_API_KEY, REACTOME_NEO4J_*, PUBMED_MONGO_URI."""
import os

from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(ROOT, '.env'), override=True)


def build_app():
    from curator_llm.adapters.chat_anthropic import AnthropicChatModel
    from curator_llm.adapters.jobs_inprocess import InProcessJobRunner
    from curator_llm.adapters.neo4j_lookup import Neo4jInstanceLookup
    from curator_llm.adapters.ols import OlsClient
    from curator_llm.adapters.sessions_memory import InMemorySessionStore
    from curator_llm.adapters.uniprot import RestUniProtClient
    from curator_llm.adapters.ws_auth import WsAuthProvider
    from curator_llm.api.app import create_app
    from curator_llm.services.pipeline_default import DefaultPipeline

    if os.getenv('SESSION_STORE', 'memory') == 'mongo':
        from curator_llm.adapters.sessions_mongo import MongoSessionStore
        store = MongoSessionStore.from_env()
    else:
        store = InMemorySessionStore()
    lookup0 = Neo4jInstanceLookup.from_env()
    pipeline = DefaultPipeline(lookup0, RestUniProtClient(), OlsClient(),
                               review=os.getenv('LLM_REVIEW') == '1', events=lookup0)
    lookup = pipeline.lookup
    uniprot, ols = pipeline.uniprot, pipeline.ontology
    from curator_llm.services.resolvers import Resolver
    origins = [o.strip() for o in os.getenv('CORS_ORIGINS', 'http://localhost:4200').split(',') if o.strip()]
    return create_app(WsAuthProvider(os.getenv('WS_BASE_URL', 'http://localhost:9090')),
                      store, InProcessJobRunner(), pipeline,
                      resolver_factory=lambda: Resolver(lookup, uniprot, ols), events=lookup,
                      upload_dir=os.getenv('UPLOAD_DIR', os.path.join(ROOT, 'data', 'uploads')),
                      cors_origins=origins, chat_model=AnthropicChatModel())


app = build_app() if os.getenv('CURATOR_LLM_AUTOSTART', '1') == '1' else None
