# CLAUDE.md

Guidance for working in this repository. See [README.md](README.md) for full user docs and
[docs/implementation-plan.md](docs/implementation-plan.md) for design decisions and status.

## What this repo is

A Reactome curator-assist tool that uses LLMs to draft annotations from literature. It has **two separate entry points** that share an install and `.env`:

1. **Gene-first CLI pipeline** (legacy, flat scripts + `reactome_llm/`): `python run_curator.py GENE` → placement → PubMed retrieval → full-text reaction extraction → QA. Output in `results/`.
2. **Paper-level REST service** (`curator_llm/`, FastAPI): annotates ONE paper (PMID or uploaded PDF), verifies quotes, resolves identifiers, emits staged instances, lists issues, and supports chat-based edits as *proposals*. Used by curator-tool-frontend (Angular, "Paper2Path" page). Most active development is here.

Python 3.10, conda env `reactome_ai`. Needs Neo4j (Reactome graph, `:7687`), MongoDB (`:27017`), `ANTHROPIC_API_KEY` (OpenAI key only for the optional cross-model review). Config is in `.env` (copy `.env.example`; loaded with `override=True`, so active values beat the shell).

## Commands

```bash
pip install -r requirements.txt            # authoritative dependency list
pytest                                      # tests/ (pytest.ini); all external services are faked
pytest tests/test_patching.py -k name       # single test
scripts/run_api.sh [--reload|--daemon|--stop]   # run REST API (uvicorn curator_llm.main:app, :8000, docs at /docs)
python scripts/export_openapi.py            # regenerate docs/openapi.json
python scripts/make_snapshot.py             # build a snapshot for replay mode
python run_curator.py SHANK3 --no-full-text --no-llm-review --no-llm-qa   # cheap CLI smoke test
```

`tests/test_ws_contract.py` needs a real curator-tool-ws plus `WS_USERNAME`/`WS_PASSWORD`; it is not part of normal offline runs.

## REST service architecture (`curator_llm/`) — ports & adapters

- `main.py` — production wiring (`build_app()`; `app` is created at import unless `CURATOR_LLM_AUTOSTART=0`). Reads env: `WS_BASE_URL`, `SESSION_STORE=memory|mongo`, `SNAPSHOT_MODE=record|replay|off`, `SNAPSHOT_DIR`, `CHAT_MODEL`, `CORS_ORIGINS`, `UPLOAD_DIR`, `LLM_REVIEW`.
- `api/app.py` — `create_app(...)` takes all dependencies by injection; routes under `/api/llm`: sessions (create / upload / get / issues / usage / export / network / existing / qa / paper search), evidence per instance, proposals (create / accept / reject), chat (SSE stream), jobs. Every request is authenticated and requires the `curator` role (401 / 403); sessions are private to their owner (others get 404).
- `ports/` — interfaces (`AuthProvider`, `InstanceLookup`, `JobRunner`, `Pipeline`, `Extractor`, `ChatModel`, `SessionStore`, external UniProt/OLS clients, events). 
- `adapters/` — production implementations: ws token verification (`ws_auth`, calls curator-tool-ws `GET /api/auth/verify`), Neo4j lookup, UniProt, OLS, Anthropic chat, in-process job runner, memory/Mongo session stores.
- `models/` — pydantic: `reactome.py` (typed `ReactomeDraft`), `evidence.py`, `session.py`, `usage.py`.
- `services/` — the logic:
  - Pipeline: `pipeline_default` (paper → extract/merge/review → verified evidence → typed draft → resolved ids), `extraction` (wraps the legacy scripts), `paper_loader`/`paper_text` (PDF via PyMuPDF or PMC JATS XML, offset-tracked), `quote_verifier` (deterministic, normalised matching), `evidence_attach`/`evidence_store`, `draft_builder` (one structured LLM call, then code), `resolvers` (names → dbIds; **the LLM never supplies a dbId**), `emitter` (draft → frontend `UserInstances` JSON; new instances get negative dbIds), `existing_events` (does Reactome already have this reaction), `qa`, `issues`, `network`, `schema_index`.
  - Editing: `patching` (RFC 6902 JSON Patch on the draft, reaction keys allowed in paths), `chat_agent` (tool-use loop that can read and *propose* only), `session_service` (owns the session lifecycle; **all store writes go through it** and it re-emits instances/issues on change).
  - `snapshots` — saves each successful run (10–15 min of LLM calls) keyed by PMID/PDF hash + focus gene; `replay` mode reuses them for UI work and demos.

Key invariants:
- Edits only take effect after a curator accepts a proposal. An edit changing what a reaction *is* (inputs, outputs, catalyst, regulators) needs a quote found in the paper, or must be marked a curator assertion.
- This service never writes to gk_central; commits go frontend → curator-tool-ws.
- Keep `WORKERS=1`: jobs run in-process and (with the memory store) state lives in the process.
- Speed: extraction runs chunks concurrently (`EXTRACT_WORKERS`, default 16; `1` = old serial mode where each chunk sees notes on the previous two, otherwise it sees the raw previous chunks), the extraction and duplicate-judge prompts are prompt-cached, and `run_merge.py` skips pairs that share no input/output entity (`services/dedup_prefilter.py`, `--no-prefilter` to disable). Which model each step uses lives in `curator_llm/model_settings.py` (roles + `LLM_MODEL_<ROLE>` / `LLM_THINKING_<ROLE>` overrides); add new LLM calls there, never hardcode a model id. Extended thinking is OFF for extraction and merge: it was most of the output tokens and latency, with no measured quality gain. Don't use Haiku as the duplicate judge (missed 10 of 24 merges). PINK1.pdf: extraction ~32s, merge ~44s (was ~5+ min and ~77s).
- A paper with no Results section (short communication, review, author manuscript) is extracted from the whole paper (`whole_paper` in `reactome_llm/FullTextPDFSections.py`: Abstract to References), flagged `via whole-paper` (no LLM boundary guess, which was not repeatable on such papers); the old behaviour was to abort.
- Draft entity kinds are protein | gene | rna | small_molecule | complex | set. A gene or transcript is an `EwasSpec` with `molecule='dna'|'rna'` (emitted with a ReferenceDNASequence / ReferenceRNASequence, no UniProt lookup); promoters/E-boxes are not entities, they are regions of the gene entity. `small_molecule` is for chemical compounds only.
- Extraction/merge/review still run as **subprocesses** (`run_extraction.py` → `run_merge.py` → `run_review.py`) on purpose, because merge/review keep state in module globals. Don't delete those scripts or inline them casually.
- Tests use fakes in `tests/fakes/ports.py`; the PINK1 paper (Kane et al. 2014, PMID 24751536) is the golden fixture.

## Gene-first pipeline (legacy)

`run_curator.py` runs three synchronous agents: Curator (`reactome_llm/ReactomeCurator.py`: placement, literature retrieval, full-text resolver, full-text analysis), Reviewer (`ReactomeReviewer.py`, decides sufficient/retry/give_up; retry cap lives in `run_curator.py`), and QA (`ReactomeQA.py`, using `reaction_to_instances.py`, schema checks vs `resources/reactome_domain_model.json`, Neo4j merge check, LLM review). Full-text extraction (`fulltext_extractor.py`) shells out to the `run_*.py` scripts. Mongo holds two DBs: PubMed abstract cache and functional-interaction partners (`idg_pairwise`). A FAIL/NEEDS_REVISION QA verdict is normal output, not a bug.

Other: `dense_retrieval/` is an experimental embedding investigation; `abstract_extractor.py`, `TextEmbedder.py` are standalone helpers; `docs/openapi.json` is generated.

## Conventions and gotchas

- Dependency injection through ports; add new external access as a port + adapter + fake, not by importing clients in services.
- Gitignored local data: `data/` caches (`snapshots`, `uploads`, `fulltext_*`), `results/`, `logs/`, `.env`, `resources/interactions/`.
- Related repos (same workspace): curator-tool-frontend (Angular UI), curator-tool-ws (Java; auth verify, gk_central reads/commit).
- Untracked scratch files in the root (`chainlit.md`, "Review curator-tool-llm …md") are not part of the code.
