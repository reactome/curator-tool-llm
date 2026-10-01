# Implementation plan: paper-level annotation and chat

Source: "Review curator-tool-llm for paper-level annotation and chat" (reviewed at commit 81a3143).

## Decisions

| Topic | Decision |
| --- | --- |
| Architecture | curator-tool-llm is a REST service. curator-tool-frontend hosts all UI (chat panel, diff accept/reject, evidence viewer). |
| Auth | Option B. The frontend calls the LLM service directly with its existing bearer token. The service verifies each token through a new curator-tool-ws endpoint, `GET /api/auth/verify`. |
| Permission | The existing `curator` role. No new flag. |
| Testability | ws calls sit behind interfaces (`AuthProvider`, `InstanceLookup`) with fakes. A separate contract test runs against a real ws. |
| Evidence in staged instances | The frontend keeps non-schema attributes and gets a UI to show evidence outside the table. |
| Cross-model review | Keep it (`run_review.py`, OpenAI). |
| Background jobs | In-process worker behind a `JobRunner` interface. Job state is stored in MongoDB. A queue (RQ) can replace it later. |
| Commit to gk_central | Frontend through ws only. This service never writes to gk_central. |

## Status (backend)

Done and tested (157 tests): evidence objects and quote verification; typed draft, resolvers (Neo4j release graph, UniProt, OLS) and emitter; LLM draft builder; structured issues; sessions (in-memory and Mongo stores); REST API incl. PDF upload, proposals (propose / accept / reject), paper search, reaction detail, issues, export, evidence by dbId; chat over SSE; CORS; ws `GET /api/auth/verify`. The API contract is `docs/openapi.json` (regenerate with `python scripts/export_openapi.py`). Verified with real runs: PINK1 paper (PDF -> ready session with 11 reactions, 10 open issues) and a real Claude chat turn (searched the paper, proposed adding CCCP as a regulator with an exact-match quote; nothing applied until accepted).

Also done since: (1) extract -> merge -> review behind a typed `Extractor` (`ExtractionResult`), still one subprocess per step on purpose: `run_merge.py` and `run_review.py` keep state in module globals, so running them in-process would let concurrent sessions corrupt each other; (2) existing-event detection (`services/existing_events.py`): PMID check plus accession-based matching against the Reactome graph (same / similar), exposed as `GET /sessions/{id}/existing`, `POST /sessions/{id}/existing/check` and the chat tool `find_existing_reactome`; reusing an existing reaction is an ordinary proposal that sets `/reactions/<key>/existing`; (3) per-reaction QA (`services/qa.py`): rule checks plus an optional model review, `POST /sessions/{id}/qa/{key}` and the chat tool `run_qa`, findings recorded as issues. Issue ids and statuses are stable across edits and re-checks.

Known limits: existing-event matching uses the release graph, so unreleased gk_central work is not seen; draft reactions with a single accession and no catalyst produce weak info-level "similar" matches; extraction and merge take about 10-15 minutes per paper.

Not done: frontend work (Phase 6), and checking `SESSION_STORE=mongo` inside the running app.

Run the service: `uvicorn curator_llm.main:app --port 8000`. Settings (in `.env`): `WS_BASE_URL`, `SESSION_STORE=memory|mongo`, `CORS_ORIGINS`, `UPLOAD_DIR`, `CHAT_MODEL`, `LLM_REVIEW=1`.

Frontend notes: the chat endpoint is a POST that streams `text/event-stream`, so use `fetch` with a streaming reader (EventSource cannot send the Authorization header). Issues (everything needing manual attention) come from `GET /sessions/{id}/issues`; each has a severity, a stable code, and the `instance_db_id` it concerns. Evidence is fetched by instance dbId, because the frontend's persist and commit paths drop unknown instance fields.

## Architecture

```
curator-tool-frontend (Angular)         curator-tool-llm (FastAPI)                  curator-tool-ws (Java)
  chat panel, diff accept/reject  ──▶   /api/llm/...  sessions, chat (SSE)   ──▶   /api/auth/verify
  evidence viewer                       pipeline functions + resolvers              gk_central read lookups
  staged instances                                                                  commit (frontend only)
```

## Phase 0: Foundations
- Create a `curator_llm/` package: `models/` (pydantic), `services/` (importable pipeline functions), `api/` (routers), `ports/` (interfaces), `adapters/` (production implementations).
- Turn the subprocess chain (`run_extraction.py` → `run_merge.py` → `run_review.py`) into functions that return objects. The CLIs become thin wrappers, and `run_curator.py` keeps working.
- Remove dead code: Phase 1/3/5 models in `ReactomeModels.py`, CrewAI config in `ModelConfig.py`.
- Interfaces:
  - `AuthProvider.verify(token) -> {username, role}`.
  - `InstanceLookup` (gk_central search by display name, PMID, dbId).
  - `JobRunner`.
- Production adapters: `WsAuthProvider`, `WsInstanceLookup`, `InProcessJobRunner`.
- Fakes in `tests/fakes/`: auth, lookup, LLM client, and a synchronous job runner.
- pytest, with the PINK1 paper (Kane et al. 2014) as the golden fixture.
- The service listens only on the internal network, reachable from the frontend and ws hosts.

## Phase 1: Evidence integrity
- `Evidence` model: `id`, `quote`, `section`, `page`, `figure`, `char_span`, `verified` (exact / fuzzy ≥ 0.9 / failed), `supports`, `system`, `experimental_species`, `claim_origin`, `strength`.
- `PaperText` store: per-page extraction (PyMuPDF), PMC `<sec>` and `<fig>` tracking, whole paper indexed with sections tagged.
- Quote verifier: pure function, exact match then fuzzy then reject. Heavily unit-tested.
- Prompt and schema (`FullTextPDFPrompts.py`): emit `supports`, `system`, `experimental_species`, `claim_origin`. Keep normalizing the entity, but stop erasing ortholog provenance.
- `reaction_to_instances.py`: reactions hold `evidence_ids`. Evidence is stored outside the LLM output, and the 2-excerpt cap is removed.
- Exit test: on PINK1, every kept quote is verified and none are truncated.

## Phase 2: Typed model and emitter
- Typed participants: `EntityWithAccessionedSequence` (with modified residues), `SimpleEntity` (ChEBI), `Complex`, `DefinedSet`, `CatalystActivity` (GO function), typed `Regulation`, `precedingEvent`, `inferredFrom`.
- Resolvers (code only): UniProt, ChEBI (OLS), GO, and gk_central lookups through `InstanceLookup`. Unresolved names get `needs_resolution`, never a guessed ID.
- Emitter: `UserInstances` JSON, with negative dbIds for new instances and shells for existing ones. Confirm the exact format against the frontend's `UserInstancesService` and `hydrateUserInstances`.
- Evidence does NOT travel inside the instances. Checked in the frontend: `cloneInstanceForCommit` (used by persist, backup and commit) rebuilds each instance from `dbId`, `displayName`, `schemaClassName`, flags and `attributes` only, so a new top-level field is silently dropped on persist. An unknown key inside `attributes` would survive persist but would also be sent to ws on commit. So the emitter returns `evidence_links` (evidence id -> instance dbId + field) separately, the session store serves them by dbId (`GET /sessions/{id}/instances/{dbId}/evidence`), and the frontend's evidence viewer calls that. Because new dbIds are negative until commit, the store keeps the negative id and maps it through `newInstOld2NewId` after commit.
- Emitted shape (matches `Instance`/`UserInstances` in the frontend): every new instance is its own `newInstances` entry with a negative dbId; references between instances are shells `{dbId, displayName, schemaClassName}`; existing gk_central instances are shells only.
- Exit test: the PINK1 output loads through "Load staged instances".

## Phase 3: Paper-first entry and existing-event detection
- `annotate_paper(pmid | pdf, focus=None)`. The gene flow becomes a caller of it.
- PMID check against gk_central literatureReference, then `dense_retrieval/ReactionMatcher.py` promoted into the pipeline as `find_existing_reactome`.
- The Neo4j release graph remains for pathway placement only.

## Phase 4: Session store and REST API
- MongoDB stores sessions (paper, reactions, evidence, change log, status) and job state.
- A router-level auth dependency applies to every route, calling `AuthProvider`.
  - 401 for a missing or invalid token. 403 for a role other than `curator`.
  - Fails closed if ws is unreachable.
  - Result cached per token for 30–60 seconds.
- Sessions are scoped to the verified username.
- Jobs check authorization when created and when results are read, not inside the worker.
- Tests:
  - A test that fails if any route lacks the auth dependency.
  - Per-user session isolation tests.
  - One contract test against a real ws, run separately from the unit suite.
- Endpoints (under `/api/llm`):
  - `POST /sessions` returns 202 with a job id.
  - `GET /jobs/{id}` for progress.
  - `GET /sessions/{id}` and `GET /sessions/{id}/reactions/{rid}`.
  - `PATCH /sessions/{id}/reactions/{rid}` (validated JSON Patch, `?propose=true`).
  - `POST /sessions/{id}/proposals/{pid}/accept` and `/reject`.
  - `GET /sessions/{id}/paper/search?q=&section=`.
  - `POST /sessions/{id}/qa/{rid}`.
  - `GET /sessions/{id}/export`.
- Publish an OpenAPI spec as the frontend contract at the start of this phase.
- In ws, add `GET /api/auth/verify` returning `{username, role}`. The existing `JwtRequestFilter` already validates the token.

## Phase 5: Chat
- `POST /sessions/{id}/chat` takes `{message, selectedDbIds}` and streams events over SSE: text deltas, tool calls, `proposal` events with diffs.
- The frontend uses `fetch` with a streaming reader, because `EventSource` cannot send the bearer header.
- Agent: Claude with tool use. Tools: `search_paper`, `get_reaction`, `list_reactions`, `propose_patch`, `add_evidence`, `resolve_identifier`, `find_existing_reactome`, `run_qa`, `rerun_extraction`.
- System prompt reuses the extraction prompt's curation rules.
- Guardrails:
  - `propose_patch` never mutates state. Only the accept endpoint does.
  - New claims need verified evidence, or are saved as `curator_assertion`.
  - Edits are per reaction.
  - Cap tool steps and tokens per turn.
- The cross-model review stays on OpenAI. The chat agent runs on Anthropic.

## Phase 6: Frontend (curator-tool-frontend)
- Chat panel, with diff accept/reject.
- Evidence viewer outside the table: quote, section, page, figure, verification status, supports link, species, system, claim origin.
- Session list.
- Accepted changes written into staged instances, with evidence preserved (see the Phase 2 check).
- Build against a mock of the OpenAPI contract first, so this does not wait on Phases 4–5.

## Phase 7: Benchmark and cleanup
- Reaction-level gold set from Reactome literatureReferences. Track extraction precision/recall, participant accuracy and quote-verification rate per release.
- Compare whole-Results-section extraction against the current 1,200-character chunks.
- Demote self-assessed `confidence` to triage only.

## Order
1. Phases 0 and 1.
2. Phase 2, then Phase 3.
3. Phase 4. Publish the OpenAPI contract at its start so Phase 6 can begin in parallel.
4. Phase 5, then Phase 7.

## Risks and open checks
- Whether the frontend preserves and strips non-schema attributes (Phase 2 check).
- ws token algorithm, only if local JWT verification is ever wanted. Not needed for option B.
- In-process jobs are lost on a restart. Job state in MongoDB lets the user see the failure and rerun. Move to RQ if this becomes a problem.
