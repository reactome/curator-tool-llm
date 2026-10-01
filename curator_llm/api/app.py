"""FastAPI app factory. Auth is a router-level dependency, so no route can be added without it, and
every session read goes through the owner check in the store."""
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from curator_llm.models.session import Issue, Session, SessionSummary
from curator_llm.ports.auth import AuthError, AuthProvider, AuthUser, ForbiddenError
from curator_llm.ports.jobs import JobRunner
from curator_llm.ports.pipeline import Pipeline, PipelineInput
from curator_llm.ports.sessions import SessionStore
from curator_llm.services.session_service import ProposalError, SessionService


def current_user(request: Request) -> AuthUser:
    header = request.headers.get('Authorization', '')
    token = header[7:].strip() if header.startswith('Bearer ') else ''
    provider: AuthProvider = request.app.state.auth
    try:
        return provider.verify(token)
    except ForbiddenError as e:
        raise HTTPException(403, str(e))
    except AuthError as e:
        raise HTTPException(401, str(e))


class StartRequest(BaseModel):
    pmid: str
    focus: Optional[str] = None


class IssueUpdate(BaseModel):
    status: str   # open | resolved | dismissed


class ProposalRequest(BaseModel):
    reason: str = ''
    ops: List[Dict[str, Any]]                                   # RFC 6902 patch against the draft
    evidence: List[Dict[str, Any]] = Field(default_factory=list)  # [{quote, supports?, ...}], verified server-side
    curator_assertion: bool = False


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class ChatRequest(BaseModel):
    message: str
    selectedDbIds: List[int] = Field(default_factory=list)


def _safe_name(name: str) -> str:
    base = os.path.basename(name or 'paper.pdf')
    return re.sub(r'[^A-Za-z0-9._-]+', '_', base)[:80] or 'paper.pdf'


def create_app(auth: AuthProvider, store: Optional[SessionStore] = None, jobs: Optional[JobRunner] = None,
               pipeline: Optional[Pipeline] = None, resolver_factory=None,
               upload_dir: Optional[str] = None, cors_origins: Optional[List[str]] = None,
               chat_model=None, events=None) -> FastAPI:
    app = FastAPI(title='curator-tool-llm')
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_credentials=True,
                           allow_methods=['*'], allow_headers=['Authorization', 'Content-Type'])
    app.state.auth = auth
    service = SessionService(store, jobs, pipeline, resolver_factory, events) if store and jobs and pipeline else None
    app.state.service, app.state.store, app.state.jobs = service, store, jobs
    router = APIRouter(prefix='/api/llm', dependencies=[Depends(current_user)])

    def svc() -> SessionService:
        if service is None:
            raise HTTPException(503, 'annotation service is not configured')
        return service

    def load(session_id: str, user: AuthUser) -> Session:
        s = svc().store.get(session_id, user.username)
        if s is None:
            raise HTTPException(404, 'no such session')      # same answer for "not yours" and "not there"
        return s

    @router.get('/whoami')
    def whoami(user: AuthUser = Depends(current_user)):
        return {'username': user.username, 'role': user.role}

    @router.post('/sessions', status_code=202)
    def start(req: StartRequest, user: AuthUser = Depends(current_user)):
        if not req.pmid.strip().isdigit():
            raise HTTPException(422, 'pmid must be numeric')
        sid, jid = svc().start(user.username, PipelineInput(pmid=req.pmid.strip(), focus=req.focus))
        return {'sessionId': sid, 'jobId': jid}

    @router.post('/sessions/upload', status_code=202)
    async def upload(file: UploadFile = File(...), focus: Optional[str] = Form(None),
                     user: AuthUser = Depends(current_user)):
        """Annotate a paper that is not in PubMed Central: upload its PDF."""
        if not upload_dir:
            raise HTTPException(503, 'uploads are not configured')
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f'PDF is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB')
        if not data.startswith(b'%PDF-'):
            raise HTTPException(422, 'the file is not a PDF')
        label = _safe_name(file.filename)
        folder = os.path.join(upload_dir, _safe_name(user.username))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'{uuid.uuid4().hex[:12]}_{label}')
        with open(path, 'wb') as f:
            f.write(data)
        sid, jid = svc().start(user.username, PipelineInput(pdf_path=path, focus=focus or None),
                               source_label=label)
        return {'sessionId': sid, 'jobId': jid}

    @router.get('/sessions')
    def list_sessions(user: AuthUser = Depends(current_user)) -> List[SessionSummary]:
        return [SessionSummary.of(s) for s in svc().store.list(user.username)]

    @router.get('/sessions/{session_id}')
    def get_session(session_id: str, user: AuthUser = Depends(current_user)):
        s = load(session_id, user)
        out = SessionSummary.of(s).model_dump()
        out.update({'focus': s.focus, 'error': s.error,
                    'reactions': [{'key': r.key, 'name': r.name, 'dbId': s.key_to_db_id.get(r.key),
                                   'evidenceIds': r.evidence_ids} for r in (s.draft.reactions if s.draft else [])]})
        return out

    @router.get('/sessions/{session_id}/issues')
    def issues(session_id: str, status: Optional[str] = None,
               user: AuthUser = Depends(current_user)) -> List[Issue]:
        s = load(session_id, user)
        return [i for i in s.issues if status is None or i.status == status]

    @router.patch('/sessions/{session_id}/issues/{issue_id}')
    def update_issue(session_id: str, issue_id: str, body: IssueUpdate, user: AuthUser = Depends(current_user)):
        if body.status not in ('open', 'resolved', 'dismissed'):
            raise HTTPException(422, 'status must be open, resolved or dismissed')
        s = load(session_id, user)
        issue = next((i for i in s.issues if i.id == issue_id), None)
        if issue is None:
            raise HTTPException(404, 'no such issue')
        issue.status = body.status
        svc().store.save(s)
        return issue

    @router.get('/sessions/{session_id}/export')
    def export(session_id: str, user: AuthUser = Depends(current_user)):
        s = load(session_id, user)
        if s.user_instances is None:
            raise HTTPException(409, f'session is {s.status}; nothing to export yet')
        return s.user_instances

    @router.get('/sessions/{session_id}/instances/{db_id}/evidence')
    def instance_evidence(session_id: str, db_id: int, user: AuthUser = Depends(current_user)):
        s = load(session_id, user)
        by_id = {e.id: e for e in s.evidence}
        return [{'field': l['field'], 'evidence': by_id[l['evidenceId']]}
                for l in s.evidence_links if l['instanceDbId'] == db_id and l['evidenceId'] in by_id]

    @router.get('/sessions/{session_id}/reactions/{key}')
    def reaction(session_id: str, key: str, user: AuthUser = Depends(current_user)):
        s = load(session_id, user)
        r = next((x for x in (s.draft.reactions if s.draft else []) if x.key == key), None)
        if r is None:
            raise HTTPException(404, 'no such reaction')
        used = set(r.inputs + r.outputs + [g.regulator for g in r.regulations] + ([r.catalyst.entity] if r.catalyst else []))
        by_id = {e.id: e for e in s.evidence}
        return {'reaction': r, 'dbId': s.key_to_db_id.get(key),
                'participants': {k: s.draft.participants[k] for k in used if k in s.draft.participants},
                'evidence': [by_id[i] for i in r.evidence_ids if i in by_id]}

    @router.get('/sessions/{session_id}/existing')
    def existing(session_id: str, user: AuthUser = Depends(current_user)):
        """Draft reactions that look like reactions Reactome already has (level same | similar)."""
        return load(session_id, user).existing

    @router.post('/sessions/{session_id}/existing/check')
    def existing_check(session_id: str, user: AuthUser = Depends(current_user)):
        load(session_id, user)
        if svc().events is None:
            raise HTTPException(503, 'existing-event check is not configured')
        try:
            s = svc().check_existing(user.username, session_id)
        except ProposalError as e:
            raise HTTPException(409, str(e))
        return s.existing

    @router.post('/sessions/{session_id}/qa/{key}')
    def qa(session_id: str, key: str, llm: bool = True, user: AuthUser = Depends(current_user)):
        """QA one reaction: rule checks always, plus an LLM review when `llm` is true and chat is configured.
        Findings also become issues (source 'qa') on the session."""
        load(session_id, user)
        try:
            return svc().run_qa(user.username, session_id, key, chat_model if llm else None)
        except ProposalError as e:
            raise HTTPException(404 if 'no reaction' in str(e) else 409, str(e))

    @router.get('/sessions/{session_id}/paper/search')
    def search_paper(session_id: str, q: str, section: Optional[str] = None, limit: int = 5,
                     user: AuthUser = Depends(current_user)):
        from curator_llm.services.paper_text import PaperText
        s = load(session_id, user)
        if not s.paper:
            raise HTTPException(409, 'this session has no paper text')
        hits = PaperText.from_dict(s.paper).search(q, section, top_k=max(1, min(limit, 20)))
        return [{'text': h.text, 'section': h.section, 'page': h.page, 'figure': h.figure,
                 'charSpan': h.char_span, 'score': h.score} for h in hits]

    @router.post('/sessions/{session_id}/proposals', status_code=201)
    def propose(session_id: str, body: ProposalRequest, user: AuthUser = Depends(current_user)):
        load(session_id, user)
        try:
            return svc().propose(user.username, session_id, body.ops, body.reason, body.evidence,
                                 actor=user.username, curator_assertion=body.curator_assertion)
        except ProposalError as e:
            raise HTTPException(422, str(e))

    @router.get('/sessions/{session_id}/proposals')
    def proposals(session_id: str, status: Optional[str] = None, user: AuthUser = Depends(current_user)):
        s = load(session_id, user)
        return [p for p in s.proposals if status is None or p.status == status]

    @router.post('/sessions/{session_id}/proposals/{proposal_id}/accept')
    def accept(session_id: str, proposal_id: str, user: AuthUser = Depends(current_user)):
        load(session_id, user)
        try:
            s = svc().accept(user.username, session_id, proposal_id)
        except ProposalError as e:
            raise HTTPException(409, str(e))
        return {'proposal': next(p for p in s.proposals if p.id == proposal_id),
                'reactions': [{'key': r.key, 'name': r.name, 'dbId': s.key_to_db_id.get(r.key)} for r in s.draft.reactions],
                'nOpenIssues': sum(i.status == 'open' for i in s.issues)}

    @router.post('/sessions/{session_id}/proposals/{proposal_id}/reject')
    def reject(session_id: str, proposal_id: str, user: AuthUser = Depends(current_user)):
        load(session_id, user)
        try:
            s = svc().reject(user.username, session_id, proposal_id)
        except ProposalError as e:
            raise HTTPException(409, str(e))
        return next(p for p in s.proposals if p.id == proposal_id)

    @router.get('/sessions/{session_id}/chat')
    def chat_history(session_id: str, user: AuthUser = Depends(current_user)):
        """The conversation so far (user and assistant text, and the ids of any proposals a reply made)."""
        return load(session_id, user).chat

    @router.post('/sessions/{session_id}/chat')
    def chat(session_id: str, body: ChatRequest, user: AuthUser = Depends(current_user)):
        """Server-sent events: text (delta), tool, proposal, error, done. Use fetch with a streaming reader
        (EventSource cannot send the Authorization header)."""
        import json as _json
        from curator_llm.services.chat_agent import ChatAgent
        if chat_model is None:
            raise HTTPException(503, 'chat is not configured')
        s = load(session_id, user)
        if s.status != 'ready':
            raise HTTPException(409, f'session is {s.status}; chat needs a ready session')
        agent = ChatAgent(svc(), chat_model)
        events = agent.run(user.username, session_id, body.message, body.selectedDbIds)

        def sse():
            for ev in events:
                yield f"event: {ev['event']}\ndata: {_json.dumps(ev['data'], default=str)}\n\n"
        return StreamingResponse(sse(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @router.get('/jobs/{job_id}')
    def job(job_id: str, user: AuthUser = Depends(current_user)):
        j = svc().jobs.get(job_id, user.username)
        if j is None:
            raise HTTPException(404, 'no such job')
        return {'id': j.id, 'status': j.status, 'progress': j.progress, 'error': j.error}

    app.include_router(router)
    return app
