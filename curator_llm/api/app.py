"""FastAPI app factory. Auth is a router-level dependency, so no route can be added without it, and
every session read goes through the owner check in the store."""
import dataclasses
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from curator_llm.models.session import Issue, PaperInfo, Session, SessionSummary
from curator_llm.models.usage import summarize
from curator_llm.ports.auth import AuthError, AuthProvider, AuthUser, ForbiddenError
from curator_llm.ports.jobs import JobRunner
from curator_llm.ports.pipeline import PaperRef, Pipeline, PipelineInput
from curator_llm.ports.sessions import SessionStore
from curator_llm.services.network import build_network
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
    pmid: Optional[str] = None                       # one paper ...
    pmids: List[str] = Field(default_factory=list)   # ... or several: merged into ONE draft, across papers
    focus: Optional[str] = None
    # Review articles keep the reactions they only cite; a research paper keeps only what its own results show (Reactome
    # needs direct evidence), and what it only cites is listed in the issues. Leave it out and each PMID's kind is read
    # from PubMed; true or false overrides that for every paper.
    review: Optional[bool] = None


class IssueUpdate(BaseModel):
    status: str   # open | resolved | dismissed


class ProposalRequest(BaseModel):
    reason: str = ''
    ops: List[Dict[str, Any]]                                   # RFC 6902 patch against the draft
    evidence: List[Dict[str, Any]] = Field(default_factory=list)  # [{quote, supports?, ...}], verified server-side
    curator_assertion: bool = False


MAX_UPLOAD_BYTES = 50 * 1024 * 1024           # per PDF
MAX_PAPERS = 10                                # per session, unless create_app is given another limit


class ChatRequest(BaseModel):
    message: str
    selectedDbIds: List[int] = Field(default_factory=list)


def parse_pmids(*groups) -> List[str]:
    """PubMed IDs from lists and/or strings (separated by commas or spaces), without repeats, in order. 422 on anything
    that is not a number."""
    out: List[str] = []
    for g in groups:
        for item in ([g] if isinstance(g, str) else (g or [])):
            for tok in re.split(r'[\s,;]+', str(item or '').strip()):
                if not tok:
                    continue
                if not tok.isdigit():
                    raise HTTPException(422, f'pmid must be numeric: {tok!r}')
                if tok not in out:
                    out.append(tok)
    return out


def _safe_name(name: str) -> str:
    base = os.path.basename(name or 'paper.pdf')
    return re.sub(r'[^A-Za-z0-9._-]+', '_', base)[:80] or 'paper.pdf'


def create_app(auth: AuthProvider, store: Optional[SessionStore] = None, jobs: Optional[JobRunner] = None,
               pipeline: Optional[Pipeline] = None, resolver_factory=None,
               upload_dir: Optional[str] = None, cors_origins: Optional[List[str]] = None,
               chat_model=None, events=None, qa_model=None, max_papers: int = MAX_PAPERS) -> FastAPI:
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
        """Annotate one paper by PubMed ID (`pmid`), or several (`pmids`) as one session: their reactions are merged into
        a single draft, and a reaction several papers describe cites all of them."""
        pmids = parse_pmids(req.pmid, req.pmids)
        if not pmids:
            raise HTTPException(422, 'give a pmid, or a list of pmids')
        if len(pmids) > max_papers:
            raise HTTPException(422, f'at most {max_papers} papers per session')
        spec = (PipelineInput(pmid=pmids[0], focus=req.focus, review=req.review) if len(pmids) == 1 else
                PipelineInput(papers=[PaperRef(pmid=p) for p in pmids], focus=req.focus, review=req.review))
        sid, jid = svc().start(user.username, spec)
        return {'sessionId': sid, 'jobId': jid}

    @router.post('/sessions/upload', status_code=202)
    async def upload(file: Optional[UploadFile] = File(None), files: Optional[List[UploadFile]] = File(None),
                     pmids: Optional[str] = Form(None), focus: Optional[str] = Form(None),
                     review: Optional[bool] = Form(None), review_files: Optional[List[str]] = Form(None),
                     user: AuthUser = Depends(current_user)):
        """Annotate papers that are not (all) in PubMed Central: upload PDFs. Send one in `file` (as before), or several
        in `files`, and optionally add PubMed IDs in `pmids` (comma or space separated). With more than one paper in all,
        they are merged into ONE draft, across papers. A PDF named in `review_files` (one field per file name, as uploaded) is a
        review article; any other PDF is research. Each PMID's kind is read from PubMed. `review=true|false` overrides
        both for every paper (see StartRequest)."""
        if not upload_dir:
            raise HTTPException(503, 'uploads are not configured')
        sent = [f for f in ([file] if file else []) + list(files or []) if f is not None and (f.filename or f.size)]
        ids = parse_pmids(pmids)
        if not sent and not ids:
            raise HTTPException(422, 'upload at least one PDF, or give pmids')
        if len(sent) + len(ids) > max_papers:
            raise HTTPException(422, f'at most {max_papers} papers per session')
        reviews = {n for n in (review_files or []) if n}
        folder = os.path.join(upload_dir, _safe_name(user.username))
        os.makedirs(folder, exist_ok=True)
        refs: List[PaperRef] = []
        saved: List[str] = []
        try:
            for f in sent:
                data = await f.read(MAX_UPLOAD_BYTES + 1)
                label = _safe_name(f.filename)
                if len(data) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f'{label}: the PDF is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB')
                if not data.startswith(b'%PDF-'):
                    raise HTTPException(422, f'{label}: the file is not a PDF')
                path = os.path.join(folder, f'{uuid.uuid4().hex[:12]}_{label}')
                with open(path, 'wb') as out:
                    out.write(data)
                saved.append(path)
                refs.append(PaperRef(pdf_path=path, label=label, review=True if f.filename in reviews else None))
        except HTTPException:
            for p in saved:                                # a refused request leaves nothing behind
                try:
                    os.remove(p)
                except OSError:
                    pass
            raise
        refs += [PaperRef(pmid=p) for p in ids]
        if review is not None:                             # the request-wide setting overrides the per-file one
            refs = [dataclasses.replace(r, review=review) for r in refs]
        if len(refs) == 1 and refs[0].pdf_path:            # one PDF: exactly as before
            sid, jid = svc().start(user.username, PipelineInput(pdf_path=refs[0].pdf_path, focus=focus or None,
                                                                review=refs[0].review), source_label=refs[0].label)
        else:
            sid, jid = svc().start(user.username, PipelineInput(papers=refs, focus=focus or None))
        return {'sessionId': sid, 'jobId': jid}

    @router.get('/sessions')
    def list_sessions(user: AuthUser = Depends(current_user)) -> List[SessionSummary]:
        return [SessionSummary.of(s) for s in svc().store.list(user.username)]

    @router.get('/sessions/{session_id}')
    def get_session(session_id: str, user: AuthUser = Depends(current_user)):
        s = load(session_id, user)
        out = SessionSummary.of(s).model_dump()
        out.update({'focus': s.focus, 'error': s.error,
                    'papers': [p.model_dump() for p in s.papers] or [
                        PaperInfo(source=s.source, kind='pmid' if s.pmid else 'pdf', pmid=s.pmid, name=s.source).model_dump()],
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

    @router.get('/sessions/{session_id}/usage')
    def usage(session_id: str, user: AuthUser = Depends(current_user)):
        """Language-model tokens spent on this session, per step (extraction, merge, review, draft, reaction checks,
        chat), with totals. When the result was replayed from a saved one (`source` is 'saved') the steps that produce
        the annotation are flagged `saved`: their numbers are from the original run and nothing was spent on them now,
        and `spent_now` totals only what was spent in this session. Steps that call no model are not listed."""
        s = load(session_id, user)
        return {'source': s.usage_source, 'entries': s.usage, **summarize(s.usage, s.usage_source == 'saved')}

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

    @router.get('/sessions/{session_id}/network')
    def network(session_id: str, user: AuthUser = Depends(current_user)):
        """The reactions and the entities they involve as nodes and edges (for drawing): input/output, catalyst,
        regulation, component/member and reaction-order edges, with each node's open-issue count."""
        s = load(session_id, user)
        return build_network(s.draft, s.issues, s.existing, s.key_to_db_id)

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

    @router.post('/sessions/{session_id}/qa')
    def qa_all(session_id: str, llm: bool = True, user: AuthUser = Depends(current_user)):
        """QA every reaction at once and return them most urgent first (priority fix > review > ok, then by number of
        action findings). Findings also become issues (source 'qa'), replacing each reaction's earlier QA issues."""
        load(session_id, user)
        try:
            return svc().run_qa_all(user.username, session_id, (qa_model or chat_model) if llm else None)
        except ProposalError as e:
            raise HTTPException(409, str(e))

    @router.post('/sessions/{session_id}/qa/{key}')
    def qa(session_id: str, key: str, llm: bool = True, user: AuthUser = Depends(current_user)):
        """QA one reaction: rule checks always, plus an LLM review when `llm` is true and chat is configured.
        Findings also become issues (source 'qa') on the session."""
        load(session_id, user)
        try:
            return svc().run_qa(user.username, session_id, key, (qa_model or chat_model) if llm else None)
        except ProposalError as e:
            raise HTTPException(404 if 'no reaction' in str(e) else 409, str(e))

    @router.get('/sessions/{session_id}/paper/search')
    def search_paper(session_id: str, q: str, section: Optional[str] = None, limit: int = 5, source: Optional[str] = None,
                     user: AuthUser = Depends(current_user)):
        """Passages of the session's papers matching `q` (best first), each with the paper it is from (`source`).
        Give `source` to search one paper only."""
        s = load(session_id, user)
        if not svc().papers_of(s):
            raise HTTPException(409, 'this session has no paper text')
        if source and source not in svc().papers_of(s):
            raise HTTPException(404, f'no such paper in this session: {source}')
        return svc().search_papers(s, q, section, max(1, min(limit, 20)), source or None)

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

    @router.post('/sessions/{session_id}/suggestions')
    def suggestions(session_id: str, user: AuthUser = Depends(current_user)):
        """Make the suggested fixes for the draft as it is now and return the pending ones. They are proposals (actor
        'suggestions') like any other: nothing changes until one is accepted. A fix the curator already decided on is
        not made again. They are also made when a session is created and after each accepted proposal."""
        load(session_id, user)
        try:
            return svc().suggest_fixes(user.username, session_id)
        except ProposalError as e:
            raise HTTPException(409, str(e))

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
