"""Owns the life of an annotation session: start it (background job), edit it through proposals, and
(re)emit instances and issues whenever the draft changes. All writes to the store go through here."""
import functools
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from curator_llm.models.evidence import ClaimOrigin, Evidence, Verification
from curator_llm.models.session import ChangeLogEntry, Issue, Proposal, Session
from curator_llm.models.usage import UsageEntry
from curator_llm.ports.jobs import JobRunner
from curator_llm.ports.pipeline import Pipeline, PipelineInput
from curator_llm.ports.sessions import SessionStore
from curator_llm.services import issues as iss
from curator_llm.services.emitter import emit_user_instances
from curator_llm.services.evidence_store import EvidenceStore
from curator_llm.services.existing_events import find_existing
from curator_llm.services.paper_text import PaperText
from curator_llm.services.patching import PatchError, apply_patch, describe_changes, touched_reactions
from curator_llm.services.qa import qa_reaction


class ProposalError(ValueError):
    """An edit, check or decision that cannot be done; the message is safe to show to the curator or the model."""


_CLAIM_FIELDS = {'inputs', 'outputs', 'catalyst', 'regulations', 'preceding'}
_locks_guard = threading.Lock()
_locks: Dict[str, threading.Lock] = {}


def _locked(method):
    """Serialise operations that read-modify-write one session (two tabs, or chat plus the UI)."""
    @functools.wraps(method)
    def wrapper(self, owner, session_id, *args, **kwargs):
        with _locks_guard:
            lock = _locks.setdefault(session_id, threading.Lock())
        with lock:
            return method(self, owner, session_id, *args, **kwargs)
    return wrapper


def _needs_evidence(ops: List[dict]) -> bool:
    """Changing what a reaction IS (participants, catalyst, regulation, order) or adding/removing a
    reaction is a claim about the paper. Renames, summations and identifier fixes are not."""
    for op in ops:
        parts = op.get('path', '').split('/')[1:]
        if len(parts) >= 2 and parts[0] == 'reactions':
            if len(parts) == 2 and op.get('op') in ('add', 'remove'):
                return True
            if len(parts) >= 3 and parts[2] in _CLAIM_FIELDS:
                return True
    return False


def _carry_over(previous: List[Issue], fresh: List[Issue], key: Callable[[Issue], tuple]) -> None:
    """Give recurring issues the id and status they had before, so a curator's reference to an issue and
    their decision to dismiss it survive a re-run."""
    old = {key(i): (i.status, i.id) for i in previous}
    for i in fresh:
        if key(i) in old:
            i.status, i.id = old[key(i)]


class SessionService:
    def __init__(self, store: SessionStore, jobs: JobRunner, pipeline: Pipeline,
                 resolver_factory: Optional[Callable] = None, events=None):
        self.store, self.jobs, self.pipeline = store, jobs, pipeline
        self.resolver_factory = resolver_factory     # () -> Resolver, to resolve entities an edit introduces
        self.events = events                         # EventLookup, to check for existing Reactome events

    def _ready(self, owner: str, session_id: str) -> Session:
        s = self.store.get(session_id, owner)
        if s is None or s.draft is None:
            raise ProposalError('session is not ready')
        return s

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self, owner: str, spec: PipelineInput, source_label: Optional[str] = None) -> Tuple[str, str]:
        """Create the session, queue the pipeline, return (session_id, job_id)."""
        s = Session(owner=owner, pmid=spec.pmid, focus=spec.focus,
                    source=source_label or spec.pmid or (spec.pdf_path or '').rsplit('/', 1)[-1])
        self.store.save(s)
        job_id = self.jobs.submit(owner, lambda report: self._run(s.id, owner, spec, report))
        return s.id, job_id

    def _run(self, session_id: str, owner: str, spec: PipelineInput, report: Callable[[str], None]) -> str:
        def progress(msg: str):
            report(msg)
            cur = self.store.get(session_id, owner)
            cur.status, cur.progress = 'running', msg
            self.store.save(cur)
        progress('starting')
        try:
            result = self.pipeline.run(spec, progress)
            s = self.store.get(session_id, owner)
            s.draft, s.evidence, s.paper = result.draft, result.evidence, result.paper
            s.existing, s.issues = list(result.existing), list(result.issues)
            s.usage, s.usage_source = list(result.usage), 'saved' if result.replayed else 'run'
            self.refresh(s)
            s.status, s.progress, s.error = 'ready', 'done', None
            s.change_log.append(ChangeLogEntry(actor='pipeline', action='created',
                                               detail=f'{len(result.draft.reactions)} reactions'))
        except Exception as e:
            s = self.store.get(session_id, owner)
            s.status, s.error = 'failed', f'{type(e).__name__}: {e}'
            self.store.save(s)
            raise
        self.store.save(s)
        return session_id

    @staticmethod
    def refresh(s: Session) -> None:
        """Re-emit instances from the draft and rebuild the issue list. Emitter issues are replaced on every
        refresh; issues that recur keep the id and status they had."""
        key = lambda i: (i.code, i.message, i.reaction_key, i.participant_key)
        kept = [i for i in s.issues if i.source != 'emitter']
        emit = emit_user_instances(s.draft, {e.id: e for e in s.evidence})
        s.user_instances, s.evidence_links, s.key_to_db_id = emit.user_instances, emit.evidence_links, emit.key_to_db_id
        have = {(i.code, i.participant_key) for i in kept}
        merged = kept + [f for f in iss.issues_from_draft(s.draft) if (f.code, f.participant_key) not in have]
        merged += iss.issues_from_notes('emitter', emit.warnings)
        iss.link_to_instances(merged, emit.key_to_db_id, emit.participant_to_db_id)
        _carry_over(s.issues, merged, key)
        s.issues = iss.assign_ids(merged)

    # ── proposals ──────────────────────────────────────────────────────────
    @staticmethod
    def _paper(s: Session) -> Optional[PaperText]:
        return PaperText.from_dict(s.paper) if s.paper else None

    def _verify_evidence(self, s: Session, drafts: List[dict]) -> List[Evidence]:
        """Quotes checked against the paper; a curator assertion needs no paper. Raises on any quote not found."""
        paper, verified, failed = self._paper(s), [], []
        for raw in drafts:
            ev = Evidence(**raw)
            if ev.claim_origin == ClaimOrigin.CURATOR_ASSERTION:
                verified.append(ev)
            elif paper is None:
                raise ProposalError('this session has no paper text to verify quotes against')
            else:
                checked = paper.verify(ev)
                (failed if checked.verified == Verification.FAILED else verified).append(
                    ev if checked.verified == Verification.FAILED else checked)
        if failed:
            raise ProposalError('these quotes were not found in the paper (quote exactly, or use '
                                'curator_assertion): ' + ' | '.join(f'"{f.quote[:100]}"' for f in failed))
        return verified

    @_locked
    def propose(self, owner: str, session_id: str, ops: List[dict], reason: str = '',
                evidence: Optional[List[dict]] = None, actor: str = 'chat',
                curator_assertion: bool = False) -> Proposal:
        """Validate an edit and hold it for review. Nothing changes until it is accepted."""
        s = self._ready(owner, session_id)
        try:
            after = apply_patch(s.draft, ops)
        except PatchError as e:
            raise ProposalError(str(e)) from e
        verified = self._verify_evidence(s, evidence or [])
        if curator_assertion and not verified:
            verified.append(Evidence(quote=reason or 'curator assertion', claim_origin=ClaimOrigin.CURATOR_ASSERTION))
        if _needs_evidence(ops) and not verified:
            raise ProposalError('this edit changes what a reaction is, so it needs at least one verified '
                                'quote from the paper, or curator_assertion=true')
        after_keys = {r.key for r in after.reactions}
        p = Proposal(id=f'p-{1 + max([int(x.id.split("-")[1]) for x in s.proposals] or [0]):03d}',
                     reason=reason, ops=ops, summary=describe_changes(s.draft, after),
                     reaction_keys=[k for k in touched_reactions(s.draft, after) if k in after_keys],
                     evidence=verified, actor=actor)
        if not p.summary:
            raise ProposalError('this edit changes nothing')
        s.proposals.append(p)
        self.store.save(s)
        return p

    def _pending(self, owner: str, session_id: str, proposal_id: str) -> Tuple[Session, Proposal]:
        s = self.store.get(session_id, owner)
        if s is None:
            raise ProposalError('no such session')
        p = next((x for x in s.proposals if x.id == proposal_id), None)
        if p is None:
            raise ProposalError('no such proposal')
        if p.status != 'pending':
            raise ProposalError(f'proposal is already {p.status}')
        return s, p

    @_locked
    def accept(self, owner: str, session_id: str, proposal_id: str) -> Session:
        """Apply a proposal: patch the draft, register its evidence, resolve what it introduced, re-emit."""
        s, p = self._pending(owner, session_id, proposal_id)
        try:
            new = apply_patch(s.draft, p.ops)
        except PatchError as e:                       # the draft changed since the proposal was made
            p.status, p.decided_by, p.decided_at = 'stale', owner, time.time()
            self.store.save(s)
            raise ProposalError(f'proposal no longer applies to the current draft: {e}') from e
        evidence = EvidenceStore.load(s.evidence, self._paper(s))
        ids = [evidence.add(e)[0].id for e in p.evidence]
        keys = [r.key for r in new.reactions if r.key in p.reaction_keys]
        for r in new.reactions:
            if r.key in keys:
                r.evidence_ids = list(dict.fromkeys(r.evidence_ids + ids))
        s.draft, s.evidence = new, evidence.all()
        if self.resolver_factory:
            have = {(i.code, i.message) for i in s.issues}
            s.issues += [i for i in iss.issues_from_notes('resolver', self.resolver_factory().resolve(s.draft))
                         if (i.code, i.message) not in have]
        self.refresh(s)
        self._recheck_existing(s, keys or None)
        p.status, p.decided_by, p.decided_at = 'accepted', owner, time.time()
        s.change_log.append(ChangeLogEntry(actor=owner, action=f'accepted {p.id}', detail='; '.join(p.summary)[:500]))
        self.store.save(s)
        return s

    @_locked
    def reject(self, owner: str, session_id: str, proposal_id: str) -> Session:
        s, p = self._pending(owner, session_id, proposal_id)
        p.status, p.decided_by, p.decided_at = 'rejected', owner, time.time()
        s.change_log.append(ChangeLogEntry(actor=owner, action=f'rejected {p.id}'))
        self.store.save(s)
        return s

    # ── checks that produce issues ─────────────────────────────────────────
    @_locked
    def run_qa(self, owner: str, session_id: str, reaction_key: str, model=None):
        """QA one reaction. Its previous QA issues are replaced (ids and statuses of recurring ones kept)."""
        s = self._ready(owner, session_id)
        try:
            result = qa_reaction(s.draft, reaction_key, s.evidence, model)
        except KeyError:
            raise ProposalError(f'no reaction with key {reaction_key!r}')
        mine = lambda i: i.source == 'qa' and i.reaction_key == reaction_key
        fresh = [Issue(source='qa', severity=f.severity, code=f.code, message=f.message, reaction_key=reaction_key,
                       instance_db_id=s.key_to_db_id.get(reaction_key)) for f in result.findings]
        _carry_over([i for i in s.issues if mine(i)], fresh, lambda i: (i.code, i.message, i.reaction_key))
        s.issues = iss.assign_ids([i for i in s.issues if not mine(i)] + fresh)
        if result.usage:
            s.usage.append(UsageEntry(step='qa', calls=1, detail=reaction_key, model=getattr(model, 'model', None),
                                      **result.usage))
        s.change_log.append(ChangeLogEntry(actor=owner, action=f'qa {reaction_key}', detail=result.verdict))
        self.store.save(s)
        return result

    @_locked
    def check_existing(self, owner: str, session_id: str, only: Optional[List[str]] = None) -> Session:
        """Compare the draft (or just the reactions in `only`) with Reactome's existing events and refresh
        the matching issues. Ids and statuses of issues that recur are kept."""
        s = self._ready(owner, session_id)
        self._recheck_existing(s, only)
        self.store.save(s)
        return s

    def _recheck_existing(self, s: Session, only: Optional[List[str]] = None) -> None:
        if self.events is None or s.draft is None:
            return
        matches, fresh = find_existing(s.draft, s.pmid, self.events, only)
        in_scope = lambda i: only is None or i.reaction_key in only
        stale = [i for i in s.issues if i.source == 'existing' and in_scope(i)]
        _carry_over(stale, fresh, lambda i: (i.code, i.message, i.reaction_key, i.participant_key))
        iss.link_to_instances(fresh, {k: v for k, v in s.key_to_db_id.items() if v is not None})
        s.issues = iss.assign_ids([i for i in s.issues if i not in stale] + fresh)
        s.existing = [m for m in s.existing if only is not None and m.reaction_key not in only] + matches
