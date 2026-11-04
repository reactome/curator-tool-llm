"""Owns the life of an annotation session: start it (background job), edit it through proposals, and
(re)emit instances and issues whenever the draft changes. All writes to the store go through here."""
import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Callable, Dict, List, Optional, Tuple

from curator_llm.models.evidence import ClaimOrigin, Evidence, Verification
from curator_llm.models.session import ChangeLogEntry, Issue, PaperInfo, Proposal, Session
from curator_llm.models.usage import UsageEntry
from curator_llm.ports.jobs import JobRunner
from curator_llm.ports.pipeline import PaperRef, Pipeline, PipelineInput
from curator_llm.ports.sessions import SessionStore
from curator_llm.services import issues as iss
from curator_llm.services.emitter import emit_user_instances
from curator_llm.services.evidence_store import EvidenceStore
from curator_llm.services.existing_events import find_existing
from curator_llm.services.paper_text import PaperText
from curator_llm.services.patching import PatchError, apply_patch, describe_changes, touched_reactions
from curator_llm.services.qa import AUTO_RULE_CODES, auto_findings, qa_reaction, rank_results
from curator_llm.services.suggestions import suggest_fixes as make_suggestions


class ProposalError(ValueError):
    """An edit, check or decision that cannot be done; the message is safe to show to the curator or the model."""


_CLAIM_FIELDS = {'inputs', 'outputs', 'catalyst', 'regulations', 'preceding'}
_locks_guard = threading.Lock()
_locks: Dict[str, threading.Lock] = {}


logger = logging.getLogger(__name__)


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
                 resolver_factory: Optional[Callable] = None, events=None, qa_model=None, auto_qa: bool = False,
                 qa_workers: int = 8):
        self.store, self.jobs, self.pipeline = store, jobs, pipeline
        self.resolver_factory = resolver_factory     # () -> Resolver, to resolve entities an edit introduces
        self.events = events                         # EventLookup, to check for existing Reactome events
        # with auto_qa, every reaction of a new draft gets the model's QA review before the session is ready (not for a
        # replayed run, which promises no model calls); the curator can always run it again (POST .../qa)
        self.qa_model, self.auto_qa, self.qa_workers = qa_model, auto_qa, qa_workers

    def _ready(self, owner: str, session_id: str) -> Session:
        s = self.store.get(session_id, owner)
        if s is None or s.draft is None:
            raise ProposalError('session is not ready')
        return s

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self, owner: str, spec: PipelineInput, source_label: Optional[str] = None) -> Tuple[str, str]:
        """Create the session, queue the pipeline, return (session_id, job_id)."""
        refs = spec.refs()
        papers = [PaperInfo(source=r.source, kind='pmid' if r.pmid else 'pdf', pmid=r.pmid,
                            name=r.label or (r.pmid if r.pmid else r.source), review=r.review) for r in refs]
        first_pmid = next((r.pmid for r in refs if r.pmid), None)
        s = Session(owner=owner, pmid=first_pmid, focus=spec.focus, papers=papers,
                    source=source_label or self._source_label(papers))
        self.store.save(s)
        job_id = self.jobs.submit(owner, lambda report: self._run(s.id, owner, spec, report))
        return s.id, job_id

    @staticmethod
    def _source_label(papers: List[PaperInfo]) -> str:
        """What the session list shows: the paper itself, or the first few of several."""
        if len(papers) == 1:
            return papers[0].name
        shown = ', '.join(p.name for p in papers[:3])
        return f'{len(papers)} papers: {shown}' + (', ...' if len(papers) > 3 else '')

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
            s.paper_texts = dict(result.papers)
            for p in s.papers:                                 # review or research, as decided for the run
                if p.source in result.reviews:
                    p.review = result.reviews[p.source]
            s.existing, s.issues = list(result.existing), list(result.issues)
            s.usage, s.usage_source = list(result.usage), 'saved' if result.replayed else 'run'
            self.refresh(s)
            if self.auto_qa and self.qa_model is not None and not result.replayed and s.draft.reactions:
                progress(f'checking each reaction ({len(s.draft.reactions)})')
                self.qa_into(s, self.qa_model, self.qa_workers)
            self.refresh_suggestions(s)
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
    def reviews_of(s: Session) -> List[str]:
        """Source tags of the session's papers that are review articles (their cited statements are their evidence)."""
        return [p.source for p in s.papers if p.review]

    @staticmethod
    def qa_into(s: Session, model, workers: int = 8) -> list:
        """QA every reaction of `s` (model calls concurrent) and put the findings into its issues; `s` is not saved here.
        Returns the results, most urgent first."""
        reviews = SessionService.reviews_of(s)
        keys = [r.key for r in s.draft.reactions]
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(keys) or 1))) as pool:
            results = list(pool.map(lambda k: qa_reaction(s.draft, k, s.evidence, model, reviews), keys))
        for res in results:
            SessionService._merge_qa(s, res, model)
        return rank_results(results)

    @staticmethod
    def refresh(s: Session) -> None:
        """Re-emit instances from the draft and rebuild the issue list. Emitter issues are replaced on every
        refresh; issues that recur keep the id and status they had."""
        key = lambda i: (i.code, i.message, i.reaction_key, i.participant_key)
        kept = [i for i in s.issues if i.source not in ('emitter', 'checks')]
        emit = emit_user_instances(s.draft, {e.id: e for e in s.evidence})
        s.user_instances, s.evidence_links, s.key_to_db_id = emit.user_instances, emit.evidence_links, emit.key_to_db_id
        have = {(i.code, i.participant_key) for i in kept}
        merged = kept + [f for f in iss.issues_from_draft(s.draft) if (f.code, f.participant_key) not in have]
        merged += iss.issues_from_notes('emitter', emit.warnings)
        merged += [Issue(source='checks', severity=f.severity, code=f.code, message=f.message, reaction_key=k)
                   for k, f in auto_findings(s.draft, s.evidence, SessionService.reviews_of(s))]
        iss.link_to_instances(merged, emit.key_to_db_id, emit.participant_to_db_id)
        _carry_over(s.issues, merged, key)
        s.issues = iss.assign_ids(merged)

    # ── proposals ──────────────────────────────────────────────────────────
    @staticmethod
    def _paper(s: Session) -> Optional[PaperText]:
        """The session's first paper."""
        return PaperText.from_dict(s.paper) if s.paper else None

    @staticmethod
    def papers_of(s: Session) -> Dict[str, PaperText]:
        """Every paper of the session by source tag, in order. A session made before several papers were supported
        has only `paper`; it is returned under its source label."""
        if s.paper_texts:
            return {src: PaperText.from_dict(d) for src, d in s.paper_texts.items()}
        if s.paper:
            return {(s.papers[0].source if s.papers else (s.source or 'paper')): PaperText.from_dict(s.paper)}
        return {}

    @staticmethod
    def pmids_of(s: Session) -> List[str]:
        return [p.pmid for p in s.papers if p.pmid] or ([s.pmid] if s.pmid else [])

    def search_papers(self, s: Session, query: str, section: Optional[str] = None, limit: int = 5,
                      source: Optional[str] = None) -> List[dict]:
        """Passages matching `query` in the session's papers (one paper if `source` is given), best first, each with the
        paper it is from."""
        hits = []
        for src, paper in self.papers_of(s).items():
            if source and src != source:
                continue
            for h in paper.search(query, section, top_k=limit):
                hits.append({'text': h.text, 'section': h.section, 'page': h.page, 'figure': h.figure,
                             'charSpan': h.char_span, 'score': h.score, 'source': src})
        hits.sort(key=lambda h: -(h['score'] or 0))
        return hits[:limit]

    def _verify_evidence(self, s: Session, drafts: List[dict]) -> List[Evidence]:
        """Quotes checked against the session's papers; a curator assertion needs no paper. A quote is found in whichever
        paper has it (`source` on the quote names one to try first). Raises on any quote not found."""
        papers, verified, failed = self.papers_of(s), [], []
        for raw in drafts:
            ev = Evidence(**raw)
            if ev.claim_origin == ClaimOrigin.CURATOR_ASSERTION:
                verified.append(ev)
            elif not papers:
                raise ProposalError('this session has no paper text to verify quotes against')
            else:
                order = ([ev.source] if ev.source in papers else []) + [k for k in papers if k != ev.source]
                found = None
                for src in order:
                    checked = papers[src].verify(ev)
                    if checked.verified != Verification.FAILED:
                        found = checked.model_copy(update={'source': src})
                        break
                (verified if found is not None else failed).append(found if found is not None else ev)
        if failed:
            raise ProposalError('these quotes were not found in the paper' + ('s' if len(papers) > 1 else '') +
                                ' (quote exactly, or use curator_assertion): ' + ' | '.join(f'"{f.quote[:100]}"' for f in failed))
        return verified

    def _build_proposal(self, s: Session, ops: List[dict], reason: str, evidence: Optional[List[dict]], actor: str,
                        curator_assertion: bool = False, skip_evidence_check: bool = False) -> Proposal:
        """Validate an edit against the session's draft and describe it. Not stored. `skip_evidence_check` is for the
        suggestions the service makes itself: the curator who accepts one is the one who decides it."""
        try:
            after = apply_patch(s.draft, ops)
        except PatchError as e:
            raise ProposalError(str(e)) from e
        verified = self._verify_evidence(s, evidence or [])
        if curator_assertion and not verified:
            verified.append(Evidence(quote=reason or 'curator assertion', claim_origin=ClaimOrigin.CURATOR_ASSERTION))
        if _needs_evidence(ops) and not verified and not skip_evidence_check:
            raise ProposalError('this edit changes what a reaction is, so it needs at least one verified '
                                'quote from the paper, or curator_assertion=true')
        after_keys = {r.key for r in after.reactions}
        p = Proposal(id=f'p-{1 + max([int(x.id.split("-")[1]) for x in s.proposals] or [0]):03d}',
                     reason=reason, ops=ops, summary=describe_changes(s.draft, after),
                     reaction_keys=[k for k in touched_reactions(s.draft, after) if k in after_keys],
                     evidence=verified, actor=actor)
        if not p.summary:
            raise ProposalError('this edit changes nothing')
        return p

    @_locked
    def propose(self, owner: str, session_id: str, ops: List[dict], reason: str = '',
                evidence: Optional[List[dict]] = None, actor: str = 'chat',
                curator_assertion: bool = False) -> Proposal:
        """Validate an edit and hold it for review. Nothing changes until it is accepted."""
        s = self._ready(owner, session_id)
        p = self._build_proposal(s, ops, reason, evidence, actor, curator_assertion)
        s.proposals.append(p)
        self.store.save(s)
        return p

    SUGGESTER = 'suggestions'

    def refresh_suggestions(self, s: Session) -> None:
        """Make the suggested fixes for the draft as it is now, as pending proposals (actor 'suggestions'). A pending one
        that no longer applies, or no longer fixes anything, is dropped; one the curator already accepted or rejected
        is not made again."""
        wanted = make_suggestions(s.draft, s.evidence, self.reviews_of(s))
        same = lambda p, ops: p.ops == ops
        s.proposals = [p for p in s.proposals
                       if not (p.actor == self.SUGGESTER and p.status == 'pending' and not any(same(p, w.ops) for w in wanted))]
        for w in wanted:
            if any(same(p, w.ops) and p.actor == self.SUGGESTER and p.status in ('pending', 'accepted', 'rejected')
                   for p in s.proposals):
                continue
            try:
                s.proposals.append(self._build_proposal(s, w.ops, w.reason, None, self.SUGGESTER, skip_evidence_check=True))
            except ProposalError as e:
                logger.info('suggestion %s for %s not made: %s', w.code, w.reaction_key, e)

    @_locked
    def suggest_fixes(self, owner: str, session_id: str) -> List[Proposal]:
        """(Re)make the suggested fixes and return the pending ones."""
        s = self._ready(owner, session_id)
        self.refresh_suggestions(s)
        self.store.save(s)
        return [p for p in s.proposals if p.actor == self.SUGGESTER and p.status == 'pending']

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
        evidence = EvidenceStore.load(s.evidence)         # already verified, against whichever paper each quote is from
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
        self.refresh_suggestions(s)
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
    def run_qa(self, owner: str, session_id: str, reaction_key: str, model=None):
        """QA one reaction. Its previous QA issues are replaced (ids and statuses of recurring ones kept).
        The model call runs outside the session lock, so chat and the UI are not held up while it waits."""
        s = self._ready(owner, session_id)
        try:
            result = qa_reaction(s.draft, reaction_key, s.evidence, model, self.reviews_of(s))
        except KeyError:
            raise ProposalError(f'no reaction with key {reaction_key!r}')
        return self._apply_qa(owner, session_id, result, model)

    def run_qa_all(self, owner: str, session_id: str, model=None, workers: int = 8):
        """QA every reaction (the model calls run concurrently), record the findings as issues, and return the
        results most urgent first."""
        s = self._ready(owner, session_id)
        draft, evidence, reviews = s.draft, s.evidence, self.reviews_of(s)
        keys = [r.key for r in draft.reactions]
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(keys) or 1))) as pool:
            results = list(pool.map(lambda k: qa_reaction(draft, k, evidence, model, reviews), keys))
        applied = []
        for res in results:
            try:
                applied.append(self._apply_qa(owner, session_id, res, model))
            except ProposalError:                # the reaction was removed by an edit while the checks ran
                continue
        return rank_results(applied)

    @_locked
    def _apply_qa(self, owner: str, session_id: str, result, model=None):
        s = self._ready(owner, session_id)
        if not any(r.key == result.reaction_key for r in s.draft.reactions):
            raise ProposalError(f'no reaction with key {result.reaction_key!r}')
        self._merge_qa(s, result, model, actor=owner)
        self.store.save(s)
        return result

    @staticmethod
    def _merge_qa(s: Session, result, model=None, actor: str = 'pipeline') -> None:
        """Replace the reaction's QA issues with this result's (ids and statuses of recurring ones kept)."""
        reaction_key = result.reaction_key
        mine = lambda i: i.source == 'qa' and i.reaction_key == reaction_key
        fresh = [Issue(source='qa', severity=f.severity, code=f.code, message=f.message, reaction_key=reaction_key,
                       instance_db_id=s.key_to_db_id.get(reaction_key))
                 for f in result.findings if f.code not in AUTO_RULE_CODES]   # those are 'checks' issues already
        _carry_over([i for i in s.issues if mine(i)], fresh, lambda i: (i.code, i.message, i.reaction_key))
        s.issues = iss.assign_ids([i for i in s.issues if not mine(i)] + fresh)
        if result.usage:
            s.usage.append(UsageEntry(step='qa', calls=1, detail=reaction_key, model=getattr(model, 'model', None),
                                      **result.usage))
        s.change_log.append(ChangeLogEntry(actor=actor, action=f'qa {reaction_key}', detail=result.verdict))

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
        matches, fresh = find_existing(s.draft, self.pmids_of(s) or None, self.events, only)
        in_scope = lambda i: only is None or i.reaction_key in only
        stale = [i for i in s.issues if i.source == 'existing' and in_scope(i)]
        _carry_over(stale, fresh, lambda i: (i.code, i.message, i.reaction_key, i.participant_key))
        iss.link_to_instances(fresh, {k: v for k, v in s.key_to_db_id.items() if v is not None})
        s.issues = iss.assign_ids([i for i in s.issues if i not in stale] + fresh)
        s.existing = [m for m in s.existing if only is not None and m.reaction_key not in only] + matches
