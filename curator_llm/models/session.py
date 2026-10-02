"""Annotation session: one paper's draft, evidence, issues and emitted instances, owned by a curator."""
import time
import uuid
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import ReactomeDraft
from curator_llm.models.usage import UsageEntry


class Issue(BaseModel):
    """Something a curator must look at. Every note the pipeline produces becomes one of these, so the
    frontend can list them all in one place instead of burying them in logs."""
    id: str = ''
    source: Literal['evidence', 'builder', 'resolver', 'emitter', 'pipeline', 'existing', 'qa']
    severity: Literal['info', 'warning', 'action'] = 'warning'
    code: str                                  # stable machine-readable kind, e.g. 'needs_resolution:uniprot'
    message: str
    reaction_key: Optional[str] = None
    participant_key: Optional[str] = None
    instance_db_id: Optional[int] = None       # filled once instances are emitted
    status: Literal['open', 'resolved', 'dismissed'] = 'open'


class Proposal(BaseModel):
    """A pending edit. Nothing changes until a curator accepts it."""
    id: str = ''
    reason: str = ''
    ops: List[dict] = Field(default_factory=list)           # RFC 6902 patch against the draft
    summary: List[str] = Field(default_factory=list)        # human-readable diff
    reaction_keys: List[str] = Field(default_factory=list)  # reactions the edit touches
    evidence: List[Evidence] = Field(default_factory=list)  # quotes already verified against the paper
    actor: str = 'chat'
    status: Literal['pending', 'accepted', 'rejected', 'stale'] = 'pending'
    created: float = Field(default_factory=time.time)
    decided_by: Optional[str] = None
    decided_at: Optional[float] = None


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    text: str
    at: float = Field(default_factory=time.time)
    proposal_ids: List[str] = Field(default_factory=list)


class ExistingMatch(BaseModel):
    """A draft reaction that looks like one Reactome already has."""
    reaction_key: str
    db_id: int
    display_name: str
    st_id: str = ''
    level: Literal['same', 'similar']
    score: float
    cites_pmid: bool = False
    catalyst_match: bool = False
    overlap: float = 0.0
    similarity: float = 0.0
    reasons: List[str] = Field(default_factory=list)


class ChangeLogEntry(BaseModel):
    at: float = Field(default_factory=time.time)
    actor: str
    action: str
    detail: str = ''


class Session(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    owner: str
    pmid: Optional[str] = None
    source: str = ''                           # pmid or uploaded file name
    focus: Optional[str] = None
    status: Literal['queued', 'running', 'ready', 'failed'] = 'queued'
    progress: str = ''
    error: Optional[str] = None
    created: float = Field(default_factory=time.time)
    updated: float = Field(default_factory=time.time)
    draft: Optional[ReactomeDraft] = None
    evidence: List[Evidence] = Field(default_factory=list)
    issues: List[Issue] = Field(default_factory=list)
    user_instances: Optional[dict] = None
    evidence_links: List[dict] = Field(default_factory=list)
    key_to_db_id: Dict[str, int] = Field(default_factory=dict)
    change_log: List[ChangeLogEntry] = Field(default_factory=list)
    paper: Optional[dict] = None                # PaperText.to_dict(): for search and quote checks
    proposals: List[Proposal] = Field(default_factory=list)
    existing: List[ExistingMatch] = Field(default_factory=list)
    usage: List[UsageEntry] = Field(default_factory=list)
    # 'saved' when the result was replayed from a snapshot: the entries are from the original run, nothing was spent now
    usage_source: Literal['run', 'saved'] = 'run'
    chat: List[ChatMessage] = Field(default_factory=list)


class SessionSummary(BaseModel):
    id: str
    pmid: Optional[str]
    source: str
    status: str
    progress: str
    created: float
    updated: float
    n_reactions: int = 0
    n_open_issues: int = 0

    @classmethod
    def of(cls, s: Session) -> 'SessionSummary':
        return cls(id=s.id, pmid=s.pmid, source=s.source, status=s.status, progress=s.progress,
                   created=s.created, updated=s.updated,
                   n_reactions=len(s.draft.reactions) if s.draft else 0,
                   n_open_issues=sum(i.status == 'open' for i in s.issues))
