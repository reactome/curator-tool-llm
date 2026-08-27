"""
Pydantic output models for the Reactome multi-agent annotation pipeline.

Each phase's Task declares one of these models via `output_pydantic=<Model>`. CrewAI
then forces the agent's final answer to match the schema and re-prompts the LLM until
it validates, so the orchestrator can read a typed object off `result.pydantic` instead
of scraping markdown-wrapped JSON out of `result.raw` after the fact.

Design notes:
- Decision-critical fields (overall_score, qa_score, decision, approval_status) are
  REQUIRED so a run can't silently succeed without them.
- Collections default to empty and descriptive strings default to "" so a sparse-but-
  valid agent answer validates on the first try instead of looping on missing optionals.
- `class` is a Python keyword, so the Reactome instance models expose it via the field
  alias "class" (populate_by_name lets CrewAI fill it; dump with by_alias=True to emit it).
"""

from typing import List
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase 1 — Literature extraction
# ---------------------------------------------------------------------------
class Interaction(BaseModel):
    partner: str = Field(..., description="Interacting partner gene/protein symbol")
    interaction_type: str = Field("", description="binding / regulation / phosphorylation / etc.")
    evidence: str = Field("", description="Experimental method or evidence description")
    confidence: str = Field("", description="Qualitative confidence: high / medium / low")
    evidence_strength_score: float = Field(0.0, description="Numeric evidence strength, 0.0-1.0")
    pmid: str = Field("", description="PubMed ID supporting this interaction")
    context: str = Field("", description="Brief description / biological context")


class PathwayInvolvement(BaseModel):
    pathway_name: str = Field(..., description="Pathway name or description")
    role: str = Field("", description="catalyst / regulator / target / etc.")
    evidence: str = Field("", description="Supporting evidence")
    confidence: str = Field("", description="Qualitative confidence: high / medium / low")
    evidence_strength_score: float = Field(0.0, description="Numeric evidence strength, 0.0-1.0")
    pmid: str = Field("", description="PubMed ID supporting this pathway role")


class GeneFunction(BaseModel):
    function: str = Field(..., description="Molecular function description")
    evidence: str = Field("", description="Experimental support")
    confidence: str = Field("", description="Qualitative confidence: high / medium / low")
    evidence_strength_score: float = Field(0.0, description="Numeric evidence strength, 0.0-1.0")
    pmid: str = Field("", description="PubMed ID supporting this function")


class LiteratureExtraction(BaseModel):
    """Phase 1 output: structured molecular evidence pulled from the literature."""
    gene: str = Field(..., description="Target gene symbol")
    interactions: List[Interaction] = Field(default_factory=list)
    pathways: List[PathwayInvolvement] = Field(default_factory=list)
    functions: List[GeneFunction] = Field(default_factory=list)
    summary: str = Field("", description="Concise functional summary of the gene")


# ---------------------------------------------------------------------------
# Phase 2 — Reactome data model creation
# ---------------------------------------------------------------------------
class ReactomeEntity(BaseModel):
    model_config = {"populate_by_name": True}
    cls: str = Field("EntityWithAccessionedSequence", alias="class", description="Reactome class name")
    displayName: str = Field(..., description="Human-readable entity name")
    identifier: str = Field("", description="UniProt accession (use the verified one)")
    species: str = Field("Homo sapiens", description="Species name")
    referenceEntity: str = Field("", description="Reference entity details")
    compartment: str = Field("", description="Subcellular compartment, only if explicitly stated")


class ReactomeComplex(BaseModel):
    model_config = {"populate_by_name": True}
    cls: str = Field("Complex", alias="class", description="Reactome class name")
    displayName: str = Field(..., description="Human-readable complex name")
    components: List[str] = Field(default_factory=list, description="Member entity names/ids")
    literatureReference: List[str] = Field(default_factory=list, description="Supporting PMIDs")


class ReactomeReaction(BaseModel):
    model_config = {"populate_by_name": True}
    cls: str = Field("Reaction", alias="class", description="Reactome class name")
    displayName: str = Field(..., description="Human-readable reaction name")
    reactionType: str = Field("", description="transition / binding / dissociation / omitted / blackBoxEvent")
    input: List[str] = Field(default_factory=list, description="Input entity names/ids")
    output: List[str] = Field(default_factory=list, description="Output entity names/ids")
    catalystActivity: List[str] = Field(default_factory=list, description="Catalyst entity names/ids")
    regulatedBy: List[str] = Field(default_factory=list,
                                   description="Regulators, each 'regulationType: regulator (note)'")
    compartment: str = Field("", description="Subcellular compartment, only if explicitly stated")
    inferredFrom: List[str] = Field(default_factory=list, description="Orthologous events")
    summation: str = Field("", description="Factual description of the molecular event")
    evidence: List[str] = Field(default_factory=list,
                                description="Verbatim source excerpts supporting this reaction")
    literatureReference: List[str] = Field(default_factory=list, description="Supporting PMIDs")
    confidence: float = Field(0.0, description="Extractor's self-assessed confidence, 0.0-1.0")
    provenance: str = Field("", description="Evidence source for this reaction: fulltext / abstract")


class ReactomePathway(BaseModel):
    model_config = {"populate_by_name": True}
    cls: str = Field("Pathway", alias="class", description="Reactome class name")
    displayName: str = Field(..., description="Human-readable pathway name")
    hasEvent: List[str] = Field(default_factory=list, description="Contained reaction names/ids")
    summation: str = Field("", description="Pathway description")
    literatureReference: List[str] = Field(default_factory=list, description="Supporting PMIDs")


class ReactomeDataModel(BaseModel):
    """Phase 2 output: Reactome instance JSON for the gene."""
    gene: str = Field(..., description="Target gene symbol")
    entities: List[ReactomeEntity] = Field(default_factory=list)
    complexes: List[ReactomeComplex] = Field(default_factory=list)
    reactions: List[ReactomeReaction] = Field(default_factory=list)
    pathways: List[ReactomePathway] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 3 — Domain expert review
# ---------------------------------------------------------------------------
class CriterionScores(BaseModel):
    biological_accuracy: float = Field(0.0, description="0.0-1.0")
    evidence_support: float = Field(0.0, description="0.0-1.0")
    mechanistic_consistency: float = Field(0.0, description="0.0-1.0")
    integration_quality: float = Field(0.0, description="0.0-1.0")


class InstanceReview(BaseModel):
    instance_id: str = Field("", description="Entity/reaction/pathway id reviewed")
    instance_type: str = Field("", description="Entity / Reaction / Pathway")
    score: float = Field(0.0, description="Per-instance quality score, 0.0-1.0")
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    evidence_assessment: str = Field("", description="strong / moderate / weak")


class ExpertReview(BaseModel):
    """Phase 3 output: domain expert validation of the generated instances."""
    gene: str = Field(..., description="Target gene symbol")
    overall_score: float = Field(..., description="Overall quality score, 0.0-1.0")
    criterion_scores: CriterionScores = Field(default_factory=CriterionScores)
    instance_reviews: List[InstanceReview] = Field(default_factory=list)
    summary: str = Field("", description="Overall assessment summary")
    recommendations: List[str] = Field(default_factory=list)
    approval_status: str = Field(..., description="approved / requires_revision / rejected")


# ---------------------------------------------------------------------------
# Phase 4 — Quality assurance
# ---------------------------------------------------------------------------
class TechnicalIssue(BaseModel):
    severity: str = Field("", description="high / medium / low")
    category: str = Field("", description="schema / integrity / consistency / integration")
    description: str = Field("", description="Issue description")
    location: str = Field("", description="Specific instance or field")
    resolution: str = Field("", description="Recommended fix")


class IntegrationAssessment(BaseModel):
    conflicts_detected: bool = Field(False, description="Whether conflicts with existing data were found")
    performance_impact: str = Field("", description="minimal / moderate / significant")
    compatibility_score: float = Field(0.0, description="0.0-1.0")


class InstanceVerdict(BaseModel):
    """Per-instance QA judgement. Only instances that are NOT curator-ready are listed here;
    every instance not named is implicitly 'good'. This is what keeps a handful of bad
    instances from masking the (usually many) good ones behind a single overall qa_score."""
    instance: str = Field("", description="Exact displayName of the instance being judged")
    instance_class: str = Field("", description="EWAS / Complex / Reaction / Pathway")
    verdict: str = Field("", description="needs_revision or bad")
    reason: str = Field("", description="One-line reason it is not curator-ready")


class QAReport(BaseModel):
    """Phase 4 output: technical QA and integration assessment."""
    gene: str = Field(..., description="Target gene symbol")
    qa_score: float = Field(..., description="Overall QA score, 0.0-1.0")
    technical_issues: List[TechnicalIssue] = Field(default_factory=list)
    integration_assessment: IntegrationAssessment = Field(default_factory=IntegrationAssessment)
    flagged_instances: List[InstanceVerdict] = Field(
        default_factory=list,
        description="Instances that are NOT curator-ready (needs_revision or bad), by exact "
                    "displayName. Everything not listed is treated as good.")


# ---------------------------------------------------------------------------
# Phase 5 — Consensus vote + final synthesis
# ---------------------------------------------------------------------------
class AgentVote(BaseModel):
    """One specialist's vote in the final virtual meeting."""
    agent_role: str = Field(..., description="Role casting the vote")
    decision: str = Field(..., description="approve / requires_revision / reject")
    confidence: float = Field(0.0, description="0.0-1.0")
    blocking_issues: List[str] = Field(default_factory=list)
    required_revisions: List[str] = Field(default_factory=list)
    summary: str = Field("", description="Short rationale")


class VoteTally(BaseModel):
    approve: int = 0
    requires_revision: int = 0
    reject: int = 0


class ConsensusDecision(BaseModel):
    """Phase 5 output: the chaired final synthesis across all votes."""
    decision: str = Field(..., description="approve / requires_revision / reject")
    confidence: float = Field(0.0, description="Average vote confidence, 0.0-1.0")
    vote_tally: VoteTally = Field(default_factory=VoteTally)
    required_revisions: List[str] = Field(default_factory=list)
    blocking_issues: List[str] = Field(default_factory=list)
    summary: str = Field("", description="Short final rationale")
    quality_threshold: float = Field(0.7, description="Threshold applied in this run")
