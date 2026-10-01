"""
Pydantic output models for the Reactome annotation pipeline: the instance data model built by
reaction_to_instances.build_instances and the QA report produced by ReactomeQA (both used via
LangChain `with_structured_output`).

Design notes:
- Decision-critical fields (overall_score, qa_score, decision, approval_status) are
  REQUIRED so a run can't silently succeed without them.
- Collections default to empty and descriptive strings default to "" so a sparse-but-
  valid agent answer validates on the first try instead of looping on missing optionals.
- `class` is a Python keyword, so the Reactome instance models expose it via the field
  alias "class" (populate_by_name lets the model fill it; dump with by_alias=True to emit it).
"""

from typing import List
from pydantic import BaseModel, Field


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
                                description="Evidence ids (ev-001, ...) supporting this reaction; quotes live in the evidence store")
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


