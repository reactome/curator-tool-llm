"""Typed draft of a Reactome annotation: what the emitter turns into curator-tool instances.

Participants are typed and referenced by `key` (a local handle), never by free-text name.
Anything already in gk_central is an ExistingRef (dbId + class), so the draft never invents an
id: unresolved things stay as new instances with `needs_resolution` set for the curator.
"""
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ExistingRef(BaseModel):
    """Pointer to an instance that already exists in gk_central."""
    db_id: int
    display_name: str
    schema_class: str


class ModifiedResidueSpec(BaseModel):
    psi_mod: Optional[str] = None            # e.g. "MOD:00046" (O-phospho-L-serine)
    mod_label: Optional[str] = None          # display label, e.g. "O-phospho-L-serine"
    short: str = 'p'                         # prefix used in display names ("p" -> p-S65-Ub)
    residue: Optional[str] = None            # one-letter residue, e.g. "S"
    coordinate: Optional[int] = None         # position in the reference sequence
    psi_mod_ref: Optional[ExistingRef] = None  # resolved PSI-MOD instance, if any


class _Participant(BaseModel):
    key: str
    name: str
    compartment: Optional[ExistingRef] = None
    compartment_name: Optional[str] = None   # kept so an unresolved compartment is not silently lost
    needs_resolution: List[str] = Field(default_factory=list)   # e.g. ["uniprot", "compartment"]
    existing: Optional[ExistingRef] = None   # set when this exact entity already exists: emit a shell


class EwasSpec(_Participant):
    kind: Literal['ewas'] = 'ewas'
    uniprot: Optional[str] = None
    reference_entity: Optional[ExistingRef] = None   # existing ReferenceGeneProduct
    species: str = 'Homo sapiens'
    start_coordinate: Optional[int] = None
    end_coordinate: Optional[int] = None
    modifications: List[ModifiedResidueSpec] = Field(default_factory=list)


class SimpleEntitySpec(_Participant):
    kind: Literal['simple'] = 'simple'
    chebi: Optional[str] = None                      # "CHEBI:30616"
    reference_entity: Optional[ExistingRef] = None   # existing ReferenceMolecule


class ComplexSpec(_Participant):
    kind: Literal['complex'] = 'complex'
    components: List[str] = Field(default_factory=list)   # participant keys


class DefinedSetSpec(_Participant):
    kind: Literal['set'] = 'set'
    members: List[str] = Field(default_factory=list)


Participant = Union[EwasSpec, SimpleEntitySpec, ComplexSpec, DefinedSetSpec]


class GoActivity(BaseModel):
    identifier: Optional[str] = None         # "GO:0004672"
    name: Optional[str] = None
    ref: Optional[ExistingRef] = None


class CatalystSpec(BaseModel):
    entity: str                              # participant key
    activity: Optional[GoActivity] = None


class RegulationSpec(BaseModel):
    kind: Literal['positive', 'negative', 'requirement']
    regulator: str                           # participant key
    note: Optional[str] = None               # condition-dependent flag for the curator


class ReactionSpec(BaseModel):
    key: str
    name: str
    reaction_type: str = 'transition'        # transition | binding | dissociation | omitted | blackBoxEvent
    inputs: List[str] = Field(default_factory=list)      # participant keys
    outputs: List[str] = Field(default_factory=list)
    catalyst: Optional[CatalystSpec] = None
    regulations: List[RegulationSpec] = Field(default_factory=list)
    compartment: Optional[ExistingRef] = None
    summation: str = ''
    pmids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    preceding: List[str] = Field(default_factory=list)   # reaction keys this one follows
    inferred_from: List[ExistingRef] = Field(default_factory=list)
    species: str = 'Homo sapiens'
    existing: Optional[ExistingRef] = None   # set when Reactome already has this reaction


class PathwaySpec(BaseModel):
    name: str
    existing: Optional[ExistingRef] = None
    reactions: List[str] = Field(default_factory=list)   # reaction keys
    summation: str = ''
    pmids: List[str] = Field(default_factory=list)


class ReactomeDraft(BaseModel):
    participants: Dict[str, Participant] = Field(default_factory=dict)
    reactions: List[ReactionSpec] = Field(default_factory=list)
    pathway: Optional[PathwaySpec] = None
    # existing instances for shared references, filled by resolvers (key: stable string)
    publications: Dict[str, ExistingRef] = Field(default_factory=dict)   # PMID -> existing LiteratureReference
