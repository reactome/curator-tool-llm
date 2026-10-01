from dataclasses import dataclass, field
from typing import List, Protocol


@dataclass
class EventRow:
    """An existing Reactome reaction-like event with the UniProt accessions of its participants
    (complex components and set members flattened; small molecules have no accession)."""
    db_id: int
    display_name: str
    st_id: str = ''
    schema_class: str = ''
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)


class EventLookup(Protocol):
    def events_citing(self, pmid: str) -> List[EventRow]:
        """Events whose literatureReference is this PMID: is the paper already curated?"""
        ...

    def candidate_events(self, catalyst_accessions: List[str], participant_accessions: List[str],
                         limit: int = 100) -> List[EventRow]:
        """Events catalysed by one of the accessions, or (when none given) sharing the most participants."""
        ...
