from typing import Dict, List, Optional, Protocol


class UniProtClient(Protocol):
    def fetch(self, accession: str) -> Optional[Dict]:
        """{accession, genes: [..], names: [..], reviewed: bool, organism: str} or None if unknown/obsolete."""
        ...

    def search_gene(self, symbol: str, taxon: int = 9606) -> Optional[str]:
        """Accession of the single reviewed entry whose gene name or synonym matches `symbol`, or None."""
        ...


class OntologyClient(Protocol):
    def search(self, name: str, ontology: str) -> List[Dict]:
        """Exact-label hits as [{identifier: 'GO:0004672', label: str}], best first."""
        ...


class PublicationTypes(Protocol):
    def publication_types(self, pmids: List[str]) -> Dict[str, List[str]]:
        """PubMed's publication types per PMID, e.g. {'25670789': ['Journal Article', 'Review']}. A PMID PubMed does not
        know is absent. Raises when PubMed cannot be asked (network, HTTP error)."""
        ...
