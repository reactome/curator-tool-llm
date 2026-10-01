"""UniProt REST client: validates accessions and finds the reviewed human entry for a gene symbol."""
from typing import Dict, Optional

import requests

BASE = 'https://rest.uniprot.org/uniprotkb'


class RestUniProtClient:
    def __init__(self, session=None, timeout: float = 20.0):
        self.http = session or requests
        self.timeout = timeout
        self._cache: Dict[str, Optional[Dict]] = {}
        self._gene_cache: Dict[tuple, Optional[str]] = {}

    def fetch(self, accession: str) -> Optional[Dict]:
        acc = accession.strip().upper()
        if acc in self._cache:
            return self._cache[acc]
        r = self.http.get(f'{BASE}/{acc}.json', timeout=self.timeout)
        out = None
        if r.status_code == 200:
            j = r.json()
            if j.get('entryType') != 'Inactive':
                genes = [g['geneName']['value'] for g in j.get('genes', []) if 'geneName' in g]
                names = []
                pd = j.get('proteinDescription', {})
                for key in ('recommendedName', 'submissionNames', 'alternativeNames'):
                    v = pd.get(key)
                    for n in ([v] if isinstance(v, dict) else v or []):
                        names.append(n.get('fullName', {}).get('value', ''))
                out = {'accession': j.get('primaryAccession', acc), 'genes': genes, 'names': names,
                       'reviewed': 'reviewed' in j.get('entryType', '').lower() and 'unreviewed' not in j.get('entryType', '').lower(),
                       'organism': j.get('organism', {}).get('scientificName', '')}
        elif r.status_code not in (400, 404, 410):
            r.raise_for_status()
        self._cache[acc] = out
        return out

    def search_gene(self, symbol: str, taxon: int = 9606) -> Optional[str]:
        key = (symbol.strip().lower(), taxon)
        if key not in self._gene_cache:
            self._gene_cache[key] = self._search_gene(symbol, taxon)
        return self._gene_cache[key]

    def _search_gene(self, symbol: str, taxon: int) -> Optional[str]:
        q = f'gene:{symbol} AND organism_id:{taxon} AND reviewed:true'     # gene field includes synonyms (Parkin -> PRKN)
        r = self.http.get(f'{BASE}/search', params={'query': q, 'fields': 'accession', 'size': 2},
                          timeout=self.timeout)
        r.raise_for_status()
        results = r.json().get('results', [])
        return results[0]['primaryAccession'] if len(results) == 1 else None   # ambiguous => no guess
