"""UniProt REST client: validates accessions and finds the reviewed entry for a gene symbol (human, or another taxon)."""
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
        # the gene field includes synonyms (Parkin -> PRKN). Outside human, entries sit under a strain or isolate of the
        # species (KSHV's LANA1 is "Human herpesvirus 8 type P (isolate GK18)"), so search the whole taxon below it;
        # human keeps organism_id so that Neanderthal and Denisovan entries are not counted
        org = f'organism_id:{taxon}' if int(taxon) == 9606 else f'taxonomy_id:{taxon}'
        q = f'gene:{symbol} AND {org} AND reviewed:true'
        r = self.http.get(f'{BASE}/search', params={'query': q, 'fields': 'accession,gene_primary', 'size': 10},
                          timeout=self.timeout)
        r.raise_for_status()
        results = r.json().get('results', [])
        # a symbol is often also another gene's synonym (SP1 is a synonym of DAND5, TRAF2 of TANK): the one entry whose
        # PRIMARY gene name it is wins. Otherwise a single hit (found by a synonym) is used; several is ambiguous.
        primary = [x for x in results if any(g.get('geneName', {}).get('value', '').upper() == symbol.strip().upper()
                                             for g in x.get('genes', []))]
        if len(primary) == 1:
            return primary[0]['primaryAccession']
        return results[0]['primaryAccession'] if len(results) == 1 else None   # ambiguous => no guess
