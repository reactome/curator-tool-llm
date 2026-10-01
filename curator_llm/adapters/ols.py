"""EBI OLS4 client for exact-label ontology lookups (GO, ChEBI, PSI-MOD)."""
from typing import Dict, List

import requests

URL = 'https://www.ebi.ac.uk/ols4/api/search'


class OlsClient:
    def __init__(self, session=None, timeout: float = 20.0):
        self.http = session or requests
        self.timeout = timeout

    def search(self, name: str, ontology: str) -> List[Dict]:
        r = self.http.get(URL, params={'q': name, 'ontology': ontology, 'exact': 'true',
                                       'queryFields': 'label,synonym', 'rows': 5, 'obsoletes': 'false'},
                          timeout=self.timeout)
        r.raise_for_status()
        out = []
        for d in r.json().get('response', {}).get('docs', []):
            ident = d.get('obo_id') or d.get('short_form', '').replace('_', ':')
            if ident:
                out.append({'identifier': ident, 'label': d.get('label', '')})
        return out
