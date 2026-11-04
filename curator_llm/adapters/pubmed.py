"""PubMed E-utilities client: the publication types NLM records for a paper (Journal Article, Review, ...).

One esummary request covers up to 200 PMIDs. Settings shared with reactome_llm/PubMedFetcher.py: PUBMED_API_KEY,
NCBI_TOOL, NCBI_EMAIL."""
import os
from typing import Dict, List

import requests

ESUMMARY_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'


class PubMedClient:
    def __init__(self, session=None, timeout: float = 15.0):
        self.http = session or requests
        self.timeout = timeout

    def publication_types(self, pmids: List[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        ids = [str(p).strip() for p in pmids if str(p).strip()]
        for i in range(0, len(ids), 200):
            params = {'db': 'pubmed', 'retmode': 'json', 'id': ','.join(ids[i:i + 200]),
                      'tool': os.getenv('NCBI_TOOL', 'curator-tool-llm'), 'email': os.getenv('NCBI_EMAIL', '')}
            if os.getenv('PUBMED_API_KEY'):
                params['api_key'] = os.environ['PUBMED_API_KEY']
            r = self.http.get(ESUMMARY_URL, params=params, timeout=self.timeout)
            r.raise_for_status()
            result = r.json().get('result') or {}
            for uid in result.get('uids', []):
                doc = result.get(uid) or {}
                if not doc.get('error'):
                    out[uid] = list(doc.get('pubtype') or [])
        return out
