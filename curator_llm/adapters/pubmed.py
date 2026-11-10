"""PubMed E-utilities client: the publication types NLM records for a paper (Journal Article, Review, ...), article
titles, and searches (to find the PubMed ID of an uploaded PDF).

One esummary request covers up to 200 PMIDs. Settings shared with reactome_llm/PubMedFetcher.py: PUBMED_API_KEY,
NCBI_TOOL, NCBI_EMAIL."""
import os
import threading
import time
from typing import Dict, List

import requests

ESUMMARY_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'


_lock = threading.Lock()
_last = [0.0]


def _wait():
    """NCBI allows 3 requests a second without an API key and 10 with one: space requests out, across threads."""
    gap = 0.11 if os.getenv('PUBMED_API_KEY') else 0.34
    with _lock:
        now = time.monotonic()
        if now - _last[0] < gap:
            time.sleep(gap - (now - _last[0]))
        _last[0] = time.monotonic()


class PubMedClient:
    def __init__(self, session=None, timeout: float = 15.0):
        self.http = session or requests
        self.timeout = timeout

    @staticmethod
    def _params(**kw) -> dict:
        params = {'db': 'pubmed', 'retmode': 'json', 'tool': os.getenv('NCBI_TOOL', 'curator-tool-llm'),
                  'email': os.getenv('NCBI_EMAIL', ''), **kw}
        if os.getenv('PUBMED_API_KEY'):
            params['api_key'] = os.environ['PUBMED_API_KEY']
        return params

    def _summaries(self, pmids: List[str]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        ids = [str(p).strip() for p in pmids if str(p).strip()]
        for i in range(0, len(ids), 200):
            _wait()
            r = self.http.get(ESUMMARY_URL, params=self._params(id=','.join(ids[i:i + 200])), timeout=self.timeout)
            r.raise_for_status()
            result = r.json().get('result') or {}
            for uid in result.get('uids', []):
                doc = result.get(uid) or {}
                if not doc.get('error'):
                    out[uid] = doc
        return out

    def publication_types(self, pmids: List[str]) -> Dict[str, List[str]]:
        return {uid: list(doc.get('pubtype') or []) for uid, doc in self._summaries(pmids).items()}

    def titles(self, pmids: List[str]) -> Dict[str, str]:
        return {uid: doc.get('title') or '' for uid, doc in self._summaries(pmids).items()}

    def search(self, term: str, limit: int = 5) -> List[str]:
        _wait()
        r = self.http.get(ESEARCH_URL, params=self._params(term=term, retmax=int(limit)), timeout=self.timeout)
        r.raise_for_status()
        return list((r.json().get('esearchresult') or {}).get('idlist') or [])
