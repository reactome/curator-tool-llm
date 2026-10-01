from typing import Dict, List, Optional

from curator_llm.ports.auth import AuthError, AuthUser, ForbiddenError
from curator_llm.ports.jobs import Job


class FakeAuthProvider:
    """tokens: {token: (username, role)}; anything else is rejected."""

    def __init__(self, tokens: Dict[str, tuple]):
        self.tokens = tokens

    def verify(self, token: str) -> AuthUser:
        if token not in self.tokens:
            raise AuthError('bad token')
        username, role = self.tokens[token]
        if role.lower() != 'curator':
            raise ForbiddenError('not a curator')
        return AuthUser(username, role)


class FakeInstanceLookup:
    """instances: dicts with dbId, displayName, schemaClassName and any searchable attributes."""

    def __init__(self, instances: Optional[List[Dict]] = None):
        self.instances = instances or []
        self.calls = []

    def find_by_display_name(self, display_name, class_names):
        self.calls.append(('name', display_name, tuple(class_names)))
        return next((i for i in self.instances if i['displayName'] == display_name
                     and i['schemaClassName'] in class_names), None)

    def search(self, class_name, attribute, value, operand='equal', limit=5):
        self.calls.append(('search', class_name, attribute, value))
        def vals(i):
            v = i.get(attribute)
            return [str(x) for x in (v if isinstance(v, list) else [v])]
        return [i for i in self.instances if i['schemaClassName'] == class_name
                and str(value) in vals(i)][:limit]

    def find_by_db_id(self, db_id):
        return next((i for i in self.instances if i['dbId'] == db_id), None)


class FakeUniProt:
    def __init__(self, entries: Dict[str, Dict], genes: Optional[Dict[str, str]] = None):
        self.entries, self.genes = entries, genes or {}

    def fetch(self, accession):
        return self.entries.get(accession.upper())

    def search_gene(self, symbol, taxon=9606):
        return self.genes.get(symbol)


class FakeOntology:
    def __init__(self, hits: Dict[tuple, List[Dict]]):
        self.hits = hits

    def search(self, name, ontology):
        return self.hits.get((name.lower(), ontology), [])


class SyncJobRunner:
    """Runs the job immediately, in the caller's thread: deterministic for tests."""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def submit(self, owner, fn):
        job = Job(id=f'job-{len(self.jobs) + 1}', owner=owner, status='running')
        self.jobs[job.id] = job
        try:
            job.result = fn(lambda m: setattr(job, 'progress', m))
            job.status = 'done'
        except Exception as e:
            job.error, job.status = f'{type(e).__name__}: {e}', 'failed'
        return job.id

    def get(self, job_id, owner):
        job = self.jobs.get(job_id)
        return job if job and job.owner == owner else None
