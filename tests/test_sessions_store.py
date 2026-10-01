import os
import uuid

import pytest

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.models.reactome import EwasSpec, ReactomeDraft
from curator_llm.models.session import Issue, Session


def sample(owner='alice'):
    d = ReactomeDraft()
    d.participants = {'a': EwasSpec(key='a', name='PINK1', uniprot='Q9BXM7')}
    return Session(owner=owner, pmid='1', draft=d, issues=[Issue(source='builder', code='note', message='m')])


def check_contract(store):
    s = sample()
    store.save(s)
    got = store.get(s.id, 'alice')
    assert got.draft.participants['a'].uniprot == 'Q9BXM7' and got.issues[0].code == 'note'
    assert store.get(s.id, 'bob') is None and store.get('nope', 'alice') is None
    got.status = 'ready'                      # mutating a returned copy must not change the store
    assert store.get(s.id, 'alice').status == 'queued'
    store.save(got)
    assert store.get(s.id, 'alice').status == 'ready'
    other = sample('bob'); store.save(other)
    assert [x.id for x in store.list('alice')] == [s.id] and [x.id for x in store.list('bob')] == [other.id]


def test_in_memory_store_contract():
    check_contract(InMemorySessionStore())


def test_mongo_store_contract_on_a_throwaway_collection():
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    try:
        from pymongo import MongoClient
        client = MongoClient(os.getenv('PUBMED_MONGO_URI'), serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
    except Exception:
        pytest.skip('MongoDB not reachable')
    from curator_llm.adapters.sessions_mongo import MongoSessionStore
    name = f'sessions_test_{uuid.uuid4().hex[:8]}'
    coll = client['curator_llm_test'][name]
    try:
        check_contract(MongoSessionStore(coll))
    finally:
        coll.drop()
