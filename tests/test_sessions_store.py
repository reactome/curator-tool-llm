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


# --- the file store (the default) ---

def test_file_store_contract(tmp_path):
    from curator_llm.adapters.sessions_file import FileSessionStore
    check_contract(FileSessionStore(str(tmp_path)))


def test_file_store_keeps_sessions_across_a_restart_in_a_folder_per_user(tmp_path):
    from curator_llm.adapters.sessions_file import FileSessionStore
    s = sample(); FileSessionStore(str(tmp_path)).save(s)
    assert (tmp_path / 'alice' / f'{s.id}.json').is_file()
    again = FileSessionStore(str(tmp_path))                 # a new process
    assert again.get(s.id, 'alice').draft.participants['a'].name == 'PINK1'
    assert [x.id for x in again.list('alice')] == [s.id]


def test_file_store_lists_newest_first(tmp_path):
    from curator_llm.adapters.sessions_file import FileSessionStore
    store = FileSessionStore(str(tmp_path))
    old, new = sample(), sample()
    old.created, new.created = 100.0, 200.0
    store.save(old); store.save(new)
    assert [x.id for x in store.list('alice')] == [new.id, old.id]


def test_file_store_keeps_ids_and_usernames_inside_their_folder(tmp_path):
    from curator_llm.adapters.sessions_file import FileSessionStore
    root = tmp_path / 'sessions'
    store = FileSessionStore(str(root))
    bob = sample('bob'); store.save(bob)
    assert store.get(f'../bob/{bob.id}', 'alice') is None            # an id cannot climb out of the owner's folder
    assert store.get(bob.id, '../bob') is None and store.list('..') == []
    odd = sample('a/b'); store.save(odd)                           # a slash in a username stays one folder
    assert [p.name for p in root.iterdir() if p.is_dir()] and store.get(odd.id, 'a/b') is not None
    assert all(p.parent.parent == root for p in root.rglob('*.json'))
    with pytest.raises(ValueError):
        bad = sample(); bad.id = '../x'; store.save(bad)


def test_file_store_skips_a_damaged_file(tmp_path):
    from curator_llm.adapters.sessions_file import FileSessionStore
    store = FileSessionStore(str(tmp_path))
    s = sample(); store.save(s)
    (tmp_path / 'alice' / 'broken.json').write_text('{not json')
    assert [x.id for x in store.list('alice')] == [s.id]


def test_file_store_marks_sessions_a_restart_cut_off_as_failed(tmp_path):
    from curator_llm.adapters.sessions_file import FileSessionStore
    store = FileSessionStore(str(tmp_path))
    running, done = sample(), sample()
    running.status, done.status = 'running', 'ready'
    store.save(running); store.save(done)
    assert FileSessionStore(str(tmp_path)).mark_interrupted() == 1
    got = store.get(running.id, 'alice')
    assert got.status == 'failed' and 'restarted' in got.error
    assert store.get(done.id, 'alice').status == 'ready'
