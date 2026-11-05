"""Command-line results the front-end can open: the whole annotation written as a file, added to a curator's annotations
directly (--owner) or later (import), and then served by the API like any other."""
import json
import os

from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_file import FileSessionStore
from curator_llm.annotation_files import main as import_main
from curator_llm.api.app import create_app
from curator_llm.cli import main as cli_main
from tests.fakes.ports import FakeAuthProvider, SyncJobRunner
from tests.test_cli import FakePipeline, pdf

TOKENS = {'t-wgm': ('wgm', 'curator'), 't-marija': ('marija', 'curator')}


def api(store_dir):
    return TestClient(create_app(FakeAuthProvider(TOKENS), FileSessionStore(str(store_dir)), SyncJobRunner(), object()))


def H(user):
    return {'Authorization': f'Bearer t-{user}'}


def test_the_command_line_writes_the_whole_annotation_and_with_owner_adds_it_for_the_front_end(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    lines = []
    code = cli_main([pdf(folder, 'paper.pdf'), '--name', 'mine', '--out', str(tmp_path / 'out'), '--owner', 'wgm',
                     '--session-dir', str(tmp_path / 'sessions')], pipeline=FakePipeline(), say=lines.append)
    assert code == 0
    doc = json.load(open(tmp_path / 'out' / 'mine.annotation.json'))
    assert doc['owner'] == 'wgm' and doc['status'] == 'ready' and doc['draft']['reactions'] and doc['user_instances']
    saved = os.listdir(tmp_path / 'sessions' / 'wgm')
    assert saved == [doc['id'] + '.json']
    assert any("added to wgm's annotations" in m for m in lines)
    c = api(tmp_path / 'sessions')
    listed = c.get('/api/llm/sessions', headers=H('wgm')).json()
    assert [s['id'] for s in listed] == [doc['id']]
    d = c.get(f"/api/llm/sessions/{doc['id']}", headers=H('wgm')).json()
    assert c.get(f"/api/llm/sessions/{doc['id']}/reactions/{d['reactions'][0]['key']}", headers=H('wgm')).status_code == 200
    assert c.get(f"/api/llm/sessions/{doc['id']}", headers=H('marija')).status_code == 404     # only its owner sees it


def test_without_owner_nothing_goes_to_a_store_and_the_file_can_be_added_later(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    cli_main([pdf(folder, 'paper.pdf'), '--name', 'mine', '--out', str(tmp_path / 'out'),
              '--session-dir', str(tmp_path / 'sessions')], pipeline=FakePipeline(), say=lambda m: None)
    assert not (tmp_path / 'sessions').exists()
    f = str(tmp_path / 'out' / 'mine.annotation.json')
    assert json.load(open(f))['owner'] == 'local'
    out = []
    assert import_main([f, '--owner', 'marija', '--session-dir', str(tmp_path / 'sessions')], say=out.append) == 0
    assert 'added to marija' in out[0]
    c = api(tmp_path / 'sessions')
    [s] = c.get('/api/llm/sessions', headers=H('marija')).json()
    assert s['status'] == 'ready'
    # the same file again is noticed, and replaced only when asked
    import_main([f, '--owner', 'marija', '--session-dir', str(tmp_path / 'sessions')], say=out.append)
    assert 'already has it' in out[-1] and len(os.listdir(tmp_path / 'sessions' / 'marija')) == 1
    import_main([f, '--owner', 'marija', '--session-dir', str(tmp_path / 'sessions'), '--replace'], say=out.append)
    assert 'added to marija' in out[-1] and len(os.listdir(tmp_path / 'sessions' / 'marija')) == 1


def test_a_file_that_is_not_an_annotation_is_refused(tmp_path):
    bad = tmp_path / 'x.issues.json'
    bad.write_text('[]')
    out = []
    assert import_main([str(bad), '--owner', 'wgm', '--session-dir', str(tmp_path / 's')], say=out.append) == 1
    assert 'not an annotation file' in out[0]
