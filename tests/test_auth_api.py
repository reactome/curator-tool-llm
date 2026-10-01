import pytest
import requests
from fastapi.testclient import TestClient

from curator_llm.adapters.ws_auth import WsAuthProvider
from curator_llm.api.app import create_app
from curator_llm.ports.auth import AuthError, ForbiddenError
from tests.fakes.ports import FakeAuthProvider, SyncJobRunner

TOKENS = {'good': ('alice', 'curator'), 'viewer': ('bob', 'viewer')}


@pytest.fixture
def client():
    return TestClient(create_app(FakeAuthProvider(TOKENS)))


def test_no_token_is_401(client):
    assert client.get('/api/llm/whoami').status_code == 401


def test_bad_token_is_401(client):
    assert client.get('/api/llm/whoami', headers={'Authorization': 'Bearer nope'}).status_code == 401


def test_non_curator_is_403(client):
    assert client.get('/api/llm/whoami', headers={'Authorization': 'Bearer viewer'}).status_code == 403


def test_curator_ok(client):
    r = client.get('/api/llm/whoami', headers={'Authorization': 'Bearer good'})
    assert r.status_code == 200 and r.json()['username'] == 'alice'


def test_every_llm_route_requires_auth(client):
    """Guards against a route being added outside the protected router."""
    checked = 0
    for path, methods in client.app.openapi()['paths'].items():
        if path.startswith('/api/llm'):
            for method in methods:
                url = path.replace('{', '').replace('}', '')
                assert client.request(method.upper(), url).status_code in (401, 403), path
                checked += 1
    assert checked


class _Resp:
    def __init__(self, status, data=None):
        self.status_code, self._d = status, data or {}

    def json(self):
        return self._d


class _Http:
    def __init__(self, resp):
        self.resp, self.calls = resp, 0

    def get(self, *a, **k):
        self.calls += 1
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


def test_ws_adapter_caches_then_expires():
    http = _Http(_Resp(200, {'username': 'alice', 'role': 'curator'}))
    now = [0.0]
    p = WsAuthProvider('http://ws', session=http, clock=lambda: now[0])
    p.verify('t'); p.verify('t')
    assert http.calls == 1
    now[0] = 100
    p.verify('t')
    assert http.calls == 2


def test_ws_adapter_rejects_non_curator_unreachable_and_401():
    with pytest.raises(ForbiddenError):
        WsAuthProvider('http://ws', session=_Http(_Resp(200, {'username': 'b', 'role': 'viewer'}))).verify('t')
    with pytest.raises(AuthError):
        WsAuthProvider('http://ws', session=_Http(requests.ConnectionError('down'))).verify('t')
    with pytest.raises(AuthError):
        WsAuthProvider('http://ws', session=_Http(_Resp(401))).verify('t')


def test_job_runner_owner_isolation():
    r = SyncJobRunner()
    jid = r.submit('alice', lambda progress: 42)
    assert r.get(jid, 'alice').result == 42 and r.get(jid, 'bob') is None
