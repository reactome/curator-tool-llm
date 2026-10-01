"""Contract test against a REAL curator-tool-ws. Skipped unless WS_BASE_URL / WS_USERNAME / WS_PASSWORD
are set in .env and ws is reachable. Run separately from the unit suite's fakes."""
import os

import pytest
import requests

from curator_llm.adapters.ws_auth import WsAuthProvider
from curator_llm.adapters.ws_login import WsLoginTokenGetter
from curator_llm.ports.auth import AuthError


@pytest.fixture(scope='module')
def ws():
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    base, user, pwd = os.getenv('WS_BASE_URL'), os.getenv('WS_USERNAME'), os.getenv('WS_PASSWORD')
    if not (base and user and pwd):
        pytest.skip('WS_BASE_URL / WS_USERNAME / WS_PASSWORD not set')
    try:
        token = WsLoginTokenGetter(base, user, pwd, timeout=5)()
    except Exception as e:
        pytest.skip(f'ws not reachable or login failed: {e}')
    return base, token


def test_verify_returns_username_and_role(ws):
    base, token = ws
    r = requests.get(base + '/api/auth/verify', headers={'Authorization': f'Bearer {token}'}, timeout=10)
    assert r.status_code == 200 and set(r.json()) == {'username', 'role'}


def test_ws_auth_provider_accepts_a_curator_and_rejects_a_bad_token(ws):
    base, token = ws
    assert WsAuthProvider(base).verify(token).role.lower() == 'curator'
    with pytest.raises(AuthError):
        WsAuthProvider(base).verify('not-a-token')
