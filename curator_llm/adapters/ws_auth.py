"""AuthProvider backed by curator-tool-ws GET /api/auth/verify (ws stays the auth authority)."""
import time
from typing import Dict, Tuple

import requests

from curator_llm.ports.auth import AuthError, AuthUser, ForbiddenError

REQUIRED_ROLE = 'curator'


class WsAuthProvider:
    def __init__(self, base_url: str, cache_seconds: float = 45.0, timeout: float = 5.0,
                 session=None, clock=time.monotonic):
        self.url = base_url.rstrip('/') + '/api/auth/verify'
        self.cache_seconds, self.timeout = cache_seconds, timeout
        self.http = session or requests
        self.clock = clock
        self._cache: Dict[str, Tuple[float, AuthUser]] = {}

    def verify(self, token: str) -> AuthUser:
        if not token:
            raise AuthError('missing token')
        hit = self._cache.get(token)
        if hit and self.clock() - hit[0] < self.cache_seconds:
            return hit[1]
        try:
            r = self.http.get(self.url, headers={'Authorization': f'Bearer {token}'},
                              timeout=self.timeout)
        except requests.RequestException as e:      # fail closed: ws unreachable means no access
            raise AuthError(f'auth service unreachable: {e}') from e
        if r.status_code in (401, 403):
            raise AuthError('token rejected')
        if r.status_code != 200:
            raise AuthError(f'auth service returned {r.status_code}')
        data = r.json()
        user = AuthUser(username=data['username'], role=(data.get('role') or ''))
        if user.role.lower() != REQUIRED_ROLE:
            raise ForbiddenError(f"role '{user.role}' may not use this feature")
        self._cache[token] = (self.clock(), user)
        return user
