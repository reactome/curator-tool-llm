"""Token source for service-to-ws calls in dev/batch runs: logs in with a service account and
re-logs-in shortly before the (5 minute) access token expires. In the REST service the caller's own
bearer token is passed through instead; this is for scripts that have no user request."""
import time

import requests


class WsLoginTokenGetter:
    def __init__(self, base_url: str, username: str, password: str, ttl_seconds: float = 240.0,
                 session=None, clock=time.monotonic, timeout: float = 10.0):
        self.url = base_url.rstrip('/') + '/api/auth/login'
        self.creds = {'username': username, 'password': password}
        self.ttl, self.clock, self.timeout = ttl_seconds, clock, timeout
        self.http = session or requests
        self._token, self._at = None, 0.0

    def __call__(self) -> str:
        if self._token is None or self.clock() - self._at >= self.ttl:
            r = self.http.post(self.url, json=self.creds, timeout=self.timeout)
            if r.status_code != 200:
                raise RuntimeError(f'ws login failed ({r.status_code})')
            self._token, self._at = r.text.strip().strip('"'), self.clock()
        return self._token
