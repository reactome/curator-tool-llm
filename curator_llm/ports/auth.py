from dataclasses import dataclass
from typing import Protocol


class AuthError(Exception):
    """Token missing, invalid or expired (HTTP 401)."""


class ForbiddenError(Exception):
    """Token valid but the user may not use this feature (HTTP 403)."""


@dataclass(frozen=True)
class AuthUser:
    username: str
    role: str


class AuthProvider(Protocol):
    def verify(self, token: str) -> AuthUser:
        """Return the user for a bearer token. Raise AuthError if it is not valid. Fail closed."""
        ...
