import copy
import threading
import time
from typing import Dict, List, Optional

from curator_llm.models.session import Session


class InMemorySessionStore:
    """Process-local store (dev and tests). Copies on the way in and out so callers cannot mutate
    stored state by accident."""

    def __init__(self):
        self._s: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def save(self, session: Session) -> None:
        session.updated = time.time()
        with self._lock:
            self._s[session.id] = copy.deepcopy(session)

    def get(self, session_id: str, owner: str) -> Optional[Session]:
        with self._lock:
            s = self._s.get(session_id)
            return copy.deepcopy(s) if s and s.owner == owner else None

    def list(self, owner: str) -> List[Session]:
        with self._lock:
            return sorted((copy.deepcopy(s) for s in self._s.values() if s.owner == owner),
                          key=lambda s: -s.created)
