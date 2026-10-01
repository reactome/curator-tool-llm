from typing import List, Optional, Protocol

from curator_llm.models.session import Session


class SessionStore(Protocol):
    def save(self, session: Session) -> None: ...

    def get(self, session_id: str, owner: str) -> Optional[Session]:
        """The session, or None if it does not exist or belongs to someone else."""
        ...

    def list(self, owner: str) -> List[Session]: ...
