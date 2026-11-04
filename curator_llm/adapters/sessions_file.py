"""File-backed SessionStore, the default: one JSON file per session (one annotation), in a folder per user.

    <directory>/<owner>/<session id>.json

Sessions survive a restart of the service without needing MongoDB up, and they can be inspected, backed up or copied
by hand. A session belongs to the folder it is in and to the owner recorded inside it; both are checked on every read.

Safe for the one-worker deployment the service requires (see README, "Running it for real"): writes within the
process are serialized by a lock, and each write goes to a temporary file that then replaces the old one, so a crash
mid-write leaves the previous version rather than a half-written file. Several worker processes writing the same
folder are not supported.
"""
import logging
import os
import re
import tempfile
import threading
import time
from typing import List, Optional
from urllib.parse import quote

from curator_llm.models.session import Session

log = logging.getLogger(__name__)

# Session ids arrive in URLs (/sessions/{id}). Only these characters become part of a path, so an id such as
# "../bob/x" can never reach outside the owner's folder. Generated ids are uuid4 hex, well inside this.
_ID = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


def _owner_dir_name(owner: str) -> str:
    """The owner's folder name: the username percent-encoded, so every username maps to its own folder and no
    character in it (a slash, say) can change the path. "." and ".." are encoded too: quote() leaves dots alone."""
    name = quote(owner, safe='')
    return name.replace('.', '%2E') if name in ('.', '..') else name


class FileSessionStore:
    def __init__(self, directory: str):
        self.directory = directory
        self._lock = threading.Lock()
        os.makedirs(directory, exist_ok=True)

    def _owner_dir(self, owner: str) -> str:
        return os.path.join(self.directory, _owner_dir_name(owner))

    def _path(self, session_id: str, owner: str) -> Optional[str]:
        if not owner or not _ID.match(session_id or ''):
            return None
        return os.path.join(self._owner_dir(owner), f'{session_id}.json')

    @staticmethod
    def _read(path: str) -> Optional[Session]:
        try:
            with open(path, encoding='utf-8') as f:
                return Session.model_validate_json(f.read())
        except FileNotFoundError:
            return None
        except Exception as e:          # a damaged or hand-edited file must not take the whole list down
            log.warning('skipping unreadable session file %s: %s', path, e)
            return None

    def save(self, session: Session) -> None:
        path = self._path(session.id, session.owner)
        if path is None:
            raise ValueError(f'cannot store session {session.id!r} of owner {session.owner!r}')
        session.updated = time.time()
        data = session.model_dump_json()
        with self._lock:
            folder = os.path.dirname(path)
            os.makedirs(folder, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=folder, prefix='.tmp-', suffix='.json')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(data)
                os.replace(tmp, path)
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise

    def get(self, session_id: str, owner: str) -> Optional[Session]:
        path = self._path(session_id, owner)
        if path is None:
            return None
        s = self._read(path)
        return s if s is not None and s.owner == owner and s.id == session_id else None

    def list(self, owner: str) -> List[Session]:
        if not owner:
            return []
        folder = self._owner_dir(owner)
        try:
            names = os.listdir(folder)
        except FileNotFoundError:
            return []
        out = []
        for name in names:
            if name.endswith('.json') and not name.startswith('.'):
                s = self._read(os.path.join(folder, name))
                if s is not None and s.owner == owner:
                    out.append(s)
        return sorted(out, key=lambda s: -s.created)

    def mark_interrupted(self) -> int:
        """Call once at startup. A session still queued or running when the service stopped has lost its job (jobs
        live in the process), so it would show "running" for ever. Mark each one failed, saying why, so the curator
        knows to start it again. Returns how many were marked."""
        n = 0
        try:
            owners = os.listdir(self.directory)
        except FileNotFoundError:
            return 0
        for owner_dir in owners:
            folder = os.path.join(self.directory, owner_dir)
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                if not name.endswith('.json') or name.startswith('.'):
                    continue
                s = self._read(os.path.join(folder, name))
                if s is not None and s.status in ('queued', 'running'):
                    s.status = 'failed'
                    s.error = 'The service restarted before this annotation finished. Please start it again.'
                    self.save(s)
                    n += 1
        return n
