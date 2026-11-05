"""Annotation files: a whole annotation (session) as one JSON file, and adding one to a curator's annotations.

The command line writes `<name>.annotation.json` next to its other result files: the complete annotation, as the service
keeps it (draft, quotes, issues and their statuses, suggested fixes, QA results, the papers' text). Adding that file to a
user's folder of the file session store makes it one of their annotations in the front-end ("Your annotations"), exactly
like one started there:

    python -m curator_llm.annotation_files results/wee1.annotation.json --owner wgm         (or scripts/import_annotation.sh)
    --session-dir DIR    the store's folder (default: SESSION_DIR from .env, else data/sessions)
    --replace            overwrite an annotation already added from the same file

The service reads the folder on every request, so no restart is needed. On a server, run it from the deployed folder, as
a user who can write data/sessions (several people deploy there: keep the files group-writable).
"""
import argparse
import os
import sys
from typing import Callable, List, Optional

from curator_llm.models.session import Session

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def default_session_dir() -> str:
    return os.getenv('SESSION_DIR') or os.path.join(ROOT, 'data', 'sessions')


def read(path: str) -> Session:
    with open(path, encoding='utf-8') as f:
        return Session.model_validate_json(f.read())


def add_to(store, session: Session, owner: str, replace: bool = False) -> Optional[str]:
    """Save `session` as one of `owner`'s annotations in `store` (a SessionStore). It keeps its id, so adding the same file
    twice is noticed: None (nothing saved) unless `replace`. Returns the session id."""
    if store.get(session.id, owner) is not None and not replace:
        return None
    session.owner = owner
    if session.status in ('queued', 'running'):      # no job runs for it here: it can only be what was finished
        session.status = 'ready'
    store.save(session)
    return session.id


def main(argv: Optional[List[str]] = None, say: Callable[[str], None] = None) -> int:
    say = say or (lambda m: print(m, flush=True))
    ap = argparse.ArgumentParser(prog='import_annotation', description='Add annotation files (<name>.annotation.json, '
                                 "written by the command line) to a curator's annotations, so the front-end can open them.")
    ap.add_argument('files', nargs='+', metavar='FILE', help='an <name>.annotation.json file; several may be given')
    ap.add_argument('--owner', required=True, help="the curator who gets them (their login name)")
    ap.add_argument('--session-dir', metavar='DIR', help='the file session store folder (default: SESSION_DIR, else data/sessions)')
    ap.add_argument('--replace', action='store_true', help='overwrite an annotation already added from the same file')
    a = ap.parse_args(argv)
    from curator_llm.adapters.sessions_file import FileSessionStore
    from curator_llm.wiring import load_env
    load_env()
    store = FileSessionStore(a.session_dir or default_session_dir())
    bad = 0
    for path in a.files:
        try:
            s = read(path)
        except Exception as e:                        # not an annotation file, or damaged
            say(f'{path}: not an annotation file ({type(e).__name__}: {e})')
            bad += 1
            continue
        sid = add_to(store, s, a.owner, a.replace)
        if sid is None:
            say(f'{path}: {a.owner} already has it ({s.id}); --replace to overwrite')
        else:
            say(f'{path}: added to {a.owner} as "{s.source}" ({len(s.draft.reactions) if s.draft else 0} reactions) -> '
                f'{os.path.join(store.directory, a.owner, sid + ".json")}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
