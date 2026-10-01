"""Write the API contract the frontend builds against:  python scripts/export_openapi.py [out.json]"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from curator_llm.adapters.sessions_memory import InMemorySessionStore  # noqa: E402
from curator_llm.api.app import create_app  # noqa: E402
from tests.fakes.ports import FakeAuthProvider, SyncJobRunner  # noqa: E402


class _NoPipeline:
    def run(self, spec, report):
        raise NotImplementedError


class _NoChat:
    def turn(self, *a):
        raise NotImplementedError


def build():
    return create_app(FakeAuthProvider({}), InMemorySessionStore(), SyncJobRunner(), _NoPipeline(),
                      upload_dir='/tmp', chat_model=_NoChat())


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'docs', 'openapi.json')
    with open(out, 'w') as f:
        json.dump(build().openapi(), f, indent=2)
    print('wrote', os.path.abspath(out))
