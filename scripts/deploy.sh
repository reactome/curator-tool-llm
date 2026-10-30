#!/usr/bin/env bash
# Package the curator-tool-llm REST service, copy it to a Linux server with scp, and (re)start it there.
#
#   scripts/deploy.sh user@server                  # package, check, upload, install if needed, restart
#   scripts/deploy.sh user@server --dry-run        # package and check only; print what it would do
#   scripts/deploy.sh user@server --path /srv/curator-llm --bind 0.0.0.0 --port 8000
#
# What is sent (the REST service only): curator_llm/, the 6 reactome_llm modules it loads, run_extraction.py,
# run_merge.py, run_review.py, resources/reactome_domain_model.json, scripts/run_api.sh, requirements-service.txt,
# .env.example and the README. NEVER .env (your keys), data/, results/, logs/ or tests/.
#
# On the server, in --path (default ~/curator-tool-llm), it:
#   * keeps the previous code as previous-code.tar.gz (roll back with: tar -xzf previous-code.tar.gz)
#   * replaces the code, but leaves .env, data/, logs/ and venv/ alone
#   * creates .env from .env.example the first time (edit it: ANTHROPIC_API_KEY, WS_BASE_URL, REACTOME_NEO4J_*, ...)
#   * creates a venv and installs requirements-service.txt, only when that file changed
#   * restarts the service in the background (scripts/run_api.sh --daemon) and checks that it answers
#
# Options:
#   --path DIR        install directory on the server           (default ~/curator-tool-llm, or $DEPLOY_PATH)
#   --bind ADDR       address the service listens on             (default 0.0.0.0)
#   --port N          port                                       (default 8000)
#   --python CMD      Python 3.10+ on the server, for the venv   (default python3)
#   --full            also send the gene-first pipeline (run_curator.py, reactome_llm/, ...; needs requirements.txt)
#   --with-snapshots  also send data/snapshots (saved results, for SNAPSHOT_MODE=replay demos)
#   --no-install      skip the venv / pip step
#   --no-restart      upload and unpack only
#   --keep-archive    keep the .tar.gz in ./dist (it is deleted otherwise)
#   --dry-run         build and check the archive; do not touch the server
#   --skip-check      skip the local import check of the unpacked archive
# Environment: DEPLOY_HOST (instead of the first argument), SSH_OPTS (e.g. "-i ~/.ssh/key -p 2222").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${DEPLOY_HOST:-}"
REMOTE_PATH="${DEPLOY_PATH:-~/curator-tool-llm}"
BIND=0.0.0.0; PORT=8000; REMOTE_PY=python3
FULL=0; SNAPSHOTS=0; INSTALL=1; RESTART=1; KEEP=0; DRY=0; CHECK=1

usage() { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --path) REMOTE_PATH="$2"; shift 2 ;;
    --bind) BIND="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --python) REMOTE_PY="$2"; shift 2 ;;
    --full) FULL=1; shift ;;
    --with-snapshots) SNAPSHOTS=1; shift ;;
    --no-install) INSTALL=0; shift ;;
    --no-restart) RESTART=0; shift ;;
    --keep-archive) KEEP=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --skip-check) CHECK=0; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) HOST="$1"; shift ;;
  esac
done
if [[ -z "$HOST" && "$DRY" -eq 0 ]]; then echo "Give the server as user@host (or set DEPLOY_HOST)." >&2; usage >&2; exit 2; fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE_NAME="curator-tool-llm-$STAMP.tar.gz"
ARCHIVE="$WORK/$ARCHIVE_NAME"
STAGE="$WORK/stage"
mkdir -p "$STAGE"

# ── 1. choose the files ────────────────────────────────────────────────────
SERVICE_FILES=(
  reactome_llm/FullTextPDFPrompts.py reactome_llm/FullTextPDFSections.py reactome_llm/ModelConfig.py
  reactome_llm/PubMedFetcher.py reactome_llm/token_profiler.py
  run_extraction.py run_merge.py run_review.py
  resources/reactome_domain_model.json
  scripts/run_api.sh requirements-service.txt .env.example README.md
)
for f in "${SERVICE_FILES[@]}"; do
  [[ -f "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done
[[ -d curator_llm ]] || { echo "Run this from the curator-tool-llm repository." >&2; exit 1; }

# tar (BSD on macOS, GNU on Linux) reading a NUL-separated list; COPYFILE_DISABLE keeps macOS ._ files out
export COPYFILE_DISABLE=1
{
  find curator_llm -type f -name '*.py' -print0
  for f in "${SERVICE_FILES[@]}"; do printf '%s\0' "$f"; done
  if [[ "$FULL" -eq 1 ]]; then
    # everything git knows about (tracked, plus new files not ignored), minus tests, docs, data and scratch
    git ls-files -co --exclude-standard -z | grep -zv -E '^(tests|test|docs|data|results|logs|\.github)/|\.pptx$|^Review |^chainlit\.md$|^\.env$' || true
    printf '%s\0' requirements.txt
  fi
  if [[ "$SNAPSHOTS" -eq 1 && -d data/snapshots ]]; then find data/snapshots -type f -print0; fi
} | sort -zu > "$WORK/files.lst"

tar -czf "$ARCHIVE" --null -T "$WORK/files.lst"
tar -xzf "$ARCHIVE" -C "$STAGE"
N_FILES=$(tr -cd '\0' < "$WORK/files.lst" | wc -c | tr -d ' ')
SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo "Packaged $N_FILES files ($SIZE): $ARCHIVE_NAME"

# ── 2. refuse to ship secrets ──────────────────────────────────────────────
[[ ! -e "$STAGE/.env" ]] || { echo "ABORT: .env is in the archive." >&2; exit 1; }
SECRET_RE='sk-ant-[A-Za-z0-9_-]{30,}|sk-(proj-)?[A-Za-z0-9_-]{40,}|-----BEGIN [A-Z ]*PRIVATE KEY-----'
if grep -rIlE "$SECRET_RE" "$STAGE" 2>/dev/null | grep -q .; then
  echo "ABORT: something in the archive looks like a secret key:" >&2
  grep -rIlE "$SECRET_RE" "$STAGE" >&2
  exit 1
fi
# .env.example may only carry commented-out secrets
if grep -E '^(ANTHROPIC_API_KEY|OPENAI_API_KEY|PUBMED_API_KEY|REACTOME_NEO4J_PWD|WS_PASSWORD)=.+' "$STAGE/.env.example" >/dev/null 2>&1; then
  echo "ABORT: .env.example holds an uncommented secret." >&2; exit 1
fi

# ── 3. check the unpacked copy really runs (imports resolve from the archive, not from this checkout) ──
if [[ "$CHECK" -eq 1 ]]; then
  if python -c 'import fastapi, anthropic, langgraph, fitz, neo4j' 2>/dev/null; then
    (
      cd "$STAGE"
      export CURATOR_LLM_AUTOSTART=0 ANTHROPIC_API_KEY=check-only
      python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
import curator_llm.main, curator_llm.api.app            # the service and everything it imports
sys.path.insert(0, os.path.join(os.getcwd(), 'reactome_llm'))
import ModelConfig, PubMedFetcher, FullTextPDFSections, FullTextPDFPrompts   # what the scripts load by path
from curator_llm.services import schema_index
assert os.path.dirname(os.path.dirname(curator_llm.main.__file__)) == os.getcwd(), 'imported from outside the archive'
schema_index.check_instance({'dbId': -1, 'displayName': 'x', 'schemaClassName': 'SimpleEntity', 'attributes': {}})
print('import check passed')
PY
      for s in run_extraction.py run_merge.py run_review.py; do python "$s" --help >/dev/null 2>&1 || { echo "ABORT: $s does not start from the archive" >&2; exit 1; }; done
      echo "script check passed"
    ) || { echo "ABORT: the packaged copy does not run (see above)." >&2; exit 1; }
  else
    echo "Skipping the import check: the current Python lacks the service dependencies (activate the environment, or pass --skip-check)." >&2
  fi
fi

if [[ "$KEEP" -eq 1 ]]; then mkdir -p dist; cp "$ARCHIVE" "dist/$ARCHIVE_NAME"; echo "Kept dist/$ARCHIVE_NAME"; fi

# ── 4. what runs on the server ─────────────────────────────────────────────
remote_script() { cat <<'REMOTE'
set -euo pipefail
DIR="$1"; ARCHIVE="$2"; PY="$3"; BIND="$4"; PORT="$5"; INSTALL="$6"; RESTART="$7"
DIR="${DIR/#\~/$HOME}"
mkdir -p "$DIR"; cd "$DIR"

if [ "$RESTART" = 1 ] && [ -x scripts/run_api.sh ]; then scripts/run_api.sh --stop || true; fi

if [ -d curator_llm ]; then                       # keep the last version so a bad deploy can be undone
  tar -czf previous-code.tar.gz curator_llm reactome_llm scripts resources run_*.py requirements*.txt 2>/dev/null || true
fi
rm -rf curator_llm reactome_llm scripts           # no stale modules from older versions; .env, data/, logs/, venv/ stay
tar -xzf "$ARCHIVE"
rm -f "$ARCHIVE"
chmod +x scripts/*.sh
mkdir -p data logs

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">>> Created $DIR/.env from .env.example. EDIT IT before using the service (ANTHROPIC_API_KEY, WS_BASE_URL, REACTOME_NEO4J_*)."
fi

if [ "$INSTALL" = 1 ]; then
  [ -d venv ] || "$PY" -m venv venv
  NEW=$(sha256sum requirements-service.txt | cut -d' ' -f1)
  if [ ! -f venv/.requirements.sha ] || [ "$(cat venv/.requirements.sha)" != "$NEW" ]; then
    echo "Installing requirements-service.txt ..."
    venv/bin/pip install --quiet --upgrade pip
    venv/bin/pip install --quiet -r requirements-service.txt
    echo "$NEW" > venv/.requirements.sha
  else
    echo "Requirements unchanged, not reinstalling."
  fi
fi

if [ "$RESTART" = 1 ]; then
  [ -f venv/bin/activate ] && . venv/bin/activate
  HOST="$BIND" PORT="$PORT" scripts/run_api.sh --daemon
  sleep 4
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/llm/sessions" || true)
  if [ "$code" = 401 ]; then echo "Service is up (HTTP 401 = it is running and asking for a token)."
  else echo "!!! Health check returned HTTP ${code:-none}. See $DIR/logs/api.log" >&2; tail -n 15 logs/api.log >&2 || true; exit 1; fi
fi
echo "Deployed to $DIR"
REMOTE
}

if [[ "$DRY" -eq 1 ]]; then
  echo
  echo "Dry run: nothing was sent. It would run:"
  echo "  scp $ARCHIVE_NAME $HOST:/tmp/"
  echo "  ssh $HOST bash -s -- '$REMOTE_PATH' /tmp/$ARCHIVE_NAME $REMOTE_PY $BIND $PORT $INSTALL $RESTART   # script below"
  echo "-------- remote script --------"
  remote_script
  echo "-------- archive contents --------"
  tar -tzf "$ARCHIVE" | sed 's/^/  /' | head -80
  exit 0
fi

# ── 5. upload and run (one connection, so a password is asked for once) ────
SSH_COMMON=(-o ControlMaster=auto -o "ControlPath=$WORK/ssh-%C" -o ControlPersist=120 ${SSH_OPTS:-})
echo "Uploading to $HOST ..."
scp "${SSH_COMMON[@]}" "$ARCHIVE" "$HOST:/tmp/$ARCHIVE_NAME"
remote_script | ssh "${SSH_COMMON[@]}" "$HOST" bash -s -- "$REMOTE_PATH" "/tmp/$ARCHIVE_NAME" "$REMOTE_PY" "$BIND" "$PORT" "$INSTALL" "$RESTART"
ssh "${SSH_COMMON[@]}" -O exit "$HOST" 2>/dev/null || true
