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
# On the server, in --path (default /usr/local/reactomes/Reactome/production/webbench/curator-tool-llm), it:
#   * keeps the previous code as previous-code.tar.gz (roll back with: tar -xzf previous-code.tar.gz)
#   * replaces the code, but leaves .env, data/, logs/ and venv/ alone
#   * creates .env from .env.example the first time (edit it: ANTHROPIC_API_KEY, WS_BASE_URL, REACTOME_NEO4J_*, ...)
#   * checks first (before touching anything) that it can build a Python environment: a Python 3.10+ venv, or conda
#   * creates that environment and installs requirements-service.txt, only when that file changed
#   * restarts the service in the background (scripts/run_api.sh --daemon) and checks that it answers
#
# Options:
#   --path DIR        install directory on the server           (default /usr/local/reactomes/Reactome/production/webbench/curator-tool-llm, or $DEPLOY_PATH)
#   --bind ADDR       address the service listens on             (default 0.0.0.0)
#   --port N          port                                       (default 8000)
#   --python CMD      Python 3.10+ on the server, for a venv      (default python3)
#   --conda NAME      use a conda environment NAME (Python 3.10, from conda-forge) instead of a venv; created if missing
#   --conda-bin PATH  the conda executable, if it is not on the server's PATH or in a usual place (~/miniconda3 ...)
#   With neither flag it uses a venv when the server has a usable Python 3.10+, otherwise conda (env "curator-llm").
#   --use-venv DIR    use the server's EXISTING venv DIR (relative to --path, e.g. .venv): nothing is created or installed;
#                     the script only checks that it has the packages the service imports (it names what is missing)
#   --stop-cmd CMD    stop the service with CMD (run in --path) instead of scripts/run_api.sh --stop
#   --start-cmd CMD   start the service with CMD instead of scripts/run_api.sh --daemon (the health check still runs on --port)
#                     e.g. on a server managed by a script:  --stop-cmd '../manage-apps.sh stop llm' --start-cmd '../manage-apps.sh start llm'
#   --group-writable  make the deployed code group-writable, so another user in the same group can redeploy over it
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
REMOTE_PATH="${DEPLOY_PATH:-/usr/local/reactomes/Reactome/production/webbench/curator-tool-llm}"
BIND=0.0.0.0; PORT=8000; REMOTE_PY=python3; ENV_MODE=auto; CONDA_ENV=curator-llm; CONDA_BIN=; USE_VENV=; STOP_CMD=; START_CMD=; GROUP_W=0
FULL=0; SNAPSHOTS=0; INSTALL=1; RESTART=1; KEEP=0; DRY=0; CHECK=1

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --path) REMOTE_PATH="$2"; shift 2 ;;
    --bind) BIND="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --python) REMOTE_PY="$2"; ENV_MODE=venv; shift 2 ;;
    --conda) CONDA_ENV="$2"; ENV_MODE=conda; shift 2 ;;
    --conda-bin) CONDA_BIN="$2"; shift 2 ;;
    --group-writable) GROUP_W=1; shift ;;
    --use-venv) USE_VENV="$2"; shift 2 ;;
    --stop-cmd) STOP_CMD="$2"; shift 2 ;;
    --start-cmd) START_CMD="$2"; shift 2 ;;
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

XATTR=(); tar --no-xattrs --version >/dev/null 2>&1 && XATTR=(--no-xattrs)    # keep macOS extended attributes out of the archive
tar -czf "$ARCHIVE" "${XATTR[@]}" --null -T "$WORK/files.lst"
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
MODE="$8"; CONDA_ENV="$9"; CONDA_BIN="${10:-}"; USE_VENV="${11:-}"; STOP_CMD="${12:-}"; START_CMD="${13:-}"; GROUP_W="${14:-0}"
DIR="${DIR/#\~/$HOME}"
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# ---- preflight: everything that can be checked is checked BEFORE the running service or its code is touched ----
fail() { echo "!!! $*" >&2; echo "!!! Nothing was changed on the server." >&2; rm -f "$ARCHIVE"; exit 1; }
find_conda() {
  [ -n "$CONDA_BIN" ] && { [ -x "$CONDA_BIN" ] && echo "$CONDA_BIN"; return; }
  command -v conda 2>/dev/null && return
  for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/opt/miniconda3" /opt/conda /opt/miniconda3 /opt/anaconda3 /opt/miniforge3; do
    [ -x "$d/bin/conda" ] && { echo "$d/bin/conda"; return; }
  done
  return 1
}
python_ok() {   # 3.10+ with a working venv module
  command -v "$PY" >/dev/null 2>&1 && "$PY" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}
mkdir -p "$DIR" 2>/dev/null && [ -w "$DIR" ] || fail "cannot create or write $DIR: check that your user may write there (or use --path)."
# An earlier deployment may already live here. Never half-replace it: refuse before deleting anything if we could not
# finish the job (files owned by someone else) or if a service we did not start is running from this folder.
if [ -d "$DIR/curator_llm" ] || [ -d "$DIR/scripts" ]; then
  LOCKED=""
  while IFS= read -r f; do
    [ -w "$f" ] || { LOCKED="$LOCKED $f"; [ "$(echo $LOCKED | wc -w)" -ge 3 ] && break; }
  done < <(find "$DIR/curator_llm" "$DIR/reactome_llm" "$DIR/scripts" 2>/dev/null)
  [ -z "$LOCKED" ] || fail "existing files here are not writable by $(whoami) (owner: $(ls -ld "$DIR/curator_llm" | awk '{print $3}')), e.g.:$LOCKED"
fi
if [ "$RESTART" = 1 ] && [ -z "$STOP_CMD" ]; then
  MANAGED=$(cat "$DIR/logs/api.pid" 2>/dev/null || true)
  FOREIGN=$(ps -eo pid=,user=,args= | grep -F "$DIR/" | grep -E "uvicorn|curator_llm" | grep -v grep | awk -v m="$MANAGED" '$1 != m' | head -3 || true)
  [ -z "$FOREIGN" ] || fail "a service is already running from $DIR that this script did not start and cannot stop: $(echo "$FOREIGN" | cut -c1-160 | tr '\n' ';'). Stop it first, or deploy to another --path and --port."
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    [ -n "$MANAGED" ] && kill -0 "$MANAGED" 2>/dev/null || fail "port $PORT is already in use by another process. Use --port."
  fi
fi
# The server's existing .env must carry what the service needs at start-up (it connects to Neo4j and verifies tokens against ws).
if [ -f "$DIR/.env" ]; then
  if [ -r "$DIR/.env" ]; then
    NEED_KEYS=""
    for k in ANTHROPIC_API_KEY WS_BASE_URL REACTOME_NEO4J_URI REACTOME_NEO4J_USER REACTOME_NEO4J_PWD; do
      grep -qE "^[[:space:]]*$k=.+" "$DIR/.env" || NEED_KEYS="$NEED_KEYS $k"
    done
    if [ -n "$NEED_KEYS" ]; then
      if [ "$RESTART" = 1 ]; then
        fail "$DIR/.env has no value for:$NEED_KEYS. The service would not start. Add them (see .env.example) or use --no-restart to deploy the code only."
      else
        echo "!!! Note: $DIR/.env has no value for:$NEED_KEYS. Add them before starting the service." >&2
      fi
    fi
  else
    echo "!!! Note: $DIR/.env is not readable by $(whoami); cannot check it, and a service started by this user could not read it." >&2
  fi
fi
command -v curl >/dev/null 2>&1 || fail "curl is not installed on the server (needed for the health check)."
KIND=""
if [ -n "$USE_VENV" ]; then                          # the server already has its venv: just check it can run the service
  VPY="$DIR/$USE_VENV/bin/python"
  [ -x "$VPY" ] || fail "no Python at $VPY (--use-venv expects an existing venv)."
  MISSING=$("$VPY" - <<'PY' 2>/dev/null || echo "?"
import importlib.util
need = {"fastapi": "fastapi", "uvicorn": "uvicorn", "anthropic": "anthropic", "langgraph": "langgraph",
        "langchain_anthropic": "langchain-anthropic", "langchain_core": "langchain-core",
        "langchain_text_splitters": "langchain-text-splitters", "rapidfuzz": "rapidfuzz", "jsonpatch": "jsonpatch",
        "multipart": "python-multipart", "neo4j": "neo4j", "dotenv": "python-dotenv", "pydantic": "pydantic",
        "requests": "requests", "pymupdf": "PyMuPDF"}
print(" ".join(pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None))
PY
)
  [ "$MISSING" != "?" ] || fail "could not run $VPY to check its packages."
  [ -z "$MISSING" ] || fail "the venv $DIR/$USE_VENV lacks: $MISSING. Install them there first (pip install -r requirements-service.txt)."
  KIND=existing
  INSTALL=0
  echo "Python environment: existing venv $DIR/$USE_VENV ($("$VPY" --version 2>&1)), all required packages present"
elif [ "$INSTALL" = 1 ] || [ "$RESTART" = 1 ]; then
  case "$MODE" in
    venv)  python_ok && KIND=venv || fail "'$PY' is not a usable Python 3.10+ with the venv module (on Debian/Ubuntu: install python3-venv). Or use --conda NAME." ;;
    conda) CONDA="$(find_conda)" && KIND=conda || fail "conda was not found. Pass --conda-bin /path/to/conda." ;;
    *)     if python_ok; then KIND=venv
           elif CONDA="$(find_conda)"; then KIND=conda
           else fail "No usable Python 3.10+ (with venv) and no conda found. Install one, or pass --python / --conda-bin."; fi ;;
  esac
fi
[ "$KIND" = conda ] && CONDA="${CONDA:-$(find_conda)}"
[ "$KIND" = conda ] && echo "Python environment: conda env '$CONDA_ENV' (using $CONDA)"
[ "$KIND" = venv ]  && echo "Python environment: venv ($("$PY" --version 2>&1), $DIR/venv)"

# ---- deploy ----
mkdir -p "$DIR"; cd "$DIR"

if [ "$RESTART" = 1 ]; then
  if [ -n "$STOP_CMD" ]; then echo "Stopping: $STOP_CMD"; bash -c "$STOP_CMD" || true
  elif [ -x scripts/run_api.sh ]; then scripts/run_api.sh --stop || true; fi
fi

if [ -d curator_llm ]; then                       # keep the last version so a bad deploy can be undone
  tar -czf previous-code.tar.gz curator_llm reactome_llm scripts resources run_*.py requirements*.txt 2>/dev/null || true
fi
rm -rf curator_llm reactome_llm scripts           # no stale modules from older versions; .env, data/, logs/, venv/ stay
tar -xzf "$ARCHIVE"
rm -f "$ARCHIVE"
chmod +x scripts/*.sh
if [ "$GROUP_W" = 1 ]; then      # shared install: the next deployer may be another user in the group
  chmod -R g+rwX curator_llm reactome_llm scripts resources run_extraction.py run_merge.py run_review.py requirements-service.txt README.md .env.example 2>/dev/null || true
fi
mkdir -p data logs

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">>> Created $DIR/.env from .env.example. EDIT IT before using the service (ANTHROPIC_API_KEY, WS_BASE_URL, REACTOME_NEO4J_*)."
fi

NEW=$(sha requirements-service.txt)
if [ "$INSTALL" = 1 ]; then
  if [ "$KIND" = venv ]; then
    [ -d venv ] || "$PY" -m venv venv
    STAMP=venv/.requirements.sha
    if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$NEW" ]; then
      echo "Installing requirements-service.txt into the venv ..."
      venv/bin/pip install --quiet --upgrade pip
      venv/bin/pip install --quiet -r requirements-service.txt
      echo "$NEW" > "$STAMP"
    else echo "Requirements unchanged, not reinstalling."; fi
  else
    if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
      echo "Creating conda env '$CONDA_ENV' (Python 3.10, conda-forge) ..."
      "$CONDA" create -y -q -n "$CONDA_ENV" -c conda-forge --override-channels python=3.10 pip
    fi
    STAMP=".requirements.conda-$CONDA_ENV.sha"
    if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$NEW" ]; then
      echo "Installing requirements-service.txt into conda env '$CONDA_ENV' ..."
      "$CONDA" run -n "$CONDA_ENV" --no-capture-output python -m pip install --quiet -r requirements-service.txt
      echo "$NEW" > "$STAMP"
    else echo "Requirements unchanged, not reinstalling."; fi
  fi
fi

if [ "$RESTART" = 1 ]; then
  if [ -n "$START_CMD" ]; then echo "Starting: $START_CMD"; bash -c "$START_CMD"
  else
    if [ "$KIND" = venv ]; then . venv/bin/activate
    elif [ "$KIND" = existing ]; then . "$USE_VENV/bin/activate"
    else . "$("$CONDA" info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; fi
    HOST="$BIND" PORT="$PORT" scripts/run_api.sh --daemon
  fi
  code=000                                  # the first start imports a lot: wait up to a minute for it to answer
  for _ in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/llm/sessions" || true)
    [ "$code" != 000 ] && break
    sleep 1
  done
  if [ "$code" = 401 ]; then echo "Service is up (HTTP 401 = it is running and asking for a token)."
  else echo "!!! Health check returned HTTP ${code:-none} after 60 s. See $DIR/logs/api.log" >&2; tail -n 15 logs/api.log >&2 || true; exit 1; fi
fi
echo "Deployed to $DIR"
REMOTE
}

if [[ "$DRY" -eq 1 ]]; then
  echo
  echo "Dry run: nothing was sent. It would run:"
  echo "  scp $ARCHIVE_NAME $HOST:/tmp/"
  echo "  ssh $HOST bash -s -- '$REMOTE_PATH' /tmp/$ARCHIVE_NAME $REMOTE_PY $BIND $PORT $INSTALL $RESTART $ENV_MODE $CONDA_ENV '$CONDA_BIN' '$USE_VENV' '$STOP_CMD' '$START_CMD' $GROUP_W   # script below"
  echo "-------- remote script --------"
  remote_script
  echo "-------- archive contents --------"
  tar -tzf "$ARCHIVE" | sed 's/^/  /' | head -80
  exit 0
fi

# ── 5. upload and run (one connection, so a password is asked for once) ────
SSH_COMMON=(-o ControlMaster=auto -o "ControlPath=/tmp/deploy-ssh-%C" -o ControlPersist=120 ${SSH_OPTS:-})   # short path: macOS limits socket paths to 104 bytes
echo "Uploading to $HOST ..."
scp "${SSH_COMMON[@]}" "$ARCHIVE" "$HOST:/tmp/$ARCHIVE_NAME"
# ssh joins its arguments into one string for the remote shell, which silently drops empty ones and splits on spaces:
# quote every argument (printf %q keeps an empty one as '') so each arrives in its own position
REMOTE_ARGS=("$REMOTE_PATH" "/tmp/$ARCHIVE_NAME" "$REMOTE_PY" "$BIND" "$PORT" "$INSTALL" "$RESTART" "$ENV_MODE" "$CONDA_ENV" "$CONDA_BIN" "$USE_VENV" "$STOP_CMD" "$START_CMD" "$GROUP_W")
remote_script | ssh "${SSH_COMMON[@]}" "$HOST" "bash -s -- $(printf '%q ' "${REMOTE_ARGS[@]}")"
ssh "${SSH_COMMON[@]}" -O exit "$HOST" 2>/dev/null || true
