#!/usr/bin/env bash
# Start the curator-tool-llm REST API (curator_llm.main:app) with uvicorn.
# Works from any directory, locally and on a server.
#
#   scripts/run_api.sh                  # local: 127.0.0.1:8000, foreground
#   scripts/run_api.sh --reload         # local, restart on code changes
#   HOST=0.0.0.0 scripts/run_api.sh     # server: listen on all interfaces
#   scripts/run_api.sh --daemon         # server: background, log to logs/api.log, pid in logs/api.pid
#   scripts/run_api.sh --stop           # stop the daemon
#
# Environment (all optional):
#   HOST (127.0.0.1)   PORT (8000)   WORKERS (1)   CONDA_ENV (reactome_ai)
#   SNAPSHOT_MODE=replay, SESSION_STORE=mongo, etc. are read from .env / the shell as usual.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
CONDA_ENV="${CONDA_ENV:-reactome_ai}"
LOG_DIR="$ROOT/logs"
PID_FILE="$LOG_DIR/api.pid"

daemon=0
extra=()
for arg in "$@"; do
  case "$arg" in
    --daemon) daemon=1 ;;
    --stop)
      if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        kill "$(cat "$PID_FILE")" && rm -f "$PID_FILE" && echo "Stopped."
      else
        rm -f "$PID_FILE"; echo "Not running."
      fi
      exit 0 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) extra+=("$arg") ;;   # passed through to uvicorn (e.g. --reload, --log-level debug)
  esac
done

# Use the active environment if it has uvicorn; otherwise try the conda env.
if ! python -c 'import uvicorn' 2>/dev/null; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
  python -c 'import uvicorn' 2>/dev/null || {
    echo "uvicorn not found. Activate your environment or set CONDA_ENV." >&2; exit 1; }
fi

cmd=(python -m uvicorn curator_llm.main:app --host "$HOST" --port "$PORT")
[[ "$WORKERS" -gt 1 ]] && cmd+=(--workers "$WORKERS")
cmd+=(${extra[@]+"${extra[@]}"})   # the ${x[@]+...} form keeps bash 3.2 (macOS) happy under set -u

if [[ "$daemon" -eq 1 ]]; then
  mkdir -p "$LOG_DIR"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (pid $(cat "$PID_FILE"))." >&2; exit 1
  fi
  nohup "${cmd[@]}" >>"$LOG_DIR/api.log" 2>&1 &
  echo $! >"$PID_FILE"
  echo "Started on $HOST:$PORT (pid $!), log: $LOG_DIR/api.log"
else
  exec "${cmd[@]}"
fi
