#!/usr/bin/env bash
# Add annotation files written by the command line (<name>.annotation.json) to a curator's annotations, so the front-end
# (Paper2Path, "Your annotations") can open them. No restart of the service is needed.
#
#   scripts/import_annotation.sh annotations/wee1.annotation.json --owner wgm
#   scripts/import_annotation.sh annotations/*.annotation.json --owner marija --session-dir /path/to/data/sessions
#
# Environment (optional): CONDA_ENV (default reactome_ai), used when the active Python lacks the dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-reactome_ai}"

if ! python -c 'import dotenv, pydantic' 2>/dev/null; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
  python -c 'import dotenv, pydantic' 2>/dev/null || {
    echo "The service's Python packages were not found. Activate your environment or set CONDA_ENV." >&2; exit 1; }
fi

exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m curator_llm.annotation_files "$@"
