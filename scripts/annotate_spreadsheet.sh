#!/usr/bin/env bash
# Annotate the papers listed in a spreadsheet (.xlsx): a PDF where the sheet names one, else the PubMed ID, whose PubMed
# Central full text is downloaded into a PMC/ folder next to the spreadsheet. Results go to annotations/ next to it.
#
#   scripts/annotate_spreadsheet.sh papers.xlsx --list            # what would be done, nothing changed
#   scripts/annotate_spreadsheet.sh papers.xlsx --download-only   # PMC full text only
#   scripts/annotate_spreadsheet.sh papers.xlsx --rows 1-3        # try a few papers first
#   scripts/annotate_spreadsheet.sh papers.xlsx                   # all of them; run it again to continue after a stop
#
# `--help` lists every option. Environment (optional): CONDA_ENV (default reactome_ai), used when the active Python
# lacks the dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-reactome_ai}"

if ! python -c 'import dotenv, anthropic, pydantic' 2>/dev/null; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
  python -c 'import dotenv, anthropic, pydantic' 2>/dev/null || {
    echo "The pipeline's Python packages were not found. Activate your environment or set CONDA_ENV." >&2; exit 1; }
fi

# the paths you give stay relative to where you run this; only the package is found through ROOT
exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m curator_llm.spreadsheet "$@"
