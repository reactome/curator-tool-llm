#!/usr/bin/env bash
# Write an annotation (<name>.annotation.json) as a Cytoscape desktop network (.cyjs) plus a visual style (.style.xml).
# In Cytoscape: File > Import > Network from File (the .cyjs), File > Import > Styles from File (the .xml), style "Paper2Path".
#
#   scripts/export_cytoscape.sh annotations/AML_drug_response.annotation.json
#   scripts/export_cytoscape.sh annotations/*.annotation.json --out cytoscape/
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

exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m curator_llm.cytoscape_export "$@"
