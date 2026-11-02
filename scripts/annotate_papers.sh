#!/usr/bin/env bash
# Annotate one or more papers (PDF files and/or PubMed IDs) from the command line. It runs the pipeline directly on
# this machine: no REST service, login or upload is needed (it needs the same .env, and Neo4j, as the service).
# By default ALL the papers go into ONE draft, merged across papers; --separate annotates each paper on its own.
#
#   scripts/annotate_papers.sh paper1.pdf paper2.pdf --pmid 24751536 --focus PINK1
#   scripts/annotate_papers.sh /path/to/pdf_folder --pmids "24751536 29168502" --out results_dir
#   scripts/annotate_papers.sh --pmid-file pmids.txt --name pink1_papers
#   scripts/annotate_papers.sh /path/to/pdf_folder --separate
#
# `scripts/annotate_papers.sh --help` lists the options and the result files. There is no limit on the number of papers.
# Environment (optional): CONDA_ENV (default reactome_ai), used when the active Python lacks the dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-reactome_ai}"

# Use the active environment if it has what the pipeline needs; otherwise try the conda env.
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
exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m curator_llm.cli "$@"
