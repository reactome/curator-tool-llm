#!/usr/bin/env bash
# Redeploy curator-tool-llm to the production webbench server (aws_curator_new) in one command.
#   scripts/deploy-production.sh              # package, check, upload, stop, replace code, start, health check
#   scripts/deploy-production.sh --dry-run    # build and check only
# That server runs the app from the folder's own .venv through ../manage-apps.sh (127.0.0.1:5000, Apache proxies /llm/),
# and several people deploy to it, so: use the existing venv, stop and start with manage-apps.sh, make the code
# group-writable. Needs REACTOME_NEO4J_*, ANTHROPIC_API_KEY and WS_BASE_URL in the server's .env (checked first).
exec "$(dirname "${BASH_SOURCE[0]}")/deploy.sh" "${DEPLOY_HOST:-aws_curator_new}" \
  --use-venv .venv --group-writable --port 5000 \
  --stop-cmd '../manage-apps.sh stop llm' --start-cmd '../manage-apps.sh start llm' "$@"
