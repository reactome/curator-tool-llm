#!/usr/bin/env bash
# Annotate every PDF in a folder through the running curator-tool-llm REST API.
# For each PDF: upload -> poll the job until done -> save the staged instances and the open issues.
#
#   scripts/annotate_pdfs.sh PDF_DIR [OUT_DIR]
#
# Output (OUT_DIR defaults to PDF_DIR/annotations):
#   <name>.staged_instances.json   the draft, as the curator tool's "load staged instances" takes it
#   <name>.issues.json             what needs a curator's attention
#   <name>.session                 the session id (so you can open it in the frontend / chat)
# PDFs that already have a .staged_instances.json in OUT_DIR are skipped.
#
# Environment:
#   WS_USERNAME, WS_PASSWORD   curator-tool-ws account with the curator role (prompted if unset)
#   TOKEN                      or give a ready bearer token instead
#   FOCUS                      optional gene to focus the extraction on
#   API (http://localhost:8000/api/llm)   WS (http://localhost:9090)   POLL (15 seconds)
#
# Requires: curl, jq. Start the service first with scripts/run_api.sh.
set -euo pipefail

[[ $# -ge 1 && $1 != -h && $1 != --help ]] || { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

PDF_DIR="$1"
OUT_DIR="${2:-$PDF_DIR/annotations}"
API="${API:-http://localhost:8000/api/llm}"
WS="${WS:-http://localhost:9090}"
POLL="${POLL:-15}"
FOCUS="${FOCUS:-}"

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
[[ -d "$PDF_DIR" ]] || { echo "Not a folder: $PDF_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

login() {
  if [[ -z "${WS_USERNAME:-}" ]]; then read -r -p "ws username: " WS_USERNAME; fi
  if [[ -z "${WS_PASSWORD:-}" ]]; then read -r -s -p "ws password: " WS_PASSWORD; echo; fi
  TOKEN=$(curl -sf -X POST "$WS/api/auth/login" -H 'Content-Type: application/json' \
            -d "$(jq -n --arg u "$WS_USERNAME" --arg p "$WS_PASSWORD" '{username:$u,password:$p}')" | tr -d '"') \
    || { echo "Login failed" >&2; exit 1; }
}

# Tokens last about five minutes, so re-login before each call when we have credentials.
auth() {
  if [[ -n "${WS_USERNAME:-}" && -n "${WS_PASSWORD:-}" ]]; then login; fi
  printf 'Authorization: Bearer %s' "$TOKEN"
}

if [[ -z "${TOKEN:-}" ]]; then login; fi

shopt -s nullglob nocaseglob
pdfs=("$PDF_DIR"/*.pdf)
shopt -u nocaseglob
[[ ${#pdfs[@]} -gt 0 ]] || { echo "No PDFs in $PDF_DIR" >&2; exit 1; }

ok=0; failed=()
for pdf in "${pdfs[@]}"; do
  name="$(basename "${pdf%.*}")"
  if [[ -s "$OUT_DIR/$name.staged_instances.json" ]]; then echo "[$name] already done, skipping"; continue; fi

  echo "[$name] uploading"
  args=(-F "file=@$pdf")
  [[ -n "$FOCUS" ]] && args+=(-F "focus=$FOCUS")
  resp=$(curl -sf -X POST "$API/sessions/upload" -H "$(auth)" "${args[@]}") \
    || { echo "[$name] upload failed"; failed+=("$name"); continue; }
  sid=$(jq -r .sessionId <<<"$resp"); jid=$(jq -r .jobId <<<"$resp")
  echo "$sid" > "$OUT_DIR/$name.session"

  status=queued
  while [[ $status == queued || $status == running ]]; do
    sleep "$POLL"
    job=$(curl -sf "$API/jobs/$jid" -H "$(auth)") || { status=failed; break; }
    status=$(jq -r .status <<<"$job")
    echo "[$name] $status: $(jq -r '.progress // ""' <<<"$job")"
  done

  if [[ $status != done ]]; then
    echo "[$name] FAILED: $(jq -r '.error // "unknown"' <<<"${job:-{\}}")"; failed+=("$name"); continue
  fi

  curl -sf "$API/sessions/$sid/export" -H "$(auth)" > "$OUT_DIR/$name.staged_instances.json"
  curl -sf "$API/sessions/$sid/issues?status=open" -H "$(auth)" > "$OUT_DIR/$name.issues.json"
  echo "[$name] done: $(jq 'length' "$OUT_DIR/$name.issues.json" 2>/dev/null || echo '?') open issues, session $sid"
  ok=$((ok + 1))
done

echo "Finished: $ok annotated, ${#failed[@]} failed${failed[*]:+ (${failed[*]})}. Output in $OUT_DIR"
[[ ${#failed[@]} -eq 0 ]]
