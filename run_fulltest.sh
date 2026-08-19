#!/bin/bash
# Full-pipeline test: PMCID -> PubMedFetcher (JATS) -> extraction -> merge.
#
# Writes results/<stem>_fulltest_<MMDD_HHMMSS>_{extraction,merged}.json — the
# run-stamped tag keeps this run from touching any existing result file, so earlier runs
# stay around to compare against, and run_extraction.py never has to refuse an overwrite.
#
# Usage:  ./run_fulltest.sh [PMID|PMCID]   (default PMC4003245 = the PINK1 paper)
#         TAG=v3 ./run_fulltest.sh         (pin the tag instead of stamping it)
# Watch:  tail -f results/fulltest_pipeline.log   (symlink -> the newest run's log)

set -o pipefail
PY=~/miniconda3/envs/paperqa/bin/python
ID="${1:-PMC4003245}"
# A run-stamped tag, so a re-run can never collide with an earlier one. run_extraction.py
# refuses to overwrite an existing extraction and reports that refusal as a SKIP with exit
# 0 — under a fixed tag a re-run therefore looked like a success here, and stage 2 went on
# to merge the PREVIOUS run's extraction. Set TAG=... to pin a name deliberately.
TAG="${TAG:-fulltest_$(date '+%m%d_%H%M%S')}"
# Mirror fetcher.output_stem(): a bare PMID's stem is "pmid<id>", not "<id>". Get this
# wrong and the count below misses the file stage 1 just wrote.
STEM="$(echo "$ID" | tr '[:upper:]' '[:lower:]')"
[[ "$ID" =~ ^[0-9]{4,9}$ ]] && STEM="pmid${STEM}"
EXTRACTION="results/${STEM}_${TAG}_extraction.json"
# Stamped like the result files, so a run's log survives the next run — it carries the
# per-stage token totals, which is what makes two runs comparable.
LOG="results/${TAG}_pipeline.log"
LATEST=results/fulltest_pipeline.log

cd ~/curator-tool-llm || exit 1
: > "$LOG"
# Keep one stable path to tail: the stamped name is unpredictable, and an editor holding
# results/fulltest_pipeline.log open follows the symlink to whichever run is current.
ln -sfn "$(basename "$LOG")" "$LATEST"

say() { echo "$@" | tee -a "$LOG"; }
stamp() { date '+%H:%M:%S'; }

say "=============================================================="
say " FULL PIPELINE TEST — $ID"
say " started $(date)"
say " log: $LOG"
say "=============================================================="

# ---- stage 1: fetch + extract ------------------------------------------------
say ""
say "[$(stamp)] STAGE 1/2  fetch (JATS) + extract  -> $EXTRACTION"
T1=$SECONDS
$PY run_extraction.py "$ID" --tag "$TAG" 2>&1 | tee -a "$LOG"
RC=$?
E1=$((SECONDS - T1))
if [ $RC -ne 0 ]; then
    say ""
    say "[$(stamp)] STAGE 1 FAILED (exit $RC) after ${E1}s — STOPPING, merge not run"
    exit $RC
fi

# Stage 1 exits 0 even when it extracted nothing — a PMID PMC has no full text for, or a
# refused overwrite, are reported as skips. So exit status alone does not mean the file
# below is this run's work; require the file to be there and to hold reactions before
# spending the judge calls on it.
if [ ! -f "$EXTRACTION" ]; then
    say ""
    say "[$(stamp)] STAGE 1 wrote no extraction file (skipped or no full text) — STOPPING"
    say "           expected: $EXTRACTION"
    exit 1
fi
COUNT=$($PY -c "import json;print(len(json.load(open('$EXTRACTION'))))" 2>/dev/null)
if [ -z "$COUNT" ] || [ "$COUNT" -eq 0 ] 2>/dev/null; then
    say ""
    say "[$(stamp)] STAGE 1 produced no reactions (or unreadable output) — STOPPING"
    exit 1
fi
say "[$(stamp)] STAGE 1 ok in ${E1}s — $COUNT reaction(s) extracted"

# ---- stage 2: merge ----------------------------------------------------------
say ""
say "[$(stamp)] STAGE 2/2  semantic merge + subsection consolidation"
T2=$SECONDS
$PY run_merge.py "$EXTRACTION" 2>&1 | tee -a "$LOG"
RC=$?
E2=$((SECONDS - T2))
if [ $RC -ne 0 ]; then
    say ""
    say "[$(stamp)] STAGE 2 FAILED (exit $RC) after ${E2}s — STOPPING"
    exit $RC
fi

say ""
say "=============================================================="
say " DONE  $(date)"
say "   stage 1 (fetch+extract): ${E1}s"
say "   stage 2 (merge):         ${E2}s"
say "   total:                   $((E1 + E2))s"
say " token totals are on the [usage] lines above, per stage"
say "=============================================================="
