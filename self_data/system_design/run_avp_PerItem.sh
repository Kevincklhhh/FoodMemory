#!/usr/bin/env bash
# Launcher for 06_avp_round1_remaining_CandList_HOI_PerItem.py.
#
# Usage:
#   ./system_design/run_avp_PerItem.sh smoke              # planner-only on 1 session
#   ./system_design/run_avp_PerItem.sh one <session>      # full run on one session
#   ./system_design/run_avp_PerItem.sh planner-all        # planner-only across all sessions
#   ./system_design/run_avp_PerItem.sh all                # full run across all sessions (with resume)
#   ./system_design/run_avp_PerItem.sh resume             # same as 'all' — just re-runs pending
#
# Override participant, tag, model via env:
#   PARTICIPANT=kailai TAG=PerItem_v1 MODEL=gpt-5.4 ./system_design/run_avp_PerItem.sh all
set -euo pipefail

cd "$(dirname "$0")/.."

PARTICIPANT="${PARTICIPANT:-kailai}"
TAG="${TAG:-PerItem_v1}"
MODEL="${MODEL:-gpt-5.4}"
SMOKE_SESSION="${SMOKE_SESSION:-20260310-195710}"
SCRIPT="system_design/06_avp_round1_remaining_CandList_HOI_PerItem.py"

mode="${1:-smoke}"

common_args=(--participant "$PARTICIPANT" --tag "$TAG" --model "$MODEL")

case "$mode" in
  smoke)
    echo "[smoke] planner-only on $SMOKE_SESSION"
    python "$SCRIPT" "${common_args[@]}" --session "$SMOKE_SESSION" --planner-only
    ;;
  one)
    sess="${2:?usage: run_avp_PerItem.sh one <session>}"
    echo "[one] full run on $sess"
    python "$SCRIPT" "${common_args[@]}" --session "$sess"
    ;;
  planner-all)
    echo "[planner-all] planner-only across all sessions"
    python "$SCRIPT" "${common_args[@]}" --all --planner-only --resume
    ;;
  all|resume)
    echo "[$mode] full run across all sessions (resume-friendly)"
    python "$SCRIPT" "${common_args[@]}" --all --resume
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    echo "Run with: smoke | one <session> | planner-all | all | resume" >&2
    exit 2
    ;;
esac
