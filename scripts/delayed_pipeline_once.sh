#!/usr/bin/env bash
set -euo pipefail

REPO="/Users/remyroche/Documents/Ares"
cd "$REPO"

mkdir -p logs
LAUNCH_EPOCH="${LAUNCH_EPOCH:-$(date +%s)}"
LOG_FILE="${LOG_FILE:-logs/delayed_pipeline_once_$(date +%Y%m%d_%H%M%S).log}"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] delayed runner started"
echo "LAUNCH_EPOCH=$LAUNCH_EPOCH"
echo "LOG_FILE=$LOG_FILE"
echo "Sleeping 7200s before check/run..."
sleep 7200

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] wake-up: checking TBM output freshness"
TBM_FILE="reports/tbm_optimized.csv"
FRESH=false
if [[ -f "$TBM_FILE" ]]; then
  MTIME=$(stat -f %m "$TBM_FILE" 2>/dev/null || echo 0)
  if [[ "$MTIME" -gt "$LAUNCH_EPOCH" ]]; then
    FRESH=true
  fi
fi

echo "Initial freshness check: file_exists=$([[ -f "$TBM_FILE" ]] && echo yes || echo no), fresh_since_launch=$FRESH"

if [[ "$FRESH" != "true" ]]; then
  echo "TBM output not fresh yet; polling up to 2h (12 x 10min)"
  for i in {1..12}; do
    sleep 600
    if [[ -f "$TBM_FILE" ]]; then
      MTIME=$(stat -f %m "$TBM_FILE" 2>/dev/null || echo 0)
      if [[ "$MTIME" -gt "$LAUNCH_EPOCH" ]]; then
        FRESH=true
        echo "TBM output became fresh on poll #$i"
        break
      fi
    fi
    echo "poll #$i: still waiting for fresh TBM file"
  done
fi

if [[ "$FRESH" != "true" ]]; then
  echo "WARNING: TBM file still not fresh; proceeding anyway per one-shot schedule"
fi

START_TS=$(date +%s)
TS=$(ls -1 data/features | sort | tail -n1)
echo "Using TS=$TS"

run_step() {
  local name="$1"
  shift
  local attempt
  for attempt in 1 2; do
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] STEP $name attempt $attempt"
    if "$@"; then
      echo "STEP $name succeeded"
      return 0
    fi
    echo "STEP $name failed (attempt $attempt)"
    if [[ "$attempt" -lt 2 ]]; then
      echo "Sleeping 60s before retrying $name"
      sleep 60
    fi
  done
  echo "STEP $name failed after retries"
  return 1
}

run_step labels python3 extreme_price_movements/run_pipeline.py labels --ts "$TS" --horizons 2 4 8
run_step train python3 extreme_price_movements/run_pipeline.py train --ts "$TS"
run_step train_meta python3 extreme_price_movements/run_pipeline.py train_meta --ts "$TS"

END_TS=$(date +%s)
echo "Total elapsed: $((END_TS - START_TS))s"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] delayed runner complete"
