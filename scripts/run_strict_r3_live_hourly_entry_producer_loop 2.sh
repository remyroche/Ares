#!/usr/bin/env bash
set -u

# Persistent scheduler for the sealed strict-R3 hourly entry producer.  The
# producer itself owns all decision-time freshness, lineage and fail-closed
# checks; this wrapper provides only singleton process management and logging.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

INFERENCE_BUNDLE="${STRICT_R3_HOURLY_INFERENCE_BUNDLE:?set STRICT_R3_HOURLY_INFERENCE_BUNDLE}"
EXECUTION_BUNDLE="${STRICT_R3_HOURLY_EXECUTION_BUNDLE:?set STRICT_R3_HOURLY_EXECUTION_BUNDLE}"
LIVE_STATE="${STRICT_R3_HOURLY_LIVE_STATE:?set STRICT_R3_HOURLY_LIVE_STATE}"
BOOTSTRAP_PREDECESSOR="${STRICT_R3_HOURLY_BOOTSTRAP_PREDECESSOR:?set STRICT_R3_HOURLY_BOOTSTRAP_PREDECESSOR}"
LOG_PATH="${STRICT_R3_HOURLY_LOG:-logs/strict_r3_live_hourly_entry_producer.log}"
PID_PATH="${STRICT_R3_HOURLY_PID:-/private/tmp/strict_r3_live_hourly_entry_producer.pid}"
POLL_SECONDS="${STRICT_R3_HOURLY_POLL_SECONDS:-3}"
SETTLED_RETRY_SCHEDULE_SECONDS="${STRICT_R3_HOURLY_SETTLED_RETRY_SCHEDULE_SECONDS:-30,60,120,180}"

mkdir -p "$(dirname "$LOG_PATH")"
if [[ -f "$PID_PATH" ]]; then
  old_pid="$(sed -n '1p' "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_command="$(ps -p "$old_pid" -o command= 2>/dev/null || true)"
    if [[ "$old_command" == *"run_strict_r3_live_hourly_entry_producer.py"* ]]; then
      exit 0
    fi
  fi
fi
echo "$$" > "$PID_PATH"
trap 'rm -f "$PID_PATH"' EXIT

exec env NUMBA_CACHE_DIR=/private/tmp/ares_numba_cache \
  MPLCONFIGDIR=/private/tmp/ares_matplotlib \
  PYTHONUNBUFFERED=1 \
  python3 scripts/run_strict_r3_live_hourly_entry_producer.py \
    --inference-bundle "$INFERENCE_BUNDLE" \
    --execution-bundle "$EXECUTION_BUNDLE" \
    --live-state "$LIVE_STATE" \
    --bootstrap-previous-run "$BOOTSTRAP_PREDECESSOR" \
    --loop --poll-seconds "$POLL_SECONDS" \
    --settled-retry-schedule-seconds "$SETTLED_RETRY_SCHEDULE_SECONDS" >> "$LOG_PATH" 2>&1
