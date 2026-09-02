#!/usr/bin/env bash
set -u

# Persistent scheduler for the sealed strict-R3 hourly entry producer.  The
# producer itself owns all decision-time freshness, lineage and fail-closed
# checks; this wrapper provides only singleton process management and logging.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

# Never reuse project-local bytecode caches: a stale or interrupted cache must
# not delay a fresh decision.  This is runtime-only and leaves all source,
# models, feature states, and sealed artifacts untouched.
PYTHON_CACHE_PREFIX="${STRICT_R3_PYTHONPYCACHEPREFIX:-/private/tmp/ares_pycache_live}"
mkdir -p "$PYTHON_CACHE_PREFIX"

# The persistent producer is warmed while idle, but a supervisor restart can
# still land immediately before a candle boundary.  Recursive bytecode-cache
# writes for the full research dependency graph can take minutes on a cold
# filesystem.  Disable those non-semantic writes so a restart imports from
# the sealed sources without blocking source refresh or order authority.
export PYTHONDONTWRITEBYTECODE=1

INFERENCE_BUNDLE="${STRICT_R3_HOURLY_INFERENCE_BUNDLE:?set STRICT_R3_HOURLY_INFERENCE_BUNDLE}"
EXECUTION_BUNDLE="${STRICT_R3_HOURLY_EXECUTION_BUNDLE:?set STRICT_R3_HOURLY_EXECUTION_BUNDLE}"
LIVE_STATE="${STRICT_R3_HOURLY_LIVE_STATE:?set STRICT_R3_HOURLY_LIVE_STATE}"
BOOTSTRAP_PREDECESSOR="${STRICT_R3_HOURLY_BOOTSTRAP_PREDECESSOR:?set STRICT_R3_HOURLY_BOOTSTRAP_PREDECESSOR}"
LOG_PATH="${STRICT_R3_HOURLY_LOG:-logs/strict_r3_live_hourly_entry_producer.log}"
PID_PATH="${STRICT_R3_HOURLY_PID:-/private/tmp/strict_r3_live_hourly_entry_producer.pid}"
POLL_SECONDS="${STRICT_R3_HOURLY_POLL_SECONDS:-3}"
SETTLED_RETRY_SCHEDULE_SECONDS="${STRICT_R3_HOURLY_SETTLED_RETRY_SCHEDULE_SECONDS:-30,60,120,180}"
FAILED_RETRY_SECONDS="${STRICT_R3_HOURLY_FAILED_RETRY_SECONDS:-30}"
START_NEXT_FRESH_HOUR="${STRICT_R3_HOURLY_START_NEXT_FRESH_HOUR:-0}"

mkdir -p "$(dirname "$LOG_PATH")"
if [[ -f "$PID_PATH" ]]; then
  old_pid="$(sed -n '1p' "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_command="$(ps -p "$old_pid" -o command= 2>/dev/null || true)"
    if [[ "$old_command" == *"run_strict_r3_live_hourly_entry_producer.py"* || "$old_command" == *"run_strict_r3_live_hourly_entry_producer_loop.sh"* ]]; then
      exit 0
    fi
  fi
fi
echo "$$" > "$PID_PATH"
child_pid=""
stopping=0
start_args=()
if [[ "$START_NEXT_FRESH_HOUR" == "1" ]]; then
  start_args+=(--start-next-fresh-hour)
fi

shutdown() {
  stopping=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap shutdown HUP INT TERM
trap 'rm -f "$PID_PATH"' EXIT

# The producer should normally never return: it owns the hourly scheduling and
# all in-hour retries.  This outer parent handles only an unexpected process
# exit, then re-enters the same sealed command.  It never creates a second
# producer and cannot re-execute a completed receipt.
while [[ "$stopping" -eq 0 ]]; do
  if (( ${#start_args[@]} )); then
    env NUMBA_CACHE_DIR=/private/tmp/ares_numba_cache \
      MPLCONFIGDIR=/private/tmp/ares_matplotlib \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPYCACHEPREFIX="$PYTHON_CACHE_PREFIX" \
      PYTHONUNBUFFERED=1 \
      python3 scripts/run_strict_r3_live_hourly_entry_producer.py \
        --inference-bundle "$INFERENCE_BUNDLE" \
        --execution-bundle "$EXECUTION_BUNDLE" \
        --live-state "$LIVE_STATE" \
        --bootstrap-previous-run "$BOOTSTRAP_PREDECESSOR" \
        --loop "${start_args[@]}" --poll-seconds "$POLL_SECONDS" \
        --failed-retry-seconds "$FAILED_RETRY_SECONDS" \
        --settled-retry-schedule-seconds "$SETTLED_RETRY_SCHEDULE_SECONDS" >> "$LOG_PATH" 2>&1 &
  else
    env NUMBA_CACHE_DIR=/private/tmp/ares_numba_cache \
      MPLCONFIGDIR=/private/tmp/ares_matplotlib \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPYCACHEPREFIX="$PYTHON_CACHE_PREFIX" \
      PYTHONUNBUFFERED=1 \
      python3 scripts/run_strict_r3_live_hourly_entry_producer.py \
        --inference-bundle "$INFERENCE_BUNDLE" \
        --execution-bundle "$EXECUTION_BUNDLE" \
        --live-state "$LIVE_STATE" \
        --bootstrap-previous-run "$BOOTSTRAP_PREDECESSOR" \
        --loop --poll-seconds "$POLL_SECONDS" \
        --failed-retry-seconds "$FAILED_RETRY_SECONDS" \
        --settled-retry-schedule-seconds "$SETTLED_RETRY_SCHEDULE_SECONDS" >> "$LOG_PATH" 2>&1 &
  fi
  child_pid="$!"
  wait "$child_pid"
  child_status="$?"
  child_pid=""
  if [[ "$stopping" -ne 0 ]]; then
    break
  fi
  printf '{"event":"producer_unexpected_exit","exit_code":%s,"restart_delay_seconds":5}\n' "$child_status" >> "$LOG_PATH"
  sleep 5
done
