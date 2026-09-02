#!/usr/bin/env bash
set -u

# Read-only xx:10 reporting loop.  This intentionally is not part of the live
# execution chain: it cannot rescore, fetch market data, alter state, or write
# to the exchange.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

RUNTIME_TAG="${STRICT_R3_REPORT_RUNTIME_TAG:?set STRICT_R3_REPORT_RUNTIME_TAG}"
LOG_PATH="${STRICT_R3_REPORT_LOG:-logs/strict_r3_live_hourly_report.log}"
PID_PATH="${STRICT_R3_REPORT_PID:-/private/tmp/strict_r3_live_hourly_report.pid}"
POLL_SECONDS="${STRICT_R3_REPORT_POLL_SECONDS:-5}"

mkdir -p "$(dirname "$LOG_PATH")"
if [[ -f "$PID_PATH" ]]; then
  old_pid="$(sed -n '1p' "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_command="$(ps -p "$old_pid" -o command= 2>/dev/null || true)"
    if [[ "$old_command" == *"run_strict_r3_live_hourly_report_loop.sh"* ]]; then
      exit 0
    fi
  fi
fi
echo "$$" > "$PID_PATH"
trap 'rm -f "$PID_PATH"' EXIT

last_reported=""
last_attempt_slot=""
while true; do
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  minute="$(date -u +%M)"
  second="$(date -u +%S)"
  decision="$(date -u +%Y-%m-%dT%H:00:00Z)"
  # Start from xx:10.  Retry every two minutes until xx:20 only if the producer
  # is still settling its bounded source retries.  A complete report is immutable.
  attempt_slot="${decision}:${minute}"
  if [[ "$minute" -ge 10 && "$minute" -lt 20 && "$decision" != "$last_reported" && "$attempt_slot" != "$last_attempt_slot" && $((10#$minute % 2)) -eq 0 ]]; then
    if python3 scripts/report_strict_r3_live_candle.py \
      --runtime-tag "$RUNTIME_TAG" --decision-ts "$decision" >> "$LOG_PATH" 2>&1; then
      last_reported="$decision"
    fi
    last_attempt_slot="$attempt_slot"
  fi
  sleep "$POLL_SECONDS"
done
