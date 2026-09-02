#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

EXECUTION_BUNDLE="${STRICT_R3_EXECUTION_BUNDLE:?set STRICT_R3_EXECUTION_BUNDLE}"
LIVE_STATE="${STRICT_R3_LIVE_STATE:?set STRICT_R3_LIVE_STATE}"
OUT_ROOT="${STRICT_R3_MONITOR_OUT_ROOT:-data_perp/artifacts/strict_r3_kraken_live_position_monitor_v1}"
LOG_PATH="${STRICT_R3_MONITOR_LOG:-logs/strict_r3_live_position_monitor.log}"
PID_PATH="${STRICT_R3_MONITOR_PID:-logs/strict_r3_live_position_monitor.pid}"
INTERVAL_SECONDS="${STRICT_R3_MONITOR_INTERVAL_SECONDS:-60}"
SUBMIT_ORDERS="${STRICT_R3_MONITOR_SUBMIT_ORDERS:-0}"

mkdir -p "$(dirname "$LOG_PATH")" "$OUT_ROOT"
if [[ -f "$PID_PATH" ]]; then
  old_pid="$(sed -n '1p' "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_command="$(ps -p "$old_pid" -o command= 2>/dev/null || true)"
    if [[ "$old_command" == *"run_strict_r3_live_position_monitor.py"* ]]; then
      exit 0
    fi
  fi
fi
echo "$$" > "$PID_PATH"
trap 'rm -f "$PID_PATH"' EXIT

submit_args=()
if [[ "$SUBMIT_ORDERS" == "1" ]]; then submit_args+=(--submit-orders); fi
exec env NUMBA_CACHE_DIR=/private/tmp/ares_numba_cache \
    MPLCONFIGDIR=/private/tmp/ares_matplotlib \
    PYTHONUNBUFFERED=1 \
    python3 scripts/run_strict_r3_live_position_monitor.py \
      --execution-bundle "$EXECUTION_BUNDLE" \
      --state "$LIVE_STATE" \
      --out-root "$OUT_ROOT" \
      --interval-seconds "$INTERVAL_SECONDS" \
      "${submit_args[@]}" >> "$LOG_PATH" 2>&1
