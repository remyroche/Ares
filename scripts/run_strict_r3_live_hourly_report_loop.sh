#!/usr/bin/env bash
set -u

# Continuous post-candle observer. From xx:03 it watches the current hour
# until a producer attempt has either executed the complete pipeline or
# rejected every opportunity. Incident attempts receive their own immutable
# report and remain pending until a tested successor provides the terminal
# result. It has no data-fetching or exchange-writing authority.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
# "any" is the production default: report the receipt's real runtime tag.
# This permits a runtime-only reseal to remain independently observable rather
# than silently orphaning the new producer namespace.
RUNTIME_TAG="${STRICT_R3_REPORT_RUNTIME_TAG:-any}"
LOG_PATH="${STRICT_R3_REPORT_LOG:-logs/strict_r3_live_hourly_report.log}"
PID_PATH="${STRICT_R3_REPORT_PID:-/private/tmp/strict_r3_live_hourly_report.pid}"
POLL_SECONDS="${STRICT_R3_REPORT_POLL_SECONDS:-15}"
LIVE_STATE="${STRICT_R3_REPORT_LIVE_STATE:-data_perp/live/strict_r3_kraken_live_state_v32_v52_full_runtime_guard.json}"
mkdir -p "$(dirname "$LOG_PATH")"
echo "$$" > "$PID_PATH"
trap 'rm -f "$PID_PATH"' EXIT

pending=""
last_started=""
last_observed_receipt=""
while true; do
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  minute="$(date -u +%M)"
  decision="$(date -u +%Y-%m-%dT%H:00:00Z)"
  if [[ "$minute" -ge 3 && "$decision" != "$last_started" ]]; then
    pending="$decision"
    last_started="$decision"
    last_observed_receipt=""
    printf '%s watch_started decision=%s\n' "$now" "$pending" >> "$LOG_PATH"
  fi
  if [[ -n "$pending" ]]; then
    stamp="$(date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$pending" '+%Y%m%dT%H%M%SZ' 2>/dev/null || true)"
    receipt=""
    if [[ -n "$stamp" ]]; then
      if [[ "$RUNTIME_TAG" == "any" || "$RUNTIME_TAG" == "*" ]]; then
        receipt="$(find data_perp/artifacts -maxdepth 2 -path "*/strict_r3_live_hourly_producer_*_${stamp}_v*/run_manifest.json" -type f -print 2>/dev/null | sort | tail -n 1)"
      else
        receipt="$(find data_perp/artifacts -maxdepth 2 -path "*/strict_r3_live_hourly_producer_${RUNTIME_TAG}_${stamp}_v*/run_manifest.json" -type f -print 2>/dev/null | sort | tail -n 1)"
      fi
    fi
    # Observe a producer receipt once. An incident remains pending, but a
    # later tested vN receipt is then reported independently without filling
    # the operational log every poll cycle.
    if [[ -n "$receipt" && "$receipt" != "$last_observed_receipt" ]]; then
      last_observed_receipt="$receipt"
      if python3 scripts/report_strict_r3_live_candle.py --runtime-tag "$RUNTIME_TAG" --decision-ts "$pending" --live-state "$LIVE_STATE" >> "$LOG_PATH" 2>&1; then
        printf '%s terminal_report decision=%s receipt=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pending" "$receipt" >> "$LOG_PATH"
        pending=""
      else
        printf '%s incident_report decision=%s receipt=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pending" "$receipt" >> "$LOG_PATH"
      fi
    fi
  fi
  sleep "$POLL_SECONDS"
done
