#!/usr/bin/env bash
set -u

# Operations controller.  It starts at xx:03, consumes every immutable report
# attempt for the current decision hour, classifies the root cause, and invokes
# only the separately configured, hash-bound safe recovery actions.  Unknown
# failures, state/feature/model/lineage failures, and all post-execution
# incidents remain fail-closed for a reviewed patch and reseal.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
# "any" consumes every immutable producer report; the receipt itself records
# the actual runtime namespace so a runtime-only reseal cannot be invisible.
RUNTIME_TAG="${STRICT_R3_SUPERVISOR_RUNTIME_TAG:-any}"
LOG_PATH="${STRICT_R3_SUPERVISOR_LOG:-logs/strict_r3_live_operations_supervisor.log}"
PID_PATH="${STRICT_R3_SUPERVISOR_PID:-/private/tmp/strict_r3_live_operations_supervisor.pid}"
POLL_SECONDS="${STRICT_R3_SUPERVISOR_POLL_SECONDS:-15}"
CONTROLLER_CONFIG="${STRICT_R3_SUPERVISOR_CONTROLLER_CONFIG:-config/strict_r3_live_operations_controller_v1.json}"
mkdir -p "$(dirname "$LOG_PATH")"
echo "$$" > "$PID_PATH"
trap 'rm -f "$PID_PATH"' EXIT

last_decision=""
seen_reports=""
while true; do
  minute="$(date -u +%M)"
  decision="$(date -u +%Y-%m-%dT%H:00:00Z)"
  if [[ "$minute" -ge 3 ]]; then
    if [[ "$decision" != "$last_decision" ]]; then
      last_decision="$decision"
      seen_reports=""
      printf '%s supervisor_watch_started decision=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$decision" >> "$LOG_PATH"
      if python3 scripts/run_strict_r3_live_operations_controller.py \
          --runtime-tag "$RUNTIME_TAG" --config "$CONTROLLER_CONFIG" \
          --decision-ts "$decision" >> "$LOG_PATH" 2>&1; then
        printf '%s controller_watchdog_checked decision=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$decision" >> "$LOG_PATH"
      else
        printf '%s controller_watchdog_failure decision=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$decision" >> "$LOG_PATH"
      fi
    fi
    stamp="$(date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$decision" '+%Y%m%dT%H%M%SZ' 2>/dev/null || true)"
    if [[ -n "$stamp" ]]; then
      if [[ "$RUNTIME_TAG" == "any" || "$RUNTIME_TAG" == "*" ]]; then
        report_pattern="strict_r3_live_candle_*_${stamp}_*.json"
      else
        report_pattern="strict_r3_live_candle_${RUNTIME_TAG}_${stamp}_*.json"
      fi
      while IFS= read -r report; do
        [[ -z "$report" ]] && continue
        case "|$seen_reports|" in
          *"|$report|"*) continue ;;
        esac
        seen_reports="${seen_reports}|${report}"
        # `any` is an observer selector, never a valid producer-runtime
        # namespace.  Ignore a legacy/manual report made under that temporary
        # name so it cannot conflict with the immutable actual-runtime report.
        case "$(basename "$report")" in
          strict_r3_live_candle_any_*)
            printf '%s supervisor_ignored_noncanonical_report report=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$report" >> "$LOG_PATH"
            continue ;;
        esac
        if python3 scripts/consume_strict_r3_live_report.py --runtime-tag "$RUNTIME_TAG" --report "$report" >> "$LOG_PATH" 2>&1; then
          printf '%s supervisor_consumed report=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$report" >> "$LOG_PATH"
        else
          printf '%s supervisor_consume_failure report=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$report" >> "$LOG_PATH"
        fi
        if python3 scripts/run_strict_r3_live_operations_controller.py \
            --runtime-tag "$RUNTIME_TAG" --config "$CONTROLLER_CONFIG" \
            --decision-ts "$decision" --report "$report" >> "$LOG_PATH" 2>&1; then
          printf '%s controller_classified report=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$report" >> "$LOG_PATH"
        else
          printf '%s controller_classification_failure report=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$report" >> "$LOG_PATH"
        fi
      done < <(find data_perp/reports -maxdepth 1 -type f -name "$report_pattern" -print 2>/dev/null | sort)
    fi
  fi
  sleep "$POLL_SECONDS"
done
