#!/usr/bin/env bash
set -u

# Comprehensive but low-frequency audit for the production inference process.
# The inference loop remains responsible for fail-closed feature/model checks;
# this process independently replays persisted decisions every three hours.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

RUN_ID="${RUN_ID:-s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2}"
MODEL_ARTIFACT_RUN_ID="${EPM_MODEL_ARTIFACT_RUN_ID:-$RUN_ID}"
POLICY_ARTIFACT_RUN_ID="${EPM_POLICY_ARTIFACT_RUN_ID:-$RUN_ID}"
INTERVAL_SECONDS="${EPM_PARITY_MONITOR_INTERVAL_SECONDS:-10800}"
MAX_PARITY_ROWS="${EPM_PARITY_MONITOR_MAX_ROWS:-500}"
LIVE_FEATURE_SOURCE_RUN_ID="${EPM_LIVE_FEATURE_SOURCE_RUN_ID:-20260711_070000}"

LIVE_ROOT="data_perp/exchanges/krakenfutures"
FEATURE_ROOT="data_perp/features"
LEDGER_PATH="$LIVE_ROOT/live_state/prediction_ledgers/$RUN_ID/prediction_ledger.parquet"
POLICY_CONFIG="data_perp/artifacts/$POLICY_ARTIFACT_RUN_ID/policy_params/optimized_portfolio_policy_config.json"
OUT_ROOT="$LIVE_ROOT/live_state/monitoring/production_parity/$RUN_ID"
SESSION_LOG="logs/kraken_perps_parity_monitor_${RUN_ID}.log"
PID_FILE="logs/kraken_perps_parity_monitor_${RUN_ID}.pid"
INFERENCE_PID_FILE="logs/kraken_perps_live_child_${RUN_ID}.pid"
INFERENCE_PID_OVERRIDE="${EPM_INFERENCE_PID:-}"
INFERENCE_LOG="${EPM_INFERENCE_LOG:-logs/kraken_perps_live_supervisor_${RUN_ID}.log}"
TRADE_LOG="inference_trades.csv"
SESSION_START_FILE="$OUT_ROOT/live_session_start_utc.txt"

mkdir -p logs "$OUT_ROOT" /private/tmp/mplconfig

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    printf '[%s] parity monitor already running pid=%s run_id=%s\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$old_pid" "$RUN_ID" | tee -a "$SESSION_LOG"
    exit 0
  fi
fi

echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

is_expected_inference_pid() {
  candidate_pid="$1"
  python3 -c 'import psutil, sys
pid = int(sys.argv[1])
run_id = sys.argv[2]
try:
    process = psutil.Process(pid)
    command = process.cmdline()
except (psutil.Error, ValueError):
    raise SystemExit(1)
joined = " ".join(command)
valid = (
    process.is_running()
    and "extreme_price_movements.inference.run_inference" in joined
    and "--live" in command
    and "--run-id" in command
    and run_id in command
)
raise SystemExit(0 if valid else 1)' "$candidate_pid" "$RUN_ID" >/dev/null 2>&1
}

find_inference_pid() {
  if [[ -n "$INFERENCE_PID_OVERRIDE" ]] && \
      is_expected_inference_pid "$INFERENCE_PID_OVERRIDE"; then
    printf '%s\n' "$INFERENCE_PID_OVERRIDE"
    return 0
  fi
  if [[ -f "$INFERENCE_PID_FILE" ]]; then
    child_pid="$(cat "$INFERENCE_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$child_pid" ]] && is_expected_inference_pid "$child_pid"; then
      printf '%s\n' "$child_pid"
      return 0
    fi
    rm -f "$INFERENCE_PID_FILE"
  fi
  while IFS= read -r candidate_pid; do
    if [[ -n "$candidate_pid" ]] && is_expected_inference_pid "$candidate_pid"; then
      printf '%s\n' "$candidate_pid"
      return 0
    fi
  done < <(pgrep -f "extreme_price_movements.inference.run_inference.*--run-id $RUN_ID" 2>/dev/null || true)
  return 1
}

find_latest_sidecar_dir() {
  candidate="$FEATURE_ROOT/$LIVE_FEATURE_SOURCE_RUN_ID/_live_latest_matrix"
  if [[ ! -d "$candidate" ]]; then
    return 1
  fi
  latest_file="$(find "$candidate" -maxdepth 1 -name 'matrix_*.parquet' -print 2>/dev/null | sort | tail -n 1)"
  if [[ -n "$latest_file" ]]; then
    printf '%s\n' "$candidate"
  fi
}

run_audit() {
  stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
  audit_dir="$OUT_ROOT/$stamp"
  mkdir -p "$audit_dir"
  inference_pid="$(find_inference_pid || true)"
  # The supervisor writes its child PID after model-process startup.  Avoid
  # recording a false failure when the parity monitor and supervisor are
  # launched together.
  startup_wait_seconds="${EPM_PARITY_MONITOR_STARTUP_WAIT_SECONDS:-120}"
  startup_waited=0
  while [[ -z "$inference_pid" && "$startup_waited" -lt "$startup_wait_seconds" ]]; do
    sleep 1
    startup_waited=$((startup_waited + 1))
    inference_pid="$(find_inference_pid || true)"
  done
  session_start="$(cat "$SESSION_START_FILE" 2>/dev/null || true)"
  if [[ -n "$inference_pid" ]]; then
    process_start="$(python3 -c 'import datetime, psutil, sys; print(datetime.datetime.fromtimestamp(psutil.Process(int(sys.argv[1])).create_time(), datetime.timezone.utc).isoformat().replace("+00:00", "Z"))' "$inference_pid" 2>/dev/null || true)"
    if [[ -n "$process_start" ]] && { [[ -z "$session_start" ]] || [[ "$process_start" > "$session_start" ]]; }; then
      session_start="$process_start"
    fi
  fi
  since_args=()
  if [[ -n "$session_start" ]]; then
    since_args=(--since "$session_start")
  fi
  overall_rc=0
  feature_parity_status="pending"
  feature_parity_reason="live_latest_matrix_sidecar_unavailable"
  execution_parity_status="pending"
  execution_parity_reason="execution_reconciliation_not_run"
  open_position_parity_status="pending"
  open_position_parity_reason="open_position_policy_audit_not_run"
  decision_chain_status="pending"
  decision_chain_reason="decision_reconciliation_not_run"

  printf '[%s] audit_start run_id=%s model=%s policy=%s inference_pid=%s\n' \
    "$stamp" "$RUN_ID" "$MODEL_ARTIFACT_RUN_ID" "$POLICY_ARTIFACT_RUN_ID" \
    "${inference_pid:-missing}"

  if [[ -z "$inference_pid" ]]; then
    printf '{"status":"fail","reason":"live_inference_process_missing","run_id":"%s"}\n' \
      "$RUN_ID" > "$audit_dir/audit_status.json"
    return 2
  fi

  env PYTHONUNBUFFERED=1 PYTHONPATH=. MPLCONFIGDIR=/private/tmp/mplconfig \
    python3 -u scripts/monitor_live_runtime_health.py \
      --pid "$inference_pid" \
      --log "$INFERENCE_LOG" \
      --trade-log "$TRADE_LOG" \
      --prediction-ledger "$LEDGER_PATH" \
      --out "$audit_dir/runtime_health.jsonl" \
      --duration-seconds 1 \
      --interval-seconds 1 \
      --max-prediction-rows "$MAX_PARITY_ROWS" \
      "${since_args[@]}" \
      > "$audit_dir/runtime_health.log" 2>&1 || overall_rc=1

  if [[ -f "$LEDGER_PATH" && -f "$POLICY_CONFIG" ]]; then
    reconciliation_rc=0
    env PYTHONUNBUFFERED=1 PYTHONPATH=. MPLCONFIGDIR=/private/tmp/mplconfig \
      python3 -u -m extreme_price_movements.inference.execution_reconciliation \
        --prediction-ledger "$LEDGER_PATH" \
        --portfolio-policy-config "$POLICY_CONFIG" \
        --output-dir "$audit_dir/execution_reconciliation" \
        --trade-log-path "$TRADE_LOG" \
        --data-root data_perp \
        --run-id "$RUN_ID" \
        --prediction-parity-max-rows "$MAX_PARITY_ROWS" \
        --shadow-tolerance-bps 50 \
        "${since_args[@]}" \
        > "$audit_dir/execution_reconciliation.log" 2>&1 || reconciliation_rc=$?

    prediction_status_path="$audit_dir/execution_reconciliation/prediction_rank_parity_reconciliation.json"
    decision_status_path="$audit_dir/execution_reconciliation/live_decision_replay_reconciliation.json"
    if [[ -f "$prediction_status_path" && -f "$decision_status_path" ]]; then
      decision_status_probe="$(
        python3 -c 'import json, sys
prediction = json.load(open(sys.argv[1], encoding="utf-8"))
decision = json.load(open(sys.argv[2], encoding="utf-8"))
active = bool(prediction.get("active_decision_chain_pass", False))
mode = str(decision.get("replay_mode") or "")
mismatches = int(decision.get("decision_mismatches", -1) or 0)
passed = active and mode == "persisted_auction_state" and mismatches == 0
reason = "active_chain_and_persisted_auction_exact" if passed else "active_or_auction_decision_mismatch"
print("{}|{}".format("pass" if passed else "fail", reason))' \
          "$prediction_status_path" "$decision_status_path" 2>/dev/null || true
      )"
      decision_chain_status="${decision_status_probe%%|*}"
      decision_chain_reason="${decision_status_probe#*|}"
      if [[ "$decision_chain_status" != "pass" ]]; then
        decision_chain_status="fail"
        overall_rc=1
      fi
    else
      decision_chain_status="fail"
      decision_chain_reason="active_decision_reconciliation_artifact_missing"
      overall_rc=1
    fi

    exit_replay_dir="$audit_dir/independent_exit_replay"
    exit_replay_rc=0
    env PYTHONUNBUFFERED=1 PYTHONPATH=. MPLCONFIGDIR=/private/tmp/mplconfig \
      python3 -u -m extreme_price_movements.scripts.live_closed_trade_exit_replay \
        --data-root data_perp \
        --run-id "$RUN_ID" \
        --closed-trades "$TRADE_LOG" \
        --out-dir "$exit_replay_dir" \
        --limit 100 \
        --exit-tolerance-bps 10 \
        --exit-time-tolerance-seconds 90 \
        --ignore-logged-exit-events \
        "${since_args[@]}" \
        > "$audit_dir/independent_exit_replay.log" 2>&1 || exit_replay_rc=$?

    execution_status_path="$exit_replay_dir/live_closed_trade_exit_replay_summary.json"
    if [[ -f "$execution_status_path" ]]; then
      execution_status_probe="$(
        python3 -c 'import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
rows = int(data.get("rows", 0) or 0)
status = str(data.get("exit_parity_status") or "pending")
reason = "independent_closed_exit_replay:{}rows".format(rows)
print("{}|{}".format(status, reason))' \
          "$execution_status_path" 2>/dev/null || true
      )"
      execution_parity_status="${execution_status_probe%%|*}"
      execution_parity_reason="${execution_status_probe#*|}"
      if [[ "$execution_parity_status" != "pass" && "$execution_parity_status" != "fail" && "$execution_parity_status" != "pending" ]]; then
        execution_parity_status="fail"
        execution_parity_reason="invalid_execution_parity_audit_status"
        overall_rc=1
      elif [[ "$execution_parity_status" == "fail" ]]; then
        overall_rc=1
      fi
    else
      execution_parity_status="fail"
      execution_parity_reason="independent_exit_replay_summary_missing"
      overall_rc=1
    fi

    open_position_dir="$audit_dir/open_position_policy_state"
    open_position_rc=0
    env PYTHONUNBUFFERED=1 PYTHONPATH=. MPLCONFIGDIR=/private/tmp/mplconfig \
      python3 -u scripts/audit_live_open_position_policy_state.py \
        --inference-log "$INFERENCE_LOG" \
        --trade-log "$TRADE_LOG" \
        --data-root data_perp \
        --run-id "$POLICY_ARTIFACT_RUN_ID" \
        --output-dir "$open_position_dir" \
        > "$audit_dir/open_position_policy_state.log" 2>&1 || open_position_rc=$?
    open_position_status_path="$open_position_dir/summary.json"
    if [[ -f "$open_position_status_path" ]]; then
      open_position_status_probe="$(
        python3 -c 'import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print("{}|{}".format(data.get("status", "fail"), data.get("reason", "unknown")))' \
          "$open_position_status_path" 2>/dev/null || true
      )"
      open_position_parity_status="${open_position_status_probe%%|*}"
      open_position_parity_reason="${open_position_status_probe#*|}"
      if [[ "$open_position_parity_status" == "fail" ]]; then
        overall_rc=1
      elif [[ "$open_position_parity_status" != "pass" && "$open_position_parity_status" != "pending" ]]; then
        open_position_parity_status="fail"
        open_position_parity_reason="invalid_open_position_parity_status"
        overall_rc=1
      fi
    else
      open_position_parity_status="fail"
      open_position_parity_reason="open_position_policy_summary_missing"
      overall_rc=1
    fi

    latest_batch_probe="$(
      python3 -c 'import os, pandas as pd, sys
p, start = sys.argv[1], pd.Timestamp(sys.argv[2])
d = pd.read_parquet(p, columns=["decision_ts"])
t = pd.to_datetime(d["decision_ts"], utc=True, errors="coerce")
x = t[t >= start]
latest = x.max() if not x.empty else pd.NaT
n = int((x == latest).sum()) if pd.notna(latest) else 0
print("{}|{}".format(latest.isoformat() if pd.notna(latest) else "", n), flush=True)
os._exit(0)' \
        "$LEDGER_PATH" "${session_start:-1970-01-01T00:00:00Z}" 2>/dev/null || true
    )"
    latest_decision_ts="${latest_batch_probe%%|*}"
    latest_decision_rows="${latest_batch_probe##*|}"
    if [[ ! "$latest_decision_rows" =~ ^[1-9][0-9]*$ ]]; then
      latest_decision_rows="$MAX_PARITY_ROWS"
    fi

    sidecar_dir="$(find_latest_sidecar_dir || true)"
    if [[ -n "$sidecar_dir" ]]; then
      env PYTHONUNBUFFERED=1 PYTHONPATH=. MPLCONFIGDIR=/private/tmp/mplconfig \
        python3 -u scripts/verify_live_ledger_feature_json_parity.py \
          --ledger "$LEDGER_PATH" \
          --sidecar-dir "$sidecar_dir" \
          --layers base,meta \
          --tolerance 1e-6 \
          --max-rows "$latest_decision_rows" \
          --min-compared-rows 1 \
          --min-common-cells 1 \
          --require-complete-sidecar-coverage \
          --output-dir "$audit_dir/feature_parity" \
          > "$audit_dir/feature_parity.log" 2>&1 || overall_rc=1
      feature_status_path="$audit_dir/feature_parity/summary.json"
      if [[ -f "$feature_status_path" ]]; then
        feature_status_probe="$(
          python3 -c 'import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
passed = bool(data.get("parity_gate_pass", False))
reason = "" if passed else ";".join(map(str, data.get("coverage_gate_failures") or [])) or "feature_parity_mismatch"
print("{}|{}".format("pass" if passed else "fail", reason))' \
            "$feature_status_path" 2>/dev/null || true
        )"
        feature_parity_status="${feature_status_probe%%|*}"
        feature_parity_reason="${feature_status_probe#*|}"
        if [[ "$feature_parity_status" != "pass" && "$feature_parity_status" != "fail" ]]; then
          feature_parity_status="fail"
          feature_parity_reason="invalid_feature_parity_audit_status"
          overall_rc=1
        elif [[ "$feature_parity_status" == "fail" ]]; then
          overall_rc=1
        fi
      else
        feature_parity_status="fail"
        feature_parity_reason="feature_parity_summary_missing"
        overall_rc=1
      fi
    else
      printf 'No live latest-matrix sidecar was available.\n' > "$audit_dir/feature_parity.log"
    fi

    if [[ -n "$latest_decision_ts" ]]; then
      env PYTHONUNBUFFERED=1 PYTHONPATH=. MPLCONFIGDIR=/private/tmp/mplconfig \
        python3 -u scripts/replay_live_signal_predictions.py \
          --data-root data_perp \
          --artifact-data-root data_perp \
          --market-mode perps \
          --exchange-id krakenfutures \
          --live-quote-currency USD \
          --source-parity-run-id "$RUN_ID" \
          --live-feature-source-run-id "$LIVE_FEATURE_SOURCE_RUN_ID" \
          --run-id "$RUN_ID" \
          --ledger "$LEDGER_PATH" \
          --decision-start "$latest_decision_ts" \
          --max-rows "$MAX_PARITY_ROWS" \
          --lookback-hours 1440 \
          --batch-by-signal-bar-cache \
          --parity-source replay \
          --fail-on-mismatch \
          --require-live-values \
          --require-policy-rank-reference \
          --tolerance 1e-5 \
          --prediction-tolerance 1e-7 \
          --output-dir "$audit_dir/exact_chain_parity" \
          > "$audit_dir/exact_chain_parity.log" 2>&1 || overall_rc=1
    else
      printf 'No prediction-ledger rows have been written since live startup.\n' \
        > "$audit_dir/exact_chain_parity.log"
    fi
  else
    printf 'Ledger or deployed policy config is not available yet.\n' \
      > "$audit_dir/execution_reconciliation.log"
    overall_rc=1
  fi

  audit_status="pass"
  audit_reason="all_required_parity_evidence_passed"
  if [[ "$overall_rc" -ne 0 ]]; then
    audit_status="fail"
    audit_reason="one_or_more_parity_checks_failed"
  elif [[ "$decision_chain_status" == "pending" ]]; then
    audit_status="pending"
    audit_reason="$decision_chain_reason"
  elif [[ "$feature_parity_status" == "pending" ]]; then
    audit_status="pending"
    audit_reason="$feature_parity_reason"
  elif [[ "$execution_parity_status" == "pending" ]]; then
    audit_status="pending"
    audit_reason="$execution_parity_reason"
  elif [[ "$open_position_parity_status" == "pending" ]]; then
    audit_status="pending"
    audit_reason="$open_position_parity_reason"
  fi

  printf '{"status":"%s","reason":"%s","run_id":"%s","model_artifact_run_id":"%s","policy_artifact_run_id":"%s","inference_pid":%s,"feature_parity_status":"%s","decision_chain_status":"%s","execution_parity_status":"%s","open_position_parity_status":"%s","audit_ts":"%s"}\n' \
    "$audit_status" "$audit_reason" \
    "$RUN_ID" "$MODEL_ARTIFACT_RUN_ID" "$POLICY_ARTIFACT_RUN_ID" \
    "$inference_pid" "$feature_parity_status" "$decision_chain_status" "$execution_parity_status" "$open_position_parity_status" "$stamp" > "$audit_dir/audit_status.json"
  ln -sfn "$stamp" "$OUT_ROOT/latest"
  if [[ -n "$session_start" ]]; then
    printf '%s\n' "$session_start" > "$audit_dir/live_session_start_utc.txt"
  fi
  printf '[%s] audit_complete rc=%s output=%s\n' "$stamp" "$overall_rc" "$audit_dir"
  return "$overall_rc"
}

printf '[%s] parity_monitor_start pid=%s interval_seconds=%s run_id=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$" "$INTERVAL_SECONDS" "$RUN_ID" \
  | tee -a "$SESSION_LOG"

while true; do
  cycle_started_epoch="$(date -u '+%s')"
  run_audit >> "$SESSION_LOG" 2>&1 || true
  cycle_finished_epoch="$(date -u '+%s')"
  cycle_elapsed=$((cycle_finished_epoch - cycle_started_epoch))
  sleep_seconds=$((INTERVAL_SECONDS - cycle_elapsed))
  if [[ "$sleep_seconds" -lt 1 ]]; then
    sleep_seconds=1
  fi
  sleep "$sleep_seconds"
done
