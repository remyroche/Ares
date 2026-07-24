#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

mkdir -p logs

RUN_ID="${RUN_ID:-s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2}"
MODEL_ARTIFACT_RUN_ID="${EPM_MODEL_ARTIFACT_RUN_ID:-$RUN_ID}"
POLICY_ARTIFACT_RUN_ID="${EPM_POLICY_ARTIFACT_RUN_ID:-$RUN_ID}"
SESSION_LOG="logs/kraken_perps_live_supervisor_${RUN_ID}.log"
PID_FILE="logs/kraken_perps_live_supervisor_${RUN_ID}.pid"
INFERENCE_PID_FILE="logs/kraken_perps_live_child_${RUN_ID}.pid"
LEDGER_PATH="data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/${RUN_ID}/prediction_ledger.parquet"
POLICY_CONFIG_PATH="data_perp/artifacts/${POLICY_ARTIFACT_RUN_ID}/policy_params/optimized_portfolio_policy_config.json"
RECONCILIATION_DIR="data_perp/exchanges/krakenfutures/live_state/reconciliation/${RUN_ID}/execution_realism"
TRADE_LOG_PATH="inference_trades.csv"
SESSION_START_FILE="data_perp/exchanges/krakenfutures/live_state/monitoring/production_parity/${RUN_ID}/live_session_start_utc.txt"
INTERNAL_RECONCILIATION_ENABLED="${EPM_LIVE_INTERNAL_RECONCILIATION_ENABLED:-1}"
RECONCILIATION_INTERVAL_SECONDS="${EPM_LIVE_RECONCILIATION_INTERVAL_SECONDS:-10800}"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  old_command="$(ps -p "$old_pid" -o command= 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null && \
      [[ "$old_command" == *"run_kraken_perps_live_supervisor.sh"* ]]; then
    printf '[%s] supervisor already running pid=%s run_id=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$old_pid" "$RUN_ID" | tee -a "$SESSION_LOG"
    exit 0
  fi
fi

rm -f "$INFERENCE_PID_FILE"
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE" "$INFERENCE_PID_FILE"' EXIT
printf '[%s] supervisor_start pid=%s run_id=%s model_artifact_run_id=%s policy_artifact_run_id=%s ledger=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$" "$RUN_ID" "$MODEL_ARTIFACT_RUN_ID" "$POLICY_ARTIFACT_RUN_ID" "$LEDGER_PATH" | tee -a "$SESSION_LOG"

reconciliation_loop() {
  while true; do
    if [[ -f "$LEDGER_PATH" && -f "$POLICY_CONFIG_PATH" ]]; then
      printf '[%s] running execution realism reconciliation\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      env \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH=. \
        MPLCONFIGDIR=/private/tmp/mplconfig \
        python3 -u -m extreme_price_movements.inference.execution_reconciliation \
          --prediction-ledger "$LEDGER_PATH" \
          --portfolio-policy-config "$POLICY_CONFIG_PATH" \
          --output-dir "$RECONCILIATION_DIR" \
          --trade-log-path "$TRADE_LOG_PATH" \
          --data-root data_perp \
          --run-id "$RUN_ID" \
          --prediction-parity-max-rows 500 \
          --shadow-tolerance-bps 50
      printf '[%s] execution realism reconciliation complete\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    else
      printf '[%s] execution realism reconciliation waiting for ledger/policy artifacts\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    fi
    sleep "$RECONCILIATION_INTERVAL_SECONDS"
  done
}

if [[ "$INTERNAL_RECONCILIATION_ENABLED" == "1" ]]; then
  reconciliation_loop >> "$SESSION_LOG" 2>&1 &
else
  printf '[%s] internal reconciliation disabled; external parity monitor is authoritative\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$SESSION_LOG"
fi

while true; do
  mkdir -p "$(dirname "$SESSION_START_FILE")"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$SESSION_START_FILE"
  {
    printf '\n[%s] starting live inference run_id=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$RUN_ID"
    env \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH=. \
      MPLCONFIGDIR=/private/tmp/mplconfig \
      EPM_EXCHANGE=krakenfutures \
      EXCHANGE_NAME=krakenfutures \
      PRIMARY_EXCHANGE=krakenfutures \
      EPM_LABEL_WEIGHT_DISABLE=1 \
      EPM_LABEL_WEIGHT_USE_BEST_DEFAULT=0 \
      EPM_FEATURE_SELECTED_LOAD_PARALLEL=0 \
      EPM_LIVE_MODEL_FEATURE_FULL_UNION_BACKGROUND_SYNC=0 \
      EPM_LIVE_LATEST_FEATURE_MATRIX_SIDECAR=1 \
      EPM_LIVE_LATEST_FEATURE_MATRIX_SIDECAR_FOR_RANGE=1 \
      EPM_HOURLY_OHLCV_DELAY_SECONDS=30 \
      EPM_LIVE_MODEL_FEATURE_AUTO_SYNC_SELECTED_CACHE=1 \
      EPM_LIVE_MODEL_FEATURE_AUTO_SYNC_BLOCKING=1 \
      EPM_FEATURE_MISSING_COLUMNS_RECENT_TAIL=1 \
      EPM_SPREAD_BLACKLIST_THRESHOLD_BPS=70 \
      EPM_STRICT_FEATURE_PARITY_NEUTRAL_FILL_NONFINITE=0 \
      python3 -u -m extreme_price_movements.inference.run_inference \
        --live \
        --perps \
        --data-root data_perp \
        --run-id "$RUN_ID" \
        --model-artifact-run-id "$MODEL_ARTIFACT_RUN_ID" \
        --policy-artifact-run-id "$POLICY_ARTIFACT_RUN_ID" \
        --run-scoped-prediction-ledger \
        --challenger-interval "${EPM_CHALLENGER_INTERVAL_SECONDS:-60}" &
    inference_pid=$!
    printf '%s\n' "$inference_pid" > "$INFERENCE_PID_FILE"
    wait "$inference_pid"
    rc=$?
    rm -f "$INFERENCE_PID_FILE"
    printf '[%s] live inference exited rc=%s; restarting in 60s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$rc"
  } >> "$SESSION_LOG" 2>&1
  sleep 60
done
