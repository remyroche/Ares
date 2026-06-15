#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

mkdir -p logs

RUN_ID="${RUN_ID:-20260612_203000_top2_fullscope_labelhpo_drift_leaflite_native}"
SESSION_LOG="logs/kraken_perps_live_supervisor_${RUN_ID}.log"
PID_FILE="logs/kraken_perps_live_supervisor_${RUN_ID}.pid"
LEDGER_PATH="data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/${RUN_ID}/prediction_ledger.parquet"
POLICY_CONFIG_PATH="data_perp/artifacts/${RUN_ID}/policy_params/best_policy_params.json"
RECONCILIATION_DIR="data_perp/exchanges/krakenfutures/live_state/reconciliation/${RUN_ID}/execution_realism"
TRADE_LOG_PATH="inference_trades.csv"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    printf '[%s] supervisor already running pid=%s run_id=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$old_pid" "$RUN_ID" | tee -a "$SESSION_LOG"
    exit 0
  fi
fi

echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT
printf '[%s] supervisor_start pid=%s run_id=%s ledger=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$" "$RUN_ID" "$LEDGER_PATH" | tee -a "$SESSION_LOG"

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
    sleep 900
  done
}

reconciliation_loop >> "$SESSION_LOG" 2>&1 &

while true; do
  {
    printf '\n[%s] starting live inference run_id=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$RUN_ID"
    env \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH=. \
      MPLCONFIGDIR=/private/tmp/mplconfig \
      EPM_EXCHANGE=kraken \
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
      python3 -u -m extreme_price_movements.inference.run_inference \
        --live \
        --perps \
        --data-root data_perp \
        --run-id "$RUN_ID" \
        --run-scoped-prediction-ledger \
        --challenger-interval 120
    rc=$?
    printf '[%s] live inference exited rc=%s; restarting in 60s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$rc"
  } >> "$SESSION_LOG" 2>&1
  sleep 60
done
