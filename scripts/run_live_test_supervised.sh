#!/usr/bin/env zsh
set -u

cd /Users/remyroche/Documents/Ares || exit 1

RUN_ID="${RUN_ID:-20260525_010004_nopenalty}"
DATA_ROOT="${DATA_ROOT:-data_perp}"
LOG_DIR="${LOG_DIR:-logs}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-30}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"
INFERENCE_INTERVAL="${INFERENCE_INTERVAL:-60}"
CHALLENGER_INTERVAL="${CHALLENGER_INTERVAL:-30}"
EPM_EXCHANGE="${EPM_EXCHANGE:-kraken}"
LIVE_DATA_ROOT="${LIVE_DATA_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.11/bin/python3}"

mkdir -p "$LOG_DIR"

stamp="$(date -u +%Y%m%d_%H%M%S)"
supervisor_log="$LOG_DIR/live_test_supervisor_${stamp}.log"
pid_file="$LOG_DIR/live_test_supervisor.pid"

echo "$$" > "$pid_file"

restart_count=0
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] supervisor_start pid=$$ run_id=$RUN_ID data_root=$DATA_ROOT exchange=$EPM_EXCHANGE inference_interval=$INFERENCE_INTERVAL challenger_interval=$CHALLENGER_INTERVAL" | tee -a "$supervisor_log"

while true; do
  child_stamp="$(date -u +%Y%m%d_%H%M%S)"
  child_log="$LOG_DIR/live_test_kraken_perps_${child_stamp}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] child_start restart_count=$restart_count log=$child_log" | tee -a "$supervisor_log"

  live_data_args=()
  if [[ -n "$LIVE_DATA_ROOT" ]]; then
    live_data_args=(--live-data-root "$LIVE_DATA_ROOT")
  fi

  env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=. \
    MPLCONFIGDIR=/private/tmp/ares_mplconfig \
    EPM_EXCHANGE="$EPM_EXCHANGE" \
    EPM_DATA_ROOT="$DATA_ROOT" \
    EPM_ARTIFACT_SOURCE_RUN_ID="${EPM_ARTIFACT_SOURCE_RUN_ID:-20260523_015947}" \
    EPM_MODEL_BACKEND="${EPM_MODEL_BACKEND:-lgbm_pipeline}" \
    EPM_DISABLE_REGIME_ADAPTORS="${EPM_DISABLE_REGIME_ADAPTORS:-1}" \
    EPM_SIMPLE_POLICY_REGIME_ADAPTOR="${EPM_SIMPLE_POLICY_REGIME_ADAPTOR:-0}" \
    EPM_LIVE_FEATURE_LAYER_DEBUG="${EPM_LIVE_FEATURE_LAYER_DEBUG:-1}" \
    EPM_LIVE_MODEL_FEATURE_AUTO_SYNC_SELECTED_CACHE="${EPM_LIVE_MODEL_FEATURE_AUTO_SYNC_SELECTED_CACHE:-1}" \
    EPM_RUN_SCOPED_PREDICTION_LEDGER="${EPM_RUN_SCOPED_PREDICTION_LEDGER:-1}" \
    "$PYTHON_BIN" -u -m extreme_price_movements.inference.run_inference \
      --live-test \
      --perps \
      --data-root "$DATA_ROOT" \
      --run-id "$RUN_ID" \
      --run-scoped-prediction-ledger \
      --inference-interval "$INFERENCE_INTERVAL" \
      --challenger-interval "$CHALLENGER_INTERVAL" \
      "${live_data_args[@]}" \
      >> "$child_log" 2>&1

  exit_code=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] child_exit restart_count=$restart_count exit_code=$exit_code log=$child_log" | tee -a "$supervisor_log"

  restart_count=$((restart_count + 1))
  if [[ "$MAX_RESTARTS" != "0" && "$restart_count" -ge "$MAX_RESTARTS" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] supervisor_stop reason=max_restarts restart_count=$restart_count" | tee -a "$supervisor_log"
    exit "$exit_code"
  fi

  sleep "$RESTART_DELAY_SECONDS"
done
