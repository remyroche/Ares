#!/usr/bin/env zsh
set -u

cd /Users/remyroche/Documents/Ares || exit 1

RUN_ID="${RUN_ID:-20260523_015947}"
DATA_ROOT="${DATA_ROOT:-data_perp}"
LOG_DIR="${LOG_DIR:-logs}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-30}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"

mkdir -p "$LOG_DIR"

stamp="$(date -u +%Y%m%d_%H%M%S)"
supervisor_log="$LOG_DIR/live_test_supervisor_${stamp}.log"
pid_file="$LOG_DIR/live_test_supervisor.pid"

echo "$$" > "$pid_file"

restart_count=0
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] supervisor_start pid=$$ run_id=$RUN_ID data_root=$DATA_ROOT" | tee -a "$supervisor_log"

while true; do
  child_stamp="$(date -u +%Y%m%d_%H%M%S)"
  child_log="$LOG_DIR/live_test_kraken_perps_${child_stamp}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] child_start restart_count=$restart_count log=$child_log" | tee -a "$supervisor_log"

  env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=. \
    MPLCONFIGDIR=/private/tmp/ares_mplconfig \
    python3 -u -m extreme_price_movements.inference.run_inference \
      --live-test \
      --perps \
      --data-root "$DATA_ROOT" \
      --run-id "$RUN_ID" \
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
