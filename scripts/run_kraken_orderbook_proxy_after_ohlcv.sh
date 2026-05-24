#!/bin/zsh
set -u

cd /Users/remyroche/Documents/Ares || exit 1

LOG="logs/kraken_orderbook_proxy_rebuild_after_ohlcv.log"
{
  echo "=== Kraken orderbook proxy watcher start $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
  OHLCV_PID="${1:-1761}"
  while ps -p "${OHLCV_PID}" >/dev/null 2>&1; do
    echo "waiting for OHLCV backfill $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    sleep 60
  done
  echo "OHLCV backfill no longer running; rebuilding orderbook proxy $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  env PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/rebuild_kraken_orderbook_proxy_from_ohlcv.py
  exit_status=$?
  echo "=== Kraken orderbook proxy rebuild exit status=${exit_status} $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
  exit "${exit_status}"
} >> "${LOG}" 2>&1
