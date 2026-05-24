#!/bin/zsh
set -u

cd /Users/remyroche/Documents/Ares || exit 1

LOG="logs/kraken_dual_perp_backfill_charts.log"
{
  echo "=== Kraken perp backfill start $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
  env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=. \
    EPM_DOWNLOAD_BACKFILL_INTERNAL_GAPS=1 \
    python3 -u scripts/download_kraken_dual_market_data.py \
      --lookback-days 1460 \
      --skip-spot \
      --perp-ohlcv-only \
      --sleep-seconds 0
  status=$?
  echo "=== Kraken perp backfill exit status=${status} $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
  exit "${status}"
} >> "${LOG}" 2>&1
