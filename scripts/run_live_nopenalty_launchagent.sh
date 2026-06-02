#!/usr/bin/env zsh
set -euo pipefail

cd /Users/remyroche/Documents/Ares

export PYTHONUNBUFFERED=1
export PYTHONPATH=/Users/remyroche/Documents/Ares
export EPM_HOURLY_OHLCV_WORKERS=48
export EPM_HOURLY_OHLCV_MAX_WORKERS=64
export EPM_HOURLY_MICRODATA_WORKERS=24

exec /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -u \
  -m extreme_price_movements.inference.run_inference \
  --live-test \
  --perps \
  --data-root data_perp \
  --run-id 20260525_010004_nopenalty
