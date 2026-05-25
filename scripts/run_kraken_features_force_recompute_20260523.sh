#!/usr/bin/env zsh
set -euo pipefail

cd /Users/remyroche/Documents/Ares

SYMS="$(
  python3 - <<'PY'
import json

path = "data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_verified_universe_latest.json"
with open(path, "r", encoding="utf-8") as fp:
    payload = json.load(fp)
symbols = [
    str(item["perp_symbol"])
    for item in payload.get("symbols", [])
    if item.get("perp_symbol")
]
print(",".join(symbols))
PY
)"

exec env PYTHONUNBUFFERED=1 PYTHONPATH=. \
  python3 -u extreme_price_movements/run_pipeline.py features \
  --market-mode perps \
  --exchange krakenfutures \
  --run-id 20260523_015947 \
  --force-feature-recompute \
  --feature-symbols "${SYMS}"
