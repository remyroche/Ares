#!/usr/bin/env bash
# Operational wrapper only: the hash-bound Python monitor remains the sole
# execution authority.  Keeping the environment in a tiny launcher avoids
# shell/session quoting failures at a live handoff.
set -euo pipefail

cd /Users/remyroche/Documents/Ares
export STRICT_R3_EXECUTION_BUNDLE=config/strict_r3_kraken_live_execution_v157_v181_hash_stability_capacity_fallback_live.json
export STRICT_R3_LIVE_STATE=data_perp/live/strict_r3_kraken_live_state_v113_v181_hash_stability_capacity_fallback_live.json
export STRICT_R3_MONITOR_OUT_ROOT=data_perp/artifacts/strict_r3_live_position_monitor_v157_v181_state113
export STRICT_R3_MONITOR_LOG=logs/strict_r3_live_position_monitor_v181.log
export STRICT_R3_MONITOR_PID=/private/tmp/strict_r3_live_position_monitor_v181.pid
export STRICT_R3_MONITOR_INTERVAL_SECONDS=30
export STRICT_R3_MONITOR_SUBMIT_ORDERS=1
export STRICT_R3_PYTHONPYCACHEPREFIX=/private/tmp/ares_pycache_live_v181

exec /bin/bash scripts/run_strict_r3_live_position_monitor_loop.sh
