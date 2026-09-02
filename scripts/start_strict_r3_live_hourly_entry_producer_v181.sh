#!/usr/bin/env bash
set -euo pipefail

# Sealed v181 successor.  The runtime only begins at the next fresh UTC hour;
# the recovered 18:00 state is a no-order bridge and can never be executed.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export STRICT_R3_HOURLY_INFERENCE_BUNDLE="config/strict_r3_inference_overlay_long_v153_v181_hash_stability_capacity_fallback.json"
export STRICT_R3_HOURLY_EXECUTION_BUNDLE="config/strict_r3_kraken_live_execution_v157_v181_hash_stability_capacity_fallback_live.json"
export STRICT_R3_HOURLY_LIVE_STATE="data_perp/live/strict_r3_kraken_live_state_v113_v181_hash_stability_capacity_fallback_live.json"
export STRICT_R3_HOURLY_BOOTSTRAP_PREDECESSOR="data_perp/artifacts/strict_r3_stateful_recovery_v181_20260823T180000Z_v1/hour_20260823T180000Z/run"
export STRICT_R3_HOURLY_LOG="logs/strict_r3_live_hourly_entry_producer_v181.log"
export STRICT_R3_HOURLY_PID="/private/tmp/strict_r3_live_hourly_entry_producer_v181.pid"
export STRICT_R3_HOURLY_POLL_SECONDS="3"
export STRICT_R3_HOURLY_FAILED_RETRY_SECONDS="30"
export STRICT_R3_HOURLY_SETTLED_RETRY_SCHEDULE_SECONDS="30,60,120,180"
export STRICT_R3_HOURLY_START_NEXT_FRESH_HOUR="1"
export STRICT_R3_PYTHONPYCACHEPREFIX="/private/tmp/ares_pycache_live_v181"

exec /bin/bash scripts/run_strict_r3_live_hourly_entry_producer_loop.sh
