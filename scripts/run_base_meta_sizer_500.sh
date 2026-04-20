#!/bin/zsh
set -euo pipefail

cd /Users/remyroche/Documents/Ares

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp

# Ensure we use the default/full search settings and strategy universe.
unset EPM_META_HPO_TRIALS
unset EPM_META_MAX_STRATEGY_IDS

timestamp() {
  /bin/date -u +"%Y-%m-%d %H:%M:%S UTC"
}

run_step() {
  local label="$1"
  shift
  echo "[$(timestamp)] START ${label}"
  "$@"
  echo "[$(timestamp)] END ${label}"
}

run_step "base_training" \
  python3 -u extreme_price_movements/run_pipeline.py base_training --planned-max-assets 500

run_step "meta_training" \
  python3 -u extreme_price_movements/run_pipeline.py meta_training --planned-max-assets 500

# The repo does not expose `simple_position_sizer.py step --planned-max-assets 500`
# as a valid CLI. `run_pipeline.py sizer --planned-max-assets 500` is the
# pipeline-equivalent entrypoint for the simple position sizer artifact step.
run_step "sizer" \
  python3 -u extreme_price_movements/run_pipeline.py sizer --planned-max-assets 500
