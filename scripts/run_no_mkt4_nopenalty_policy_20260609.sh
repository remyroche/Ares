#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-20260609_210500}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-20260523_015947}"
PRESET_RUN_ID="${PRESET_RUN_ID:-20260606_recency5fold_current4_deployed_preset_extended}"
STRATEGY_SOURCE_CSV="${STRATEGY_SOURCE_CSV:-data_perp/artifacts/20260523_015947/policy_oos_retrain_strategy_source_perps.csv}"
DATA_ROOT="${DATA_ROOT:-data_perp}"
LOG_DIR="${LOG_DIR:-logs}"

STRATS="${STRATS:-dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_leverage_build_score_0_45107844_return_autocorr_48_1_18643_rolling_range_20_-0_25967735,bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828,bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_price_rv_15d_robust_z_0_060036644,asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597}"
SIMPLE_POLICY_STRATS="${SIMPLE_POLICY_STRATS:-long_dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_leverage_build_score_0_45107844_return_autocorr_48_1_18643_rolling_range_20_-0_25967735,long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828,short_bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_price_rv_15d_robust_z_0_060036644,short_asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597}"

mkdir -p "$LOG_DIR" "$DATA_ROOT/artifacts/$RUN_ID/slices"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/mplconfig}"
mkdir -p "$MPLCONFIGDIR"

fail_if_log_failed() {
  local log_file="$1"
  if grep -E "PIPELINE FAILED|ERROR: No alpha label datasets found|ERROR: Base models intermediate not found|Traceback" "$log_file" >/dev/null 2>&1; then
    echo "Failure marker found in $log_file" >&2
    tail -80 "$log_file" >&2 || true
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo "Required artifact missing or empty: $path" >&2
    exit 1
  fi
}

RUN_ID="$RUN_ID" SOURCE_RUN_ID="$SOURCE_RUN_ID" python3 - <<'PY'
import json
import os
from pathlib import Path

from extreme_price_movements.simple_policy_optimiser import (
    _load_policy_stage_view,
    _load_slice_plan_source_validation,
)

run_id = os.environ["RUN_ID"]
source_run_id = os.environ["SOURCE_RUN_ID"]
source = Path("data_perp/artifacts") / source_run_id / "slices" / "slice_plan.json"
target = Path("data_perp/artifacts") / run_id / "slices" / "slice_plan.json"
payload = json.loads(source.read_text())
payload["run_id"] = run_id
target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
validation = _load_slice_plan_source_validation(target)
stage_view, stage_name = _load_policy_stage_view(target)
if stage_name != "policy_optimiser" or not stage_view.get("allowed_periods"):
    raise SystemExit(f"missing strict policy_optimiser stage view in {target}")
if not validation.get("oos_policy_slice_verified"):
    raise SystemExit(f"policy-OOS slice is not verified in {target}: {validation}")
if validation.get("policy_optimiser_holdout_start_months_ago") != [16]:
    raise SystemExit(f"unexpected holdout start months: {validation}")
if validation.get("policy_optimiser_holdout_end_months_ago") != [12]:
    raise SystemExit(f"unexpected holdout end months: {validation}")
if "policy_holdout_middle" not in set(validation.get("policy_holdout_predict_roles") or []):
    raise SystemExit(f"policy holdout is not middle holdout: {validation}")
print(
    "Copied verified middle-holdout slice plan to "
    f"{target} (periods={len(stage_view.get('allowed_periods') or [])}, "
    f"predict={validation.get('policy_optimiser_predict_start')}.."
    f"{validation.get('policy_optimiser_predict_end')})"
)
PY

COMMON_ENV=(
  PYTHONUNBUFFERED=1
  PYTHONPATH=.
  MPLCONFIGDIR=/private/tmp/mplconfig
  EPM_ARTIFACT_SOURCE_RUN_ID="$SOURCE_RUN_ID"
  EPM_TRAIN_SLICE_PLAN_EVENT_RUN_ID="$SOURCE_RUN_ID"
  EPM_MASK_STRATEGY_SOURCE_CSV="$STRATEGY_SOURCE_CSV"
  EPM_MASK_STRATEGY_TOP_N=10
  EPM_MASK_STRATEGY_RANKING_METRIC=stage_e_rank_score
  EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID="$PRESET_RUN_ID"
  EPM_LGBM_USE_NATIVE_PRESET=1
  EPM_LGBM_REQUIRE_NATIVE_PRESET=1
  EPM_BASE_HPO_TRIALS=0
  EPM_META_HPO_TRIALS=0
  EPM_MIN_TRAIN_SAMPLES=800
  EPM_BASE_MIN_SAMPLES_HARD_FLOOR=200
  EPM_SAMPLE_WEIGHT_OPT_MIN_SAMPLES=50
  EPM_BASE_REQUIRE_POSITIVE_OOF_EXPECTANCY=0
  EPM_META_BASE_QUALITY_GATE_ENABLE=0
  EPM_BASE_STRATEGY_IDS="$STRATS"
  EPM_META_STRATEGY_IDS="$STRATS"
  EPM_LABEL_STRATEGY_IDS="$STRATS"
  EPM_LABEL_PERSIST_INCREMENTAL=0
  EPM_LABEL_INCREMENTAL_ONLY_MISSING=0
  EPM_POLICY_STRATEGY_IDS="$STRATS"
  EPM_REQUIRE_STRATEGY_ALLOWLIST=1
  EPM_MODEL_BACKEND=lgbm_pipeline
  EPM_TRAINING_MODEL_BACKEND=lgbm_pipeline
  EPM_TRAIN_EXTEND_TO_LATEST=0
  EPM_LABEL_WEIGHT_DISABLE=1
  EPM_EXCHANGE=kraken
)

check_required_h5_labels() {
  python3 - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

root = Path("data_perp/artifacts/20260523_015947/labels")
min_rows = 50000
min_span_days = 365
max_age_days = 7
required = [
    "train_long_dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_leverage_build_score_0_45107844_return_autocorr_48_1_18643_rolling_range_20_-0_25967735_5.parquet",
    "train_long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828_5.parquet",
    "train_short_bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_price_rv_15d_robust_z_0_060036644_5.parquet",
    "train_short_asset_minus_mkt_oi_1d_peer_resid_0_34164831_oi_expansion_compression_balance_24h_0_42287597_5.parquet",
]
missing = [str(root / name) for name in required if not (root / name).exists() or (root / name).stat().st_size <= 0]
if missing:
    raise SystemExit("Missing required H5 label parquet(s): " + ", ".join(missing))
bad = []
for name in required:
    path = root / name
    pf = pq.ParquetFile(path)
    rows = int(pf.metadata.num_rows)
    try:
        ts = pq.read_table(path, columns=["__ts__"]).to_pandas()["__ts__"]
        ts = pd.to_datetime(ts, utc=True, errors="coerce")
        ts_min = ts.min()
        ts_max = ts.max()
        span_days = float((ts_max - ts_min).total_seconds() / 86400.0) if pd.notna(ts_min) and pd.notna(ts_max) else 0.0
        age_days = float((pd.Timestamp.utcnow() - ts_max).total_seconds() / 86400.0) if pd.notna(ts_max) else 9999.0
    except Exception as exc:
        bad.append(f"{name}: timestamp_read_failed={exc}")
        continue
    if rows < min_rows or span_days < min_span_days or age_days > max_age_days:
        bad.append(
            f"{name}: rows={rows:,} span_days={span_days:.1f} "
            f"max_ts={ts_max} age_days={age_days:.1f}"
        )
if bad:
    raise SystemExit(
        "Required H5 label parquet coverage is insufficient for full retrain:\n"
        + "\n".join(f"  - {item}" for item in bad)
    )
print("Required H5 label parquets present with full-history coverage: 4/4")
PY
}

if check_required_h5_labels; then
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] labels already present; skipping label generation"
else
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] labels start source_run_id=$SOURCE_RUN_ID"
  python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import shutil

root = Path("data_perp/artifacts/20260523_015947/labels")
if root.exists():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = root.parent / f"labels_backup_before_full_regen_{stamp}"
    shutil.copytree(root, backup)
    print(f"Backed up existing labels to {backup}")
PY
  env "${COMMON_ENV[@]}" \
    python3 -u extreme_price_movements/run_pipeline.py labels \
      --perps --exchange krakenfutures --model-backend lgbm_pipeline --run-id "$SOURCE_RUN_ID" --horizons 5 \
    > "$LOG_DIR/labels_${SOURCE_RUN_ID}_no_mkt4_H5.log" 2>&1
  fail_if_log_failed "$LOG_DIR/labels_${SOURCE_RUN_ID}_no_mkt4_H5.log"
  check_required_h5_labels
fi
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] labels complete"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] train_base start run_id=$RUN_ID"
env "${COMMON_ENV[@]}" \
  python3 -u extreme_price_movements/run_pipeline.py train_base \
    --perps --exchange krakenfutures --model-backend lgbm_pipeline --run-id "$RUN_ID" \
  > "$LOG_DIR/train_${RUN_ID}_no_mkt4_train_base.log" 2>&1
fail_if_log_failed "$LOG_DIR/train_${RUN_ID}_no_mkt4_train_base.log"
require_file "$DATA_ROOT/artifacts/$RUN_ID/base_models_intermediate.pkl"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] train_base complete"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] train_meta start run_id=$RUN_ID"
env "${COMMON_ENV[@]}" \
  python3 -u extreme_price_movements/run_pipeline.py train_meta \
    --perps --exchange krakenfutures --model-backend lgbm_pipeline --run-id "$RUN_ID" \
  > "$LOG_DIR/train_${RUN_ID}_no_mkt4_train_meta.log" 2>&1
fail_if_log_failed "$LOG_DIR/train_${RUN_ID}_no_mkt4_train_meta.log"
require_file "$DATA_ROOT/artifacts/$RUN_ID/models/model_state_meta.pkl"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] train_meta complete"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] policy_oos_predictions start run_id=$RUN_ID"
env "${COMMON_ENV[@]}" \
  python3 -u scripts/generate_policy_oos_predictions.py \
    --data-root "$DATA_ROOT" --run-id "$RUN_ID" --market-mode perps \
  > "$LOG_DIR/policy_oos_predictions_${RUN_ID}_no_mkt4.log" 2>&1
fail_if_log_failed "$LOG_DIR/policy_oos_predictions_${RUN_ID}_no_mkt4.log"
require_file "$DATA_ROOT/artifacts/$RUN_ID/policy_oos_predictions/manifest.json"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] policy_oos_predictions complete"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] simple_policy_optimiser start run_id=$RUN_ID"
env "${COMMON_ENV[@]}" \
  EPM_POLICY_STRATEGY_IDS="$SIMPLE_POLICY_STRATS" \
  python3 -u extreme_price_movements/simple_policy_optimiser.py \
    --data_root "$DATA_ROOT" --run_id "$RUN_ID" --market-mode perps --strategy-ids "$SIMPLE_POLICY_STRATS" \
  > "$LOG_DIR/simple_policy_optimiser_${RUN_ID}_no_mkt4.log" 2>&1
fail_if_log_failed "$LOG_DIR/simple_policy_optimiser_${RUN_ID}_no_mkt4.log"
require_file "$DATA_ROOT/artifacts/$RUN_ID/simple_policy_optimiser/deployment/best_policy_params.json"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] simple_policy_optimiser complete"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] sequence complete: $DATA_ROOT/artifacts/$RUN_ID"
