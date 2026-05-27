#!/usr/bin/env zsh
set -euo pipefail

RUN_ID="20260525_010004_nopenalty"
SOURCE_RUN_ID="20260523_015947"
DATA_ROOT="data_perp"
MARKET_MODE="perps"
EXCHANGE="kraken"
SLEEP_SECONDS="${EPM_MONITOR_SLEEP_SECONDS:-900}"
LOG_DIR="logs"
STATE_DIR="${LOG_DIR}/${RUN_ID}_monitor_state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

TARGET_STRATEGY="loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_mkt_ret_eq_24h_1_1280091_mkt_ret_eq_24h_-0_81129736_up_down_semivol_ratio_tanh_-0_39156261"
STRATEGY_2="dist_rolling_7d_high_0_13977644_mkt_ret_eq_24h_-0_56630391_rolling_range_20_-0_40672407"
STRATEGY_3="loc_prev_week_range_pos_48_0_42586401_loc_vwap_dev_z_24_0_10701825_zscore_price_50_1_0128103_mkt_ret_eq_24h_-0_78752208_up_down_return_mass_ratio_tanh_1_1231147"
STRATEGY_4="dist_weekly_vwap_0_074823022_loc_prev_week_range_pos_48_0_48354843_mkt_ret_eq_24h_-0_43956268_volume_autocorr_48_-0_38378653"
ALL_STRATEGIES="${TARGET_STRATEGY},${STRATEGY_2},${STRATEGY_3},${STRATEGY_4}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export EPM_ARTIFACT_SOURCE_RUN_ID="${SOURCE_RUN_ID}"
export EPM_DATA_ROOT="${DATA_ROOT}"
export EPM_MODEL_BACKEND=lgbm_pipeline
export EPM_DISABLE_REGIME_ADAPTORS=1
export EPM_SIMPLE_POLICY_REGIME_ADAPTOR=0
export EPM_BASE_STRATEGY_IDS="${ALL_STRATEGIES}"
export EPM_META_STRATEGY_IDS="${ALL_STRATEGIES}"
export EPM_POLICY_STRATEGY_IDS="${ALL_STRATEGIES}"

status_json() {
  python3 - "$DATA_ROOT" "$RUN_ID" "$ALL_STRATEGIES" <<'PY'
import json
import pickle
import sys
from pathlib import Path

data_root, run_id, strategies_csv = sys.argv[1:4]
strategies = [s for s in strategies_csv.split(",") if s]
root = Path(data_root) / "artifacts" / run_id
bundle_path = root / "base_models_intermediate.pkl"
alpha = {}
if bundle_path.exists():
    with bundle_path.open("rb") as f:
        bundle = pickle.load(f)
    alpha = bundle.get("alpha_models", {}) or {}
present = set()
for side, side_models in alpha.items():
    if isinstance(side_models, dict):
        present.update(str(sid) for sid in side_models)
oof_present = {
    sid: (root / "oof" / f"oof_{sid}_H10.parquet").exists()
    for sid in strategies
}
meta_oofs = list((root / "meta_oof").glob("meta_oof_*_H*.parquet"))
native_dirs = list((root / "models" / "native").glob("*_H10"))
print(json.dumps({
    "base_bundle_count": len(present),
    "base_bundle_missing": [sid for sid in strategies if sid not in present],
    "base_oof_count": sum(1 for ok in oof_present.values() if ok),
    "base_oof_missing": [sid for sid, ok in oof_present.items() if not ok],
    "native_dir_count": len(native_dirs),
    "meta_oof_count": len(meta_oofs),
}, sort_keys=True))
PY
}

base_complete() {
  python3 - "$DATA_ROOT" "$RUN_ID" "$ALL_STRATEGIES" <<'PY'
import pickle
import sys
from pathlib import Path

data_root, run_id, strategies_csv = sys.argv[1:4]
strategies = [s for s in strategies_csv.split(",") if s]
root = Path(data_root) / "artifacts" / run_id
bundle_path = root / "base_models_intermediate.pkl"
if not bundle_path.exists():
    raise SystemExit(1)
with bundle_path.open("rb") as f:
    bundle = pickle.load(f)
alpha = bundle.get("alpha_models", {}) or {}
present = set()
for side_models in alpha.values():
    if isinstance(side_models, dict):
        present.update(str(sid) for sid in side_models)
if any(sid not in present for sid in strategies):
    raise SystemExit(1)
if any(not (root / "oof" / f"oof_{sid}_H10.parquet").exists() for sid in strategies):
    raise SystemExit(1)
PY
}

while true; do
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] monitor status: $(status_json)"

  if ! base_complete; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] base training not complete; sleeping ${SLEEP_SECONDS}s"
    sleep "${SLEEP_SECONDS}"
    continue
  fi

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] base training complete"

  if [[ ! -f "${STATE_DIR}/meta.done" ]]; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] starting train_meta"
    python3 -u extreme_price_movements/run_pipeline.py train_meta \
      --market-mode "${MARKET_MODE}" \
      --exchange "${EXCHANGE}" \
      --run-id "${RUN_ID}"
    touch "${STATE_DIR}/meta.done"
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] train_meta complete"
  fi

  if [[ ! -f "${STATE_DIR}/simple_policy.done" ]]; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] starting simple_policy_optimiser"
    python3 -u extreme_price_movements/simple_policy_optimiser.py \
      --data_root "${DATA_ROOT}" \
      --run_id "${RUN_ID}" \
      --market-mode "${MARKET_MODE}" \
      --strategy-ids "${ALL_STRATEGIES}" \
      --no-regime-adaptor
    touch "${STATE_DIR}/simple_policy.done"
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] simple_policy_optimiser complete"
  fi

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] monitor complete; all requested stages finished"
  exit 0
done
