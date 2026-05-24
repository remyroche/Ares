#!/usr/bin/env zsh
set -euo pipefail

BASE_ROOT="/Users/remyroche/Documents/Ares"
OLD_CODE_ROOT="/private/tmp/ares_selector_old"
EXP_ROOT="${BASE_ROOT}/data_perp_policy_rollout_feature_selector_experiment_v3"
SOURCE_RUN_ID="20260520_004500"
SOURCE_DATA_ROOT="${BASE_ROOT}/data_perp"
SOURCE_FEATURES="${SOURCE_DATA_ROOT}/features/${SOURCE_RUN_ID}"
SOURCE_EXCHANGE="${SOURCE_DATA_ROOT}/exchanges/krakenfutures"
SOURCE_ARTIFACT_FEATURES="${SOURCE_DATA_ROOT}/artifacts/${SOURCE_RUN_ID}/features"

CORE_STRATEGY="loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_mkt_ret_eq_24h_1_1280091_mkt_ret_eq_24h_-0_81129736_up_down_semivol_ratio_tanh_-0_39156261"
POLICY_STRATEGY="short_${CORE_STRATEGY}"

mkdir -p "${EXP_ROOT}/logs"

ensure_links() {
  local variant="$1"
  local run_id="$2"
  local root="${EXP_ROOT}/${variant}_perp"
  mkdir -p "${root}/features" "${root}/exchanges" "${root}/artifacts/${run_id}"
  [[ -e "${root}/features/${run_id}" ]] || ln -s "${SOURCE_FEATURES}" "${root}/features/${run_id}"
  [[ -e "${root}/exchanges/krakenfutures" ]] || ln -s "${SOURCE_EXCHANGE}" "${root}/exchanges/krakenfutures"
  [[ -e "${root}/artifacts/${run_id}/features" ]] || ln -s "${SOURCE_ARTIFACT_FEATURES}" "${root}/artifacts/${run_id}/features"
}

require_labels() {
  local manifest="$1"
  python3 - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"missing labels manifest: {path}")
payload = json.loads(path.read_text())
datasets = payload.get("datasets") or {}
if not datasets:
    raise SystemExit(f"empty labels manifest datasets: {path}")
print(f"verified labels manifest: {len(datasets)} datasets")
PY
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    return 1
  fi
  echo "verified ${label}: ${path}"
}

require_dir_nonempty() {
  setopt local_options extended_glob
  local path="$1"
  local label="$2"
  local files=()
  if [[ -d "${path}" ]]; then
    files=("${path}"/**/*(.N))
  fi
  if (( ${#files} == 0 )); then
    echo "ERROR: missing or empty ${label}: ${path}" >&2
    return 1
  fi
  echo "verified ${label}: ${path}"
}

run_variant() {
  local variant="$1"
  local run_id="$2"
  local rollout="$3"
  local selector="$4"
  local code_root="$5"
  local data_root="${EXP_ROOT}/${variant}"
  local effective_data_root="${EXP_ROOT}/${variant}_perp"
  local artifacts="${effective_data_root}/artifacts/${run_id}"
  local log="${EXP_ROOT}/logs/${variant}.log"

  ensure_links "${variant}" "${run_id}"
  {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] START variant=${variant} run_id=${run_id} rollout=${rollout} selector=${selector} code_root=${code_root}"
    cd "${code_root}"
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="${code_root}"
    export EPM_EXCHANGE=kraken
    export EPM_DATA_ROOT="${data_root}"
    export EPM_MODEL_BACKEND=lgbm_pipeline
    export EPM_POLICY_ROLLOUT_LABELING_ENABLE="${rollout}"
    export EPM_LABEL_STRATEGY_IDS="${CORE_STRATEGY}"
    export EPM_BASE_STRATEGY_IDS="${CORE_STRATEGY}"
    export EPM_META_STRATEGY_IDS="${CORE_STRATEGY}"
    export EPM_POLICY_STRATEGY_IDS="${POLICY_STRATEGY}"
    export EPM_BASE_MAX_STRATEGY_IDS=1
    export EPM_META_MAX_STRATEGY_IDS=1
    export EPM_BASE_HPO_TRIALS="${EPM_BASE_HPO_TRIALS:-80}"
    export EPM_META_HPO_TRIALS="${EPM_META_HPO_TRIALS:-80}"
    export EPM_LGBM_HPO_TRIALS="${EPM_LGBM_HPO_TRIALS:-80}"
    export EPM_LGBM_N_JOBS="${EPM_LGBM_N_JOBS:-3}"
    export EPM_LGBM_FINAL_MODEL_COUNT="${EPM_LGBM_FINAL_MODEL_COUNT:-3}"
    export EPM_LGBM_OOF_DISTILLATION_PASSES="${EPM_LGBM_OOF_DISTILLATION_PASSES:-2}"
    export EPM_LGBM_MIN_OOF_DISTILLATION_PASSES="${EPM_LGBM_MIN_OOF_DISTILLATION_PASSES:-2}"
    export EPM_LGBM_META_MIN_OOF_DISTILLATION_PASSES="${EPM_LGBM_META_MIN_OOF_DISTILLATION_PASSES:-2}"
    export EPM_LGBM_FINAL_FIT_USE_ALL_ROWS="${EPM_LGBM_FINAL_FIT_USE_ALL_ROWS:-1}"
    if [[ -s "${artifacts}/labels/labels_manifest.json" ]]; then
      echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] SKIP labels: existing artifact found"
    else
      python3 -u extreme_price_movements/run_pipeline.py labels --market-mode perps --exchange kraken --run-id "${run_id}"
    fi
    require_labels "${artifacts}/labels/labels_manifest.json"
    if [[ -s "${artifacts}/base_models_intermediate.pkl" ]]; then
      echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] SKIP train_base: existing artifact found"
    else
      python3 -u extreme_price_movements/run_pipeline.py train_base --market-mode perps --exchange kraken --model-backend lgbm_pipeline --run-id "${run_id}"
    fi
    require_file "${artifacts}/base_models_intermediate.pkl" "base intermediate model state"
    if [[ -s "${artifacts}/models/model_state_meta.pkl" ]]; then
      echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] SKIP train_meta: existing artifact found"
    else
      python3 -u extreme_price_movements/run_pipeline.py train_meta --market-mode perps --exchange kraken --model-backend lgbm_pipeline --run-id "${run_id}"
    fi
    require_file "${artifacts}/models/model_state_meta.pkl" "meta model state"
    require_dir_nonempty "${artifacts}/meta_oof" "meta OOF directory"
    if [[ -s "${artifacts}/policy_optimisation.json" ]]; then
      echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] SKIP simple_policy_optimiser: existing artifact found"
    else
      python3 -u extreme_price_movements/simple_policy_optimiser.py --data_root "${effective_data_root}" --run_id "${run_id}" --market-mode perps --max-strategies 1
    fi
    require_file "${artifacts}/policy_optimisation.json" "policy optimisation output"
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] DONE variant=${variant}"
  } 2>&1 | tee "${log}"
}

select_best_new_rollout() {
  python3 - "${EXP_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
variants = [
    ("rollout_on_new", "20260522_101001", "1"),
    ("rollout_off_new", "20260522_101002", "0"),
]


def first_strategy(payload: dict) -> dict:
    for key, value in payload.items():
        if str(key).startswith("__"):
            continue
        if isinstance(value, dict):
            return value
    return {}


def finite_float(value, default=-1e100) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


rows = []
for variant, run_id, rollout in variants:
    path = root / f"{variant}_perp" / "artifacts" / run_id / "policy_optimisation.json"
    if not path.exists():
        raise SystemExit(f"missing policy optimisation output for {variant}: {path}")
    payload = json.loads(path.read_text())
    strategy = first_strategy(payload)
    metrics = strategy.get("final_policy_deployment_metrics") or {}
    rows.append(
        {
            "variant": variant,
            "run_id": run_id,
            "rollout": rollout,
            "mean_net_trade": finite_float(metrics.get("mean_net_trade")),
            "net_pnl": finite_float(metrics.get("net_pnl")),
            "sortino": finite_float(metrics.get("sortino")),
            "n_trades": finite_float(metrics.get("n_trades"), default=0.0),
        }
    )

best = max(
    rows,
    key=lambda row: (
        row["mean_net_trade"],
        row["net_pnl"],
        row["sortino"],
        row["n_trades"],
    ),
)
print(
    "selected_new_rollout "
    f"variant={best['variant']} "
    f"run_id={best['run_id']} "
    f"rollout={best['rollout']} "
    f"mean_net_trade={best['mean_net_trade']:.8f} "
    f"net_pnl={best['net_pnl']:.8f} "
    f"sortino={best['sortino']:.8f} "
    f"n_trades={int(best['n_trades'])}"
)
print(best["rollout"])
PY
}

run_variant "rollout_on_new" "20260522_101001" "1" "new" "${BASE_ROOT}"
run_variant "rollout_off_new" "20260522_101002" "0" "new" "${BASE_ROOT}"
best_rollout_output="$(select_best_new_rollout)"
echo "${best_rollout_output}" | head -n 1
best_rollout="$(echo "${best_rollout_output}" | tail -n 1)"
if [[ "${best_rollout}" == "1" ]]; then
  run_variant "rollout_on_old_selector" "20260522_101003" "1" "old_selector" "${OLD_CODE_ROOT}"
else
  run_variant "rollout_off_old_selector" "20260522_101004" "0" "old_selector" "${OLD_CODE_ROOT}"
fi

python3 "${BASE_ROOT}/scripts/summarize_policy_rollout_selector_experiment.py" "${EXP_ROOT}"
