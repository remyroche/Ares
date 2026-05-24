#!/usr/bin/env zsh
set -euo pipefail

BASE_ROOT="/Users/remyroche/Documents/Ares"
OLD_CODE_ROOT="/private/tmp/ares_selector_old"
EXP_ROOT="${BASE_ROOT}/data_perp_policy_rollout_feature_selector_experiment_v3"
RUN_ID="20260522_101003"
CORE_STRATEGY="loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_mkt_ret_eq_24h_1_1280091_mkt_ret_eq_24h_-0_81129736_up_down_semivol_ratio_tanh_-0_39156261"
POLICY_STRATEGY="short_${CORE_STRATEGY}"

cd "${OLD_CODE_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${OLD_CODE_ROOT}"
export EPM_EXCHANGE=kraken
export EPM_DATA_ROOT="${EXP_ROOT}/rollout_on_old_selector"
export EPM_MODEL_BACKEND=lgbm_pipeline
export EPM_POLICY_ROLLOUT_LABELING_ENABLE=1
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
export EPM_SIMPLE_POLICY_RUN_PORTFOLIO_REPLAY=0

python3 -u extreme_price_movements/run_pipeline.py train_base --market-mode perps --exchange kraken --model-backend lgbm_pipeline --run-id "${RUN_ID}"
python3 -u extreme_price_movements/run_pipeline.py train_meta --market-mode perps --exchange kraken --model-backend lgbm_pipeline --run-id "${RUN_ID}"
python3 -u extreme_price_movements/simple_policy_optimiser.py --data_root "${EXP_ROOT}/rollout_on_old_selector_perp" --run_id "${RUN_ID}" --market-mode perps --max-strategies 1
