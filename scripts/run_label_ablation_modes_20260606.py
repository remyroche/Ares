#!/usr/bin/env python3
"""Run one-strategy label/sample-weight ablation modes sequentially."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

STRATEGY_ID = (
    "dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_"
    "leverage_build_score_0_45107844_return_autocorr_48_1_18643_"
    "rolling_range_20_-0_25967735"
)

BASE_ENV = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONPATH": ".",
    "EPM_ARTIFACT_SOURCE_RUN_ID": "20260523_015947",
    "EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID": "20260525_010004_nopenalty",
    "EPM_MODEL_BACKEND": "lgbm_pipeline",
    "EPM_TRAINING_NO_PENALTY": "1",
    "EPM_LGBM_USE_NATIVE_PRESET": "1",
    "EPM_LGBM_CV_SPLITS": "3",
    "EPM_LGBM_RECENCY_WEIGHTING": "1",
    "EPM_LGBM_BASE_RECENCY_HALF_LIFE_DAYS": "365",
    "EPM_LGBM_META_RECENCY_HALF_LIFE_DAYS": "182.5",
    "EPM_LGBM_TRUE_SOFT_LABELS": "1",
    "EPM_LGBM_REBALANCE_EFFECTIVE_CLASSES": "1",
    "EPM_LGBM_REBALANCE_POS_MASS_MIN": "0.25",
    "EPM_LGBM_REBALANCE_POS_MASS_MAX": "0.55",
    "EPM_LGBM_REBALANCE_MAX_MULTIPLIER": "2.0",
    "EPM_TRAIN_EXTEND_TO_LATEST": "1",
    "EPM_TRAIN_RECENT_DAYS": "730",
    "EPM_BASE_STRATEGY_IDS": STRATEGY_ID,
    "EPM_META_STRATEGY_IDS": STRATEGY_ID,
    "EPM_REQUIRE_STRATEGY_ALLOWLIST": "1",
    "EPM_SKIP_MASK_STRATEGY_PARAMS": "1",
    "EPM_EXECUTION_AWARE_COST_BPS": "68.83",
}

RUNS = [
    (
        "1_current_soft_current_weights_interleaved_default_3fold_2y",
        "20260606_label_ablation_1_current_default_interleaved_3fold_2y",
        {
            "EPM_LABEL_ABLATION_MODE": "current",
            "EPM_LGBM_PURGED_CV": "0",
            "EPM_LGBM_CV_MODE": "interleaved_spread",
        },
    ),
    (
        "2_current_soft_execution_aware_weights_interleaved_default_3fold_2y",
        "20260606_label_ablation_2_exec_weights_interleaved_3fold_2y",
        {
            "EPM_LABEL_ABLATION_MODE": "execution_aware_weights",
            "EPM_LGBM_PURGED_CV": "0",
            "EPM_LGBM_CV_MODE": "interleaved_spread",
        },
    ),
    (
        "3_net_executable_soft_current_weights_interleaved_default_3fold_2y",
        "20260606_label_ablation_3_net_exec_soft_interleaved_3fold_2y",
        {
            "EPM_LABEL_ABLATION_MODE": "net_executable_soft_label",
            "EPM_LGBM_PURGED_CV": "0",
            "EPM_LGBM_CV_MODE": "interleaved_spread",
        },
    ),
]


def main() -> int:
    for mode_name, run_id, extra_env in RUNS:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(extra_env)
        log_path = LOG_DIR / f"train_{run_id}.log"
        cmd = [
            sys.executable,
            "-u",
            "extreme_price_movements/run_pipeline.py",
            "train",
            "--market-mode",
            "perps",
            "--exchange",
            "kraken",
            "--ts",
            "20260523_015947",
            "--run-id",
            run_id,
        ]
        with log_path.open("ab", buffering=0) as log_fp:
            log_fp.write(f"\n=== START {mode_name} run_id={run_id} ===\n".encode())
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                env=env,
            )
            ret = proc.wait()
            log_fp.write(f"\n=== END {mode_name} run_id={run_id} ret={ret} ===\n".encode())
        if ret != 0:
            return ret
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
