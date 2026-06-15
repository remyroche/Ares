#!/usr/bin/env python3
"""Run fixed-contract base recency-HPO for the top loc_ema_stack head."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from run_mkt_eq_stripped_heads_hpo_20260609 import (
    DEFAULT_RUN_ID,
    FEATURE_SOURCE_RUN_ID,
    LOG_DIR,
    ROOT,
    _base_env,
    _write_registry,
)


STRATEGY_ID = (
    "loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_"
    "up_down_semivol_ratio_tanh_-0_39156261"
)
TOP3_GRID_PAIRS = "9:0.4,12:0.3,9:0.3"


def main() -> int:
    run_id = os.environ.get("EPM_STRIPPED_MKT_EQ_RUN_ID", DEFAULT_RUN_ID).strip()
    registry_path = _write_registry(run_id)
    env = _base_env(run_id, registry_path)
    winner_root = (
        ROOT
        / "data_perp"
        / "artifacts"
        / run_id
        / "recency_hpo"
        / STRATEGY_ID
        / "top3_confirmed"
    )
    env.update(
        {
            "EPM_LABEL_STRATEGY_IDS": STRATEGY_ID,
            "EPM_BASE_STRATEGY_IDS": STRATEGY_ID,
            "EPM_META_STRATEGY_IDS": STRATEGY_ID,
            "EPM_POLICY_STRATEGY_IDS": STRATEGY_ID,
            "EPM_ARTIFACT_SOURCE_RUN_ID": run_id,
            "EPM_LABEL_SOURCE_RUN_ID": run_id,
            "EPM_LABEL_ARTIFACT_RUN_ID": run_id,
            "EPM_LGBM_USE_NATIVE_PRESET": "1",
            "EPM_LGBM_REQUIRE_NATIVE_PRESET": "1",
            "EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID": run_id,
            "EPM_RECENCY_HPO_ENABLED": "1",
            "EPM_RECENCY_HPO_ONLY": "1",
            "EPM_RECENCY_HPO_SCOPE": "base",
            "EPM_RECENCY_HPO_SCOPE_KEY": STRATEGY_ID,
            "EPM_RECENCY_HPO_STRATEGY_ID": STRATEGY_ID,
            "EPM_RECENCY_HPO_BASE_GRID_PAIRS": TOP3_GRID_PAIRS,
            "EPM_RECENCY_HPO_CONFIRMATION_TOP_N": "3",
            "EPM_RECENCY_HPO_ROOT": str(winner_root),
            "EPM_RECENCY_HPO_BASE_WINNER_PATH": str(winner_root / "base_winner.json"),
            "EPM_TRAIN_EXTEND_TO_LATEST": "1",
            "EPM_TRAIN_EXTEND_DISABLE_EXACT_PLAN_FILTER": "1",
        }
    )
    cmd = [
        sys.executable,
        "-u",
        "extreme_price_movements/run_pipeline.py",
        "recency_hpo",
        "--market-mode",
        "perps",
        "--exchange",
        "kraken",
        "--model-backend",
        "lgbm_pipeline",
        "--ts",
        FEATURE_SOURCE_RUN_ID,
        "--run-id",
        run_id,
    ]
    log_path = LOG_DIR / f"recency_hpo_top3_{STRATEGY_ID}_{run_id}.log"
    with log_path.open("ab", buffering=0) as log_fp:
        log_fp.write(b"\n=== START recency_hpo_loc_ema_stack ===\n")
        log_fp.write(("RUN_ID " + run_id + "\n").encode())
        log_fp.write(("STRATEGY_ID " + STRATEGY_ID + "\n").encode())
        log_fp.write(("GRID_PAIRS " + TOP3_GRID_PAIRS + "\n").encode())
        log_fp.write(("WINNER_PATH " + env["EPM_RECENCY_HPO_BASE_WINNER_PATH"] + "\n").encode())
        log_fp.write(("CMD " + " ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
        )
        ret = proc.wait()
        log_fp.write(f"\n=== END recency_hpo_loc_ema_stack ret={ret} ===\n".encode())
    return int(ret)


if __name__ == "__main__":
    raise SystemExit(main())
