#!/usr/bin/env python3
"""Run train_meta only for the top loc_ema mkt_ret_eq_24h-stripped head."""
from __future__ import annotations

import os
import subprocess
import sys

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
POLICY_SLICE_SOURCE_RUN_ID = os.environ.get(
    "EPM_TOP2_POLICY_SLICE_SOURCE_RUN_ID",
    DEFAULT_RUN_ID,
).strip()


def main() -> int:
    run_id = os.environ.get("EPM_STRIPPED_MKT_EQ_RUN_ID", DEFAULT_RUN_ID).strip()
    registry_path = _write_registry(run_id)
    env = _base_env(run_id, registry_path)
    env.update(
        {
            "EPM_ARTIFACT_SOURCE_RUN_ID": run_id,
            "EPM_LABEL_SOURCE_RUN_ID": run_id,
            "EPM_FEATURE_SOURCE_RUN_ID": FEATURE_SOURCE_RUN_ID,
            "EPM_META_STRATEGY_IDS": STRATEGY_ID,
            "EPM_POLICY_STRATEGY_IDS": STRATEGY_ID,
            "EPM_TRAIN_SLICE_PLAN_PATH": str(
                ROOT
                / "data_perp"
                / "artifacts"
                / POLICY_SLICE_SOURCE_RUN_ID
                / "slices"
                / "slice_plan.json"
            ),
            "EPM_TRAIN_SLICE_PLAN_EVENT_RUN_ID": POLICY_SLICE_SOURCE_RUN_ID,
        }
    )
    cmd = [
        sys.executable,
        "-u",
        "extreme_price_movements/run_pipeline.py",
        "train_meta",
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
    log_path = LOG_DIR / f"train_meta_loc_ema_{run_id}.log"
    with log_path.open("ab", buffering=0) as log_fp:
        log_fp.write(b"\n=== START train_meta_loc_ema ===\n")
        log_fp.write(("RUN_ID " + run_id + "\n").encode())
        log_fp.write(("META_STRATEGY_IDS " + STRATEGY_ID + "\n").encode())
        log_fp.write(("CMD " + " ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
        )
        ret = proc.wait()
        log_fp.write(f"\n=== END train_meta_loc_ema ret={ret} ===\n".encode())
    return int(ret)


if __name__ == "__main__":
    raise SystemExit(main())
