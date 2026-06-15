#!/usr/bin/env python3
"""Run train_meta only for the top two mkt_ret_eq_24h-stripped heads."""
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


TOP2_STRATEGY_IDS = [
    "loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_up_down_semivol_ratio_tanh_-0_39156261",
    "dist_rolling_7d_high_0_13977644_rolling_range_20_-0_40672407",
]


def main() -> int:
    run_id = os.environ.get("EPM_STRIPPED_MKT_EQ_RUN_ID", DEFAULT_RUN_ID).strip()
    registry_path = _write_registry(run_id)
    env = _base_env(run_id, registry_path)
    ids_csv = ",".join(TOP2_STRATEGY_IDS)
    env.update(
        {
            "EPM_ARTIFACT_SOURCE_RUN_ID": run_id,
            "EPM_LABEL_SOURCE_RUN_ID": run_id,
            "EPM_FEATURE_SOURCE_RUN_ID": FEATURE_SOURCE_RUN_ID,
            "EPM_META_STRATEGY_IDS": ids_csv,
            "EPM_POLICY_STRATEGY_IDS": ids_csv,
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
    log_path = LOG_DIR / f"train_meta_top2_{run_id}.log"
    with log_path.open("ab", buffering=0) as log_fp:
        log_fp.write(b"\n=== START train_meta_top2 ===\n")
        log_fp.write(("RUN_ID " + run_id + "\n").encode())
        log_fp.write(("META_STRATEGY_IDS " + ids_csv + "\n").encode())
        log_fp.write(("CMD " + " ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
        )
        ret = proc.wait()
        log_fp.write(f"\n=== END train_meta_top2 ret={ret} ===\n".encode())
    return int(ret)


if __name__ == "__main__":
    raise SystemExit(main())
