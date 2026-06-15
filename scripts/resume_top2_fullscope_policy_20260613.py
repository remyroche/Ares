#!/usr/bin/env python3
"""Resume the top-two full-scope run after train_meta.

This starts only the downstream policy stages for the full-scope target:
policy-OOS prediction handoff, simple policy with regime adaptor, and the
no-regime comparison variant.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "EPM_TOP2_RESELECT_RUN_ID",
    "20260612_183500_top2_reselect_labelhpo_drift_leaflite_native",
)
os.environ.setdefault(
    "EPM_TOP2_FULLSCOPE_RUN_ID",
    "20260612_203000_top2_fullscope_labelhpo_drift_leaflite_native",
)

from scripts.run_top2_recency_pipeline_20260611 import (  # noqa: E402
    DATA_ROOT,
    POLICY_SLICE_SOURCE_RUN_ID,
    SOURCE_RUN_ID,
    STAGE1_RUN_ID,
    STAGE3_RUN_ID,
    TOP2,
    _append,
    _build_recent_tail_slice_plan,
    _pipeline_cmd,
    _require_file,
    _run_step,
    _train_env,
    _winner_paths,
)


def _copy_policy_variant_safe(run_id: str, suffix: str) -> None:
    src = DATA_ROOT / "artifacts" / run_id / "simple_policy_optimiser"
    dst = DATA_ROOT / "artifacts" / run_id / f"simple_policy_optimiser_{suffix}"
    _require_file(src / "manifest.json", f"simple_policy_optimiser {suffix} manifest")
    if dst.exists():
        _append(f"Preserving existing simple_policy_optimiser variant: {dst}")
        return
    shutil.copytree(src, dst)
    _append(f"Copied simple_policy_optimiser variant: {dst}")


def _restore_policy_variant_safe(run_id: str, suffix: str) -> None:
    src = DATA_ROOT / "artifacts" / run_id / f"simple_policy_optimiser_{suffix}"
    dst = DATA_ROOT / "artifacts" / run_id / "simple_policy_optimiser"
    _require_file(src / "manifest.json", f"simple_policy_optimiser {suffix} manifest")
    if dst.exists():
        backup = DATA_ROOT / "artifacts" / run_id / "simple_policy_optimiser_last_no_regime"
        if not backup.exists():
            shutil.copytree(dst, backup)
            _append(f"Backed up current simple_policy_optimiser before restore: {backup}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    _append(f"Restored simple_policy_optimiser variant: {dst}")


def run_fullscope_policy_resume() -> None:
    base_winner, meta_winner = _winner_paths()
    _require_file(base_winner, "base recency winner")
    _require_file(meta_winner, "meta recency winner")

    run_root = DATA_ROOT / "artifacts" / STAGE3_RUN_ID
    _require_file(run_root / "base_models_intermediate.pkl", "full-scope base models")
    _require_file(run_root / "models" / "trained_state.pkl", "full-scope trained state")
    _require_file(run_root / "models" / "model_state_meta.pkl", "full-scope meta state")
    _require_file(run_root / "models" / "model_state_meta.manifest.json", "full-scope meta manifest")
    _require_file(run_root / "meta_oof" / "meta_feature_contract.json", "full-scope meta feature contract")
    for row in TOP2:
        strategy_id = row["strategy_id"]
        head = f"short_{strategy_id}_tbm_clf"
        _require_file(run_root / "oof" / f"oof_{strategy_id}_H10.parquet", f"base OOF {strategy_id}")
        _require_file(run_root / "meta_oof" / f"meta_oof_{head}.parquet", f"meta OOF {head}")

    slice_plan_path = _build_recent_tail_slice_plan(POLICY_SLICE_SOURCE_RUN_ID, STAGE3_RUN_ID)
    env = _train_env(
        run_id=STAGE3_RUN_ID,
        label_source_run_id=SOURCE_RUN_ID,
        preset_source_run_id=STAGE1_RUN_ID,
        slice_plan_path=slice_plan_path,
        params_only=False,
        full_scope=True,
        base_winner=base_winner,
        meta_winner=meta_winner,
    )
    env.update(
        {
            "EPM_SIMPLE_POLICY_BASE_TO_META_TOP_FRAC": "0.40",
            "EPM_META_BASE_QUALITY_GATE_ENABLE": "0",
            "EPM_LGBM_N_JOBS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    for row in TOP2:
        strategy_id = row["strategy_id"]
        out_path = run_root / "policy_oos_predictions" / f"policy_oos_{strategy_id}_clf.parquet"
        if out_path.exists() and out_path.stat().st_size > 0:
            _append(f"Policy-OOS already present for {strategy_id}: {out_path}")
            continue
        _run_step(
            f"stage3_policy_oos_{strategy_id}",
            [
                sys.executable,
                "-u",
                "scripts/generate_policy_oos_predictions.py",
                "--data-root",
                "data_perp",
                "--run-id",
                STAGE3_RUN_ID,
                "--market-mode",
                "perps",
                "--strategy-id",
                strategy_id,
            ],
            env,
        )
        _require_file(out_path, f"policy-OOS predictions {strategy_id}")
        _require_file(out_path.with_suffix(".manifest.json"), f"policy-OOS manifest {strategy_id}")

    policy_cmd = [
        sys.executable,
        "-u",
        "extreme_price_movements/simple_policy_optimiser.py",
        "--data_root",
        "data_perp",
        "--run_id",
        STAGE3_RUN_ID,
        "--market-mode",
        "perps",
        "--strategy-ids",
        ",".join(f"{row['side']}_{row['strategy_id']}" for row in TOP2),
    ]

    with_env = dict(env)
    with_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "1"
    _run_step("stage3_simple_policy_with_regime_adaptor", policy_cmd, with_env)
    _copy_policy_variant_safe(STAGE3_RUN_ID, "with_regime_adaptor")

    no_env = dict(env)
    no_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "0"
    _run_step("stage3_simple_policy_without_regime_adaptor", policy_cmd + ["--no-regime-adaptor"], no_env)
    _copy_policy_variant_safe(STAGE3_RUN_ID, "without_regime_adaptor")
    _restore_policy_variant_safe(STAGE3_RUN_ID, "with_regime_adaptor")

    marker = run_root / "top2_fullscope_policy_complete.json"
    marker.write_text(
        json.dumps(
            {
                "run_id": STAGE3_RUN_ID,
                "complete": True,
                "strategy_ids": [row["strategy_id"] for row in TOP2],
                "base_to_meta_top_frac": 0.40,
                "variants": [
                    "simple_policy_optimiser_with_regime_adaptor",
                    "simple_policy_optimiser_without_regime_adaptor",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _append(f"Resume top2 full-scope policy stages completed: {marker}")


def main() -> int:
    run_fullscope_policy_resume()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
