#!/usr/bin/env python3
"""Resume stage1 simple-policy with the selected side-qualified top-two heads.

The current top-two registry contains two short strategies. Passing unprefixed
strategy IDs to simple_policy_optimiser expands the allowlist to long+short and
causes the optimiser to wait for long policy-OOS predictions that should not
exist for this run. This wrapper preserves prior partial policy folders and
reruns the stage with explicit short_* IDs.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
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
    STAGE1_RUN_ID,
    TOP2,
    _append,
    _common_env,
    _require_file,
    _run_step,
)

REPORT_FILES = (
    "policy_optimisation.json",
    "policy_optimisation_perps.json",
    "policy_optimisation_oos_metrics.json",
    "policy_optimisation_oos_metrics_perps.json",
    "best_policy_params.json",
    "best_policy_params_perps.json",
    "strategy_for_inference.json",
    "strategy_for_inference_perps.json",
)
REPORT_DIRS = (
    "policy_params",
    "portfolio_policy_replay",
)
REQUIRED_REPORT_OUTPUTS = (
    "policy_optimisation_oos_metrics.json",
    "policy_optimisation_oos_metrics_perps.json",
    "policy_params/training_live_parity_contract.json",
    "policy_params/best_policy_params.json",
    "portfolio_policy_replay/portfolio_policy_replay_report.json",
    "portfolio_policy_replay/per_fold_validation_metrics.json",
)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _preserve_existing(path: Path, reason: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}_{reason}_{_timestamp()}")
    path.rename(backup)
    _append(f"Preserved existing policy output: {path} -> {backup}")
    return backup


def _require_policy_oos(run_root: Path, side_strategy_id: str) -> None:
    parquet = run_root / "policy_oos_predictions" / f"policy_oos_{side_strategy_id}_clf.parquet"
    manifest = parquet.with_suffix(".manifest.json")
    _require_file(parquet, f"policy-OOS predictions {side_strategy_id}")
    _require_file(manifest, f"policy-OOS manifest {side_strategy_id}")


def _copy_complete_variant(run_root: Path, suffix: str) -> Path:
    src = run_root / "simple_policy_optimiser"
    dst = run_root / f"simple_policy_optimiser_{suffix}"
    _require_policy_output(src, suffix)
    if dst.exists():
        _preserve_existing(dst, "pre_short_only")
    shutil.copytree(src, dst)
    _append(f"Copied complete simple_policy_optimiser variant: {dst}")
    return dst


def _restore_variant(run_root: Path, suffix: str) -> None:
    src = run_root / f"simple_policy_optimiser_{suffix}"
    dst = run_root / "simple_policy_optimiser"
    _require_policy_output(src, suffix)
    if dst.exists():
        _preserve_existing(dst, "last_no_regime")
    shutil.copytree(src, dst)
    _append(f"Restored canonical simple_policy_optimiser from {src}")


def _report_output_complete(path: Path) -> bool:
    return path.exists() and all(
        (path / rel).exists() and (path / rel).stat().st_size > 0
        for rel in REQUIRED_REPORT_OUTPUTS
    )


def _require_report_output(path: Path, label: str) -> None:
    if _report_output_complete(path):
        return
    missing = [
        str(path / rel)
        for rel in REQUIRED_REPORT_OUTPUTS
        if not ((path / rel).exists() and (path / rel).stat().st_size > 0)
    ]
    raise RuntimeError(f"simple-policy report output {label} incomplete; missing={missing}")


def _copy_complete_report_variant(run_root: Path, suffix: str) -> Path:
    dst = run_root / f"simple_policy_reports_{suffix}"
    if dst.exists():
        _preserve_existing(dst, "pre_short_only")
    dst.mkdir(parents=True, exist_ok=False)
    for rel in REPORT_FILES:
        src = run_root / rel
        if src.exists() and src.stat().st_size > 0:
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
    for rel in REPORT_DIRS:
        src = run_root / rel
        if src.exists():
            shutil.copytree(src, dst / rel)
    _require_report_output(dst, suffix)
    _append(f"Copied complete simple-policy report variant: {dst}")
    return dst


def _restore_report_variant(run_root: Path, suffix: str) -> None:
    src = run_root / f"simple_policy_reports_{suffix}"
    _require_report_output(src, suffix)
    for rel in REPORT_FILES:
        source = src / rel
        if not source.exists():
            continue
        target = run_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for rel in REPORT_DIRS:
        source = src / rel
        if not source.exists():
            continue
        target = run_root / rel
        if target.exists():
            _preserve_existing(target, "last_no_regime_report")
        shutil.copytree(source, target)
    _append(f"Restored canonical simple-policy reports from {src}")


def _policy_output_complete(path: Path) -> bool:
    required = (
        path / "deployment" / "best_policy_params.json",
        path / "simple_policy_candidates_deployable.parquet",
        path / "simple_policy_candidates_metadata.json",
        path / "rank_reference" / "manifest.json",
        path / "training_live_parity_contract.json",
    )
    return path.exists() and all(p.exists() and p.stat().st_size > 0 for p in required)


def _require_policy_output(path: Path, label: str) -> None:
    if _policy_output_complete(path):
        return
    required = (
        path / "deployment" / "best_policy_params.json",
        path / "simple_policy_candidates_deployable.parquet",
        path / "simple_policy_candidates_metadata.json",
        path / "rank_reference" / "manifest.json",
        path / "training_live_parity_contract.json",
    )
    missing = [str(p) for p in required if not (p.exists() and p.stat().st_size > 0)]
    raise RuntimeError(f"simple_policy_optimiser {label} output incomplete; missing={missing}")


def main() -> int:
    run_root = DATA_ROOT / "artifacts" / STAGE1_RUN_ID
    side_strategy_ids = [f"{row['side']}_{row['strategy_id']}" for row in TOP2]

    _require_file(run_root / "base_models_intermediate.pkl", "stage1 base models")
    _require_file(run_root / "models" / "model_state_meta.pkl", "stage1 meta state")
    _require_file(run_root / "slices" / "slice_plan.json", "stage1 slice plan")
    for side_strategy_id in side_strategy_ids:
        _require_policy_oos(run_root, side_strategy_id)

    for name in (
        "simple_policy_optimiser",
        "simple_policy_optimiser_with_regime_adaptor",
        "simple_policy_optimiser_without_regime_adaptor",
        "simple_policy_reports_with_regime_adaptor",
        "simple_policy_reports_without_regime_adaptor",
    ):
        path = run_root / name
        if name.startswith("simple_policy_reports"):
            complete = _report_output_complete(path)
        else:
            complete = _policy_output_complete(path)
        if path.exists() and not complete:
            _preserve_existing(path, "incomplete")

    env = _common_env(STAGE1_RUN_ID)
    env.update(
        {
            "EPM_MASK_STRATEGY_SOURCE_CSV": str(
                run_root / "strategy_registry" / "top2_mkt_eq_stripped_rule_registry.csv"
            ),
            "EPM_SIMPLE_POLICY_BASE_TO_META_TOP_FRAC": "0.40",
            "EPM_SIMPLE_POLICY_USE_POLICY_OOS_PREDICTIONS": "1",
            "EPM_SIMPLE_POLICY_USE_PRECOMPUTED_META_OOF": "0",
            "EPM_SIMPLE_POLICY_ALLOW_FINAL_FIT_POLICY_GENERATION": "0",
            "EPM_SIMPLE_POLICY_ALLOW_META_OOF_POLICY_SOURCE": "0",
            "EPM_LGBM_N_JOBS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    policy_cmd = [
        sys.executable,
        "-u",
        "extreme_price_movements/simple_policy_optimiser.py",
        "--data_root",
        "data_perp",
        "--run_id",
        STAGE1_RUN_ID,
        "--market-mode",
        "perps",
        "--strategy-ids",
        ",".join(side_strategy_ids),
    ]

    with_env = dict(env)
    with_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "1"
    _run_step("stage1_simple_policy_short_only_with_regime_adaptor", policy_cmd, with_env)
    _copy_complete_variant(run_root, "with_regime_adaptor")
    _copy_complete_report_variant(run_root, "with_regime_adaptor")

    no_env = dict(env)
    no_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "0"
    _run_step(
        "stage1_simple_policy_short_only_without_regime_adaptor",
        policy_cmd + ["--no-regime-adaptor"],
        no_env,
    )
    _copy_complete_variant(run_root, "without_regime_adaptor")
    _copy_complete_report_variant(run_root, "without_regime_adaptor")
    _restore_variant(run_root, "with_regime_adaptor")
    _restore_report_variant(run_root, "with_regime_adaptor")

    marker = run_root / "top2_stage1_simple_policy_short_only_complete.json"
    marker.write_text(
        json.dumps(
            {
                "run_id": STAGE1_RUN_ID,
                "complete": True,
                "strategy_ids": side_strategy_ids,
                "base_to_meta_top_frac": 0.40,
                "variants": [
                    "simple_policy_optimiser_with_regime_adaptor",
                    "simple_policy_optimiser_without_regime_adaptor",
                ],
                "report_variants": [
                    "simple_policy_reports_with_regime_adaptor",
                    "simple_policy_reports_without_regime_adaptor",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _append(f"Stage1 short-only simple-policy resume completed: {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
