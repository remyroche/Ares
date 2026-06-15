#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import scripts.run_top2_recency_pipeline_20260611 as pipeline


BASELINE_RUN_ID = "20260612_183500_top2_reselect_labelhpo_drift_leaflite_native"


def _assert_same_strategies_as_baseline() -> None:
    registry = (
        pipeline.DATA_ROOT
        / "artifacts"
        / BASELINE_RUN_ID
        / "strategy_registry"
        / "top2_mkt_eq_stripped_rule_registry.csv"
    )
    if not registry.exists():
        raise SystemExit(f"baseline strategy registry missing: {registry}")
    with registry.open("r", encoding="utf-8", newline="") as f:
        baseline_ids = [row["strategy_id"] for row in csv.DictReader(f)]
    current_ids = [row["strategy_id"] for row in pipeline.TOP2]
    if baseline_ids != current_ids:
        raise SystemExit(
            "MR/TF runner strategy ids differ from baseline artifact: "
            f"baseline={baseline_ids}, current={current_ids}"
        )
    pipeline._append(
        "Strategy registry parity verified against "
        f"{BASELINE_RUN_ID}: {current_ids}"
    )


def _enable_full_lgbm_search(env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    lgbm_hpo_trials = os.environ.get("EPM_TOP2_LGBM_HPO_TRIALS", "150")
    out.update(
        {
            "EPM_LGBM_USE_NATIVE_PRESET": "0",
            "EPM_LGBM_REQUIRE_NATIVE_PRESET": "0",
            "EPM_LGBM_NATIVE_PRESET_PARAMS_ONLY": "0",
            "EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID": "",
            "EPM_LGBM_HPO_TRIALS": lgbm_hpo_trials,
            "EPM_LGBM_HPO_EARLY_STOP_PATIENCE": os.environ.get(
                "EPM_TOP2_LGBM_HPO_PATIENCE",
                "40",
            ),
            "EPM_LGBM_BASE_LABEL_WEIGHT_HPO": "1",
            # Keep feature selection/HPO/label-HPO active, but bound the
            # highest-cost selector stages so this MR/TF run can coexist with
            # another full training job without swap/I/O stalls.
            "EPM_LGBM_RACE_MAX_ROWS": os.environ.get(
                "EPM_TOP2_LGBM_RACE_MAX_ROWS",
                "60000",
            ),
            "EPM_LGBM_UNIVARIATE_MAX_ROWS": os.environ.get(
                "EPM_TOP2_LGBM_UNIVARIATE_MAX_ROWS",
                "8000",
            ),
            "EPM_LGBM_RELIEF_ENABLED": os.environ.get(
                "EPM_TOP2_LGBM_RELIEF_ENABLED",
                "0",
            ),
            "EPM_LGBM_HPO_MAX_ROWS": os.environ.get(
                "EPM_TOP2_LGBM_HPO_MAX_ROWS",
                "6000",
            ),
            "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS": os.environ.get(
                "EPM_TOP2_LABEL_HPO_LAYER1_TRIALS",
                "300",
            ),
            "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER1_PATIENCE": os.environ.get(
                "EPM_TOP2_LABEL_HPO_LAYER1_PATIENCE",
                "40",
            ),
            "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS": os.environ.get(
                "EPM_TOP2_LABEL_HPO_LAYER2_TRIALS",
                "150",
            ),
            "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER2_PATIENCE": os.environ.get(
                "EPM_TOP2_LABEL_HPO_LAYER2_PATIENCE",
                "30",
            ),
            "EPM_LGBM_LABEL_WEIGHT_HPO_MAX_ROWS": os.environ.get(
                "EPM_TOP2_LABEL_HPO_MAX_ROWS",
                "6000",
            ),
            "EPM_LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS": os.environ.get(
                "EPM_TOP2_LABEL_HPO_ELECTION_MAX_ROWS",
                "20000",
            ),
        }
    )
    out.pop("EPM_BASE_HPO_TRIALS", None)
    out.pop("EPM_META_HPO_TRIALS", None)
    return out


def _run_reselect_and_policy_full_hpo(base_winner: Path, meta_winner: Path) -> str:
    run_id = pipeline.STAGE1_RUN_ID
    marker = pipeline.DATA_ROOT / "artifacts" / run_id / "top2_reselect_policy_complete.json"
    if marker.exists():
        pipeline._append(f"Stage1 already complete: {marker}")
        return run_id
    slice_plan_path = pipeline._build_recent_tail_slice_plan(
        pipeline.POLICY_SLICE_SOURCE_RUN_ID,
        run_id,
    )
    env = pipeline._train_env(
        run_id=run_id,
        label_source_run_id=pipeline.SOURCE_RUN_ID,
        preset_source_run_id=pipeline.SOURCE_RUN_ID,
        slice_plan_path=slice_plan_path,
        params_only=False,
        full_scope=False,
        base_winner=base_winner,
        meta_winner=meta_winner,
    )
    env = _enable_full_lgbm_search(env)
    pipeline._append(
        "Full LGBM search enabled for MR/TF run: "
        f"native_preset={env.get('EPM_LGBM_USE_NATIVE_PRESET')}, "
        f"lgbm_hpo_trials={env.get('EPM_LGBM_HPO_TRIALS')}, "
        f"label_hpo_layer1={env.get('EPM_LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS')}, "
        f"label_hpo_layer2={env.get('EPM_LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS')}"
    )
    if pipeline._base_artifacts_ready(run_id):
        pipeline._append(
            "Stage1 base artifacts already present; skipping train_base and "
            "resuming at train_meta."
        )
    else:
        pipeline._run_step(
            "stage1_train_base_lgbm_fs_hpo_labelhpo_mrtf",
            pipeline._pipeline_cmd("train_base", run_id),
            env,
        )
    pipeline._require_file(
        pipeline.DATA_ROOT / "artifacts" / run_id / "base_models_intermediate.pkl",
        "stage1 base models",
    )

    meta_env = _enable_full_lgbm_search(pipeline._meta_fallback_env(env))
    meta_env["EPM_META_HPO_TRIALS"] = os.environ.get("EPM_TOP2_META_HPO_TRIALS", "150")
    if pipeline._meta_artifacts_ready(run_id):
        pipeline._append(
            "Stage1 meta artifacts already present; skipping train_meta and "
            "resuming at policy-OOS."
        )
    else:
        pipeline._run_step(
            "stage1_train_meta_lgbm_fs_hpo_labelhpo_mrtf",
            pipeline._pipeline_cmd("train_meta", run_id),
            meta_env,
        )
    pipeline._require_file(
        pipeline.DATA_ROOT / "artifacts" / run_id / "models" / "model_state_meta.pkl",
        "stage1 meta state",
    )

    for row in pipeline.TOP2:
        pipeline._run_step(
            f"stage1_policy_oos_{row['strategy_id']}",
            [
                pipeline.sys.executable,
                "-u",
                "scripts/generate_policy_oos_predictions.py",
                "--data-root",
                "data_perp",
                "--run-id",
                run_id,
                "--market-mode",
                "perps",
                "--strategy-id",
                row["strategy_id"],
            ],
            env,
        )
    policy_cmd = [
        pipeline.sys.executable,
        "-u",
        "extreme_price_movements/simple_policy_optimiser.py",
        "--data_root",
        "data_perp",
        "--run_id",
        run_id,
        "--market-mode",
        "perps",
        "--strategy-ids",
        ",".join(f"{row['side']}_{row['strategy_id']}" for row in pipeline.TOP2),
    ]
    with_env = dict(env)
    with_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "1"
    pipeline._run_step("stage1_simple_policy_with_regime_adaptor", policy_cmd, with_env)
    pipeline._copy_policy_variant(run_id, "with_regime_adaptor")
    for row in pipeline.TOP2:
        adaptor = (
            pipeline.DATA_ROOT
            / "artifacts"
            / run_id
            / "simple_policy_optimiser"
            / "regime_adaptors"
            / row["strategy_id"]
            / "regime_adaptor.json"
        )
        if adaptor.exists():
            pipeline._append(f"Regime adaptor artifact present for {row['strategy_id']}: {adaptor}")
    no_env = dict(env)
    no_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "0"
    pipeline._run_step(
        "stage1_simple_policy_without_regime_adaptor",
        policy_cmd + ["--no-regime-adaptor"],
        no_env,
    )
    pipeline._copy_policy_variant(run_id, "without_regime_adaptor")
    pipeline._restore_policy_variant(run_id, "with_regime_adaptor")
    pipeline._verify_stage_logs(run_id)
    marker.write_text(json.dumps({"run_id": run_id, "complete": True}, indent=2) + "\n")
    return run_id


def main() -> int:
    pipeline._append("MR/TF stage1-only comparable run starting")
    _assert_same_strategies_as_baseline()
    base_winner = pipeline.ensure_base_recency_winner()
    pipeline._build_recent_tail_slice_plan(
        pipeline.POLICY_SLICE_SOURCE_RUN_ID,
        pipeline.SOURCE_RUN_ID,
    )
    pipeline.ensure_loc_ema_meta()
    meta_winner = pipeline.ensure_meta_recency_winner()
    pipeline._append(f"Base winner: {base_winner}")
    pipeline._append(f"Meta winner: {meta_winner}")
    stage1_dir = _run_reselect_and_policy_full_hpo(base_winner, meta_winner)
    pipeline._append(f"MR/TF stage1-only comparable run complete: {stage1_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
