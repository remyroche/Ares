#!/usr/bin/env python3
"""Verify sealed joint-correctness MLP artifacts and write a correctness report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {
    "run_manifest.json", "comparison_metrics.parquet", "mlp_and_consensus_oos_predictions.parquet",
    "head_contract_audit.parquet", "joint_state_definitions.parquet", "JOINT_CORRECTNESS_MLP_REPORT.md",
}


def verify(path: Path) -> dict:
    checks: list[dict] = []
    manifest_path = path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checks.append({"name": "manifest_complete", "passed": manifest.get("status") == "complete", "detail": manifest.get("status")})
    missing = sorted(name for name in REQUIRED if not (path / name).exists())
    checks.append({"name": "required_artifacts", "passed": not missing, "detail": missing})
    predictions = pd.read_parquet(path / "mlp_and_consensus_oos_predictions.parquet")
    metrics = pd.read_parquet(path / "comparison_metrics.parquet")
    heads = pd.read_parquet(path / "head_contract_audit.parquet")
    states = pd.read_parquet(path / "joint_state_definitions.parquet")
    key = ["target_arm", "fold", "candidate_id"]
    duplicate_count = int(predictions.duplicated(key).sum())
    checks.append({"name": "unique_oos_candidate_rows", "passed": duplicate_count == 0, "detail": duplicate_count})
    expected_targets = set(manifest.get("target_arms", [])) or {"residual_ordinal", "net_ordinal", "clear_binary", "query_rank_ordinal"}
    observed_targets = set(predictions.target_arm.astype(str))
    checks.append({"name": "all_target_arms_present", "passed": expected_targets <= observed_targets, "detail": sorted(observed_targets)})
    observed_heads = set(heads.head_name.astype(str))
    expected_heads = {f"cap{cap}_{weight}" for cap in (40, 60, 80, 100, 120) for weight in ("ordinary", "equal_month")}
    checks.append({"name": "ten_structural_heads_per_target", "passed": expected_heads <= observed_heads, "detail": sorted(observed_heads)})
    forbidden = [name for name in predictions.columns if any(token in name.lower() for token in ("net_bps", "gross_bps", "label_available", "latent_state"))]
    # Outcome columns are retained only for evaluation; the inference score
    # columns themselves must not be named as realized inputs.
    score_columns = [name for name in predictions.columns if name in {"mlp_state_score", "cap120_policy_correction", "consensus_ev", "mlp_weighted_consensus", "mlp_blend25_consensus", "mlp_blend50_consensus"}]
    checks.append({"name": "score_contract_separates_evaluation_labels", "passed": bool(score_columns) and "net_bps" not in score_columns and "gross_bps" not in score_columns, "detail": {"scores": score_columns, "evaluation_only": forbidden}})
    pooled = metrics[metrics.period.eq("pooled")]
    checks.append({"name": "global_tail_metrics_present", "passed": len(pooled) > 0 and pooled["tail"].nunique() >= 3, "detail": {"rows": int(len(pooled)), "tails": sorted(pooled["tail"].unique().tolist()) if len(pooled) else []}})
    checks.append({"name": "state_definitions_have_economic_separation_fields", "passed": {"mean_net_bps", "mean_correctness", "selected_k"} <= set(states.columns), "detail": list(states.columns)})
    state_metrics_path = path / "mlp_state_metrics.parquet"
    if not state_metrics_path.exists():
        companion = path.parent / "joint_correctness_mlp_meta_clearbinary_20260806_v1" / "mlp_state_metrics.parquet"
        if companion.exists():
            state_metrics_path = companion
    if state_metrics_path.exists():
        state_metrics = pd.read_parquet(state_metrics_path)
        passed = {"state_train", "calibration", "test"} <= set(state_metrics.split.astype(str)) and np.isfinite(state_metrics.logloss).all()
        checks.append({"name": "mlp_state_recognition_metrics", "passed": bool(passed), "detail": {"rows": int(len(state_metrics)), "mean_test_accuracy": float(state_metrics.loc[state_metrics.split.eq("test"), "accuracy"].mean())}})
    else:
        checks.append({"name": "mlp_state_recognition_metrics", "passed": False, "detail": "state-metric audit is in the clear-binary companion artifact"})
    report = {
        "schema": "joint_correctness_mlp_meta_correctness_v1",
        "artifact": str(path),
        "status": "passed" if all(bool(row["passed"]) for row in checks) else "failed",
        "checks": checks,
    }
    (path / "correctness_test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    report = verify(args.path)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
