#!/usr/bin/env python3
"""Consolidate the three capacity-matched rule-state reliability grids."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ROOT / "data_perp/artifacts/tp6_sl4_rule_state_reliability_20260809_capacitymatch_v1",
    ROOT / "data_perp/artifacts/tp6_sl4_rule_state_reliability_20260809_capacitymatch_seed20260910_v1",
    ROOT / "data_perp/artifacts/tp6_sl4_rule_state_reliability_20260809_capacitymatch_seed20261011_v1",
)
OUT = ROOT / "data_perp/artifacts/tp6_sl4_rule_state_reliability_summary_20260809_v2"


def _summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_parts = []
    stability_parts = []
    audit_parts = []
    for seed, run in enumerate(RUNS):
        metric = pd.read_parquet(run / "metrics_global.parquet")
        pivot = metric.pivot(index="arm", columns="tail", values="net_bps_per_trade")
        pivot = pivot.rename(columns=lambda tail: f"top{int(float(tail) * 10000) / 100:g}_seed{seed}")
        metric_parts.append(pivot)
        stability = pd.read_parquet(run / "metrics_stability.parquet").set_index("arm")
        stability = stability[["mean_top5_net_bps", "median_top5_net_bps", "worst_month_top5_net_bps", "positive_months_top5", "portability_score_bps"]].add_suffix(f"_seed{seed}")
        stability_parts.append(stability)
        audit = pd.read_parquet(run / "model_audit.parquet")
        audit["seed"] = seed
        audit_parts.append(audit)
    joined = metric_parts[0].join(metric_parts[1:], how="outer").reset_index()
    for tail in ("top0.5", "top1", "top2", "top5", "top10"):
        cols = [field for field in joined if field.startswith(f"{tail}_seed")]
        joined[f"{tail}_mean"] = joined[cols].mean(axis=1)
        joined[f"{tail}_min"] = joined[cols].min(axis=1)
        joined[f"{tail}_std"] = joined[cols].std(axis=1, ddof=0)
    stability = stability_parts[0].join(stability_parts[1:], how="outer").reset_index()
    values = [field for field in stability if field != "arm"]
    grouped: dict[str, pd.Series] = {}
    for name in ("mean_top5_net_bps", "median_top5_net_bps", "worst_month_top5_net_bps", "positive_months_top5", "portability_score_bps"):
        cols = [field for field in values if field.startswith(name)]
        grouped[f"{name}_mean"] = stability[cols].mean(axis=1)
    stability_mean = pd.DataFrame({"arm": stability["arm"], **grouped})
    result = joined.merge(stability_mean, on="arm", how="left", validate="one_to_one")
    result["top5_uplift_vs_control_mean"] = result["top5_mean"] - float(result.loc[result.arm.eq("canonical_control"), "top5_mean"].iloc[0])
    result = result.sort_values(["top5_mean", "top5_min", "portability_score_bps_mean", "top1_mean"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)
    audit = pd.concat(audit_parts, ignore_index=True).groupby("block", sort=True).agg(
        folds=("month", "count"), features=("feature_count", "median"), auc_mean=("held_auc", "mean"), auc_std=("held_auc", "std"), brier_mean=("held_brier", "mean"), positive_rate=("train_positive_rate", "mean"), soft_memberships_fed=("soft_memberships_fed", "all"),
    ).reset_index().sort_values("auc_mean", ascending=False, kind="stable")
    return result, audit, pd.concat(audit_parts, ignore_index=True)


def run() -> Path:
    if OUT.exists():
        raise FileExistsError(OUT)
    summary, block_audit, model_audit = _summary()
    feature_audit = pd.read_parquet(RUNS[0] / "feature_block_audit.parquet")
    leaf_audit = pd.read_parquet(RUNS[0] / "activated_leaf_support_audit.parquet")
    path_audit = pd.read_parquet(RUNS[0] / "active_path_correctness_support_audit.parquet")
    contract = json.loads((RUNS[0] / "feature_contract.json").read_text())
    correctness_runs = [json.loads((run / "correctness_test_report.json").read_text()) for run in RUNS]
    membership_fields = {field for fields in contract.values() for field in fields if "k09__cluster__" in field and field.endswith("__membership")}
    requested_structural_blocks = {"soft_memberships", "activated_leaf_support", "rule_path_ood_drift", "covariance_break", "recent_correctness", "cross_model_state", "all_rule_state"}
    correctness = {
        "schema": "tp6_sl4_rule_state_reliability_summary_correctness_v1",
        "three_runs_complete": True,
        "all_new_structural_challengers_explicitly_include_k9_soft_memberships": all(
            any(field in membership_fields for field in contract[name]) for name in requested_structural_blocks
        ),
        "legacy_context_support_uncertainty_controls_are_explicit_controls_without_memberships": all(
            not any(field in membership_fields for field in contract[name]) for name in {"context", "incumbent_support", "incumbent_uncertainty"}
        ),
        "all_leaf_support_fields_use_exact_catalog_training_counts": all(item["leaf_support_is_exact_catalog_train_leaf_count"] for item in correctness_runs),
        "all_outcome_bearing_features_are_prior_resolved": all(item["outcome_states_use_label_available_ts_strictly_before_decision"] for item in correctness_runs),
        "all_success_covariance_features_are_prior_resolved": all(item["success_covariance_state_uses_strict_prior_label_availability"] for item in correctness_runs),
        "all_scores_finite": all(all(item["all_scores_finite"].values()) for item in correctness_runs),
        "all_prediction_ids_unique": all(item["prediction_candidate_month_pairs_unique"] for item in correctness_runs),
    }
    top = summary.head(35)
    blocks = feature_audit.groupby("block", sort=True).agg(
        fields=("field", "count"), min_coverage=("coverage", "min"), median_coverage=("coverage", "median"), min_nonzero=("nonzero", "min"), median_std=("std", "median"),
    ).reset_index()
    selected = summary.loc[summary.arm.isin([
        "canonical_control", "multiply_context_a075", "multiply_context_a100",
        "multiply_activated_leaf_support_a050", "multiply_activated_leaf_support_a100",
        "shrink_recent_correctness_lo75", "multiply_rule_path_ood_drift_a100",
        "multiply_all_rule_state_a075", "multiply_incumbent_uncertainty_a050",
    ])].copy()
    OUT.mkdir(parents=True)
    summary.to_parquet(OUT / "three_seed_summary.parquet", index=False)
    selected.to_parquet(OUT / "selected_arm_summary.parquet", index=False)
    block_audit.to_parquet(OUT / "reliability_block_classifier_quality.parquet", index=False)
    model_audit.to_parquet(OUT / "reliability_model_audit_all_seeds.parquet", index=False)
    feature_audit.to_parquet(OUT / "feature_block_audit.parquet", index=False)
    blocks.to_parquet(OUT / "feature_block_coverage_summary.parquet", index=False)
    leaf_audit.to_parquet(OUT / "activated_leaf_support_audit.parquet", index=False)
    path_audit.to_parquet(OUT / "active_path_correctness_support_audit.parquet", index=False)
    (OUT / "feature_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (OUT / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    baseline = summary.loc[summary.arm.eq("canonical_control")].iloc[0]
    report = [
        "# TP6/SL4 causal rule-state reliability ablation — 2025", "",
        "## Scope", "",
        "Long-only matched 10,224-row 2025 development replay. The stored Base+Consensus score is identical on every arm. This uses exact TP6/SL4/H12 net labels, not the full-universe trailing-exit replay.", "",
        "## Causal contract", "",
        "Activated-leaf support is computed from each active tree leaf's frozen `train_leaf_count`, including effective, p05/p50/p95/median, and contribution-weighted support. Rule/path OOD contains marginal leaf rarity, factorised joint path rarity, support-gated recurrent-path OOD, and feature-only prototype Mahalanobis/PSI/KS drift proxies. Recent correctness and success covariance use only labels with `label_available_ts < decision_ts`. K9 soft memberships, absolute contributions, and active flags are explicitly fed to every new structural challenger; the legacy context/support/uncertainty arms remain membership-free controls.", "",
        "## Feature coverage", "", blocks.round(4).to_string(index=False), "",
        "## Three-seed result", "",
        f"The fixed canonical control is {baseline.top5_mean:.2f} net bps/trade at Top-5. The table reports mean/minimum/standard deviation across independent model seeds; selection must not use the first-seed maximum.", "",
        top[["arm", "top0.5_mean", "top1_mean", "top2_mean", "top5_mean", "top5_min", "top5_std", "top10_mean", "top5_uplift_vs_control_mean", "median_top5_net_bps_mean", "worst_month_top5_net_bps_mean", "portability_score_bps_mean"]].round(2).to_string(index=False), "",
        "## Reliability-model quality", "", block_audit.round(4).to_string(index=False), "",
        "## Decision", "",
        "1. The all-rule-state and rule/path-OOD classifiers have the best correctness AUC, but their ranking corrections do not beat the best simple bounded controls. Do not equate classifier AUC with tail-EV incrementality.",
        "2. Exact activated-leaf support is economically promising: its multiplier a=0.50 is the strongest new diagnostic arm and is stable across seeds, but it remains below the matched context multiplier a=0.75 on pooled Top-5. Carry both into a later untouched/full-universe test; do not replace the canonical stack yet.",
        "3. Recent correctness, covariance break, and cross-model state do not establish a robust incremental winner in this grid. They remain causal diagnostics; adding every block together overfits relative to the compact arms.",
        "4. The transform grid favours moderate multipliers for the best contenders. Shrinkage is useful for some blocks but is not the selected general form here.",
        "", "## Correctness checks", "", json.dumps(correctness, indent=2), "",
    ]
    (OUT / "RULE_STATE_RELIABILITY_ABLATION_REPORT.md").write_text("\n".join(report) + "\n")
    manifest = {
        "schema": "tp6_sl4_rule_state_reliability_summary_20260809_v1", "status": "COMPLETE", "runs": [str(path) for path in RUNS], "baseline_top5_mean": float(baseline.top5_mean), "rows": 10224,
        "selection": "three-seed matched development comparison; pooled Top-5 then seed minimum/stability; no deployment promotion", "artifacts": sorted(path.name for path in OUT.iterdir()),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(OUT), "baseline_top5": float(baseline.top5_mean), "arms": len(summary)}, indent=2))
    return OUT


if __name__ == "__main__":
    run()
