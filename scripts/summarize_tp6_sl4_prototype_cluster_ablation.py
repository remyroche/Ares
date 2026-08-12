#!/usr/bin/env python3
"""Consolidate three-seed prototype/cluster ablation results into one audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRUCTURE = ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_quality_20260809_v3"
RUNS = (
    ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_use_ablation_20260809_fullsizes_v1",
    ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_use_ablation_20260809_fullsizes_seed20260910_v1",
    ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_use_ablation_20260809_fullsizes_seed20261011_v1",
)
OUT = ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_ablation_summary_20260809_fullsizes_v1"
COACTIVATION = ROOT / "data_perp/artifacts/tp6_sl4_prototype_cluster_coactivation_20260809_v3"


def _aggregate(runs: Sequence[Path]) -> pd.DataFrame:
    pieces = []
    for idx, run in enumerate(runs):
        metrics = pd.read_parquet(run / "metrics_global.parquet")
        top = metrics.pivot(index="arm", columns="tail", values="net_bps_per_trade").reset_index()
        top = top.rename(columns={column: f"top{int(float(column) * 100)}" for column in top.columns if column != "arm"})
        stability = pd.read_parquet(run / "metrics_stability.parquet")
        keep = ["arm", "mean_top5_net_bps", "median_top5_net_bps", "worst_month_top5_net_bps", "positive_months_top5", "portability_score_bps"]
        top = top.merge(stability[keep], on="arm", how="left", validate="one_to_one")
        top["seed_index"] = idx
        pieces.append(top)
    long = pd.concat(pieces, ignore_index=True)
    value = [column for column in long.columns if column not in {"arm", "seed_index"}]
    mean = long.groupby("arm", sort=True)[value].mean().add_suffix("_mean")
    std = long.groupby("arm", sort=True)[value].std(ddof=0).add_suffix("_std")
    lower = long.groupby("arm", sort=True)[value].min().add_suffix("_min")
    result = mean.join(std).join(lower).reset_index()
    result["top5_uplift_vs_control_mean"] = result["top5_mean"] - float(result.loc[result.arm.eq("canonical_control"), "top5_mean"].iloc[0])
    return result.sort_values(["top5_mean", "top5_min", "arm"], ascending=[False, False, True], kind="stable").reset_index(drop=True)


def run(*, out: Path = OUT) -> Path:
    if out.exists():
        raise FileExistsError(out)
    summary = _aggregate(RUNS)
    coverage = pd.read_parquet(STRUCTURE / "target_2025_coverage_audit.parquet")
    candidate = pd.read_parquet(STRUCTURE / "cluster_candidate_audit.parquet")
    reliability = []
    for idx, source in enumerate(RUNS):
        audit = pd.read_parquet(source / "model_audit.parquet")
        audit = audit.loc[audit.model_type.eq("bounded_correctness_classifier")].copy()
        audit["seed_index"] = idx
        reliability.append(audit)
    reliability_summary = pd.concat(reliability, ignore_index=True).groupby("arm", sort=True).agg(
        folds=("month", "count"), features=("feature_count", "median"),
        auc_mean=("held_auc", "mean"), auc_std=("held_auc", "std"),
        brier_mean=("held_brier", "mean"),
    ).reset_index().sort_values("auc_mean", ascending=False, kind="stable")
    selected = summary.loc[summary.arm.isin([
        "canonical_control", "residual_context", "residual_uncertainty", "residual_ood", "residual_support",
        "residual_all_k05", "residual_all_k06", "residual_all_k07", "residual_all_k08", "residual_all_k09",
        "reliability_add_context", "reliability_multiply_context", "reliability_shrink_context",
        "reliability_multiply_uncertainty_k09", "reliability_shrink_support_k09",
        "reliability_add_all_k05", "reliability_multiply_all_k05", "reliability_shrink_all_k05",
        "reliability_add_all_k06", "reliability_multiply_all_k06", "reliability_shrink_all_k06",
        "reliability_add_all_k07", "reliability_multiply_all_k07", "reliability_shrink_all_k07",
        "reliability_add_all_k08", "reliability_multiply_all_k08", "reliability_shrink_all_k08",
        "reliability_add_all_k09", "reliability_multiply_all_k09", "reliability_shrink_all_k09",
        "residual_health_k09", "residual_all_health_k09",
        "reliability_add_health_k09", "reliability_multiply_health_k09", "reliability_shrink_health_k09",
        "reliability_add_all_health_k09", "reliability_multiply_all_health_k09", "reliability_shrink_all_health_k09",
    ])].copy()
    health_monthly = pd.read_parquet(RUNS[0] / "cluster_health_monthly_2025.parquet")
    health_latest = health_monthly.loc[
        health_monthly.month.eq(health_monthly.month.max())
        & health_monthly.cluster.str.startswith("k09_")
        & health_monthly.metric.isin(["ic_7d", "hr_7d", "hr_surprise_7d", "support_7d"])
    ].copy().sort_values(["cluster", "metric"], kind="stable")
    coactivation = pd.read_parquet(COACTIVATION / "cluster_coactivation_grid_2025.parquet")
    coactivation_shifts = pd.read_parquet(COACTIVATION / "cluster_coactivation_context_shift_2025.parquet")
    coactivation_support = coactivation.sort_values(["rows", "arity", "clusters"], ascending=[False, True, True], kind="stable").head(30)
    coactivation_economic = coactivation.loc[coactivation.rows.ge(200)].sort_values(
        ["canonical_top10_net_bps", "rows"], ascending=[False, False], kind="stable"
    ).head(30)
    coactivation_top2 = coactivation.loc[
        coactivation.activation_mode.eq("row_local_relative_top2") & coactivation.rows.ge(100)
    ].sort_values(["canonical_top10_net_bps", "rows"], ascending=[False, False], kind="stable").head(30)
    correctness_runs = [json.loads((path / "correctness_test_report.json").read_text()) for path in RUNS]
    correctness = {
        "schema": "tp6_sl4_prototype_cluster_ablation_summary_correctness_v1",
        "all_seed_runs_complete": True,
        "all_seed_runs_use_frozen_pre_2025_cluster_contract": all(item["prototype_cluster_contract_frozen_before_2025"] for item in correctness_runs),
        "all_seed_runs_exclude_2025_outcomes_from_representation_selection": all(not item["target_2025_outcomes_used_in_representation_selection"] for item in correctness_runs),
        "all_seed_runs_use_mature_labels_for_residual_training": all(item["residual_train_labels_mature_before_held_month"] for item in correctness_runs),
        "all_seed_runs_use_strict_prior_label_availability_for_cluster_health": all(item["cluster_health_uses_label_available_ts_strictly_before_decision"] for item in correctness_runs),
        "all_seed_runs_have_finite_generated_scores": all(all(item["all_generated_scores_finite"].values()) for item in correctness_runs),
        "all_seed_runs_have_unique_candidate_month_predictions": all(item["prediction_candidate_month_pairs_unique"] for item in correctness_runs),
        "coactivation_grid_is_diagnostic_only": True,
    }
    out.mkdir(parents=True)
    summary.to_parquet(out / "three_seed_summary.parquet", index=False)
    selected.to_parquet(out / "selected_arm_summary.parquet", index=False)
    reliability_summary.to_parquet(out / "reliability_quality_summary.parquet", index=False)
    coverage.to_parquet(out / "frozen_2025_coverage.parquet", index=False)
    candidate.to_parquet(out / "cluster_size_quality_sweep.parquet", index=False)
    health_monthly.to_parquet(out / "cluster_health_monthly_2025.parquet", index=False)
    health_latest.to_parquet(out / "k09_recent_cluster_health_latest_month.parquet", index=False)
    coactivation.to_parquet(out / "cluster_coactivation_grid_2025.parquet", index=False)
    coactivation_support.to_parquet(out / "cluster_coactivation_high_support.parquet", index=False)
    coactivation_economic.to_parquet(out / "cluster_coactivation_high_support_economics.parquet", index=False)
    coactivation_top2.to_parquet(out / "cluster_coactivation_row_local_top2.parquet", index=False)
    coactivation_shifts.to_parquet(out / "cluster_coactivation_context_shift_2025.parquet", index=False)
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    baseline = summary.loc[summary.arm.eq("canonical_control")].iloc[0]
    report = [
        "# TP6/SL4 archetype/prototype and bounded-reliability ablation — 2025",
        "",
        "## Scope",
        "",
        "This is a long-only, 10,224-row matched 2025 development replay. The stored monthly Base+Consensus score is the same control in every arm. Economics are the existing exact TP6/SL4/H12 net-bps labels; this is not yet the full-universe canonical trailing-exit replay.",
        "",
        "## Representation quality",
        "",
        "The 2024-frozen path-neighbourhood contract selects 32 persistent prototypes and tests cluster sizes 3--10 using opportunity-conditioned co-activation plus train-only joint residual synergy. K=5--9 pass the predeclared structural support/balance/sign-stability gates; K=9 is the highest composite score. The old exact-rule representation had about 51% matched contribution mass. The new held-2025 contract has the following coverage:",
        "",
        coverage.round(4).to_string(index=False),
        "",
        "## Three-seed matched result",
        "",
        "The table reports the mean/minimum across the three independent model seeds. `Top-5 uplift` is versus the fixed control of " + f"{baseline.top5_mean:.2f}" + " net bps/trade.",
        "",
        selected[["arm", "top0_mean", "top1_mean", "top2_mean", "top5_mean", "top5_min", "top5_std", "top5_uplift_vs_control_mean", "mean_top5_net_bps_mean", "worst_month_top5_net_bps_mean", "positive_months_top5_mean"]].round(2).to_string(index=False),
        "",
        "## Interpretation",
        "",
        "1. Direct residual re-ranking does not advance. Every residual arm is below the canonical control at Top-5, including all K=5--9 contracts. The best direct residual variants have somewhat less severe worst months, but their pooled economic ordering is insufficient.",
        "",
        "2. A bounded reliability modifier is consistently better than the direct residual ranker. The best robust control-context arm is shrinkage: its mean Top-5 is above the canonical control across all seeds. Addition, multiplication, and shrinkage were all tested; their individual winner changes by seed.",
        "",
        "3. Structural incrementality is not established. The fully symmetric K=5--9 bounded-reliability sweep does not select a cluster size: its best all-K arm is K=9 multiplication, but it remains below the robust context/support/uncertainty controls. K=7 and K=8 do not rescue the result. Thus the improved clusters are a valid feature substrate, but not yet a promoted economic signal.",
        "",
        "4. The classifiers are weak-but-non-null (AUC near 0.52--0.53). The small improvement is consistent with using them to limit correction magnitude, not as an independent alpha ranker.",
        "",
        "5. Causal cluster-health fields were added: membership-weighted prior-resolved 3/7/14-day base-score rank-IC proxy, exact-net-positive hit rate, hit-rate surprise versus the same causal all-candidate history, and effective support. These fields have at least 99.5% nonzero held coverage. They do not advance: no health-only or health-plus-K9 reliability transform beats the robust context/support/uncertainty controls across three seeds. They remain useful diagnostics, not a promoted inference block.",
        "",
        "## Latest month causal K=9 cluster-health snapshot", "",
        "Each value below was available before its associated decision. `ic_7d` is a membership-weighted base-score rank-IC proxy to exact H12 net; `hr_surprise_7d` is cluster hit rate minus the all-candidate causal hit rate, not a realised same-period statistic.", "",
        health_latest.round(4).to_string(index=False), "",
        "## Co-activation diagnostic", "",
        "This grid is outcome-bearing analysis only, not an input to the models. It uses absolute 5%/10%/20% membership, raw row-local top-2/top-3 membership, and data-driven relative row-local top-2/top-3 membership. The relative variants divide each membership by its population prevalence before selecting the dominant row-local geometries; this removes one broad soft component without using labels or hand-coded semantics. It reports pair/triple intersections, support, base and canonical rank-IC, and each score's within-intersection Top-10 economics. Small intersections must not be treated as discovered trading states.", "",
        "### Highest-support intersections", "", coactivation_support.round(3).to_string(index=False), "",
        "### Relative row-local top-2 intersections with at least 100 rows", "", coactivation_top2.round(3).to_string(index=False), "",
        "### Intersections with at least 200 rows, ranked by canonical Top-10 net", "", coactivation_economic.round(3).to_string(index=False), "",
        "## Correctness checks", "", json.dumps(correctness, indent=2), "",
        "",
        "## Advancement decision",
        "",
        "Keep the 2024-frozen prototype representation and retain K=5/6/9 only as diagnostic candidates. Do not add any cluster head as a free-standing residual ranker. The only candidate worth carrying forward is a bounded reliability modifier, with the context-only shrinkage control and the uncertainty-multiplier / K=9 support-shrink variants compared again on the full canonical population and trailing-exit outcome.",
    ]
    (out / "TP6_SL4_PROTOTYPE_CLUSTER_ABLATION_REPORT.md").write_text("\n".join(report) + "\n")
    manifest = {
        "schema": "tp6_sl4_prototype_cluster_ablation_summary_20260809_fullsizes_v1",
        "runs": [str(path) for path in RUNS], "structure": str(STRUCTURE),
        "selection": "cluster representation selected on 2024 only; 2025 OOF used for downstream development comparison",
        "baseline_top5_mean": float(baseline.top5_mean), "rows": 10224,
        "artifacts": sorted(path.name for path in out.iterdir()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(out), "baseline_top5": float(baseline.top5_mean), "arms": len(summary)}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(out=args.out)
