#!/usr/bin/env python3
"""Recompute Stage-3/4 economics with exact global top-k selection.

The training runner also emits side-local metrics for head diagnosis.  Those
must not be averaged and presented as book economics.  This read-only
postprocessor freezes the held-out predictions and ranks them globally, with a
deterministic candidate-ID tie break, exactly as required by the evaluation
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
DEFAULT_STAGE3 = ART / "root_cause_base_residual_learning_20260731_v1"
DEFAULT_OUTPUT = ART / "root_cause_global_learning_economics_20260731_v1"
FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_global_topk(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    count = max(1, int(np.ceil(len(frame) * fraction)))
    ranked = frame.assign(__tie__=frame.candidate_id.astype(str)).sort_values(
        ["combined_economic_prediction_bps", "__tie__"],
        ascending=[False, True], kind="stable",
    )
    return ranked.head(count).drop(columns="__tie__")


def compute(predictions: pd.DataFrame, *, bootstrap_reps: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = {
        "candidate_id", "__ts__", "side_name", "gross_h12_bps", "net_h12_bps",
        "combined_economic_prediction_bps", "model_family", "seed", "split", "evaluation_scope",
    }
    missing = needed - set(predictions)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    heldout = predictions.loc[
        predictions.evaluation_scope.eq("outer_heldout") & predictions.split.eq("later_oos")
    ].copy()
    if heldout.empty:
        raise ValueError("no later outer-heldout predictions")
    heldout["__ts__"] = pd.to_datetime(heldout["__ts__"], utc=True)
    if heldout.duplicated(["model_family", "seed", "candidate_id"]).any():
        raise ValueError("duplicate candidate within a later heldout model arm")

    rows: list[dict] = []
    top10: dict[tuple[str, int], pd.DataFrame] = {}
    for (family, seed), arm in heldout.groupby(["model_family", "seed"], observed=True):
        for fraction in FRACTIONS:
            selected = exact_global_topk(arm, fraction)
            month = selected.assign(month=selected["__ts__"].dt.to_period("M").astype(str)).groupby("month")[["gross_h12_bps", "net_h12_bps"]].mean()
            side = selected.groupby("side_name")[["gross_h12_bps", "net_h12_bps"]].mean()
            rows.append({
                "record_type": "global_topk_arm",
                "model_family": family,
                "seed": int(seed),
                "top_fraction": fraction,
                "population_rows": len(arm),
                "selected_rows": len(selected),
                "gross_bps": float(selected.gross_h12_bps.mean()),
                "net_bps": float(selected.net_h12_bps.mean()),
                "worst_month_gross_bps": float(month.gross_h12_bps.min()),
                "worst_month_net_bps": float(month.net_h12_bps.min()),
                "worst_side_gross_bps": float(side.gross_h12_bps.min()),
                "worst_side_net_bps": float(side.net_h12_bps.min()),
                "selection_scope": "GLOBAL_TOP_K_NOT_PER_TIMESTAMP_OR_SIDE",
            })
            if fraction == 0.10:
                top10[(str(family), int(seed))] = selected
    arm_frame = pd.DataFrame(rows)

    comparisons: list[dict] = []
    means = arm_frame.loc[arm_frame.top_fraction.eq(.10)].groupby("model_family")[["gross_bps", "net_bps"]].mean()
    for name, left, right in (
        ("null_to_causal", "prior", "causal_capacity_oracle"),
        ("production_to_causal", "production_like_lgbm", "causal_capacity_oracle"),
        ("causal_to_future", "causal_capacity_oracle", "future_feature_oracle"),
    ):
        if left in means.index and right in means.index:
            comparisons.append({
                "record_type": "named_global_gap", "comparison": name,
                "left_model": left, "right_model": right,
                "left_gross_bps": float(means.loc[left, "gross_bps"]),
                "right_gross_bps": float(means.loc[right, "gross_bps"]),
                "gross_gap_bps": float(means.loc[right, "gross_bps"] - means.loc[left, "gross_bps"]),
                "left_net_bps": float(means.loc[left, "net_bps"]),
                "right_net_bps": float(means.loc[right, "net_bps"]),
                "net_gap_bps": float(means.loc[right, "net_bps"] - means.loc[left, "net_bps"]),
                "selection_scope": "GLOBAL_TOP_10",
            })

    prior_keys = sorted(key for key in top10 if key[0] == "prior")
    if prior_keys:
        baseline = top10[prior_keys[0]]
        baseline_day = baseline.assign(day=baseline["__ts__"].dt.floor("D")).groupby("day").net_h12_bps.mean()
        for (family, seed), selected in sorted(top10.items()):
            if family == "prior":
                continue
            arm_day = selected.assign(day=selected["__ts__"].dt.floor("D")).groupby("day").net_h12_bps.mean()
            days = baseline_day.index.union(arm_day.index)
            delta = arm_day.reindex(days, fill_value=0.0).to_numpy() - baseline_day.reindex(days, fill_value=0.0).to_numpy()
            rng = np.random.default_rng(20260731 + int(seed) % 10_000)
            draws = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(bootstrap_reps)])
            comparisons.append({
                "record_type": "paired_utc_day_bootstrap_vs_prior", "comparison": f"{family}_vs_prior",
                "model_family": family, "seed": int(seed), "paired_days": len(days),
                "mean_delta_net_bps": float(delta.mean()),
                "ci_low_bps": float(np.quantile(draws, .025)),
                "ci_high_bps": float(np.quantile(draws, .975)),
                "probability_positive": float((draws > 0).mean()),
                "selection_scope": "GLOBAL_TOP_10",
            })
    return arm_frame, pd.DataFrame(comparisons)


def causal_metric_concordance(metrics: pd.DataFrame, arms: pd.DataFrame) -> pd.DataFrame:
    """Associate development base metrics with later *global* economics.

    The hindsight M7 arm is reported separately by the ladder but excluded
    here: including it would mechanically inflate metric concordance for
    deployable model selection.
    """
    dev = metrics.loc[
        metrics.split.eq("development_oof")
        & metrics.evaluation_scope.eq("outer_heldout")
        & metrics.component.eq("base_directional")
        & ~metrics.model_family.eq("future_feature_oracle")
        & ~metrics.model_family.eq("frozen_oof_reference")
    ].copy()
    candidate_metrics = [
        "base_directional__roc_auc", "base_directional__pr_auc",
        "base_directional__log_loss", "base_directional__brier",
        "base_directional__ece", "base_directional__spearman_ic",
        "base_directional__mae", "base_directional__calibration_slope",
    ]
    available = [name for name in candidate_metrics if name in dev]
    left = dev.groupby(["model_family", "seed"], as_index=False)[available].mean()
    right = arms.loc[arms.top_fraction.eq(.10) & ~arms.model_family.eq("future_feature_oracle")].copy()
    outcomes = [
        "gross_bps", "net_bps", "worst_month_gross_bps", "worst_month_net_bps",
        "worst_side_gross_bps", "worst_side_net_bps",
    ]
    joined = left.merge(right[["model_family", "seed", *outcomes]], on=["model_family", "seed"], validate="one_to_one")
    rows = []
    for metric in available:
        for outcome in outcomes:
            valid = joined[metric].notna() & joined[outcome].notna()
            rows.append({
                "record_type": "causal_only_metric_concordance",
                "development_base_metric": metric,
                "later_global_economic_metric": outcome,
                "arms": int(valid.sum()),
                "pearson": float(joined.loc[valid, metric].corr(joined.loc[valid, outcome], method="pearson")) if valid.sum() > 2 else np.nan,
                "spearman": float(joined.loc[valid, metric].corr(joined.loc[valid, outcome], method="spearman")) if valid.sum() > 2 else np.nan,
                "excluded_noncausal_families": "future_feature_oracle,frozen_oof_reference",
                "selection_scope": "GLOBAL_TOP_10",
            })
    return pd.DataFrame(rows)


def run(stage3: Path = DEFAULT_STAGE3, output: Path = DEFAULT_OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((stage3 / "run_manifest.json").read_text())
    predictions_path = stage3 / "base_residual_oof_predictions.parquet"
    metrics_path = stage3 / "model_learning_efficiency.parquet"
    declared = manifest.get("outputs_sha256", {}).get(predictions_path.name)
    if declared != sha256(predictions_path):
        raise ValueError("Stage3 prediction digest mismatch")
    predictions = pd.read_parquet(predictions_path)
    arms, gaps = compute(predictions)
    metrics = pd.read_parquet(metrics_path)
    concordance = causal_metric_concordance(metrics, arms)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arms.to_parquet(stage / "global_topk_learning_economics.parquet", index=False)
        gaps.to_parquet(stage / "global_topk_learning_gaps.parquet", index=False)
        concordance.to_parquet(stage / "causal_only_global_metric_concordance.parquet", index=False)
        result = {
            "schema": "root_cause_global_learning_economics_v1",
            "status": "COMPLETE_DIAGNOSTIC_ONLY",
            "selection_scope": "GLOBAL_TOP_K_NOT_PER_TIMESTAMP_OR_SIDE",
            "stage3_manifest_sha256": sha256(stage3 / "run_manifest.json"),
            "prediction_sha256": declared,
            "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__))},
            "outputs_sha256": {p.name: sha256(p) for p in stage.iterdir()},
        }
        (stage / "run_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output)
        return result
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.stage3, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
