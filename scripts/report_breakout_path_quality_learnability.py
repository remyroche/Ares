#!/usr/bin/env python3
"""Consolidate chronological breakout-label learnability experiment shards.

This report is intentionally descriptive. It never selects a trading policy or
modifies any base/meta artifact; it only decides whether a label has stable
pre-entry learnability under its frozen feature contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_QUARTERS = ("2025q3", "2025q4", "2026q1", "2026q2")
DEFAULT_PREFIXES = {
    "all_observable": "breakout_path_quality_learnability_20260713_observable_",
    "exclude_lagged_path_state": "breakout_path_quality_learnability_20260713_no_lagpath_",
}


def _read_variant(
    reports_root: Path,
    variant: str,
    prefix: str,
    quarters: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for quarter in quarters:
        shard = reports_root / f"{prefix}{quarter}"
        metric_path = shard / "chronological_learnability_metrics.csv"
        importance_path = shard / "feature_importance_by_fold.csv"
        prediction_path = shard / "oof_predictions.parquet"
        if not metric_path.exists():
            raise FileNotFoundError(f"Missing learnability shard: {metric_path}")
        metric = pd.read_csv(metric_path)
        metric["feature_variant"] = variant
        metric["quarter"] = quarter
        metrics.append(metric)
        if importance_path.exists():
            importance = pd.read_csv(importance_path)
            importance["feature_variant"] = variant
            importance["quarter"] = quarter
            importances.append(importance)
        if prediction_path.exists():
            prediction = pd.read_parquet(prediction_path)
            prediction["feature_variant"] = variant
            prediction["quarter"] = quarter
            prediction["__ts__"] = pd.to_datetime(prediction["__ts__"], utc=True, errors="coerce")
            predictions.append(prediction.dropna(subset=["__ts__"]))
    return (
        pd.concat(metrics, ignore_index=True, copy=False),
        pd.concat(importances, ignore_index=True, copy=False) if importances else pd.DataFrame(),
        pd.concat(predictions, ignore_index=True, copy=False) if predictions else pd.DataFrame(),
    )


def _calibration_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Use equal-count probability bins independently within each OOS fold."""

    if predictions.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    keys = ["feature_variant", "quarter", "fold_start", "target", "model"]
    for key, local in predictions.groupby(keys, observed=True, sort=False):
        local = local.copy()
        local["calibration_decile"] = pd.qcut(
            local["prediction"].rank(method="first"), q=10, labels=False
        ).astype("int8") + 1
        part = local.groupby("calibration_decile", observed=True).agg(
            rows=("target_realized", "size"),
            mean_prediction=("prediction", "mean"),
            observed_rate=("target_realized", "mean"),
        ).reset_index()
        for name, value in zip(keys, key):
            part[name] = value
        part["calibration_gap"] = part["mean_prediction"] - part["observed_rate"]
        rows.append(part)
    return pd.concat(rows, ignore_index=True, copy=False)


def _event_day_rows(predictions: pd.DataFrame, event_dates: tuple[pd.Timestamp, ...]) -> pd.DataFrame:
    """Place specified dates in their OOS-fold distribution without policy thresholds."""

    if predictions.empty:
        return pd.DataFrame()
    keys = ["feature_variant", "quarter", "fold_start", "target", "model"]
    local = predictions.copy()
    local["date"] = local["__ts__"].dt.normalize()
    daily = local.groupby([*keys, "date"], observed=True).agg(
        rows=("target_realized", "size"),
        mean_probability=("prediction", "mean"),
        p90_probability=("prediction", lambda value: value.quantile(0.90)),
        observed_rate=("target_realized", "mean"),
    ).reset_index()
    daily["mean_probability_percentile_within_oos_quarter"] = daily.groupby(
        keys, observed=True
    )["mean_probability"].rank(pct=True, method="average")
    events = daily.loc[daily["date"].isin(event_dates)].copy()
    return events.sort_values(["date", "feature_variant", "target", "model"], kind="stable")


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    quarters = tuple(item.strip() for item in args.quarters.split(",") if item.strip())
    all_metrics: list[pd.DataFrame] = []
    all_importances: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    for variant, prefix in DEFAULT_PREFIXES.items():
        metrics, importances, predictions = _read_variant(args.reports_root, variant, prefix, quarters)
        all_metrics.append(metrics)
        if not importances.empty:
            all_importances.append(importances)
        if not predictions.empty:
            all_predictions.append(predictions)
    metrics = pd.concat(all_metrics, ignore_index=True, copy=False)
    importances = pd.concat(all_importances, ignore_index=True, copy=False)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)

    summary = metrics.groupby(["feature_variant", "target", "model"], observed=True).agg(
        folds=("quarter", "nunique"),
        mean_auc=("roc_auc", "mean"), min_auc=("roc_auc", "min"),
        mean_ap=("average_precision", "mean"),
        mean_top10_lift=("top10_lift", "mean"), min_top10_lift=("top10_lift", "min"),
        mean_brier=("brier", "mean"), mean_ece10=("ece10", "mean"),
    ).reset_index()
    summary["chronologically_learnable"] = (
        summary["folds"].eq(len(quarters))
        & summary["mean_auc"].gt(0.55)
        & summary["min_auc"].gt(0.50)
        & summary["mean_top10_lift"].gt(1.25)
        & summary["min_top10_lift"].gt(1.0)
    )
    summary.to_csv(args.output / "model_summary.csv", index=False)

    value_columns = ["mean_auc", "min_auc", "mean_ap", "mean_top10_lift", "min_top10_lift", "mean_brier", "mean_ece10"]
    baseline = summary.loc[summary["feature_variant"].eq("all_observable"), ["target", "model", *value_columns]]
    challenger = summary.loc[summary["feature_variant"].eq("exclude_lagged_path_state"), ["target", "model", *value_columns]]
    delta = challenger.merge(baseline, on=["target", "model"], suffixes=("_no_lagged_path", "_all_observable"))
    for column in value_columns:
        delta[f"delta_{column}_no_lagged_minus_all"] = (
            delta[f"{column}_no_lagged_path"] - delta[f"{column}_all_observable"]
        )
    delta.to_csv(args.output / "lagged_path_state_ablation_delta.csv", index=False)

    predictions = pd.concat(all_predictions, ignore_index=True, copy=False) if all_predictions else pd.DataFrame()
    calibration = _calibration_rows(predictions)
    calibration.to_csv(args.output / "calibration_by_fold_decile.csv", index=False)
    event_dates = tuple(pd.Timestamp(value, tz="UTC") for value in args.event_dates.split(",") if value)
    event_days = _event_day_rows(predictions, event_dates)
    event_days.to_csv(args.output / "event_day_path_quality_scores.csv", index=False)

    if not importances.empty:
        importance_summary = importances.groupby(
            ["feature_variant", "target", "model", "feature"], observed=True
        ).agg(mean_importance=("importance", "mean"), folds=("quarter", "nunique")).reset_index()
        importance_summary.sort_values(
            ["feature_variant", "target", "model", "mean_importance"],
            ascending=[True, True, True, False],
            inplace=True,
        )
        importance_summary.to_csv(args.output / "feature_importance_summary.csv", index=False)

    manifest = {
        "schema": "breakout_path_quality_learnability_report_v1",
        "status": "diagnostic_only_no_policy_effect",
        "quarters": list(quarters),
        "variants": list(DEFAULT_PREFIXES),
        "event_dates": [str(value.date()) for value in event_dates],
        "decision_contract": {
            "minimum_folds": len(quarters),
            "mean_auc_gt": 0.55,
            "min_auc_gt": 0.50,
            "mean_top10_lift_gt": 1.25,
            "min_top10_lift_gt": 1.0,
        },
        "leakage_contract": (
            "Every input shard used train-only thresholds, train-only preprocessing, and an "
            "8-hour label-horizon purge before its scored OOS quarter. The no-lagged-path "
            "variant removes prior-bar path-state indicators but retains all other observable features."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "metric_rows": int(len(metrics)), "summary_rows": int(len(summary))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("data_perp/reports"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarters", default=",".join(DEFAULT_QUARTERS))
    parser.add_argument("--event-dates", default="2026-06-17")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
