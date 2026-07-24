#!/usr/bin/env python3
"""Summarize OOS failure-mode detection quality without averaging weak arms away."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _safe_average_precision(target: np.ndarray, score: np.ndarray) -> float:
    return (
        float(average_precision_score(target, score))
        if len(target) and np.unique(target).size > 1
        else np.nan
    )


def _quality(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame["target"].astype(bool).to_numpy()
    score = pd.to_numeric(frame["risk"], errors="coerce").to_numpy(np.float64)
    alert = frame["alert"].astype(bool).to_numpy()
    finite = np.isfinite(score)
    target, score, alert = target[finite], score[finite], alert[finite]
    positives = int(target.sum())
    alerts = int(alert.sum())
    prevalence = float(target.mean()) if len(target) else np.nan
    precision = float(target[alert].mean()) if alerts else np.nan
    recall = float((target & alert).sum() / max(positives, 1))
    realized_severity = pd.to_numeric(
        frame.loc[finite, "target_failure_severity"], errors="coerce"
    ).fillna(0.0)
    expected_severity = pd.to_numeric(
        frame.loc[finite, "expected_failure_severity"], errors="coerce"
    ).fillna(0.0)
    return {
        "oos_rows": int(len(target)),
        "positive_days": positives,
        "alert_days": alerts,
        "alert_positive_days": int((target & alert).sum()),
        "prevalence": prevalence,
        "precision": precision,
        "recall": recall,
        "lift": precision / max(prevalence, 1e-12)
        if np.isfinite(precision) and np.isfinite(prevalence)
        else np.nan,
        "average_precision": _safe_average_precision(target.astype(np.int8), score),
        "brier": float(np.mean((score - target.astype(np.float64)) ** 2)),
        "severity_mae": float(np.mean(np.abs(expected_severity - realized_severity))),
        "mean_expected_severity": float(expected_severity.mean()),
        "mean_realized_severity": float(realized_severity.mean()),
    }


def _group_quality(frame: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(list(keys), observed=True, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        rows.append({**dict(zip(keys, values)), **_quality(group)})
    return pd.DataFrame(rows)


def _classify(row: pd.Series) -> str:
    adequate_support = (
        int(row["folds"]) >= 4
        and int(row["positive_days"]) >= 20
        and int(row["alert_days"]) >= 10
        and int(row["months"]) >= 3
    )
    if float(row["prevalence"]) >= 0.80:
        return "diagnostic_only_high_prevalence"
    if not adequate_support:
        return "diagnostic_only_sparse"
    separation = (
        float(row["lift"]) >= 1.50
        and float(row["average_precision"])
        >= float(row["prevalence"]) + 0.05
    )
    stability = (
        float(row["positive_lift_fold_fraction"]) >= 0.60
        and float(row["positive_lift_month_fraction"]) >= 0.60
    )
    if separation and stability:
        return "promotable_research_signal"
    if separation:
        return "promising_but_unstable"
    if float(row["lift"]) <= 1.10:
        return "diagnostic_only_unpredictable"
    return "diagnostic_only_weak_separation"


def _scope_report(root: Path, output: Path, scope: str) -> dict[str, Any]:
    predictions = pd.read_parquet(root / f"{scope}_oos_predictions.parquet")
    metrics = pd.read_csv(root / f"{scope}_oos_metrics.csv")
    predictions["day"] = pd.to_datetime(predictions["day"], utc=True)
    predictions["month"] = predictions["day"].dt.to_period("M").astype(str)

    target = _group_quality(predictions, ["failure_mode"])
    local = _group_quality(
        predictions, ["side_name", "archetype_policy_key", "failure_mode"]
    )
    monthly = _group_quality(
        predictions,
        ["month", "side_name", "archetype_policy_key", "failure_mode"],
    )

    fold = metrics.loc[
        :,
        [
            "side_name",
            "archetype_policy_key",
            "failure_mode",
            "fold_index",
            "lift",
        ],
    ].copy()
    fold_summary = (
        fold.groupby(
            ["side_name", "archetype_policy_key", "failure_mode"],
            observed=True,
            as_index=False,
        )
        .agg(
            folds=("fold_index", "nunique"),
            worst_fold_lift=("lift", "min"),
            median_fold_lift=("lift", "median"),
            positive_lift_fold_fraction=("lift", lambda value: float(value.gt(1.0).mean())),
        )
    )
    month_summary = (
        monthly.groupby(
            ["side_name", "archetype_policy_key", "failure_mode"],
            observed=True,
            as_index=False,
        )
        .agg(
            months=("month", "nunique"),
            worst_month_lift=("lift", "min"),
            median_month_lift=("lift", "median"),
            positive_lift_month_fraction=("lift", lambda value: float(value.gt(1.0).mean())),
        )
    )
    quality = local.merge(
        fold_summary,
        on=["side_name", "archetype_policy_key", "failure_mode"],
        how="left",
        validate="one_to_one",
    ).merge(
        month_summary,
        on=["side_name", "archetype_policy_key", "failure_mode"],
        how="left",
        validate="one_to_one",
    )
    quality["status"] = quality.apply(_classify, axis=1)
    quality = quality.sort_values(
        ["status", "lift", "positive_days"], ascending=[True, False, False]
    )

    target.to_csv(output / f"{scope}_target_quality.csv", index=False)
    local.to_csv(output / f"{scope}_side_archetype_mode_quality.csv", index=False)
    monthly.to_csv(output / f"{scope}_monthly_mode_quality.csv", index=False)
    quality.to_csv(output / f"{scope}_promotion_status.csv", index=False)
    status_counts = quality["status"].value_counts().to_dict()
    return {
        "prediction_rows": int(len(predictions)),
        "evaluated_arms": int(len(quality)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "promotable_research_signals": int(
            quality["status"].eq("promotable_research_signal").sum()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "prospective_failure_mode_quality_v1",
        "input": str(root.resolve()),
        "promotion_contract": (
            "Research-only: >=4 folds, >=20 positive days, >=10 alerts, >=3 months, "
            "lift>=1.5, AP>=prevalence+0.05, and lift>1 in >=60% of folds and months. "
            "No result is a live hard-gate approval."
        ),
        "local": _scope_report(root, output, "local"),
        "parent": _scope_report(root, output, "parent"),
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
