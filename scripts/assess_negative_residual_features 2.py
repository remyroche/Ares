#!/usr/bin/env python3
"""Assess causal negative-residual composites against calendar episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
)


def _metrics(score: pd.Series, event: pd.Series, threshold: float) -> dict[str, float]:
    valid = score.notna() & event.notna()
    x = score.loc[valid].to_numpy(dtype=np.float64)
    y = event.loc[valid].astype(bool).to_numpy()
    selected = x >= threshold
    prevalence = float(y.mean()) if len(y) else np.nan
    precision = float(y[selected].mean()) if selected.any() else 0.0
    corr = float(np.corrcoef(x, y.astype(float))[0, 1]) if len(x) and np.std(x) and np.std(y) else np.nan
    return {
        "correlation": corr,
        "precision": precision,
        "lift": precision / max(prevalence, 1e-9),
        "recall": float((selected & y).sum() / max(y.sum(), 1)),
        "false_positive_rate": float((selected & ~y).sum() / max((~y).sum(), 1)),
        "selected_days": int(selected.sum()),
        "event_days": int(y.sum()),
        "recognized_event_days": int((selected & y).sum()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.feature_file, columns=NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.loc[
        (frame.index >= pd.Timestamp(args.start, tz="UTC"))
        & (frame.index < pd.Timestamp(args.end, tz="UTC"))
    ]
    daily_parts = []
    composite_set = set(NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS)
    for feature in NEGATIVE_RESIDUAL_META_FEATURE_KEYS:
        grouped = frame[feature].groupby(frame.index.floor("D"))
        series = grouped.max() if feature in composite_set else grouped.mean()
        daily_parts.append(series.rename(feature))
    daily = pd.concat(daily_parts, axis=1)

    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    calendar = calendar.loc[calendar["adverse_event_rows"].gt(0)]
    train_mask = daily.index.year == 2025
    oos_mask = daily.index.year == 2026
    rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    for (side, archetype), local_events in calendar.groupby(
        ["side_name", "archetype_policy_key"], observed=True
    ):
        event_days = set(local_events["day"])
        event = pd.Series(daily.index.isin(event_days), index=daily.index)
        for feature in NEGATIVE_RESIDUAL_META_FEATURE_KEYS:
            raw = daily[feature]
            train_corr = raw.loc[train_mask].corr(event.loc[train_mask].astype(float))
            direction = 1.0 if not np.isfinite(train_corr) or train_corr >= 0 else -1.0
            score = direction * raw
            threshold = float(score.loc[train_mask].quantile(0.90))
            train_metrics = _metrics(score.loc[train_mask], event.loc[train_mask], threshold)
            oos_metrics = _metrics(score.loc[oos_mask], event.loc[oos_mask], threshold)
            rows.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "feature": feature,
                    "direction": direction,
                    "train_threshold_q90": threshold,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"oos_{key}": value for key, value in oos_metrics.items()},
                }
            )
            ignored = local_events.loc[local_events.get("status", "").eq("ignored")] if "status" in local_events else local_events.iloc[0:0]
            for day in ignored["day"]:
                if day not in score.index:
                    continue
                history = score.loc[score.index < day]
                causal_threshold = float(history.quantile(0.90)) if len(history) >= 90 else np.nan
                gap_rows.append(
                    {
                        "day": day,
                        "side_name": side,
                        "archetype_policy_key": archetype,
                        "feature": feature,
                        "score": float(score.loc[day]),
                        "causal_q90": causal_threshold,
                        "incrementally_recognized": bool(
                            np.isfinite(causal_threshold) and score.loc[day] >= causal_threshold
                        ),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics["promotion_score"] = (
        metrics["oos_correlation"].fillna(0.0)
        + 0.25 * metrics["oos_recall"].fillna(0.0)
        + 0.10 * np.log1p(metrics["oos_lift"].clip(lower=0.0).fillna(0.0))
        - 0.50 * metrics["oos_false_positive_rate"].fillna(1.0)
    )
    metrics.to_csv(args.output / "side_archetype_feature_metrics.csv", index=False)
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(args.output / "ignored_cell_incremental_recognition.csv", index=False)
    summary = (
        metrics.groupby("feature", observed=True)
        .agg(
            archetypes=("archetype_policy_key", "nunique"),
            mean_oos_correlation=("oos_correlation", "mean"),
            worst_oos_correlation=("oos_correlation", "min"),
            mean_oos_lift=("oos_lift", "mean"),
            mean_oos_fpr=("oos_false_positive_rate", "mean"),
            total_oos_recognized=("oos_recognized_event_days", "sum"),
            mean_promotion_score=("promotion_score", "mean"),
        )
        .reset_index()
    )
    if not gaps.empty:
        gap_summary = gaps.groupby("feature", observed=True)["incrementally_recognized"].sum()
        summary["ignored_cells_recognized"] = summary["feature"].map(gap_summary).fillna(0).astype(int)
    else:
        summary["ignored_cells_recognized"] = 0
    correlation = daily.corr(method="spearman").abs()
    np.fill_diagonal(correlation.values, np.nan)
    summary["max_library_spearman"] = summary["feature"].map(correlation.max())
    summary = summary.sort_values("mean_promotion_score", ascending=False, kind="stable")
    summary.to_csv(args.output / "feature_summary.csv", index=False)
    manifest = {
        "schema": "negative_residual_feature_assessment_v1",
        "feature_file": str(args.feature_file),
        "period": [args.start, args.end],
        "train_direction_threshold_period": "2025",
        "oos_period": "2026 through end",
        "features": int(len(summary)),
        "side_archetype_rows": int(len(metrics)),
        "best_features": summary.head(10)["feature"].tolist(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, default=Path("data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/residual_episode_recognition_calendar_20260712_v1/calendar_recognized_vs_ignored.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/negative_residual_features_assessment_20260712_v1"))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-07-10")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
