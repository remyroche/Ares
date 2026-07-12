#!/usr/bin/env python3
"""Summarize OOS transfer evidence from hierarchical AE/GMM state research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT = Path(
    "data_perp/reports/hierarchical_aegmm_state_validation_20260712_v1"
)
FINAL_FOLDS = {"2026-04", "2026-05", "2026-06"}


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0.0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _summarize(metrics: pd.DataFrame, *, cohort: str) -> pd.DataFrame:
    current = metrics.loc[metrics["scope"].eq("side_archetype")].copy()
    if current.empty:
        return current
    grouping = ["arm", "state_block", "zone", "side_name", "archetype_policy_key"]
    rows: list[dict[str, Any]] = []
    for keys, group in current.groupby(grouping, observed=True, sort=True):
        row = dict(zip(grouping, keys, strict=True))
        weights = pd.to_numeric(group["selected_rows"], errors="coerce").fillna(0.0)
        row.update(
            {
                "cohort": cohort,
                "folds": int(group["fold"].nunique()),
                "rows": int(weights.sum()),
                "mean_ev_after_1pct": _weighted_mean(
                    group["mean_ev_after_1pct"], weights
                ),
                "worst_fold_mean_ev": float(group["mean_ev_after_1pct"].min()),
                "worst_week_mean_ev": float(group["worst_week_mean_ev"].min()),
                "worst_month_mean_ev": float(group["worst_month_mean_ev"].min()),
                "mean_expected_ev_spearman": _weighted_mean(
                    group["state_expected_ev_spearman"], weights
                ),
                "positive_ev_spearman_fold_share": float(
                    group["state_expected_ev_spearman"].gt(0.0).mean()
                ),
                "mean_ev_top_minus_bottom": _weighted_mean(
                    group["expected_ev_top_minus_bottom"], weights
                ),
                "positive_ev_lift_fold_share": float(
                    group["expected_ev_top_minus_bottom"].gt(0.0).mean()
                ),
                "mean_bad_mae_spearman": _weighted_mean(
                    group["state_expected_bad_mae_spearman"], weights
                ),
                "positive_bad_mae_ordering_fold_share": float(
                    group["state_expected_bad_mae_spearman"].gt(0.0).mean()
                ),
                "mean_bad_mae_high_minus_low": _weighted_mean(
                    group["pred_bad_mae_high_minus_low"], weights
                ),
                "clean_exec_precision": _weighted_mean(
                    group["clean_exec_precision"], weights
                ),
                "first_touch_bad_mae_rate": _weighted_mean(
                    group["first_touch_bad_mae_rate"], weights
                ),
                "timeout_rate": _weighted_mean(group["timeout_rate"], weights),
                "mean_hit_surprise": _weighted_mean(
                    group["mean_hit_surprise"], weights
                ),
            }
        )
        # This is a research gate, not a policy rule. It deliberately requires
        # both opportunity and adverse-path ordering across multiple OOS folds.
        row["state_transfer_research_pass"] = bool(
            row["folds"] >= 3
            and np.isfinite(row["mean_expected_ev_spearman"])
            and row["mean_expected_ev_spearman"] > 0.0
            and np.isfinite(row["mean_ev_top_minus_bottom"])
            and row["mean_ev_top_minus_bottom"] > 0.0
            and np.isfinite(row["mean_bad_mae_spearman"])
            and row["mean_bad_mae_spearman"] > 0.0
            and row["positive_ev_spearman_fold_share"] >= 0.60
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _autocorr_summary(autocorr: pd.DataFrame, *, cohort: str) -> pd.DataFrame:
    if autocorr.empty:
        return autocorr
    grouping = ["arm", "state_block", "zone", "side_name", "archetype_policy_key"]
    rows: list[dict[str, Any]] = []
    for keys, group in autocorr.groupby(grouping, observed=True, sort=True):
        row = dict(zip(grouping, keys, strict=True))
        weights = pd.to_numeric(group["days"], errors="coerce").fillna(0.0)
        row.update(
            {
                "cohort": cohort,
                "folds": int(group["fold"].nunique()),
                "days": int(weights.sum()),
                "signed_hit_surprise_autocorr_lag1": _weighted_mean(
                    group["signed_hit_surprise_autocorr_lag1"], weights
                ),
                "negative_hit_surprise_autocorr_lag1": _weighted_mean(
                    group["negative_hit_surprise_autocorr_lag1"], weights
                ),
                "positive_hit_surprise_autocorr_lag1": _weighted_mean(
                    group["positive_hit_surprise_autocorr_lag1"], weights
                ),
                "worst_day_mean_ev": float(group["worst_day_mean_ev"].min()),
                "mean_daily_ev": _weighted_mean(group["mean_daily_ev"], weights),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _fold_coverage(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_dir in sorted((input_dir / "folds").glob("*")):
        prediction_path = fold_dir / "oos_state_predictions.parquet"
        manifest_path = fold_dir / "manifest.json"
        if not prediction_path.exists() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        values = pd.read_parquet(prediction_path, columns=["__ts__"])
        timestamp = pd.to_datetime(values["__ts__"], utc=True, errors="coerce")
        start = pd.Timestamp(manifest["oos_start"])
        end = pd.Timestamp(manifest["oos_end_exclusive"])
        expected_days = max(int((end.normalize() - start.normalize()).days), 1)
        observed_days = int(timestamp.dt.normalize().nunique())
        raw_fold = str(manifest.get("fold", fold_dir.name))
        fold = (
            f"{raw_fold[:4]}-{raw_fold[4:6]}"
            if raw_fold.isdigit() and len(raw_fold) == 6
            else raw_fold
        )
        rows.append(
            {
                "fold": fold,
                "oos_start": str(start),
                "oos_end_exclusive": str(end),
                "observed_start": str(timestamp.min()),
                "observed_end": str(timestamp.max()),
                "expected_days": expected_days,
                "observed_days": observed_days,
                "day_coverage": float(observed_days / expected_days),
                "rows": int(len(values)),
                "complete_month": bool(observed_days / expected_days >= 0.80),
            }
        )
    return pd.DataFrame(rows)


def run(input_dir: Path) -> dict[str, Any]:
    metrics_path = input_dir / "oos_zone_metrics_by_side_archetype.csv"
    autocorr_path = input_dir / "oos_hit_surprise_autocorrelation_by_side_archetype.csv"
    if not metrics_path.exists() or not autocorr_path.exists():
        raise FileNotFoundError("State validation aggregate tables are not present yet")
    metrics = pd.read_csv(metrics_path)
    autocorr = pd.read_csv(autocorr_path)
    coverage = _fold_coverage(input_dir)
    final_mask = metrics["fold"].astype(str).isin(FINAL_FOLDS)
    dev = _summarize(metrics.loc[~final_mask], cohort="development_2025_to_2026_03")
    final = _summarize(metrics.loc[final_mask], cohort="final_2026_04_to_06")
    combined = pd.concat([dev, final], ignore_index=True)
    ac_final_mask = autocorr["fold"].astype(str).isin(FINAL_FOLDS)
    ac_dev = _autocorr_summary(
        autocorr.loc[~ac_final_mask], cohort="development_2025_to_2026_03"
    )
    ac_final = _autocorr_summary(
        autocorr.loc[ac_final_mask], cohort="final_2026_04_to_06"
    )
    ac_combined = pd.concat([ac_dev, ac_final], ignore_index=True)
    dev.to_csv(
        input_dir / "state_transfer_development_by_side_archetype.csv", index=False
    )
    final.to_csv(
        input_dir / "state_transfer_final_holdout_by_side_archetype.csv", index=False
    )
    combined.to_csv(
        input_dir / "state_transfer_summary_by_side_archetype.csv", index=False
    )
    ac_combined.to_csv(
        input_dir / "state_transfer_autocorrelation_by_side_archetype.csv", index=False
    )
    coverage.to_csv(input_dir / "state_transfer_fold_coverage.csv", index=False)
    overall = (
        combined.groupby(["cohort", "arm", "state_block", "zone"], observed=True)
        .agg(
            side_archetype_cells=("archetype_policy_key", "size"),
            research_pass_cells=("state_transfer_research_pass", "sum"),
            mean_ev_spearman=("mean_expected_ev_spearman", "mean"),
            mean_ev_lift=("mean_ev_top_minus_bottom", "mean"),
            mean_bad_mae_spearman=("mean_bad_mae_spearman", "mean"),
            worst_week_mean_ev=("worst_week_mean_ev", "min"),
            worst_month_mean_ev=("worst_month_mean_ev", "min"),
        )
        .reset_index()
    )
    overall.to_csv(input_dir / "state_transfer_overall_summary.csv", index=False)
    result = {
        "development_rows": int(len(dev)),
        "final_holdout_rows": int(len(final)),
        "development_pass_cells": int(dev["state_transfer_research_pass"].sum())
        if not dev.empty
        else 0,
        "final_holdout_pass_cells": int(final["state_transfer_research_pass"].sum())
        if not final.empty
        else 0,
        "incomplete_final_folds": coverage.loc[
            coverage["fold"].astype(str).isin(FINAL_FOLDS)
            & ~coverage["complete_month"],
            "fold",
        ]
        .astype(str)
        .tolist()
        if not coverage.empty
        else [],
        "output": str(input_dir),
    }
    (input_dir / "state_transfer_report.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    print(json.dumps(_safe(run(args.input_dir)), indent=2), flush=True)


if __name__ == "__main__":
    main()
