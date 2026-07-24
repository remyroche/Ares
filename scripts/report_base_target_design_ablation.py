#!/usr/bin/env python3
"""Consolidate comparable target-design ablation artifacts.

The report only aggregates precomputed chronological OOS fold metrics.  It does
not select a target, fit a model, or run a threshold/policy search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUTS = (
    Path("data_perp/reports/base_target_design_ablation_100k_april_20260714_v3"),
    Path("data_perp/reports/base_target_design_ablation_100k_may_20260714_v1"),
    Path("data_perp/reports/base_target_design_ablation_100k_june_20260714_v1"),
)
DEFAULT_OUTPUT = Path("data_perp/reports/base_target_design_ablation_100k_apr_jun_20260714_v1")
BASELINE_ARM = "production_frozen"


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    weights = pd.to_numeric(frame["selected_rows"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    return float(np.average(values[mask], weights=weights[mask])) if np.any(mask) else float("nan")


def _summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    mean_columns = (
        "mean_ev_after_1pct",
        "positive_ev_rate",
        "clean_exec_precision",
        "full_path_bad_mae_rate",
        "timeout_rate",
        "stop_or_adverse_rate",
        "score_net_spearman",
    )
    rows: list[dict[str, Any]] = []
    for keys, sub in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        row["folds"] = int(sub["fold"].nunique())
        row["selected_rows"] = int(sub["selected_rows"].sum())
        row["sum_ev_after_1pct"] = float(sub["sum_ev_after_1pct"].sum())
        row["worst_fold_mean_ev"] = float(sub["mean_ev_after_1pct"].min())
        row["worst_week_mean_ev"] = float(sub["worst_week_mean_ev"].min())
        for column in mean_columns:
            row[column] = _weighted_mean(sub, column)
        rows.append(row)
    return pd.DataFrame(rows)


def _deltas(summary: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    baseline = summary.loc[summary["arm"].eq(BASELINE_ARM)].set_index(group_cols)
    rows: list[dict[str, Any]] = []
    columns = (
        "mean_ev_after_1pct",
        "clean_exec_precision",
        "full_path_bad_mae_rate",
        "timeout_rate",
        "stop_or_adverse_rate",
        "worst_fold_mean_ev",
        "worst_week_mean_ev",
    )
    for row in summary.to_dict(orient="records"):
        if row["arm"] == BASELINE_ARM:
            continue
        key = tuple(row[column] for column in group_cols)
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        result = {column: row[column] for column in ["arm", *group_cols]}
        for column in columns:
            result[f"delta_{column}"] = float(row[column] - base[column])
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=",".join(map(str, DEFAULT_INPUTS)))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = [Path(value.strip()) for value in str(args.inputs).split(",") if value.strip()]
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for root in inputs:
        metrics_path = root / "metrics.csv"
        manifest_path = root / "manifest.json"
        if not metrics_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Expected complete ablation input under {root}")
        frames.append(pd.read_csv(metrics_path))
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    metrics = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "combined_metrics.csv", index=False)
    for name, grouping_filter in {
        "overall": metrics["grouping"].eq("overall"),
        "by_side": metrics["grouping"].eq("side"),
        "by_archetype": metrics["grouping"].eq("archetype"),
        "by_week": metrics["grouping"].eq("week"),
    }.items():
        section = metrics.loc[grouping_filter].copy()
        group_cols = ["arm", "top_fraction", "group_value"] if name != "overall" else ["arm", "top_fraction"]
        summary = _summary(section, group_cols)
        summary.to_csv(args.output_dir / f"{name}_summary.csv", index=False)
        _deltas(summary, group_cols=[column for column in group_cols if column != "arm"]).to_csv(
            args.output_dir / f"{name}_delta_vs_production.csv", index=False
        )
    manifest = {
        "status": "complete",
        "inputs": [str(path) for path in inputs],
        "folds": sorted(metrics["fold"].dropna().unique().tolist()),
        "training_contract": "Each source is a fixed-feature/fixed-parameter chronological target ablation; no feature selection, HPO, threshold, or policy fitting was rerun.",
        "promotion_status": "research_only_capacity_limited",
        "source_train_caps": [item.get("fold_metadata", {}) for item in manifests],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
