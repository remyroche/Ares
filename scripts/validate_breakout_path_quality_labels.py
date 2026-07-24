#!/usr/bin/env python3
"""Materialize and audit train-only breakout path-quality labels by fold.

The script deliberately validates each outcome component before a combined
breakout model is considered.  All columns used to build the labels are
realized path outcomes and are excluded from inference feature contracts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.breakout_path_quality_labels import (
    breakout_path_quality_label_manifest,
    fit_breakout_path_quality_thresholds,
    materialize_breakout_path_quality_labels,
)


REQUIRED = [
    "__ts__",
    "side_name",
    "__archetype_policy_key__",
    "__path_trailing_success__",
    "__first_touch_mfe_norm__",
    "__first_touch_full_path_mae_norm__",
    "__first_touch_mfe_to_tp__",
    "__path_post_mfe_drawdown_norm__",
]
OUTCOME_NAMES = [
    "breakout_retention_outcome",
    "breakout_path_efficiency_outcome",
    "breakout_participation_outcome",
    "breakout_reversal_magnitude_outcome",
]


def _derive_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Create four post-entry outcomes with comparable orientation.

    Retention and participation are higher-is-better.  Efficiency is the
    favorable MFE share of observed directional path.  Reversal is the
    post-MFE drawdown and is higher-is-worse.
    """

    output = frame.loc[:, ["__ts__", "side_name", "__archetype_policy_key__"]].copy()
    mfe = pd.to_numeric(frame["__first_touch_mfe_norm__"], errors="coerce").clip(lower=0.0)
    mae = pd.to_numeric(frame["__first_touch_full_path_mae_norm__"], errors="coerce").clip(lower=0.0)
    output["breakout_retention_outcome"] = pd.to_numeric(
        frame["__path_trailing_success__"], errors="coerce"
    ).clip(0.0, 1.0)
    output["breakout_path_efficiency_outcome"] = (
        mfe / (mfe + mae + np.float32(1e-6))
    ).clip(0.0, 1.0)
    output["breakout_participation_outcome"] = pd.to_numeric(
        frame["__first_touch_mfe_to_tp__"], errors="coerce"
    ).clip(0.0, 10.0)
    output["breakout_reversal_magnitude_outcome"] = pd.to_numeric(
        frame["__path_post_mfe_drawdown_norm__"], errors="coerce"
    ).clip(0.0, 20.0)
    output = output.rename(columns={"__archetype_policy_key__": "archetype_policy_key"})
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="coerce")
    return output.dropna(subset=["__ts__", "side_name", "archetype_policy_key"])


def _load_labels(labels_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(labels_dir.glob("train_global_*.parquet")):
        available = set(pq.read_schema(path).names)
        missing = set(REQUIRED).difference(available)
        if missing:
            continue
        raw = pd.read_parquet(path, columns=REQUIRED)
        parts.append(_derive_outcomes(raw))
    if not parts:
        raise FileNotFoundError("No compatible labeled outcome partitions found")
    return pd.concat(parts, ignore_index=True, copy=False)


def _quarter_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    naive_start = start.tz_localize(None)
    quarter_start = naive_start.to_period("Q").start_time.tz_localize("UTC")
    return list(pd.date_range(quarter_start, end, freq="QS", tz="UTC"))


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    values = _load_labels(args.labels_dir)
    start = pd.Timestamp(args.eval_start, tz="UTC")
    end = pd.Timestamp(args.eval_end, tz="UTC")
    values = values.loc[values["__ts__"].lt(end)].copy()
    metric_rows: list[dict[str, object]] = []
    materialized: list[pd.DataFrame] = []
    for side, archetype in (
        values.loc[:, ["side_name", "archetype_policy_key"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        local = values.loc[
            values["side_name"].eq(side) & values["archetype_policy_key"].eq(archetype)
        ].sort_values("__ts__", kind="stable")
        for fold_start in _quarter_starts(start, end):
            fold_end = min(fold_start + pd.DateOffset(months=3), end)
            train = local.loc[local["__ts__"].lt(fold_start)]
            scored = local.loc[local["__ts__"].ge(fold_start) & local["__ts__"].lt(fold_end)]
            if len(train) < args.minimum_train_rows or len(scored) < args.minimum_eval_rows:
                continue
            try:
                thresholds = fit_breakout_path_quality_thresholds(train.loc[:, OUTCOME_NAMES])
            except ValueError:
                continue
            labels = materialize_breakout_path_quality_labels(scored.loc[:, OUTCOME_NAMES], thresholds)
            summary = {
                "fold_start": fold_start,
                "fold_end": fold_end,
                "side_name": side,
                "archetype_policy_key": archetype,
                "train_rows": int(len(train)),
                "eval_rows": int(len(scored)),
                "label_valid_rate": float(labels["breakout_quality_label_valid"].mean()),
                "retention_failure_rate": float(labels["breakout_retention_failure"].mean()),
                "low_efficiency_rate": float(labels["breakout_low_efficiency"].mean()),
                "participation_failure_rate": float(labels["breakout_participation_failure"].mean()),
                "rapid_reversal_rate": float(labels["breakout_rapid_reversal"].mean()),
                "combined_failure_rate": float(labels["breakout_any_path_quality_failure"].mean()),
                "combined_soft_risk": float(labels["breakout_path_quality_soft_risk"].mean()),
                **breakout_path_quality_label_manifest(thresholds)["thresholds"],
            }
            metric_rows.append(summary)
            if side == "short" and archetype == "short_breakout_precision":
                part = scored.loc[:, ["__ts__", "side_name", "archetype_policy_key"]].copy()
                part = pd.concat([part.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
                part["fold_start"] = fold_start
                materialized.append(part)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "breakout_path_quality_fold_metrics.csv", index=False)
    if materialized:
        pd.concat(materialized, ignore_index=True, copy=False).to_parquet(
            args.output / "short_breakout_oof_path_quality_labels.parquet",
            index=False,
            compression="zstd",
        )
    focus = metrics.loc[
        metrics["archetype_policy_key"].isin(
            ["short_breakout_precision", "long_breakout_diagnostic_candidate"]
        )
    ]
    manifest = {
        "schema": "breakout_path_quality_validation_v1",
        "status": "labels_materialized_for_component_validation_only",
        "labels_dir": str(args.labels_dir),
        "evaluation": {"start": str(start), "end": str(end)},
        "rows": int(len(values)),
        "fold_rows": int(len(metrics)),
        "focus_rows": int(len(focus)),
        "outcome_contract": {
            "retention": "__path_trailing_success__",
            "efficiency": "first-touch MFE / (MFE + full-path MAE)",
            "participation": "__first_touch_mfe_to_tp__",
            "reversal": "__path_post_mfe_drawdown_norm__",
        },
        "leakage_contract": (
            "All label cutoffs are fitted only on observations before each fold. "
            "The four realized path outcomes are excluded from inference features."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-start", default="2025-07-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--minimum-train-rows", type=int, default=500)
    parser.add_argument("--minimum-eval-rows", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
