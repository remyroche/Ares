#!/usr/bin/env python3
"""Report signed hit-surprise autocorrelation for fixed-meta state ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path(
    "data_perp/reports/hierarchical_aegmm_fixed_meta_ablation_20260712_v1"
)


def _selection_mask(frame: pd.DataFrame, score_col: str) -> pd.Series:
    score = pd.to_numeric(frame[score_col], errors="coerce")
    timestamp = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    return score.groupby(timestamp, sort=False).rank(method="first", pct=True).ge(0.90)


def _autocorr(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    if (
        len(array) < 5
        or float(np.std(array[:-1])) <= 1e-10
        or float(np.std(array[1:])) <= 1e-10
    ):
        return float("nan")
    return float(np.corrcoef(array[:-1], array[1:])[0, 1])


def _daily_and_autocorr(
    frame: pd.DataFrame, arm: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = frame.loc[_selection_mask(frame, "score_alternative")].copy()
    selected["date"] = pd.to_datetime(selected["__ts__"], utc=True).dt.normalize()
    selected["hit_surprise"] = pd.to_numeric(
        selected["clean_exec"], errors="coerce"
    ) - pd.to_numeric(selected["hit_prob_alternative"], errors="coerce")
    selected["negative_hit_surprise"] = selected["hit_surprise"].clip(upper=0.0)
    selected["positive_hit_surprise"] = selected["hit_surprise"].clip(lower=0.0)
    daily = (
        selected.groupby(
            ["date", "side_name", "archetype_policy_key"], observed=True, sort=True
        )
        .agg(
            selected_rows=("hit_surprise", "size"),
            mean_hit_surprise=("hit_surprise", "mean"),
            negative_hit_surprise=("negative_hit_surprise", "mean"),
            positive_hit_surprise=("positive_hit_surprise", "mean"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            clean_exec_precision=("clean_exec", "mean"),
            first_touch_bad_mae_rate=("first_touch_bad_mae_1r", "mean"),
            timeout_rate=("timeout", "mean"),
        )
        .reset_index()
    )
    daily["arm"] = arm
    rows: list[dict[str, Any]] = []
    for (side, archetype), group in daily.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        ordered = group.sort_values("date", kind="stable")
        rows.append(
            {
                "arm": arm,
                "side_name": side,
                "archetype_policy_key": archetype,
                "days": int(len(ordered)),
                "signed_hit_surprise_autocorr_lag1": _autocorr(
                    ordered["mean_hit_surprise"]
                ),
                "negative_hit_surprise_autocorr_lag1": _autocorr(
                    ordered["negative_hit_surprise"]
                ),
                "positive_hit_surprise_autocorr_lag1": _autocorr(
                    ordered["positive_hit_surprise"]
                ),
                "mean_daily_ev": float(ordered["mean_ev_after_1pct"].mean()),
                "worst_day_ev": float(ordered["mean_ev_after_1pct"].min()),
            }
        )
    return daily, pd.DataFrame(rows)


def _weighted_abs(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid].abs(), weights=weights.loc[valid]))


def run(input_dir: Path) -> dict[str, Any]:
    paths = sorted(input_dir.glob("*/oos_predictions.parquet"))
    if not paths:
        raise FileNotFoundError(f"No arm predictions under {input_dir}")
    daily_frames: list[pd.DataFrame] = []
    autocorr_frames: list[pd.DataFrame] = []
    for path in paths:
        arm = path.parent.name
        frame = pd.read_parquet(path)
        daily, autocorr = _daily_and_autocorr(frame, arm)
        daily_frames.append(daily)
        autocorr_frames.append(autocorr)
    daily = pd.concat(daily_frames, ignore_index=True)
    autocorr = pd.concat(autocorr_frames, ignore_index=True)
    summary_rows: list[dict[str, Any]] = []
    for arm, group in autocorr.groupby("arm", observed=True, sort=True):
        weights = pd.to_numeric(group["days"], errors="coerce").fillna(0.0)
        summary_rows.append(
            {
                "arm": arm,
                "side_archetype_cells": int(len(group)),
                "days": int(weights.sum()),
                "mean_abs_signed_hit_surprise_autocorr": _weighted_abs(
                    group["signed_hit_surprise_autocorr_lag1"], weights
                ),
                "mean_abs_negative_hit_surprise_autocorr": _weighted_abs(
                    group["negative_hit_surprise_autocorr_lag1"], weights
                ),
                "mean_abs_positive_hit_surprise_autocorr": _weighted_abs(
                    group["positive_hit_surprise_autocorr_lag1"], weights
                ),
                "worst_daily_ev": float(group["worst_day_ev"].min()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    baseline = summary.loc[summary["arm"].eq("baseline_retrained")]
    if len(baseline) == 1:
        base = baseline.iloc[0]
        for name in (
            "mean_abs_signed_hit_surprise_autocorr",
            "mean_abs_negative_hit_surprise_autocorr",
            "mean_abs_positive_hit_surprise_autocorr",
        ):
            summary[f"delta_{name}_vs_baseline"] = summary[name] - float(base[name])
    daily.to_csv(input_dir / "selected_top10_signed_surprise_daily.csv", index=False)
    autocorr.to_csv(
        input_dir / "selected_top10_signed_surprise_autocorrelation.csv", index=False
    )
    summary.to_csv(
        input_dir / "selected_top10_signed_surprise_summary.csv", index=False
    )
    return {
        "output": str(input_dir),
        "arms": summary.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.input_dir), indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
