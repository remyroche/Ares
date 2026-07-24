#!/usr/bin/env python3
"""Robustness report for the strict extreme-local champion overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_20260712_v4"
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()

    path = args.artifact_dir / "oos_predictions.parquet"
    frame = pd.read_parquet(path)
    timestamp = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["week_start"] = timestamp.dt.floor("D") - pd.to_timedelta(
        timestamp.dt.weekday.to_numpy(), unit="D"
    )
    parent = frame["selected_parent"].astype(bool).to_numpy()
    strict = frame["selected_strict_extreme_local"].astype(bool).to_numpy()
    parent_week = frame.loc[parent].groupby("week_start")["ev_after_1pct"].mean()
    strict_week = frame.loc[strict].groupby("week_start")["ev_after_1pct"].mean()
    weekly = pd.concat(
        [parent_week.rename("parent_ev"), strict_week.rename("strict_ev")], axis=1
    ).dropna()
    weekly["delta_ev"] = weekly["strict_ev"] - weekly["parent_ev"]
    delta = weekly["delta_ev"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(20260712)
    sampled = delta[
        rng.integers(0, len(delta), size=(int(args.bootstrap_draws), len(delta)))
    ].mean(axis=1)
    weekly.reset_index().to_csv(args.artifact_dir / "paired_weekly_delta.csv", index=False)

    dropped = frame.loc[parent & ~strict].copy()
    added = frame.loc[~parent & strict].copy()
    dropped.groupby(
        ["side_name", "archetype_policy_key"], observed=True, dropna=False
    ).agg(
        rows=("ev_after_1pct", "size"),
        mean_ev_after_1pct=("ev_after_1pct", "mean"),
        clean_exec_precision=("clean_exec", "mean"),
        dirty_positive_rate=("dirty_positive", "mean"),
        bad_mae_rate=("full_path_bad_mae_1r", "mean"),
        timeout_rate=("timeout", "mean"),
    ).reset_index().to_csv(
        args.artifact_dir / "dropped_rows_by_side_archetype.csv", index=False
    )
    _write_json(
        args.artifact_dir / "robustness.json",
        {
            "schema": "meta_residual_extreme_local_champion_robustness_v1",
            "artifact": str(path),
            "weeks": int(len(delta)),
            "paired_weekly_mean_ev_delta": float(delta.mean()),
            "bootstrap_draws": int(args.bootstrap_draws),
            "bootstrap_ci025": float(np.quantile(sampled, 0.025)),
            "bootstrap_ci975": float(np.quantile(sampled, 0.975)),
            "bootstrap_positive_probability": float(np.mean(sampled > 0.0)),
            "parent_rows": int(parent.sum()),
            "strict_rows": int(strict.sum()),
            "dropped_rows": int(len(dropped)),
            "added_rows": int(len(added)),
            "dropped_mean_ev_after_1pct": float(dropped["ev_after_1pct"].mean()),
            "dropped_clean_exec_precision": float(dropped["clean_exec"].mean()),
            "dropped_bad_mae_rate": float(dropped["full_path_bad_mae_1r"].mean()),
            "dropped_timeout_rate": float(dropped["timeout"].mean()),
        },
    )


if __name__ == "__main__":
    main()
