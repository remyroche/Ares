#!/usr/bin/env python3
"""Merge independently executed strict-OOF transition-head weekly shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_execution_ev_market_state_transition_heads import _metric_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.shard_root
    shards = sorted(root.glob("h*_w*/strict_weekly_oof_transition_predictions.parquet"))
    if not shards:
        raise FileNotFoundError(f"no transition prediction shards beneath {root}")
    predictions = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    # A complete result needs exactly one feature-set prediction per side,
    # horizon and decision row.  Duplicate shards are a hard error, rather
    # than silently cherry-picking an arbitrary re-run.
    key = ["candidate_id", "side_name", "horizon_hours", "feature_set"]
    if predictions.duplicated(key).any():
        duplicates = predictions.loc[predictions.duplicated(key, keep=False), key]
        raise ValueError(f"duplicate transition OOF shard rows: {duplicates.head().to_dict('records')}")
    expected_horizons = {1, 3, 6, 12}
    observed_horizons = set(pd.to_numeric(predictions["horizon_hours"], errors="coerce").dropna().astype(int))
    if observed_horizons != expected_horizons:
        raise ValueError(f"incomplete horizon coverage: observed={sorted(observed_horizons)}")
    weekly = pd.concat(
        [pd.read_csv(path.parent / "strict_weekly_oof_transition_metrics.csv") for path in shards],
        ignore_index=True,
    )
    aggregate_rows = []
    for (feature_set, side, horizon), group in predictions.groupby(
        ["feature_set", "side_name", "horizon_hours"], sort=True
    ):
        aggregate_rows.append(
            _metric_row(
                group,
                prediction_column="oof_transition_probability",
                target_column="raw_state_transition_label",
                label=str(feature_set),
                side=str(side),
                horizon=int(horizon),
                week_start=pd.NaT,
            )
        )
    aggregate = pd.DataFrame(aggregate_rows)
    latest_week = pd.to_datetime(predictions["week_start"], utc=True).max()
    latest = weekly.loc[pd.to_datetime(weekly["week_start"], utc=True).eq(latest_week)].copy()
    predictions.to_parquet(root / "strict_weekly_oof_transition_predictions.parquet", index=False)
    weekly.to_csv(root / "strict_weekly_oof_transition_metrics.csv", index=False)
    aggregate.to_csv(root / "strict_weekly_oof_transition_aggregate_metrics.csv", index=False)
    latest.to_csv(root / "latest_week_transition_metrics.csv", index=False)
    summary = {
        "schema": "execution_ev_raw_market_state_transition_head_merged_v1",
        "shards": [str(path.parent) for path in shards],
        "rows": int(len(predictions)),
        "horizons_hours": sorted(observed_horizons),
        "latest_week_start": str(latest_week),
        "strict_oof": True,
        "outputs": {
            "predictions": str(root / "strict_weekly_oof_transition_predictions.parquet"),
            "weekly_metrics": str(root / "strict_weekly_oof_transition_metrics.csv"),
            "aggregate_metrics": str(root / "strict_weekly_oof_transition_aggregate_metrics.csv"),
            "latest_week_metrics": str(root / "latest_week_transition_metrics.csv"),
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
