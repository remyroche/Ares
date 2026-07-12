#!/usr/bin/env python3
"""Test whether historical-rank surprise events survive nearby causal definitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_train_meta_residual_archetype_enhancement import (
    DEFAULT_OUT_DIR,  # noqa: E402
)

QUANTILES = (0.85, 0.90, 0.95)
EMA_HOURS = (3, 6, 12)
GAP_HOURS = (1, 2, 4)
REFERENCE = (0.90, 6, 2)


def _event_ids(
    timestamps: pd.Series,
    side: pd.Series,
    archetype: pd.Series,
    active: pd.Series,
    max_gap_hours: int,
) -> pd.Series:
    output = pd.Series(pd.NA, index=timestamps.index, dtype="Int64")
    work = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce"),
            "side": side.astype(str),
            "archetype": archetype.astype(str),
            "active": active.fillna(False).astype(bool),
        },
        index=timestamps.index,
    )
    next_event = 0
    for _, group in work[work["active"]].groupby(["side", "archetype"], sort=True):
        group = group.sort_values("timestamp")
        gaps = group["timestamp"].diff().dt.total_seconds().div(3600.0)
        local = gaps.isna() | gaps.gt(float(max_gap_hours))
        ids = local.cumsum().to_numpy(dtype=np.int64) - 1 + next_event
        output.loc[group.index] = ids
        next_event = int(ids.max()) + 1 if len(ids) else next_event
    return output


def _hourly_surprise(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        pd.to_numeric(frame["historical_rank_current_reference"], errors="coerce").ge(
            0.90
        )
    ].copy()
    selected["__ts__"] = pd.to_datetime(selected["__ts__"], utc=True, errors="coerce")
    selected["surprise"] = pd.to_numeric(
        selected["clean_exec"], errors="coerce"
    ) - pd.to_numeric(selected["hit_prob_current_reference"], errors="coerce")
    return (
        selected.groupby(
            ["__ts__", "side_name", "archetype_policy_key"],
            dropna=False,
            sort=True,
        )
        .agg(rows=("surprise", "size"), surprise=("surprise", "mean"))
        .reset_index()
    )


def _configuration(
    hourly: pd.DataFrame,
    quantile: float,
    ema_hours: int,
    gap_hours: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    for _, group in hourly.groupby(
        ["side_name", "archetype_policy_key"], dropna=False, sort=True
    ):
        group = group.sort_values("__ts__").copy()
        group["smoothed_surprise"] = (
            group["surprise"]
            .ewm(
                span=int(ema_hours),
                adjust=False,
                min_periods=max(2, int(ema_hours // 2)),
            )
            .mean()
        )
        group["causal_tail_threshold"] = (
            group["smoothed_surprise"]
            .abs()
            .shift(1)
            .rolling(24 * 30, min_periods=24)
            .quantile(float(quantile))
        )
        group["active"] = (
            group["smoothed_surprise"].abs().ge(group["causal_tail_threshold"])
        )
        parts.append(group)
    output = pd.concat(parts, ignore_index=True)
    output["event_id"] = _event_ids(
        output["__ts__"],
        output["side_name"],
        output["archetype_policy_key"],
        output["active"],
        gap_hours,
    )
    active = output[output["active"]].copy()
    events = active.groupby("event_id", dropna=True).agg(
        start=("__ts__", "min"),
        end=("__ts__", "max"),
        event_hours=("__ts__", "nunique"),
        rows=("rows", "sum"),
        surprise_mass=("smoothed_surprise", lambda values: float(np.abs(values).sum())),
    )
    total_mass = float(events["surprise_mass"].sum()) if len(events) else 0.0
    summary = {
        "quantile": quantile,
        "ema_hours": ema_hours,
        "gap_hours": gap_hours,
        "active_hours": int(len(active)),
        "event_count": int(len(events)),
        "median_event_duration_hours": float(events["event_hours"].median())
        if len(events)
        else np.nan,
        "maximum_event_duration_hours": float(events["event_hours"].max())
        if len(events)
        else np.nan,
        "median_rows_per_event": float(events["rows"].median())
        if len(events)
        else np.nan,
        "largest_event_surprise_share": (
            float(events["surprise_mass"].max() / total_mass)
            if total_mass > 0.0
            else np.nan
        ),
    }
    active["event_key"] = (
        active["__ts__"].astype(str)
        + "|"
        + active["side_name"].astype(str)
        + "|"
        + active["archetype_policy_key"].astype(str)
    )
    return active, summary


def main() -> None:
    root = DEFAULT_OUT_DIR
    report_dir = root / "final_report"
    source = pd.read_parquet(
        root / "historical_rank_oos" / "oos_predictions_historical_rank.parquet"
    )
    hourly = _hourly_surprise(source)
    active_by_config: dict[tuple[float, int, int], set[str]] = {}
    summaries: list[dict[str, Any]] = []
    for quantile in QUANTILES:
        for ema_hours in EMA_HOURS:
            for gap_hours in GAP_HOURS:
                active, summary = _configuration(hourly, quantile, ema_hours, gap_hours)
                key = (quantile, ema_hours, gap_hours)
                active_by_config[key] = set(active["event_key"].astype(str))
                summaries.append(summary)
    reference = active_by_config[REFERENCE]
    jaccard_rows: list[dict[str, Any]] = []
    for key, values in active_by_config.items():
        union = reference | values
        jaccard_rows.append(
            {
                "quantile": key[0],
                "ema_hours": key[1],
                "gap_hours": key[2],
                "jaccard_vs_reference": float(
                    len(reference & values) / max(len(union), 1)
                ),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    jaccard = pd.DataFrame(jaccard_rows)
    summary_frame.to_csv(
        report_dir / "stage4_event_definition_sensitivity.csv", index=False
    )
    jaccard.to_csv(report_dir / "stage4_event_jaccard.csv", index=False)
    non_reference = jaccard[
        ~(
            jaccard["quantile"].eq(REFERENCE[0])
            & jaccard["ema_hours"].eq(REFERENCE[1])
            & jaccard["gap_hours"].eq(REFERENCE[2])
        )
    ]
    local_neighborhood = non_reference[
        (
            non_reference["ema_hours"].eq(REFERENCE[1])
            & non_reference["gap_hours"].eq(REFERENCE[2])
        )
        | (
            non_reference["quantile"].eq(REFERENCE[0])
            & non_reference["gap_hours"].eq(REFERENCE[2])
        )
        | (
            non_reference["quantile"].eq(REFERENCE[0])
            & non_reference["ema_hours"].eq(REFERENCE[1])
        )
    ]
    local_neighborhood.to_csv(
        report_dir / "stage4_event_local_neighborhood_jaccard.csv",
        index=False,
    )
    local_median = float(local_neighborhood["jaccard_vs_reference"].median())
    manifest = {
        "schema": "meta_residual_event_sensitivity_v1",
        "reference": {
            "quantile": REFERENCE[0],
            "ema_hours": REFERENCE[1],
            "gap_hours": REFERENCE[2],
        },
        "configurations": int(len(summary_frame)),
        "local_neighborhood_median_jaccard": local_median,
        "full_factorial_median_jaccard": float(
            non_reference["jaccard_vs_reference"].median()
        ),
        "minimum_jaccard_vs_reference": float(
            non_reference["jaccard_vs_reference"].min()
        ),
        "maximum_largest_event_surprise_share": float(
            summary_frame["largest_event_surprise_share"].max()
        ),
        "event_stability_pass": bool(
            local_median >= 0.60
            and summary_frame["largest_event_surprise_share"].max() <= 0.50
        ),
        "causal_contract": (
            "Every threshold is a shifted rolling 30-day quantile; the current hour never enters its threshold."
        ),
    }
    (report_dir / "stage4_event_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
