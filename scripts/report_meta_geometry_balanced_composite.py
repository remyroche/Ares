#!/usr/bin/env python3
"""Detailed OOS report for the balanced market/geometry composite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.tune_meta_geometry_rank_nudge import _prepare, _top10_mask

KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _aggregate(frame: pd.DataFrame, groups: list[str], selector: str) -> pd.DataFrame:
    grouped: Iterable[tuple[Any, pd.DataFrame]] = (
        [((), frame)]
        if not groups
        else frame.groupby(groups, observed=True, dropna=False, sort=True)
    )
    rows: list[dict[str, Any]] = []
    for key, part in grouped:
        values = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = {
            "selector": selector,
            "selected_rows": int(len(part)),
            "symbols": int(part["__symbol__"].nunique()),
            "mean_ev_after_1pct": float(part["ev_after_1pct"].mean()),
            "sum_ev_after_1pct": float(part["ev_after_1pct"].sum(min_count=1)),
            "clean_exec_precision": float(part["clean_exec"].mean()),
            "full_path_bad_mae_rate": float(part["full_path_bad_mae_1r"].mean()),
            "timeout_rate": float(part["timeout"].mean()),
        }
        for name, value in zip(groups, values, strict=False):
            row[name] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _state_label(values: pd.Series, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    label = np.full(len(values), f"{prefix}_neutral", dtype=object)
    label[numeric.le(-0.5).fillna(False).to_numpy()] = f"{prefix}_unfavorable"
    label[numeric.ge(0.5).fillna(False).to_numpy()] = f"{prefix}_favorable"
    return pd.Series(label, index=values.index)


def _autocorrelation(frame: pd.DataFrame, selector: str) -> pd.DataFrame:
    daily = (
        frame.groupby(
            ["day", "side_name", "archetype_policy_key"], observed=True, dropna=False
        )["ev_after_1pct"]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for (side, archetype), part in daily.groupby(
        ["side_name", "archetype_policy_key"], observed=True, dropna=False
    ):
        values = part.sort_values("day")["ev_after_1pct"]
        loss = values.lt(0.0).astype(np.float32)
        rows.append(
            {
                "selector": selector,
                "side_name": side,
                "archetype_policy_key": archetype,
                "days": int(len(values)),
                "mean_ev": float(values.mean()),
                "negative_day_rate": float(loss.mean()),
                "ev_autocorr_lag1": float(values.autocorr(1))
                if len(values) >= 3
                else np.nan,
                "loss_autocorr_lag1": float(loss.autocorr(1))
                if len(values) >= 3
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--nudge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_columns = [
        *KEYS,
        "calendar_month",
        "evaluation_scope",
        "base_batch_rank",
        "overlay_expected_ev_z",
        "ev_after_1pct",
        "clean_exec",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    source = pd.read_parquet(
        args.ablation_dir / "cross_sectional_geometry_predictions.parquet",
        columns=source_columns,
    )
    market = pd.read_parquet(
        args.ablation_dir / "market_state_predictions.parquet",
        columns=KEYS + ["evaluation_scope", "overlay_expected_ev_z"],
    ).rename(columns={"overlay_expected_ev_z": "market_state_ev_z"})
    balanced = pd.read_parquet(
        args.nudge_dir / "balanced_composite_predictions.parquet"
    )
    source = source.merge(
        balanced[KEYS + ["evaluation_scope", "selected_top10_balanced_composite"]],
        on=KEYS + ["evaluation_scope"],
        how="inner",
        validate="one_to_one",
    ).merge(
        market,
        on=KEYS + ["evaluation_scope"],
        how="left",
        validate="one_to_one",
    )
    source = source.rename(columns={"overlay_expected_ev_z": "geometry_state_ev_z"})
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    source["day"] = source["__ts__"].dt.floor("D")
    source["week_start"] = source["day"] - pd.to_timedelta(
        source["day"].dt.weekday, unit="D"
    )
    source["market_state"] = _state_label(source["market_state_ev_z"], "market")
    source["geometry_state"] = _state_label(source["geometry_state_ev_z"], "geometry")
    source["joint_state"] = source["market_state"] + "__" + source["geometry_state"]

    metric_parts: list[pd.DataFrame] = []
    autocorr_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    scopes = {
        "overall": [],
        "month": ["calendar_month"],
        "week": ["week_start"],
        "day": ["day"],
        "side": ["side_name"],
        "archetype": ["archetype_policy_key"],
        "month_side_archetype": ["calendar_month", "side_name", "archetype_policy_key"],
    }
    for evaluation_scope, frame in source.groupby("evaluation_scope", observed=True):
        prepared, offsets = _prepare(frame)
        baseline_mask = _top10_mask(
            prepared["base_batch_rank"].to_numpy(dtype=np.float32), offsets
        )
        balanced_mask = (
            prepared["selected_top10_balanced_composite"]
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        for selector, mask in (
            ("baseline", baseline_mask),
            ("balanced_composite_v1", balanced_mask),
        ):
            selected = prepared.loc[
                mask & prepared["ev_after_1pct"].notna().to_numpy()
            ].copy()
            selected["evaluation_scope"] = evaluation_scope
            for scope, groups in scopes.items():
                metrics = _aggregate(selected, groups, selector)
                metrics["scope"] = scope
                metrics["evaluation_scope"] = evaluation_scope
                metric_parts.append(metrics)
            autocorr_parts.append(
                _autocorrelation(selected, f"{evaluation_scope}__{selector}")
            )
            daily = _aggregate(selected, ["day"], selector)
            daily["evaluation_scope"] = evaluation_scope
            daily_parts.append(daily)

    top10 = source.loc[
        source["base_batch_rank"].ge(0.90) & source["ev_after_1pct"].notna()
    ]
    state_metrics = _aggregate(
        top10,
        [
            "evaluation_scope",
            "side_name",
            "archetype_policy_key",
            "market_state",
            "geometry_state",
        ],
        "baseline_top10_state_catalog",
    )
    local_baseline = _aggregate(
        top10,
        ["evaluation_scope", "side_name", "archetype_policy_key"],
        "baseline_top10_local",
    ).rename(columns={"mean_ev_after_1pct": "local_mean_ev_after_1pct"})
    state_metrics = state_metrics.merge(
        local_baseline[
            [
                "evaluation_scope",
                "side_name",
                "archetype_policy_key",
                "local_mean_ev_after_1pct",
            ]
        ],
        on=["evaluation_scope", "side_name", "archetype_policy_key"],
        how="left",
        validate="many_to_one",
    )
    state_metrics["ev_lift_vs_local"] = (
        state_metrics["mean_ev_after_1pct"] - state_metrics["local_mean_ev_after_1pct"]
    )

    metrics = pd.concat(metric_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    metrics.to_csv(args.output_dir / "metrics_by_scope.csv", index=False)
    daily.to_csv(args.output_dir / "daily_metrics.csv", index=False)
    pd.concat(autocorr_parts, ignore_index=True).to_csv(
        args.output_dir / "side_archetype_autocorrelation.csv", index=False
    )
    state_metrics.to_csv(args.output_dir / "state_catalog_metrics.csv", index=False)
    july_hourly = source.loc[source["evaluation_scope"].eq("july_oos")].copy()
    july_hourly["hour"] = july_hourly["__ts__"].dt.floor("h")
    hourly_parts = []
    for selector, mask_col in (
        ("baseline", None),
        ("balanced_composite_v1", "selected_top10_balanced_composite"),
    ):
        if mask_col is None:
            prepared, offsets = _prepare(july_hourly)
            mask = _top10_mask(
                prepared["base_batch_rank"].to_numpy(dtype=np.float32), offsets
            )
            selected = prepared.loc[mask]
        else:
            selected = july_hourly.loc[july_hourly[mask_col].fillna(False)]
        hourly = _aggregate(selected, ["hour"], selector)
        hourly_parts.append(hourly)
    pd.concat(hourly_parts, ignore_index=True).to_csv(
        args.output_dir / "july_hourly_metrics.csv", index=False
    )
    manifest = {
        "schema": "meta_geometry_balanced_composite_report_v1",
        "selection_contract": "exact global within-timestamp top-10 count",
        "cost_contract": "ev_after_1pct includes 1% round-trip cost",
        "state_contract": (
            "market and geometry states are frozen-model expected-EV z bands: <=-0.5 unfavorable, "
            ">=0.5 favorable, otherwise neutral"
        ),
        "historical_scope": "walk-forward OOS September 2025 through June 2026",
        "july_scope": "untouched July 1-10 evaluation",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
