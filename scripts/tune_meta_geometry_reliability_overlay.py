#!/usr/bin/env python3
"""Test causal recent side/archetype reliability on the balanced geometry score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.tune_meta_geometry_rank_nudge import (
    _metric_context,
    _metrics,
    _prepare,
    _top10_mask,
)

KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
HALF_LIVES = (3.0, 7.0, 14.0)
BETAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)


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


def _recent_reliability(frame: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    selected = frame.loc[
        frame["selected_top10_balanced_composite"].fillna(False)
        & frame["ev_after_1pct"].notna()
    ].copy()
    selected["day"] = selected["__ts__"].dt.floor("D")
    daily = (
        selected.groupby(
            ["day", "side_name", "archetype_policy_key"], observed=True, sort=True
        )
        .agg(
            ev=("ev_after_1pct", "mean"),
            hit=("clean_exec", "mean"),
            rows=("ev_after_1pct", "size"),
        )
        .reset_index()
    )
    all_days = pd.date_range(
        frame["__ts__"].min().floor("D"),
        frame["__ts__"].max().floor("D"),
        freq="D",
        tz="UTC",
    )
    outputs: list[pd.DataFrame] = []
    for (side, archetype), part in daily.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        values = part.set_index("day").reindex(all_days)
        # Two-day delay is conservative for the <=12h execution horizon: all
        # contributing paths are resolved before they can affect a new day.
        ev_observed = values["ev"].shift(2)
        hit_observed = values["hit"].shift(2)
        ev_recent = ev_observed.ewm(
            halflife=float(half_life_days), min_periods=3, adjust=False, ignore_na=True
        ).mean()
        hit_recent = hit_observed.ewm(
            halflife=float(half_life_days), min_periods=3, adjust=False, ignore_na=True
        ).mean()
        ev_mean = ev_observed.expanding(min_periods=20).mean()
        hit_mean = hit_observed.expanding(min_periods=20).mean()
        ev_std = ev_observed.expanding(min_periods=20).std(ddof=0).clip(lower=1e-4)
        hit_std = hit_observed.expanding(min_periods=20).std(ddof=0).clip(lower=0.02)
        ev_z = ((ev_recent - ev_mean) / ev_std).clip(-3.0, 3.0)
        hit_z = ((hit_recent - hit_mean) / hit_std).clip(-3.0, 3.0)
        output = pd.DataFrame(
            {
                "day": all_days,
                "side_name": str(side),
                "archetype_policy_key": str(archetype),
                "recent_ev_z": ev_z.fillna(0.0).to_numpy(dtype=np.float32),
                "recent_hit_z": hit_z.fillna(0.0).to_numpy(dtype=np.float32),
            }
        )
        output["recent_reliability_z"] = (
            (0.70 * output["recent_ev_z"] + 0.30 * output["recent_hit_z"])
            .clip(-3.0, 3.0)
            .astype(np.float32)
        )
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def _loss_autocorr(frame: pd.DataFrame, selected: np.ndarray) -> float:
    part = frame.loc[
        selected
        & frame["ev_after_1pct"].notna().to_numpy(dtype=bool)
        & frame["side_name"].astype(str).str.lower().eq("short").to_numpy()
        & frame["archetype_policy_key"]
        .astype(str)
        .str.contains("breakout", case=False, na=False)
        .to_numpy()
    ].copy()
    daily = part.groupby(part["__ts__"].dt.floor("D"), observed=True)[
        "ev_after_1pct"
    ].mean()
    loss = daily.lt(0.0).astype(np.float32)
    return float(loss.autocorr(1)) if len(loss) >= 3 else np.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--nudge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(
        args.ablation_dir / "cross_sectional_geometry_predictions.parquet",
        columns=KEYS
        + [
            "calendar_month",
            "evaluation_scope",
            "ev_after_1pct",
            "clean_exec",
            "full_path_bad_mae_1r",
            "timeout",
        ],
    )
    balanced = pd.read_parquet(
        args.nudge_dir / "balanced_composite_predictions.parquet"
    )
    frame = source.merge(
        balanced,
        on=KEYS + ["calendar_month", "evaluation_scope"],
        how="inner",
        validate="one_to_one",
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)
    reliability_by_half_life = {
        half_life: _recent_reliability(frame, half_life) for half_life in HALF_LIVES
    }
    search_rows: list[dict[str, Any]] = []
    historical = frame[
        frame["evaluation_scope"].eq("historical_walkforward_oos")
    ].copy()
    historical, historical_offsets = _prepare(historical)
    historical_context = _metric_context(historical)
    baseline_selected = (
        historical["selected_top10_balanced_composite"]
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    baseline_metrics = _metrics(historical_context, baseline_selected)
    baseline_loss_ac = _loss_autocorr(historical, baseline_selected)
    for half_life, reliability in reliability_by_half_life.items():
        probe = historical.merge(
            reliability,
            left_on=["day", "side_name", "archetype_policy_key"],
            right_on=["day", "side_name", "archetype_policy_key"],
            how="left",
            validate="many_to_one",
        )
        reliability_z = (
            probe["recent_reliability_z"].fillna(0.0).to_numpy(dtype=np.float32)
        )
        base_score = probe["score_balanced_composite"].to_numpy(dtype=np.float32)
        for beta in BETAS:
            score = np.clip(base_score + float(beta) * reliability_z, 0.0, 1.0).astype(
                np.float32
            )
            selected = _top10_mask(score, historical_offsets)
            metrics = _metrics(historical_context, selected)
            loss_ac = _loss_autocorr(probe, selected)
            search_rows.append(
                {
                    "half_life_days": half_life,
                    "beta": beta,
                    **metrics,
                    "short_breakout_loss_autocorr": loss_ac,
                    "mean_ev_delta": metrics["mean_ev"] - baseline_metrics["mean_ev"],
                    "worst_day_delta": metrics["worst_day_ev"]
                    - baseline_metrics["worst_day_ev"],
                    "worst_week_delta": metrics["worst_week_ev"]
                    - baseline_metrics["worst_week_ev"],
                    "breakout_ev_delta": metrics["breakout_ev"]
                    - baseline_metrics["breakout_ev"],
                    "loss_autocorr_delta": loss_ac - baseline_loss_ac,
                    "reliability_objective": metrics["objective"]
                    - 0.0010 * max(float(loss_ac), 0.0),
                }
            )
    search = pd.DataFrame(search_rows)
    safe = search.loc[
        search["mean_ev_delta"].ge(0.0)
        & search["worst_day_delta"].ge(-0.0005)
        & search["worst_week_delta"].ge(-0.00025)
        & search["breakout_ev_delta"].ge(0.0)
        & search["loss_autocorr_delta"].le(0.0)
    ]
    if safe.empty:
        safe = search
    best = (
        safe.sort_values(
            ["reliability_objective", "mean_ev", "worst_day_ev"],
            ascending=False,
            kind="stable",
        )
        .iloc[0]
        .to_dict()
    )
    search.to_csv(
        args.output_dir / "historical_oos_reliability_search.csv", index=False
    )

    july = frame[frame["evaluation_scope"].eq("july_oos")].copy()
    july, july_offsets = _prepare(july)
    july_context = _metric_context(july)
    reliability = reliability_by_half_life[float(best["half_life_days"])]
    july_probe = july.merge(
        reliability,
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
        validate="many_to_one",
    )
    baseline_july = (
        july_probe["selected_top10_balanced_composite"]
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    score = np.clip(
        july_probe["score_balanced_composite"].to_numpy(dtype=np.float32)
        + float(best["beta"])
        * july_probe["recent_reliability_z"].fillna(0.0).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    ).astype(np.float32)
    selected_july = _top10_mask(score, july_offsets)
    baseline_july_metrics = _metrics(july_context, baseline_july)
    july_metrics = _metrics(july_context, selected_july)
    scorecard = pd.DataFrame(
        [
            {
                "scope": "historical_tuning",
                "selector": "balanced_composite_v1",
                **baseline_metrics,
                "short_breakout_loss_autocorr": baseline_loss_ac,
            },
            {"scope": "historical_tuning", "selector": "reliability_overlay", **best},
            {
                "scope": "july_untouched",
                "selector": "balanced_composite_v1",
                **baseline_july_metrics,
                "short_breakout_loss_autocorr": _loss_autocorr(
                    july_probe, baseline_july
                ),
            },
            {
                "scope": "july_untouched",
                "selector": "reliability_overlay",
                **july_metrics,
                "short_breakout_loss_autocorr": _loss_autocorr(
                    july_probe, selected_july
                ),
            },
        ]
    )
    scorecard.to_csv(args.output_dir / "scorecard.csv", index=False)
    manifest = {
        "schema": "meta_geometry_reliability_overlay_v1",
        "selected_on": "historical walk-forward OOS predictions only",
        "july_role": "untouched evaluation with causal day-by-day updates",
        "outcome_availability_lag_days": 2,
        "selected_parameters": best,
        "selection_contract": "fixed global top-10 activity",
        "cost_contract": "ev_after_1pct includes 1% round-trip cost",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(scorecard.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
