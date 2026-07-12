#!/usr/bin/env python3
"""Evaluate the historical-walk-forward sparse shock overlay."""

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

from scripts.report_meta_residual_archetype_final import (  # noqa: E402
    _autocorr_components,
    _calendar_components_preselected,
    _event_table,
    _group_metrics,
    _true_monday_week_start,
)
from scripts.run_meta_residual_sparse_shock_composite import ARM, CHAMPION  # noqa: E402
from scripts.run_train_meta_residual_archetype_enhancement import (
    DEFAULT_OUT_DIR,  # noqa: E402
)

CURRENT = "current_reference"


def _selected(frame: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    return frame.loc[pd.to_numeric(frame[rank_col], errors="coerce").ge(0.90)].copy()


def _summary(selected: pd.DataFrame, selector: str) -> dict[str, Any]:
    selected = selected.copy()
    selected["week_start"] = _true_monday_week_start(selected["__ts__"])
    weekly = selected.groupby("week_start", sort=True)["ev_after_1pct"].mean()
    monthly = selected.groupby("calendar_month", sort=True)["ev_after_1pct"].mean()
    return {
        "selector": selector,
        "selected_rows": int(len(selected)),
        "mean_ev_after_1pct": float(selected["ev_after_1pct"].mean()),
        "clean_exec_precision": float(selected["clean_exec"].mean()),
        "dirty_positive_rate": float(selected["dirty_positive"].mean()),
        "first_touch_bad_mae_rate": float(selected["first_touch_bad_mae_1r"].mean()),
        "full_path_bad_mae_rate": float(selected["full_path_bad_mae_1r"].mean()),
        "timeout_rate": float(selected["timeout"].mean()),
        "worst_week_ev": float(weekly.min()),
        "worst_month_ev": float(monthly.min()),
        "positive_weeks": int(weekly.gt(0.0).sum()),
        "weeks": int(len(weekly)),
    }


def _daily(selected: pd.DataFrame, probability_col: str, arm: str) -> pd.DataFrame:
    return _calendar_components_preselected(selected, prob_col=probability_col, arm=arm)


def _comparison(current_daily: pd.DataFrame, final_daily: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "side_name", "archetype_policy_key"]
    left = current_daily.rename(
        columns={
            "rows": "rows_base",
            "hit_rate": "hit_rate_base",
            "mean_ev_after_1pct": "mean_ev_after_1pct_base",
            "signed_surprise": "mean_hit_surprise_base",
        }
    )[
        keys
        + [
            "rows_base",
            "hit_rate_base",
            "mean_ev_after_1pct_base",
            "mean_hit_surprise_base",
        ]
    ]
    right = final_daily.rename(
        columns={
            "rows": "rows_alt",
            "hit_rate": "hit_rate_alt",
            "mean_ev_after_1pct": "mean_ev_after_1pct_alt",
            "signed_surprise": "mean_hit_surprise_alt",
        }
    )[
        keys
        + [
            "rows_alt",
            "hit_rate_alt",
            "mean_ev_after_1pct_alt",
            "mean_hit_surprise_alt",
        ]
    ]
    out = left.merge(right, on=keys, how="outer")
    for col in (
        "rows_base",
        "hit_rate_base",
        "mean_ev_after_1pct_base",
        "mean_hit_surprise_base",
        "rows_alt",
        "hit_rate_alt",
        "mean_ev_after_1pct_alt",
        "mean_hit_surprise_alt",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    thresholds = out.groupby(["side_name", "archetype_policy_key"], sort=False)[
        "mean_hit_surprise_base"
    ].transform(lambda x: float(np.nanquantile(np.abs(x), 0.90)))
    out["baseline_tail_threshold"] = thresholds
    out["baseline_high_surprise"] = out["mean_hit_surprise_base"].abs().ge(thresholds)
    out["surprise_abs_improvement"] = (
        out["mean_hit_surprise_base"].abs() - out["mean_hit_surprise_alt"].abs()
    )
    out["ev_delta"] = out["mean_ev_after_1pct_alt"] - out["mean_ev_after_1pct_base"]
    out["high_surprise_significantly_improved"] = out["surprise_abs_improvement"].ge(
        0.20 * out["mean_hit_surprise_base"].abs()
    )
    return out.sort_values(keys, kind="stable")


def _bootstrap(
    frame: pd.DataFrame, left_rank: str, right_rank: str, draws: int = 2_000
) -> dict[str, Any]:
    left = _selected(frame, left_rank)
    right = _selected(frame, right_rank)
    left["week"] = _true_monday_week_start(left["__ts__"])
    right["week"] = _true_monday_week_start(right["__ts__"])
    left_week = left.groupby("week", sort=True)["ev_after_1pct"].mean()
    right_week = right.groupby("week", sort=True)["ev_after_1pct"].mean()
    aligned = pd.concat(
        [left_week.rename("left"), right_week.rename("right")], axis=1
    ).dropna()
    delta = (aligned["right"] - aligned["left"]).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(20260711)
    boot = np.mean(delta[rng.integers(0, len(delta), size=(draws, len(delta)))], axis=1)
    return {
        "weeks": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci025": float(np.quantile(boot, 0.025)),
        "ci975": float(np.quantile(boot, 0.975)),
        "positive_probability": float(np.mean(boot > 0.0)),
    }


def main() -> None:
    root = DEFAULT_OUT_DIR
    arm_dir = root / ARM
    report_dir = root / "final_report"
    frame = pd.read_parquet(arm_dir / "oos_predictions_historical_rank.parquet")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    current = _selected(frame, "historical_rank_current_reference")
    parent = _selected(frame, "historical_rank_alternative")
    final = _selected(frame, "historical_rank_adjusted")
    summaries = pd.DataFrame(
        [
            _summary(current, CURRENT),
            _summary(parent, CHAMPION),
            _summary(final, ARM),
        ]
    )
    summaries.to_csv(report_dir / "shock_overlay_top10_summary.csv", index=False)

    current_daily = _daily(current, "hit_prob_current_reference", CURRENT)
    parent_daily = _daily(parent, "hit_prob_alternative", CHAMPION)
    final_daily = _daily(final, "hit_prob_adjusted", ARM)
    calendars = pd.concat([current_daily, parent_daily, final_daily], ignore_index=True)
    autocorr = _autocorr_components(calendars)
    calendars.to_csv(
        report_dir / "shock_overlay_hit_surprise_calendar.csv", index=False
    )
    autocorr.to_csv(
        report_dir / "shock_overlay_hit_surprise_autocorrelation.csv", index=False
    )
    comparison = _comparison(current_daily, final_daily)
    comparison.to_csv(
        report_dir / "shock_overlay_high_surprise_comparison.csv", index=False
    )
    events = _event_table(comparison)
    events.to_csv(report_dir / "shock_overlay_high_surprise_events.csv", index=False)

    frame["week_start"] = _true_monday_week_start(frame["__ts__"])
    breakdowns: list[pd.DataFrame] = []
    for name, selected in ((CURRENT, current), (CHAMPION, parent), (ARM, final)):
        selected = selected.copy()
        selected["week_start"] = _true_monday_week_start(selected["__ts__"])
        for scope, cols in (
            ("month", ["calendar_month"]),
            ("week", ["week_start"]),
            ("side_archetype", ["side_name", "archetype_policy_key"]),
            (
                "month_side_archetype",
                ["calendar_month", "side_name", "archetype_policy_key"],
            ),
        ):
            table = _group_metrics(selected, cols, name)
            table["scope"] = scope
            breakdowns.append(table)
    pd.concat(breakdowns, ignore_index=True).to_csv(
        report_dir / "shock_overlay_breakdowns.csv", index=False
    )

    event_tail = events[events["baseline_signed_surprise"].abs().gt(0.0)]
    unimproved = event_tail.loc[~event_tail["improved_either_way"]]
    material_unimproved = unimproved.loc[unimproved["material_persistent_event"]]
    current_ac = autocorr[autocorr["arm"].eq(CURRENT)]
    parent_ac = autocorr[autocorr["arm"].eq(CHAMPION)]
    final_ac = autocorr[autocorr["arm"].eq(ARM)]
    manifest = {
        "schema": "meta_residual_shock_champion_report_v1",
        "arm": ARM,
        "parent": CHAMPION,
        "current_to_final_weekly_bootstrap": _bootstrap(
            frame, "historical_rank_current_reference", "historical_rank_adjusted"
        ),
        "parent_to_final_weekly_bootstrap": _bootstrap(
            frame, "historical_rank_alternative", "historical_rank_adjusted"
        ),
        "mean_abs_signed_autocorr": {
            CURRENT: float(current_ac["signed_surprise_autocorr_lag1"].abs().mean()),
            CHAMPION: float(parent_ac["signed_surprise_autocorr_lag1"].abs().mean()),
            ARM: float(final_ac["signed_surprise_autocorr_lag1"].abs().mean()),
        },
        "high_surprise_cells": int(comparison["baseline_high_surprise"].sum()),
        "high_surprise_cells_significantly_improved": int(
            comparison.loc[
                comparison["baseline_high_surprise"],
                "high_surprise_significantly_improved",
            ].sum()
        ),
        "unimproved_events": int(len(unimproved)),
        "material_persistent_unimproved_events": int(len(material_unimproved)),
        "current_model_overwritten": False,
    }
    (report_dir / "shock_overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
