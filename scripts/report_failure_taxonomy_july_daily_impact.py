#!/usr/bin/env python3
"""Report July day-level diagnostic impact of the failure-state detector.

The detector is trained against labels from the frozen three-year taxonomy
backcast.  It is therefore deliberately *not* treated as a deployable gate in
this report.  Where strict meta-OOS outcomes exist, the report shows the
economic effect of removing detector-alerted side x archetype cells as a
cross-source diagnostic only.  This makes both useful questions visible:

* did the state detector identify the weaker selected cells on a given day?
* would a hard gate have discarded useful edge along with them?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STRICT = Path(
    "data_perp/reports/failure_taxonomy_strict_oos_sensitivity_20260719_v8_corrected"
)
DEFAULT_TAXONOMY = Path(
    "data_perp/reports/failure_episode_taxonomy_20260719_v17_three_year_taxonomy"
)
DEFAULT_DETECTOR = Path(
    "data_perp/reports/prospective_failure_mode_detection_20260719_v7_three_year"
)
DEFAULT_OUTPUT = Path("data_perp/reports/failure_taxonomy_july_daily_impact_20260719_v1")
KEYS = ["day", "side_name", "archetype_policy_key"]


def _utc_day(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce").dt.floor("D")


def _weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    values = pd.to_numeric(frame[value], errors="coerce")
    weights = pd.to_numeric(frame[weight], errors="coerce")
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _detector_slice(detector: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return one causal detector output per daily side x archetype cell."""
    work = detector.loc[
        detector["failure_mode"].eq(name) & detector["target_horizon_days"].eq(0)
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=KEYS + [f"{name}_risk", f"{name}_alert"])
    work["risk"] = pd.to_numeric(work["risk"], errors="coerce")
    work["alert"] = work["alert"].fillna(False).astype(bool)
    # The detector contract emits one row; max makes a future accidental
    # duplicate conservative without mixing any future labels into this report.
    result = work.groupby(KEYS, observed=True, as_index=False).agg(
        **{f"{name}_risk": ("risk", "max"), f"{name}_alert": ("alert", "max")}
    )
    return result


def _strict_daily_impact(strict_daily: pd.DataFrame, detector: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = _detector_slice(detector, "negative_ev_day")
    onset = _detector_slice(detector, "negative_ev_onset")
    health = strict_daily.merge(current, on=KEYS, how="left").merge(onset, on=KEYS, how="left")
    health["negative_ev_day_alert"] = health["negative_ev_day_alert"].fillna(False).astype(bool)
    health["negative_ev_onset_alert"] = health["negative_ev_onset_alert"].fillna(False).astype(bool)
    health["detector_available"] = health["negative_ev_day_risk"].notna()
    health["alerted_rows"] = np.where(health["negative_ev_day_alert"], health["rows"], 0)
    health["alerted_ev"] = np.where(health["negative_ev_day_alert"], health["sum_ev_after_1pct"], 0.0)

    rows: list[dict[str, Any]] = []
    for day, part in health.groupby("day", observed=True, sort=True):
        total_rows = int(part["rows"].sum())
        total_ev = float(pd.to_numeric(part["sum_ev_after_1pct"], errors="coerce").sum())
        alert_rows = int(part["alerted_rows"].sum())
        alert_ev = float(pd.to_numeric(part["alerted_ev"], errors="coerce").sum())
        retained_rows = total_rows - alert_rows
        retained_ev = total_ev - alert_ev
        mean_ev = total_ev / total_rows if total_rows else np.nan
        retained_mean = retained_ev / retained_rows if retained_rows else np.nan
        row = {
            "day": day,
            "strict_top10_rows": total_rows,
            "strict_top10_net_ev": total_ev,
            "strict_top10_net_ev_per_trade": mean_ev,
            "strict_negative_cells": int(part["negative_ev_day"].fillna(False).sum()),
            "strict_cells": int(len(part)),
            "detector_cell_coverage": float(part["detector_available"].mean()),
            "mean_negative_ev_day_risk": float(pd.to_numeric(part["negative_ev_day_risk"], errors="coerce").mean()),
            "max_negative_ev_day_risk": float(pd.to_numeric(part["negative_ev_day_risk"], errors="coerce").max()),
            "alerted_cells": int(part["negative_ev_day_alert"].sum()),
            "alerted_rows": alert_rows,
            "alerted_net_ev": alert_ev,
            "alerted_net_ev_per_trade": alert_ev / alert_rows if alert_rows else np.nan,
            "retained_rows_if_hard_gate": retained_rows,
            "retained_net_ev_if_hard_gate": retained_ev,
            "retained_net_ev_per_trade_if_hard_gate": retained_mean,
            "hard_gate_delta_net_ev_per_trade": retained_mean - mean_ev if retained_rows else np.nan,
            "hard_gate_delta_total_net_ev": retained_ev - total_ev,
            "onset_alerted_cells": int(part["negative_ev_onset_alert"].sum()),
        }
        if not alert_rows:
            row["diagnostic_interpretation"] = "no_detector_alert"
        elif alert_ev < 0:
            row["diagnostic_interpretation"] = "would_remove_negative_ev_cells"
        else:
            row["diagnostic_interpretation"] = "would_remove_positive_ev_cells"
        rows.append(row)
    daily = pd.DataFrame(rows)
    return health, daily


def _taxonomy_daily_context(taxonomy_daily: pd.DataFrame, detector: pd.DataFrame) -> pd.DataFrame:
    current = _detector_slice(detector, "negative_ev_day")
    onset = _detector_slice(detector, "negative_ev_onset")
    health = taxonomy_daily.merge(current, on=KEYS, how="left").merge(onset, on=KEYS, how="left")
    health["negative_ev_day_alert"] = health["negative_ev_day_alert"].fillna(False).astype(bool)
    health["negative_ev_onset_alert"] = health["negative_ev_onset_alert"].fillna(False).astype(bool)
    rows: list[dict[str, Any]] = []
    for day, part in health.groupby("day", observed=True, sort=True):
        rows.append(
            {
                "day": day,
                "taxonomy_rows": int(part["selected_rows"].sum()),
                "taxonomy_net_ev": float((part["mean_ev_after_cost"] * part["selected_rows"]).sum()),
                "taxonomy_net_ev_per_trade": _weighted_mean(part, "mean_ev_after_cost", "selected_rows"),
                "taxonomy_negative_cells": int(part["mean_ev_after_cost"].lt(0).sum()),
                "taxonomy_cells": int(len(part)),
                "taxonomy_detector_coverage": float(part["negative_ev_day_risk"].notna().mean()),
                "taxonomy_alerted_cells": int(part["negative_ev_day_alert"].sum()),
                "taxonomy_onset_alerted_cells": int(part["negative_ev_onset_alert"].sum()),
                "taxonomy_max_detector_risk": float(pd.to_numeric(part["negative_ev_day_risk"], errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    strict = pd.read_csv(Path(args.strict) / "strict_base_meta_oos_daily_health.csv")
    strict["day"] = _utc_day(strict["day"])
    strict = strict.loc[strict["day"].between(start, end)].copy()
    taxonomy_daily = pd.read_parquet(Path(args.taxonomy) / "daily_side_archetype_health.parquet")
    taxonomy_daily["day"] = _utc_day(taxonomy_daily["day"])
    taxonomy_daily = taxonomy_daily.loc[taxonomy_daily["day"].between(start, end)].copy()
    detector = pd.read_parquet(Path(args.detector) / "local_oos_predictions.parquet")
    detector["day"] = _utc_day(detector["day"])
    detector = detector.loc[detector["day"].between(start, end)].copy()

    strict_cells, strict_day = _strict_daily_impact(strict, detector)
    taxonomy_day = _taxonomy_daily_context(taxonomy_daily, detector)
    combined = pd.merge(strict_day, taxonomy_day, on="day", how="outer", validate="one_to_one").sort_values("day")
    strict_cells.sort_values(KEYS).to_csv(output / "strict_meta_oos_july_cell_impact.csv", index=False)
    strict_day.sort_values("day").to_csv(output / "strict_meta_oos_july_daily_impact.csv", index=False)
    taxonomy_day.sort_values("day").to_csv(output / "taxonomy_july_daily_context.csv", index=False)
    combined.to_csv(output / "july_daily_impact_combined.csv", index=False)

    summary = {
        "schema": "failure_taxonomy_july_daily_impact_v1",
        "period": {"start": str(start), "end": str(end)},
        "strict_meta_oos_contract": (
            "Strict meta-OOS economic rows are available only through 2026-07-10. "
            "The hard-gate calculation is a cross-source diagnostic: detector labels "
            "come from the frozen taxonomy backcast, not from this strict handoff."
        ),
        "taxonomy_contract": (
            "The taxonomy panel covers the wider selected-state universe through "
            "2026-07-17. It is diagnostic state context, not the strict meta policy universe."
        ),
        "strict_oos_days": int(strict_day["day"].nunique()),
        "taxonomy_days": int(taxonomy_day["day"].nunique()),
        "strict_total_rows": int(strict_day["strict_top10_rows"].sum()),
        "strict_total_net_ev": float(strict_day["strict_top10_net_ev"].sum()),
        "strict_total_net_ev_per_trade": float(
            strict_day["strict_top10_net_ev"].sum() / strict_day["strict_top10_rows"].sum()
        ) if len(strict_day) else np.nan,
        "hard_gate_alerted_rows": int(strict_day["alerted_rows"].sum()),
        "hard_gate_alerted_net_ev": float(strict_day["alerted_net_ev"].sum()),
        "hard_gate_retained_net_ev_per_trade": float(
            strict_day["retained_net_ev_if_hard_gate"].sum()
            / strict_day["retained_rows_if_hard_gate"].sum()
        ) if strict_day["retained_rows_if_hard_gate"].sum() else np.nan,
        "hard_gate_contract": "illustrative only; do not promote without a dedicated strict OOS ablation",
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-07-31")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
