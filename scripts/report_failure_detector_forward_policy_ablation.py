#!/usr/bin/env python3
"""Evaluate frozen failure-detector alerts on matured deployed-policy exits.

``policy_exit`` and ``timeout_close`` rows are realized exits. Only
``marked_at_symbol_cutoff`` rows stay out of the economic ablation because
their eventual exit is still unknown. A date without a frozen detector score
is reported as unavailable, never as a negative alert. This prevents
accidental look-ahead or optimistic coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPLAY = Path("data_perp/reports/deployed_policy_replay_last3d_20260719_v1")
DEFAULT_DETECTOR = Path("data_perp/reports/prospective_failure_mode_detection_20260719_v7_three_year")
DEFAULT_OUTPUT = Path("data_perp/reports/failure_detector_forward_policy_ablation_20260719_v1")


def _day(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce").dt.floor("D")


def _unprefix_archetype(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"^(long|short)__", "", regex=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    replay_dir, detector_dir, output = Path(args.replay), Path(args.detector), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trades = pd.read_parquet(replay_dir / "trade_replay.parquet")
    candidates = pd.read_parquet(
        replay_dir / "admitted_candidates.parquet",
        columns=["timestamp", "symbol", "side", "policy_archetype"],
    )
    trades["entry_ts_utc"] = pd.to_datetime(trades["entry_ts_utc"], utc=True)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True)
    work = trades.merge(
        candidates,
        left_on=["entry_ts_utc", "symbol", "side"],
        right_on=["timestamp", "symbol", "side"],
        how="left",
        validate="one_to_one",
    )
    work["entry_day"] = _day(work["entry_day_utc"])
    work["archetype_policy_key"] = _unprefix_archetype(work["policy_archetype"])
    work["mature_exit"] = work["resolution"].isin(["policy_exit", "timeout_close"])

    detector = pd.read_parquet(
        detector_dir / "local_oos_predictions.parquet",
        columns=["day", "side_name", "archetype_policy_key", "failure_mode", "target_horizon_days", "risk", "threshold", "alert"],
    )
    detector["day"] = _day(detector["day"])
    detector["side_name"] = detector["side_name"].astype(str).str.lower()
    same_day = detector.loc[
        detector["failure_mode"].eq("negative_ev_day")
        & detector["target_horizon_days"].eq(0)
    ].copy()
    same_day = same_day.groupby(
        ["day", "side_name", "archetype_policy_key"], observed=True, as_index=False
    ).agg(risk=("risk", "max"), threshold=("threshold", "max"), alert=("alert", "max"))
    same_day = same_day.rename(columns={"day": "entry_day", "side_name": "side"})
    work = work.merge(same_day, on=["entry_day", "side", "archetype_policy_key"], how="left")
    work["detector_score_available"] = work["risk"].notna()
    work["alert"] = work["alert"].astype("boolean").fillna(False).astype(bool)
    work["risk_minus_threshold"] = pd.to_numeric(work["risk"], errors="coerce") - pd.to_numeric(work["threshold"], errors="coerce")
    mature = work.loc[work["mature_exit"]].copy()

    day_rows: list[dict[str, Any]] = []
    for day, part in work.groupby("entry_day", observed=True, sort=True):
        exits = part.loc[part["mature_exit"]]
        scored = exits.loc[exits["detector_score_available"]]
        alerted = scored.loc[scored["alert"]]
        retained = scored.loc[~scored["alert"]]
        day_rows.append(
            {
                "entry_day": day,
                "admitted_trades": int(len(part)),
                "mature_exits": int(len(exits)),
                "marked_unmatured": int((~part["mature_exit"]).sum()),
                "mature_net_ev_per_trade": float(exits["net_return_after_1pct"].mean()) if len(exits) else np.nan,
                "detector_score_coverage": float(exits["detector_score_available"].mean()) if len(exits) else np.nan,
                "detector_alerted_mature_exits": int(len(alerted)),
                "alerted_mature_ev_per_trade": float(alerted["net_return_after_1pct"].mean()) if len(alerted) else np.nan,
                "retained_mature_ev_per_trade_if_hard_gate": float(retained["net_return_after_1pct"].mean()) if len(retained) else np.nan,
                "hard_gate_delta_ev_per_trade": (
                    float(retained["net_return_after_1pct"].mean() - exits["net_return_after_1pct"].mean())
                    if len(retained) and len(exits) else np.nan
                ),
                "max_risk": float(scored["risk"].max()) if len(scored) else np.nan,
                "max_risk_minus_threshold": float(scored["risk_minus_threshold"].max()) if len(scored) else np.nan,
                "interpretation": (
                    "scored_no_alert"
                    if len(scored) and not len(alerted)
                    else "alert_ablation_available"
                    if len(alerted)
                    else "no_frozen_detector_score_for_entry_day"
                ),
            }
        )
    daily = pd.DataFrame(day_rows)
    work.sort_values(["entry_day", "entry_ts_utc", "symbol"]).to_csv(output / "forward_policy_trade_level.csv", index=False)
    daily.to_csv(output / "forward_policy_mature_exit_daily.csv", index=False)

    summary: dict[str, Any] = {
        "schema": "failure_detector_forward_policy_ablation_v1",
        "detector_contract": "frozen same-day negative_ev_day OOS outputs only",
        "maturity_contract": "policy_exit and timeout_close; only marked_at_symbol_cutoff paths are censored",
        "admitted_trades": int(len(work)),
        "mature_exits": int(len(mature)),
        "mature_detector_coverage": float(mature["detector_score_available"].mean()) if len(mature) else np.nan,
        "mature_alerted_exits": int(mature["alert"].sum()),
        "date_range": [str(work["entry_day"].min()), str(work["entry_day"].max())],
        "conclusion": (
            "No policy conclusion without score coverage. The frozen detector is evaluated only where it emitted a score; "
            "unscored later days remain an explicit deployment gap."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
