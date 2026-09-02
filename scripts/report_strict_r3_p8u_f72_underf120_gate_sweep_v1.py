#!/usr/bin/env python3
"""Replay frozen P8U/F72/Under-F120 MC1 scores at multiple admission floors.

The MC1 maps are already fitted and target-free score coordinates are frozen.
This utility changes only the terminal dual-MC1 admission floor, runs the
existing chronological constrained auction once per floor, and reports the
resulting participation and lower-tail diagnostics.  It never fits a model,
touches a live artifact, or performs exchange I/O.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_aug27_20260828_v1/dual_predictions.parquet"
DEFAULT_FLOORS = (30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0)
QUANTILES = (0.05, 0.10, 0.15, 0.20)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _accepted_rows(decisions: pd.DataFrame) -> pd.DataFrame:
    work = decisions.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="raise")
    work["net_bps"] = _finite(work["position_net_return"]) * 10_000.0
    return work.loc[
        work["accepted"].fillna(False).astype(bool)
        & work["policy_outcome_available"].fillna(False).astype(bool)
        & work["net_bps"].notna()
    ].copy()


def _calendar_daily(accepted: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    daily = accepted.assign(day=accepted["timestamp"].dt.normalize()).groupby("day", sort=True)["net_bps"].agg(
        trades="size", net_ev_bps_per_trade="mean", net_total_bps="sum"
    )
    result = pd.DataFrame(index=dates).join(daily, how="left")
    result.index.name = "day"
    result["trades"] = result["trades"].fillna(0).astype(int)
    result["net_total_bps"] = result["net_total_bps"].fillna(0.0)
    return result.reset_index()


def _calendar_weekly(accepted: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_week = start.normalize() - pd.Timedelta(days=int(start.dayofweek))
    end_week = end.normalize() - pd.Timedelta(days=int(end.dayofweek))
    weeks = pd.date_range(start_week, end_week, freq="7D", tz="UTC")
    token = accepted["timestamp"].dt.normalize() - pd.to_timedelta(accepted["timestamp"].dt.dayofweek, unit="D")
    weekly = accepted.assign(week=token).groupby("week", sort=True)["net_bps"].agg(
        trades="size", net_ev_bps_per_trade="mean", net_total_bps="sum"
    )
    result = pd.DataFrame(index=weeks).join(weekly, how="left")
    result.index.name = "week"
    result["trades"] = result["trades"].fillna(0).astype(int)
    result["net_total_bps"] = result["net_total_bps"].fillna(0.0)
    return result.reset_index()


def _quantiles(series: pd.Series, prefix: str) -> dict[str, float]:
    valid = _finite(series).dropna()
    return {
        f"{prefix}_q{int(q * 100):02d}_net_ev_bps_per_trade": float(valid.quantile(q)) if len(valid) else float("nan")
        for q in QUANTILES
    }


def _participation(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict[str, object]:
    data: dict[str, object] = {
        "calendar_days": int(len(daily)),
        "active_days": int(daily["trades"].gt(0).sum()),
        "mean_trades_per_calendar_day": float(daily["trades"].mean()),
        "max_trades_per_day": int(daily["trades"].max()),
        "calendar_weeks": int(len(weekly)),
        "active_weeks": int(weekly["trades"].gt(0).sum()),
        "mean_trades_per_calendar_week": float(weekly["trades"].mean()),
        "max_trades_per_week": int(weekly["trades"].max()),
    }
    for cutoff in (1, 5, 10):
        data[f"days_lt_{cutoff}_trades"] = int(daily["trades"].lt(cutoff).sum())
        data[f"pct_days_lt_{cutoff}_trades"] = float(daily["trades"].lt(cutoff).mean())
        data[f"weeks_lt_{cutoff}_trades"] = int(weekly["trades"].lt(cutoff).sum())
        data[f"pct_weeks_lt_{cutoff}_trades"] = float(weekly["trades"].lt(cutoff).mean())
    data.update(_quantiles(daily.loc[daily["trades"].gt(0), "net_ev_bps_per_trade"], "daily"))
    data.update(_quantiles(weekly.loc[weekly["trades"].gt(0), "net_ev_bps_per_trade"], "weekly"))
    return data


def _period_token(frame: pd.DataFrame) -> str:
    timestamps = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return f"{timestamps.min():%Y%m}_{timestamps.max():%Y%m}"


def _raw_admission_summary(frame: pd.DataFrame, floor: float) -> dict[str, object]:
    current = _finite(frame["current_mc1_expected_bps"])
    bcf = _finite(frame["bcf_mc1_expected_bps"])
    outcome = _finite(frame["policy_net_bps"])
    admitted = current.ge(floor) & bcf.ge(floor) & outcome.notna()
    selected = frame.loc[admitted].copy()
    net = _finite(selected["policy_net_bps"])
    return {
        "raw_dual_admitted_rows": int(len(selected)),
        "raw_dual_admission_net_ev_bps_per_trade": float(net.mean()) if len(net) else float("nan"),
        "raw_dual_admission_total_net_bps": float(net.sum()) if len(net) else 0.0,
        "raw_dual_admission_positive_fraction": float(net.gt(0).mean()) if len(net) else float("nan"),
    }


def _run_floor(
    frame: pd.DataFrame,
    floor: float,
    capacity: int,
    out: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    # Reuse the one source of truth for dual admission, candidate normalisation,
    # auction constraints, and wallet path.  The global is restored even on an
    # error so this offline report cannot perturb another research process.
    import run_strict_r3_enhanced_base_live_stack_challenger as parent

    previous = parent.MC1_THRESHOLD_BPS
    period = _period_token(frame)
    label = f"gate_{int(floor):02d}_cap{int(capacity)}"
    try:
        parent.MC1_THRESHOLD_BPS = float(floor)
        metric = parent._portfolio_metrics(
            frame,
            label,
            period,
            out,
            max_new_entries_per_bar=int(capacity),
        )
    finally:
        parent.MC1_THRESHOLD_BPS = previous
    decisions = pd.read_parquet(out / f"{label}_{period}_decisions.parquet")
    accepted = _accepted_rows(decisions)
    start = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise").min()
    end = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise").max()
    daily = _calendar_daily(accepted, start, end)
    weekly = _calendar_weekly(accepted, start, end)
    summary: dict[str, object] = {
        "dual_mc1_floor_bps": float(floor),
        "max_new_entries_per_bar": int(capacity),
        **_raw_admission_summary(frame, floor),
        **metric,
        **_participation(daily, weekly),
    }
    summary["daily_q_definition"] = "quantile of UTC-calendar-day mean net EV/trade over active days"
    summary["weekly_q_definition"] = "quantile of UTC-calendar-week mean net EV/trade over active weeks"
    summary["outcome_contract"] = "frozen rich policy net bps; fixed 100-bps round-trip policy cost embedded exactly once"
    return summary, daily, weekly


def _markdown(summary: pd.DataFrame) -> str:
    columns = [
        "dual_mc1_floor_bps", "max_new_entries_per_bar", "candidate_admitted_rows", "accepted_rows",
        "raw_dual_admitted_rows", "raw_dual_admission_net_ev_bps_per_trade",
        "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps",
        "worst_week_bps", "max_drawdown", "mean_trades_per_calendar_day",
        "max_trades_per_day", "days_lt_1_trades", "days_lt_5_trades", "days_lt_10_trades",
        "daily_q05_net_ev_bps_per_trade", "daily_q10_net_ev_bps_per_trade",
        "daily_q15_net_ev_bps_per_trade", "daily_q20_net_ev_bps_per_trade",
        "weekly_q05_net_ev_bps_per_trade", "weekly_q10_net_ev_bps_per_trade",
        "weekly_q15_net_ev_bps_per_trade", "weekly_q20_net_ev_bps_per_trade",
    ]
    frame = summary.loc[:, columns].copy()
    lines = [
        "# P8U/F72/Under-F120 gate-only constrained replay",
        "",
        "Frozen target-free scores and already-fitted MC1 maps; only the common dual-MC1 floor changes. "
        "Every arm uses the same chronological global auction and rich policy outcomes.",
        "",
        "## Aggregate and participation diagnostics",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append("" if not np.isfinite(value) else f"{float(value):.3f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    lines.extend([
        "",
        "Daily and weekly Q05/Q10/Q15/Q20 are lower-tail quantiles of active-period mean EV per trade; "
        "days with no entries are separately captured by the participation columns.",
        "",
    ])
    return "\n".join(lines)


def run(
    input_path: Path,
    floors: Iterable[float],
    capacities: Iterable[int],
    out: Path,
    *,
    end_exclusive: pd.Timestamp | None = None,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    floors = tuple(sorted({float(value) for value in floors}))
    capacities = tuple(sorted({int(value) for value in capacities}))
    if not floors or any(value <= 0.0 for value in floors):
        raise ValueError("floors must be non-empty positive bps values")
    if not capacities or any(value <= 0 for value in capacities):
        raise ValueError("capacities must be non-empty positive integers")
    out.mkdir(parents=True)
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "enhanced_base_routed",
        "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
        "current_final_score", "bcf_final_score", "current_mc1_expected_bps", "bcf_mc1_expected_bps",
    ]
    frame = pd.read_parquet(input_path, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if end_exclusive is not None:
        frame = frame.loc[frame["__decision_ts__"].lt(end_exclusive)].copy()
        if frame.empty:
            raise ValueError("end_exclusive excludes every score row")
    if frame.duplicated(["candidate_id", "__decision_ts__"]).any():
        raise AssertionError("frozen dual-MC1 input has duplicate candidate identity")
    forbidden = {"policy_label_available_ts"}.difference(frame.columns)
    if forbidden:
        raise AssertionError("frozen dual-MC1 input lacks label-availability provenance")
    summaries: list[dict[str, object]] = []
    for floor in floors:
        for capacity in capacities:
            summary, daily, weekly = _run_floor(frame, floor, capacity, out)
            suffix = f"{int(floor):02d}bps_cap{int(capacity)}"
            daily.to_parquet(out / f"daily_{suffix}.parquet", index=False, compression="zstd")
            weekly.to_parquet(out / f"weekly_{suffix}.parquet", index=False, compression="zstd")
            summaries.append(summary)
    summary = pd.DataFrame(summaries).sort_values(["dual_mc1_floor_bps", "max_new_entries_per_bar"], kind="stable").reset_index(drop=True)
    summary.to_parquet(out / "gate_summary.parquet", index=False, compression="zstd")
    (out / "GATE_SWEEP_RECEIPT.md").write_text(_markdown(summary))
    manifest = {
        "schema": "strict_r3_p8u_f72_underf120_gate_sweep_v1",
        "scope": "offline gate-only replay; frozen MC1 predictions; no fitting, live mutation, or exchange I/O",
        "input": str(input_path), "input_sha256": _sha256(input_path),
        "floors_bps": list(floors), "max_new_entries_per_bar": list(capacities),
        "evaluation": f"{frame['__decision_ts__'].min()} through {frame['__decision_ts__'].max()}",
        "end_exclusive": None if end_exclusive is None else str(end_exclusive),
        "portfolio": "existing global chronological auction / same rich policy outcome contract",
        "cost": "policy_net_bps embeds the 100-bps round-trip policy cost exactly once",
        "outputs": ["gate_summary.parquet", "daily_*bps.parquet", "weekly_*bps.parquet", "GATE_SWEEP_RECEIPT.md"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--floors", default=",".join(str(value) for value in DEFAULT_FLOORS))
    parser.add_argument("--capacities", default="1,2,3,4")
    parser.add_argument("--end-exclusive", type=str, default=None,
                        help="optional UTC decision cutoff; rows at/after it are excluded from terminal replay")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    floors = tuple(float(value.strip()) for value in args.floors.split(",") if value.strip())
    capacities = tuple(int(value.strip()) for value in args.capacities.split(",") if value.strip())
    cutoff = pd.Timestamp(args.end_exclusive, tz="UTC") if args.end_exclusive else None
    print(run(args.input.resolve(), floors, capacities, args.out.resolve(), end_exclusive=cutoff))


if __name__ == "__main__":
    main()
