#!/usr/bin/env python3
"""Detailed offline quality receipt for the frozen P8U/F72/Under-F120 sweep.

This reader consumes a completed gate/capacity sweep only.  It never fits or
scores a model, alters source data, or accesses an exchange.  Outcome labels
are already embedded in the immutable policy replay decisions; raw admission
diagnostics are recomputed from the corresponding frozen MC1 ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = ROOT / "data_perp/artifacts/strict_r3_p8u_f72_underf120_gate_capacity_sweep_aug27_20260828_v2"
DEFAULT_MC1 = ROOT / "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_aug27_20260828_v1/dual_predictions.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _equity_risk(equity: pd.DataFrame) -> dict[str, object]:
    work = equity.loc[:, ["timestamp", "wallet"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["wallet"] = pd.to_numeric(work["wallet"], errors="coerce")
    work = work.dropna().sort_values("timestamp", kind="stable")
    if len(work) < 2:
        return {"max_drawdown": float("nan"), "sortino_weekly_annualized": float("nan"), "weekly_q05_wallet_return_pct": float("nan"), "weekly_mad_wallet_return_pct": float("nan"), "weekly_std_wallet_return_pct": float("nan"), "weeks": 0}
    drawdown = work.wallet / work.wallet.cummax() - 1.0
    work["week"] = work.timestamp.dt.normalize() - pd.to_timedelta(work.timestamp.dt.dayofweek, unit="D")
    weekly = work.groupby("week", sort=True).wallet.agg(["first", "last"])
    ret = ((weekly["last"] / weekly["first"] - 1.0) * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return {"max_drawdown": float(drawdown.min()), "sortino_weekly_annualized": float("nan"), "weekly_q05_wallet_return_pct": float("nan"), "weekly_mad_wallet_return_pct": float("nan"), "weekly_std_wallet_return_pct": float("nan"), "weeks": 0}
    downside = np.minimum(ret.to_numpy(float), 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    result: dict[str, object] = {
        "max_drawdown": float(drawdown.min()),
        "sortino_weekly_annualized": float(ret.mean() / downside_dev * np.sqrt(52.0)) if downside_dev > 1e-12 else float("nan"),
        "weekly_q05_wallet_return_pct": float(ret.quantile(.05)),
        "weekly_mad_wallet_return_pct": float(np.median(np.abs(ret - ret.median()))),
        "weekly_std_wallet_return_pct": float(ret.std(ddof=0)),
        "weeks": int(len(ret)),
        "negative_weeks": int(ret.lt(0.0).sum()),
    }
    for multiplier in (0.5, 1.0, 1.5, 2.0):
        result[f"weeks_below_mean_minus_{multiplier:.1f}std"] = int(ret.lt(ret.mean() - multiplier * ret.std(ddof=0)).sum())
    return result


def _accepted(decisions: pd.DataFrame) -> pd.DataFrame:
    work = decisions.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="raise")
    work["net_bps"] = pd.to_numeric(work["position_net_return"], errors="coerce") * 10_000.0
    return work.loc[
        work["accepted"].fillna(False).astype(bool)
        & work["policy_outcome_available"].fillna(False).astype(bool)
        & np.isfinite(work["net_bps"])
    ].copy()


def _calendar_metrics(accepted: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar_days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    by_day = accepted.assign(day=accepted.timestamp.dt.normalize()).groupby("day", sort=True).net_bps.agg(
        trades="size", net_ev_bps_per_trade="mean", net_total_bps="sum"
    )
    daily = pd.DataFrame(index=calendar_days).join(by_day, how="left").reset_index(names="day")
    daily["trades"] = daily.trades.fillna(0).astype(int)
    daily["net_total_bps"] = daily.net_total_bps.fillna(0.0)
    daily["month"] = daily.day.dt.strftime("%Y-%m")
    daily["quarter"] = daily.day.dt.to_period("Q").astype(str)
    daily["year"] = daily.day.dt.year.astype(int)

    work = accepted.assign(
        week=accepted.timestamp.dt.normalize() - pd.to_timedelta(accepted.timestamp.dt.dayofweek, unit="D")
    )
    by_week = work.groupby("week", sort=True).net_bps.agg(trades="size", net_ev_bps_per_trade="mean", net_total_bps="sum")
    calendar_weeks = pd.date_range(
        start.normalize() - pd.Timedelta(days=int(start.dayofweek)),
        end.normalize() - pd.Timedelta(days=int(end.dayofweek)),
        freq="7D", tz="UTC",
    )
    weekly = pd.DataFrame(index=calendar_weeks).join(by_week, how="left").reset_index(names="week")
    weekly["trades"] = weekly.trades.fillna(0).astype(int)
    weekly["net_total_bps"] = weekly.net_total_bps.fillna(0.0)
    return daily, weekly


def _temporal_summary(accepted: pd.DataFrame, equity: pd.DataFrame, *, label: str, floor: float, capacity: int) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if accepted.empty:
        raise AssertionError("no accepted policy-resolved rows")
    start, end = accepted.timestamp.min(), accepted.timestamp.max()
    daily, weekly = _calendar_metrics(accepted, start=start, end=end)
    groups: list[tuple[str, pd.Series]] = [
        ("global", pd.Series(True, index=accepted.index)),
        ("year", accepted.timestamp.dt.year.astype(str)),
        ("quarter", accepted.timestamp.dt.to_period("Q").astype(str)),
        ("month", accepted.timestamp.dt.strftime("%Y-%m")),
    ]
    rows: list[dict[str, object]] = []
    for kind, membership in groups:
        grouped = [("all", accepted)] if kind == "global" else accepted.groupby(membership, sort=True)
        for period, part in grouped:
            days = pd.date_range(part.timestamp.min().normalize(), part.timestamp.max().normalize(), freq="D", tz="UTC")
            rows.append({
                "scope": kind, "period": str(period), "dual_mc1_floor_bps": floor, "max_new_entries_per_bar": capacity,
                "entries": int(len(part)), "calendar_days": int(len(days)), "trades_per_calendar_day": float(len(part) / len(days)),
                "net_ev_bps_per_trade": float(part.net_bps.mean()), "net_total_bps": float(part.net_bps.sum()),
                "positive_trade_fraction": float(part.net_bps.gt(0).mean()), "precision_gt50": float(part.net_bps.gt(50).mean()),
                "worst_trade_bps": float(part.net_bps.min()), "best_trade_bps": float(part.net_bps.max()),
            })
    temporal = pd.DataFrame(rows)
    risk = _equity_risk(equity)
    global_row = temporal.loc[temporal.scope.eq("global")].iloc[0].to_dict()
    global_row.update({
        "label": label,
        **risk,
        "worst_month_bps": float(temporal.loc[temporal.scope.eq("month"), "net_ev_bps_per_trade"].min()),
        "worst_week_bps": float(weekly.loc[weekly.trades.gt(0), "net_ev_bps_per_trade"].min()),
        "active_days": int(daily.trades.gt(0).sum()),
        "days_lt_1_trades": int(daily.trades.lt(1).sum()),
        "days_lt_5_trades": int(daily.trades.lt(5).sum()),
        "days_lt_10_trades": int(daily.trades.lt(10).sum()),
        "max_trades_per_day": int(daily.trades.max()),
        "q05_week_net_ev_bps_per_trade": float(weekly.loc[weekly.trades.gt(0), "net_ev_bps_per_trade"].quantile(.05)),
    })
    return global_row, temporal, daily, weekly


def _raw_monthly(mc1: pd.DataFrame, floor: float) -> pd.DataFrame:
    work = mc1.copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    net = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    admitted = (
        pd.to_numeric(work["current_mc1_expected_bps"], errors="coerce").ge(floor)
        & pd.to_numeric(work["bcf_mc1_expected_bps"], errors="coerce").ge(floor)
        & work["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(net)
    )
    work = work.loc[admitted].copy()
    work["policy_net_bps"] = net.loc[admitted].to_numpy(float)
    work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    result = work.groupby("month", sort=True).policy_net_bps.agg(
        raw_dual_admitted_rows="size", raw_dual_admission_net_ev_bps_per_trade="mean", raw_dual_admission_total_net_bps="sum"
    ).reset_index()
    result["dual_mc1_floor_bps"] = float(floor)
    return result


def _markdown(summary: pd.DataFrame, temporal: pd.DataFrame, august: pd.DataFrame, raw: pd.DataFrame) -> str:
    def table(frame: pd.DataFrame, columns: list[str], rows: int = 100) -> list[str]:
        view = frame.loc[:, [column for column in columns if column in frame]].head(rows)
        lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
        for row in view.itertuples(index=False, name=None):
            rendered = [f"{float(value):.3f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else ("" if isinstance(value, (float, np.floating)) else str(value)) for value in row]
            lines.append("| " + " | ".join(rendered) + " |")
        return lines
    canonical = summary.loc[(summary.dual_mc1_floor_bps.eq(50.0)) & (summary.max_new_entries_per_bar.eq(2))]
    lines = [
        "# P8U/F72/Under-F120 extended quality receipt", "",
        "Offline research only. The report uses precomputed target-free scores and strict-prequential MC1 maps. It joins rich-policy outcomes only for evaluation. The frozen rich policy embeds its 100-bps round-trip cost exactly once.", "",
        "## Gate and capacity summary", "",
        *table(summary, ["dual_mc1_floor_bps", "max_new_entries_per_bar", "raw_dual_admitted_rows", "raw_dual_admission_net_ev_bps_per_trade", "entries", "net_ev_bps_per_trade", "net_total_bps", "worst_month_bps", "worst_week_bps", "max_drawdown", "sortino_weekly_annualized", "trades_per_calendar_day", "days_lt_1_trades", "days_lt_5_trades", "days_lt_10_trades"]), "",
        "## Frozen 50-bps / two-entry contract: temporal metrics", "",
        *table(temporal.loc[(temporal.dual_mc1_floor_bps.eq(50.0)) & (temporal.max_new_entries_per_bar.eq(2))], ["scope", "period", "entries", "trades_per_calendar_day", "net_ev_bps_per_trade", "net_total_bps", "positive_trade_fraction", "precision_gt50", "worst_trade_bps", "best_trade_bps"]), "",
        "## August 1–27 2026: constrained daily outcome", "",
        *table(august, ["day", "trades", "net_ev_bps_per_trade", "net_total_bps"]), "",
        "## August raw dual-admission conversion (before portfolio constraints)", "",
        *table(raw, ["month", "dual_mc1_floor_bps", "raw_dual_admitted_rows", "raw_dual_admission_net_ev_bps_per_trade", "raw_dual_admission_total_net_bps"]), "",
        "The 50-bps/two-entry row is a frozen control, not a threshold selection from this report. August is a reconciliation extension only, not a promotion period.", "",
    ]
    if canonical.empty:
        raise AssertionError("canonical 50-bps/two-entry row missing")
    return "\n".join(lines)


def run(sweep: Path, mc1_path: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    summary_rows: list[dict[str, object]] = []
    temporal_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    weekly_rows: list[pd.DataFrame] = []
    gate = pd.read_parquet(sweep / "gate_summary.parquet")
    for row in gate.itertuples(index=False):
        floor, capacity = float(row.dual_mc1_floor_bps), int(row.max_new_entries_per_bar)
        suffix = f"gate_{int(floor):02d}_cap{capacity}_202511_202608"
        decisions = pd.read_parquet(sweep / f"{suffix}_decisions.parquet")
        equity = pd.read_parquet(sweep / f"{suffix}_equity.parquet")
        global_row, temporal, daily, weekly = _temporal_summary(_accepted(decisions), equity, label=suffix, floor=floor, capacity=capacity)
        # Preserve gate-level raw admission metrics alongside the portfolio path.
        for column in ("raw_dual_admitted_rows", "raw_dual_admission_net_ev_bps_per_trade", "raw_dual_admission_total_net_bps"):
            global_row[column] = getattr(row, column)
        summary_rows.append(global_row)
        temporal_rows.append(temporal.assign(label=suffix))
        daily_rows.append(daily.assign(dual_mc1_floor_bps=floor, max_new_entries_per_bar=capacity, label=suffix))
        weekly_rows.append(weekly.assign(dual_mc1_floor_bps=floor, max_new_entries_per_bar=capacity, label=suffix))
    summary = pd.DataFrame(summary_rows).sort_values(["dual_mc1_floor_bps", "max_new_entries_per_bar"], kind="stable")
    temporal = pd.concat(temporal_rows, ignore_index=True)
    daily = pd.concat(daily_rows, ignore_index=True)
    weekly = pd.concat(weekly_rows, ignore_index=True)
    mc1 = pd.read_parquet(mc1_path, columns=["__decision_ts__", "policy_path_valid", "policy_net_bps", "current_mc1_expected_bps", "bcf_mc1_expected_bps"])
    sweep_manifest = json.loads((sweep / "run_manifest.json").read_text())
    cutoff_raw = sweep_manifest.get("end_exclusive")
    if cutoff_raw:
        cutoff = pd.Timestamp(cutoff_raw)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        mc1["__decision_ts__"] = pd.to_datetime(mc1["__decision_ts__"], utc=True, errors="raise")
        mc1 = mc1.loc[mc1["__decision_ts__"].lt(cutoff)].copy()
    raw = pd.concat([_raw_monthly(mc1, floor) for floor in (30.0, 40.0, 50.0)], ignore_index=True)
    canonical_august = daily.loc[
        daily.dual_mc1_floor_bps.eq(50.0)
        & daily.max_new_entries_per_bar.eq(2)
        & daily.day.ge(pd.Timestamp("2026-08-01", tz="UTC"))
        & daily.day.lt(pd.Timestamp("2026-08-28", tz="UTC"))
    ].copy()
    summary.to_parquet(out / "portfolio_gate_capacity_global.parquet", index=False, compression="zstd")
    temporal.to_parquet(out / "portfolio_temporal_metrics.parquet", index=False, compression="zstd")
    daily.to_parquet(out / "portfolio_daily_metrics.parquet", index=False, compression="zstd")
    weekly.to_parquet(out / "portfolio_weekly_metrics.parquet", index=False, compression="zstd")
    raw.to_parquet(out / "raw_dual_admission_monthly.parquet", index=False, compression="zstd")
    canonical_august.to_parquet(out / "canonical_50bps_cap2_august01_27_daily.parquet", index=False, compression="zstd")
    (out / "EXTENDED_QUALITY_RECEIPT.md").write_text(_markdown(summary, temporal, canonical_august, raw))
    _write_json_once(out / "run_manifest.json", {
        "schema": "strict_r3_p8u_f72_underf120_extended_quality_v1",
        "scope": "offline evaluation only; no fitting, scoring, exchange I/O, or live mutation",
        "sweep": str(sweep), "sweep_gate_summary_sha256": _sha256(sweep / "gate_summary.parquet"),
        "mc1": str(mc1_path), "mc1_sha256": _sha256(mc1_path),
        "coverage": "2025-11 through 2026-08-27 inclusive",
        "policy": "frozen rich SimplePolicyOptimiser policy net labels; 100 bps cost embedded once",
        "canonical_control": "dual MC1 >=50 bps; max two new entries per decision timestamp",
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--mc1", type=Path, default=DEFAULT_MC1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.sweep.resolve(), args.mc1.resolve(), args.out.resolve()))


if __name__ == "__main__":
    main()
