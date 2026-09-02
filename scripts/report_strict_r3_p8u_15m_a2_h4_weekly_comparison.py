#!/usr/bin/env python3
"""Build a receipt-backed weekly comparison of constrained H0+H4 portfolios.

This is an offline reporting utility.  It reads immutable accepted-decision and
equity receipts; it does not refit, score, contact an exchange, or change any
research/live artifact.  Weeks are grouped by entry decision timestamp.  The
weekly return and local drawdown are calculated from the constrained MTM equity
path, seeded by the immediately preceding equity observation where available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


H4 = "H4_l1_d4_l15_leaf5_reg20"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _period_start(series: pd.Series, frequency: str) -> pd.Series:
    base = pd.to_datetime(series, utc=True, errors="raise").dt.normalize()
    if frequency == "week":
        return base - pd.to_timedelta(base.dt.weekday, unit="D")
    if frequency == "month":
        return base - pd.to_timedelta(base.dt.day - 1, unit="D")
    raise ValueError(f"unsupported period frequency: {frequency}")


def period_from_receipts(root: Path, arm: str, frequency: str) -> pd.DataFrame:
    accepted = pd.read_parquet(root / f"{arm}__all_oos_accepted.parquet")
    equity = pd.read_parquet(root / f"{arm}__all_oos_equity.parquet")
    accepted["timestamp"] = pd.to_datetime(accepted.timestamp, utc=True, errors="raise")
    equity["timestamp"] = pd.to_datetime(equity.timestamp, utc=True, errors="raise")
    accepted["period"] = _period_start(accepted.timestamp, frequency)
    equity["period"] = _period_start(equity.timestamp, frequency)
    accepted["net_bps"] = pd.to_numeric(accepted.position_net_return, errors="raise") * 10_000.0
    trade = accepted.groupby("period", as_index=False).agg(
        trades=("candidate_id", "size"), total_net_bps=("net_bps", "sum"),
        net_ev_per_trade_bps=("net_bps", "mean"), mean_position_size=("position_size", "mean"),
        max_open_positions_after=("open_positions_after", "max"),
    )
    # A week starts from the immediately preceding constrained MTM observation,
    # then measures drawdown only within that week.  This avoids attributing an
    # older peak-to-trough move to every following week.
    rows: list[dict[str, object]] = []
    equity = equity.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for period, frame in equity.groupby("period", sort=True):
        first_i = int(frame.index.min())
        seeded = equity.iloc[[first_i - 1]].copy() if first_i else frame.iloc[0:0].copy()
        path = pd.concat([seeded, frame], ignore_index=True)
        mtm = pd.to_numeric(path.mtm_equity, errors="raise").to_numpy(float)
        peak = np.maximum.accumulate(mtm)
        local_dd = float(np.min(mtm / peak - 1.0)) if len(mtm) else np.nan
        start_equity = float(mtm[0])
        end_equity = float(mtm[-1])
        rows.append({
            "period": period,
            "period_start_mtm_equity": start_equity,
            "period_end_mtm_equity": end_equity,
            "period_mtm_return": end_equity / start_equity - 1.0 if start_equity else np.nan,
            "period_local_max_drawdown": local_dd,
            "mean_open_positions": float(pd.to_numeric(frame.open_positions, errors="raise").mean()),
            "max_open_positions": int(pd.to_numeric(frame.open_positions, errors="raise").max()),
            "mean_wallet_cap_utilization": float(pd.to_numeric(frame.wallet_cap_utilization, errors="raise").mean()),
            "max_wallet_cap_utilization": float(pd.to_numeric(frame.wallet_cap_utilization, errors="raise").max()),
        })
    risk = pd.DataFrame(rows)
    result = risk.merge(trade, on="period", how="outer", validate="one_to_one").sort_values("period", kind="stable")
    for field in ("trades", "total_net_bps"):
        result[field] = result[field].fillna(0)
    result["net_ev_per_trade_bps"] = result.net_ev_per_trade_bps.where(result.trades.gt(0))
    result["arm"] = arm
    result["frequency"] = frequency
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--challenger-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    control_root, challenger_root = args.control_root.resolve(), args.challenger_root.resolve()
    control = period_from_receipts(control_root, H4, "week").assign(comparison_arm="R0_H0_plus_H4")
    challenger = period_from_receipts(challenger_root, H4, "week").assign(comparison_arm="A2_expanded_H0_plus_H4")
    key = "period"
    common = control.merge(challenger, on=key, how="outer", suffixes=("_control", "_a2"), validate="one_to_one")
    for metric in ("trades", "total_net_bps", "net_ev_per_trade_bps", "period_mtm_return", "period_local_max_drawdown", "mean_open_positions", "max_open_positions", "mean_wallet_cap_utilization", "max_wallet_cap_utilization"):
        common[f"delta_{metric}_a2_minus_control"] = pd.to_numeric(common[f"{metric}_a2"], errors="coerce") - pd.to_numeric(common[f"{metric}_control"], errors="coerce")
    common = common.sort_values(key, kind="stable")
    monthly_control = period_from_receipts(control_root, H4, "month").assign(comparison_arm="R0_H0_plus_H4")
    monthly_challenger = period_from_receipts(challenger_root, H4, "month").assign(comparison_arm="A2_expanded_H0_plus_H4")
    monthly = pd.concat([monthly_control, monthly_challenger], ignore_index=True)
    monthly_pivot = monthly.pivot(index="period", columns="comparison_arm")
    monthly_delta = pd.DataFrame({"month": monthly_pivot.index.astype(str)})
    for metric in ("trades", "total_net_bps", "net_ev_per_trade_bps", "period_mtm_return", "period_local_max_drawdown", "mean_wallet_cap_utilization", "max_wallet_cap_utilization", "max_open_positions"):
        monthly_delta[f"delta_{metric}_a2_minus_control"] = (
            monthly_pivot[(metric, "A2_expanded_H0_plus_H4")].to_numpy()
            - monthly_pivot[(metric, "R0_H0_plus_H4")].to_numpy()
        )
    out.mkdir(parents=True, exist_ok=False)
    control.to_parquet(out / "weekly_control.parquet", index=False)
    challenger.to_parquet(out / "weekly_a2.parquet", index=False)
    common.to_parquet(out / "weekly_comparison.parquet", index=False)
    monthly.to_parquet(out / "monthly_metrics.parquet", index=False)
    monthly_delta.to_parquet(out / "monthly_deltas.parquet", index=False)
    summary = pd.concat([
        pd.read_parquet(control_root / "portfolio_summary.parquet").assign(comparison_arm="R0_H0_plus_H4"),
        pd.read_parquet(challenger_root / "portfolio_summary.parquet").assign(comparison_arm="A2_expanded_H0_plus_H4"),
    ], ignore_index=True)
    summary.to_parquet(out / "portfolio_summaries.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-a2-h4-weekly-comparison-v1",
        "scope": "read-only offline comparison of existing constrained portfolio receipts",
        "control_root": str(control_root),
        "challenger_root": str(challenger_root),
        "control_manifest_sha256": sha256(control_root / "run_manifest.json"),
        "challenger_manifest_sha256": sha256(challenger_root / "run_manifest.json"),
        "week_definition": "Monday 00:00 UTC through Sunday 23:59 UTC, by entry decision timestamp",
        "month_definition": "calendar month UTC, by entry decision timestamp",
        "return_definition": "constrained MTM equity path, seeded from immediately preceding observation",
        "drawdown_definition": "local within-week constrained MTM drawdown, seeded from immediately preceding observation",
        "no_refit_no_exchange_io": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
