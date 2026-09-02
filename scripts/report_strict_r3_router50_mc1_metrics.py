#!/usr/bin/env python3
"""Compare router-free and router-aware MC1 maps on identical meta scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EVAL_START = pd.Timestamp("2026-04-01T00:00:00Z")
EVAL_END = pd.Timestamp("2026-08-01T00:00:00Z")


def _risk_metrics(frame: pd.DataFrame) -> dict[str, object]:
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    monthly = frame.assign(month=pd.to_datetime(frame["__decision_ts__"], utc=True).dt.strftime("%Y-%m")).groupby("month", sort=True)["policy_net_bps"].mean()
    weekly = frame.assign(week=pd.to_datetime(frame["__decision_ts__"], utc=True).dt.strftime("%G-W%V")).groupby("week", sort=True)["policy_net_bps"].mean()
    all_days = pd.date_range(EVAL_START.floor("D"), (EVAL_END - pd.Timedelta(days=1)).floor("D"), freq="D", tz="UTC")
    observed_days = pd.DatetimeIndex(pd.to_datetime(frame["__decision_ts__"], utc=True).dt.floor("D").unique()) if len(frame) else pd.DatetimeIndex([], tz="UTC")
    return {
        "rows": int(len(frame)),
        "net_ev_bps_per_trade": float(net.mean()) if len(net) else np.nan,
        "net_sum_bps": float(net.sum()) if len(net) else 0.0,
        "worst_month_bps": float(monthly.min()) if len(monthly) else np.nan,
        "worst_week_bps": float(weekly.min()) if len(weekly) else np.nan,
        "positive_month_fraction": float(monthly.gt(0).mean()) if len(monthly) else np.nan,
        "days_without_entries": int(all_days.difference(observed_days).size),
    }


def _daily_sortino(equity: pd.DataFrame) -> float:
    if not {"timestamp", "wallet"}.issubset(equity.columns):
        return float("nan")
    work = equity.loc[:, ["timestamp", "wallet"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["wallet"] = pd.to_numeric(work["wallet"], errors="coerce")
    work = work.dropna().sort_values("timestamp", kind="stable")
    work["day"] = work["timestamp"].dt.floor("D")
    ret = work.groupby("day", sort=True)["wallet"].last().pct_change().dropna().to_numpy(float)
    down = float(np.sqrt(np.mean(np.minimum(ret, 0.0) ** 2))) if len(ret) else np.nan
    return float(np.sqrt(365.0) * np.mean(ret) / down) if np.isfinite(down) and down > 0 else float("nan")


def _dual(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    valid = (
        frame["enhanced_base_routed"].fillna(False).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_exit_bar_15m"], errors="coerce"))
        & pd.to_numeric(frame["current_mc1_expected_bps"], errors="coerce").ge(threshold)
        & pd.to_numeric(frame["bcf_mc1_expected_bps"], errors="coerce").ge(threshold)
    )
    return frame.loc[valid].copy()


def _global_rows(root: Path, threshold: float) -> dict[str, object]:
    metrics = pd.read_parquet(root / "portfolio_metrics.parquet")
    metric = metrics.loc[pd.to_numeric(metrics["threshold_bps"], errors="coerce").eq(float(threshold))].iloc[0].to_dict()
    decisions = pd.read_parquet(root / f"routed_base_dual_{int(threshold)}_2026_marjul_decisions.parquet")
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    timestamp = pd.to_datetime(accepted["timestamp"], utc=True)
    maximum = int(timestamp.value_counts().max()) if len(timestamp) else 0
    if maximum > 2:
        raise AssertionError(f"portfolio replay breached max-two timestamp cap: {maximum}")
    equity = pd.read_parquet(root / f"routed_base_dual_{int(threshold)}_2026_marjul_equity.parquet")
    return {
        **{f"portfolio_{key}": value for key, value in metric.items()},
        "portfolio_max_entries_per_timestamp": maximum,
        "portfolio_sortino_daily_annualized": _daily_sortino(equity),
    }


def _arm(label: str, root: Path, thresholds: tuple[float, ...]) -> list[dict[str, object]]:
    frame = pd.read_parquet(root / "dual_mc1_predictions.parquet")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        admitted = _dual(frame, threshold)
        top2 = admitted.sort_values(
            ["__decision_ts__", "bcf_mc1_expected_bps", "candidate_id"], ascending=[True, False, True], kind="stable",
        ).groupby("__decision_ts__", sort=False, group_keys=False).head(2).copy()
        rows.append({
            "arm": label, "threshold_bps": threshold,
            **{f"admitted_{key}": value for key, value in _risk_metrics(admitted).items()},
            **{f"unconstrained_top2_{key}": value for key, value in _risk_metrics(top2).items()},
            **_global_rows(root, threshold),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-router-root", type=Path, required=True)
    parser.add_argument("--router-mc1-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    thresholds = (30.0, 40.0, 50.0)
    table = pd.DataFrame([
        *_arm("MC1_without_router_outputs", args.no_router_root, thresholds),
        *_arm("MC1_with_router_outputs", args.router_mc1_root, thresholds),
    ])
    table.to_parquet(args.out / "mc1_router_variant_metrics.parquet", index=False, compression="zstd")
    deltas: list[dict[str, object]] = []
    left = table.loc[table["arm"].eq("MC1_without_router_outputs")].set_index("threshold_bps")
    right = table.loc[table["arm"].eq("MC1_with_router_outputs")].set_index("threshold_bps")
    for threshold in thresholds:
        record = {"threshold_bps": threshold}
        for field in (
            "admitted_rows", "admitted_net_ev_bps_per_trade", "admitted_net_sum_bps",
            "unconstrained_top2_rows", "unconstrained_top2_net_ev_bps_per_trade", "unconstrained_top2_net_sum_bps",
            "portfolio_accepted_rows", "portfolio_net_ev_bps_per_realised_trade", "portfolio_net_sum_bps_realised",
            "portfolio_worst_month_bps", "portfolio_worst_week_bps", "portfolio_max_drawdown", "portfolio_sortino_daily_annualized",
        ):
            record[f"delta_{field}"] = float(right.loc[threshold, field]) - float(left.loc[threshold, field])
        deltas.append(record)
    pd.DataFrame(deltas).to_parquet(args.out / "delta_router_mc1.parquet", index=False, compression="zstd")
    args.out.joinpath("run_manifest.json").write_text(json.dumps({
        "scope": "offline research only; exact shared meta score panels and canonical rich-policy outcomes",
        "thresholds_bps": list(thresholds),
        "unconstrained_definition": "dual admission then BCF-MC1 priority, maximum two entries per timestamp, no cross-time portfolio constraints",
        "portfolio_definition": "standard global constrained replay; independently asserted maximum two new entries per timestamp",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
