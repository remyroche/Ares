#!/usr/bin/env python3
"""Replay recent production admissions with the deployed 1-minute exit policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.inference.simple_policy_stop import (
    SIMPLE_POLICY_STOP_PARAM_KEYS,
)
from extreme_price_movements.scripts.live_closed_trade_exit_replay import (
    ParsedRecap,
    _read_cached_execution_1m,
    replay_one_anchor,
)


DEFAULT_RUN_ID = (
    "s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_"
    "geometry_20260717_v2"
)


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _strategy_params(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["strategy_id"]): dict(row)
        for row in policy.get("strategies", [])
        if row.get("strategy_id")
    }


def _net_return(side: str, entry: float, exit_: float, cost: float) -> float:
    gross = (exit_ / entry - 1.0) if side == "long" else (entry / exit_ - 1.0)
    return float(gross - cost)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=480,
        help="Maximum executable path length; unresolved trades exit at the final available close.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ledger_path = (
        args.data_root
        / "exchanges/krakenfutures/live_state/prediction_ledgers"
        / args.run_id
        / "prediction_ledger.parquet"
    )
    policy_path = (
        args.data_root
        / "artifacts"
        / args.run_id
        / "simple_policy_optimiser/deployment/best_policy_params.json"
    )
    ledger = pd.read_parquet(ledger_path)
    ledger["timestamp"] = pd.to_datetime(ledger["timestamp"], utc=True)
    admitted = ledger.loc[ledger["portfolio_decision"].astype(str).eq("traded")].copy()
    end_day = admitted["timestamp"].max().floor("D")
    start_day = end_day - pd.Timedelta(days=args.days - 1)
    admitted = admitted.loc[admitted["timestamp"].ge(start_day)].copy()
    admitted = admitted.sort_values(["timestamp", "symbol"], kind="mergesort")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    params_by_strategy = _strategy_params(policy)
    fee = 0.01
    cached_bars: dict[str, pd.DataFrame] = {}
    for symbol in admitted["symbol"].astype(str).unique():
        bars = _read_cached_execution_1m(
            data_root=args.data_root,
            symbol=symbol,
            start=start_day,
            end=end_day + pd.Timedelta(days=2),
        )
        cached_bars[symbol] = bars
    if not any(not bars.empty for bars in cached_bars.values()):
        raise RuntimeError("No canonical execution 1-minute bars cover admitted rows")

    empty_recap = ParsedRecap(
        bars=pd.DataFrame(),
        stop_fill_ts=None,
        stop_fill_price=np.nan,
        stop_reason="",
        source="independent_execution_1m_replay",
    )
    results: list[dict[str, Any]] = []
    for row in admitted.to_dict(orient="records"):
        symbol = str(row["symbol"])
        side = str(row["side"]).lower()
        entry_ts = _utc(row["timestamp"])
        entry = pd.to_numeric(row.get("realized_entry_price"), errors="coerce")
        if not np.isfinite(entry):
            entry = pd.to_numeric(row.get("entry_price_actual"), errors="coerce")
        if not np.isfinite(entry) or float(entry) <= 0:
            raise ValueError(f"Missing valid actual entry price for {symbol} at {entry_ts}")
        entry = float(entry)
        strategy_id = str(row.get("strategy_id") or f"{side}_s52_meta_threshold_handoff")
        params = {
            key: value
            for key, value in params_by_strategy[strategy_id].items()
            if key in SIMPLE_POLICY_STOP_PARAM_KEYS
        }
        overrides: dict[str, Any] = {}
        try:
            overrides = json.loads(row.get("policy_replay_params_json") or "{}")
            params.update(
                {
                    key: value
                    for key, value in overrides.items()
                    if key in SIMPLE_POLICY_STOP_PARAM_KEYS
                }
            )
        except json.JSONDecodeError:
            pass
        params["params_hash"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()[:16]
        params["_loaded_from_simple_policy_artifact"] = True
        params["_artifact_path"] = str(policy_path.resolve())
        params["_artifact_mtime_ns"] = int(policy_path.stat().st_mtime_ns)
        params["replay_timeframe"] = "1m"
        row["initial_stop_price"] = row.get("policy_stop_price")
        row["entry_price"] = entry
        bars = cached_bars[symbol]
        horizon_end = entry_ts + pd.Timedelta(minutes=int(args.horizon_minutes))
        symbol_cutoff = _utc(bars["ts"].max()) if not bars.empty else pd.NaT
        path_end = min(symbol_cutoff, horizon_end) if pd.notna(symbol_cutoff) else horizon_end
        if bars.empty or "ts" not in bars.columns:
            path = pd.DataFrame()
        else:
            bar_ts = pd.to_datetime(bars["ts"], utc=True)
            path = bars.loc[
                bar_ts.ge(entry_ts.ceil("min")) & bar_ts.le(path_end)
            ].copy()
        replay = replay_one_anchor(
            row=row,
            policy_params=params,
            entry_price=entry,
            entry_anchor="realized_entry",
            bars=path,
            recap=empty_recap,
            ignore_logged_exit_events=True,
        )
        hit = bool(replay.get("replay_hit", False))
        if hit:
            exit_price = float(replay["replay_exit_price"])
            exit_ts = _utc(replay["replay_exit_ts"])
            resolution = "policy_exit"
        elif not path.empty:
            exit_price = float(path.iloc[-1]["close"])
            exit_ts = _utc(path.iloc[-1]["ts"])
            resolution = "timeout_close" if symbol_cutoff >= horizon_end.floor("min") else "marked_at_symbol_cutoff"
        else:
            exit_price = np.nan
            exit_ts = pd.NaT
            resolution = "missing_path"
        net = _net_return(side, entry, exit_price, fee) if np.isfinite(exit_price) else np.nan
        notional = pd.to_numeric(row.get("entry_notional_quote"), errors="coerce")
        results.append(
            {
                "entry_day_utc": entry_ts.floor("D"),
                "entry_ts_utc": entry_ts,
                "exit_ts_utc": exit_ts,
                "symbol": symbol,
                "side": side,
                "strategy_id": strategy_id,
                "entry_price": entry,
                "exit_price": exit_price,
                "resolution": resolution,
                "exit_reason": replay.get("replay_exit_reason", ""),
                "net_return_after_1pct": net,
                "entry_notional_quote": notional,
                "net_pnl_quote": float(net * notional) if np.isfinite(net) and np.isfinite(notional) else np.nan,
                "path_minutes": int(len(path)),
                "coverage_status": replay.get("coverage_status", ""),
                "replay_error": replay.get("error", ""),
            }
        )

    trades = pd.DataFrame(results)
    daily = (
        trades.groupby("entry_day_utc", observed=True)
        .agg(
            admitted_trades=("symbol", "size"),
            resolved_policy_exits=("resolution", lambda x: int((x == "policy_exit").sum())),
            open_marked_or_timeout=("resolution", lambda x: int(x.isin(["marked_at_symbol_cutoff", "timeout_close"]).sum())),
            missing_paths=("resolution", lambda x: int((x == "missing_path").sum())),
            mean_net_return=("net_return_after_1pct", "mean"),
            sum_net_return=("net_return_after_1pct", "sum"),
            net_pnl_quote=("net_pnl_quote", "sum"),
            gross_entry_notional_quote=("entry_notional_quote", lambda x: float(x[trades.loc[x.index, "net_return_after_1pct"].notna()].sum())),
            positive_trade_rate=("net_return_after_1pct", lambda x: float((x.dropna() > 0).mean()) if x.notna().any() else np.nan),
        )
        .reset_index()
    )
    daily["net_pnl_pct_of_deployed_notional"] = (
        daily["net_pnl_quote"] / daily["gross_entry_notional_quote"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(args.output_dir / "trade_replay.parquet", index=False)
    daily.to_csv(args.output_dir / "daily_metrics.csv", index=False)
    manifest = {
        "run_id": args.run_id,
        "ledger": str(ledger_path),
        "policy": str(policy_path),
        "execution_store": str(args.data_root / "exchanges/krakenfutures/execution_1m"),
        "entry_start_utc": str(admitted["timestamp"].min()),
        "entry_end_utc": str(admitted["timestamp"].max()),
        "horizon_minutes": int(args.horizon_minutes),
        "replay_cutoff_contract": (
            "per-symbol latest canonical minute, capped at entry plus "
            f"{int(args.horizon_minutes)} minutes; unresolved paths exit at the final available close"
        ),
        "cost_contract": "1% round-trip fee subtracted once; replay exit fills include modeled exit spread/gap",
        "selection_contract": "actual production portfolio_decision=traded admissions",
        "partial_days": [str(start_day.date()), str(end_day.date())],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(daily.to_string(index=False))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
