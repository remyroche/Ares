#!/usr/bin/env python3
"""Join canonical close-price exits to causal Kraken L2 execution states.

This is an offline oracle.  It does not alter the parent exit policy.  For an
existing canonical exit ``t*`` it compares that exit with predeclared earlier
times and quantifies whether book-cost savings exceed the close-price PnL
sacrificed by leaving earlier.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.surface import _notional_token  # noqa: E402


PREEMPT_MINUTES = (5, 3, 2, 1, 0)


def _utc(values: object) -> pd.Timestamp | pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_datetime(values, utc=True, errors="coerce")
    stamp = pd.Timestamp(values)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _read_many(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        raise FileNotFoundError("no parquet sources supplied")
    return pd.concat(frames, ignore_index=True, copy=False)


def _find_parquet(root: Path, name: str) -> list[Path]:
    values = sorted(root.rglob(name))
    if not values:
        raise FileNotFoundError(f"no {name!r} files under {root}")
    return values


def _cost_columns(frame: pd.DataFrame, side: str, notionals: Sequence[float]) -> list[str]:
    prefix = "sell_book_cost_bps_" if side == "long" else "buy_book_cost_bps_"
    columns = [prefix + _notional_token(value) for value in notionals]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"execution states lack declared cost grid columns: {missing}")
    return columns


def _interpolated_book_cost(row: pd.Series, *, side: str, position_notional: float, notionals: Sequence[float]) -> tuple[float, bool]:
    if not np.isfinite(position_notional) or position_notional <= 0.0:
        return float("nan"), True
    ordered = np.asarray(sorted(float(value) for value in notionals), dtype=float)
    if position_notional > ordered[-1]:
        return float("nan"), True
    cost_columns = _cost_columns(pd.DataFrame([row]), side, ordered)
    insuff_prefix = "sell_insufficient_depth_" if side == "long" else "buy_insufficient_depth_"
    costs = pd.to_numeric(row[cost_columns], errors="coerce").to_numpy(float)
    insuff = np.asarray([bool(row.get(insuff_prefix + _notional_token(value), True)) for value in ordered], dtype=bool)
    if not np.isfinite(costs).all() or bool(insuff[ordered <= position_notional].any()):
        return float("nan"), True
    return float(np.interp(position_notional, ordered, costs)), False


def _first_available_state(
    states: pd.DataFrame,
    *,
    symbol: str,
    decision_time: pd.Timestamp,
    latency: pd.Timedelta,
    max_lag: pd.Timedelta,
) -> pd.Series | None:
    group = states.loc[states["symbol"].astype(str).eq(str(symbol))]
    if group.empty:
        return None
    times = group["available_ts"].to_numpy(dtype="datetime64[ns]")
    target = (decision_time + latency).tz_localize(None).to_datetime64()
    index = int(np.searchsorted(times, target, side="left"))
    if index >= len(group):
        return None
    selected = group.iloc[index]
    available = pd.Timestamp(selected["available_ts"])
    if available.tzinfo is None:
        available = available.tz_localize("UTC")
    if available - (decision_time + latency) > max_lag:
        return None
    return selected


def _price_lookup(prices: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    output: dict[tuple[str, pd.Timestamp], float] = {}
    for row in prices.loc[:, ["symbol", "timestamp", "close"]].dropna().itertuples(index=False):
        timestamp = _utc(row.timestamp)
        close = float(row.close)
        if np.isfinite(close) and close > 0.0:
            output[(str(row.symbol), timestamp)] = close
    return output


def _close_pnl_bps(*, side: str, entry_price: float, close: float, policy_cost_bps: float) -> float:
    if side == "long":
        gross = (close / entry_price - 1.0) * 10_000.0
    elif side == "short":
        gross = (entry_price / close - 1.0) * 10_000.0
    else:
        raise ValueError(f"unsupported side {side!r}")
    return float(gross - policy_cost_bps)


def build_execution_oracle(
    *,
    exits: pd.DataFrame,
    states: pd.DataFrame,
    prices: pd.DataFrame,
    notionals: Sequence[float],
    latency_ms: Sequence[int] = (0,),
    max_state_lag_seconds: float = 5.0,
    policy_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Return predeclared exit-time candidates and their executable oracle PnL."""
    required_exits = {"exit_id", "symbol", "side", "exit_ts", "entry_price", "position_notional", "exit_reason"}
    missing = required_exits.difference(exits.columns)
    if missing:
        raise ValueError(f"exits lack required columns: {sorted(missing)}")
    required_states = {"symbol", "available_ts", "mid", "spread_bps", "book_valid"}
    missing = required_states.difference(states.columns)
    if missing:
        raise ValueError(f"execution states lack required columns: {sorted(missing)}")
    states = states.copy()
    states["available_ts"] = _utc(states["available_ts"])
    states = states.loc[states["book_valid"].fillna(False).astype(bool)].sort_values(
        ["symbol", "available_ts"], kind="stable"
    ).reset_index(drop=True)
    prices = prices.copy()
    prices["timestamp"] = _utc(prices["timestamp"])
    close_by_key = _price_lookup(prices)
    output: list[dict[str, Any]] = []
    max_lag = pd.Timedelta(seconds=float(max_state_lag_seconds))
    for exit_row in exits.to_dict("records"):
        symbol = str(exit_row["symbol"])
        side = str(exit_row["side"]).lower()
        t_star = _utc(exit_row["exit_ts"])
        entry = float(exit_row["entry_price"])
        notional = float(exit_row["position_notional"])
        if side not in {"long", "short"} or not (np.isfinite(entry) and entry > 0.0):
            continue
        canonical_exit_close = close_by_key.get((symbol, t_star), float("nan"))
        exit_close_pnl = _close_pnl_bps(side=side, entry_price=entry, close=canonical_exit_close, policy_cost_bps=policy_cost_bps) if np.isfinite(canonical_exit_close) else float("nan")
        for preempt in PREEMPT_MINUTES:
            candidate_time = t_star - pd.Timedelta(minutes=int(preempt))
            candidate_close = close_by_key.get((symbol, candidate_time), float("nan"))
            candidate_close_pnl = _close_pnl_bps(side=side, entry_price=entry, close=candidate_close, policy_cost_bps=policy_cost_bps) if np.isfinite(candidate_close) else float("nan")
            for latency in latency_ms:
                state = _first_available_state(
                    states, symbol=symbol, decision_time=candidate_time,
                    latency=pd.Timedelta(milliseconds=int(latency)), max_lag=max_lag,
                )
                base = {
                    "exit_id": str(exit_row["exit_id"]), "symbol": symbol, "side": side,
                    "exit_reason": str(exit_row["exit_reason"]), "t_star": t_star,
                    "candidate_time": candidate_time, "preempt_minutes": int(preempt),
                    "latency_ms": int(latency), "canonical_close": canonical_exit_close,
                    "candidate_close": candidate_close, "close_pnl_bps": candidate_close_pnl,
                    "canonical_exit_close_pnl_bps": exit_close_pnl, "position_notional": notional,
                    "policy_cost_bps": float(policy_cost_bps),
                }
                if state is None:
                    output.append({**base, "data_quality_flags": "missing_or_late_complete_book", "insufficient_depth": True})
                    continue
                cost, insufficient = _interpolated_book_cost(state, side=side, position_notional=notional, notionals=notionals)
                mid = float(state["mid"])
                vwap = mid * (1.0 - cost / 10_000.0) if side == "long" else mid * (1.0 + cost / 10_000.0)
                executable_pnl = candidate_close_pnl - cost if np.isfinite(candidate_close_pnl) and np.isfinite(cost) else float("nan")
                canonical_cost, canonical_insufficient = (float("nan"), True)
                canonical_state = _first_available_state(
                    states, symbol=symbol, decision_time=t_star, latency=pd.Timedelta(milliseconds=int(latency)), max_lag=max_lag,
                )
                if canonical_state is not None:
                    canonical_cost, canonical_insufficient = _interpolated_book_cost(canonical_state, side=side, position_notional=notional, notionals=notionals)
                price_effect = candidate_close_pnl - exit_close_pnl if np.isfinite(candidate_close_pnl) and np.isfinite(exit_close_pnl) else float("nan")
                execution_saving = canonical_cost - cost if np.isfinite(canonical_cost) and np.isfinite(cost) else float("nan")
                output.append({
                    **base,
                    "available_ts": state["available_ts"], "book_mid": mid,
                    "spread_bps": float(state["spread_bps"]), "book_vwap": vwap,
                    "book_cost_bps": cost, "canonical_book_cost_bps": canonical_cost,
                    "executable_pnl_bps": executable_pnl, "price_effect_bps": price_effect,
                    "execution_saving_bps": execution_saving,
                    "preemption_gain_bps": price_effect + execution_saving if np.isfinite(price_effect) and np.isfinite(execution_saving) else float("nan"),
                    "insufficient_depth": bool(insufficient or canonical_insufficient),
                    "data_quality_flags": "" if not insufficient else "insufficient_book_depth",
                })
    return pd.DataFrame.from_records(output)


def oracle_summary(oracle: pd.DataFrame) -> pd.DataFrame:
    """Required robust distribution slices for the pre-emption opportunity."""
    if oracle.empty:
        return pd.DataFrame()
    valid = oracle.loc[~oracle["insufficient_depth"].fillna(True)].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in valid.groupby(["preempt_minutes", "exit_reason"], dropna=False, sort=True):
        values = pd.to_numeric(group["preemption_gain_bps"], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "preempt_minutes": int(keys[0]), "exit_reason": str(keys[1]), "rows": int(len(values)),
            "mean_preemption_gain_bps": float(values.mean()), "median_preemption_gain_bps": float(values.median()),
            "p10_preemption_gain_bps": float(values.quantile(.10)), "p25_preemption_gain_bps": float(values.quantile(.25)),
            "p75_preemption_gain_bps": float(values.quantile(.75)), "p90_preemption_gain_bps": float(values.quantile(.90)),
            "fraction_gain_gt_0": float(values.gt(0.0).mean()), "fraction_gain_gt_25": float(values.gt(25.0).mean()),
            "fraction_gain_gt_50": float(values.gt(50.0).mean()), "fraction_gain_gt_100": float(values.gt(100.0).mean()),
        })
    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exits", type=Path, required=True, help="Canonical policy-exit parquet")
    parser.add_argument("--prices", type=Path, required=True, help="Canonical minute close parquet: symbol/timestamp/close")
    parser.add_argument("--states-root", type=Path, required=True, help="Surface root or detailed message-state root")
    parser.add_argument("--state-file-name", default="states.parquet", help="Use surface.parquet for minute-only approximation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--notionals", type=float, nargs="+", required=True, help="Same quote-notional grid used by state builder")
    parser.add_argument("--latency-ms", type=int, nargs="+", default=[0, 250, 1000, 5000])
    parser.add_argument("--max-state-lag-seconds", type=float, default=5.0)
    parser.add_argument("--policy-cost-bps", type=float, default=0.0, help="Only set when close PnL source excludes the declared parent-policy cost")
    parser.add_argument("--exit-id-column", default="candidate_id")
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument("--side-column", default="side_name")
    parser.add_argument("--exit-ts-column", default="exit_timestamp")
    parser.add_argument("--entry-column", default="entry_price")
    parser.add_argument("--notional-column", default="position_notional")
    parser.add_argument("--reason-column", default="simple_policy_exit_reason")
    args = parser.parse_args()

    exits = pd.read_parquet(args.exits).rename(columns={
        args.exit_id_column: "exit_id", args.symbol_column: "symbol", args.side_column: "side",
        args.exit_ts_column: "exit_ts", args.entry_column: "entry_price", args.notional_column: "position_notional",
        args.reason_column: "exit_reason",
    })
    prices = pd.read_parquet(args.prices)
    paths = _find_parquet(args.states_root, args.state_file_name)
    states = _read_many(paths)
    oracle = build_execution_oracle(
        exits=exits, states=states, prices=prices, notionals=tuple(args.notionals),
        latency_ms=tuple(args.latency_ms), max_state_lag_seconds=float(args.max_state_lag_seconds),
        policy_cost_bps=float(args.policy_cost_bps),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    oracle.to_parquet(args.out, index=False)
    summary = oracle_summary(oracle)
    summary.to_parquet(args.out.with_name(args.out.stem + "_summary.parquet"), index=False)
    receipt = {
        "schema": "ares.policy_exit_execution_oracle.v1",
        "oracle": str(args.out), "rows": int(len(oracle)), "valid_rows": int((~oracle.get("insufficient_depth", pd.Series(dtype=bool)).fillna(True)).sum()),
        "canonical_policy_modified": False, "preempt_minutes": list(PREEMPT_MINUTES),
        "latency_ms": list(args.latency_ms), "state_source": str(args.states_root),
        "timestamp_rule": "first complete valid state at or after candidate decision plus declared latency, bounded by max-state-lag",
    }
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
