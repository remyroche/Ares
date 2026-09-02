#!/usr/bin/env python3
"""Materialise post-score frozen-policy labels for one strict-R3 phase.

This is deliberately an *evaluation overlay*.  Candidate identities and all
score inputs are read from the target-free point-in-time closure.  It never
filters candidates based on a future path: incomplete 15-minute paths are
retained as ``policy_path_valid=false`` and have no numerical outcome.

The output is compatible with the prequential MC1 replay.  It applies the
frozen SimplePolicyOptimiser contract exactly once: decision-time 15-minute
open, H12, SL 3 ATR, trailing activation .5 ATR, giveback .25 ATR, then a
single 100-bps deduction from simulator gross return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The historical contract has no hidden policy-engine execution adjustments.
os.environ["EPM_EXCHANGE"] = "krakenfutures"
os.environ["EPM_SIMPLE_POLICY_15M_DOWNLOAD"] = "0"
os.environ["EPM_SIMPLE_POLICY_15M_CHART_ONLY"] = "0"
os.environ["EPM_SIMPLE_POLICY_STOP_EXIT_BASE_GAP_BPS"] = "0"
os.environ["EPM_SIMPLE_POLICY_STOP_EXIT_MAX_GAP_BPS"] = "0"
os.environ["EPM_SIMPLE_POLICY_SPREAD_MODEL_ENABLED"] = "0"

from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _PerpPolicy15mReplayStore,
    simulate_and_score,
)

COST_BPS = 100.0
HORIZON_BARS = 48
POLICY_COLUMNS = [
    "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_label_available_ts", "policy_outcome_source",
    "policy_cost_bps",
]


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True, type=Path)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--data-root", default=ROOT / "data_perp", type=Path)
    p.add_argument("--policy-json", default=ROOT / "config/strict_r3_frozen_15m_policy.json", type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    return p.parse_args()


def _utc(value: object) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC")


def _policy(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text())
    winner = raw.get("winner", {})
    if float(raw.get("cost_bps_once", COST_BPS)) != COST_BPS or int(raw.get("timeout_hours", 12)) != 12:
        raise ValueError("only frozen H12 / 100-bps policy contract is permitted")
    result = {key: float(winner[key]) for key in ("sl_mult", "trailing_activation_mult", "fixed_trailing_gap_mult")}
    if result != {"sl_mult": 3.0, "trailing_activation_mult": 0.5, "fixed_trailing_gap_mult": 0.25}:
        raise ValueError(f"unexpected frozen policy geometry: {result}")
    return result


def _empty(frame: pd.DataFrame, reason: str = "incomplete_15m_path") -> pd.DataFrame:
    out = pd.DataFrame({"candidate_id": frame["candidate_id"].astype(str).to_numpy()})
    out["policy_path_valid"] = False
    out["policy_gross_bps"] = np.nan
    out["policy_net_bps"] = np.nan
    out["policy_exit_bar_15m"] = -1
    out["policy_exit_reason"] = reason
    out["policy_entry_price"] = np.nan
    out["policy_exit_price"] = np.nan
    out["policy_label_available_ts"] = pd.to_datetime(frame["__decision_ts__"], utc=True) + pd.Timedelta(hours=12)
    out["policy_outcome_source"] = "unavailable"
    out["policy_cost_bps"] = np.nan
    return out.loc[:, POLICY_COLUMNS]


def _paths(frame: pd.DataFrame, bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate positions and exact 48-bar OHLC paths, never filling gaps."""
    starts = pd.DatetimeIndex(pd.to_datetime(frame["__decision_ts__"], utc=True))
    # Work in UTC nanoseconds rather than NumPy object timestamps.  The latter
    # makes ``datetime + timedelta64`` depend on pandas/NumPy coercion rules.
    expected_ns = starts.asi8[:, None] + (
        np.arange(HORIZON_BARS, dtype=np.int64)[None, :] * 15 * 60 * 1_000_000_000
    )
    expected = pd.DatetimeIndex(pd.to_datetime(expected_ns.ravel(), utc=True))
    lookup = bars.index.get_indexer(expected).reshape(len(frame), HORIZON_BARS)
    usable = (lookup >= 0).all(axis=1)
    positions = np.flatnonzero(usable)
    if not len(positions):
        empty = np.empty((0, HORIZON_BARS), dtype=np.float64)
        return positions, empty, empty, empty, empty
    indexer = lookup[positions]
    arrays = []
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(bars[column], errors="coerce").to_numpy(np.float64)
        arrays.append(values[indexer])
    finite = np.isfinite(np.stack(arrays)).all(axis=(0, 2))
    positions = positions[finite]
    return positions, *(value[finite] for value in arrays)


def _replay_symbol(frame: pd.DataFrame, store: _PerpPolicy15mReplayStore, policy: dict[str, float]) -> pd.DataFrame:
    out = _empty(frame)
    symbol = str(frame["__symbol__"].iloc[0])
    begin = pd.to_datetime(frame["__decision_ts__"], utc=True).min()
    finish = pd.to_datetime(frame["__decision_ts__"], utc=True).max() + pd.Timedelta(hours=12) - pd.Timedelta(minutes=15)
    bars = store.load(symbol, columns=["open", "high", "low", "close"], start_ts=begin, end_ts=finish)
    if bars.empty:
        return out
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    positions, opens, highs, lows, closes = _paths(frame, bars)
    if not len(positions):
        return out
    atr = pd.to_numeric(frame.iloc[positions]["signal_atr"], errors="coerce").to_numpy(np.float64)
    entry = opens[:, 0]
    keep = np.isfinite(atr) & (atr > 0.0) & np.isfinite(entry) & (entry > 0.0)
    positions, opens, highs, lows, closes, atr, entry = (
        positions[keep], opens[keep], highs[keep], lows[keep], closes[keep], atr[keep], entry[keep],
    )
    if not len(positions):
        return out
    run = pd.DataFrame({
        "timestamp": frame.iloc[positions]["__ts__"].to_numpy(),
        "symbol": frame.iloc[positions]["__symbol__"].astype(str).to_numpy(),
        "side": np.ones(len(positions), dtype=np.float32),
        "rank_pct": np.ones(len(positions), dtype=np.float32),
        "barrier_pct": atr / entry,
        "expected_half_spread_bps": np.zeros(len(positions)),
        "exit_quote_half_spread_bps": np.zeros(len(positions)),
        "entry_slippage_proxy_bps": np.zeros(len(positions)),
        "market_mode": "perps",
    })
    sim = simulate_and_score(
        run, opens, highs, lows, closes, cost_pct=0.0, size_power=1.0,
        replay_timeframe="15m", market_mode="perps", sl_mult=policy["sl_mult"], sl_abs_cap_pct=0.0,
        trailing_activation_mult=policy["trailing_activation_mult"], trailing_activation_cap_pct=0.0,
        trailing_activation_max_bars=HORIZON_BARS, fixed_trailing_gap_mult=policy["fixed_trailing_gap_mult"],
        capital_protect_mfe_mult=0.0, adverse_exit_enabled=False, hard_tp_abs_pct=0.0,
        max_concurrent_trades=max(len(run), 1), max_concurrent_per_asset=max(len(run), 1),
        max_new_entries_per_bar=max(len(run), 1),
    )
    if not np.asarray(sim["selected_mask"], dtype=bool).all():
        raise AssertionError("label replay applied a portfolio constraint")
    gross = np.asarray(sim["gross_returns"], dtype=np.float64) * 10_000.0
    valid = np.isfinite(gross)
    rows = positions[valid]
    out.loc[rows, "policy_path_valid"] = True
    out.loc[rows, "policy_gross_bps"] = gross[valid]
    out.loc[rows, "policy_net_bps"] = gross[valid] - COST_BPS
    out.loc[rows, "policy_exit_bar_15m"] = np.asarray(sim["exit_bars"], dtype=np.int16)[valid]
    out.loc[rows, "policy_exit_reason"] = np.asarray(sim["exit_reason"], dtype=object)[valid]
    out.loc[rows, "policy_entry_price"] = np.asarray(sim["entry_prices"], dtype=np.float64)[valid]
    out.loc[rows, "policy_exit_price"] = np.asarray(sim["exit_prices"], dtype=np.float64)[valid]
    out.loc[rows, "policy_outcome_source"] = "krakenfutures_dedicated_15m_phase_replay"
    out.loc[rows, "policy_cost_bps"] = COST_BPS
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    start, end = _utc(args.start), _utc(args.end)
    if end <= start:
        raise ValueError("end must exceed start")
    columns = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "signal_atr"]
    candidates = pd.read_parquet(args.candidates, columns=columns)
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
    candidates["__decision_ts__"] = pd.to_datetime(candidates["__decision_ts__"], utc=True)
    candidates = candidates.loc[(candidates["__decision_ts__"] >= start) & (candidates["__decision_ts__"] < end)].copy()
    if candidates.empty:
        raise ValueError("no candidates in requested decision interval")
    if not candidates["candidate_id"].is_unique:
        raise ValueError("candidate identities must be unique")
    if not candidates["side_name"].astype(str).eq("long").all():
        raise ValueError("phase replay is long only")
    if not (candidates["__decision_ts__"] - candidates["__ts__"] == pd.Timedelta(hours=1)).all():
        raise AssertionError("candidate decision convention is not signal + 1h")
    policy = _policy(args.policy_json)
    args.out_dir.mkdir(parents=True)
    store = _PerpPolicy15mReplayStore(args.data_root, "perps")
    parts = []
    for number, (_, group) in enumerate(candidates.groupby("__symbol__", sort=True), start=1):
        parts.append(_replay_symbol(group.reset_index(drop=True), store, policy))
        if number % 20 == 0 or number == candidates["__symbol__"].nunique():
            print(f"labels: {number}/{candidates['__symbol__'].nunique()} symbols", flush=True)
    labels = pd.concat(parts, ignore_index=True).sort_values("candidate_id", kind="stable")
    if len(labels) != len(candidates) or not labels["candidate_id"].is_unique:
        raise AssertionError("labels must preserve every target-free identity exactly once")
    valid = labels["policy_path_valid"].to_numpy(bool)
    if valid.any() and not np.allclose(labels.loc[valid, "policy_gross_bps"], labels.loc[valid, "policy_net_bps"] + COST_BPS, rtol=0.0, atol=1e-10):
        raise AssertionError("frozen 100-bps cost was not applied exactly once")
    output = args.out_dir / "canonical_policy_contract.parquet"
    labels.to_parquet(output, index=False)
    manifest = {
        "schema": "strict_r3_phase_h1_policy_labels_v1",
        "target_free_input": str(args.candidates), "target_free_input_sha256": _sha(args.candidates),
        "interval": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "policy": {**policy, "timeout_hours": 12, "cost_bps_once": COST_BPS},
        "rows": int(len(labels)), "valid_rows": int(valid.sum()), "invalid_rows": int((~valid).sum()),
        "output_sha256": _sha(output),
        "invalid_semantics": "retained but never numerical; excluded from fitting and realised evaluation",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
