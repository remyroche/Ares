#!/usr/bin/env python3
"""Materialise exact TP6/SL4 H12 labels beside the full-universe panel.

This is a standalone *label sidecar*, not a feature or model trainer.  It
binds every output row to the existing ``candidate_id`` and reopens the exact
one-minute entry bar at ``__decision_ts__``.  The contract is deliberately
identical to the frozen T2/T4 panel except for its TP=6 ATR, SL=4 ATR
geometry:

* entry is the next-minute open one completed hour after signal close;
* first TP/SL touch wins, with SL precedence on a same-minute conflict;
* no-hit trades settle at minute 720; and
* all labels become available exactly 12 hours after entry.

The stored terminal return is always the side-normalised minute-720 close,
including rows that hit an earlier barrier.  It therefore supplies a true
terminal-H12 label rather than reusing the stopped exit PnL.

Run with ``--resume`` after an interrupted full build.  A completed partition
is immutable: reruns validate it rather than overwrite it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from numba import njit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
ONE_MINUTE = ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv"
TP_ATR = 6.0
SL_ATR = 4.0
HORIZON_MINUTES = 720
ROUND_TRIP_COST_BPS = 100.0
LABEL_COLUMNS = (
    "t2_tp6_sl4_event", "t2_tp6_sl4_exit_minute", "t4_tp6_sl4_exit_pnl_atr",
    "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "t4_tp6_sl4_terminal_pnl_atr",
)


@njit(cache=True)
def _first_touch_tp6_sl4(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         starts: np.ndarray, entry: np.ndarray,
                         atr: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all TP6/SL4 labels in one compiled pass over each H12 path."""
    n = len(starts)
    event = np.full(n, 2, np.int8)  # 0=TP, 1=SL, 2=timeout
    exit_minute = np.full(n, HORIZON_MINUTES, np.int16)
    exit_pnl = np.empty(n, np.float32)
    terminal_pnl = np.empty(n, np.float32)
    exit_pnl[:] = np.nan
    terminal_pnl[:] = np.nan
    for row in range(n):
        start = starts[row]
        if start < 0 or start + HORIZON_MINUTES - 1 >= len(close):
            continue
        e = entry[row]
        a = atr[row]
        if not np.isfinite(e) or not np.isfinite(a) or a <= 0.0:
            continue
        s = side[row]
        resolved = False
        for offset in range(HORIZON_MINUTES):
            pos = start + offset
            if not np.isfinite(high[pos]) or not np.isfinite(low[pos]) or not np.isfinite(close[pos]):
                resolved = True  # remains NaN: callers reject incomplete paths
                exit_pnl[row] = np.nan
                break
            if s > 0.0:
                favorable = (high[pos] - e) / a
                adverse = (e - low[pos]) / a
            else:
                favorable = (e - low[pos]) / a
                adverse = (high[pos] - e) / a
            # Exact contract: adverse checks first, so a minute containing
            # both touches is an SL conflict, not an optimistic TP fill.
            if not resolved and adverse >= SL_ATR:
                event[row] = 1
                exit_minute[row] = offset + 1
                exit_pnl[row] = -SL_ATR
                resolved = True
            elif not resolved and favorable >= TP_ATR:
                event[row] = 0
                exit_minute[row] = offset + 1
                exit_pnl[row] = TP_ATR
                resolved = True
        if np.isfinite(close[start + HORIZON_MINUTES - 1]):
            terminal_pnl[row] = s * (close[start + HORIZON_MINUTES - 1] - e) / a
            if not resolved:
                exit_pnl[row] = terminal_pnl[row]
    return event, exit_minute, exit_pnl, terminal_pnl


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--minute-root", type=Path, default=ONE_MINUTE)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true", help="emit only candidates with exact contiguous H12 paths and record excluded rows")
    p.add_argument("--symbol", action="append", default=[], help="optional explicit symbol, repeatable")
    return p.parse_args()


def _load_candidates(part: Path) -> pd.DataFrame:
    cols = ["candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__", "decision_price", "atr_1h"]
    x = pd.read_parquet(part, columns=cols)
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    x["__decision_ts__"] = pd.to_datetime(x["__decision_ts__"], utc=True)
    if x.candidate_id.duplicated().any() or len(x) == 0:
        raise ValueError(f"invalid candidate identity in {part}")
    if x.__symbol__.nunique() != 1:
        raise ValueError(f"panel partition must contain exactly one symbol: {part}")
    expected = x.__ts__ + pd.Timedelta(hours=1)
    if not x.__decision_ts__.eq(expected).all():
        raise ValueError(f"candidate decision timestamp is not signal-close + 1h: {part}")
    if not x.side_name.isin(("long", "short")).all():
        raise ValueError(f"unknown side in {part}")
    return x


def _minute_path(root: Path, symbol: str, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    location = root / f"symbol={symbol}"
    if not location.exists():
        raise FileNotFoundError(f"minute OHLC root absent for {symbol}: {location}")
    years = list(range(start.year, (end_exclusive - pd.Timedelta(minutes=1)).year + 1))
    table = ds.dataset(location, format="parquet", partitioning="hive").to_table(
        filter=(ds.field("year").isin(years)) & (ds.field("ts") >= start) & (ds.field("ts") < end_exclusive),
        columns=["ts", "open", "high", "low", "close"],
    )
    raw = table.to_pandas()
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    raw = raw.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    # Reindex to an exact 1m grid.  Missing bars become NaN and are rejected,
    # never silently skipped by positional path traversal.
    grid = pd.date_range(start.floor("min"), (end_exclusive - pd.Timedelta(minutes=1)).floor("min"), freq="min", tz="UTC")
    return raw.reindex(grid)


def _complete_paths(minute: pd.DataFrame, starts: np.ndarray) -> np.ndarray:
    finite = np.isfinite(minute[["open", "high", "low", "close"]].to_numpy(float)).all(axis=1).astype(np.int64)
    cumulative = np.r_[0, np.cumsum(finite)]
    return cumulative[starts + HORIZON_MINUTES] - cumulative[starts] == HORIZON_MINUTES


def _materialise_part(part: Path, minute_root: Path, *, allow_incomplete: bool) -> tuple[pd.DataFrame, int]:
    candidates = _load_candidates(part)
    symbol = str(candidates.__symbol__.iloc[0])
    path_start = candidates.__decision_ts__.min()
    path_end = candidates.__decision_ts__.max() + pd.Timedelta(minutes=HORIZON_MINUTES)
    minute = _minute_path(minute_root, symbol, path_start, path_end)
    starts = minute.index.get_indexer(candidates.__decision_ts__)
    if (starts < 0).any():
        raise ValueError("exact next-minute entry timestamp absent from minute history")
    complete = _complete_paths(minute, starts)
    excluded = int((~complete).sum())
    if excluded and not allow_incomplete:
        raise ValueError(f"{excluded} candidate paths lack an exact, contiguous H12 minute history")
    if excluded:
        candidates = candidates.iloc[np.flatnonzero(complete)].reset_index(drop=True)
        starts = starts[complete]
    entry = minute.open.to_numpy(float)[starts]
    stored = candidates.decision_price.to_numpy(float)
    if not np.allclose(entry, stored, rtol=0., atol=1e-10, equal_nan=False):
        raise ValueError("panel decision_price does not equal exact reopened next-minute entry open")
    side = np.where(candidates.side_name.eq("long").to_numpy(), 1., -1.)
    event, exit_minute, exit_pnl_atr, terminal_pnl_atr = _first_touch_tp6_sl4(
        minute.high.to_numpy(float), minute.low.to_numpy(float), minute.close.to_numpy(float),
        starts.astype(np.int64), entry, candidates.atr_1h.to_numpy(float), side,
    )
    if not np.isfinite(exit_pnl_atr).all() or not np.isfinite(terminal_pnl_atr).all():
        raise ValueError("TP6/SL4 materialiser emitted incomplete labels")
    gross = exit_pnl_atr.astype(np.float64) * candidates.atr_1h.to_numpy(float) / entry * 10_000.
    out = candidates[["candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__"]].copy()
    out["tp6_sl4_entry_price"] = entry.astype(np.float64)
    out["t2_tp6_sl4_event"] = event
    out["t2_tp6_sl4_exit_minute"] = exit_minute
    out["t4_tp6_sl4_exit_pnl_atr"] = exit_pnl_atr
    out["t4_tp6_sl4_gross_bps"] = gross.astype(np.float32)
    out["t4_tp6_sl4_net_bps"] = (gross - ROUND_TRIP_COST_BPS).astype(np.float32)
    out["t4_tp6_sl4_terminal_pnl_atr"] = terminal_pnl_atr
    out["__label_available_at__"] = out.__decision_ts__ + pd.Timedelta(hours=12)
    if not np.allclose(out.t4_tp6_sl4_gross_bps.to_numpy(float) - ROUND_TRIP_COST_BPS, out.t4_tp6_sl4_net_bps.to_numpy(float), atol=2e-3, rtol=0.):
        raise AssertionError("gross/net identity failed")
    return out, excluded


def _validate_existing(path: Path) -> None:
    needed = ["candidate_id", "__decision_ts__", *LABEL_COLUMNS]
    x = pd.read_parquet(path, columns=needed)
    if len(x) == 0 or x.candidate_id.duplicated().any() or x.loc[:, LABEL_COLUMNS].isna().any().any():
        raise ValueError(f"incomplete existing sidecar partition: {path}")


def main() -> None:
    a = _args()
    if not (a.panel / "parts").exists():
        raise FileNotFoundError(f"not a full-universe panel: {a.panel}")
    if a.out_dir.exists() and not a.resume:
        raise FileExistsError(f"output already exists; use --resume after validating scope: {a.out_dir}")
    parts_out = a.out_dir / "parts"
    parts_out.mkdir(parents=True, exist_ok=True)
    requested = set(a.symbol)
    report = []
    for part in sorted((a.panel / "parts").glob("*.parquet")):
        # Reading this tiny identity projection before the minute path is a
        # safe filter and prevents a symbol-selection glob from changing scope.
        symbol = str(pd.read_parquet(part, columns=["__symbol__"])["__symbol__"].iloc[0])
        if requested and symbol not in requested:
            continue
        destination = parts_out / part.name
        if destination.exists():
            if not a.resume:
                raise FileExistsError(destination)
            _validate_existing(destination)
            report.append({"symbol": symbol, "rows": int(len(pd.read_parquet(destination, columns=["candidate_id"]))), "status": "reused"})
            continue
        out, excluded = _materialise_part(part, a.minute_root, allow_incomplete=bool(a.allow_incomplete))
        out.to_parquet(destination, index=False, compression="zstd")
        report.append({"symbol": symbol, "rows": int(len(out)), "excluded_incomplete_h12_paths": excluded, "status": "materialised"})
        print(json.dumps({"event": "symbol_complete", **report[-1]}), flush=True)
    missing = requested.difference({x["symbol"] for x in report})
    if missing:
        raise ValueError(f"requested symbols absent from panel: {sorted(missing)}")
    complete = not requested and len(report) == len(list((a.panel / "parts").glob("*.parquet")))
    manifest = {
        "schema": "full_universe_tp6_sl4_h12_sidecar_v1", "complete": complete,
        "source_panel": str(a.panel), "minute_root": str(a.minute_root),
        "entry": "bound candidate __decision_ts__ = signal close + 1h; exact one-minute open, equality-validated against panel decision_price",
        "exit": "TP=+6 ATR / SL=-4 ATR; first touch; same-minute TP/SL conflict resolves adversely to SL; timeout at H12",
        "terminal": "side-normalised H12 close return in ATR, retained even after an earlier barrier exit",
        "cost": {"round_trip_bps": ROUND_TRIP_COST_BPS, "net_formula": "gross_bps - 100"},
        "label_availability": "__decision_ts__ + 12h", "outputs": list(LABEL_COLUMNS), "parts": report,
    }
    (a.out_dir / ("manifest.json" if complete else "checkpoint_manifest.json")).write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
