#!/usr/bin/env python3
"""Materialise a declared exact one-minute TP/SL/H horizon sidecar.

Unlike the frozen TP6/SL4 producer, this writes generic outcome names and a
fully explicit manifest.  It is therefore safe to use for nearby-contract
stability audits without allowing a different geometry to masquerade as the
selected winner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.materialize_full_universe_tp6_sl4_h12_sidecar import DEFAULT_PANEL, ONE_MINUTE, ROUND_TRIP_COST_BPS, _load_candidates, _minute_path


@njit(cache=True)
def _first_touch(high: np.ndarray, low: np.ndarray, close: np.ndarray, starts: np.ndarray, entry: np.ndarray, atr: np.ndarray, side: np.ndarray, tp: float, sl: float, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(starts); event = np.full(n, 2, np.int8); exit_minute = np.full(n, horizon, np.int16); pnl = np.empty(n, np.float32); pnl[:] = np.nan
    for row in range(n):
        start = starts[row]
        if start < 0 or start + horizon - 1 >= len(close) or not np.isfinite(entry[row]) or not np.isfinite(atr[row]) or atr[row] <= 0:
            continue
        resolved = False
        for offset in range(horizon):
            pos = start + offset
            if not np.isfinite(high[pos]) or not np.isfinite(low[pos]) or not np.isfinite(close[pos]):
                resolved = True; break
            if side[row] > 0:
                favorable = (high[pos] - entry[row]) / atr[row]; adverse = (entry[row] - low[pos]) / atr[row]
            else:
                favorable = (entry[row] - low[pos]) / atr[row]; adverse = (high[pos] - entry[row]) / atr[row]
            if not resolved and adverse >= sl:
                event[row] = 1; exit_minute[row] = offset + 1; pnl[row] = -sl; resolved = True
            elif not resolved and favorable >= tp:
                event[row] = 0; exit_minute[row] = offset + 1; pnl[row] = tp; resolved = True
        if not resolved and np.isfinite(close[start + horizon - 1]):
            pnl[row] = side[row] * (close[start + horizon - 1] - entry[row]) / atr[row]
    return event, exit_minute, pnl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--panel", type=Path, default=DEFAULT_PANEL); p.add_argument("--minute-root", type=Path, default=ONE_MINUTE)
    p.add_argument("--tp-atr", type=float, required=True); p.add_argument("--sl-atr", type=float, required=True); p.add_argument("--horizon-hours", type=float, required=True)
    p.add_argument("--symbol", action="append", default=[]); p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    horizon = int(round(a.horizon_hours * 60))
    if a.tp_atr <= 0 or a.sl_atr <= 0 or horizon <= 0 or not np.isclose(horizon, a.horizon_hours * 60): raise ValueError("positive whole-minute geometry required")
    if a.out.exists() and not a.resume: raise FileExistsError(a.out)
    out_parts = a.out / "parts"; out_parts.mkdir(parents=True, exist_ok=True); wanted = set(a.symbol); report=[]
    for part in sorted((a.panel / "parts").glob("*.parquet")):
        symbol = str(pd.read_parquet(part, columns=["__symbol__"])["__symbol__"].iloc[0])
        if wanted and symbol not in wanted: continue
        dst = out_parts / part.name
        if dst.exists():
            if not a.resume: raise FileExistsError(dst)
            report.append({"symbol":symbol,"status":"reused"}); continue
        candidates = _load_candidates(part); start = candidates.__decision_ts__.min(); end = candidates.__decision_ts__.max() + pd.Timedelta(minutes=horizon)
        minute = _minute_path(a.minute_root, symbol, start, end); starts = minute.index.get_indexer(candidates.__decision_ts__)
        entry = np.full(len(candidates), np.nan); valid_start = starts >= 0; entry[valid_start] = minute.open.to_numpy(float)[starts[valid_start]]
        side = np.where(candidates.side_name.eq("long"), 1., -1.); event, exit_minute, pnl = _first_touch(minute.high.to_numpy(float), minute.low.to_numpy(float), minute.close.to_numpy(float), starts.astype(np.int64), entry, candidates.atr_1h.to_numpy(float), side, a.tp_atr, a.sl_atr, horizon)
        # Label validity is deliberately stricter than path resolution.  A TP/SL
        # may occur before a later missing minute, but the declared H-hour
        # contract is still incomplete and must not become an ordinary outcome.
        # This keeps missing supervision distinct from a timeout or a loss.
        ohlc = minute[["open", "high", "low", "close"]].to_numpy(float)
        finite = np.isfinite(ohlc).all(axis=1).astype(np.int64)
        cumulative_finite = np.concatenate(([0], np.cumsum(finite)))
        complete_path = np.zeros(len(candidates), dtype=bool)
        in_bounds = valid_start & (starts + horizon <= len(minute))
        good_starts = starts[in_bounds]
        complete_path[in_bounds] = (
            cumulative_finite[good_starts + horizon] - cumulative_finite[good_starts]
        ) == horizon
        valid = complete_path & np.isfinite(pnl) & np.isclose(entry, candidates.decision_price.to_numpy(float), rtol=0., atol=1e-10)
        gross = pnl.astype(float) * candidates.atr_1h.to_numpy(float) / entry * 10_000.
        out = candidates[["candidate_id","__ts__","__symbol__","side_name","__decision_ts__"]].copy(); out["label_valid"] = valid; out["event"] = np.where(valid,event,np.nan); out["exit_minute"] = np.where(valid,exit_minute,np.nan); out["gross_bps"] = np.where(valid,gross,np.nan); out["net_bps"] = np.where(valid,gross-ROUND_TRIP_COST_BPS,np.nan); out["__label_available_at__"] = out.__decision_ts__ + pd.Timedelta(minutes=horizon)
        out.to_parquet(dst,index=False,compression="zstd"); report.append({"symbol":symbol,"rows":len(out),"valid":int(valid.sum()),"status":"materialised"}); print(json.dumps(report[-1]),flush=True)
    complete = not wanted and len(report)==len(list((a.panel/'parts').glob('*.parquet')))
    manifest={"schema":"exact_geometry_sidecar_v1","complete":complete,"contract":{"tp_atr":a.tp_atr,"sl_atr":a.sl_atr,"horizon_minutes":horizon,"entry":"signal close +1h exact next-minute open","same_minute_conflict":"adverse first","cost_bps":ROUND_TRIP_COST_BPS},"parts":report}
    (a.out/("manifest.json" if complete else "checkpoint_manifest.json")).write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__": main()
