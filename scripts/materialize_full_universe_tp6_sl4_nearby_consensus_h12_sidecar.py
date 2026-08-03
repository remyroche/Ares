#!/usr/bin/env python3
"""Materialise exact nearby-geometry consensus labels for TP6/SL4 BW3.

This is a resolved training-label sidecar only.  Each candidate is reopened at
the existing exact one-minute entry and evaluated over the same contiguous H12
path as the focal TP6/SL4 label.  It writes nine first-touch event codes for
TP={5.5,6,6.5} × SL={3.5,4,4.5}, plus the fraction agreeing with the central
6/4 event.  No result is a model feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from materialize_full_universe_tp6_sl4_h12_sidecar import (
    DEFAULT_PANEL, HORIZON_MINUTES, ONE_MINUTE, _complete_paths,
    _first_touch_tp6_sl4, _load_candidates, _minute_path,
)

TP_GRID = np.asarray((5.5, 6.0, 6.5), dtype=np.float64)
SL_GRID = np.asarray((3.5, 4.0, 4.5), dtype=np.float64)


@njit(cache=True)
def _nearby_events(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   starts: np.ndarray, entry: np.ndarray, atr: np.ndarray,
                   side: np.ndarray) -> np.ndarray:
    events = np.full((len(starts), 9), 2, np.int8)
    for row in range(len(starts)):
        start, e, a, s = starts[row], entry[row], atr[row], side[row]
        if start < 0 or start + HORIZON_MINUTES - 1 >= len(close) or not np.isfinite(e) or not np.isfinite(a) or a <= 0.:
            continue
        for index in range(9):
            tp = TP_GRID[index // 3]
            sl = SL_GRID[index % 3]
            for offset in range(HORIZON_MINUTES):
                pos = start + offset
                if not np.isfinite(high[pos]) or not np.isfinite(low[pos]):
                    break
                favorable = (high[pos] - e) / a if s > 0. else (e - low[pos]) / a
                adverse = (e - low[pos]) / a if s > 0. else (high[pos] - e) / a
                # Same adverse-first tie convention as the central sidecar.
                if adverse >= sl:
                    events[row, index] = 1
                    break
                if favorable >= tp:
                    events[row, index] = 0
                    break
    return events


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--minute-root", type=Path, default=ONE_MINUTE)
    parser.add_argument("--central-sidecar", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--symbol", action="append", default=[])
    return parser.parse_args()


def _part(part: Path, minute_root: Path, central: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    candidates = _load_candidates(part)
    symbol = str(candidates.__symbol__.iloc[0])
    minute = _minute_path(minute_root, symbol, candidates.__decision_ts__.min(), candidates.__decision_ts__.max() + pd.Timedelta(minutes=HORIZON_MINUTES))
    starts = minute.index.get_indexer(candidates.__decision_ts__)
    complete = (starts >= 0) & _complete_paths(minute, np.maximum(starts, 0))
    candidates = candidates.loc[complete].reset_index(drop=True)
    starts = starts[complete]
    entry = minute.open.to_numpy(float)[starts]
    if not np.allclose(entry, candidates.decision_price.to_numpy(float), rtol=0., atol=1e-10):
        raise ValueError(f"{symbol}: entry-open parity failed")
    side = np.where(candidates.side_name.eq("long"), 1., -1.)
    events = _nearby_events(minute.high.to_numpy(float), minute.low.to_numpy(float), minute.close.to_numpy(float), starts.astype(np.int64), entry, candidates.atr_1h.to_numpy(float), side)
    central_events, _, central_pnl, _ = _first_touch_tp6_sl4(minute.high.to_numpy(float), minute.low.to_numpy(float), minute.close.to_numpy(float), starts.astype(np.int64), entry, candidates.atr_1h.to_numpy(float), side)
    if not np.isfinite(central_pnl).all() or not np.array_equal(events[:, 4], central_events):
        raise AssertionError(f"{symbol}: centre contract does not reproduce TP6/SL4")
    expected = central.set_index("candidate_id").reindex(candidates.candidate_id)
    if expected.t2_tp6_sl4_event.isna().any() or not np.array_equal(expected.t2_tp6_sl4_event.to_numpy(np.int8), central_events):
        raise AssertionError(f"{symbol}: frozen central sidecar parity failed")
    output = candidates[["candidate_id", "__ts__", "__decision_ts__"]].copy()
    for index, (tp, sl) in enumerate(((tp, sl) for tp in TP_GRID for sl in SL_GRID)):
        output[f"t2_tp{tp:g}_sl{sl:g}_event"] = events[:, index]
    output["t2_tp6_sl4_event"] = central_events
    output["tp6_sl4_contract_consensus"] = (events == central_events[:, None]).mean(axis=1).astype(np.float32)
    output["tp6_sl4_contract_mode_fraction"] = np.maximum.reduce([(events == value).mean(axis=1) for value in (0, 1, 2)]).astype(np.float32)
    output["__label_available_at__"] = output.__decision_ts__ + pd.Timedelta(hours=12)
    return output, int((~complete).sum())


def main() -> None:
    args = _args()
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(args.out_dir)
    central_parts = sorted((args.central_sidecar / "parts").glob("*.parquet"))
    if not central_parts:
        raise FileNotFoundError("central TP6/SL4 sidecar has no parts")
    central_by_name = {path.name: pd.read_parquet(path, columns=["candidate_id", "t2_tp6_sl4_event"]) for path in central_parts}
    destination = args.out_dir / "parts"; destination.mkdir(parents=True, exist_ok=True)
    report = []
    requested = set(args.symbol)
    for part in sorted((args.panel / "parts").glob("*.parquet")):
        symbol = str(pd.read_parquet(part, columns=["__symbol__"])["__symbol__"].iloc[0])
        if requested and symbol not in requested:
            continue
        target = destination / part.name
        if target.exists():
            report.append({"symbol": symbol, "status": "reused"}); continue
        output, excluded = _part(part, args.minute_root, central_by_name[part.name])
        output.to_parquet(target, index=False, compression="zstd")
        report.append({"symbol": symbol, "rows": len(output), "excluded_incomplete_h12_paths": excluded})
        print(json.dumps(report[-1]), flush=True)
    complete = not requested and len(report) == len(list((args.panel / "parts").glob("*.parquet")))
    (args.out_dir / ("manifest.json" if complete else "checkpoint_manifest.json")).write_text(json.dumps({
        "schema": "full_universe_tp6_sl4_nearby_consensus_h12_sidecar_v1", "complete": complete,
        "geometries": {"tp_atr": TP_GRID.tolist(), "sl_atr": SL_GRID.tolist()},
        "central_index": 4, "tie_policy": "adverse-first", "horizon_minutes": HORIZON_MINUTES,
        "label_availability": "decision_ts + 12h", "training_only": True, "parts": report,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
