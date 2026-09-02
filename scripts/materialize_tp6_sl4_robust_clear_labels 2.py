#!/usr/bin/env python3
"""Materialise exact pre-adverse robust-clear labels for the TP6/SL4 winner.

The label is a training-only primitive.  It measures the best favourable
movement *strictly before* the first meaningful adverse (SL=4 ATR) touch,
starting at the same exact next-minute entry as the winning H12 sidecar.
Incomplete paths are retained with ``label_valid=false`` and no numerical
target; they can be audited but cannot enter supervised fitting.
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

from scripts.materialize_full_universe_tp6_sl4_h12_sidecar import (
    DEFAULT_PANEL, HORIZON_MINUTES, ONE_MINUTE, SL_ATR, TP_ATR, _complete_paths,
    _load_candidates, _minute_path,
)


PANEL = DEFAULT_PANEL
COST_BPS = 100.0


@njit(cache=True)
def _pre_adverse_mfe(high: np.ndarray, low: np.ndarray, close: np.ndarray, starts: np.ndarray,
                      entry: np.ndarray, atr: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validity, max favourable ATR strictly before lower touch, and lower minute."""
    n = len(starts)
    valid = np.zeros(n, np.bool_)
    result = np.empty(n, np.float32); result[:] = np.nan
    lower_minute = np.full(n, -1, np.int16)
    for row in range(n):
        start, e, a, s = starts[row], entry[row], atr[row], side[row]
        if start < 0 or start + HORIZON_MINUTES > len(close) or not np.isfinite(e) or not np.isfinite(a) or e <= 0.0 or a <= 0.0:
            continue
        best = 0.0
        complete = True
        for offset in range(HORIZON_MINUTES):
            pos = start + offset
            if not np.isfinite(high[pos]) or not np.isfinite(low[pos]) or not np.isfinite(close[pos]):
                complete = False
                break
            if s > 0.0:
                favorable = (high[pos] - e) / a
                adverse = (e - low[pos]) / a
            else:
                favorable = (e - low[pos]) / a
                adverse = (high[pos] - e) / a
            # Strictly pre-adverse: a same-minute double touch belongs to the
            # adverse path and contributes no favourable evidence.
            if adverse >= SL_ATR:
                lower_minute[row] = offset + 1
                break
            if favorable > best:
                best = favorable
        if complete:
            valid[row] = True
            result[row] = best
    return valid, result, lower_minute


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35., 35.)))


def _materialise_part(part: Path, minute_root: Path) -> tuple[pd.DataFrame, int]:
    candidates = _load_candidates(part)
    symbol = str(candidates.__symbol__.iloc[0])
    path_start = candidates.__decision_ts__.min()
    path_end = candidates.__decision_ts__.max() + pd.Timedelta(minutes=HORIZON_MINUTES)
    minute = _minute_path(minute_root, symbol, path_start, path_end)
    starts = minute.index.get_indexer(candidates.__decision_ts__)
    complete = _complete_paths(minute, starts)
    entry = np.full(len(candidates), np.nan, dtype=float)
    entry[starts >= 0] = minute.open.to_numpy(float)[starts[starts >= 0]]
    stored = candidates.decision_price.to_numpy(float)
    executable = complete & np.isfinite(entry) & np.isfinite(stored) & np.isclose(entry, stored, rtol=0., atol=1e-10)
    side = np.where(candidates.side_name.eq("long").to_numpy(), 1., -1.)
    valid, pre_mfe, lower_minute = _pre_adverse_mfe(
        minute.high.to_numpy(float), minute.low.to_numpy(float), minute.close.to_numpy(float),
        starts.astype(np.int64), entry, candidates.atr_1h.to_numpy(float), side,
    )
    valid &= executable
    atr_bps = candidates.atr_1h.to_numpy(float) / entry * 10_000.
    pre_mfe_bps = pre_mfe.astype(float) * atr_bps
    out = candidates[["candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__"]].copy()
    out["label_valid"] = valid
    out["target_invalid"] = ~valid
    out["invalid_reason"] = np.where(valid, "complete_executable_path", "incomplete_or_nonexecutable_h12_path")
    out["tp6_sl4_entry_price"] = entry
    out["pre_adverse_mfe_atr"] = np.where(valid, pre_mfe, np.nan)
    out["pre_adverse_mfe_bps"] = np.where(valid, pre_mfe_bps, np.nan)
    out["lower_touch_minute"] = np.where(valid, lower_minute, -1)
    out["atr_bps"] = np.where(valid, atr_bps, np.nan)
    for buffer in (0., 25., 50.):
        margin = pre_mfe_bps - COST_BPS - buffer
        out[f"robust_clear_margin_bps_b{int(buffer)}"] = np.where(valid, margin, np.nan)
        out[f"robust_clear_event_b{int(buffer)}"] = np.where(valid, margin > 0., np.nan)
        for temp in (25., 50., 100.):
            out[f"robust_clear_soft_b{int(buffer)}_t{int(temp)}"] = np.where(valid, _sigmoid(margin / temp), np.nan)
    out["__label_available_at__"] = out.__decision_ts__ + pd.Timedelta(hours=12)
    return out, int((~valid).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--minute-root", type=Path, default=ONE_MINUTE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.resume:
        raise FileExistsError(args.out)
    destination = args.out / "parts"; destination.mkdir(parents=True, exist_ok=True)
    requested = set(args.symbol)
    report = []
    for part in sorted((args.panel / "parts").glob("*.parquet")):
        symbol = str(pd.read_parquet(part, columns=["__symbol__"])["__symbol__"].iloc[0])
        if requested and symbol not in requested:
            continue
        out_path = destination / part.name
        if out_path.exists():
            if not args.resume:
                raise FileExistsError(out_path)
            old = pd.read_parquet(out_path, columns=["candidate_id"])
            report.append({"symbol": symbol, "rows": len(old), "status": "reused"})
            continue
        out, invalid = _materialise_part(part, args.minute_root)
        out.to_parquet(out_path, index=False, compression="zstd")
        report.append({"symbol": symbol, "rows": len(out), "invalid_rows": invalid, "status": "materialised"})
        print(json.dumps(report[-1]), flush=True)
    complete = not requested and len(report) == len(list((args.panel / "parts").glob("*.parquet")))
    manifest = {"schema": "tp6_sl4_robust_clear_label_v1", "complete": complete,
                "contract": {"entry": "signal close +1h, exact next-minute open", "geometry": f"TP {TP_ATR:g} ATR / SL {SL_ATR:g} ATR / H12", "pre_adverse": "max favourable strictly before lower touch; same-minute conflict adverse", "cost_bps": COST_BPS, "buffers_bps": [0, 25, 50], "temperatures_bps": [25, 50, 100], "label_availability": "decision +12h"},
                "invalid_semantics": "invalid paths retain NaN targets and are excluded from fitting", "parts": report}
    (args.out / ("manifest.json" if complete else "checkpoint_manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
