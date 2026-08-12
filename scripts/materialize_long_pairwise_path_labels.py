#!/usr/bin/env python3
"""Exact H12 geometry/path labels for the long executable-net rank ablation.

The sidecar is deliberately label-only.  It binds to the current long
candidate IDs, reopens the entry minute at ``signal + 1h``, and rejects any
incomplete path.  It never writes path outcomes into model inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_full_universe_tp6_sl4_h12_sidecar import (
    HORIZON_MINUTES, ONE_MINUTE, _complete_paths, _minute_path,
)

SIDE = "long"
ROUND_TRIP_COST_BPS = 100.0


@njit(cache=True)
def _path_statistics(high, low, close, starts, entry, atr):
    """Compute all required path primitives in one compiled, sequential pass."""
    n = len(starts)
    valid = np.zeros(n, np.bool_)
    terminal = np.full(n, np.nan, np.float32)
    max_mfe = np.full(n, np.nan, np.float32)
    mae_before_peak = np.full(n, np.nan, np.float32)
    peak_minute = np.full(n, -1, np.int16)
    tp4 = np.full(n, -1, np.int16); tp6 = np.full(n, -1, np.int16)
    sl3 = np.full(n, -1, np.int16); sl5 = np.full(n, -1, np.int16)
    for row in range(n):
        start, e, a = starts[row], entry[row], atr[row]
        if start < 0 or start + HORIZON_MINUTES > len(close) or not np.isfinite(e) or not np.isfinite(a) or e <= 0.0 or a <= 0.0:
            continue
        best, prior_mae = 0.0, 0.0
        peak_mae = 0.0
        complete = True
        for offset in range(HORIZON_MINUTES):
            pos = start + offset
            if not np.isfinite(high[pos]) or not np.isfinite(low[pos]) or not np.isfinite(close[pos]):
                complete = False
                break
            favorable = (high[pos] - e) / a
            adverse = (e - low[pos]) / a
            # Pre-peak MAE is strictly prior to the minute that makes a new
            # high; a same-minute high/low conflict is never optimistically
            # treated as risk-free.
            if favorable > best:
                best = favorable
                peak_mae = prior_mae
                peak_minute[row] = offset + 1
            if adverse > prior_mae:
                prior_mae = adverse
            if tp4[row] < 0 and favorable >= 4.0: tp4[row] = offset + 1
            if tp6[row] < 0 and favorable >= 6.0: tp6[row] = offset + 1
            if sl3[row] < 0 and adverse >= 3.0: sl3[row] = offset + 1
            if sl5[row] < 0 and adverse >= 5.0: sl5[row] = offset + 1
        if complete:
            valid[row] = True
            max_mfe[row] = best
            mae_before_peak[row] = peak_mae
            terminal[row] = (close[start + HORIZON_MINUTES - 1] - e) / a
    return valid, terminal, max_mfe, mae_before_peak, peak_minute, tp4, tp6, sl3, sl5


def _barrier_grade(tp4, tp6, sl3, sl5, terminal_net_bps):
    """Declared 0--5 barrier-aware relevance; only unresolved rows use H12 PnL."""
    n = len(tp4)
    grade = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        # The wide contract is examined first: it separates a genuinely deep
        # loss from a path that merely touched the 3-ATR control stop.
        if sl5[i] >= 0 and (tp6[i] < 0 or sl5[i] <= tp6[i]):
            grade[i] = 0
        elif tp6[i] >= 0 and (sl5[i] < 0 or tp6[i] < sl5[i]):
            grade[i] = 5
        elif sl3[i] >= 0 and (tp4[i] < 0 or sl3[i] <= tp4[i]):
            grade[i] = 1
        elif tp4[i] >= 0 and (sl3[i] < 0 or tp4[i] < sl3[i]):
            grade[i] = 4
        elif terminal_net_bps[i] <= -50.0:
            grade[i] = 1
        elif terminal_net_bps[i] <= 50.0:
            grade[i] = 2
        else:
            grade[i] = 3
    return grade


def _ratio_grade(mfe, mae_before_peak, atr_return):
    """MFE/MAE relevance for an economically material H12 opportunity.

    Both conditions are deliberately required: an ATR-relative move above three
    ATR and an absolute favourable move of at least 1.5%.  The latter prevents
    a tiny absolute move in an ultra-low-ATR regime from being treated as
    supervised path support.
    """
    favourable_move_fraction = mfe * atr_return
    eligible = (
        np.isfinite(mfe)
        & np.isfinite(mae_before_peak)
        & np.isfinite(favourable_move_fraction)
        & (mfe > 3.0)
        & (favourable_move_fraction >= 0.015)
    )
    safe_mae = np.maximum(mae_before_peak, 0.05)
    ratio = np.where(eligible, mfe / safe_mae, np.nan)
    grade = np.full(len(mfe), -1, dtype=np.int8)
    bins = [0.7, 1.3, 1.6, 2.0, 2.5]
    for value, pos in zip(ratio[eligible], np.flatnonzero(eligible)):
        grade[pos] = 0 if value < bins[0] else 1 if value < bins[1] else 2 if value < bins[2] else 3 if value < bins[3] else 4 if value < bins[4] else 5
    return (
        eligible,
        np.minimum(ratio, 25.0).astype(np.float32),
        grade,
        favourable_move_fraction.astype(np.float32),
    )


def _candidates_from_part(part: Path, candidate_ids: set[str]) -> pd.DataFrame:
    cols = ["candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__", "decision_price", "atr_1h"]
    frame = pd.read_parquet(part, columns=cols)
    frame = frame.loc[frame["candidate_id"].isin(candidate_ids) & frame["side_name"].astype(str).str.lower().eq(SIDE)].copy()
    if frame.empty:
        return frame
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    if frame["candidate_id"].duplicated().any() or frame["__symbol__"].nunique() != 1:
        raise ValueError(f"invalid candidate identity/symbol surface in {part}")
    if not frame["__decision_ts__"].eq(frame["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("entry timestamp must equal signal close + 1h")
    return frame


def _one_part(part: Path, candidate_ids: set[str], minute_root: Path) -> pd.DataFrame:
    candidates = _candidates_from_part(part, candidate_ids)
    if candidates.empty:
        return candidates
    symbol = str(candidates["__symbol__"].iloc[0])
    start = candidates["__decision_ts__"].min()
    end = candidates["__decision_ts__"].max() + pd.Timedelta(minutes=HORIZON_MINUTES)
    minute = _minute_path(minute_root, symbol, start, end)
    starts = minute.index.get_indexer(candidates["__decision_ts__"])
    complete = _complete_paths(minute, starts)
    entry = np.full(len(candidates), np.nan, dtype=np.float64)
    entry[starts >= 0] = minute.open.to_numpy(float)[starts[starts >= 0]]
    expected = candidates["decision_price"].to_numpy(float)
    executable = complete & np.isfinite(entry) & np.isclose(entry, expected, rtol=0.0, atol=1e-10)
    valid, terminal_atr, mfe, mae_pre_peak, peak_minute, tp4, tp6, sl3, sl5 = _path_statistics(
        minute.high.to_numpy(float), minute.low.to_numpy(float), minute.close.to_numpy(float),
        starts.astype(np.int64), entry, candidates["atr_1h"].to_numpy(float),
    )
    valid &= executable
    atr_return = candidates["atr_1h"].to_numpy(float) / entry
    terminal_net_bps = terminal_atr.astype(float) * atr_return * 10_000.0 - ROUND_TRIP_COST_BPS
    barrier = _barrier_grade(tp4, tp6, sl3, sl5, terminal_net_bps)
    ratio_valid, ratio, ratio_grade, favourable_move_fraction = _ratio_grade(
        mfe.astype(float), mae_pre_peak.astype(float), atr_return
    )
    out = candidates.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__"]].copy()
    out["label_valid"] = valid
    out["barrier_relevance_0_5"] = np.where(valid, barrier, -1).astype(np.int8)
    out["mfe_mae_label_valid"] = valid & ratio_valid
    # Explicit material-path support contract consumed by B4.  It is a
    # training/diagnostic condition only and never an inference-time filter.
    out["support_h12_mfe_mae"] = valid & ratio_valid
    out["mfe_mae_relevance_0_5"] = np.where(valid & ratio_valid, ratio_grade, -1).astype(np.int8)
    out["terminal_h12_net_bps"] = np.where(valid, terminal_net_bps, np.nan).astype(np.float32)
    out["peak_mfe_atr"] = np.where(valid, mfe, np.nan).astype(np.float32)
    out["peak_favourable_move_pct"] = np.where(
        valid, favourable_move_fraction * 100.0, np.nan
    ).astype(np.float32)
    out["mae_before_peak_atr"] = np.where(valid, mae_pre_peak, np.nan).astype(np.float32)
    out["mfe_mae_ratio"] = np.where(valid & ratio_valid, ratio, np.nan).astype(np.float32)
    out["atr_bps"] = np.where(valid, atr_return * 10_000.0, np.nan).astype(np.float32)
    out["first_tp4_minute"] = np.where(valid, tp4, -1).astype(np.int16)
    out["first_tp6_minute"] = np.where(valid, tp6, -1).astype(np.int16)
    out["first_sl3_minute"] = np.where(valid, sl3, -1).astype(np.int16)
    out["first_sl5_minute"] = np.where(valid, sl5, -1).astype(np.int16)
    out["__label_available_at__"] = out["__decision_ts__"] + pd.Timedelta(hours=12)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    parser.add_argument("--minute-root", type=Path, default=ONE_MINUTE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--symbol", action="append", default=[], help="optional exact symbol partition; repeatable")
    parser.add_argument("--skip-manifest", action="store_true", help="for parallel partition workers; final manifest is written by the coordinator")
    parser.add_argument("--bucket-index", type=int, default=None, help="zero-based deterministic partition-worker index")
    parser.add_argument("--bucket-count", type=int, default=None, help="number of deterministic partition workers")
    args = parser.parse_args()
    ledger = pd.read_parquet(args.ledger, columns=["candidate_id", "side_name"])
    candidate_ids = set(ledger.loc[ledger["side_name"].astype(str).str.lower().eq(SIDE), "candidate_id"].astype(str))
    out_parts = args.out_dir / "parts"; out_parts.mkdir(parents=True, exist_ok=True)
    requested = set(map(str, args.symbol))
    if (args.bucket_index is None) != (args.bucket_count is None):
        raise ValueError("bucket-index and bucket-count must be supplied together")
    if args.bucket_count is not None and not (0 <= args.bucket_index < args.bucket_count):
        raise ValueError("invalid deterministic worker bucket")
    records = []
    for part_index, part in enumerate(sorted((args.panel / "parts").glob("*.parquet"))):
        if args.bucket_count is not None and part_index % args.bucket_count != args.bucket_index:
            continue
        symbol = str(pd.read_parquet(part, columns=["__symbol__"])["__symbol__"].iloc[0])
        if requested and symbol not in requested:
            continue
        destination = out_parts / part.name
        if destination.exists():
            if not args.resume:
                raise FileExistsError(destination)
            prior = pd.read_parquet(destination, columns=["candidate_id"])
            records.append({"symbol": symbol, "rows": len(prior), "status": "reused"})
            continue
        output = _one_part(part, candidate_ids, args.minute_root)
        if output.empty:
            continue
        output.to_parquet(destination, index=False, compression="zstd")
        records.append({"symbol": symbol, "rows": len(output), "valid": int(output.label_valid.sum()), "ratio_eligible": int(output.mfe_mae_label_valid.sum()), "status": "materialized"})
        print(json.dumps(records[-1]), flush=True)
    if args.skip_manifest:
        return
    combined_ids = set()
    for p in out_parts.glob("*.parquet"):
        combined_ids.update(pd.read_parquet(p, columns=["candidate_id"])["candidate_id"].astype(str))
    missing = len(candidate_ids.difference(combined_ids))
    manifest = {
        "schema": "long_pairwise_path_labels_v1", "side": SIDE,
        "candidate_ids": len(candidate_ids), "materialized_ids": len(combined_ids), "missing_candidate_ids": missing,
        "entry": "signal close + 1h exact next-minute open; equality-validated to panel decision price",
        "horizon": "12h contiguous minute path", "label_available": "entry + 12h",
        "barrier_relevance": "0 severe: SL5 before TP6; 1 mild: SL3 before TP4 or timeout net <=-50; 2 timeout net [-50,+50]; 3 timeout net >+50; 4 TP4 before SL3; 5 TP6 before SL5",
        "mfe_mae_relevance": "peak MFE / strictly pre-peak MAE; only MFE >3 ATR and peak gross move >=1.5%; MAE denominator floor=0.05 ATR; grades [<.7, .7-1.3, 1.3-1.6, 1.6-2, 2-2.5, >=2.5]",
        "paths": records,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
