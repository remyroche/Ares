#!/usr/bin/env python3
"""Causal 14-day side-local EV admission replay for the v3 arms.

This is an evaluation utility, not a replacement production map.  It uses
only the v3 OOS predictions and labels resolved in the preceding 14 days.  The
first 14 OOS days are deliberately unavailable because no prior arm scores
exist in this artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_pair_condition_specialists import (
    PATH_ARTIFACT,
    PATH_ROOT,
    _source,
    net_bps,
    simulate_h12_stop_trailing_grid,
)
from scripts.run_broad_multiview_specialist_lambdarank import _base


LOOKBACK = pd.Timedelta(days=14)
MIN_GLOBAL_ROWS = 32
MIN_BIN_ROWS = 20
MAP_BINS = 20
ADMISSION_BPS = 20.0
TAILS = (0.01, 0.05, 0.10)


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Small weighted non-decreasing isotonic solver."""
    level: list[float] = []
    mass: list[float] = []
    start: list[int] = []
    end: list[int] = []
    for i, (value, weight) in enumerate(zip(values, weights)):
        level.append(float(value)); mass.append(float(weight)); start.append(i); end.append(i + 1)
        while len(level) >= 2 and level[-2] > level[-1]:
            w = mass[-2] + mass[-1]
            level[-2] = (level[-2] * mass[-2] + level[-1] * mass[-1]) / w
            mass[-2] = w; end[-2] = end[-1]
            level.pop(); mass.pop(); start.pop(); end.pop()
    out = np.empty(len(values), dtype=float)
    for value, left, right in zip(level, start, end):
        out[left:right] = value
    return out


def rolling_side_map(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Map scores with a causal rolling 14-day side-local EV map.

    Bin boundaries and payoffs are recomputed for each UTC day from rows whose
    labels were available before that day's start.  This prevents later score
    distribution shifts (especially on the short side) from being interpreted
    through stale bins learned during the warm-up period.
    """
    work = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True).copy()
    ts = pd.to_datetime(work["__ts__"], utc=True).to_numpy(dtype="datetime64[ns]")
    available = pd.to_datetime(work["label_available_ts"], utc=True).to_numpy(dtype="datetime64[ns]")
    score = pd.to_numeric(work[score_col], errors="coerce").to_numpy(float)
    net = pd.to_numeric(work["net_bps"], errors="coerce").to_numpy(float)
    finite_score = np.isfinite(score)
    if finite_score.sum() < MIN_GLOBAL_ROWS:
        work["ev14_mapped_bps"] = np.nan; work["ev14_map_support"] = 0; work["ev14_map_available"] = False; work["ev14_side_mean_bps"] = np.nan; work["ev14_side_gate"] = False; work["ev14_admitted"] = False
        return work

    available_ns = available.astype("datetime64[ns]").astype("int64")
    available_valid = ~np.isnat(available)
    order_available = np.argsort(np.where(available_valid, available_ns, np.iinfo(np.int64).max), kind="stable")
    sorted_available = available_ns[order_available]
    day_values = pd.to_datetime(ts, utc=True).normalize().to_numpy(dtype="datetime64[ns]")
    mapped = np.full(len(work), np.nan, dtype=np.float32); support = np.zeros(len(work), dtype=np.int32)
    side_mean = np.full(len(work), np.nan, dtype=np.float32)
    for day in np.unique(day_values):
        day_ns = day.astype("datetime64[ns]").astype("int64")
        cutoff_ns = day_ns - int(LOOKBACK.value)
        left = int(np.searchsorted(sorted_available, cutoff_ns, side="left"))
        right = int(np.searchsorted(sorted_available, day_ns, side="left"))
        hist_idx = order_available[left:right]
        hist_idx = hist_idx[available_valid[hist_idx] & finite_score[hist_idx] & np.isfinite(net[hist_idx])]
        total = int(len(hist_idx))
        if total < MIN_GLOBAL_ROWS:
            continue
        hist_scores = score[hist_idx]
        quantiles = np.nanquantile(hist_scores, np.linspace(0.0, 1.0, MAP_BINS + 1))
        edges = np.unique(quantiles)
        if len(edges) < 3:
            lo, hi = float(np.nanmin(hist_scores)), float(np.nanmax(hist_scores))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if hi <= lo:
                hi = lo + 1e-6
            edges = np.linspace(lo, hi, MAP_BINS + 1)
        n_bins = len(edges) - 1
        hist_bins = np.clip(np.searchsorted(edges[1:-1], hist_scores, side="right"), 0, n_bins - 1)
        counts = np.bincount(hist_bins, minlength=n_bins).astype(np.int64)
        sums = np.bincount(hist_bins, weights=net[hist_idx], minlength=n_bins).astype(np.float64)
        means = np.divide(sums, counts, out=np.full(n_bins, np.nan), where=counts > 0)
        global_mean = float(np.sum(net[hist_idx]) / total)
        means = np.where(np.isfinite(means), means, global_mean)
        fitted = _pava(means, np.maximum(counts, 1))
        idx = np.flatnonzero(day_values == day)
        valid_idx = idx[finite_score[idx]]
        current_bins = np.clip(np.searchsorted(edges[1:-1], score[valid_idx], side="right"), 0, n_bins - 1)
        mapped[valid_idx] = fitted[current_bins].astype(np.float32)
        support[valid_idx] = counts[current_bins].astype(np.int32)
        side_mean[idx] = float(global_mean)
    work["ev14_mapped_bps"] = mapped
    work["ev14_map_support"] = support
    work["ev14_side_mean_bps"] = side_mean
    work["ev14_map_available"] = np.isfinite(mapped) & (support >= MIN_BIN_ROWS)
    # A side-wide gate prevents a few high-scoring candidates from reviving a
    # side whose recent 14-day conversion has fallen below the economic floor.
    work["ev14_side_gate"] = np.isfinite(side_mean) & (side_mean > ADMISSION_BPS)
    work["ev14_admitted"] = work.ev14_map_available & work.ev14_side_gate & (mapped > ADMISSION_BPS)
    return work


def _replay_exit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, g in frame.groupby(frame.candidate_id.str.split("|").str[0], sort=False):
        bars = _source(symbol)
        path_file = PATH_ARTIFACT / f"symbol={symbol}.parquet"
        if not path_file.exists():
            continue
        meta = pd.read_parquet(path_file, columns=["candidate_id", "entry_price", "atr_bps"])
        g = g.merge(meta, on="candidate_id", how="inner", validate="one_to_one")
        if g.empty:
            continue
        starts = bars.index.get_indexer(pd.to_datetime(g.__ts__, utc=True)); valid = starts >= 0
        if not valid.any():
            continue
        g = g.loc[valid].copy(); starts = starts[valid]
        entry = g.entry_price.to_numpy(float); atr_bps = g.atr_bps.to_numpy(float); atr = entry * atr_bps / 10_000.0
        side = np.where(g.side_name.eq("long").to_numpy(), 1.0, -1.0)
        grid = simulate_h12_stop_trailing_grid(
            bars.high.to_numpy(float), bars.low.to_numpy(float), bars.close.to_numpy(float), starts.astype(np.int64),
            entry.astype(np.float32), atr.astype(np.float32), side.astype(np.float32),
            np.asarray([3.0], np.float32), np.asarray([.5], np.float32), np.asarray([.25], np.float32), horizon_bars=48,
        )
        g["exit_net_bps"] = net_bps(grid, atr_bps, cost_bps=100.0).reshape(-1)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(out: Path, *, long_only: bool = False) -> None:
    pred_path = out / "predictions.parquet"
    pred = pd.read_parquet(pred_path).copy()
    base = _base()[["candidate_id", "label_available_ts"]].drop_duplicates("candidate_id")
    pred = pred.merge(base, on="candidate_id", how="left", validate="one_to_one")
    pred["__ts__"] = pd.to_datetime(pred["__ts__"], utc=True)
    pred["label_available_ts"] = pd.to_datetime(pred["label_available_ts"], utc=True)
    score_cols = [c for c in pred.columns if c.startswith("score__")]
    mapped_parts: list[pd.DataFrame] = []
    for col in score_cols:
        for side, side_frame in pred.groupby("side_name", sort=True):
            x = rolling_side_map(side_frame[["candidate_id", "__ts__", "label_available_ts", "side_name", "net_bps", col]].copy(), col)
            x["arm"] = col.removeprefix("score__")
            mapped_parts.append(x[["candidate_id", "__ts__", "side_name", "net_bps", "label_available_ts", "arm", "ev14_mapped_bps", "ev14_map_support", "ev14_map_available", "ev14_side_mean_bps", "ev14_side_gate", "ev14_admitted"]])
    mapped = pd.concat(mapped_parts, ignore_index=True)
    if long_only:
        mapped["ev14_admitted"] &= mapped["side_name"].eq("long")
    mapped.to_parquet(out / "ev14_admission_predictions.parquet", index=False)

    exit_rows: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for arm, g in mapped.groupby("arm", sort=True):
        accepted = g[g.ev14_admitted].copy()
        accepted["month"] = accepted.__ts__.dt.strftime("%Y-%m")
        replay = _replay_exit(accepted)
        if replay.empty:
            continue
        replay["arm"] = arm; replay["month"] = pd.to_datetime(replay.__ts__, utc=True).dt.strftime("%Y-%m")
        exit_rows.append(replay[["candidate_id", "__ts__", "side_name", "month", "arm", "ev14_mapped_bps", "ev14_map_support", "ev14_side_mean_bps", "ev14_side_gate", "ev14_admitted", "net_bps", "exit_net_bps"]])
        for scope, groups in [("global", [("all", replay)]), ("side", list(replay.groupby("side_name"))), ("month", list(replay.groupby("month"))), ("side_month", list(replay.groupby(["side_name", "month"])))]:
            for key, sub in groups:
                rec: dict[str, Any] = {"arm": arm, "scope": scope, "key": str(key), "rows": int(len(sub)), "accept_rate": float(len(sub) / max(1, len(g))), "mean_expected_ev_bps": float(sub.ev14_mapped_bps.mean()), "mean_h12_net_bps": float(sub.net_bps.mean()), "mean_exit_net_bps": float(sub.exit_net_bps.mean()), "mean_exit_gross_bps": float(sub.exit_net_bps.mean() + 100.0)}
                metrics.append(rec)
        for tail in TAILS:
            n = max(1, int(math.ceil(len(replay) * tail)))
            top = replay.sort_values(["ev14_mapped_bps", "candidate_id"], ascending=[False, True], kind="stable").head(n)
            metrics.append({"arm": arm, "scope": "accepted_global_tail", "key": f"top{int(tail*100)}", "rows": int(n), "accept_rate": float(len(accepted) / max(1, len(g))), "mean_expected_ev_bps": float(top.ev14_mapped_bps.mean()), "mean_h12_net_bps": float(top.net_bps.mean()), "mean_exit_net_bps": float(top.exit_net_bps.mean()), "mean_exit_gross_bps": float(top.exit_net_bps.mean() + 100.0)})
    if exit_rows:
        pd.concat(exit_rows, ignore_index=True).to_parquet(out / "ev14_admission_exit_rows.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(out / "ev14_admission_metrics.parquet", index=False)
    summary = pd.DataFrame(metrics)
    summary.to_csv(out / "ev14_admission_metrics.csv", index=False)
    (out / "ev14_admission_manifest.json").write_text(json.dumps({
        "schema": "causal_side_local_ev_admission_14d_v1",
        "lookback_days": 14,
        "admission_threshold_net_bps": ADMISSION_BPS,
        "mapping_bins": MAP_BINS,
        "minimum_global_rows": MIN_GLOBAL_ROWS,
        "minimum_bin_rows": MIN_BIN_ROWS,
        "label_rule": "label_available_ts < decision_ts and label_available_ts >= decision_ts - 14 days",
        "score_bin_rule": "recomputed per UTC day from the preceding 14 days of side/arm scores; payoffs and PAVA fit use only labels available before day start",
        "side_gate_rule": "require rolling side-wide mean net bps over the same preceding 14-day resolved window to exceed the 20 bps admission floor",
        "side_policy": "long_only" if long_only else "both_sides",
        "warmup_rule": "no admission until at least 14 prior calendar days of resolved rows exist",
        "exit_policy": {"stop_loss_atr": 3.0, "trailing_activation_atr": 0.5, "trailing_giveback_atr": 0.25, "path_minutes": 15, "horizon_hours": 12, "cost_bps_once": 100.0},
    }, indent=2) + "\n")
    print(summary[summary.scope.isin(["global", "accepted_global_tail"])].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--long-only", action="store_true")
    args = parser.parse_args()
    run(args.out, long_only=args.long_only)
