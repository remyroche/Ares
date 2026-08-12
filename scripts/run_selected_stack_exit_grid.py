#!/usr/bin/env python3
"""Vectorized coarse (15-minute) SL/trailing-profit grid for the selected stack."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.trailing_exit_grid import net_bps, simulate_h12_stop_trailing_grid

PRED = ROOT / "data_perp/artifacts/frozen_residual_query_hpo_20260810_v1/predictions.parquet"
PATH_ROOT = ROOT / "15m_ohlcv_perp"
PATH_ARTIFACT = ROOT / "data_perp/artifacts/h12_query_path_grid_20260805_v2"
OUT = ROOT / "data_perp/artifacts/frozen_selected_stack_exit_grid_20260810_v1"
STOPS = np.asarray([1., 1.5, 2., 2.5, 3.], dtype=np.float32)
ACTIVATIONS = np.asarray([.5, 1., 1.5, 2., 3.], dtype=np.float32)
GIVEBACKS = np.asarray([.25, .5, .75, 1.], dtype=np.float32)


def _source(symbol: str) -> pd.DataFrame:
    path = PATH_ROOT / (symbol.lower().replace("_", "") + "_15m.parquet")
    raw = pd.read_parquet(path)
    col = next((c for c in ("ts", "timestamp", "__index_level_0__") if c in raw.columns), None)
    if col is not None:
        raw = raw.set_index(col)
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw.loc[:, ["open", "high", "low", "close"]]
    return raw[~raw.index.duplicated(keep="last")].sort_index()


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(PRED)
    pred["month"] = pd.to_datetime(pred["__ts__"], utc=True).dt.to_period("M").astype(str)
    # The exit grid is evaluated on the same global score ordering as the
    # selected stack.  Retain only top-10% to keep the per-symbol path tensor
    # bounded; top-1/5/10 metrics are then read from this common population.
    n = max(1, int(np.ceil(len(pred) * .10)))
    chosen = pred.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(n).copy()
    chosen["gross_atr_grid"] = None
    pieces = []
    for symbol, g in chosen.groupby(chosen.candidate_id.str.split("|").str[0], sort=False):
        try:
            bars = _source(symbol)
        except FileNotFoundError:
            continue
        g = g.copy()
        path_file = PATH_ARTIFACT / f"symbol={symbol}.parquet"
        if not path_file.exists():
            continue
        path_meta = pd.read_parquet(path_file, columns=["candidate_id", "entry_price", "atr_bps"])
        g = g.merge(path_meta, on="candidate_id", how="inner", validate="one_to_one")
        if g.empty:
            continue
        ts = pd.to_datetime(g["__ts__"], utc=True)
        starts = bars.index.get_indexer(ts)
        valid = starts >= 0
        if not valid.any():
            continue
        g = g.loc[valid].copy(); starts = starts[valid]
        e = g.entry_price.to_numpy(float)
        atr_bps = g.atr_bps.to_numpy(float)
        atr = e * atr_bps / 10_000.
        side = np.where(g.side_name.eq("long").to_numpy(), 1., -1.)
        grid = simulate_h12_stop_trailing_grid(
            bars.high.to_numpy(float), bars.low.to_numpy(float), bars.close.to_numpy(float),
            starts.astype(np.int64), e.astype(np.float32), atr.astype(np.float32), side.astype(np.float32),
            STOPS, ACTIVATIONS, GIVEBACKS, horizon_bars=48,
        )
        for si, stop in enumerate(STOPS):
            for ai, activation in enumerate(ACTIVATIONS):
                for gi, giveback in enumerate(GIVEBACKS):
                    z = g[["candidate_id", "score", "net_bps", "gross_bps", "month"]].copy()
                    z["stop_atr"], z["activation_atr"], z["giveback_atr"] = float(stop), float(activation), float(giveback)
                    z["exit_net_bps"] = net_bps(grid[:, si:si+1, ai:ai+1, gi:gi+1], atr_bps, cost_bps=100.).reshape(-1)
                    pieces.append(z)
    if not pieces:
        raise RuntimeError("no 15-minute paths matched selected predictions")
    rows = pd.concat(pieces, ignore_index=True)
    metrics = []
    for (stop, activation, giveback), g in rows.groupby(["stop_atr", "activation_atr", "giveback_atr"], sort=False):
        g = g.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
        rec = {"stop_atr": stop, "activation_atr": activation, "giveback_atr": giveback}
        for tail in (.01, .05, .10):
            m = max(1, int(np.ceil(len(g) * tail)))
            rec[f"top{int(tail*100)}_net_bps"] = float(g.head(m).exit_net_bps.mean())
            rec[f"top{int(tail*100)}_gross_bps"] = float(g.head(m).exit_net_bps.mean() + 100.)
        metrics.append(rec)
    result = pd.DataFrame(metrics).sort_values(["top5_net_bps", "top1_net_bps"], ascending=False)
    result.to_parquet(out / "exit_grid_metrics.parquet", index=False)
    result.head(20).to_csv(out / "exit_grid_top20.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"schema": "frozen_selected_stack_exit_grid_v1", "prediction_source": str(PRED), "path_source": str(PATH_ROOT), "resolution_minutes": 15, "horizon_bars": 48, "cost_bps": 100, "stops_atr": STOPS.tolist(), "activations_atr": ACTIVATIONS.tolist(), "givebacks_atr": GIVEBACKS.tolist(), "selection": "top5 net, then top1 net", "note": "coarse 15-minute exit proxy; not minute execution"}, indent=2) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); args = ap.parse_args(); print(run(args.out))
