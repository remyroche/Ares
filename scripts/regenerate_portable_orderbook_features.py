#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG, enable_perp_feature_keys
from extreme_price_movements.data_store import PartitionedOHLCVStore, save_features, to_panel
from extreme_price_movements.features import add_regime_gates, compute_market_features
from extreme_price_movements.pipeline_steps import (
    _compute_features_hourly_runtime,
    _load_saved_microdata_for_symbols,
)
from extreme_price_movements.utils import tprint


def _snapshot_symbols(feature_dir: Path) -> list[str]:
    symbols: list[str] = []
    for path in sorted(feature_dir.glob("symbol=*.parquet")):
        name = path.stem
        if not name.startswith("symbol="):
            continue
        symbols.append(name.split("=", 1)[1].replace("_", "/"))
    return symbols


def _load_symbol_frames(
    store: PartitionedOHLCVStore,
    symbols: list[str],
    *,
    end_ts: pd.Timestamp,
    min_rows: int,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = store.load(symbol, end_ts=end_ts)
        if df.empty or len(df) < min_rows:
            continue
        out[symbol] = df
    return out


def _portable_orderbook_keys(cfg: dict) -> list[str]:
    return sorted(set(str(k) for k in cfg.get("ORDERBOOK_FEATURE_KEYS", []) if str(k)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally regenerate portable orderbook features only."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ts", required=True)
    parser.add_argument("--market-mode", choices=["spot", "perps"], required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    cfg = CFG.copy()
    if args.market_mode == "perps":
        cfg = enable_perp_feature_keys(cfg)
    cfg["market_mode"] = args.market_mode
    cfg["use_perps"] = args.market_mode == "perps"
    cfg["data_root"] = os.path.abspath(args.data_root)
    cfg["feature_save_workers"] = max(1, int(cfg.get("feature_save_workers", 2)))
    cfg["enable_orderbook_features"] = True
    cfg["enable_orderbook_wall_features"] = False

    ts_sig = pd.Timestamp(args.ts.replace("_", " "), tz="UTC")
    feature_end_lag_days = int(cfg.get("feature_generation_end_lag_days", 3))
    end_ts = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(feature_end_lag_days, 0))
    ).floor("h")
    if end_ts < ts_sig:
        end_ts = ts_sig

    feature_dir = Path(cfg["data_root"]) / "features" / args.run_id
    symbols = _snapshot_symbols(feature_dir)
    keys = _portable_orderbook_keys(cfg)
    if not symbols:
        raise RuntimeError(f"No existing symbol parquet files under {feature_dir}")
    if not keys:
        raise RuntimeError("ORDERBOOK_FEATURE_KEYS is empty")

    tprint(
        f"Portable orderbook regeneration: mode={args.market_mode} "
        f"symbols={len(symbols)} keys={len(keys)} run_id={args.run_id} "
        f"end_ts={end_ts}"
    )

    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    total_saved_symbols = 0
    total_saved_keys = 0
    chunk_size = max(1, int(args.chunk_size))

    for chunk_idx, start in enumerate(range(0, len(symbols), chunk_size), start=1):
        chunk_symbols = symbols[start : start + chunk_size]
        t0 = time.perf_counter()
        dfs = _load_symbol_frames(
            store,
            chunk_symbols,
            end_ts=end_ts,
            min_rows=24 * 15,
        )
        if not dfs:
            tprint(f"[chunk {chunk_idx}] no loadable symbols; skipping")
            continue

        panel = to_panel(dfs)
        microdata_panel, orderbook_by_symbol = _load_saved_microdata_for_symbols(
            cfg["data_root"],
            list(dfs),
            panel["close"].index,
        )
        for key, frame in microdata_panel.items():
            existing = panel.get(key)
            if (
                isinstance(existing, pd.DataFrame)
                and not existing.empty
                and np.isfinite(existing.to_numpy(dtype=np.float32, copy=False)).any()
            ):
                continue
            panel[key] = frame

        mkt_df = compute_market_features(panel, cfg["market_basket"])
        mkt_gates = add_regime_gates(
            mkt_df, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"]
        )
        feats, feat_index, feat_columns = _compute_features_hourly_runtime(
            panel,
            mkt_gates,
            cfg,
            orderbook_by_symbol,
            requested_feature_keys=keys,
        )
        selected = {key: value for key, value in feats.items() if key in set(keys)}
        if not selected:
            tprint(f"[chunk {chunk_idx}] computed no portable orderbook keys")
            continue

        save_features(
            selected,
            ts_sig,
            cfg["data_root"],
            feat_index=feat_index,
            feat_columns=feat_columns,
            save_workers=int(cfg.get("feature_save_workers", 2)),
            replace_existing=False,
        )
        total_saved_symbols += len(feat_columns)
        total_saved_keys += len(selected)
        tprint(
            f"[chunk {chunk_idx}] saved symbols={len(feat_columns)} "
            f"keys={len(selected)} elapsed={time.perf_counter() - t0:.1f}s"
        )

    tprint(
        f"Portable orderbook regeneration complete: "
        f"saved_symbol_chunks_total={total_saved_symbols} "
        f"saved_key_chunks_total={total_saved_keys}"
    )


if __name__ == "__main__":
    main()
