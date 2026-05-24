#!/usr/bin/env python3
"""Rebuild Kraken Futures orderbook-proxy sidecars from local native OHLCV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    build_hourly_orderbook_proxy_from_ohlcv,
    normalize_orderbook_proxy_frame,
)
from extreme_price_movements.utils import tprint


def _symbol_to_filename(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perp-root", default="data_perp/exchanges/krakenfutures")
    parser.add_argument("--manifest", default="data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_universe_latest.json")
    parser.add_argument("--start-ts", default="")
    parser.add_argument("--end-ts", default="")
    args = parser.parse_args()

    root = Path(args.perp_root)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    symbols = [
        str(item.get("perp_symbol", "")).strip()
        for item in manifest.get("symbols", [])
        if item.get("perp_symbol")
    ]
    start_ts = pd.Timestamp(args.start_ts, tz="UTC") if args.start_ts else None
    end_ts = pd.Timestamp(args.end_ts, tz="UTC") if args.end_ts else None

    store = PartitionedOHLCVStore(str(root), "1h")
    out_dir = root / "orderbook_hourly"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    for i, symbol in enumerate(symbols, start=1):
        df = store.load(symbol, start_ts=start_ts, end_ts=end_ts)
        if df.empty:
            skipped += 1
            tprint(f"[{i:04d}/{len(symbols):04d}] skip empty {symbol}")
            continue
        proxy = build_hourly_orderbook_proxy_from_ohlcv(df)
        proxy = normalize_orderbook_proxy_frame(proxy)
        if proxy.empty:
            skipped += 1
            tprint(f"[{i:04d}/{len(symbols):04d}] skip no proxy {symbol}")
            continue
        out_path = out_dir / f"{_symbol_to_filename(symbol)}.parquet"
        proxy.to_parquet(out_path, compression="zstd")
        ok += 1
        if ok % 25 == 0 or i == len(symbols):
            tprint(f"Rebuilt Kraken orderbook proxies: ok={ok} skipped={skipped}")

    tprint(f"Kraken orderbook proxy rebuild complete: ok={ok} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
