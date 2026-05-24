#!/usr/bin/env python3
"""Incrementally download Kraken Futures 15m OHLCV into the HF cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from extreme_price_movements.data_store import (
    _fetch_ohlcv_paged,
    _load_local_env_if_present,
    make_perp_exchange,
)
from extreme_price_movements.utils import tprint


def _load_symbols(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("symbols") if isinstance(payload, dict) else payload
    symbols: list[str] = []
    for row in rows or []:
        if isinstance(row, dict):
            sym = row.get("perp_symbol") or row.get("symbol")
        else:
            sym = row
        if sym:
            symbols.append(str(sym))
    return list(dict.fromkeys(symbols))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_verified_universe_latest.json",
    )
    parser.add_argument("--lookback-days", type=float, default=1460.0)
    parser.add_argument("--hf-data-dir", default="15m_ohlcv_perp")
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument(
        "--order",
        choices=("alpha_asc", "alpha_desc"),
        default="alpha_asc",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--rate-limit-ms", type=int, default=1500)
    args = parser.parse_args()

    partition_count = max(1, int(args.partition_count))
    partition_id = int(args.partition_id)
    if partition_id < 0 or partition_id >= partition_count:
        raise ValueError(
            f"--partition-id must be in [0, {partition_count - 1}], got {partition_id}"
        )

    hf_dir = Path(args.hf_data_dir)
    if not hf_dir.is_absolute():
        hf_dir = Path.cwd() / hf_dir
    hf_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EPM_HF_DATA_DIR"] = str(hf_dir)
    os.environ["EPM_EXCHANGE"] = "kraken"

    from extreme_price_movements.hf_data_loader import _load_existing_data, _save_data

    _load_local_env_if_present()
    ex = make_perp_exchange()
    ex.rateLimit = max(int(getattr(ex, "rateLimit", 0) or 0), int(args.rate_limit_ms))

    all_symbols = sorted(_load_symbols(Path(args.manifest)))
    if args.order == "alpha_desc":
        all_symbols = list(reversed(all_symbols))
    symbols = [sym for idx, sym in enumerate(all_symbols) if idx % partition_count == partition_id]

    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=float(args.lookback_days))).floor("15min")
    until = pd.Timestamp.now(tz="UTC").floor("15min")
    tprint(
        "Kraken 15m HF download start: "
        f"symbols={len(symbols)}/{len(all_symbols)} partition={partition_id}/{partition_count} "
        f"since={since} until={until} hf_dir={hf_dir}"
    )

    def _missing_ranges(symbol: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        existing = _load_existing_data(symbol, allow_quote_fallback=False)
        if existing is None or existing.empty:
            return [(since, until)]
        ex_start = existing.index.min()
        ex_end = existing.index.max()
        if ex_start <= since and ex_end >= until:
            return []
        ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if since < ex_start:
            left_end = min(until, ex_start - pd.Timedelta(minutes=15))
            if since <= left_end:
                ranges.append((since, left_end))
        if until > ex_end:
            right_start = max(since, ex_end + pd.Timedelta(minutes=15))
            if right_start <= until:
                ranges.append((right_start, until))
        return ranges

    def _download_charts_range(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        frame_ms = int(pd.Timedelta(minutes=15).total_seconds() * 1000)
        chunk_ms = 2000 * frame_ms
        cursor_ms = int(start.value // 10**6)
        end_ms = int((end + pd.Timedelta(minutes=15)).value // 10**6)
        while cursor_ms < end_ms:
            chunk_end_ms = min(cursor_ms + chunk_ms, end_ms)
            part = _fetch_ohlcv_paged(
                ex,
                symbol,
                cursor_ms,
                chunk_end_ms,
                timeframe="15m",
                limit=2000,
            )
            if part is not None and not part.empty:
                frames.append(part)
            cursor_ms = chunk_end_ms
            time.sleep(max(float(args.sleep_seconds), ex.rateLimit / 1000.0))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out[(out.index >= start) & (out.index <= end)]
        return out

    stats = {"ok": 0, "skipped": 0, "empty": 0, "failed": 0, "symbols": len(symbols)}
    for i, symbol in enumerate(symbols, start=1):
        try:
            ranges = _missing_ranges(symbol)
            if not ranges:
                existing = _load_existing_data(symbol, allow_quote_fallback=False)
                stats["skipped"] += 1
                tprint(
                    f"[{i:04d}/{len(symbols):04d}] {symbol} skipped "
                    f"rows={len(existing)} span={existing.index.min()}->{existing.index.max()}"
                )
                continue
            chunks: list[pd.DataFrame] = []
            for range_start, range_end in ranges:
                tprint(f"Downloading 15m charts data for {symbol}: {range_start} to {range_end}")
                part = _download_charts_range(symbol, range_start, range_end)
                if part is not None and not part.empty:
                    chunks.append(part)
            existing = _load_existing_data(symbol, allow_quote_fallback=False)
            combined_parts = ([] if existing is None or existing.empty else [existing]) + chunks
            if combined_parts:
                df = pd.concat(combined_parts).sort_index()
                df = df[~df.index.duplicated(keep="last")]
                _save_data(symbol, df)
                stats["ok"] += 1
                tprint(
                    f"[{i:04d}/{len(symbols):04d}] {symbol} ok "
                    f"rows={len(df)} span={df.index.min()}->{df.index.max()}"
                )
            else:
                stats["empty"] += 1
                tprint(f"[{i:04d}/{len(symbols):04d}] {symbol} empty")
        except Exception as exc:
            stats["failed"] += 1
            tprint(f"[{i:04d}/{len(symbols):04d}] {symbol} failed: {exc}")
        time.sleep(max(0.0, float(args.sleep_seconds)))

    tprint(f"Kraken 15m HF download complete: {stats}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if stats["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
