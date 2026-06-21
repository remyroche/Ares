#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    _fetch_kraken_futures_chart_ohlcv,
    make_perp_exchange,
)
from extreme_price_movements.utils import tprint


def _load_symbols(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("symbols") if isinstance(payload, dict) else payload
    out: list[str] = []
    for row in rows or []:
        sym = row.get("perp_symbol") if isinstance(row, dict) else row
        if sym:
            out.append(str(sym))
    return list(dict.fromkeys(out))


def _gap_ranges(
    index: pd.DatetimeIndex,
    *,
    end_ts: pd.Timestamp,
    max_gap_hours: int,
    start_ts: pd.Timestamp | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if index.empty:
        return []
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True)).floor("h")
    idx = idx[~idx.isna()].drop_duplicates().sort_values()
    if idx.empty:
        return []
    if start_ts is None:
        range_start_bound = idx.min()
    else:
        range_start_bound = pd.Timestamp(start_ts)
        if range_start_bound.tzinfo is None:
            range_start_bound = range_start_bound.tz_localize("UTC")
        else:
            range_start_bound = range_start_bound.tz_convert("UTC")
        range_start_bound = range_start_bound.floor("h")
    range_start_bound = max(range_start_bound, idx.min())
    full = pd.date_range(range_start_bound, end_ts, freq="1h", tz="UTC")
    idx = idx[(idx >= range_start_bound) & (idx <= end_ts)]
    missing = full.difference(idx)
    if missing.empty:
        return []
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    range_start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        contiguous = ts == prev + pd.Timedelta(hours=1)
        within_cap = (ts - range_start) <= pd.Timedelta(hours=max_gap_hours - 1)
        if contiguous and within_cap:
            prev = ts
            continue
        ranges.append((range_start, prev + pd.Timedelta(hours=1)))
        range_start = ts
        prev = ts
    ranges.append((range_start, prev + pd.Timedelta(hours=1)))
    return ranges


def _nonempty_ohlcv_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if len(cols) < 5:
        return pd.DataFrame()
    out = df.loc[:, cols].copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    no_trade_price = out["open"].eq(out["close"])
    out.loc[no_trade_price & out["volume"].isna(), "volume"] = 0.0
    zero_no_trade = out["volume"].eq(0) & no_trade_price
    prev_linked = zero_no_trade & zero_no_trade.shift(1, fill_value=False) & out["close"].shift(1).eq(out["open"])
    next_linked = zero_no_trade & zero_no_trade.shift(-1, fill_value=False) & out["close"].eq(out["open"].shift(-1))
    suspicious_zero_run = zero_no_trade & (prev_linked | next_linked)
    volume_ok = out["volume"].gt(0) | (zero_no_trade & ~suspicious_zero_run)
    valid = (
        out["open"].gt(0)
        & out["high"].gt(0)
        & out["low"].gt(0)
        & out["close"].gt(0)
        & volume_ok
    )
    return out.loc[valid].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_verified_universe_latest.json")
    parser.add_argument("--perp-root", default="data_perp/exchanges/krakenfutures")
    parser.add_argument("--end-ts", default="")
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument("--max-gap-hours", type=int, default=720)
    parser.add_argument(
        "--lookback-days",
        type=float,
        default=0.0,
        help=(
            "Only inspect gaps in the trailing N days before --end-ts. "
            "Use 0 to scan from each symbol's first local row."
        ),
    )
    parser.add_argument("--rate-limit-ms", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = _load_symbols(Path(args.manifest))
    partition_count = max(1, int(args.partition_count))
    partition_id = int(args.partition_id)
    if partition_id < 0 or partition_id >= partition_count:
        raise ValueError(f"invalid partition {partition_id}/{partition_count}")
    symbols = [s for i, s in enumerate(symbols) if i % partition_count == partition_id]
    end_ts = (
        pd.Timestamp(args.end_ts, tz="UTC").floor("h")
        if args.end_ts
        else pd.Timestamp.utcnow().floor("h")
    )
    start_ts = (
        end_ts - pd.Timedelta(days=float(args.lookback_days))
        if float(args.lookback_days or 0.0) > 0.0
        else None
    )

    store = PartitionedOHLCVStore(args.perp_root, "1h")
    exchange = None
    stats = {"symbols": len(symbols), "updated": 0, "skipped_no_local": 0, "no_gaps": 0, "fetched_rows": 0, "failed": []}
    for i, symbol in enumerate(symbols, start=1):
        try:
            existing = store.load(symbol)
            if existing.empty:
                stats["skipped_no_local"] += 1
                tprint(f"[{i:04d}/{len(symbols):04d}] {symbol}: skip no local OHLCV seed")
                continue
            ranges = _gap_ranges(
                existing.index,
                end_ts=end_ts,
                max_gap_hours=int(args.max_gap_hours),
                start_ts=start_ts,
            )
            if not ranges:
                stats["no_gaps"] += 1
                continue
            tprint(f"[{i:04d}/{len(symbols):04d}] {symbol}: gap_ranges={len(ranges)}")
            if args.dry_run:
                continue
            if exchange is None:
                exchange = make_perp_exchange()
                exchange.rateLimit = max(
                    int(getattr(exchange, "rateLimit", 0) or 0),
                    int(args.rate_limit_ms),
                )
            frames = []
            for start, end in ranges:
                since_ms = int(start.value // 10**6)
                until_ms = int(end.value // 10**6)
                fetched = _fetch_kraken_futures_chart_ohlcv(
                    exchange,
                    symbol,
                    "trade",
                    since_ms,
                    until_ms,
                    timeframe="1h",
                )
                fetched = _nonempty_ohlcv_rows(fetched)
                if not fetched.empty:
                    frames.append(fetched)
                time.sleep(max(0.0, float(args.sleep)))
            if not frames:
                continue
            incoming = pd.concat(frames).sort_index()
            incoming = incoming[~incoming.index.duplicated(keep="last")]
            existing_hours = pd.DatetimeIndex(existing.index).floor("h")
            incoming = incoming.loc[~incoming.index.floor("h").isin(existing_hours)]
            if incoming.empty:
                continue
            stats["fetched_rows"] += int(len(incoming))
            stats["updated"] += 1
            if not args.dry_run:
                store.save_partitioned(symbol, incoming, defer_compact=True)
                for year in sorted(set(int(y) for y in incoming.index.year)):
                    store.compact_partition(symbol, year)
                store._write_meta(symbol, {"last_gap_backfill_ts": end_ts.isoformat()})
            tprint(f"  {symbol}: wrote_missing_rows={len(incoming)}")
        except Exception as exc:
            stats["failed"].append(f"{symbol}: {exc.__class__.__name__}: {exc}")
            tprint(f"[{i:04d}/{len(symbols):04d}] {symbol}: FAIL {exc}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if not stats["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
