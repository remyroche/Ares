#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import (
    _fetch_kraken_futures_open_interest_analytics,
    make_perp_exchange,
)
from extreme_price_movements.utils import tprint


def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _load_symbols(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("symbols") if isinstance(payload, dict) else payload
    out: list[str] = []
    for row in rows or []:
        sym = row.get("perp_symbol") if isinstance(row, dict) else row
        if sym:
            out.append(str(sym))
    return list(dict.fromkeys(out))


def _load_oi(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=np.float32)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" not in df.columns:
            return pd.Series(dtype=np.float32)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce").floor("h")
    if "open_interest" not in df.columns:
        return pd.Series(dtype=np.float32)
    s = pd.to_numeric(df["open_interest"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.where(s > 0.0).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index().astype(np.float32)


def _gap_ranges(
    series: pd.Series,
    *,
    end_ts: pd.Timestamp,
    max_gap_hours: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if series.empty:
        return []
    idx = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True)).floor("h")
    idx = idx[~idx.isna()].drop_duplicates().sort_values()
    if idx.empty:
        return []
    full = pd.date_range(idx.min(), end_ts, freq="1h", tz="UTC")
    missing = full.difference(idx)
    if missing.empty:
        return []
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        contiguous = ts == prev + pd.Timedelta(hours=1)
        within_cap = (ts - start) <= pd.Timedelta(hours=max_gap_hours - 1)
        if contiguous and within_cap:
            prev = ts
            continue
        ranges.append((start, prev + pd.Timedelta(hours=1)))
        start = ts
        prev = ts
    ranges.append((start, prev + pd.Timedelta(hours=1)))
    return ranges


def _write_oi(path: Path, existing: pd.Series, incoming: pd.Series) -> tuple[int, int]:
    before = int(existing.notna().sum())
    merged = pd.concat([existing, incoming]).sort_index().groupby(level=0).last()
    merged = merged.replace([np.inf, -np.inf], np.nan).where(lambda s: s > 0.0).dropna()
    out = merged.rename("open_interest").to_frame().astype({"open_interest": "float32"})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return before, int(out["open_interest"].notna().sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_verified_universe_latest.json")
    parser.add_argument("--out-dir", default="data_perp/exchanges/krakenfutures/open_interest_hourly")
    parser.add_argument("--seed-dir", default="data_perp/exchanges/krakenfutures/funding_hourly")
    parser.add_argument("--end-ts", default="")
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument("--max-gap-hours", type=int, default=720)
    parser.add_argument("--rate-limit-ms", type=int, default=150)
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
    out_dir = Path(args.out_dir)
    seed_dir = Path(args.seed_dir)
    exchange = None
    stats = {"symbols": len(symbols), "updated": 0, "no_gaps": 0, "skipped_no_seed": 0, "fetched_rows": 0, "failed": []}
    for i, symbol in enumerate(symbols, start=1):
        try:
            filename = f"{_safe_symbol(symbol)}.parquet"
            out_path = out_dir / filename
            existing = _load_oi(out_path)
            seed = _load_oi(seed_dir / filename)
            combined = pd.concat([seed, existing]).sort_index().groupby(level=0).last()
            combined = combined.replace([np.inf, -np.inf], np.nan).where(lambda s: s > 0.0).dropna()
            if combined.empty:
                stats["skipped_no_seed"] += 1
                tprint(f"[{i:04d}/{len(symbols):04d}] {symbol}: skip no local OI seed")
                continue
            ranges = _gap_ranges(combined, end_ts=end_ts, max_gap_hours=int(args.max_gap_hours))
            if not ranges:
                stats["no_gaps"] += 1
                if not args.dry_run and not out_path.exists():
                    _write_oi(out_path, combined.iloc[:0], combined)
                continue
            tprint(f"[{i:04d}/{len(symbols):04d}] {symbol}: oi_gap_ranges={len(ranges)}")
            if args.dry_run:
                continue
            if exchange is None:
                exchange = make_perp_exchange()
                exchange.rateLimit = max(
                    int(getattr(exchange, "rateLimit", 0) or 0),
                    int(args.rate_limit_ms),
                )
            fetched_parts = []
            for start, end in ranges:
                oi = _fetch_kraken_futures_open_interest_analytics(
                    exchange,
                    symbol,
                    int(start.value // 10**6),
                    int(end.value // 10**6),
                    timeframe="1h",
                )
                if not oi.empty:
                    fetched_parts.append(oi)
                time.sleep(max(0.0, float(args.sleep)))
            if not fetched_parts:
                if not args.dry_run and not out_path.exists():
                    _write_oi(out_path, combined.iloc[:0], combined)
                continue
            incoming = pd.concat(fetched_parts).sort_index().groupby(level=0).last()
            incoming = incoming.replace([np.inf, -np.inf], np.nan).where(lambda s: s > 0.0).dropna()
            incoming = incoming.loc[~incoming.index.floor("h").isin(pd.DatetimeIndex(combined.index).floor("h"))]
            if incoming.empty:
                if not args.dry_run and not out_path.exists():
                    _write_oi(out_path, combined.iloc[:0], combined)
                continue
            stats["fetched_rows"] += int(len(incoming))
            stats["updated"] += 1
            if not args.dry_run:
                before, after = _write_oi(out_path, combined, incoming)
                tprint(f"  {symbol}: oi_rows={before}->{after} added_missing={len(incoming)}")
        except Exception as exc:
            stats["failed"].append(f"{symbol}: {exc.__class__.__name__}: {exc}")
            tprint(f"[{i:04d}/{len(symbols):04d}] {symbol}: FAIL {exc}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if not stats["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
