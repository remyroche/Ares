#!/usr/bin/env python3
"""Backfill Kraken Futures open interest from the native analytics chart API."""

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


def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _default_since_ms(days: float) -> int:
    ts = pd.Timestamp.utcnow() - pd.Timedelta(days=float(days))
    return int(ts.value // 10**6)


def _positive_count(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return int((values.notna() & (values > 0.0)).sum())


def _merge_sidecar(path: Path, oi: pd.Series) -> tuple[int, int, int, int]:
    if oi.empty:
        return 0, 0, 0, 0
    incoming = oi.rename("open_interest").to_frame()
    incoming.index = pd.to_datetime(incoming.index, utc=True).floor("h")
    incoming = incoming[~incoming.index.duplicated(keep="last")].sort_index()
    before = 0
    before_positive = 0
    if path.exists():
        existing = pd.read_parquet(path)
        if not isinstance(existing.index, pd.DatetimeIndex):
            if "ts" not in existing.columns:
                existing = pd.DataFrame(index=incoming.index)
            else:
                existing["ts"] = pd.to_datetime(existing["ts"], utc=True, errors="coerce")
                existing = existing.set_index("ts")
        existing.index = pd.to_datetime(existing.index, utc=True, errors="coerce").floor("h")
        existing = existing[~existing.index.duplicated(keep="last")].sort_index()
        if "open_interest" in existing.columns:
            before = int(existing["open_interest"].replace([np.inf, -np.inf], np.nan).notna().sum())
            before_positive = _positive_count(existing["open_interest"])
        merged = pd.concat([existing, incoming]).sort_index().groupby(level=0).last()
    else:
        merged = incoming
    merged["open_interest"] = pd.to_numeric(
        merged["open_interest"], errors="coerce"
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, compression="zstd")
    tmp.replace(path)
    after = int(merged["open_interest"].replace([np.inf, -np.inf], np.nan).notna().sum())
    after_positive = _positive_count(merged["open_interest"])
    return before, after, before_positive, after_positive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data_perp/exchanges/krakenfutures/manifests/"
            "kraken_dual_market_verified_universe_latest.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_perp/exchanges/krakenfutures/open_interest_hourly"),
    )
    parser.add_argument("--days", type=float, default=1465.0)
    parser.add_argument("--start", default=None, help="UTC start timestamp, e.g. 2022-05-20")
    parser.add_argument("--end", default=None, help="UTC end timestamp; default now")
    parser.add_argument("--limit-symbols", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument("--rate-limit-ms", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = list(manifest.get("symbols") or [])
    partition_count = max(1, int(args.partition_count))
    partition_id = int(args.partition_id)
    if partition_id < 0 or partition_id >= partition_count:
        raise ValueError(
            f"--partition-id must be in [0, {partition_count - 1}], got {partition_id}"
        )
    items = [item for idx, item in enumerate(items) if idx % partition_count == partition_id]
    if args.limit_symbols > 0:
        items = items[: int(args.limit_symbols)]
    since_ms = (
        int(pd.Timestamp(args.start, tz="UTC").value // 10**6)
        if args.start
        else _default_since_ms(args.days)
    )
    until_ms = (
        int(pd.Timestamp(args.end, tz="UTC").value // 10**6)
        if args.end
        else int(pd.Timestamp.utcnow().value // 10**6)
    )

    exchange = make_perp_exchange()
    exchange.rateLimit = max(0, int(args.rate_limit_ms))
    updated = 0
    empty = 0
    failed: list[str] = []
    for i, item in enumerate(items, start=1):
        perp_symbol = str(item.get("perp_symbol") or "")
        if not perp_symbol:
            continue
        try:
            oi = _fetch_kraken_futures_open_interest_analytics(
                exchange,
                perp_symbol,
                since_ms,
                until_ms,
                timeframe="1h",
            )
            if oi.empty:
                empty += 1
                print(f"{i:03d}/{len(items)} {perp_symbol}: empty", flush=True)
            else:
                path = args.out_dir / f"{_safe_symbol(perp_symbol)}.parquet"
                if args.dry_run:
                    before, after = 0, int(oi.notna().sum())
                    before_positive, after_positive = 0, _positive_count(oi)
                else:
                    before, after, before_positive, after_positive = _merge_sidecar(path, oi)
                updated += 1
                print(
                    f"{i:03d}/{len(items)} {perp_symbol}: "
                    f"oi_rows={len(oi):,} sidecar={before:,}->{after:,} "
                    f"positive={before_positive:,}->{after_positive:,}",
                    flush=True,
                )
        except Exception as exc:
            failed.append(f"{perp_symbol}: {exc.__class__.__name__}: {exc}")
            print(f"{i:03d}/{len(items)} {perp_symbol}: FAIL {exc}", flush=True)
        time.sleep(max(0.0, float(args.sleep)))

    result = {
        "symbols": len(items),
        "updated": updated,
        "empty": empty,
        "failed": failed,
        "dry_run": bool(args.dry_run),
        "since": pd.to_datetime(since_ms, unit="ms", utc=True).isoformat(),
        "until": pd.to_datetime(until_ms, unit="ms", utc=True).isoformat(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
