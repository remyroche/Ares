#!/usr/bin/env python3
"""Backfill exact 1m execution paths for simple-policy candidate rows.

Workers may share one output root only when ``partition-id`` assigns every
symbol to exactly one worker.  Windows are half-open and derived from candidate
timestamps plus the requested replay horizon.  Existing candles are preserved;
only missing minute buckets are fetched from Kraken Futures charts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    _fetch_ohlcv_paged,
    append_missing_kraken_execution_1m,
    canonical_kraken_execution_1m_root,
    make_perp_exchange,
)


def _owned(symbol: str, partition_count: int, partition_id: int) -> bool:
    digest = hashlib.blake2b(symbol.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % partition_count == partition_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _merge_windows(
    starts: Iterable[pd.Timestamp], horizon_minutes: int, warmup_minutes: int = 0
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    horizon = pd.Timedelta(minutes=int(horizon_minutes))
    warmup = pd.Timedelta(minutes=max(int(warmup_minutes), 0))
    def _utc_minute(ts: pd.Timestamp) -> pd.Timestamp:
        value = pd.Timestamp(ts)
        value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return value.floor("min")

    raw = sorted((_utc_minute(ts) - warmup, _utc_minute(ts) + horizon) for ts in starts)
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in raw:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _minute_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return pd.DatetimeIndex([], tz="UTC")
    idx = frame.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.floor("min").unique().sort_values()


def _missing_buckets(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    existing: pd.DatetimeIndex,
    chunk_minutes: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunk_minutes = max(1, min(int(chunk_minutes), 1_440))
    chunk = pd.Timedelta(minutes=chunk_minutes)
    bucket_ns = int(chunk.value)
    existing_ns = set(existing.asi8.tolist())
    buckets: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for start, end in windows:
        expected = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
        missing_ns = [int(v) for v in expected.asi8 if int(v) not in existing_ns]
        for value in missing_ns:
            bucket_start_ns = (value // bucket_ns) * bucket_ns
            bucket_start = pd.Timestamp(bucket_start_ns, tz="UTC")
            bucket_end = bucket_start + chunk
            clipped = (max(start, bucket_start), min(end, bucket_end))
            if bucket_start_ns in buckets:
                prior = buckets[bucket_start_ns]
                buckets[bucket_start_ns] = (
                    min(prior[0], clipped[0]),
                    max(prior[1], clipped[1]),
                )
            else:
                buckets[bucket_start_ns] = clipped
    return [buckets[key] for key in sorted(buckets)]


def _load_existing(
    store: PartitionedOHLCVStore,
    symbol: str,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    if not windows:
        return pd.DataFrame()
    out = store.load(
        symbol,
        columns=["ts", "open", "high", "low", "close", "volume"],
        start_ts=min(start for start, _ in windows),
        end_ts=max(end for _, end in windows),
    )
    return out if out is not None else pd.DataFrame()


def _coverage(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]], frame: pd.DataFrame
) -> tuple[int, int, float]:
    existing = set(_minute_index(frame).asi8.tolist())
    required = 0
    covered = 0
    for start, end in windows:
        expected = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
        required += len(expected)
        covered += sum(int(v) in existing for v in expected.asi8)
    return covered, required, float(covered / required) if required else 1.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--store-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--horizon-minutes", type=int, default=1_440)
    parser.add_argument("--warmup-minutes", type=int, default=0)
    parser.add_argument("--chunk-minutes", type=int, default=1_440)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--no-compact", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--stage-manifest",
        default="",
        help="Optional frozen request-stage manifest to hash-bind into the download audit.",
    )
    args = parser.parse_args()

    partition_count = max(1, int(args.partition_count))
    partition_id = int(args.partition_id)
    if not 0 <= partition_id < partition_count:
        raise ValueError("partition-id must be in [0, partition-count)")
    if int(args.horizon_minutes) <= 0:
        raise ValueError("horizon-minutes must be positive")

    os.environ.setdefault("EPM_EXCHANGE", "krakenfutures")
    candidate_path = Path(args.candidates)
    parquet_columns = set(pq.read_schema(candidate_path).names)
    timestamp_column = "timestamp" if "timestamp" in parquet_columns else "__decision_ts__"
    symbol_column = "symbol" if "symbol" in parquet_columns else "__symbol__"
    if timestamp_column not in parquet_columns or symbol_column not in parquet_columns:
        raise ValueError(
            "candidate input requires timestamp/symbol or canonical __decision_ts__/__symbol__"
        )
    candidate_columns = [timestamp_column, symbol_column]
    if "product_id" in parquet_columns:
        candidate_columns.append("product_id")
    candidates = pd.read_parquet(candidate_path, columns=candidate_columns).rename(
        columns={timestamp_column: "timestamp", symbol_column: "symbol"}
    )
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates = candidates.dropna(subset=["timestamp", "symbol"])
    product_ids: dict[str, str] = {}
    if "product_id" in candidates.columns:
        for symbol, group in candidates.groupby("symbol", sort=True):
            values = sorted(
                {
                    str(value).strip()
                    for value in group["product_id"].dropna()
                    if str(value).strip()
                }
            )
            if len(values) != 1:
                raise ValueError(
                    f"{symbol} must map to exactly one frozen product_id, got {values}"
                )
            product_ids[str(symbol)] = values[0]
    grouped = {
        str(symbol): _merge_windows(
            group["timestamp"], int(args.horizon_minutes), int(args.warmup_minutes)
        )
        for symbol, group in candidates.groupby("symbol", sort=True)
        if _owned(str(symbol), partition_count, partition_id)
    }

    canonical_root = canonical_kraken_execution_1m_root(args.data_root)
    if args.store_root is not None and Path(args.store_root).resolve() != canonical_root.resolve():
        raise ValueError(f"execution_1m store-root must be canonical: {canonical_root}")
    store = PartitionedOHLCVStore(str(canonical_root), timeframe="1m")
    exchange = None if args.verify_only else make_perp_exchange()
    results: list[dict[str, Any]] = []
    total = len(grouped)
    for number, (symbol, windows) in enumerate(grouped.items(), start=1):
        started = time.monotonic()
        status = "ok"
        error = None
        fetched_rows = 0
        fetched_requests = 0
        product_id = product_ids.get(symbol)
        try:
            existing = _load_existing(store, symbol, windows)
            before_covered, required, before_fraction = _coverage(windows, existing)
            existing_index = _minute_index(existing)
            existing_minutes = set(existing_index)
            buckets = _missing_buckets(
                windows,
                existing_index,
                int(args.chunk_minutes),
            )
            if not args.verify_only:
                pending_frames: list[pd.DataFrame] = []
                for start, end in buckets:
                    fresh = _fetch_ohlcv_paged(
                        exchange,
                        symbol,
                        int(start.value // 10**6),
                        int(end.value // 10**6),
                        timeframe="1m",
                        limit=2_000,
                        params={"product_id": product_id} if product_id else None,
                    )
                    fetched_requests += 1
                    if fresh is not None and not fresh.empty:
                        fresh = fresh.loc[(fresh.index >= start) & (fresh.index < end)]
                        # The canonical store is immutable. A fetch bucket can
                        # overlap already persisted live rows even though it was
                        # requested to repair a later gap; only append timestamps
                        # that were absent when this repair started.
                        fresh = fresh.loc[~fresh.index.isin(existing_minutes)]
                        if not fresh.empty:
                            pending_frames.append(fresh)
                            existing_minutes.update(_minute_index(fresh))
                    if float(args.sleep_seconds) > 0.0:
                        time.sleep(float(args.sleep_seconds))
                if pending_frames:
                    pending = pd.concat(pending_frames).sort_index()
                    pending = pending.loc[~pending.index.duplicated(keep="last")]
                    append_result = append_missing_kraken_execution_1m(
                        args.data_root, symbol, pending
                    )
                    fetched_rows += int(append_result["appended_rows"])
            final = _load_existing(store, symbol, windows)
            covered, required, fraction = _coverage(windows, final)
            if covered < required:
                status = "incomplete"
        except Exception as exc:  # keep other symbol shards progressing
            before_covered = covered = fetched_rows = fetched_requests = 0
            required = sum(int((end - start) / pd.Timedelta(minutes=1)) for start, end in windows)
            before_fraction = fraction = 0.0
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        result = {
            "symbol": symbol,
            "product_id": product_id,
            "windows": len(windows),
            "required_minutes": required,
            "covered_before": before_covered,
            "coverage_before": before_fraction,
            "fetched_requests": fetched_requests,
            "fetched_rows": fetched_rows,
            "covered_after": covered,
            "coverage_after": fraction,
            "status": status,
            "error": error,
            "elapsed_seconds": time.monotonic() - started,
        }
        results.append(result)
        print(
            f"[{partition_id}:{number:03d}/{total:03d}] {symbol} {status} "
            f"coverage={covered}/{required} ({fraction:.2%}) "
            f"requests={fetched_requests} fetched={fetched_rows} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )

    manifest = {
        "generated_by": "download_policy_execution_1m",
        "candidate_path": str(args.candidates),
        "candidate_sha256": _sha256(candidate_path),
        "stage_manifest": (
            {
                "path": str(Path(args.stage_manifest).resolve()),
                "sha256": _sha256(Path(args.stage_manifest)),
            }
            if args.stage_manifest
            else None
        ),
        "store_root": str(canonical_root),
        "timeframe": "1m",
        "horizon_minutes": int(args.horizon_minutes),
        "warmup_minutes": int(args.warmup_minutes),
        "window_semantics": "half_open_[decision_ts-warmup,decision_ts+horizon)",
        "partition_count": partition_count,
        "partition_id": partition_id,
        "symbol_ownership": "blake2b_modulo_partition_count",
        "symbols": total,
        "verify_only": bool(args.verify_only),
        "storage_contract": "canonical_kraken_execution_1m_immutable_append_missing_v1",
        "product_mapping_contract": (
            "frozen_product_id_from_candidate_input"
            if product_ids
            else "legacy_current_catalog_or_pf_fallback_not_historical_lineage_safe"
        ),
        "summary": {
            "required_minutes": int(sum(row["required_minutes"] for row in results)),
            "covered_minutes": int(sum(row["covered_after"] for row in results)),
            "fetched_requests": int(sum(row["fetched_requests"] for row in results)),
            "fetched_rows": int(sum(row["fetched_rows"] for row in results)),
            "ok_symbols": int(sum(row["status"] == "ok" for row in results)),
            "incomplete_symbols": int(sum(row["status"] == "incomplete" for row in results)),
            "failed_symbols": int(sum(row["status"] == "failed" for row in results)),
        },
        "results": results,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True), flush=True)
    return 0 if all(row["status"] == "ok" for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
