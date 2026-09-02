#!/usr/bin/env python3
"""Backfill exact 1m execution paths for simple-policy candidate rows.

Workers may share one output root only when ``partition-id`` assigns every
symbol to exactly one worker.  Windows are half-open and derived from candidate
timestamps plus the requested replay horizon.  Existing candles are preserved;
only missing minute buckets are fetched from Kraken Futures charts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import subprocess
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


_INVALID_PART_RE = re.compile(
    r"invalid execution_1m part (?P<path>.+?\.parquet):"
)


def _quarantine_exact_invalid_part(
    *,
    error: Exception,
    data_root: str,
    receipt_dir: Path,
) -> dict[str, str] | None:
    """Quarantine only the exact shard named by the immutable append validator.

    The standalone helper performs a second full Parquet read and refuses to
    move a readable source part.  This function is deliberately opt-in for
    historical source repair; the generic canonical store remains fail-closed.
    """
    match = _INVALID_PART_RE.search(str(error))
    if match is None:
        return None
    raw_path = match.group("path")
    source_path = Path(raw_path)
    if not source_path.is_absolute():
        source_path = (ROOT / source_path).resolve()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:16]
    receipt = receipt_dir / f"{digest}.json"
    if receipt.exists():
        raise FileExistsError(
            f"auto-quarantine receipt already exists for unresolved invalid part: {receipt}"
        )
    command = [
        sys.executable,
        str(ROOT / "scripts/quarantine_corrupt_kraken_execution_1m_parts.py"),
        "--data-root",
        str(data_root),
        "--part",
        str(source_path),
        "--receipt",
        str(receipt),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # A full independent Parquet validation plus SHA-256 on a large corrupt
        # shard can exceed thirty seconds on the shared local volume.  This
        # remains a bounded, single-file check; it must not turn into a broad
        # store scan or an unbounded recovery operation.
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().replace("\n", " ")
        raise RuntimeError(
            f"auto-quarantine failed for validator-named part {source_path}: {detail}"
        )
    return {"source_path": str(source_path), "receipt": str(receipt)}


def _merge_windows(
    decision_starts: Iterable[pd.Timestamp], horizon_minutes: int,
    warmup_minutes: int = 0, path_starts: Iterable[pd.Timestamp] | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return merged causal source intervals.

    Feature warm-up is anchored at the decision timestamp, while the execution
    path begins at the actual entry timestamp when one is supplied.  Treating
    both as the decision timestamp truncates a delayed-entry H12 path by the
    delay at the final candidate in a request.
    """
    horizon = pd.Timedelta(minutes=int(horizon_minutes))
    warmup = pd.Timedelta(minutes=max(int(warmup_minutes), 0))
    def _utc_minute(ts: pd.Timestamp) -> pd.Timestamp:
        value = pd.Timestamp(ts)
        value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return value.floor("min")

    decisions = list(decision_starts)
    paths = decisions if path_starts is None else list(path_starts)
    if len(decisions) != len(paths):
        raise ValueError("decision and execution path timestamp counts differ")
    raw = sorted(
        (_utc_minute(decision) - warmup, _utc_minute(path_start) + horizon)
        for decision, path_start in zip(decisions, paths, strict=True)
    )
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
    """Read only the yearly partitions intersecting the requested windows.

    ``PartitionedOHLCVStore.load`` is deliberately general-purpose and walks a
    symbol's complete historical directory before applying filename-level
    filtering.  The exact-exit backfill can have thousands of old one-minute
    parts for a symbol but needs a small, explicitly bounded date range.  This
    local fast path keeps the same immutable parquet source and timestamp
    de-duplication semantics while avoiding unrelated years (and the hourly
    actual-volume overlay, which is not an input to the OHLC path contract).
    """
    if not windows:
        return pd.DataFrame()
    start = min(value for value, _ in windows)
    end = max(value for _, value in windows)
    start = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    start_s, end_s = int(start.timestamp()), int(end.timestamp())
    symbol_dir = Path(store._get_symbol_dir(symbol))
    files: list[Path] = []
    for year in range(start.year, end.year + 1):
        for path in sorted((symbol_dir / f"year={year}").glob("*.parquet")):
            parts = path.stem.split("-")
            try:
                file_start, file_end = int(parts[-2]), int(parts[-1])
            except (IndexError, ValueError):
                continue
            if file_end >= start_s and file_start <= end_s:
                files.append(path)
    if not files:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        ).set_index(pd.DatetimeIndex([], tz="UTC", name="ts"))
    columns = ["ts", "open", "high", "low", "close", "volume"]
    def _read_exact_part(path: Path) -> pd.DataFrame | None:
        try:
            # ``pandas.read_parquet(Path)`` can dispatch through Arrow's
            # dataset discovery machinery.  For a fragmented but bounded
            # one-minute source that needlessly opens every sibling fragment
            # while discovering a schema.  We have already selected this
            # exact immutable file by its encoded bounds, so read it directly
            # as one Parquet file instead.  Values, column selection and the
            # fail-closed behaviour are unchanged.
            return pq.ParquetFile(path).read(columns=columns).to_pandas()
        except Exception:
            return None

    # Legacy writers sometimes left hundreds of tiny immutable parts for one
    # month.  Their reads are independent and already bounded by ``files``;
    # parallelising this local I/O avoids serial metadata latency without
    # modifying, compacting, or reordering source rows.
    workers = min(8, len(files))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            loaded = list(executor.map(_read_exact_part, files))
    else:
        loaded = [_read_exact_part(path) for path in files]
    frames = [frame for frame in loaded if frame is not None]
    if not frames:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        ).set_index(pd.DatetimeIndex([], tz="UTC", name="ts"))
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).set_index("ts").sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    out = out.loc[(out.index >= start) & (out.index <= end)]
    return store._downcast(out)


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
    parser.add_argument(
        "--product-id-override",
        action="append",
        default=[],
        metavar="SYMBOL=PRODUCT_ID",
        help=(
            "Frozen historical product mapping for a requested symbol. May be "
            "repeated; it is used only when the candidate input has no matching "
            "product_id and is persisted in the download manifest."
        ),
    )
    parser.add_argument("--no-compact", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--require-frozen-product-id", action="store_true",
        help=(
            "never query a current catalog for symbols lacking an explicit product_id in the candidate request; "
            "record those rows as skipped instead"
        ),
    )
    parser.add_argument(
        "--quarantine-invalid-source-parts",
        action="store_true",
        help=(
            "Historical-repair mode only: when the immutable append validator names "
            "an unreadable canonical part, independently validate and quarantine that "
            "one part, then retry the same append."
        ),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--stage-manifest",
        default="",
        help="Optional frozen request-stage manifest to hash-bind into the download audit.",
    )
    args = parser.parse_args()
    manifest_path = Path(args.manifest)

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
    entry_column = "entry_ts" if "entry_ts" in parquet_columns else timestamp_column
    if entry_column not in candidate_columns:
        candidate_columns.append(entry_column)
    if "product_id" in parquet_columns:
        candidate_columns.append("product_id")
    rename_columns = {timestamp_column: "timestamp", symbol_column: "symbol"}
    if entry_column != timestamp_column:
        rename_columns[entry_column] = "entry_ts"
    candidates = pd.read_parquet(candidate_path, columns=candidate_columns).rename(columns=rename_columns)
    if entry_column == timestamp_column:
        candidates["entry_ts"] = candidates["timestamp"]
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates["entry_ts"] = pd.to_datetime(candidates["entry_ts"], utc=True, errors="coerce")
    candidates = candidates.dropna(subset=["timestamp", "entry_ts", "symbol"])
    if (candidates["entry_ts"] < candidates["timestamp"]).any():
        raise ValueError("candidate entry timestamp precedes the decision timestamp")
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
                if bool(args.require_frozen_product_id) and not values:
                    # Retain the symbol in ``grouped`` so the main loop emits
                    # an auditable skipped result rather than silently
                    # dropping a frozen-universe member.  It will never use a
                    # current-catalog fallback under this flag.
                    continue
                raise ValueError(
                    f"{symbol} must map to exactly one frozen product_id, got {values}"
                )
            product_ids[str(symbol)] = values[0]
    overrides: dict[str, str] = {}
    for raw in args.product_id_override:
        symbol, separator, product_id = str(raw).partition("=")
        symbol = symbol.strip()
        product_id = product_id.strip()
        if not separator or not symbol or not product_id:
            raise ValueError("product-id-override must be SYMBOL=PRODUCT_ID")
        if symbol not in set(candidates["symbol"].astype(str)):
            raise ValueError(f"product-id-override symbol is absent from candidates: {symbol}")
        prior = product_ids.get(symbol)
        if prior is not None and prior != product_id:
            raise ValueError(
                f"product-id-override conflicts with candidate product_id for {symbol}: {prior}"
            )
        overrides[symbol] = product_id
    product_ids.update(overrides)
    grouped = {
        str(symbol): _merge_windows(
            group["timestamp"], int(args.horizon_minutes), int(args.warmup_minutes), group["entry_ts"]
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
        auto_quarantined_parts: list[dict[str, str]] = []
        product_id = product_ids.get(symbol)
        if args.require_frozen_product_id and not product_id:
            required = sum(int((end - start) / pd.Timedelta(minutes=1)) for start, end in windows)
            result = {
                "symbol": symbol,
                "product_id": None,
                "windows": len(windows),
                "required_minutes": required,
                "covered_before": 0,
                "coverage_before": 0.0,
                "fetched_requests": 0,
                "fetched_rows": 0,
                "covered_after": 0,
                "coverage_after": 0.0,
                "status": "skipped_missing_frozen_product_id",
                "error": "candidate request has no frozen historical product_id",
                "elapsed_seconds": time.monotonic() - started,
                "auto_quarantined_parts": [],
            }
            results.append(result)
            print(
                f"[{partition_id}:{number:03d}/{total:03d}] {symbol} {result['status']} ",
                "requests=0 fetched=0", flush=True,
            )
            continue
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
                    for attempt in range(32):
                        try:
                            append_result = append_missing_kraken_execution_1m(
                                args.data_root, symbol, pending
                            )
                            break
                        except Exception as exc:
                            if not args.quarantine_invalid_source_parts:
                                raise
                            quarantined = _quarantine_exact_invalid_part(
                                error=exc,
                                data_root=args.data_root,
                                receipt_dir=(
                                    manifest_path.parent
                                    / "auto_quarantine_validator_named_parts"
                                ),
                            )
                            if quarantined is None:
                                raise
                            auto_quarantined_parts.append(quarantined)
                    else:
                        raise RuntimeError(
                            f"exhausted auto-quarantine retries for {symbol}"
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
            "auto_quarantined_parts": auto_quarantined_parts,
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
        "window_semantics": "half_open_[decision_ts-warmup,entry_ts+horizon)",
        "partition_count": partition_count,
        "partition_id": partition_id,
        "symbol_ownership": "blake2b_modulo_partition_count",
        "symbols": total,
        "verify_only": bool(args.verify_only),
        "require_frozen_product_id": bool(args.require_frozen_product_id),
        "storage_contract": "canonical_kraken_execution_1m_immutable_append_missing_v1",
        "product_mapping_contract": (
            "frozen_product_id_from_candidate_input_or_explicit_override"
            if product_ids
            else "legacy_current_catalog_or_pf_fallback_not_historical_lineage_safe"
        ),
        "product_id_overrides": dict(sorted(overrides.items())),
        "summary": {
            "required_minutes": int(sum(row["required_minutes"] for row in results)),
            "covered_minutes": int(sum(row["covered_after"] for row in results)),
            "fetched_requests": int(sum(row["fetched_requests"] for row in results)),
            "fetched_rows": int(sum(row["fetched_rows"] for row in results)),
            "ok_symbols": int(sum(row["status"] == "ok" for row in results)),
            "incomplete_symbols": int(sum(row["status"] == "incomplete" for row in results)),
            "failed_symbols": int(sum(row["status"] == "failed" for row in results)),
            "skipped_missing_frozen_product_id_symbols": int(sum(row["status"] == "skipped_missing_frozen_product_id" for row in results)),
        },
        "results": results,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True), flush=True)
    return 0 if all(row["status"] in {"ok", "skipped_missing_frozen_product_id"} for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
