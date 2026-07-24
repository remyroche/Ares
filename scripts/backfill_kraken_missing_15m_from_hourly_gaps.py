#!/usr/bin/env python3
"""Backfill only 15-minute Kraken raw bars underlying hourly raw-data gaps.

This intentionally stops at the canonical raw 15-minute cache. It does not
resample hourly bars, materialize features, regenerate labels, or score models.
The separate repair stages can therefore validate the downloaded source before
it affects any model input.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import _resolve_perp_symbol, make_perp_exchange
from extreme_price_movements.hf_data_loader import (
    configure_hf_data_dirs,
    sync_15m_ohlcv_range,
)
from extreme_price_movements.timestamp_contract import to_utc_timestamp


def _utc(value: str) -> pd.Timestamp:
    return to_utc_timestamp(pd.Timestamp(value))


def _candidate_symbols(path: Path) -> list[str]:
    frame = pd.read_parquet(path, columns=[])
    if "symbol" in frame.columns:
        values = frame["symbol"].astype(str)
    else:
        values = pd.Index(frame.index).astype(str)
    return sorted({symbol for symbol in values if "/" in symbol})


def _hourly_index(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    partition = root / f"symbol={symbol.replace('/', '_')}"
    files = sorted(partition.rglob("*.parquet"))
    if not files:
        return pd.DatetimeIndex([], tz="UTC")
    # The canonical store names every 1h part as part-<first_epoch>-<last_epoch>.
    # Reading whole parquet histories merely to find raw gaps is expensive and
    # can itself exhaust memory. Use the partition metadata first; malformed
    # legacy files fall back to a one-column read only for that file.
    seconds: list[int] = []
    fallback_files: list[Path] = []
    start_s, end_s = int(start.timestamp()), int(end.timestamp())
    for file_path in files:
        parts = file_path.stem.split("-")
        if len(parts) < 3:
            fallback_files.append(file_path)
            continue
        try:
            first_s, last_s = int(parts[-2]), int(parts[-1])
        except ValueError:
            fallback_files.append(file_path)
            continue
        if last_s < start_s or first_s > end_s:
            continue
        seconds.extend(range(max(first_s, start_s), min(last_s, end_s) + 1, 3600))

    timestamps: list[pd.Series] = []
    for file_path in fallback_files:
        values = pd.read_parquet(file_path, columns=["ts"])["ts"]
        timestamps.append(pd.to_datetime(values, utc=True, errors="coerce"))
    if timestamps:
        seconds.extend(
            (pd.DatetimeIndex(pd.concat(timestamps, ignore_index=True).dropna())
             .floor("h").asi8 // 1_000_000_000).tolist()
        )
    if not seconds:
        return pd.DatetimeIndex([], tz="UTC")
    index = pd.to_datetime(np.asarray(seconds, dtype=np.int64), unit="s", utc=True)
    return pd.DatetimeIndex(index.unique()).sort_values()


def _contiguous_ranges(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield hourly holes inside observed coverage, excluding listing boundaries."""
    if index.empty:
        return []
    observed_start = max(start, index.min().floor("h"))
    observed_end = min(end, index.max().floor("h"))
    if observed_start > observed_end:
        return []
    expected = pd.date_range(observed_start, observed_end, freq="h", tz="UTC")
    missing = expected.difference(index.floor("h").unique())
    if missing.empty:
        return []
    values = missing.asi8
    breaks = np.flatnonzero(np.diff(values) > pd.Timedelta(hours=1).value)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(missing) - 1]
    return [(missing[int(left)], missing[int(right)]) for left, right in zip(starts, ends)]


def _gap_manifest(symbols: list[str], hourly_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        index = _hourly_index(hourly_root, symbol, start, end)
        for gap_start, gap_end in _contiguous_ranges(index, start, end):
            rows.append(
                {
                    "symbol": symbol,
                    "hourly_gap_start_utc": gap_start,
                    "hourly_gap_end_utc": gap_end,
                    "missing_hourly_bars": int((gap_end - gap_start) / pd.Timedelta(hours=1)) + 1,
                    "raw_15m_start_utc": gap_start,
                    "raw_15m_end_utc": gap_end + pd.Timedelta(minutes=45),
                }
            )
    return pd.DataFrame(rows)


def _valid_15m_rows(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    if frame.empty:
        return 0
    subset = frame.loc[(frame.index >= start) & (frame.index <= end)]
    required = [column for column in ("open", "high", "low", "close", "volume") if column in subset]
    if len(required) < 4:
        return 0
    return int(subset[required].notna().all(axis=1).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--candidate-matrix", required=True, type=Path)
    parser.add_argument("--start", required=True, help="Inclusive UTC timestamp")
    parser.add_argument("--end", required=True, help="Inclusive UTC timestamp")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--max-ranges", type=int, default=0, help="0 means all gaps")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start, end = _utc(args.start), _utc(args.end)
    if end < start:
        raise ValueError("--end must be on or after --start")
    data_root = Path(args.data_root)
    hourly_root = data_root / "exchanges" / "krakenfutures" / "ohlcv"
    if not hourly_root.exists():
        raise FileNotFoundError(f"Hourly Kraken store not found: {hourly_root}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    symbols = _candidate_symbols(args.candidate_matrix)
    manifest = _gap_manifest(symbols, hourly_root, start, end)
    manifest_path = args.report_dir / "raw_missing_15m_gap_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"candidate_symbols={len(symbols)} gap_ranges={len(manifest)} missing_hourly_bars={int(manifest['missing_hourly_bars'].sum()) if not manifest.empty else 0}")
    print(f"manifest={manifest_path}")
    if args.dry_run or manifest.empty:
        return 0

    if args.max_ranges > 0:
        manifest = manifest.iloc[: args.max_ranges].copy()

    canonical_root = data_root / "exchanges" / "krakenfutures"
    cache_15m, _ = configure_hf_data_dirs(
        market_data_root=canonical_root,
        exchange_id="kraken",
        market_mode="perps",
        force_canonical=True,
    )
    print(f"canonical_15m_cache={cache_15m}")
    exchange = make_perp_exchange()
    results: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        gap_start = _utc(str(row.raw_15m_start_utc))
        gap_end = _utc(str(row.raw_15m_end_utc))
        resolved = _resolve_perp_symbol(exchange, str(row.symbol))
        result = {
            "symbol": row.symbol,
            "resolved_exchange_symbol": resolved or "",
            "raw_15m_start_utc": gap_start,
            "raw_15m_end_utc": gap_end,
            "missing_hourly_bars": int(row.missing_hourly_bars),
            "expected_15m_bars": int((gap_end - gap_start) / pd.Timedelta(minutes=15)) + 1,
            "valid_15m_bars": 0,
            "status": "pending",
            "error": "",
        }
        if not resolved:
            result["status"] = "unavailable_on_kraken"
            results.append(result)
            continue
        try:
            downloaded = sync_15m_ohlcv_range(
                exchange,
                resolved,
                gap_start,
                gap_end,
                full_backfill=False,
            )
            result["valid_15m_bars"] = _valid_15m_rows(downloaded, gap_start, gap_end)
            result["status"] = "complete" if result["valid_15m_bars"] == result["expected_15m_bars"] else "partial_or_unavailable"
        except Exception as exc:  # Keep going so a delisted symbol cannot block the repair.
            result["status"] = "download_error"
            result["error"] = repr(exc)
        results.append(result)
        print(f"{result['status']} {row.symbol} {gap_start}..{gap_end} {result['valid_15m_bars']}/{result['expected_15m_bars']}")

    result_frame = pd.DataFrame(results)
    result_path = args.report_dir / "raw_missing_15m_fetch_results.csv"
    result_frame.to_csv(result_path, index=False)
    complete = int((result_frame["status"] == "complete").sum()) if not result_frame.empty else 0
    print(f"completed_ranges={complete}/{len(result_frame)} results={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
