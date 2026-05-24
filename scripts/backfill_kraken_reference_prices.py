#!/usr/bin/env python3
"""Incrementally backfill Kraken spot/index/mark/premium reference prices.

This script updates only missing reference columns in existing Kraken Futures
OHLCV partitions. It uses:

* Kraken spot OHLCV for ``spot_*`` columns.
* Kraken Futures public chart ticks for ``mark_*``, ``index_*`` and
  ``premium_index_*`` columns.

It intentionally does not synthesize index from spot or spot from index. If an
official source has no rows for a gap, the gap remains missing and is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    _fetch_kraken_futures_chart_ohlcv,
    make_perp_exchange,
    make_spot_exchange,
)
from extreme_price_movements.utils import tprint


REFERENCE_TICKS = {
    "mark": {
        "tick": "mark",
        "cols": ("mark_open", "mark_high", "mark_low", "mark_close"),
        "price_col": "mark_price",
    },
    "index": {
        "tick": "index",
        "cols": ("index_open", "index_high", "index_low", "index_close"),
        "price_col": "index_price",
    },
    "premium": {
        "tick": "premiumIndex",
        "cols": (
            "premium_index_open",
            "premium_index_high",
            "premium_index_low",
            "premium_index_close",
        ),
        "price_col": "premium_index",
    },
}

KRAKEN_FUTURES_CHART_TICKS = {"mark", "spot", "trade"}


def _load_manifest(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("symbols") if isinstance(payload, dict) else payload
    out: list[dict[str, str]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        perp = str(row.get("perp_symbol") or "").strip()
        spot = str(row.get("spot_symbol") or "").strip()
        if perp:
            out.append({"perp_symbol": perp, "spot_symbol": spot})
    return out


def _to_utc_hour_index(index: Iterable[Any]) -> pd.DatetimeIndex:
    idx = pd.to_datetime(pd.Index(index), utc=True, errors="coerce")
    idx = pd.DatetimeIndex(idx).dropna().floor("h")
    return idx.drop_duplicates().sort_values()


def _finite_positive(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.where(np.isfinite(vals) & (vals > 0.0))


def _finite_any(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.where(np.isfinite(vals))


def _missing_ranges(
    timestamps: pd.DatetimeIndex,
    present_mask: pd.Series,
    *,
    max_gap_hours: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if timestamps.empty:
        return []
    aligned = present_mask.reindex(timestamps)
    mask = pd.Series(False, index=timestamps)
    valid_idx = aligned.index[aligned.notna()]
    if len(valid_idx):
        mask.loc[valid_idx] = aligned.loc[valid_idx].astype(bool)
    missing = timestamps[~mask.to_numpy()]
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


def _ccxt_ohlcv_frame(rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce").dt.floor("h")
    df = df.dropna(subset=["ts"]).drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = df["open"].gt(0) & df["high"].gt(0) & df["low"].gt(0) & df["close"].gt(0)
    return df.loc[valid, ["open", "high", "low", "close", "volume"]].astype(np.float32)


def _fetch_spot_ohlcv(
    exchange: Any,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    sleep: float,
) -> pd.DataFrame:
    rows: list[list[Any]] = []
    since_ms = int(start.value // 10**6)
    end_ms = int(end.value // 10**6)
    retry_sleep = max(1.0, sleep)
    while since_ms < end_ms:
        batch = None
        for attempt in range(8):
            try:
                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe="1h",
                    since=since_ms,
                    limit=720,
                )
                retry_sleep = max(1.0, sleep)
                break
            except Exception as exc:
                text = str(exc)
                rate_limited = (
                    "too many requests" in text.lower()
                    or "rate limit" in text.lower()
                    or "EGeneral:Too many requests" in text
                )
                if not rate_limited or attempt >= 7:
                    raise
                wait = min(90.0, retry_sleep * (1.7 ** attempt))
                tprint(f"Kraken spot rate limited for {symbol}; retrying in {wait:.1f}s")
                time.sleep(wait)
        if not batch:
            break
        for row in batch:
            try:
                ts_ms = int(row[0])
            except Exception:
                continue
            if ts_ms >= end_ms:
                continue
            rows.append(row)
        max_seen = max(int(row[0]) for row in batch)
        if max_seen <= since_ms:
            break
        since_ms = max_seen + 60 * 60 * 1000
        time.sleep(max(0.0, sleep))
    return _ccxt_ohlcv_frame(rows)


def _load_spot_reference(
    *,
    spot_store: PartitionedOHLCVStore,
    spot_exchange: Any | None,
    spot_symbol: str,
    timestamps: pd.DatetimeIndex,
    max_gap_hours: int,
    sleep: float,
    dry_run: bool,
) -> tuple[pd.DataFrame, int, int]:
    existing = spot_store.load(spot_symbol)
    if existing.empty:
        existing = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    existing.index = _to_utc_hour_index(existing.index)
    present = (
        _finite_positive(existing["close"]).notna()
        if "close" in existing.columns
        else pd.Series(False, index=existing.index)
    )
    ranges = _missing_ranges(timestamps, present, max_gap_hours=max_gap_hours)
    fetched_rows = 0
    if ranges and not dry_run and spot_exchange is not None:
        frames = []
        for start, end in ranges:
            fetched = _fetch_spot_ohlcv(
                spot_exchange,
                spot_symbol,
                start,
                end,
                sleep=sleep,
            )
            if not fetched.empty:
                frames.append(fetched)
            time.sleep(max(0.0, sleep))
        if frames:
            incoming = pd.concat(frames).sort_index()
            incoming = incoming[~incoming.index.duplicated(keep="last")]
            existing_hours = _to_utc_hour_index(existing.index)
            new_rows = incoming.loc[~incoming.index.isin(existing_hours)]
            if not new_rows.empty:
                fetched_rows = int(len(new_rows))
                spot_store.save_partitioned(spot_symbol, new_rows, defer_compact=True)
                for year in sorted(set(int(y) for y in new_rows.index.year)):
                    spot_store.compact_partition(spot_symbol, year)
            existing = spot_store.load(spot_symbol)
            existing.index = _to_utc_hour_index(existing.index)
    return existing, len(ranges), fetched_rows


def _reference_frame_from_tick(fetch: pd.DataFrame, kind: str) -> pd.DataFrame:
    spec = REFERENCE_TICKS[kind]
    out = pd.DataFrame(index=fetch.index)
    for src, dst in zip(("open", "high", "low", "close"), spec["cols"]):
        out[dst] = _finite_any(fetch[src]) if src in fetch.columns else np.nan
    out[spec["price_col"]] = out[spec["cols"][-1]]
    if kind in {"mark", "index"}:
        for col in list(out.columns):
            out[col] = _finite_positive(out[col])
    else:
        for col in list(out.columns):
            out[col] = _finite_any(out[col])
    return out.dropna(how="all").astype(np.float32)


def _reference_tick_available(
    exchange: Any,
    symbol: str,
    tick: str,
    *,
    probe_since: pd.Timestamp,
) -> tuple[bool, str]:
    """Return whether Kraken accepts this official reference tick for symbol.

    Some Kraken Futures contracts expose mark charts but not index/premium
    charts. Probing once prevents repeatedly querying known-unavailable
    endpoints for every historical gap.
    """

    try:
        rows = exchange.fetch_ohlcv(
            symbol,
            timeframe="1h",
            since=int(probe_since.value // 10**6),
            limit=3,
            params={"price": tick},
        )
        return True, f"ok_rows={len(rows or [])}"
    except Exception as exc:
        text = str(exc)
        if "400" in text or "Bad Request" in text or "not found" in text.lower():
            return False, f"{exc.__class__.__name__}: {text[:220]}"
        return True, f"probe_warning={exc.__class__.__name__}: {text[:220]}"


def _merge_update_perp_years(
    *,
    store: PartitionedOHLCVStore,
    symbol: str,
    updates: pd.DataFrame,
    dry_run: bool,
) -> int:
    if updates.empty:
        return 0
    existing = store.load(symbol)
    if existing.empty:
        return 0
    existing.index = _to_utc_hour_index(existing.index)
    updates = updates.copy()
    updates.index = _to_utc_hour_index(updates.index)
    common = existing.index.intersection(updates.index)
    if common.empty:
        return 0
    merged_rows = existing.loc[common].copy()
    changed = pd.Series(False, index=common)
    for col in updates.columns:
        incoming = updates[col].reindex(common)
        if col not in merged_rows.columns:
            merged_rows[col] = np.nan
        before = pd.to_numeric(merged_rows[col], errors="coerce")
        before_missing = before.isna()
        incoming_num = pd.to_numeric(incoming, errors="coerce")
        if col in {
            "premium_index",
            "premium_index_open",
            "premium_index_high",
            "premium_index_low",
            "premium_index_close",
        }:
            can_update = before_missing & incoming_num.notna()
        elif col.endswith("_volume") or col == "volume":
            can_update = (
                (before_missing & incoming_num.notna())
                | (before.le(0) & incoming_num.gt(0))
            )
        else:
            before_missing |= before.le(0)
            can_update = before_missing & incoming_num.gt(0)
        if can_update.any():
            merged_rows.loc[can_update, col] = incoming_num.loc[can_update]
            changed |= can_update
    if not changed.any():
        return 0
    out = merged_rows.loc[changed].copy()
    if not dry_run:
        store.save_partitioned(symbol, out, defer_compact=True)
        for year in sorted(set(int(y) for y in out.index.year)):
            store.compact_partition(symbol, year)
    return int(len(out))


def _spot_update_frame(spot: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    if spot.empty:
        return pd.DataFrame(index=timestamps)
    aligned = spot.reindex(timestamps)
    out = pd.DataFrame(index=timestamps)
    mapping = {
        "open": "spot_open",
        "high": "spot_high",
        "low": "spot_low",
        "close": "spot_close",
        "volume": "spot_volume",
    }
    for src, dst in mapping.items():
        if src not in aligned.columns:
            continue
        if src == "volume":
            out[dst] = _finite_any(aligned[src])
        else:
            out[dst] = _finite_positive(aligned[src])
    return out.dropna(how="all").astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_verified_universe_latest.json")
    parser.add_argument("--perp-root", default="data_perp/exchanges/krakenfutures")
    parser.add_argument("--spot-root", default="data_spot/exchanges/kraken")
    parser.add_argument("--start-ts", default="")
    parser.add_argument("--end-ts", default="")
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument("--max-gap-hours", type=int, default=720)
    parser.add_argument("--rate-limit-ms", type=int, default=800)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--ticks", default="spot,mark,index,premium")
    parser.add_argument("--reference-probe-lookback-hours", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = _load_manifest(Path(args.manifest))
    ticks = {str(t).strip().lower() for t in args.ticks.split(",") if str(t).strip()}
    partition_count = max(1, int(args.partition_count))
    partition_id = int(args.partition_id)
    if partition_id < 0 or partition_id >= partition_count:
        raise ValueError(f"invalid partition {partition_id}/{partition_count}")
    rows = [row for i, row in enumerate(rows) if i % partition_count == partition_id]

    start_ts = pd.to_datetime(args.start_ts, utc=True, errors="coerce") if args.start_ts else None
    end_ts = (
        pd.Timestamp(args.end_ts, tz="UTC").floor("h")
        if args.end_ts
        else pd.Timestamp.utcnow().floor("h")
    )

    os.environ["EPM_EXCHANGE"] = "kraken"
    perp_store = PartitionedOHLCVStore(args.perp_root, "1h")
    spot_store = PartitionedOHLCVStore(args.spot_root, "1h")
    spot_exchange = None
    perp_exchange = None
    if not args.dry_run:
        if "spot" in ticks:
            spot_exchange = make_spot_exchange()
            spot_exchange.rateLimit = max(
                int(getattr(spot_exchange, "rateLimit", 0) or 0),
                int(args.rate_limit_ms),
            )
        if ticks.intersection({"mark", "index", "premium"}):
            perp_exchange = make_perp_exchange()
            perp_exchange.rateLimit = max(
                int(getattr(perp_exchange, "rateLimit", 0) or 0),
                int(args.rate_limit_ms),
            )

    stats: dict[str, Any] = {
        "symbols": len(rows),
        "dry_run": bool(args.dry_run),
        "ticks": sorted(ticks),
        "spot_gap_ranges": 0,
        "spot_downloaded_rows": 0,
        "perp_reference_gap_ranges": 0,
        "perp_reference_downloaded_rows": 0,
        "perp_rows_updated": 0,
        "reference_tick_unavailable": {},
        "skipped_no_perp": 0,
        "failed": [],
    }
    tick_availability: dict[tuple[str, str], bool] = {}

    for i, row in enumerate(rows, start=1):
        perp_symbol = row["perp_symbol"]
        spot_symbol = row.get("spot_symbol") or ""
        try:
            perp = perp_store.load(perp_symbol)
            if perp.empty:
                stats["skipped_no_perp"] += 1
                tprint(f"[{i:04d}/{len(rows):04d}] {perp_symbol}: skip no local perp rows")
                continue
            perp.index = _to_utc_hour_index(perp.index)
            if start_ts is not None:
                perp = perp.loc[perp.index >= start_ts]
            perp = perp.loc[perp.index <= end_ts]
            if perp.empty:
                continue
            timestamps = _to_utc_hour_index(perp.index)
            symbol_updates: list[pd.DataFrame] = []

            if "spot" in ticks and spot_symbol:
                spot, gap_ranges, fetched = _load_spot_reference(
                    spot_store=spot_store,
                    spot_exchange=spot_exchange,
                    spot_symbol=spot_symbol,
                    timestamps=timestamps,
                    max_gap_hours=int(args.max_gap_hours),
                    sleep=float(args.sleep),
                    dry_run=bool(args.dry_run),
                )
                stats["spot_gap_ranges"] += int(gap_ranges)
                stats["spot_downloaded_rows"] += int(fetched)
                symbol_updates.append(_spot_update_frame(spot, timestamps))

            for kind, spec in REFERENCE_TICKS.items():
                if kind not in ticks:
                    continue
                if str(spec["tick"]) not in KRAKEN_FUTURES_CHART_TICKS:
                    stats["reference_tick_unavailable"].setdefault(
                        f"GLOBAL:{spec['tick']}",
                        "Kraken Futures chart candle tick types are mark, spot and trade; "
                        "this tick is not an official historical chart source.",
                    )
                    continue
                target_cols = list(spec["cols"]) + [str(spec["price_col"])]
                present = pd.Series(True, index=timestamps)
                for col in target_cols:
                    if col not in perp.columns:
                        col_present = pd.Series(False, index=timestamps)
                    elif kind == "premium":
                        col_present = _finite_any(perp[col]).notna().reindex(timestamps).fillna(False)
                    else:
                        col_present = _finite_positive(perp[col]).notna().reindex(timestamps).fillna(False)
                    present &= col_present.astype(bool)
                ranges = _missing_ranges(
                    timestamps,
                    present,
                    max_gap_hours=int(args.max_gap_hours),
                )
                stats["perp_reference_gap_ranges"] += int(len(ranges))
                if not ranges or args.dry_run or perp_exchange is None:
                    continue
                availability_key = (perp_symbol, str(spec["tick"]))
                if availability_key not in tick_availability:
                    probe_since = max(
                        timestamps.min(),
                        end_ts - pd.Timedelta(hours=max(1, int(args.reference_probe_lookback_hours))),
                    )
                    available, reason = _reference_tick_available(
                        perp_exchange,
                        perp_symbol,
                        str(spec["tick"]),
                        probe_since=probe_since,
                    )
                    tick_availability[availability_key] = bool(available)
                    if not available:
                        stats["reference_tick_unavailable"][
                            f"{perp_symbol}:{spec['tick']}"
                        ] = reason
                        tprint(
                            f"[{i:04d}/{len(rows):04d}] {perp_symbol}: "
                            f"skip unavailable {spec['tick']} reference ({reason})"
                        )
                if not tick_availability[availability_key]:
                    continue
                frames = []
                for start, end in ranges:
                    fetched = _fetch_kraken_futures_chart_ohlcv(
                        perp_exchange,
                        perp_symbol,
                        spec["tick"],
                        int(start.value // 10**6),
                        int(end.value // 10**6),
                        timeframe="1h",
                    )
                    if not fetched.empty:
                        frames.append(_reference_frame_from_tick(fetched, kind))
                    time.sleep(max(0.0, float(args.sleep)))
                if frames:
                    ref = pd.concat(frames).sort_index()
                    ref = ref[~ref.index.duplicated(keep="last")]
                    stats["perp_reference_downloaded_rows"] += int(len(ref))
                    symbol_updates.append(ref)

            if symbol_updates:
                updates = pd.concat(symbol_updates, axis=1)
                updates = updates.loc[:, ~updates.columns.duplicated(keep="last")]
                updated_rows = _merge_update_perp_years(
                    store=perp_store,
                    symbol=perp_symbol,
                    updates=updates,
                    dry_run=bool(args.dry_run),
                )
                stats["perp_rows_updated"] += int(updated_rows)
                if updated_rows or args.dry_run:
                    tprint(
                        f"[{i:04d}/{len(rows):04d}] {perp_symbol}: "
                        f"updated_rows={updated_rows}"
                    )
        except Exception as exc:
            stats["failed"].append(f"{perp_symbol}: {exc.__class__.__name__}: {exc}")
            tprint(f"[{i:04d}/{len(rows):04d}] {perp_symbol}: FAIL {exc}")

    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if not stats["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
