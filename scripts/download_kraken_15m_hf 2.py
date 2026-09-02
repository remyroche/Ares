#!/usr/bin/env python3
"""Incrementally download Kraken Futures 15m OHLCV into the HF cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _load_label_requirements(
    labels_dir: Path,
    *,
    signal_timeframe: str,
    path_padding_hours: float,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the exact causal 15m range required by retained label rows."""
    manifest_path = labels_dir / "labels_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing label manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", {}) if isinstance(payload, dict) else {}
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError(f"No datasets found in {manifest_path}")

    starts: dict[str, pd.Timestamp] = {}
    ends: dict[str, pd.Timestamp] = {}
    signal_delta = pd.Timedelta(signal_timeframe)
    path_padding = pd.Timedelta(hours=max(0.0, float(path_padding_hours)))
    files = sorted(
        {
            str(meta.get("file"))
            for meta in datasets.values()
            if isinstance(meta, dict) and str(meta.get("file") or "").endswith(".parquet")
        }
    )
    for file_name in files:
        path = labels_dir / file_name
        frame = pd.read_parquet(path, columns=["__ts__", "__symbol__"])
        ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        symbols = frame["__symbol__"].astype(str)
        valid = ts.notna() & symbols.ne("")
        if not bool(valid.any()):
            continue
        compact = pd.DataFrame(
            {"symbol": symbols.loc[valid].to_numpy(), "ts": ts.loc[valid].to_numpy()}
        )
        grouped = compact.groupby("symbol", sort=False)["ts"].agg(["min", "max"])
        for symbol, row in grouped.iterrows():
            required_start = pd.Timestamp(row["min"]).tz_convert("UTC") + signal_delta
            required_end = (
                pd.Timestamp(row["max"]).tz_convert("UTC")
                + signal_delta
                + path_padding
            )
            starts[str(symbol)] = min(starts.get(str(symbol), required_start), required_start)
            ends[str(symbol)] = max(ends.get(str(symbol), required_end), required_end)
    return {symbol: (starts[symbol], ends[symbol]) for symbol in sorted(starts)}


def _regularize_15m_candles(frame: pd.DataFrame) -> pd.DataFrame:
    """Regularize a 15-minute series while preserving source provenance.

    ``exchange_observed`` is true only for a candle returned by Kraken.  A
    flat/zero-volume row introduced locally to bridge a missing timestamp is
    explicitly false.  Older cache rows, written before this contract was
    introduced, remain unknown (``<NA>``) so downstream consumers can retain
    their conservative legacy treatment rather than silently trusting them.
    """
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame
    out = frame.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.loc[~out.index.isna()]
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    if "exchange_observed" not in out.columns:
        out["exchange_observed"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    else:
        out["exchange_observed"] = out["exchange_observed"].astype("boolean")
    if len(out) <= 1:
        return out
    full_index = pd.date_range(
        out.index.min().floor("15min"),
        out.index.max().floor("15min"),
        freq="15min",
        tz="UTC",
    )
    out = out.reindex(full_index)
    previous_close = pd.to_numeric(out["close"], errors="coerce").ffill()
    missing = out["close"].isna() & previous_close.notna()
    for column in ("open", "high", "low", "close"):
        out.loc[missing, column] = previous_close.loc[missing]
    if "volume" in out.columns:
        out.loc[missing, "volume"] = 0.0
    out.loc[missing, "exchange_observed"] = False
    return out.dropna(subset=["open", "high", "low", "close"])


def _merge_refresh_frames(
    existing: pd.DataFrame | None,
    refreshed: list[pd.DataFrame],
) -> pd.DataFrame:
    """Merge a forced refresh atomically at returned timestamps only.

    A short Kraken response must never erase an older cached bar merely
    because that bar falls inside the requested refresh interval.  Refreshed
    candles override cache values at the exact timestamps returned by Kraken;
    missing timestamps retain their existing values and provenance.
    """
    if (existing is None or existing.empty) and not any(
        frame is not None and not frame.empty for frame in refreshed
    ):
        return pd.DataFrame()
    # ``sort_index`` does not provide a safe precedence rule for duplicate
    # timestamps.  Explicitly remove the returned timestamps from the cache
    # before appending each source frame, so an authoritative Kraken candle
    # always replaces an older local fill while a short response leaves all
    # other cache timestamps untouched.
    combined = (
        pd.DataFrame() if existing is None or existing.empty else existing.copy()
    )
    for frame in refreshed:
        if frame is None or frame.empty:
            continue
        replacement = frame.copy()
        replacement.index = pd.to_datetime(replacement.index, utc=True, errors="coerce")
        replacement = replacement.loc[~replacement.index.isna()]
        replacement = replacement.loc[
            ~replacement.index.duplicated(keep="last")
        ].sort_index()
        if replacement.empty:
            continue
        if not combined.empty:
            combined.index = pd.to_datetime(combined.index, utc=True, errors="coerce")
            combined = combined.loc[~combined.index.isin(replacement.index)]
        combined = pd.concat([combined, replacement], axis=0)
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    return _regularize_15m_candles(combined)


def _partition_symbols(
    symbols: list[str], *, partition_count: int, partition_id: int, order: str
) -> list[str]:
    selected = [
        symbol
        for idx, symbol in enumerate(sorted(symbols))
        if idx % int(partition_count) == int(partition_id)
    ]
    return list(reversed(selected)) if str(order) == "alpha_desc" else selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_verified_universe_latest.json",
    )
    parser.add_argument(
        "--target-free-manifest", type=Path,
        help=(
            "Optional strict-R3 target-free candidate manifest.  Its frozen "
            "source_map keys replace the broader exchange universe."
        ),
    )
    parser.add_argument("--lookback-days", type=float, default=1460.0)
    parser.add_argument(
        "--symbol",
        action="append",
        default=None,
        help=(
            "Restrict the refresh to one or more explicit canonical symbols. "
            "Repeat the option for multiple symbols; default universe behavior "
            "is unchanged when omitted."
        ),
    )
    parser.add_argument(
        "--force-start",
        type=str,
        default=None,
        help="Explicit UTC start for a refresh range; bypasses the cache-span shortcut.",
    )
    parser.add_argument(
        "--force-end",
        type=str,
        default=None,
        help="Explicit UTC end (exclusive) for a refresh range; requires --force-start.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help=(
            "Optional labels directory. When set, download only each label "
            "symbol's required causal path range instead of a blanket lookback."
        ),
    )
    parser.add_argument("--signal-timeframe", default="1h")
    parser.add_argument(
        "--path-padding-hours",
        type=float,
        default=24.0,
        help="Future path required after signal close (96 15m bars = 24h).",
    )
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
    parser.add_argument(
        "--regularize-only",
        action="store_true",
        help="Do not download; regularize existing sparse Kraken candles in place.",
    )
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

    requirements: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    if args.labels_dir is not None:
        requirements = _load_label_requirements(
            Path(args.labels_dir),
            signal_timeframe=str(args.signal_timeframe),
            path_padding_hours=float(args.path_padding_hours),
        )
        all_symbols = sorted(requirements)
    elif args.target_free_manifest is not None:
        target_free = json.loads(args.target_free_manifest.read_text())
        all_symbols = sorted(str(value) for value in target_free.get("source_map", {}))
        if not all_symbols:
            raise ValueError("target-free manifest has no frozen source_map universe")
    else:
        all_symbols = sorted(_load_symbols(Path(args.manifest)))
    if args.symbol:
        requested = {str(symbol) for symbol in args.symbol}
        # Explicit repair requests may legitimately target delisted contracts
        # that are absent from the *current* verified universe.  Preserve the
        # exact requested identity and let the exchange/archive adapter report
        # whether historical charts remain available.
        all_symbols = sorted(requested)
    # Partition membership must be invariant to traversal order. Reversing the
    # full list before modulo partitioning makes asc/desc workers overlap and
    # leaves other symbols uncovered.
    symbols = _partition_symbols(
        all_symbols,
        partition_count=partition_count,
        partition_id=partition_id,
        order=str(args.order),
    )

    if bool(args.regularize_only):
        stats = {"ok": 0, "empty": 0, "failed": 0, "symbols": len(symbols)}
        for i, symbol in enumerate(symbols, start=1):
            try:
                existing = _load_existing_data(symbol, allow_quote_fallback=False)
                if existing is None or existing.empty:
                    stats["empty"] += 1
                    continue
                regularized = _regularize_15m_candles(existing)
                _save_data(symbol, regularized)
                stats["ok"] += 1
                tprint(
                    f"[{i:04d}/{len(symbols):04d}] {symbol} regularized "
                    f"rows={len(existing)}->{len(regularized)}"
                )
            except Exception as exc:
                stats["failed"] += 1
                tprint(f"[{i:04d}/{len(symbols):04d}] {symbol} failed: {exc}")
        tprint(f"Kraken 15m HF regularization complete: {stats}")
        print(json.dumps(stats, indent=2, sort_keys=True))
        return 0 if stats["failed"] == 0 else 2

    _load_local_env_if_present()
    ex = make_perp_exchange()
    ex.rateLimit = max(int(getattr(ex, "rateLimit", 0) or 0), int(args.rate_limit_ms))

    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=float(args.lookback_days))).floor("15min")
    until = pd.Timestamp.now(tz="UTC").floor("15min")
    if args.force_start is not None:
        if args.force_end is None:
            raise ValueError("--force-end is required with --force-start")
        since = pd.Timestamp(args.force_start)
        until = pd.Timestamp(args.force_end)
        if since.tzinfo is None:
            since = since.tz_localize("UTC")
        else:
            since = since.tz_convert("UTC")
        if until.tzinfo is None:
            until = until.tz_localize("UTC")
        else:
            until = until.tz_convert("UTC")
        since, until = since.floor("15min"), until.ceil("15min")
    tprint(
        "Kraken 15m HF download start: "
        f"symbols={len(symbols)}/{len(all_symbols)} partition={partition_id}/{partition_count} "
        f"since={since} until={until} hf_dir={hf_dir} "
        f"label_aware={bool(requirements)}"
    )

    def _missing_ranges(symbol: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        if args.force_start is not None:
            return [(since, until)]
        required_since, required_until = requirements.get(symbol, (since, until))
        required_since = required_since.floor("15min")
        required_until = required_until.ceil("15min")
        existing = _load_existing_data(symbol, allow_quote_fallback=False)
        if existing is None or existing.empty:
            return [(required_since, required_until)]
        ex_start = existing.index.min()
        ex_end = existing.index.max()
        if ex_start <= required_since and ex_end >= required_until:
            return []
        ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if required_since < ex_start:
            left_end = min(required_until, ex_start - pd.Timedelta(minutes=15))
            if required_since <= left_end:
                ranges.append((required_since, left_end))
        if required_until > ex_end:
            right_start = max(required_since, ex_end + pd.Timedelta(minutes=15))
            if right_start <= required_until:
                ranges.append((right_start, required_until))
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
                observed = part.copy()
                # This provenance is written before any local regularisation.
                # It distinguishes a genuine no-trade candle returned by
                # Kraken from a flat bar synthesized to bridge a timestamp
                # absent from the source response.
                observed["exchange_observed"] = True
                frames.append(observed)
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
            if (existing is not None and not existing.empty) or chunks:
                # Forced refreshes are atomic at the timestamp level.  New
                # exchange rows replace cache rows only where Kraken actually
                # returned a candle; an incomplete response cannot erase a
                # previously retained signal-hour bar.
                df = _merge_refresh_frames(existing, chunks)
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
