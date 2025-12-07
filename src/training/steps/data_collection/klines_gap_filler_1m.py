"""Ad-hoc 1m synthetic gap filler for klines data.

This module provides a small, focused utility that can:
- Inspect existing 1m raw klines for an exchange/symbol stored via KlinesParquetManager
- Detect remaining timestamp gaps on the 1m grid
- Fill small gaps synthetically (forward/backward flat carry with zero volume)
- Persist the updated 1m raw data back to parquet
- Rebuild the corresponding resampled higher-timeframe data (5m/15m/30m/1h by default)

It is designed to be used in two ways:
- Standalone CLI script: python -m .../klines_gap_filler_1m --exchange binance --asset ETH
- Programmatically from the enhanced_klines_processing_pipeline for "unfillable" gaps
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.data.klines_parquet import KlinesParquetManager


def _get_logger_functions():  # pragma: no cover - thin wrapper
    """Lazy import of tprint helpers with simple fallbacks.

    This mirrors the pattern used in enhanced_klines_processing_pipeline without
    introducing a hard dependency during import.
    """

    try:
        from src.utils.tprint import (
            tprint,
            tprint_info,
            tprint_warning,
            tprint_error,
            tprint_success,
        )

        return tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    except Exception:  # pragma: no cover - defensive fallback
        def _print(*args, **kwargs):  # type: ignore[override]
            print(*args, **kwargs)

        return _print, _print, _print, _print, _print


# Local logger-style functions
_tprint, _tprint_info, _tprint_warning, _tprint_error, _tprint_success = _get_logger_functions()


@dataclass
class SyntheticGapFillStats:
    """Summary of a synthetic 1m gap-filling run."""

    exchange: str
    symbol: str
    interval: str
    rows_before: int
    rows_after: int
    synthetic_rows_added: int
    gap_segments_considered: int
    gap_segments_filled: int
    max_gap_bars: int
    resampled_intervals_updated: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "interval": self.interval,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "synthetic_rows_added": self.synthetic_rows_added,
            "gap_segments_considered": self.gap_segments_considered,
            "gap_segments_filled": self.gap_segments_filled,
            "max_gap_bars": self.max_gap_bars,
            "resampled_intervals_updated": list(self.resampled_intervals_updated),
        }


def _detect_and_fill_small_1m_gaps(
    df: pd.DataFrame,
    max_gap_bars: int,
) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """Detect and synthetically fill small 1m gaps in a 1m OHLCV frame.

    The algorithm:
    - Reindexes the frame to a complete 1m grid from min→max timestamp
    - Identifies contiguous runs of missing "close" values
    - For gaps with length <= max_gap_bars, inserts synthetic bars where:
      * open/high/low/close are flat at the previous real close (or next if no prev)
      * volume/quote_volume/trade counts are set to zero
      * symbol/interval/exchange columns are preserved
      * a boolean flag "is_synthetic_gap_fill" is set to True
    - Larger gaps are left untouched (their timestamps are removed again)

    Returns the filled DataFrame (with any large-gap NaNs dropped) and the list
    of filled (start_ts, end_ts) intervals.
    """

    if df is None or df.empty:
        return df, []

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.loc[df.index.notna()]

    if df.empty:
        return df, []

    df = df.sort_index()

    # Build a complete 1m grid over the observed range
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1T")
    df_full = df.reindex(full_index)

    if "is_synthetic_gap_fill" not in df_full.columns:
        df_full["is_synthetic_gap_fill"] = False

    # Use close as the primary completeness signal
    if "close" not in df_full.columns:
        return df, []

    is_missing = df_full["close"].isna()
    if not is_missing.any():
        # Nothing to fill; drop any all-NaN rows that might have been introduced
        df_clean = df_full.dropna(subset=["open", "high", "low", "close"], how="all")
        return df_clean, []

    gap_group = (is_missing != is_missing.shift()).cumsum()
    filled_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    # Convenience: columns we treat specially
    ohlc_cols = {"open", "high", "low", "close"}

    for _, gap_block in df_full[is_missing].groupby(gap_group[is_missing]):
        start_ts = gap_block.index[0]
        end_ts = gap_block.index[-1]
        length = len(gap_block)

        # Only synthetically fill gaps up to the configured threshold
        if length <= 0 or length > max_gap_bars:
            continue

        try:
            pos_start = full_index.get_loc(start_ts)
        except KeyError:
            continue

        pos_end = full_index.get_loc(end_ts)

        prev_ts: Optional[pd.Timestamp] = full_index[pos_start - 1] if pos_start > 0 else None
        next_ts: Optional[pd.Timestamp] = (
            full_index[pos_end + 1] if pos_end + 1 < len(full_index) else None
        )

        base_row = None
        if prev_ts is not None and not pd.isna(df_full.at[prev_ts, "close"]):
            base_row = df_full.loc[prev_ts]
        elif next_ts is not None and not pd.isna(df_full.at[next_ts, "close"]):
            base_row = df_full.loc[next_ts]

        if base_row is None:
            # No reliable neighbour; skip this gap
            continue

        try:
            base_price = float(base_row["close"])  # type: ignore[arg-type]
        except Exception:
            # If we cannot coerce, skip this gap entirely
            continue

        for ts in gap_block.index:
            for col in df_full.columns:
                if col in ohlc_cols:
                    df_full.at[ts, col] = base_price
                elif "volume" in col or "trade" in col:
                    # Any volume-like or trade count columns are zeroed for synthetic bars
                    df_full.at[ts, col] = 0.0
                elif col == "is_synthetic_gap_fill":
                    df_full.at[ts, col] = True
                elif col in ("symbol", "interval", "exchange"):
                    # Preserve metadata from base_row if possible
                    df_full.at[ts, col] = base_row.get(col, df_full.at[ts, col])
                else:
                    # For other columns, best-effort copy from neighbour without
                    # attempting to be clever about semantics.
                    df_full.at[ts, col] = base_row.get(col, df_full.at[ts, col])

        filled_ranges.append((start_ts, end_ts))

    # Drop any timestamps that still have missing OHLC after the synthetic fill
    df_filled = df_full.dropna(subset=["open", "high", "low", "close"], how="any")

    return df_filled, filled_ranges


def _resample_from_1m(
    df_1m: pd.DataFrame,
    exchange: str,
    symbol: str,
    target_intervals: Sequence[str],
) -> Dict[str, int]:
    """Resample a 1m OHLCV frame to higher intervals and return record counts.

    The aggregation mirrors the logic used in EnhancedKlinesProcessingPipeline._perform_resampling.
    """

    if df_1m is None or df_1m.empty:
        return {}

    if not isinstance(df_1m.index, pd.DatetimeIndex):
        df_1m = df_1m.copy()
        df_1m.index = pd.to_datetime(df_1m.index, errors="coerce")
        df_1m = df_1m.loc[df_1m.index.notna()]

    if df_1m.empty:
        return {}

    df_1m = df_1m.sort_index()

    if "symbol" not in df_1m.columns:
        df_1m = df_1m.copy()
        df_1m["symbol"] = symbol
    if "exchange" not in df_1m.columns:
        df_1m = df_1m.copy()
        df_1m["exchange"] = exchange

    def _interval_to_freq(interval: str) -> Optional[str]:
        mapping = {
            "1m": "1T",
            "3m": "3T",
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6H",
            "8h": "8H",
            "12h": "12H",
            "1d": "1D",
            "3d": "3D",
            "1w": "1W",
            "1M": "1M",
        }
        return mapping.get(interval)

    counts: Dict[str, int] = {}

    for interval in target_intervals:
        if interval == "1m":
            # No resample needed for the base interval
            continue

        freq = _interval_to_freq(interval)
        if freq is None:
            _tprint_warning(f"⚠️ Unsupported resample interval: {interval}")
            continue

        try:
            resampled = df_1m.resample(freq).agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            ).dropna()
        except Exception as e:  # pragma: no cover - defensive
            _tprint_warning(f"⚠️ Resample to {interval} failed: {e}")
            continue

        if resampled.empty:
            continue

        resampled["symbol"] = symbol
        resampled["interval"] = interval
        resampled["exchange"] = exchange

        manager = KlinesParquetManager(data_dir="historical_data", exchange=exchange)
        success = manager.write_data(resampled, symbol, interval, data_type="processed", overwrite=True)
        if success:
            counts[interval] = len(resampled)
            _tprint_success(
                f"✅ Resampled and stored {symbol} {interval}: {len(resampled):,} records"
            )
        else:
            _tprint_warning(f"⚠️ Failed to store resampled data for {symbol} {interval}")

    return counts


def fill_1m_gaps_and_resample_for_symbol(
    *,
    exchange: str,
    symbol: str,
    data_dir: str = "historical_data",
    target_intervals: Optional[Sequence[str]] = None,
    max_gap_bars: int = 30,
    dry_run: bool = False,
) -> Dict[str, object]:
    """High-level entrypoint: fill 1m gaps and refresh resampled data.

    This function is safe to call from both the enhanced klines pipeline and
    from the CLI. When ``dry_run`` is True, it only reports what it *would*
    change, without writing anything.
    """

    interval = "1m"
    manager = KlinesParquetManager(data_dir=data_dir, exchange=exchange)

    _tprint_info(f"🔍 Loading existing {exchange.upper()} {symbol} {interval} raw data…")
    df_1m = manager.read_data(symbol, interval, data_type="raw")

    if df_1m is None or df_1m.empty:
        _tprint_warning(f"⚠️ No existing raw data found for {exchange.upper()} {symbol} {interval}")
        return {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "rows_before": 0,
            "rows_after": 0,
            "synthetic_rows_added": 0,
            "gap_segments_considered": 0,
            "gap_segments_filled": 0,
            "max_gap_bars": max_gap_bars,
            "resampled_intervals_updated": [],
            "dry_run": dry_run,
        }

    rows_before = len(df_1m)

    _tprint_info(
        f"📊 Loaded {rows_before:,} rows of {exchange.upper()} {symbol} {interval} data "
        f"from {df_1m.index.min()} to {df_1m.index.max()}"
    )

    filled_df, filled_ranges = _detect_and_fill_small_1m_gaps(df_1m, max_gap_bars=max_gap_bars)

    gap_segments_considered = len(filled_ranges)
    rows_after = len(filled_df)
    synthetic_rows_added = max(rows_after - rows_before, 0)

    _tprint_info(
        f"📈 Synthetic gap fill summary for {symbol} {interval}: "
        f"{gap_segments_considered} small gaps filled, +{synthetic_rows_added:,} rows"
    )

    resampled_counts: Dict[str, int] = {}

    if not dry_run:
        # Persist updated 1m raw data by fully rewriting the monthly shards
        _tprint_info(
            f"💾 Writing updated {exchange.upper()} {symbol} {interval} raw data "
            f"back to parquet (overwrite=True)…"
        )
        write_ok = manager.write_data(
            filled_df,
            symbol,
            interval,
            data_type="raw",
            overwrite=True,
        )
        if not write_ok:
            _tprint_warning(
                f"⚠️ Failed to persist updated raw {symbol} {interval} data; "
                f"resampling will still proceed in-memory"
            )

        # Rebuild requested resampled intervals from the new 1m base
        effective_targets: Sequence[str]
        if target_intervals is None:
            effective_targets = ("5m", "15m", "30m", "1h")
        else:
            effective_targets = tuple(target_intervals)

        _tprint_info(
            f"🔄 Rebuilding resampled data for intervals: {', '.join(effective_targets)}"
        )
        resampled_counts = _resample_from_1m(
            filled_df,
            exchange=exchange,
            symbol=symbol,
            target_intervals=effective_targets,
        )

    stats = SyntheticGapFillStats(
        exchange=exchange,
        symbol=symbol,
        interval=interval,
        rows_before=rows_before,
        rows_after=rows_after,
        synthetic_rows_added=synthetic_rows_added,
        gap_segments_considered=gap_segments_considered,
        gap_segments_filled=gap_segments_considered,
        max_gap_bars=max_gap_bars,
        resampled_intervals_updated=sorted(list(resampled_counts.keys())),
    )

    result = stats.to_dict()
    result["dry_run"] = dry_run
    result["resampled_counts"] = resampled_counts

    return result


def _parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ad-hoc 1m synthetic gap filler. Loads existing raw 1m data for an "
            "exchange/symbol, fills small gaps synthetically, rewrites the raw "
            "1m parquet shards, and rebuilds higher-timeframe resampled data."
        )
    )

    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange name (e.g. binance, bingx, okx)",
    )
    parser.add_argument(
        "--asset",
        type=str,
        required=False,
        help=(
            "Base asset (e.g. ETH). If --symbol is not provided, the symbol "
            "is derived as <asset>USDT."
        ),
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help=(
            "Full trading symbol (e.g. ETHUSDT). If omitted, it is derived "
            "from --asset as <asset>USDT."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="historical_data",
        help="Base data directory used by KlinesParquetManager",
    )
    parser.add_argument(
        "--target-intervals",
        type=str,
        default="5m,15m,30m,1h",
        help=(
            "Comma-separated list of higher intervals to rebuild from 1m "
            "(e.g. '5m,15m,1h'). The base 1m data is always updated."
        ),
    )
    parser.add_argument(
        "--max-gap-bars",
        type=int,
        default=30,
        help=(
            "Maximum length (in 1m bars) of a gap segment to fill synthetically. "
            "Larger gaps are left untouched."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report, but do not write any parquet files.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - CLI glue
    args = _parse_cli_args(argv)

    symbol: Optional[str] = args.symbol
    if not symbol:
        if not args.asset:
            _tprint_error("❌ Either --symbol or --asset must be provided")
            return
        symbol = f"{args.asset.upper()}USDT"

    target_intervals: Optional[List[str]] = None
    if args.target_intervals:
        target_intervals = [
            item.strip()
            for item in str(args.target_intervals).split(",")
            if item.strip()
        ]

    _tprint("=" * 80)
    _tprint("🚀 1M SYNTHETIC GAP FILLER")
    _tprint("=" * 80)
    _tprint_info(f"Exchange: {args.exchange}")
    _tprint_info(f"Symbol:   {symbol}")
    _tprint_info(f"Data dir: {args.data_dir}")
    _tprint_info(f"Target intervals: {', '.join(target_intervals or [])}")
    _tprint_info(f"Max gap bars: {args.max_gap_bars}")
    _tprint_info(f"Dry run: {args.dry_run}")

    stats = fill_1m_gaps_and_resample_for_symbol(
        exchange=args.exchange,
        symbol=symbol,
        data_dir=args.data_dir,
        target_intervals=target_intervals,
        max_gap_bars=args.max_gap_bars,
        dry_run=bool(args.dry_run),
    )

    _tprint("""
📊 Synthetic gap fill completed
-------------------------------
""")
    for key, value in stats.items():
        _tprint_info(f"{key}: {value}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
