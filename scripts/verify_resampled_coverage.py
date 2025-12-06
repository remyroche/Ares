"""
Verify that resampled klines (5m, 15m, 1h) exist and cover the full span of 1m raw klines.

Checks each symbol under historical_data/{exchange}/{symbol}/raw for 1m files and confirms:
1) processed/{symbol}_{interval} exists for target intervals.
2) processed span (min/max timestamp) fully covers the raw 1m span.

Exit code is non-zero if any interval is missing or has a span mismatch.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Ensure project root on path
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow.dataset as ds

from src.utils.data.klines_parquet import KlinesParquetManager

# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

TimestampSpan = Tuple[pd.Timestamp, pd.Timestamp]


def _first_available_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the first column name present in columns from candidates."""
    for cand in candidates:
        if cand in columns:
            return cand
    return None


def _dataset_span(paths: Iterable[Path], candidate_cols: Sequence[str]) -> Optional[TimestampSpan]:
    """Compute min/max timestamp across parquet files using the first available candidate column."""
    paths = list(paths)
    if not paths:
        return None

    dataset = ds.dataset(paths, format="parquet")
    col_name = _first_available_column(dataset.schema.names, candidate_cols)
    if not col_name:
        return None

    table = ds.Scanner.from_dataset(dataset, columns=[col_name]).to_table()
    if table.num_rows == 0:
        return None

    arr = table.column(0).combine_chunks().to_pandas()
    ts = pd.to_datetime(arr, errors="coerce", utc=True).dropna()
    if ts.empty:
        return None

    # Normalize to UTC naive if tz-aware; otherwise leave as is
    if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.min(), ts.max()


def _span_str(span: Optional[TimestampSpan]) -> str:
    if span is None:
        return "n/a"
    return f"{span[0]} → {span[1]}"


# --------------------------------------------------------------------------- #
# Verification logic
# --------------------------------------------------------------------------- #

@dataclass
class IntervalStatus:
    interval: str
    exists: bool
    span: Optional[TimestampSpan]
    covers_raw: bool
    reason: Optional[str] = None


@dataclass
class SymbolReport:
    symbol: str
    raw_span: Optional[TimestampSpan]
    intervals: List[IntervalStatus]
    regenerated: List[str]

    @property
    def ok(self) -> bool:
        return all(s.exists and s.covers_raw for s in self.intervals)


def verify_symbol(
    base_dir: Path,
    exchange: str,
    symbol: str,
    base_interval: str,
    target_intervals: Sequence[str],
    manager: KlinesParquetManager,
) -> SymbolReport:
    raw_dir = base_dir / exchange / symbol.lower() / "raw"
    processed_dir = base_dir / exchange / symbol.lower() / "processed"

    raw_files = list(raw_dir.glob(f"{symbol.lower()}_{base_interval}_*.parquet"))
    raw_span = _dataset_span(raw_files, ["timestamp", "open_time", "__index_level_0__", "index"])

    regenerated: List[str] = []
    intervals: List[IntervalStatus] = []
    for target in target_intervals:
        target_path = processed_dir / f"{symbol.lower()}_{target}"
        if not target_path.exists():
            # Attempt to resample from raw
            if _resample_from_raw(raw_files, symbol, target, base_interval, manager):
                regenerated.append(target)
            else:
                intervals.append(
                    IntervalStatus(
                        interval=target,
                        exists=False,
                        span=None,
                        covers_raw=False,
                        reason="missing processed directory",
                    )
                )
                continue

        processed_span = _dataset_span(target_path.glob("**/*.parquet"), ["timestamp", "open_time", "__index_level_0__", "index"])
        covers_raw = False
        reason = None
        if raw_span and processed_span:
            # Allow end tolerance up to one target interval
            interval_minutes = _interval_to_minutes(target)
            tolerance = pd.Timedelta(minutes=max(interval_minutes - 1, 0)) if interval_minutes else pd.Timedelta(0)
            covers_raw = processed_span[0] <= raw_span[0] and processed_span[1] + tolerance >= raw_span[1]
            if not covers_raw:
                reason = (
                    f"span mismatch raw {raw_span[0]}→{raw_span[1]} "
                    f"vs {processed_span[0]}→{processed_span[1]} (tol {tolerance})"
                )
        elif raw_span and not processed_span:
            reason = "cannot read processed span"

        intervals.append(
            IntervalStatus(
                interval=target,
                exists=True,
                span=processed_span,
                covers_raw=covers_raw,
                reason=reason,
            )
        )

    return SymbolReport(symbol=symbol, raw_span=raw_span, intervals=intervals, regenerated=regenerated)


def _load_raw_df(raw_files: List[Path], candidate_cols: Sequence[str]) -> Optional[pd.DataFrame]:
    if not raw_files:
        return None
    dataset = ds.dataset(raw_files, format="parquet")
    col_name = _first_available_column(dataset.schema.names, candidate_cols)
    if not col_name:
        return None
    needed_cols = list({col_name} | {"open", "high", "low", "close", "volume"})
    table = ds.Scanner.from_dataset(dataset, columns=needed_cols).to_table()
    if table.num_rows == 0:
        return None
    df = table.to_pandas()
    series = df[col_name] if col_name in df.columns else None
    if series is None and isinstance(df.index, pd.DatetimeIndex):
        series = df.index
    if series is None:
        return None

    ts_raw = pd.to_datetime(series, errors="coerce", utc=True)
    mask = pd.Series(ts_raw.notna(), index=df.index)
    if not mask.any():
        return None

    df = df.loc[mask].copy()
    ts_values = ts_raw[mask.values]
    df.index = ts_values.tz_convert("UTC").tz_localize(None)
    # Keep all rows (do not drop OHLCV rows to avoid truncation); just dedupe and sort
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Remove true duplicate rows (same index and same OHLCV)
    df = df[~df.reset_index().duplicated(subset=["index", "open", "high", "low", "close", "volume"])].set_index(df.index)
    return df


def _resample_from_raw(
    raw_files: List[Path],
    symbol: str,
    target_interval: str,
    base_interval: str,
    manager: KlinesParquetManager,
) -> bool:
    """Resample raw 1m klines to target interval and write to processed/."""
    freq_map = {"1m": "1T", "3m": "3T", "5m": "5T", "15m": "15T", "30m": "30T", "1h": "1H"}
    if target_interval not in freq_map or base_interval not in freq_map:
        return False

    df = _load_raw_df(raw_files, ["timestamp", "open_time", "__index_level_0__", "index"])
    if df is None or df.empty:
        return False

    freq = freq_map[target_interval]
    resampled = df.resample(freq, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna(how="all")

    if resampled.empty:
        return False

    # Drop true duplicates on index and OHLCV
    resampled = resampled[~resampled.index.duplicated(keep="last")]
    resampled = resampled[~resampled.reset_index().duplicated(subset=["index", "open", "high", "low", "close", "volume"])].set_index(resampled.index)

    resampled["symbol"] = symbol.upper()
    resampled["interval"] = target_interval
    resampled["exchange"] = manager.exchange

    return manager.write_data(resampled, symbol=symbol, interval=target_interval, data_type="processed", overwrite=True)


def _interval_to_minutes(interval: str) -> Optional[int]:
    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
    }
    return mapping.get(interval)


def collect_symbols(base_dir: Path, exchange: str) -> List[str]:
    exchange_dir = base_dir / exchange
    if not exchange_dir.exists():
        return []
    return sorted(
        [
            p.name
            for p in exchange_dir.iterdir()
            if p.is_dir() and (p / "raw").exists()
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that resampled klines exist and cover raw 1m data."
    )
    parser.add_argument("--data-dir", default="historical_data", help="Base data directory")
    parser.add_argument("--exchange", default="binance", help="Exchange name (e.g., binance)")
    parser.add_argument("--base-interval", default="1m", help="Base interval to verify (default: 1m)")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["5m", "15m", "1h"],
        help="Target resampled intervals to verify",
    )
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional symbol list; defaults to all under exchange")

    args = parser.parse_args()
    base_dir = Path(args.data_dir)
    exchange = args.exchange.lower()
    manager = KlinesParquetManager(data_dir=args.data_dir, exchange=exchange)

    symbols = args.symbols or collect_symbols(base_dir, exchange)
    if not symbols:
        print("No symbols found to verify.")
        return 0

    reports: List[SymbolReport] = []
    for symbol in symbols:
        reports.append(verify_symbol(base_dir, exchange, symbol, args.base_interval, args.targets, manager))

    missing = 0
    span_mismatches = 0

    print(f"\nVerification for exchange={exchange} base_interval={args.base_interval}")
    for rep in reports:
        print(f"\nSymbol: {rep.symbol}")
        print(f"  Raw span: {_span_str(rep.raw_span)}")
        for st in rep.intervals:
            status = "OK" if (st.exists and st.covers_raw) else "FAIL"
            span_str = _span_str(st.span)
            reason = f" ({st.reason})" if st.reason else ""
            print(f"  {st.interval:>4}: {status} | span={span_str}{reason}")
            if not st.exists:
                missing += 1
            elif not st.covers_raw:
                span_mismatches += 1

    if missing or span_mismatches:
        print(f"\nResult: FAIL — missing={missing}, span_mismatches={span_mismatches}")
        return 1

    print("\nResult: OK — all target intervals present and covering raw span.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
