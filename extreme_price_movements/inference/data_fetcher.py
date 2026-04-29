"""
OHLCV Data Fetcher for Inference with Incremental Updates.

This module handles fetching OHLCV data for inference with:
- Incremental updates (only fetch missing data)
- 15m OHLCV fetching with immediate resampling to 1h
- Proper time indexation (floor to hour)
- Rate limiting between requests
"""

import random
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from extreme_price_movements import hf_data_loader
from extreme_price_movements.data_store import PartitionedOHLCVStore, make_spot_exchange
from extreme_price_movements.utils import tprint

# Default configuration
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LOOKBACK_HOURS = 24 * 60
MAX_RETRIES = 3
BACKOFF_BASE = 1.0
RATE_LIMIT_DELAY = 0.1  # seconds between requests


def classify_api_error(exc: Exception) -> str:
    """Classify exchange/API failures into stable operational buckets."""
    text = f"{exc.__class__.__name__} {exc}".lower()
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if "retry-after" in text:
        return "retry_after"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "network" in text or "connection" in text or "temporarily unavailable" in text:
        return "network"
    if "auth" in text or "permission" in text or "forbidden" in text or "401" in text:
        return "auth_or_permission"
    if "invalid symbol" in text or "bad symbol" in text or "unknown symbol" in text:
        return "invalid_symbol"
    return "api_error"


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Extract Retry-After from common ccxt/request exception shapes."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    value = None
    try:
        value = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        return None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _sleep_for_api_error(exc: Exception, *, attempt: int) -> None:
    """Respect Retry-After when present, otherwise apply small jittered backoff."""
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        tprint(f"API retry-after received; sleeping {retry_after:.2f}s")
        time.sleep(retry_after)
        return
    delay = min(8.0, BACKOFF_BASE * (2.0 ** max(attempt - 1, 0)))
    delay += random.uniform(0.0, 0.25)
    time.sleep(delay)


class DataFetcher:
    """Data fetcher with incremental updates for inference."""

    def __init__(self, exchange: Any = None, data_root: str = "data"):
        """Initialize the DataFetcher.

        Args:
            exchange: ccxt exchange instance (created if None)
            data_root: Root directory for data storage
        """
        self.exchange = exchange if exchange is not None else make_spot_exchange()
        self.data_root = data_root
        self.ohlcv_store = PartitionedOHLCVStore(data_root, timeframe="1h")
        self.api_error_counts: Dict[str, int] = {}
        self.dead_letter_symbols: Dict[str, str] = {}

    def _record_api_error(self, symbol: str, exc: Exception, *, context: str) -> None:
        category = classify_api_error(exc)
        self.api_error_counts[category] = self.api_error_counts.get(category, 0) + 1
        self.dead_letter_symbols[symbol] = f"{context}:{category}:{exc}"
        tprint(f"[DataFetcher] {context} failed for {symbol}: {category}: {exc}")

    def initialize_with_historical_data(
        self, symbols: List[str], lookback_hours: int = DEFAULT_LOOKBACK_HOURS
    ):
        """On startup: Use existing data + download missing.

        1. Check what data we already have
        2. Only fetch what's missing until current time
        3. Resample 15m -> 1h

        Args:
            symbols: List of trading symbols
            lookback_hours: Number of hours to look back if no data exists
        """
        # Get current time
        now = pd.Timestamp.now(tz="UTC")

        updated = 0
        skipped = 0
        failed = 0
        total = len(symbols)
        tprint(f"Initializing historical data batch: symbols={total}")

        for i, symbol in enumerate(symbols, start=1):
            if i == 1 or i == total or i % 25 == 0:
                tprint(f"Historical data init progress: {i}/{total}")

            # Check existing data range
            try:
                existing_data = self.ohlcv_store.load(symbol, start_ts=None, end_ts=now)
            except Exception as exc:
                failed += 1
                tprint(f"Error loading stored OHLCV for {symbol}: {exc}")
                tprint(traceback.format_exc())
                continue

            # Safely check existing data
            try:
                existing_not_empty = (
                    existing_data is not None
                    and isinstance(existing_data, (pd.DataFrame, pd.Series))
                    and not (hasattr(existing_data, "empty") and existing_data.empty)
                )
            except Exception:
                existing_not_empty = False

            if existing_not_empty:
                # Find gap from last timestamp to now
                last_ts = existing_data.index.max()
                if (now - last_ts) > pd.Timedelta(hours=1):
                    # Fetch missing data
                    missing_data = self.fetch_ohlcv(
                        symbol, start=last_ts + pd.Timedelta(hours=1), end=now
                    )
                    if (
                        missing_data is not None
                        and isinstance(missing_data, (pd.DataFrame, pd.Series))
                        and not (hasattr(missing_data, "empty") and missing_data.empty)
                    ):
                        # Resample and merge
                        existing_data = self._resample_and_merge(
                            existing_data, missing_data
                        )
                        self.ohlcv_store.save_partitioned(
                            symbol=symbol, df=existing_data
                        )
                        updated += 1
                else:
                    skipped += 1
            else:
                # No data - fetch from lookback
                start = now - pd.Timedelta(hours=lookback_hours)
                data = self.fetch_ohlcv(symbol, start=start, end=now)
                # Validate data is a proper DataFrame before saving
                if isinstance(data, pd.DataFrame) and not (
                    hasattr(data, "empty") and data.empty
                ):
                    self.ohlcv_store.save_partitioned(symbol=symbol, df=data)
                    updated += 1
                else:
                    failed += 1
                    tprint(f"Warning: No valid data returned for {symbol}; skipping")
        tprint(
            "Historical data init batch complete: "
            f"updated={updated} up_to_date={skipped} failed={failed}"
        )

    def fetch_ohlcv(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Fetch 15m OHLCV and resample to 1h.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            start: Start timestamp
            end: End timestamp

        Returns:
            DataFrame with 1h OHLCV data
        """
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        try:
            # Fetch 15m data from exchange
            ohlcv_15m = self._fetch_with_retry(
                symbol,
                timeframe="15m",
                since=int(start.timestamp() * 1000),
                limit=1200,  # Max for 15m
            )
        except Exception as exc:
            self._record_api_error(symbol, exc, context="fetch_ohlcv")
            return pd.DataFrame()

        # Validate response
        if ohlcv_15m is None:
            tprint(f"  Warning: None returned for {symbol}, returning empty DataFrame")
            return pd.DataFrame()
        if isinstance(ohlcv_15m, str):
            tprint(f"  Warning: Error string returned for {symbol}: {ohlcv_15m[:100]}")
            return pd.DataFrame()
        if not isinstance(ohlcv_15m, list):
            tprint(
                f"  Warning: Unexpected type {type(ohlcv_15m)} for {symbol}, returning empty DataFrame"
            )
            return pd.DataFrame()
        if len(ohlcv_15m) == 0:
            tprint(
                f"  Warning: Empty list returned for {symbol}, returning empty DataFrame"
            )
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv_15m, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Convert timestamp column to datetime with timezone
        timestamps = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = timestamps
        df.set_index("timestamp", inplace=True)

        # Filter to requested range
        df = df[(df.index >= start) & (df.index <= end)]

        # Safely check for empty df
        try:
            is_empty = (
                df is None
                or not isinstance(df, (pd.DataFrame, pd.Series))
                or (hasattr(df, "empty") and df.empty)
            )
        except Exception:
            is_empty = True

        if is_empty:
            return pd.DataFrame()

        # Resample to 1h
        df_1h = self._resample_to_hourly(df)

        return df_1h

    def _fetch_with_retry(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> List[List]:
        """Fetch OHLCV with retry logic and rate limiting.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for OHLCV
            since: Start time in milliseconds
            limit: Number of candles

        Returns:
            List of OHLCV candles
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if attempt > 1:
                    tprint(
                        f"[DataFetcher] retrying OHLCV {symbol} {timeframe}: "
                        f"attempt={attempt}/{MAX_RETRIES}"
                    )
                return self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                )
            except Exception as exc:
                last_exc = exc
                category = classify_api_error(exc)
                tprint(
                    f"[DataFetcher] OHLCV API error for {symbol} {timeframe}: "
                    f"category={category} attempt={attempt}/{MAX_RETRIES}: {exc}"
                )
                if attempt < MAX_RETRIES:
                    _sleep_for_api_error(exc, attempt=attempt)
        assert last_exc is not None
        raise last_exc

    def _resample_to_hourly(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling 1h aggregation over 15m data to produce overlapping 1h bars.

        This retains the 15m timestamps but computes OHLCV over the preceding 60 minutes.

        Args:
            df_15m: DataFrame with 15m OHLCV data

        Returns:
            DataFrame with 1h rolling OHLCV data on 15m timestamps
        """
        # Safely check df_15m
        try:
            is_empty = (
                df_15m is None
                or not isinstance(df_15m, (pd.DataFrame, pd.Series))
                or (hasattr(df_15m, "empty") and df_15m.empty)
            )
        except Exception:
            is_empty = True

        if is_empty:
            return df_15m.copy()

        # Ensure we have a clean copy sorted by index
        df_15m = df_15m.copy().sort_index()

        # We need a 60-minute rolling window, which is 4 bars for 15m data
        # We use a time-based rolling window to be robust against missing data
        # '1h' implies closing the window on the right and including the current row
        rolling = df_15m.rolling("1h")

        # Compute rolling OHLCV
        df_1h = pd.DataFrame(index=df_15m.index)
        df_1h["open"] = rolling["open"].apply(
            lambda x: x.iloc[0] if len(x) > 0 else np.nan, raw=False
        )
        df_1h["high"] = rolling["high"].max()
        df_1h["low"] = rolling["low"].min()
        df_1h["close"] = rolling["close"].apply(
            lambda x: x.iloc[-1] if len(x) > 0 else np.nan, raw=False
        )
        df_1h["volume"] = rolling["volume"].sum()

        # Drop rows with all NaN
        df_1h.dropna(how="all", inplace=True)

        return df_1h

    def _resample_and_merge(
        self, existing: pd.DataFrame, new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge existing data with new data after resampling.

        Args:
            existing: Existing DataFrame
            new_data: New DataFrame to merge

        Returns:
            Merged DataFrame
        """
        # Safely check new_data and existing
        try:
            new_not_empty = (
                new_data is not None
                and isinstance(new_data, (pd.DataFrame, pd.Series))
                and not (hasattr(new_data, "empty") and new_data.empty)
            )
            existing_not_empty = (
                existing is not None
                and isinstance(existing, (pd.DataFrame, pd.Series))
                and not (hasattr(existing, "empty") and existing.empty)
            )
        except Exception:
            new_not_empty = False
            existing_not_empty = False

        if not new_not_empty:
            return existing

        if not existing_not_empty:
            return new_data

        # Concatenate and remove duplicates
        merged = pd.concat([existing, new_data])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()

        return merged

    def fetch_incremental(self, symbol: str) -> pd.DataFrame:
        """At runtime: only fetch missing data since last fetch.

        Args:
            symbol: Trading symbol

        Returns:
            Updated DataFrame with all data (existing + new)
        """
        now = pd.Timestamp.now(tz="UTC")

        # Load existing data
        existing = self.ohlcv_store.load(symbol, start_ts=None, end_ts=now)

        # Safely check existing
        try:
            existing_not_empty = (
                existing is not None
                and isinstance(existing, (pd.DataFrame, pd.Series))
                and not (hasattr(existing, "empty") and existing.empty)
            )
        except Exception:
            existing_not_empty = False

        if existing_not_empty:
            last_ts = existing.index.max()

            # Check if we need to fetch new data (more than 15m gap)
            if (now - last_ts) > pd.Timedelta(minutes=15):
                # Fetch only from last timestamp + 15m buffer
                new_data = self.fetch_ohlcv(
                    symbol, start=last_ts + pd.Timedelta(minutes=15), end=now
                )
                if (
                    new_data is not None
                    and isinstance(new_data, (pd.DataFrame, pd.Series))
                    and not (hasattr(new_data, "empty") and new_data.empty)
                ):
                    # Merge and save
                    merged = self._resample_and_merge(existing, new_data)
                    # Safely check merged before saving
                    try:
                        merged_valid = (
                            merged is not None
                            and isinstance(merged, (pd.DataFrame, pd.Series))
                            and not (hasattr(merged, "empty") and merged.empty)
                        )
                    except Exception:
                        merged_valid = False

                    if merged_valid:
                        self.ohlcv_store.save_partitioned(merged, symbol)
                        tprint(
                            f"Incremental update for {symbol}: added {len(new_data)} new rows"
                        )
                        return merged
                    else:
                        tprint(
                            f"Warning: Invalid merged data for {symbol}, skipping save"
                        )
                        return existing
            return existing
        else:
            # No existing data - fetch full lookback
            return self.fetch_ohlcv(
                symbol, start=now - pd.Timedelta(hours=DEFAULT_LOOKBACK_HOURS), end=now
            )

    def fetch_latest_hourly_symbol(
        self,
        symbol: str,
        *,
        target_hour: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Fetch and persist the latest closed 1h candle for one symbol."""
        now = pd.Timestamp.now(tz="UTC")
        if target_hour is None:
            target_hour = now.floor("h") - pd.Timedelta(hours=1)
        target_hour = pd.Timestamp(target_hour)
        if target_hour.tzinfo is None:
            target_hour = target_hour.tz_localize("UTC")
        else:
            target_hour = target_hour.tz_convert("UTC")

        ohlcv = self._fetch_with_retry(
            symbol,
            timeframe="1h",
            since=int(target_hour.timestamp() * 1000),
            limit=1,
        )
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[df.index <= target_hour]
        if df.empty:
            return pd.DataFrame()
        df = df[~df.index.duplicated(keep="last")]
        self.ohlcv_store.save_partitioned(symbol=symbol, df=df, defer_compact=True)
        return df

    def fetch_hourly_universe_once(
        self,
        symbols: List[str],
        *,
        max_workers: int = 16,
        no_progress_timeout_seconds: float = 60.0,
        target_hour: Optional[pd.Timestamp] = None,
        check_recent_gaps_days: int = 7,
        backfill_fn: Optional[Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch one closed 1h candle for the full live universe.

        The batch uses bounded fanout and stops waiting when no symbol has
        produced new OHLCV for ``no_progress_timeout_seconds``.
        """
        workers = max(1, min(int(max_workers), 32))
        if target_hour is None:
            target_hour = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
        target_hour = pd.Timestamp(target_hour)
        if target_hour.tzinfo is None:
            target_hour = target_hour.tz_localize("UTC")
        else:
            target_hour = target_hour.tz_convert("UTC")

        out: Dict[str, pd.DataFrame] = {}
        failed = 0
        empty = 0
        canceled = 0
        gap_backfills = 0
        tprint(
            "Hourly OHLCV universe batch start: "
            f"symbols={len(symbols)} workers={workers} target_hour={target_hour}"
        )
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(
                self.fetch_latest_hourly_symbol, sym, target_hour=target_hour
            ): sym
            for sym in symbols
        }
        pending = set(futures.keys())
        last_data_time = time.monotonic()
        try:
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                if not done:
                    quiet_for = time.monotonic() - last_data_time
                    if quiet_for >= float(no_progress_timeout_seconds):
                        canceled = len(pending)
                        for fut in pending:
                            fut.cancel()
                        tprint(
                            "Hourly OHLCV batch no-progress timeout: "
                            f"quiet_for={quiet_for:.1f}s canceled={canceled}"
                        )
                        break
                    continue
                for fut in done:
                    sym = futures[fut]
                    try:
                        df = fut.result()
                    except Exception as exc:
                        failed += 1
                        self._record_api_error(sym, exc, context="hourly_ohlcv")
                        continue
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        out[sym] = df
                        last_data_time = time.monotonic()
                    else:
                        empty += 1
                    if check_recent_gaps_days > 0 and self.has_recent_gap(
                        sym, days=check_recent_gaps_days
                    ):
                        try:
                            self.trigger_gap_backfill(
                                sym,
                                days=check_recent_gaps_days,
                                backfill_fn=backfill_fn,
                            )
                            gap_backfills += 1
                        except Exception as exc:
                            failed += 1
                            self._record_api_error(sym, exc, context="gap_backfill")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        tprint(
            "Hourly OHLCV universe batch complete: "
            f"requested={len(symbols)} updated={len(out)} empty={empty} "
            f"failed={failed} canceled={canceled} gap_backfills={gap_backfills} "
            f"dead_letters={len(self.dead_letter_symbols)} "
            f"errors={self.api_error_counts}"
        )
        return out

    def has_recent_gap(self, symbol: str, days: int = 7) -> bool:
        """Return True if stored hourly OHLCV has gaps in the recent window."""
        end_ts = pd.Timestamp.now(tz="UTC")
        start_ts = end_ts - pd.Timedelta(days=int(days))
        df = self.ohlcv_store.load(symbol, start_ts=start_ts, end_ts=end_ts)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return True
        idx = pd.DatetimeIndex(df.index).sort_values().unique()
        if len(idx) <= 1:
            return True
        diffs = idx.to_series().diff().dropna()
        return bool((diffs > pd.Timedelta(hours=1, minutes=5)).any())

    def trigger_gap_backfill(
        self,
        symbol: str,
        *,
        days: int = 7,
        backfill_fn: Optional[Any] = None,
    ) -> Optional[pd.DataFrame]:
        """Backfill recent missing bars via existing downloader tooling."""
        end_ts = pd.Timestamp.now(tz="UTC")
        start_ts = end_ts - pd.Timedelta(days=int(days))
        fn = backfill_fn or hf_data_loader.sync_15m_ohlcv_range
        try:
            return fn(self.exchange, symbol, start_ts, end_ts, full_backfill=False)
        except TypeError:
            # Backward-compatible signatures in tests/mocks.
            return fn(self.exchange, symbol, start_ts, end_ts)

    def needs_incremental_update(self, symbol: str) -> bool:
        """Cheap probe using latest exchange kline (limit=1) vs local store tail."""
        try:
            latest = fetch_latest_ohlcv(self.exchange, symbol, timeframe="1h")
        except Exception:
            return True
        if latest is None or not isinstance(latest, pd.DataFrame) or latest.empty:
            return True
        local = self.ohlcv_store.load(symbol, start_ts=None, end_ts=None)
        if local is None or not isinstance(local, pd.DataFrame) or local.empty:
            return True
        remote_ts = pd.Timestamp(latest.index.max())
        local_ts = pd.Timestamp(local.index.max())
        return bool(remote_ts > local_ts)

    def fetch_incremental_universe(
        self,
        symbols: List[str],
        *,
        max_workers: int = 8,
        check_recent_gaps_days: int = 7,
        backfill_fn: Optional[Any] = None,
        use_lightweight_probe: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Incrementally update a symbol universe using bounded worker fanout."""
        workers = max(1, min(int(max_workers), 32))
        out: Dict[str, pd.DataFrame] = {}
        submitted = 0
        skipped_probe = 0
        failed = 0
        gap_backfills = 0
        tprint(
            f"Incremental universe batch start: symbols={len(symbols)} workers={workers} "
            f"lightweight_probe={use_lightweight_probe}"
        )
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for sym in symbols:
                if use_lightweight_probe and (not self.needs_incremental_update(sym)):
                    skipped_probe += 1
                    continue
                futures[ex.submit(self.fetch_incremental, sym)] = sym
                submitted += 1
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    out[sym] = fut.result()
                except Exception as exc:
                    failed += 1
                    self._record_api_error(sym, exc, context="fetch_incremental")
                    tprint(traceback.format_exc())
                    continue
                if check_recent_gaps_days > 0 and self.has_recent_gap(
                    sym, days=check_recent_gaps_days
                ):
                    tprint(f"Detected recent gap for {sym}; invoking backfill")
                    try:
                        self.trigger_gap_backfill(
                            sym,
                            days=check_recent_gaps_days,
                            backfill_fn=backfill_fn,
                        )
                        gap_backfills += 1
                    except Exception as exc:
                        failed += 1
                        self._record_api_error(sym, exc, context="gap_backfill")
        tprint(
            f"Incremental universe batch complete: requested={len(symbols)} "
            f"submitted={submitted} skipped_probe={skipped_probe} updated={len(out)} "
            f"failed={failed} gap_backfills={gap_backfills} workers={workers} "
            f"dead_letters={len(self.dead_letter_symbols)} errors={self.api_error_counts}"
        )
        return out

    def get_panel(
        self, symbols: List[str], lookback_hours: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Get OHLCV panel for given symbols.

        Args:
            symbols: List of trading symbols
            lookback_hours: Optional number of recent hours to load

        Returns:
            Panel dictionary with open, high, low, close, volume DataFrames
        """
        # Fetch OHLCV data for all symbols
        ohlcv_data = {}
        start_ts = None
        if lookback_hours is not None:
            start_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
                hours=int(lookback_hours)
            )
        for symbol in symbols:
            try:
                data = self.ohlcv_store.load(symbol, start_ts=start_ts, end_ts=None)
                # Safely check data
                try:
                    data_not_empty = (
                        data is not None
                        and isinstance(data, (pd.DataFrame, pd.Series))
                        and not (hasattr(data, "empty") and data.empty)
                    )
                except Exception:
                    data_not_empty = False

                if data_not_empty:
                    ohlcv_data[symbol] = data
            except Exception as e:
                tprint(f"Warning: Could not load data for {symbol}: {e}")

        # Convert to panel format
        return get_panel_from_dict(ohlcv_data)


# Backwards compatibility: Keep existing functions for non-class usage
def make_exchange() -> Any:
    """Create and return a Binance spot exchange instance.

    Returns:
        ccxt.binance exchange instance with rate limiting enabled
    """
    try:
        ex = make_spot_exchange()
        tprint("Created Binance spot exchange and loaded markets")
        return ex
    except Exception as exc:
        tprint(
            f"Failed to create Binance spot exchange: {classify_api_error(exc)}: {exc}"
        )
        raise


def fetch_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str = DEFAULT_TIMEFRAME,
    since: Optional[int] = None,
    limit: int = 100,
) -> List[List]:
    """Fetch OHLCV data for a single symbol using ccxt.

    Args:
        exchange: ccxt exchange instance
        symbol: Trading symbol (e.g., "BTC/USDT")
        timeframe: Timeframe (e.g., "1h", "4h", "1d")
        since: Start time in milliseconds (optional)
        limit: Number of candles to fetch

    Returns:
        List of OHLCV candles [timestamp, open, high, low, close, volume]
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            return exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )
        except Exception as exc:
            last_exc = exc
            category = classify_api_error(exc)
            tprint(
                f"Error fetching OHLCV for {symbol}: category={category} "
                f"attempt={attempt}/{MAX_RETRIES}: {exc}"
            )
            if attempt < MAX_RETRIES:
                _sleep_for_api_error(exc, attempt=attempt)
    assert last_exc is not None
    raise last_exc


def convert_ohlcv_to_dataframe(ohlcv: List[List], symbol: str) -> pd.DataFrame:
    """Convert ccxt OHLCV format to pandas DataFrame.

    Args:
        ohlcv: List of OHLCV candles
        symbol: Symbol for the data

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if not ohlcv:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["symbol"] = symbol

    return df


def fetch_ohlcv_for_symbols(
    symbols: List[str],
    exchange: Optional[Any] = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    lookback_periods: int = 48,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple symbols.

    Args:
        symbols: List of trading symbols
        exchange: ccxt exchange instance (created if None)
        timeframe: Timeframe for OHLCV data
        lookback_periods: Number of periods to look back

    Returns:
        Dictionary mapping symbol to OHLCV DataFrame
    """
    if exchange is None:
        exchange = make_exchange()

    # Calculate start time based on lookback periods
    # Add some buffer to account for weekends/gaps
    periods_per_day = 24 if timeframe == "1h" else (24 // 4 if timeframe == "4h" else 1)
    limit = lookback_periods + 24  # Add buffer

    results = {}
    failed = 0
    tprint(
        f"Fetching OHLCV batch: symbols={len(symbols)} timeframe={timeframe} "
        f"lookback_periods={lookback_periods}"
    )

    for symbol in symbols:
        try:
            ohlcv = fetch_ohlcv(exchange, symbol, timeframe, limit=limit)
            if ohlcv:
                df = convert_ohlcv_to_dataframe(ohlcv, symbol)
                results[symbol] = df
            else:
                tprint(f"No data for {symbol}")
        except Exception as e:
            failed += 1
            tprint(f"Failed to fetch {symbol}: {classify_api_error(e)}: {e}")
            continue

    tprint(
        f"OHLCV batch complete: requested={len(symbols)} fetched={len(results)} "
        f"failed={failed}"
    )
    return results


def fetch_latest_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> Optional[pd.DataFrame]:
    """Fetch the latest OHLCV candle for a symbol.

    Args:
        exchange: ccxt exchange instance
        symbol: Trading symbol
        timeframe: Timeframe

    Returns:
        DataFrame with latest candle or None if failed
    """
    try:
        ohlcv = fetch_ohlcv(exchange, symbol, timeframe, limit=1)
        if ohlcv:
            return convert_ohlcv_to_dataframe(ohlcv, symbol)
        return None
    except Exception as e:
        tprint(
            f"Error fetching latest OHLCV for {symbol}: "
            f"{classify_api_error(e)}: {e}"
        )
        return None


def get_panel_from_dict(
    ohlcv_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Convert dictionary of OHLCV DataFrames to panel format.

    Creates a dictionary of DataFrames for each price type (open, high, low, close, volume)
    indexed by datetime and columns by symbol.

    Args:
        ohlcv_data: Dictionary mapping symbol to OHLCV DataFrame

    Returns:
        Dictionary with keys: open, high, low, close, volume
    """
    panel = {
        "open": pd.DataFrame(),
        "high": pd.DataFrame(),
        "low": pd.DataFrame(),
        "close": pd.DataFrame(),
        "volume": pd.DataFrame(),
    }

    # Find common index (union of all datetimes)
    all_indexes = []
    for df in ohlcv_data.values():
        # Safely check df
        try:
            df_not_empty = (
                df is not None
                and isinstance(df, (pd.DataFrame, pd.Series))
                and not (hasattr(df, "empty") and df.empty)
            )
        except Exception:
            df_not_empty = False

        if df_not_empty:
            all_indexes.append(df.index)

    if not all_indexes:
        return panel

    # Union of all timestamps
    common_index = all_indexes[0]
    for idx in all_indexes[1:]:
        common_index = common_index.union(idx)

    common_index = sorted(common_index)

    # Build panel
    for symbol, df in ohlcv_data.items():
        # Safely check df
        try:
            df_not_empty = (
                df is not None
                and isinstance(df, (pd.DataFrame, pd.Series))
                and not (hasattr(df, "empty") and df.empty)
            )
        except Exception:
            df_not_empty = False

        if not df_not_empty:
            continue

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                series = df[col].rename(symbol)
                panel[col] = panel[col].join(series, how="outer")

    # Reindex to common_index
    for col in panel:
        panel[col] = panel[col].reindex(common_index)
        panel[col] = panel[col].sort_index()

    return panel


def fetch_and_build_panel(
    symbols: List[str],
    exchange: Optional[Any] = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    lookback_periods: int = 48,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data and build panel format.

    Args:
        symbols: List of trading symbols
        exchange: ccxt exchange instance
        timeframe: Timeframe for OHLCV
        lookback_periods: Number of periods to look back

    Returns:
        Panel dictionary with open, high, low, close, volume DataFrames
    """
    ohlcv_data = fetch_ohlcv_for_symbols(
        symbols=symbols,
        exchange=exchange,
        timeframe=timeframe,
        lookback_periods=lookback_periods,
    )

    return get_panel_from_dict(ohlcv_data)
