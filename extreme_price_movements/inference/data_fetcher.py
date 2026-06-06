"""
OHLCV Data Fetcher for Inference with Incremental Updates.

This module handles fetching OHLCV data for inference with:
- Incremental updates (only fetch missing data)
- 15m OHLCV fetching with immediate resampling to 1h
- Proper time indexation (floor to hour)
- Rate limiting between requests
"""

import os
import random
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from extreme_price_movements import hf_data_loader
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    _compute_missing_funding_ranges,
    _compute_missing_hourly_ranges,
    _fetch_ccxt_history_paged,
    _resolve_perp_symbol,
    build_hourly_orderbook_proxy_from_ohlcv,
    exchange_data_component,
    fetch_hourly_orderbook_proxy,
    make_perp_exchange,
    make_ohlcv_store,
    make_spot_exchange,
    normalize_orderbook_proxy_frame,
    scoped_data_root,
)
from extreme_price_movements.utils import tprint

# Default configuration
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LOOKBACK_HOURS = 24 * 60
MAX_RETRIES = 3
BACKOFF_BASE = 1.0
RATE_LIMIT_DELAY = 0.1  # seconds between requests

MICRODATA_FRAME_FIELDS = (
    "funding_rate",
    "open_interest",
    "mark_price",
    "index_price",
    "premium_index",
)

PERP_OHLCV_EXTRA_FIELDS = (
    "funding_rate",
    "open_interest",
    "spot_open",
    "spot_high",
    "spot_low",
    "spot_close",
    "spot_volume",
    "mark_open",
    "mark_high",
    "mark_low",
    "mark_close",
    "mark_price",
    "index_open",
    "index_high",
    "index_low",
    "index_close",
    "index_price",
    "premium_index_open",
    "premium_index_high",
    "premium_index_low",
    "premium_index_close",
    "premium_index",
)


def _read_int_env(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw not in (None, "") else int(default)
    except (TypeError, ValueError):
        value = int(default)
    value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return value


def _normalise_market_mode(market_mode: Optional[str] = None) -> str:
    raw = str(market_mode or "spot").strip().lower()
    return "perps" if raw in {"perp", "perps", "future", "futures", "swap"} else "spot"


def _is_exchange_scoped_data_root(
    data_root: str | os.PathLike[str],
    *,
    exchange_id: Optional[str],
    market_mode: str,
) -> bool:
    path = Path(data_root)
    parts = path.parts
    if len(parts) < 2 or parts[-2] != "exchanges":
        return False
    component = exchange_data_component(exchange_id, market_mode)
    return parts[-1] == component


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

    def __init__(
        self, exchange: Any = None, data_root: str = "data", market_mode: str = "spot"
    ):
        """Initialize the DataFetcher.

        Args:
            exchange: ccxt exchange instance (created if None)
            data_root: Root directory for data storage
        """
        self.market_mode = _normalise_market_mode(market_mode)
        if exchange is not None:
            self.exchange = exchange
        elif self.market_mode == "perps":
            self.exchange = make_perp_exchange()
        else:
            self.exchange = make_spot_exchange()
        self.data_root = data_root
        exchange_id = str(getattr(self.exchange, "id", "") or "").strip() or None
        cfg = {
            "data_root": data_root,
            "exchange_id": exchange_id,
            "market_mode": self.market_mode,
        }
        if _is_exchange_scoped_data_root(
            data_root,
            exchange_id=exchange_id,
            market_mode=self.market_mode,
        ):
            market_data_root = Path(data_root)
            self.ohlcv_store = PartitionedOHLCVStore(
                root_dir=str(market_data_root),
                timeframe="1h",
            )
        else:
            self.ohlcv_store = make_ohlcv_store(cfg, timeframe="1h")
            market_data_root = Path(scoped_data_root(cfg))
        self.orderbook_dir = market_data_root / "orderbook_hourly"
        self.funding_dir = market_data_root / "funding_hourly"
        self.open_interest_dir = market_data_root / "open_interest_hourly"
        self.orderbook_dir.mkdir(parents=True, exist_ok=True)
        self.funding_dir.mkdir(parents=True, exist_ok=True)
        self.open_interest_dir.mkdir(parents=True, exist_ok=True)
        self.api_error_counts: Dict[str, int] = {}
        self.dead_letter_symbols: Dict[str, str] = {}
        self._perp_exchange: Optional[Any] = None
        self._symbols_without_perp_funding: set[str] = set()
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}
        self._microdata_symbol_cache: Dict[
            str, tuple[Optional[float], Optional[float], Optional[float], Dict[str, pd.Series]]
        ] = {}
        self._ticker_snapshot_cache: tuple[float, Dict[str, Dict[str, Any]]] | None = None
        self._ticker_snapshot_lock = threading.Lock()

    def _exchange_symbol(self, symbol: str) -> str:
        if self.market_mode != "perps" or ":" in str(symbol):
            return symbol
        return _resolve_perp_symbol(self.exchange, symbol) or symbol

    def _invalidate_symbol_cache(self, symbol: str, *, microdata: bool = False) -> None:
        """Invalidate in-memory panel cache entries after local data writes."""
        self._ohlcv_cache.pop(symbol, None)
        if microdata:
            self._microdata_symbol_cache.pop(symbol, None)

    @staticmethod
    def _tail_frame(df: pd.DataFrame, start_ts: Optional[pd.Timestamp]) -> pd.DataFrame:
        if start_ts is None or df is None or df.empty:
            return df
        return df[df.index >= start_ts]

    @staticmethod
    def _cache_covers(df: pd.DataFrame, start_ts: Optional[pd.Timestamp]) -> bool:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return False
        if start_ts is None:
            return True
        idx = pd.DatetimeIndex(df.index)
        return bool(len(idx) and pd.Timestamp(idx.min()) <= pd.Timestamp(start_ts))

    def _load_ohlcv_symbol_cached(
        self, symbol: str, start_ts: Optional[pd.Timestamp]
    ) -> pd.DataFrame:
        cached = self._ohlcv_cache.get(symbol)
        if self._cache_covers(cached, start_ts):
            return self._tail_frame(cached, start_ts)
        data = self.ohlcv_store.load(symbol, start_ts=start_ts, end_ts=None)
        if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
            data = data.sort_index()
            for col in ("open", "high", "low", "close", "volume"):
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors="coerce").astype(
                        np.float32
                    )
            self._ohlcv_cache[symbol] = data
        return data

    def _merge_symbol_cache(self, symbol: str, new_df: pd.DataFrame) -> None:
        if new_df is None or not isinstance(new_df, pd.DataFrame) or new_df.empty:
            return
        cached = self._ohlcv_cache.get(symbol)
        if cached is None or cached.empty:
            self._ohlcv_cache[symbol] = new_df.sort_index()
            return
        merged = pd.concat([cached, new_df]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        self._ohlcv_cache[symbol] = merged

    def _load_microdata_symbol_cached(
        self, symbol: str
    ) -> tuple[Optional[pd.Index], Dict[str, pd.Series]]:
        key = self._symbol_file_key(symbol)
        obp = self.orderbook_dir / f"{key}.parquet"
        frp = self.funding_dir / f"{key}.parquet"
        oip = self.open_interest_dir / f"{key}.parquet"
        ob_mtime = obp.stat().st_mtime if obp.exists() else None
        fr_mtime = frp.stat().st_mtime if frp.exists() else None
        oi_mtime = oip.stat().st_mtime if oip.exists() else None
        cached = self._microdata_symbol_cache.get(symbol)
        if (
            cached
            and cached[0] == ob_mtime
            and cached[1] == fr_mtime
            and len(cached) >= 4
            and cached[2] == oi_mtime
        ):
            by_field = cached[3]
            idx_union = None
            for series in by_field.values():
                idx_union = (
                    series.index if idx_union is None else idx_union.union(series.index)
                )
            return idx_union, by_field

        by_field: Dict[str, pd.Series] = {}
        idx_union = None
        if obp.exists():
            ob = pd.read_parquet(obp)
            ob.index = pd.to_datetime(ob.index, utc=True)
            for field_name in (
                "mid",
                "best_bid",
                "best_ask",
                "bid_qty_1",
                "ask_qty_1",
                "cum_bid_qty_l10",
                "cum_ask_qty_l10",
                "cum_bid_qty_l20",
                "cum_ask_qty_l20",
                "snapshot_ts",
                "trade_count_1h",
                "buy_qty_1h",
                "sell_qty_1h",
                "notional_1h",
                "buy_notional_1h",
                "sell_notional_1h",
                "vwap_1h",
                "mean_trade_qty_1h",
                "signed_flow_imbalance_1h",
            ):
                if field_name in ob.columns:
                    by_field[f"orderbook_{field_name}"] = pd.to_numeric(
                        ob[field_name], errors="coerce"
                    ).astype(np.float32)
            idx_union = ob.index if idx_union is None else idx_union.union(ob.index)
        if frp.exists():
            fr = pd.read_parquet(frp)
            fr.index = pd.to_datetime(fr.index, utc=True)
            for field_name in MICRODATA_FRAME_FIELDS:
                if field_name in fr.columns:
                    by_field[field_name] = pd.to_numeric(
                        fr[field_name], errors="coerce"
                    ).astype(np.float32)
            idx_union = fr.index if idx_union is None else idx_union.union(fr.index)
        if oip.exists():
            oi = pd.read_parquet(oip)
            oi.index = pd.to_datetime(oi.index, utc=True)
            oi = oi[~oi.index.duplicated(keep="last")].sort_index()
            oi_col = None
            for candidate in (
                "open_interest",
                "openInterestValue",
                "sumOpenInterestValue",
                "openInterestAmount",
                "openInterest",
                "sumOpenInterest",
            ):
                if candidate in oi.columns:
                    oi_col = candidate
                    break
            if oi_col is not None:
                oi_series = pd.to_numeric(oi[oi_col], errors="coerce").astype(
                    np.float32
                )
                existing = by_field.get("open_interest")
                if existing is not None:
                    # Funding-hourly is updated from the live ticker and can be
                    # fresher; open_interest_hourly provides the dense
                    # historical contract needed by 3d/7d OI features.
                    by_field["open_interest"] = existing.combine_first(oi_series)
                else:
                    by_field["open_interest"] = oi_series
                idx_union = oi.index if idx_union is None else idx_union.union(oi.index)

        self._microdata_symbol_cache[symbol] = (ob_mtime, fr_mtime, oi_mtime, by_field)
        return idx_union, by_field

    def _record_api_error(self, symbol: str, exc: Exception, *, context: str) -> None:
        category = classify_api_error(exc)
        self.api_error_counts[category] = self.api_error_counts.get(category, 0) + 1
        self.dead_letter_symbols[symbol] = f"{context}:{category}:{exc}"
        tprint(f"[DataFetcher] {context} failed for {symbol}: {category}: {exc}")

    def _log_microdata_error(
        self, symbol: str, exc: Exception, *, context: str
    ) -> None:
        category = classify_api_error(exc)
        tprint(f"[DataFetcher] {context} failed for {symbol}: {category}: {exc}")

    def _get_funding_exchange_and_symbol(
        self, symbol: str
    ) -> tuple[Optional[Any], Optional[str]]:
        if symbol in self._symbols_without_perp_funding:
            return None, None
        if symbol and ":" in symbol:
            return self.exchange, symbol
        if self._perp_exchange is None:
            self._perp_exchange = make_perp_exchange()
        perp_symbol = _resolve_perp_symbol(self._perp_exchange, symbol)
        if not perp_symbol:
            self._symbols_without_perp_funding.add(symbol)
            return None, None
        return self._perp_exchange, perp_symbol

    def _ticker_lookup_key(self, symbol: str) -> str:
        try:
            market = self.exchange.market(symbol)
            market_id = str(market.get("id") or "").upper()
            if market_id:
                return market_id
        except Exception:
            pass
        return str(symbol).replace("/", "").replace(":USD", "").upper()

    def _public_ticker_snapshot_map(self) -> Dict[str, Dict[str, Any]]:
        """Return a short-lived Kraken Futures ticker map keyed by exchange market id."""
        now_mono = time.monotonic()
        if (
            self._ticker_snapshot_cache is not None
            and now_mono - self._ticker_snapshot_cache[0] < 10.0
        ):
            return self._ticker_snapshot_cache[1]
        with self._ticker_snapshot_lock:
            now_mono = time.monotonic()
            if (
                self._ticker_snapshot_cache is not None
                and now_mono - self._ticker_snapshot_cache[0] < 10.0
            ):
                return self._ticker_snapshot_cache[1]
            out: Dict[str, Dict[str, Any]] = {}
            if hasattr(self.exchange, "publicGetTickers"):
                payload = self.exchange.publicGetTickers()
                rows = payload.get("tickers") if isinstance(payload, dict) else None
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        market_id = str(row.get("symbol") or "").upper()
                        if market_id:
                            out[market_id] = row
            self._ticker_snapshot_cache = (now_mono, out)
            return out

    def _fetch_live_derivative_snapshot(
        self,
        symbol: str,
        *,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        if self.market_mode != "perps":
            return pd.DataFrame()
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        row: Dict[str, float] = {}
        try:
            ticker_map = self._public_ticker_snapshot_map()
            raw = ticker_map.get(self._ticker_lookup_key(symbol).upper())
            if isinstance(raw, dict):
                mapping = {
                    "funding_rate": "fundingRate",
                    "open_interest": "openInterestValue",
                    "mark_price": "markPrice",
                    "index_price": "indexPrice",
                }
                for out_col, src_col in mapping.items():
                    value = pd.to_numeric(raw.get(src_col), errors="coerce")
                    if pd.notna(value) and np.isfinite(float(value)):
                        row[out_col] = float(value)
                if "open_interest" not in row:
                    value = pd.to_numeric(raw.get("openInterest"), errors="coerce")
                    ref_price = pd.to_numeric(
                        raw.get("markPrice") or raw.get("indexPrice"), errors="coerce"
                    )
                    if (
                        pd.notna(value)
                        and np.isfinite(float(value))
                        and pd.notna(ref_price)
                        and np.isfinite(float(ref_price))
                        and float(ref_price) > 0.0
                    ):
                        row["open_interest"] = float(value) * float(ref_price)
                if "mark_price" in row and "index_price" in row and row["index_price"] > 0:
                    row["premium_index"] = float(row["mark_price"] / row["index_price"] - 1.0)
        except Exception as exc:
            self._log_microdata_error(symbol, exc, context="microdata_ticker_snapshot")
        if not row:
            return pd.DataFrame()
        return pd.DataFrame(row, index=pd.DatetimeIndex([ts.floor("1h")], name="ts"))

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
        now = pd.Timestamp.now(tz="UTC")
        latest_closed_hour = now.floor("h") - pd.Timedelta(hours=1)

        updated = 0
        skipped = 0
        failed = 0
        total = len(symbols)
        tprint(f"Initializing historical data batch: symbols={total}")
        fetch_timeout_seconds = float(
            os.environ.get("EPM_STARTUP_OHLCV_FETCH_TIMEOUT_SECONDS", "45") or "45"
        )

        def fetch_with_timeout(
            symbol_: str, start_: pd.Timestamp, end_: pd.Timestamp
        ) -> pd.DataFrame:
            executor = ThreadPoolExecutor(max_workers=1)
            fut = executor.submit(self.fetch_ohlcv, symbol_, start_, end_)
            try:
                return fut.result(timeout=fetch_timeout_seconds)
            except Exception as exc:
                fut.cancel()
                tprint(
                    "Startup historical OHLCV fetch skipped after timeout/error: "
                    f"symbol={symbol_} timeout={fetch_timeout_seconds:.1f}s "
                    f"error={exc}"
                )
                return pd.DataFrame()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        for i, symbol in enumerate(symbols, start=1):
            if i == 1 or i == total or i % 25 == 0:
                tprint(f"Historical data init progress: {i}/{total}")

            # Check only recent existing data. Startup only needs to know
            # whether the live tail is current; loading four years per symbol
            # makes every process restart expensive.
            try:
                meta = getattr(self.ohlcv_store, "_read_meta", lambda _s: {})(symbol)
                last_ts_ms = int((meta or {}).get("last_ts_ms", 0) or 0)
                if last_ts_ms > 0:
                    existing_data = pd.DataFrame(
                        index=pd.DatetimeIndex(
                            [pd.Timestamp(last_ts_ms, unit="ms", tz="UTC")]
                        )
                    )
                else:
                    tail_start = now - pd.Timedelta(
                        hours=max(int(lookback_hours), 48)
                    )
                    existing_data = self.ohlcv_store.load(
                        symbol, start_ts=tail_start, end_ts=now
                    )
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
                # Startup should only warm availability metadata. The hourly
                # refresh path fetches the latest closed bar, and the explicit
                # gap backfill path repairs older gaps. Doing catch-up work
                # here rewrites many yearly partitions on every process start.
                skipped += 1
            else:
                # Do not bootstrap historical data during live startup. A
                # symbol with no stored history cannot satisfy the strict model
                # feature contract anyway, and the hourly refresh path will add
                # the latest closed bar without rewriting existing partitions.
                skipped += 1
        tprint(
            "Historical data init batch complete: "
            f"updated={updated} up_to_date={skipped} failed={failed}"
        )

    def fetch_ohlcv(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Fetch strict closed 1h OHLCV for the inference hourly store.

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
            hours = max(
                1,
                int(
                    np.ceil(
                        (
                            pd.Timestamp(end).tz_convert("UTC")
                            - pd.Timestamp(start).tz_convert("UTC")
                        ).total_seconds()
                        / 3600.0
                    )
                )
                + 2,
            )
            ohlcv_1h = self._fetch_with_retry(
                symbol,
                timeframe="1h",
                since=int(start.timestamp() * 1000),
                limit=min(max(hours, 1), 1200),
            )
        except Exception as exc:
            self._record_api_error(symbol, exc, context="fetch_ohlcv")
            return pd.DataFrame()

        # Validate response
        if ohlcv_1h is None:
            tprint(f"  Warning: None returned for {symbol}, returning empty DataFrame")
            return pd.DataFrame()
        if isinstance(ohlcv_1h, str):
            tprint(f"  Warning: Error string returned for {symbol}: {ohlcv_1h[:100]}")
            return pd.DataFrame()
        if not isinstance(ohlcv_1h, list):
            tprint(
                f"  Warning: Unexpected type {type(ohlcv_1h)} for {symbol}, returning empty DataFrame"
            )
            return pd.DataFrame()
        if len(ohlcv_1h) == 0:
            tprint(
                f"  Warning: Empty list returned for {symbol}, returning empty DataFrame"
            )
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv_1h, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Convert timestamp column to datetime with timezone
        timestamps = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = timestamps
        df.set_index("timestamp", inplace=True)

        # Filter to requested range
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        df = df[df.index == df.index.floor("h")]

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

        return df[["open", "high", "low", "close", "volume"]].astype(np.float32)

    def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
        params: Optional[Dict[str, Any]] = None,
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
                exchange_symbol = self._exchange_symbol(symbol)
                return self.exchange.fetch_ohlcv(
                    symbol=exchange_symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                    params=dict(params or {}),
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
        """Aggregate intrahour candles into strict hourly bars on hour timestamps.

        Args:
            df_15m: DataFrame with 15m OHLCV data

        Returns:
            DataFrame with 1h OHLCV data on hourly timestamps
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

        df_1h = df_15m.resample("1h", label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

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
        df = df[df.index == target_hour]
        if df.empty:
            return pd.DataFrame()
        df = df[~df.index.duplicated(keep="last")]
        if self.market_mode == "perps":
            api_symbol = self._exchange_symbol(symbol)
            row_ts = pd.Timestamp(df.index.max())
            derivative_snapshot = self._fetch_live_derivative_snapshot(
                api_symbol,
                timestamp=row_ts,
            )
            if isinstance(derivative_snapshot, pd.DataFrame) and not derivative_snapshot.empty:
                aligned = derivative_snapshot.reindex(df.index).ffill().bfill()
                for col in MICRODATA_FRAME_FIELDS:
                    if col in aligned.columns:
                        df[col] = pd.to_numeric(aligned[col], errors="coerce").astype(
                            np.float32
                        )
            try:
                mark_df = self._fetch_with_retry(
                    api_symbol,
                    timeframe="1h",
                    since=int(row_ts.timestamp() * 1000),
                    limit=1,
                    params={"price": "mark"},
                )
                if mark_df:
                    mark_row = pd.DataFrame(
                        mark_df,
                        columns=[
                            "timestamp",
                            "mark_open",
                            "mark_high",
                            "mark_low",
                            "mark_close",
                            "mark_volume",
                        ],
                    )
                    mark_row["timestamp"] = pd.to_datetime(
                        mark_row["timestamp"], unit="ms", utc=True
                    )
                    mark_row = mark_row.set_index("timestamp").reindex(df.index)
                    for col in ("mark_open", "mark_high", "mark_low", "mark_close"):
                        if col in mark_row.columns:
                            df[col] = pd.to_numeric(
                                mark_row[col], errors="coerce"
                            ).astype(np.float32)
                    if "mark_close" in df.columns:
                        df["mark_price"] = df["mark_close"]
            except Exception as exc:
                self._log_microdata_error(symbol, exc, context="hourly_mark_ohlcv")
            try:
                exchange_id = str(getattr(self.exchange, "id", "")).lower()
                price_type = "spot" if exchange_id == "krakenfutures" else "index"
                index_df = self._fetch_with_retry(
                    api_symbol,
                    timeframe="1h",
                    since=int(row_ts.timestamp() * 1000),
                    limit=1,
                    params={"price": price_type},
                )
            except Exception:
                try:
                    price_type = "spot"
                    index_df = self._fetch_with_retry(
                        api_symbol,
                        timeframe="1h",
                        since=int(row_ts.timestamp() * 1000),
                        limit=1,
                        params={"price": "spot"},
                    )
                except Exception as exc:
                    index_df = None
                    self._log_microdata_error(symbol, exc, context="hourly_index_ohlcv")
            if index_df:
                prefix = "spot" if str(price_type) == "spot" else "index"
                price_row = pd.DataFrame(
                    index_df,
                    columns=[
                        "timestamp",
                        f"{prefix}_open",
                        f"{prefix}_high",
                        f"{prefix}_low",
                        f"{prefix}_close",
                        f"{prefix}_volume",
                    ],
                )
                price_row["timestamp"] = pd.to_datetime(
                    price_row["timestamp"], unit="ms", utc=True
                )
                price_row = price_row.set_index("timestamp").reindex(df.index)
                for col in (
                    f"{prefix}_open",
                    f"{prefix}_high",
                    f"{prefix}_low",
                    f"{prefix}_close",
                ):
                    if col in price_row.columns:
                        df[col] = pd.to_numeric(price_row[col], errors="coerce").astype(
                            np.float32
                        )
                if prefix == "index" and "index_close" in df.columns:
                    df["index_price"] = df["index_close"]
        self.ohlcv_store.save_partitioned(symbol=symbol, df=df, defer_compact=True)
        self._merge_symbol_cache(symbol, df)
        return df

    def fetch_hourly_universe_once(
        self,
        symbols: List[str],
        *,
        max_workers: int = 16,
        microdata_max_workers: Optional[int] = None,
        no_progress_timeout_seconds: float = 60.0,
        target_hour: Optional[pd.Timestamp] = None,
        check_recent_gaps_days: int = 7,
        backfill_fn: Optional[Any] = None,
        refresh_microdata: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch one closed 1h candle for the full live universe.

        The batch uses bounded fanout and stops waiting when no symbol has
        produced new OHLCV for ``no_progress_timeout_seconds``.
        """
        worker_cap = _read_int_env(
            "EPM_HOURLY_OHLCV_MAX_WORKERS",
            64,
            minimum=1,
            maximum=128,
        )
        workers = max(1, min(int(max_workers), worker_cap))
        if microdata_max_workers is None:
            microdata_workers = _read_int_env(
                "EPM_HOURLY_MICRODATA_WORKERS",
                min(workers, 24),
                minimum=1,
                maximum=worker_cap,
            )
        else:
            microdata_workers = max(1, min(int(microdata_max_workers), worker_cap))
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
        skipped_existing = 0
        microdata_symbols: list[str] = []
        microdata_refreshed = 0
        microdata_failed = 0
        microdata_canceled = 0
        symbols_to_fetch: list[str] = []
        skipped_existing_symbols: list[str] = []
        for sym in symbols:
            try:
                cached_bar = self.ohlcv_store.load(
                    sym,
                    start_ts=target_hour,
                    end_ts=target_hour + pd.Timedelta(hours=1),
                )
                if isinstance(cached_bar, pd.DataFrame) and not cached_bar.empty:
                    hourly_index = pd.DatetimeIndex(cached_bar.index)
                    if bool((hourly_index == target_hour).any()):
                        skipped_existing += 1
                        skipped_existing_symbols.append(sym)
                        continue
            except Exception:
                pass
            symbols_to_fetch.append(sym)
        tprint(
            "Hourly OHLCV universe batch start: "
            f"symbols={len(symbols)} fetch={len(symbols_to_fetch)} "
            f"skipped_existing={skipped_existing} workers={workers} "
            f"worker_cap={worker_cap} microdata_workers={microdata_workers} "
            f"target_hour={target_hour}"
        )
        started_at = time.monotonic()
        if refresh_microdata and skipped_existing_symbols:
            # OHLCV target-hour bars and derivative microdata have independent
            # freshness. A restart can have the candle already on disk while
            # the current ticker-derived OI/funding/mark snapshot is missing;
            # refreshing skipped-current symbols keeps OI-derived model
            # contracts finite without redownloading the hourly candle.
            microdata_symbols.extend(skipped_existing_symbols)

        target_hour_first = str(
            os.environ.get("EPM_HOURLY_FETCH_TARGET_HOUR_FIRST", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        if (
            check_recent_gaps_days > 0
            and skipped_existing_symbols
            and not target_hour_first
        ):
            def _gap_backfill_skipped(sym: str) -> Optional[pd.DataFrame]:
                if not self.has_recent_gap(sym, days=check_recent_gaps_days):
                    return None
                backfilled = self.trigger_hourly_gap_backfill(
                    sym,
                    days=check_recent_gaps_days,
                )
                self._invalidate_symbol_cache(sym, microdata=False)
                return backfilled

            gap_workers = max(1, min(workers, len(skipped_existing_symbols)))
            with ThreadPoolExecutor(max_workers=gap_workers) as gap_executor:
                gap_futures = {
                    gap_executor.submit(_gap_backfill_skipped, sym): sym
                    for sym in skipped_existing_symbols
                }
                for fut in as_completed(gap_futures):
                    sym = gap_futures[fut]
                    try:
                        backfilled = fut.result()
                        if isinstance(backfilled, pd.DataFrame) and not backfilled.empty:
                            out[sym] = backfilled
                            gap_backfills += 1
                    except Exception as exc:
                        failed += 1
                        self._record_api_error(sym, exc, context="gap_backfill")

        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(
                self.fetch_latest_hourly_symbol, sym, target_hour=target_hour
            ): sym
            for sym in symbols_to_fetch
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
                        if refresh_microdata:
                            microdata_symbols.append(sym)
                    else:
                        empty += 1
                    if check_recent_gaps_days > 0 and self.has_recent_gap(
                        sym, days=check_recent_gaps_days
                    ):
                        try:
                            backfilled = self.trigger_hourly_gap_backfill(
                                sym,
                                days=check_recent_gaps_days,
                            )
                            self._invalidate_symbol_cache(sym, microdata=False)
                            if isinstance(backfilled, pd.DataFrame) and not backfilled.empty:
                                out[sym] = backfilled
                            gap_backfills += 1
                        except Exception as exc:
                            failed += 1
                            self._record_api_error(sym, exc, context="gap_backfill")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if (
            check_recent_gaps_days > 0
            and skipped_existing_symbols
            and target_hour_first
            and str(
                os.environ.get("EPM_HOURLY_GAP_BACKFILL_AFTER_TARGET_FETCH", "1")
            ).strip().lower()
            not in {"0", "false", "no", "off"}
        ):
            tprint(
                "Hourly target-hour fetch finished; starting optional recent-gap "
                f"repair for skipped-current symbols={len(skipped_existing_symbols)}"
            )

            def _gap_backfill_skipped(sym: str) -> Optional[pd.DataFrame]:
                if not self.has_recent_gap(sym, days=check_recent_gaps_days):
                    return None
                backfilled = self.trigger_hourly_gap_backfill(
                    sym,
                    days=check_recent_gaps_days,
                )
                self._invalidate_symbol_cache(sym, microdata=False)
                return backfilled

            gap_workers = max(1, min(workers, len(skipped_existing_symbols)))
            with ThreadPoolExecutor(max_workers=gap_workers) as gap_executor:
                gap_futures = {
                    gap_executor.submit(_gap_backfill_skipped, sym): sym
                    for sym in skipped_existing_symbols
                }
                for fut in as_completed(gap_futures):
                    sym = gap_futures[fut]
                    try:
                        backfilled = fut.result()
                        if isinstance(backfilled, pd.DataFrame) and not backfilled.empty:
                            out[sym] = backfilled
                            gap_backfills += 1
                    except Exception as exc:
                        failed += 1
                        self._record_api_error(sym, exc, context="gap_backfill")

        if refresh_microdata and microdata_symbols:
            microdata_symbols = list(dict.fromkeys(microdata_symbols))
            microdata_workers = max(
                1, min(int(microdata_workers), len(microdata_symbols), worker_cap)
            )
            tprint(
                "Hourly microdata refresh batch start: "
                f"symbols={len(microdata_symbols)} workers={microdata_workers} "
                f"target_hour={target_hour}"
            )
            micro_executor = ThreadPoolExecutor(max_workers=microdata_workers)
            micro_futures = {
                micro_executor.submit(
                    self.update_microdata_symbol,
                    sym,
                    start_ts=target_hour,
                    end_ts=target_hour,
                ): sym
                for sym in microdata_symbols
            }
            micro_pending = set(micro_futures.keys())
            last_microdata_time = time.monotonic()
            try:
                while micro_pending:
                    done, micro_pending = wait(
                        micro_pending, timeout=1.0, return_when=FIRST_COMPLETED
                    )
                    if not done:
                        quiet_for = time.monotonic() - last_microdata_time
                        if quiet_for >= float(no_progress_timeout_seconds):
                            microdata_canceled = len(micro_pending)
                            for fut in micro_pending:
                                fut.cancel()
                            tprint(
                                "Hourly microdata no-progress timeout: "
                                f"quiet_for={quiet_for:.1f}s "
                                f"canceled={microdata_canceled}"
                            )
                            break
                        continue
                    for fut in done:
                        sym = micro_futures[fut]
                        try:
                            fut.result()
                            self._microdata_symbol_cache.pop(sym, None)
                            microdata_refreshed += 1
                            last_microdata_time = time.monotonic()
                        except Exception as exc:
                            microdata_failed += 1
                            self._log_microdata_error(
                                sym, exc, context="microdata_refresh"
                            )
            finally:
                micro_executor.shutdown(wait=False, cancel_futures=True)

        tprint(
            "Hourly OHLCV universe batch complete: "
            f"requested={len(symbols)} fetch={len(symbols_to_fetch)} "
            f"skipped_existing={skipped_existing} updated={len(out)} empty={empty} "
            f"failed={failed} canceled={canceled} gap_backfills={gap_backfills} "
            f"microdata_refreshed={microdata_refreshed} "
            f"microdata_failed={microdata_failed} "
            f"microdata_canceled={microdata_canceled} "
            f"elapsed={time.monotonic() - started_at:.3f}s "
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

    def trigger_hourly_gap_backfill(
        self,
        symbol: str,
        *,
        days: int = 7,
    ) -> Optional[pd.DataFrame]:
        """Backfill recent missing 1h OHLCV rows in the inference hourly store."""
        end_ts = pd.Timestamp.now(tz="UTC")
        start_ts = end_ts - pd.Timedelta(days=int(days))
        new_data = self.fetch_ohlcv(symbol, start=start_ts, end=end_ts)
        if new_data is None or not isinstance(new_data, pd.DataFrame) or new_data.empty:
            return pd.DataFrame()
        existing = self.ohlcv_store.load(symbol, start_ts=None, end_ts=None)
        merged = (
            self._resample_and_merge(existing, new_data)
            if isinstance(existing, pd.DataFrame) and not existing.empty
            else new_data
        )
        if isinstance(merged, pd.DataFrame) and not merged.empty:
            self.ohlcv_store.save_partitioned(symbol=symbol, df=merged)
        return new_data

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
        refresh_microdata: bool = True,
        microdata_start_ts: Optional[pd.Timestamp] = None,
        microdata_end_ts: Optional[pd.Timestamp] = None,
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
                if refresh_microdata:
                    try:
                        self.update_microdata_symbol(
                            sym,
                            start_ts=microdata_start_ts,
                            end_ts=microdata_end_ts,
                        )
                    except Exception as exc:
                        self._log_microdata_error(sym, exc, context="microdata_refresh")
                if check_recent_gaps_days > 0 and self.has_recent_gap(
                    sym, days=check_recent_gaps_days
                ):
                    tprint(f"Detected recent gap for {sym}; invoking backfill")
                    try:
                        self.trigger_hourly_gap_backfill(
                            sym,
                            days=check_recent_gaps_days,
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

    def _symbol_file_key(self, symbol: str) -> str:
        return symbol.replace("/", "_").replace(":", "_")

    def update_microdata_symbol(
        self,
        symbol: str,
        backfill_days: int = 180,
        start_ts: pd.Timestamp | None = None,
        end_ts: pd.Timestamp | None = None,
    ) -> Dict[str, bool]:
        """Incrementally refresh orderbook/funding snapshots for one symbol."""
        microdata_min_ranges = 10
        now_h = pd.Timestamp.now(tz="UTC").floor("1h")
        end_h = (
            pd.Timestamp(end_ts).tz_convert("UTC").floor("1h")
            if end_ts is not None and pd.Timestamp(end_ts).tzinfo is not None
            else (
                pd.Timestamp(end_ts).tz_localize("UTC").floor("1h")
                if end_ts is not None
                else now_h
            )
        )
        if start_ts is not None:
            start_h = (
                pd.Timestamp(start_ts).tz_convert("UTC").floor("1h")
                if pd.Timestamp(start_ts).tzinfo is not None
                else pd.Timestamp(start_ts).tz_localize("UTC").floor("1h")
            )
        else:
            start_h = end_h - pd.Timedelta(days=int(backfill_days))
        if start_h > end_h:
            start_h, end_h = end_h, start_h
        ob_path = self.orderbook_dir / f"{self._symbol_file_key(symbol)}.parquet"
        fr_path = self.funding_dir / f"{self._symbol_file_key(symbol)}.parquet"
        out = {"orderbook": False, "funding": False}

        try:
            existing_ob = pd.read_parquet(ob_path) if ob_path.exists() else None
            existing_idx = (
                pd.to_datetime(existing_ob.index, utc=True, errors="coerce")
                if existing_ob is not None and not existing_ob.empty
                else None
            )
            missing_ob_ranges = list(
                _compute_missing_hourly_ranges(existing_idx, start_h, end_h)
            )
            if len(missing_ob_ranges) < microdata_min_ranges:
                missing_ob_ranges = [
                    (range_start, range_end)
                    for range_start, range_end in missing_ob_ranges
                    if pd.Timestamp(range_end) >= end_h
                ]
            ob_frames = []
            for range_start, range_end in missing_ob_ranges:
                proxy_df = fetch_hourly_orderbook_proxy(
                    self.exchange,
                    symbol,
                    int(range_start.value // 10**6),
                    int(range_end.value // 10**6),
                )
                if proxy_df is None or proxy_df.empty:
                    local_ohlcv = self._load_ohlcv_symbol_cached(symbol, range_start)
                    if (
                        isinstance(local_ohlcv, pd.DataFrame)
                        and not local_ohlcv.empty
                    ):
                        local_ohlcv = local_ohlcv.loc[
                            (local_ohlcv.index >= range_start)
                            & (local_ohlcv.index <= range_end)
                        ]
                        proxy_df = build_hourly_orderbook_proxy_from_ohlcv(local_ohlcv)
                if proxy_df is not None and not proxy_df.empty:
                    ob_frames.append(proxy_df)
            if existing_ob is not None and not existing_ob.empty:
                ob_frames.insert(0, existing_ob)
            if ob_frames:
                rec = pd.concat(ob_frames).sort_index().groupby(level=0).last()
                rec = normalize_orderbook_proxy_frame(rec)
                rec.to_parquet(ob_path)
                out["orderbook"] = True
        except Exception as exc:
            self._log_microdata_error(symbol, exc, context="microdata_orderbook")

        try:
            fr_df = None
            existing_funding = None
            funding_exchange, funding_symbol = self._get_funding_exchange_and_symbol(
                symbol
            )
            if not funding_symbol or funding_exchange is None:
                return out
            if hasattr(funding_exchange, "fetch_funding_rate_history"):
                if fr_path.exists():
                    existing_funding = pd.read_parquet(fr_path)
                until_ms = int((end_h + pd.Timedelta(hours=1)).value // 10**6)
                existing_idx = (
                    pd.to_datetime(existing_funding.index, utc=True, errors="coerce")
                    if existing_funding is not None and not existing_funding.empty
                    else None
                )
                funding_missing_ranges = list(
                    _compute_missing_funding_ranges(existing_idx, start_h, end_h)
                )
                if len(funding_missing_ranges) < microdata_min_ranges:
                    funding_missing_ranges = []
                funding_frames = []
                for range_start, range_end in funding_missing_ranges:
                    hist = _fetch_ccxt_history_paged(
                        funding_exchange.fetch_funding_rate_history,
                        funding_symbol,
                        int(range_start.value // 10**6),
                        int(range_end.value // 10**6),
                        value_keys=["fundingRate", "funding_rate", "rate"],
                        exchange=funding_exchange,
                        limit=1000,
                    )
                    if len(hist) > 0:
                        funding_frames.append(hist.to_frame(name="funding_rate"))
                if funding_frames:
                    fr_df = pd.concat(funding_frames).sort_index()
            if fr_df is None and hasattr(funding_exchange, "fetch_funding_rate"):
                fr_df = pd.DataFrame(
                    [funding_exchange.fetch_funding_rate(funding_symbol)]
                )
            live_derivatives = self._fetch_live_derivative_snapshot(
                funding_symbol,
                timestamp=end_h,
            )
            if fr_df is not None and not fr_df.empty:
                if "funding_rate" in fr_df.columns and isinstance(
                    fr_df.index, pd.DatetimeIndex
                ):
                    fr_df = fr_df.copy()
                    fr_df.index = pd.to_datetime(fr_df.index, utc=True).floor("1h")
                    fr_df = fr_df[["funding_rate"]]
                else:
                    ts_col = (
                        "timestamp"
                        if "timestamp" in fr_df.columns
                        else "fundingTimestamp"
                    )
                    rate_col = (
                        "fundingRate"
                        if "fundingRate" in fr_df.columns
                        else "funding_rate"
                    )
                    fr_df["ts"] = pd.to_datetime(
                        fr_df[ts_col], unit="ms", utc=True
                    ).dt.floor("1h")
                    fr_df = (
                        fr_df[["ts", rate_col]]
                        .rename(columns={rate_col: "funding_rate"})
                        .set_index("ts")
                    )
                fr_df["funding_rate"] = pd.to_numeric(
                    fr_df["funding_rate"], errors="coerce"
                ).astype(np.float32)
            if (
                live_derivatives is not None
                and isinstance(live_derivatives, pd.DataFrame)
                and not live_derivatives.empty
            ):
                if fr_df is not None and not fr_df.empty:
                    fr_df = pd.concat([fr_df, live_derivatives], sort=True)
                else:
                    fr_df = live_derivatives
            if fr_df is not None and not fr_df.empty:
                fr_df.index = pd.to_datetime(fr_df.index, utc=True).floor("1h")
                for col in MICRODATA_FRAME_FIELDS:
                    if col in fr_df.columns:
                        fr_df[col] = pd.to_numeric(fr_df[col], errors="coerce").astype(
                            np.float32
                        )
                keep_cols = [col for col in MICRODATA_FRAME_FIELDS if col in fr_df.columns]
                fr_df = fr_df[keep_cols]
                if existing_funding is None and fr_path.exists():
                    existing_funding = pd.read_parquet(fr_path)
                if existing_funding is not None and not existing_funding.empty:
                    fr_df = pd.concat([existing_funding, fr_df], sort=True)
                fr_df = fr_df.sort_index().groupby(level=0).last()
                fr_df.to_parquet(fr_path)
                out["funding"] = "funding_rate" in fr_df.columns
        except Exception as exc:
            self._log_microdata_error(symbol, exc, context="microdata_funding")
        return out

    def _load_microdata_panel(
        self,
        symbols: List[str],
        *,
        start_ts: Optional[pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        idx_union = None
        orderbook_fields = {
            "mid": {},
            "best_bid": {},
            "best_ask": {},
            "bid_qty_1": {},
            "ask_qty_1": {},
            "cum_bid_qty_l10": {},
            "cum_ask_qty_l10": {},
            "cum_bid_qty_l20": {},
            "cum_ask_qty_l20": {},
            "snapshot_ts": {},
            "trade_count_1h": {},
            "buy_qty_1h": {},
            "sell_qty_1h": {},
            "notional_1h": {},
            "buy_notional_1h": {},
            "sell_notional_1h": {},
            "vwap_1h": {},
            "mean_trade_qty_1h": {},
            "signed_flow_imbalance_1h": {},
        }
        microdata_fields = {field_name: {} for field_name in MICRODATA_FRAME_FIELDS}
        for sym in symbols:
            sym_idx, by_field = self._load_microdata_symbol_cached(sym)
            if sym_idx is not None:
                if start_ts is not None:
                    sym_idx = pd.DatetimeIndex(sym_idx)
                    sym_idx = sym_idx[sym_idx >= pd.Timestamp(start_ts)]
                    if sym_idx.empty:
                        continue
                idx_union = sym_idx if idx_union is None else idx_union.union(sym_idx)
            for field_name in orderbook_fields:
                series = by_field.get(f"orderbook_{field_name}")
                if series is not None:
                    if start_ts is not None:
                        series = series[series.index >= pd.Timestamp(start_ts)]
                    orderbook_fields[field_name][sym] = series
            for field_name in microdata_fields:
                series = by_field.get(field_name)
                if series is not None:
                    if start_ts is not None:
                        series = series[series.index >= pd.Timestamp(start_ts)]
                    microdata_fields[field_name][sym] = series
        if idx_union is None:
            return {}
        idx_union = pd.DatetimeIndex(idx_union).sort_values().unique()
        out = {}
        if orderbook_fields["mid"]:
            out["orderbook_hourly"] = (
                pd.DataFrame(orderbook_fields["mid"])
                .reindex(idx_union)
                .astype(np.float32)
            )
        for field_name, by_symbol in orderbook_fields.items():
            if not by_symbol:
                continue
            out[f"orderbook_{field_name}"] = (
                pd.DataFrame(by_symbol).reindex(idx_union).astype(np.float32)
            )
        for field_name, by_symbol in microdata_fields.items():
            if not by_symbol:
                continue
            out[field_name] = (
                pd.DataFrame(by_symbol).reindex(idx_union).ffill().astype(np.float32)
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
        cache_hits = 0
        cache_misses = 0
        for symbol in symbols:
            try:
                cached = self._ohlcv_cache.get(symbol)
                if self._cache_covers(cached, start_ts):
                    cache_hits += 1
                else:
                    cache_misses += 1
                data = self._load_ohlcv_symbol_cached(symbol, start_ts=start_ts)
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
        tprint(
            "DataFetcher panel load: "
            f"symbols={len(symbols)} cache_hits={cache_hits} "
            f"cache_misses={cache_misses} lookback_hours={lookback_hours}"
        )
        panel = get_panel_from_dict(ohlcv_data)
        micro_panel = self._load_microdata_panel(symbols, start_ts=start_ts)
        for key, frame in micro_panel.items():
            existing = panel.get(key)
            if (
                isinstance(existing, pd.DataFrame)
                and not existing.empty
                and isinstance(frame, pd.DataFrame)
                and not frame.empty
            ):
                panel[key] = frame.combine_first(existing).sort_index()
            else:
                panel[key] = frame
        return panel


# Backwards compatibility: Keep existing functions for non-class usage
def make_exchange(market_mode: str = "spot") -> Any:
    """Create and return the configured exchange instance for spot or perps.

    Returns:
        ccxt exchange instance with rate limiting enabled
    """
    mode = _normalise_market_mode(market_mode)
    try:
        ex = make_perp_exchange() if mode == "perps" else make_spot_exchange()
        label = "perp/swap" if mode == "perps" else "spot"
        exchange_id = str(getattr(ex, "id", "exchange")).upper()
        tprint(f"Created {exchange_id} {label} exchange and loaded markets")
        return ex
    except Exception as exc:
        exchange_label = str(os.environ.get("EPM_EXCHANGE") or "binance").upper()
        tprint(
            f"Failed to create {exchange_label} {mode} exchange: "
            f"{classify_api_error(exc)}: {exc}"
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
    for col in PERP_OHLCV_EXTRA_FIELDS:
        panel[col] = pd.DataFrame()

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

        for col in ["open", "high", "low", "close", "volume", *PERP_OHLCV_EXTRA_FIELDS]:
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
