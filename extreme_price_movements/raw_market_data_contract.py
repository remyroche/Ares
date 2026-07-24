"""Canonical raw Kraken market-data refresh and panel-loading contract.

Feature parity starts before feature formulas: every scoring path must observe
the same persisted UTC OHLCV/derivatives rows and construct panels with the
same field, symbol, and timestamp semantics.  This module is the shared entry
point for live inference, historical replay, and manual replay.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import os
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from extreme_price_movements.utils import tprint


RAW_MARKET_DATA_CONTRACT_VERSION = "kraken_raw_market_data_v1"


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _bounded_workers(requested: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 1
    if requested is None:
        try:
            requested = int(os.getenv("EPM_RAW_MARKET_DATA_WORKERS", "8") or "8")
        except (TypeError, ValueError):
            requested = 8
    return max(1, min(int(requested), int(task_count), 32))


@dataclass(frozen=True)
class RawMarketDataContract:
    """Persisted acquisition semantics shared by research and production."""

    version: str = RAW_MARKET_DATA_CONTRACT_VERSION
    exchange_id: str = "krakenfutures"
    market_mode: str = "perps"
    timezone: str = "UTC"
    decision_timeframe: str = "1h"
    gap_repair_timeframe: str = "15m"
    open_interest_unit: str = "quote_notional"
    panel_timestamp_semantics: str = "bar_open_for_completed_hour_utc"
    deterministic_symbol_order: bool = True

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawMarketRefreshResult:
    contract: Mapping[str, Any]
    requested_symbols: int
    updated_symbols: tuple[str, ...]
    failed_symbols: Mapping[str, str]
    max_workers: int
    read_only: bool

    @property
    def ok(self) -> bool:
        return not self.failed_symbols


def repair_hourly_from_complete_15m(
    *,
    store: Any,
    symbol: str,
    frame_15m: pd.DataFrame,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    """Materialize only absent completed hours backed by four 15m candles.

    This is the sole 15m-to-hourly repair rule used by download and inference.
    It never overwrites an existing hourly row and never emits the current,
    potentially incomplete hour.
    """

    required = ("open", "high", "low", "close", "volume")
    if not isinstance(frame_15m, pd.DataFrame) or frame_15m.empty:
        return pd.DataFrame(columns=required)
    if any(column not in frame_15m.columns for column in required):
        return pd.DataFrame(columns=required)

    frame = frame_15m.loc[:, required].copy(deep=False)
    index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    valid = ~index.isna()
    frame = frame.loc[valid]
    frame.index = pd.DatetimeIndex(index[valid])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        return pd.DataFrame(columns=required)

    hourly = frame.resample("1h", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = (
        pd.to_numeric(frame["close"], errors="coerce")
        .resample("1h", label="left", closed="left")
        .count()
    )
    last_complete = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta(hours=1)
    hourly = hourly.loc[counts.reindex(hourly.index).eq(4)]
    hourly = hourly.loc[hourly.index <= last_complete]
    if start_ts is not None:
        hourly = hourly.loc[hourly.index >= _utc_timestamp(start_ts).floor("1h")]
    if end_ts is not None:
        hourly = hourly.loc[hourly.index <= _utc_timestamp(end_ts).floor("1h")]
    hourly = hourly.dropna(subset=["open", "high", "low", "close"])
    if hourly.empty:
        return hourly

    existing = store.load(
        symbol,
        start_ts=pd.Timestamp(hourly.index.min()),
        end_ts=pd.Timestamp(hourly.index.max()),
    )
    present = (
        pd.DatetimeIndex(existing.index).floor("1h").unique()
        if isinstance(existing, pd.DataFrame) and not existing.empty
        else pd.DatetimeIndex([], tz="UTC")
    )
    repaired = hourly.loc[~hourly.index.isin(present)]
    if repaired.empty:
        return repaired
    repaired = repaired.astype({column: "float32" for column in required})
    if persist:
        store.save_partitioned(symbol=symbol, df=repaired)
    return repaired


def refresh_raw_market_rows(
    *,
    fetcher: Any,
    symbols: Sequence[str],
    target_hour: pd.Timestamp | None = None,
    max_workers: int | None = None,
    microdata_max_workers: int | None = None,
    no_progress_timeout_seconds: float = 60.0,
    check_recent_gaps_days: int = 7,
    refresh_microdata: bool = True,
    microdata_lookback_hours: int = 48,
    microdata_allow_live_snapshot: bool = True,
    read_only: bool = False,
    contract: RawMarketDataContract | None = None,
) -> RawMarketRefreshResult:
    """Refresh one completed hourly cross-section through the shared fetcher.

    ``DataFetcher.fetch_hourly_universe_once`` already implements Kraken's
    bounded concurrent 1h tail fetch, 15m internal-gap repair, derivative
    snapshots, and atomic persistence.  Keeping the call behind this contract
    makes every caller share those semantics while allowing parity replays to
    remain explicitly read-only.
    """

    contract = contract or RawMarketDataContract()
    ordered = tuple(sorted(dict.fromkeys(str(symbol) for symbol in symbols)))
    workers = _bounded_workers(max_workers, len(ordered))
    if read_only or not ordered:
        return RawMarketRefreshResult(
            contract=contract.manifest(),
            requested_symbols=len(ordered),
            updated_symbols=(),
            failed_symbols={},
            max_workers=workers,
            read_only=bool(read_only),
        )
    refreshed = fetcher.fetch_hourly_universe_once(
        list(ordered),
        max_workers=workers,
        microdata_max_workers=microdata_max_workers,
        no_progress_timeout_seconds=float(no_progress_timeout_seconds),
        target_hour=(
            _utc_timestamp(target_hour) if target_hour is not None else None
        ),
        check_recent_gaps_days=int(check_recent_gaps_days),
        refresh_microdata=bool(refresh_microdata),
        microdata_lookback_hours=max(0, int(microdata_lookback_hours)),
        microdata_allow_live_snapshot=bool(microdata_allow_live_snapshot),
    )
    updated = tuple(sorted(str(symbol) for symbol in (refreshed or {})))
    return RawMarketRefreshResult(
        contract=contract.manifest(),
        requested_symbols=len(ordered),
        updated_symbols=updated,
        failed_symbols={},
        max_workers=workers,
        read_only=False,
    )


def refresh_raw_market_history(
    *,
    store: Any,
    exchange: Any,
    symbols: Sequence[str],
    since_ts: pd.Timestamp,
    market_mode: str = "perps",
    spot_exchange: Any = None,
    max_workers: int | None = None,
    read_only: bool = False,
    contract: RawMarketDataContract | None = None,
) -> RawMarketRefreshResult:
    """Incrementally refresh historical rows with bounded symbol concurrency.

    The underlying partitioned-store updater is authoritative for Kraken OHLCV,
    funding, mark/spot references, and quote-notional OI.  Per-symbol file locks
    preserve atomicity while workers overlap network-bound requests.
    """

    contract = contract or RawMarketDataContract(market_mode=str(market_mode))
    ordered = tuple(sorted(dict.fromkeys(str(symbol) for symbol in symbols)))
    workers = _bounded_workers(max_workers, len(ordered))
    if read_only or not ordered:
        return RawMarketRefreshResult(
            contract=contract.manifest(),
            requested_symbols=len(ordered),
            updated_symbols=(),
            failed_symbols={},
            max_workers=workers,
            read_only=bool(read_only),
        )
    since_ms = int(_utc_timestamp(since_ts).value // 10**6)

    def _update(symbol: str) -> str:
        if str(market_mode).lower() in {"perp", "perps"}:
            store.update_symbol_perp(
                exchange,
                symbol,
                since_ms,
                spot_exchange=spot_exchange,
            )
        else:
            store.update_symbol(exchange, symbol, since_ms)
        return symbol

    updated: list[str] = []
    failed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_update, symbol): symbol for symbol in ordered}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                updated.append(str(future.result()))
            except Exception as exc:  # each symbol remains independently retryable
                failed[symbol] = f"{type(exc).__name__}: {exc}"
    if failed:
        tprint(
            "Raw market historical refresh completed with failures: "
            f"updated={len(updated)} failed={len(failed)} workers={workers}"
        )
    return RawMarketRefreshResult(
        contract=contract.manifest(),
        requested_symbols=len(ordered),
        updated_symbols=tuple(sorted(updated)),
        failed_symbols=dict(sorted(failed.items())),
        max_workers=workers,
        read_only=False,
    )


def load_raw_market_panel(
    *,
    store: Any,
    symbols: Sequence[str],
    panel_fields: Sequence[str],
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    max_workers: int | None = None,
    symbol_loader: Callable[[str, pd.Timestamp | None, pd.Timestamp | None], pd.DataFrame]
    | None = None,
    microdata_loader: Callable[
        [Sequence[str], pd.Timestamp | None, pd.Timestamp | None],
        Mapping[str, pd.DataFrame],
    ]
    | None = None,
) -> dict[str, pd.DataFrame]:
    """Load one deterministic UTC panel through bounded concurrent reads."""

    ordered = tuple(sorted(dict.fromkeys(str(symbol) for symbol in symbols)))
    start = _utc_timestamp(start_ts) if start_ts is not None else None
    end = _utc_timestamp(end_ts) if end_ts is not None else None
    workers = _bounded_workers(max_workers, len(ordered))
    fields = tuple(dict.fromkeys(str(field) for field in panel_fields))
    by_symbol: dict[str, pd.DataFrame] = {}

    def _load(symbol: str) -> pd.DataFrame:
        if symbol_loader is not None:
            frame = symbol_loader(symbol, start, end)
        else:
            frame = store.load(
                symbol,
                columns=None,
                start_ts=start,
                end_ts=end,
            )
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame()
        out = frame.copy(deep=False)
        index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out.loc[~index.isna()]
        out.index = pd.DatetimeIndex(index[~index.isna()])
        out = out[~out.index.duplicated(keep="last")].sort_index()
        if start is not None:
            out = out.loc[out.index >= start]
        if end is not None:
            out = out.loc[out.index <= end]
        return out

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_load, symbol): symbol for symbol in ordered}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                frame = future.result()
            except Exception as exc:
                tprint(f"Raw market panel load failed for {symbol}: {exc}")
                continue
            if not frame.empty:
                by_symbol[symbol] = frame

    panel: dict[str, pd.DataFrame] = {}
    for field in fields:
        series = [
            pd.to_numeric(by_symbol[symbol][field], errors="coerce").rename(symbol)
            for symbol in ordered
            if symbol in by_symbol and field in by_symbol[symbol]
        ]
        if series:
            panel[field] = pd.concat(series, axis=1).sort_index()

    if microdata_loader is not None:
        microdata = microdata_loader(ordered, start, end) or {}
        for name, frame in microdata.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            existing = panel.get(str(name))
            panel[str(name)] = (
                frame.combine_first(existing).sort_index()
                if isinstance(existing, pd.DataFrame) and not existing.empty
                else frame.sort_index()
            )
    return panel
