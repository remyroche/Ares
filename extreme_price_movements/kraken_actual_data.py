from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ACTUAL_VOLUME_VALID_STATUSES = frozenset({"actual_trades", "confirmed_no_trades"})
ACTUAL_VOLUME_TERMINAL_STATUSES = frozenset(
    {"actual_trades", "confirmed_no_trades", "source_conflict", "unavailable"}
)


@dataclass(frozen=True)
class SymbolCoverage:
    symbol: str
    price_rows: int
    missing_oi: int
    missing_volume: int
    missing_funding: int
    missing_any: int
    valid_all: int
    chart_positive_volume: int
    actual_trades: int
    confirmed_no_trades: int
    source_conflict_volume: int
    unavailable_volume: int
    linked_zero_carry: int
    isolated_zero_chart: int


def safe_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def load_verified_perp_symbols(manifest_path: Path) -> list[str]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = payload.get("symbols") if isinstance(payload, dict) else payload
    out: list[str] = []
    for row in rows or []:
        sym = row.get("perp_symbol") if isinstance(row, dict) else row
        if sym:
            out.append(str(sym))
    return list(dict.fromkeys(out))


def symbol_key_from_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "_")


def _coerce_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        idx = pd.to_datetime(out.index, utc=True, errors="coerce")
    elif "ts" in out.columns:
        idx = pd.to_datetime(out["ts"], utc=True, errors="coerce")
        out = out.drop(columns=["ts"])
    else:
        idx = pd.to_datetime(out.index, utc=True, errors="coerce")
    out.index = pd.DatetimeIndex(idx, name="ts")
    out = out.loc[~out.index.isna()].sort_index()
    return out.loc[~out.index.duplicated(keep="last")]


def _sidecar_candidates(root: Path, symbol_or_key: str) -> list[Path]:
    text = str(symbol_or_key)
    return [
        root / f"{safe_symbol(text)}.parquet",
        root / f"{symbol_key_from_symbol(text)}.parquet",
        root / f"{text}.parquet",
    ]


def load_sidecar_frame(root: Path, symbol_or_key: str) -> pd.DataFrame:
    for path in _sidecar_candidates(Path(root), symbol_or_key):
        if not path.exists():
            continue
        try:
            return _coerce_utc_index(pd.read_parquet(path))
        except Exception:
            continue
    return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))


def load_sidecar_series(root: Path, symbol_or_key: str, column: str) -> pd.Series:
    frame = load_sidecar_frame(root, symbol_or_key)
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.sort_index()


def load_partitioned_ohlcv_symbol(ohlcv_root: Path, symbol_key: str) -> pd.DataFrame:
    files = sorted((Path(ohlcv_root) / f"symbol={symbol_key}").glob("year=*/compact-*.parquet"))
    if not files:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    frames = [pd.read_parquet(path) for path in files]
    return _coerce_utc_index(pd.concat(frames, ignore_index=True))


def price_mask(df: pd.DataFrame) -> pd.Series:
    required = ["open", "high", "low", "close"]
    if any(col not in df.columns for col in required):
        return pd.Series(False, index=df.index)
    prices = df.loc[:, required].apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)
    return prices.gt(0.0).all(axis=1)


def chart_volume_classification(df: pd.DataFrame) -> pd.DataFrame:
    index = df.index
    out = pd.DataFrame(index=index)
    if df.empty or "volume" not in df.columns:
        out["chart_volume_valid"] = False
        out["chart_positive_volume"] = False
        out["linked_zero_carry"] = False
        out["isolated_zero_chart"] = False
        return out
    volume = pd.to_numeric(df["volume"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    open_ = pd.to_numeric(df.get("open"), errors="coerce")
    close = pd.to_numeric(df.get("close"), errors="coerce")
    positive = volume.gt(0.0)
    zero_no_trade = volume.eq(0.0) & open_.eq(close)
    prev_linked = (
        zero_no_trade
        & zero_no_trade.shift(1, fill_value=False)
        & close.shift(1).eq(open_)
    )
    next_linked = (
        zero_no_trade
        & zero_no_trade.shift(-1, fill_value=False)
        & close.eq(open_.shift(-1))
    )
    linked = zero_no_trade & (prev_linked | next_linked)
    isolated = zero_no_trade & ~linked
    out["chart_volume_valid"] = positive | isolated
    out["chart_positive_volume"] = positive
    out["linked_zero_carry"] = linked
    out["isolated_zero_chart"] = isolated
    return out


def chart_confirmed_no_trade_mask(ohlcv: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    if ohlcv is None or ohlcv.empty:
        return pd.Series(False, index=index)
    aligned = _coerce_utc_index(ohlcv).reindex(index)
    required = ["open", "high", "low", "close"]
    if any(col not in aligned.columns for col in required):
        return pd.Series(False, index=index)
    prices = aligned.loc[:, required].apply(pd.to_numeric, errors="coerce")
    flat = (
        prices["open"].notna()
        & prices["open"].eq(prices["high"])
        & prices["open"].eq(prices["low"])
        & prices["open"].eq(prices["close"])
    )
    if "volume" in aligned.columns:
        volume = pd.to_numeric(aligned["volume"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        volume_not_positive = ~volume.gt(0.0)
    else:
        volume_not_positive = pd.Series(True, index=index)
    return flat & volume_not_positive


def actual_volume_valid_mask(
    sidecar: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    ohlcv: pd.DataFrame | None = None,
) -> pd.Series:
    if sidecar.empty or "coverage_status" not in sidecar.columns:
        return pd.Series(False, index=index)
    aligned = sidecar.reindex(index)
    status = aligned["coverage_status"].astype(str)
    actual_trades = status.eq("actual_trades")
    confirmed_no_trades = status.eq("confirmed_no_trades")
    return actual_trades | (
        confirmed_no_trades & chart_confirmed_no_trade_mask(ohlcv, index)
    )


def actual_volume_terminal_mask(sidecar: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    if sidecar.empty or "coverage_status" not in sidecar.columns:
        return pd.Series(False, index=index)
    aligned = sidecar.reindex(index)
    status = aligned["coverage_status"].astype(str)
    return status.isin(ACTUAL_VOLUME_TERMINAL_STATUSES)


def overlay_actual_volume_sidecar(
    ohlcv: pd.DataFrame,
    *,
    root_dir: str | Path,
    symbol: str,
    sidecar_name: str = "actual_volume_hourly",
) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty or "volume" not in ohlcv.columns:
        return ohlcv
    out = _coerce_utc_index(ohlcv)
    sidecar = load_sidecar_frame(Path(root_dir) / sidecar_name, symbol)
    if sidecar.empty:
        return out
    valid = actual_volume_valid_mask(sidecar, out.index, ohlcv=out)
    if not bool(valid.any()):
        return out
    aligned = sidecar.reindex(out.index)
    if "volume" in aligned.columns:
        out.loc[valid, "volume"] = pd.to_numeric(aligned.loc[valid, "volume"], errors="coerce")
    if "quote_volume" in aligned.columns:
        out.loc[valid, "quote_volume"] = pd.to_numeric(
            aligned.loc[valid, "quote_volume"], errors="coerce"
        )
    if "trade_count" in aligned.columns:
        out.loc[valid, "trade_count"] = pd.to_numeric(
            aligned.loc[valid, "trade_count"], errors="coerce"
        )
    if "vwap" in aligned.columns:
        out.loc[valid, "vwap"] = pd.to_numeric(aligned.loc[valid, "vwap"], errors="coerce")
    return out


def contiguous_hour_ranges(
    missing_index: Iterable[pd.Timestamp],
    *,
    max_gap_hours: int = 720,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    idx = pd.DatetimeIndex(pd.to_datetime(list(missing_index), utc=True, errors="coerce"))
    idx = idx[~idx.isna()].floor("h").drop_duplicates().sort_values()
    if idx.empty:
        return []
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = idx[0]
    prev = idx[0]
    max_delta = pd.Timedelta(hours=max(1, int(max_gap_hours)) - 1)
    for ts in idx[1:]:
        if ts == prev + pd.Timedelta(hours=1) and ts - start <= max_delta:
            prev = ts
            continue
        ranges.append((start, prev + pd.Timedelta(hours=1)))
        start = ts
        prev = ts
    ranges.append((start, prev + pd.Timedelta(hours=1)))
    return ranges


def plan_symbol_coverage(
    *,
    symbol_key: str,
    ohlcv: pd.DataFrame,
    oi_sidecar: pd.Series | None = None,
    actual_volume_sidecar: pd.DataFrame | None = None,
    funding_sidecar: pd.Series | None = None,
    max_gap_hours: int = 720,
    retry_unavailable_volume: bool = False,
) -> tuple[SymbolCoverage, list[tuple[pd.Timestamp, pd.Timestamp]], list[tuple[pd.Timestamp, pd.Timestamp]]]:
    if ohlcv.empty:
        empty = SymbolCoverage(symbol_key, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return empty, [], []
    df = _coerce_utc_index(ohlcv)
    p_mask = price_mask(df)
    work = df.loc[p_mask].copy()
    if work.empty:
        empty = SymbolCoverage(symbol_key, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return empty, [], []
    index = work.index

    raw_oi = (
        pd.to_numeric(work["open_interest"], errors="coerce")
        if "open_interest" in work.columns
        else pd.Series(np.nan, index=index)
    )
    if oi_sidecar is not None and not oi_sidecar.empty:
        raw_oi = raw_oi.combine_first(oi_sidecar.reindex(index))
    oi_valid = raw_oi.replace([np.inf, -np.inf], np.nan).gt(0.0)

    raw_funding = (
        pd.to_numeric(work["funding_rate"], errors="coerce")
        if "funding_rate" in work.columns
        else pd.Series(np.nan, index=index)
    )
    if funding_sidecar is not None and not funding_sidecar.empty:
        raw_funding = raw_funding.combine_first(funding_sidecar.reindex(index))
    funding_valid = raw_funding.replace([np.inf, -np.inf], np.nan).notna()

    chart = chart_volume_classification(work)
    sidecar = actual_volume_sidecar if actual_volume_sidecar is not None else pd.DataFrame()
    actual_valid = actual_volume_valid_mask(sidecar, index, ohlcv=work)
    terminal = actual_volume_terminal_mask(sidecar, index)
    if retry_unavailable_volume:
        terminal = actual_valid
    volume_valid = chart["chart_positive_volume"].astype(bool) | actual_valid
    volume_missing_for_fetch = ~volume_valid & ~terminal
    missing_oi = ~oi_valid
    missing_volume = ~volume_valid
    missing_funding = ~funding_valid
    missing_any = missing_oi | missing_volume | missing_funding

    aligned_sidecar = sidecar.reindex(index) if not sidecar.empty else pd.DataFrame(index=index)
    status = (
        aligned_sidecar["coverage_status"].astype(str)
        if "coverage_status" in aligned_sidecar.columns
        else pd.Series("", index=index)
    )
    coverage = SymbolCoverage(
        symbol=symbol_key,
        price_rows=int(len(index)),
        missing_oi=int(missing_oi.sum()),
        missing_volume=int(missing_volume.sum()),
        missing_funding=int(missing_funding.sum()),
        missing_any=int(missing_any.sum()),
        valid_all=int((~missing_any).sum()),
        chart_positive_volume=int(chart["chart_positive_volume"].sum()),
        actual_trades=int(status.eq("actual_trades").sum()),
        confirmed_no_trades=int(status.eq("confirmed_no_trades").sum()),
        source_conflict_volume=int(status.eq("source_conflict").sum()),
        unavailable_volume=int(status.eq("unavailable").sum()),
        linked_zero_carry=int(chart["linked_zero_carry"].sum()),
        isolated_zero_chart=int(chart["isolated_zero_chart"].sum()),
    )
    oi_ranges = contiguous_hour_ranges(index[missing_oi], max_gap_hours=max_gap_hours)
    volume_ranges = contiguous_hour_ranges(index[volume_missing_for_fetch], max_gap_hours=max_gap_hours)
    return coverage, oi_ranges, volume_ranges


def aggregate_trades_to_hourly(
    trades: list[dict[str, Any]] | pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    source: str,
    fill_empty_hours: bool,
    empty_status: str = "confirmed_no_trades",
) -> pd.DataFrame:
    start = pd.Timestamp(start_ts).tz_convert("UTC") if pd.Timestamp(start_ts).tzinfo else pd.Timestamp(start_ts, tz="UTC")
    end = pd.Timestamp(end_ts).tz_convert("UTC") if pd.Timestamp(end_ts).tzinfo else pd.Timestamp(end_ts, tz="UTC")
    hours = pd.date_range(start.floor("h"), end.floor("h") - pd.Timedelta(hours=1), freq="1h", tz="UTC")
    columns = ["volume", "quote_volume", "trade_count", "vwap", "source", "coverage_status"]
    if len(hours) == 0:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC", name="ts"))

    if isinstance(trades, pd.DataFrame):
        raw = trades.copy()
    else:
        raw = pd.DataFrame(list(trades or []))
    if raw.empty:
        out = pd.DataFrame(index=hours, columns=columns)
        out.index.name = "ts"
        if fill_empty_hours:
            out["volume"] = 0.0
            out["quote_volume"] = 0.0
            out["trade_count"] = 0
            out["vwap"] = np.nan
            out["source"] = source
            out["coverage_status"] = empty_status
        return out

    if "timestamp" in raw.columns:
        ts = pd.to_datetime(pd.to_numeric(raw["timestamp"], errors="coerce"), unit="ms", utc=True)
    elif "datetime" in raw.columns:
        ts = pd.to_datetime(raw["datetime"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(raw.index, utc=True, errors="coerce")
    raw.index = pd.DatetimeIndex(ts, name="ts")
    raw = raw.loc[(raw.index >= start) & (raw.index < end)]
    if raw.empty:
        return aggregate_trades_to_hourly(
            [],
            start_ts=start,
            end_ts=end,
            source=source,
            fill_empty_hours=fill_empty_hours,
            empty_status=empty_status,
        )
    price = pd.to_numeric(raw.get("price"), errors="coerce")
    amount = pd.to_numeric(raw.get("amount"), errors="coerce")
    cost = (
        pd.to_numeric(raw.get("cost"), errors="coerce")
        if "cost" in raw.columns
        else price * amount
    )
    valid = price.gt(0) & amount.gt(0)
    raw = raw.loc[valid].copy()
    raw["amount"] = amount.loc[valid].astype("float64")
    raw["cost"] = cost.loc[valid].astype("float64")
    raw["price"] = price.loc[valid].astype("float64")
    if raw.empty:
        return aggregate_trades_to_hourly(
            [],
            start_ts=start,
            end_ts=end,
            source=source,
            fill_empty_hours=fill_empty_hours,
            empty_status=empty_status,
        )
    raw["trade_count"] = 1
    grouped = raw.resample("1h").agg(
        volume=("amount", "sum"),
        quote_volume=("cost", "sum"),
        trade_count=("trade_count", "sum"),
    )
    grouped = grouped.reindex(hours)
    grouped["vwap"] = grouped["quote_volume"] / grouped["volume"].replace(0.0, np.nan)
    grouped["source"] = source
    grouped["coverage_status"] = np.where(
        grouped["trade_count"].fillna(0).gt(0), "actual_trades", empty_status
    )
    if not fill_empty_hours:
        grouped = grouped[grouped["trade_count"].fillna(0).gt(0)]
    else:
        grouped["volume"] = grouped["volume"].fillna(0.0)
        grouped["quote_volume"] = grouped["quote_volume"].fillna(0.0)
        grouped["trade_count"] = grouped["trade_count"].fillna(0).astype("int32")
    grouped.index.name = "ts"
    return grouped.loc[:, columns]


def fetch_trades_paged(
    exchange: Any,
    symbol: str,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    limit: int = 1000,
    max_pages: int = 200,
    sleep_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], bool, str]:
    start_ms = int(pd.Timestamp(start_ts).value // 10**6)
    end_ms = int(pd.Timestamp(end_ts).value // 10**6)
    cursor = start_ms
    out: list[dict[str, Any]] = []
    last_seen = -1
    for _page in range(max(1, int(max_pages))):
        try:
            batch = exchange.fetch_trades(symbol, since=cursor, limit=int(limit))
        except Exception as exc:
            return out, False, f"{exc.__class__.__name__}: {exc}"
        if not batch:
            return out, True, "empty"
        advanced = False
        for trade in batch:
            try:
                ts = int(trade.get("timestamp"))
            except Exception:
                continue
            if ts < start_ms:
                continue
            if ts >= end_ms:
                return out, True, "complete"
            out.append(trade)
            if ts > last_seen:
                last_seen = ts
                advanced = True
        if not advanced:
            return out, False, "pagination_stalled"
        cursor = last_seen + 1
        time.sleep(max(0.0, float(sleep_seconds)))
    return out, False, "max_pages_exceeded"


def write_actual_volume_sidecar(path: Path, incoming: pd.DataFrame) -> tuple[int, int]:
    if incoming is None or incoming.empty:
        return 0, 0
    incoming = _coerce_utc_index(incoming)
    before = 0
    if path.exists():
        existing = _coerce_utc_index(pd.read_parquet(path))
        before = int(len(existing))
        merged = pd.concat([existing, incoming]).sort_index().groupby(level=0).last()
    else:
        merged = incoming
    for col in ("volume", "quote_volume", "trade_count", "vwap"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    for col in ("source", "coverage_status"):
        if col in merged.columns:
            merged[col] = merged[col].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return before, int(len(merged))
