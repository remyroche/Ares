import concurrent.futures
import fcntl
import gc as _gc
import glob
import json
import os
import shutil
import tempfile
import time
import zipfile
from datetime import timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import ccxt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter

from extreme_price_movements.utils import retry_with_backoff, tprint

SPOT_QUOTE_SUFFIXES = (
    "USDT",
    "USDC",
    "BUSD",
    "USD1",
    "FDUSD",
    "EUR",
    "BTC",
    "ETH",
)
BINANCE_PUBLIC_SPOT_DATA_BASE = "https://data.binance.vision/data/spot"
BINANCE_PUBLIC_UM_FUTURES_DATA_BASE = "https://data.binance.vision/data/futures/um"
BINANCE_PUBLIC_DATA_BASE = BINANCE_PUBLIC_SPOT_DATA_BASE
_ARCHIVE_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
)
HTTP_POOL_MAXSIZE = int(os.getenv("EPM_HTTP_POOL_MAXSIZE", "64") or "64")
HTTP_POOL_CONNECTIONS = int(os.getenv("EPM_HTTP_POOL_CONNECTIONS", "64") or "64")
_PUBLIC_DATA_SESSION: requests.Session | None = None


def _configure_requests_session_pool(session: Any) -> None:
    """Increase requests/urllib3 pool capacity for shared Binance clients."""
    if session is None or not hasattr(session, "mount"):
        return
    adapter = HTTPAdapter(
        pool_connections=max(1, HTTP_POOL_CONNECTIONS),
        pool_maxsize=max(1, HTTP_POOL_MAXSIZE),
        max_retries=0,
        pool_block=False,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)


def _configure_exchange_http_pool(exchange: Any) -> Any:
    """Configure ccxt's requests session pool when the sync client exposes it."""
    session = getattr(exchange, "session", None)
    _configure_requests_session_pool(session)
    return exchange


def _public_data_session() -> requests.Session:
    global _PUBLIC_DATA_SESSION
    if _PUBLIC_DATA_SESSION is None:
        _PUBLIC_DATA_SESSION = requests.Session()
        _configure_requests_session_pool(_PUBLIC_DATA_SESSION)
    return _PUBLIC_DATA_SESSION


def _download_public_zip_to_tmp(url: str) -> Optional[str]:
    """Download a zip URL to a temporary file with resilient TLS behavior."""
    last_exc: Exception | None = None
    tmp_path: Optional[str] = None
    for attempt, verify_ssl in enumerate((True, False), start=1):
        try:
            headers = {"User-Agent": _ARCHIVE_USER_AGENT}
            response = _public_data_session().get(
                url, stream=True, timeout=60, headers=headers, verify=verify_ssl
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            tmp_path = tmp.name
            with tmp:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp.write(chunk)
            return tmp_path
        except Exception as exc:  # pragma: no branch
            last_exc = exc
            mode = "strict" if verify_ssl else "insecure"
            tprint(
                f"WARN public archive download attempt {attempt} ({mode}) for {url}: {exc}"
            )
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                tmp_path = None
    if last_exc is not None:
        tprint(f"WARN public archive download exhausted retries for {url}: {last_exc}")
    return None


def _normalize_spot_symbol(symbol: str) -> str:
    norm = str(symbol or "").upper().strip().replace("_", "/")
    if not norm:
        return norm
    if "/" in norm:
        return norm
    for quote in SPOT_QUOTE_SUFFIXES:
        if norm.endswith(quote) and len(norm) > len(quote):
            return f"{norm[:-len(quote)]}/{quote}"
    return norm


def _load_dotenv_if_present(path: str = ".env") -> None:
    """Load simple KEY=VALUE env files without overwriting process env."""
    env_path = os.path.abspath(path)
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
    except Exception as exc:
        tprint(f"Could not load dotenv file {env_path}: {exc}")


def _load_local_env_if_present() -> None:
    """Load local dotenv files without overriding process env variables."""
    _load_dotenv_if_present(".env.local")
    _load_dotenv_if_present(".env")


def _symbol_alias_candidates(symbol: str) -> list[str]:
    canonical = _normalize_spot_symbol(symbol)
    raw = str(symbol or "").upper().strip()
    candidates: list[str] = []
    for candidate in (
        canonical,
        canonical.replace("/", "_"),
        canonical.replace("/", ""),
        raw,
        raw.replace("/", "_"),
        raw.replace("/", ""),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _normalize_feature_index(
    idx_vals,
    val_array=None,
):
    idx = pd.Index(idx_vals)
    if isinstance(idx, pd.DatetimeIndex):
        normalized = idx.tz_localize(None) if idx.tz is not None else idx
        return normalized, val_array, None

    if np.issubdtype(idx.dtype, np.number):
        return None, None, "numeric_index"

    if idx.dtype == object:
        safe_values = []
        for value in idx_vals:
            if isinstance(value, (int, np.integer, float, np.floating)) and not pd.isna(
                value
            ):
                safe_values.append(pd.NaT)
            else:
                safe_values.append(value)
        converted = pd.to_datetime(safe_values, utc=True, errors="coerce")
    else:
        converted = pd.to_datetime(idx_vals, utc=True, errors="coerce")
    valid_mask = ~pd.isna(converted)
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return None, None, "unparseable_index"

    normalized = pd.DatetimeIndex(converted[valid_mask]).tz_localize(None)
    if val_array is None:
        filtered_values = None
    else:
        filtered_values = np.asarray(val_array)[valid_mask]

    if valid_count < len(idx):
        return normalized, filtered_values, "partial_unparseable_index"
    return normalized, filtered_values, "coerced_index"


def _coerce_feature_values_float32(values) -> np.ndarray:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=np.float32, copy=False)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32, copy=False)


def _normalize_allowed_periods(
    allowed_periods,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    normalized: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if not allowed_periods:
        return normalized
    for period in allowed_periods:
        if isinstance(period, dict):
            start = period.get("start_ts") or period.get("start")
            end = period.get("end_ts") or period.get("end")
        elif isinstance(period, (list, tuple)) and len(period) >= 2:
            start, end = period[0], period[1]
        else:
            continue
        start_ts = pd.to_datetime(start, utc=True, errors="coerce")
        end_ts = pd.to_datetime(end, utc=True, errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
            continue
        normalized.append((start_ts.tz_localize(None), end_ts.tz_localize(None)))
    return normalized


def _apply_allowed_periods_mask(
    df: pd.DataFrame,
    allowed_periods,
) -> pd.DataFrame:
    periods = _normalize_allowed_periods(allowed_periods)
    if not periods or df.empty:
        return df
    idx = pd.to_datetime(df.index, utc=True, errors="coerce").tz_localize(None)
    mask = np.zeros(len(df), dtype=bool)
    for start_ts, end_ts in periods:
        mask |= (idx >= start_ts) & (idx < end_ts)
    if mask.any():
        return df.loc[mask]
    return df.iloc[0:0]


def _build_parquet_ts_filters(
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
    allowed_periods=None,
):
    """Build pyarrow parquet filters for timestamp pushdown when possible."""

    def _to_utc_filter_ts(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
        if ts is None:
            return None
        out = pd.Timestamp(ts)
        if out.tzinfo is None:
            out = out.tz_localize("UTC")
        else:
            out = out.tz_convert("UTC")
        return out

    periods = _normalize_allowed_periods(allowed_periods)
    start_ts = pd.Timestamp(start_ts) if start_ts is not None else None
    end_ts = pd.Timestamp(end_ts) if end_ts is not None else None
    start_ts = _to_utc_filter_ts(start_ts)
    end_ts = _to_utc_filter_ts(end_ts)

    if start_ts is not None or end_ts is not None:
        if not periods:
            periods = [(start_ts, end_ts)]
        else:
            clipped: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for period_start, period_end in periods:
                period_start = _to_utc_filter_ts(period_start)
                period_end = _to_utc_filter_ts(period_end)
                if start_ts is not None:
                    period_start = max(period_start, start_ts)
                if end_ts is not None:
                    period_end = min(period_end, end_ts)
                if (
                    period_end is not None
                    and period_start is not None
                    and period_end > period_start
                ):
                    clipped.append((period_start, period_end))
            periods = clipped

    if not periods:
        return None

    filters = []
    for period_start, period_end in periods:
        if period_start is None or period_end is None:
            continue
        filters.append(
            [
                ("ts", ">=", period_start.to_pydatetime()),
                ("ts", "<", period_end.to_pydatetime()),
            ]
        )
    return filters if filters else None


def _ensure_feature_frame_index(
    df: pd.DataFrame,
    parquet_path: Optional[str] = None,
) -> tuple[pd.DataFrame, str | None]:
    frame = df.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        return frame, None

    if "ts" in frame.columns:
        ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        valid_mask = ~pd.isna(ts)
        if not valid_mask.any():
            return frame, "invalid_ts_column"
        frame = frame.loc[valid_mask].copy()
        frame = frame.drop(columns=["ts"])
        frame.index = pd.DatetimeIndex(ts[valid_mask], tz="UTC")
        return frame, "ts_column_indexed"

    normalized_idx, _, reason = _normalize_feature_index(frame.index.values)
    if normalized_idx is None:
        if parquet_path is not None:
            recovered = _recover_feature_index_from_metadata(parquet_path, len(frame))
            if recovered is not None:
                frame.index = recovered
                return frame, "recovered_from_metadata"
        return frame, reason
    frame.index = pd.DatetimeIndex(normalized_idx, tz="UTC")
    return frame, reason


def _recover_feature_index_from_metadata(
    parquet_path: str,
    row_count: int,
) -> Optional[pd.DatetimeIndex]:
    meta = _read_feature_metadata(parquet_path)
    if not meta:
        return None
    first_ts = meta.get("first_ts")
    last_ts = meta.get("last_ts")
    expected_rows = int(meta.get("rows", 0) or 0)
    if not first_ts or not last_ts or expected_rows <= 0 or expected_rows != row_count:
        return None
    start = pd.Timestamp(first_ts)
    end = pd.Timestamp(last_ts)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    if row_count == 1:
        return pd.DatetimeIndex([start], tz="UTC")
    return pd.date_range(start=start, end=end, periods=row_count, tz="UTC")


class FileLock:
    """
    Simple file-based lock using fcntl for Unix-like systems.
    """

    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.handle = None

    def __enter__(self):
        try:
            self.handle = open(self.lock_file, "w")
            # Blocking exclusive lock
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        except Exception as e:
            tprint(f"Error acquiring lock {self.lock_file}: {e}")
            if self.handle:
                self.handle.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle:
            try:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
                self.handle.close()
            except Exception as e:
                tprint(f"Error releasing lock {self.lock_file}: {e}")


def _configured_exchange_id() -> str:
    raw = (
        os.environ.get("EPM_EXCHANGE")
        or os.environ.get("EXCHANGE_NAME")
        or os.environ.get("PRIMARY_EXCHANGE")
        or "binance"
    )
    exchange_id = str(raw or "binance").strip().lower()
    if exchange_id in {"okx", "okex"}:
        return "okx"
    if exchange_id in {"kraken", "krakenfutures", "kraken_futures"}:
        return "kraken"
    return "binance"


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _exchange_auth_config(exchange_id: str) -> Dict[str, Any]:
    if exchange_id == "okx":
        api_key = _env_first("OKX_API_KEY")
        api_secret = _env_first("OKX_API_SECRET", "OKX_SECRET_KEY")
        passphrase = _env_first("OKX_API_PASSPHRASE", "OKX_PASSPHRASE", "OKX_PASSWORD")
        auth: Dict[str, Any] = {}
        if api_key and api_secret:
            auth.update({"apiKey": api_key, "secret": api_secret})
            if passphrase:
                auth["password"] = passphrase
        return auth
    if exchange_id == "kraken":
        api_key = _env_first("KRAKEN_API_KEY")
        api_secret = _env_first("KRAKEN_API_SECRET")
        return {"apiKey": api_key, "secret": api_secret} if api_key and api_secret else {}
    api_key = _env_first("BINANCE_API_KEY")
    api_secret = _env_first("BINANCE_API_SECRET")
    return {"apiKey": api_key, "secret": api_secret} if api_key and api_secret else {}


def _perp_exchange_auth_config(exchange_id: str) -> Dict[str, Any]:
    if exchange_id == "kraken":
        api_key = _env_first("KRAKENFUTURES_API_KEY", "KRAKEN_API_KEY")
        api_secret = _env_first("KRAKENFUTURES_API_SECRET", "KRAKEN_API_SECRET")
        return {"apiKey": api_key, "secret": api_secret} if api_key and api_secret else {}
    return _exchange_auth_config(exchange_id)


def make_spot_exchange():
    _load_local_env_if_present()
    exchange_id = _configured_exchange_id()
    config: Dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": int(os.getenv("EPM_CCXT_TIMEOUT_MS", "20000") or "20000"),
        "options": {"fetchCurrencies": False},
    }
    config.update(_exchange_auth_config(exchange_id))
    if exchange_id == "okx":
        config["options"].update({"defaultType": "spot"})
        ex = _configure_exchange_http_pool(ccxt.okx(config))
    elif exchange_id == "kraken":
        ex = _configure_exchange_http_pool(ccxt.kraken(config))
    else:
        ex = _configure_exchange_http_pool(ccxt.binance(config))
    ex.load_markets()
    return ex


def make_perp_exchange():
    _load_local_env_if_present()
    exchange_id = _configured_exchange_id()
    config: Dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": int(os.getenv("EPM_CCXT_TIMEOUT_MS", "20000") or "20000"),
        "options": {"fetchCurrencies": False},
    }
    config.update(_perp_exchange_auth_config(exchange_id))
    if exchange_id == "okx":
        config["options"].update({"defaultType": "swap"})
        ex = _configure_exchange_http_pool(ccxt.okx(config))
    elif exchange_id == "kraken":
        ex = _configure_exchange_http_pool(ccxt.krakenfutures(config))
    else:
        config["options"].update({"defaultType": "future"})
        ex = _configure_exchange_http_pool(ccxt.binanceusdm(config))
    ex.load_markets()
    return ex


def _resolve_perp_symbol(exchange, spot_symbol: str) -> Optional[str]:
    if not spot_symbol or "/" not in spot_symbol:
        return None
    base, quote = spot_symbol.split("/", 1)
    quote = quote.split(":", 1)[0].upper()
    preferred_quotes = [quote]
    for fallback_quote in ("USD", "USDC", "USDT"):
        if fallback_quote != quote:
            preferred_quotes.append(fallback_quote)
    seen_quotes = set()
    candidates = []
    for perp_quote in preferred_quotes:
        if not perp_quote or perp_quote in seen_quotes:
            continue
        seen_quotes.add(perp_quote)
        candidates.extend(
            [
                f"{base}/{perp_quote}:{perp_quote}",
                f"{base}/{perp_quote}",
                f"{base}{perp_quote}",
            ]
        )
    for cand in candidates:
        if cand in getattr(exchange, "markets", {}):
            return cand
        if cand in getattr(exchange, "symbols", []):
            return cand
    return None


def _extract_float(row: dict, keys: List[str]) -> Optional[float]:
    if not isinstance(row, dict):
        return None
    info = row.get("info", {}) if isinstance(row.get("info"), dict) else {}
    for k in keys:
        val = row.get(k)
        if val is None:
            val = info.get(k)
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            continue
    return None


def _extract_timestamp_ms(row: dict) -> Optional[int]:
    if not isinstance(row, dict):
        return None
    info = row.get("info", {}) if isinstance(row.get("info"), dict) else {}
    for k in [
        "timestamp",
        "fundingTimestamp",
        "time",
        "transactTime",
        "calcTime",
        "fundingTime",
    ]:
        val = row.get(k)
        if val is None:
            val = info.get(k)
        if val is None:
            continue
        try:
            return int(val)
        except Exception:
            continue
    return None


def _fetch_ccxt_history_paged(
    fetch_fn: Callable,
    symbol: str,
    since_ms: int,
    until_ms: int,
    *,
    value_keys: list[str],
    exchange=None,
    timeframe: Optional[str] = None,
    limit: int = 1000,
) -> pd.Series:
    cursor = int(since_ms)
    rows: list[tuple[int, float]] = []

    while cursor < int(until_ms):
        try:
            if timeframe is None:
                batch = fetch_fn(symbol, since=cursor, limit=limit)
            else:
                batch = fetch_fn(symbol, timeframe=timeframe, since=cursor, limit=limit)
        except Exception as exc:
            tprint(f"WARN history fetch failed for {symbol}: {exc}")
            break

        if not batch:
            break

        max_seen = cursor
        for item in batch:
            ts = _extract_timestamp_ms(item)
            if ts is None:
                continue
            if ts <= max_seen:
                max_seen = max(max_seen, ts)
            else:
                max_seen = ts
            if ts < since_ms or ts >= until_ms:
                continue
            val = _extract_float(item, value_keys)
            if val is None or not np.isfinite(val):
                continue
            rows.append((ts, val))

        if max_seen <= cursor:
            break
        cursor = max_seen + 1
        if exchange is not None:
            time.sleep(float(getattr(exchange, "rateLimit", 100)) / 1000.0)

        if len(batch) < limit:
            # Provider returned fewer rows than requested; likely no more data.
            if cursor >= until_ms:
                break

    if not rows:
        return pd.Series(dtype=np.float32)

    df = pd.DataFrame(rows, columns=["ts", "value"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("h")
    out = df.groupby("ts")["value"].last().sort_index().astype(np.float32)
    return out


def _coerce_trade_side_sign(item: dict) -> float:
    info = item.get("info") if isinstance(item, dict) else None
    if isinstance(info, dict) and "m" in info:
        maker_flag = str(info.get("m")).strip().lower()
        if maker_flag in {"true", "1"}:
            return -1.0
        if maker_flag in {"false", "0"}:
            return 1.0
    side = str(item.get("side", "")).strip().lower() if isinstance(item, dict) else ""
    if side == "buy":
        return 1.0
    if side == "sell":
        return -1.0
    return 0.0


def _fetch_trade_history_paged(
    exchange,
    symbol: str,
    since_ms: int,
    until_ms: int,
    *,
    limit: int = 1000,
) -> pd.DataFrame:
    cursor = int(since_ms)
    rows: list[tuple[int, float, float, float]] = []

    while cursor < int(until_ms):
        try:
            batch = exchange.fetch_trades(symbol, since=cursor, limit=limit)
        except Exception as exc:
            tprint(f"WARN trade history fetch failed for {symbol}: {exc}")
            break

        if not batch:
            break

        max_seen = cursor
        for item in batch:
            ts = _extract_timestamp_ms(item)
            if ts is None:
                continue
            max_seen = max(max_seen, int(ts))
            if ts < since_ms or ts >= until_ms:
                continue
            price = _extract_float(item, ["price", "p"])
            amount = _extract_float(item, ["amount", "qty", "q"])
            if price is None or amount is None:
                continue
            if not np.isfinite(price) or not np.isfinite(amount) or amount <= 0.0:
                continue
            rows.append(
                (int(ts), float(price), float(amount), _coerce_trade_side_sign(item))
            )

        if max_seen <= cursor:
            break
        cursor = max_seen + 1
        time.sleep(float(getattr(exchange, "rateLimit", 100)) / 1000.0)
        if len(batch) < limit and cursor >= until_ms:
            break

    if not rows:
        return pd.DataFrame(columns=["price", "amount", "side_sign"]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="ts")
        )

    df = pd.DataFrame(rows, columns=["ts", "price", "amount", "side_sign"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["ts", "price", "amount", "side_sign"])
    df = df.set_index("ts").sort_index()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(np.float32)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(np.float32)
    df["side_sign"] = pd.to_numeric(df["side_sign"], errors="coerce").astype(np.float32)
    return df


def _iter_binance_public_aggtrade_archives(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[tuple[str, pd.Timestamp]]:
    compact = _normalize_spot_symbol(symbol).replace("/", "")
    start = pd.Timestamp(start_ts)
    end = pd.Timestamp(end_ts)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    start = start.floor("D")
    end = end.floor("D")
    current_month = pd.Timestamp.utcnow()
    if current_month.tzinfo is None:
        current_month = current_month.tz_localize("UTC")
    else:
        current_month = current_month.tz_convert("UTC")
    current_month = current_month.floor("D").replace(day=1)
    month_cursor = start.replace(day=1)
    specs: list[tuple[str, pd.Timestamp]] = []
    while month_cursor <= end:
        month_end = (month_cursor + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
        if month_cursor < current_month and month_end <= end:
            specs.append(("monthly", month_cursor))
        else:
            day_cursor = max(start, month_cursor)
            day_stop = min(end, month_end)
            while day_cursor <= day_stop:
                specs.append(("daily", day_cursor))
                day_cursor += pd.Timedelta(days=1)
        month_cursor = (month_cursor + pd.offsets.MonthBegin(1)).normalize()
    out = []
    for granularity, stamp in specs:
        out.append((granularity, pd.Timestamp(stamp), compact))
    return out


def _binance_public_aggtrade_url(
    compact_symbol: str,
    granularity: str,
    stamp: pd.Timestamp,
) -> str:
    if granularity == "monthly":
        suffix = stamp.strftime("%Y-%m")
        return (
            f"{BINANCE_PUBLIC_DATA_BASE}/monthly/aggTrades/{compact_symbol}/"
            f"{compact_symbol}-aggTrades-{suffix}.zip"
        )
    suffix = stamp.strftime("%Y-%m-%d")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/daily/aggTrades/{compact_symbol}/"
        f"{compact_symbol}-aggTrades-{suffix}.zip"
    )


def _read_binance_public_aggtrades_archive(
    url: str,
    start_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    tmp_path = None
    try:
        tmp_path = _download_public_zip_to_tmp(url)
        if not tmp_path:
            return pd.DataFrame()
        rows: list[pd.DataFrame] = []
        with zipfile.ZipFile(tmp_path) as zf:
            members = [name for name in zf.namelist() if name.endswith(".csv")]
            if not members:
                return pd.DataFrame()
            with zf.open(members[0]) as handle:
                reader = pd.read_csv(
                    handle,
                    header=None,
                    usecols=[1, 2, 5, 6],
                    names=["price", "amount", "ts_raw", "buyer_maker"],
                    chunksize=250_000,
                )
                for chunk in reader:
                    ts_raw = pd.to_numeric(chunk["ts_raw"], errors="coerce")
                    if ts_raw.isna().all():
                        continue
                    time_unit = "us" if float(ts_raw.max()) >= 1e14 else "ms"
                    ts = pd.to_datetime(
                        ts_raw.astype("int64"), unit=time_unit, utc=True
                    )
                    ts_ms = (
                        (ts.astype("int64") // 10**6)
                        if time_unit == "us"
                        else ts_raw.astype("int64")
                    )
                    mask = (ts_ms >= int(start_ms)) & (ts_ms < int(until_ms))
                    if not mask.any():
                        continue
                    part = pd.DataFrame(
                        {
                            "price": pd.to_numeric(
                                chunk.loc[mask, "price"], errors="coerce"
                            ).astype(np.float32),
                            "amount": pd.to_numeric(
                                chunk.loc[mask, "amount"], errors="coerce"
                            ).astype(np.float32),
                            "side_sign": np.where(
                                chunk.loc[mask, "buyer_maker"]
                                .astype(str)
                                .str.lower()
                                .isin({"true", "1"}),
                                -1.0,
                                1.0,
                            ).astype(np.float32),
                        },
                        index=ts[mask],
                    )
                    rows.append(part)
        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out
    except Exception as exc:
        tprint(f"WARN public aggTrades archive fetch failed {url}: {exc}")
        return pd.DataFrame()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _fetch_trade_history_public_aggtrades(
    symbol: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    if ":" in str(symbol or ""):
        return pd.DataFrame()
    start_ts = pd.to_datetime(since_ms, unit="ms", utc=True)
    end_ts = pd.to_datetime(until_ms - 1, unit="ms", utc=True)
    parts: list[pd.DataFrame] = []
    for granularity, stamp, compact in _iter_binance_public_aggtrade_archives(
        symbol, start_ts, end_ts
    ):
        url = _binance_public_aggtrade_url(compact, granularity, stamp)
        part = _read_binance_public_aggtrades_archive(url, since_ms, until_ms)
        if part is not None and not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _binance_public_kline_url(
    compact_symbol: str,
    granularity: str,
    stamp: pd.Timestamp,
    interval: str = "1h",
    data_base: str = BINANCE_PUBLIC_SPOT_DATA_BASE,
) -> str:
    if granularity == "monthly":
        suffix = stamp.strftime("%Y-%m")
        return (
            f"{data_base}/monthly/klines/{compact_symbol}/{interval}/"
            f"{compact_symbol}-{interval}-{suffix}.zip"
        )
    suffix = stamp.strftime("%Y-%m-%d")
    return (
        f"{data_base}/daily/klines/{compact_symbol}/{interval}/"
        f"{compact_symbol}-{interval}-{suffix}.zip"
    )


def _read_binance_public_kline_archive(
    url: str,
    start_ms: int,
    until_ms: int,
    interval: str = "1h",
) -> pd.DataFrame:
    tmp_path = None
    try:
        tmp_path = _download_public_zip_to_tmp(url)
        if not tmp_path:
            return pd.DataFrame()
        with zipfile.ZipFile(tmp_path) as zf:
            members = [name for name in zf.namelist() if name.endswith(".csv")]
            if not members:
                return pd.DataFrame()
            with zf.open(members[0]) as handle:
                df = pd.read_csv(
                    handle,
                    header=None,
                    usecols=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10],
                    names=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "quote_volume",
                        "trade_count",
                        "taker_buy_base",
                        "taker_buy_quote",
                    ],
                )
        open_time = pd.to_numeric(df["open_time"], errors="coerce")
        if open_time.isna().all():
            return pd.DataFrame()
        valid_open_time = open_time.notna()
        if not bool(valid_open_time.all()):
            df = df.loc[valid_open_time].copy()
            open_time = open_time.loc[valid_open_time]
        time_unit = "us" if float(open_time.max()) >= 1e14 else "ms"
        ts = pd.to_datetime(open_time.astype("int64"), unit=time_unit, utc=True)
        ts_ms = (
            (ts.astype("int64") // 10**6)
            if time_unit == "us"
            else open_time.astype("int64")
        )
        interval_ms = int(pd.Timedelta(interval).total_seconds() * 1000)
        snapshot_ms = ts_ms + interval_ms
        mask = (snapshot_ms >= int(start_ms)) & (snapshot_ms < int(until_ms))
        if not mask.any():
            return pd.DataFrame()
        out = df.loc[mask].copy()
        out.index = pd.to_datetime(snapshot_ms[mask], unit="ms", utc=True)
        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_base",
            "taker_buy_quote",
        ]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out
    except Exception as exc:
        tprint(f"WARN public kline archive fetch failed {url}: {exc}")
        return pd.DataFrame()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _fetch_orderbook_proxy_from_public_klines(
    symbol: str,
    since_ms: int,
    until_ms: int,
    *,
    data_base: str = BINANCE_PUBLIC_SPOT_DATA_BASE,
) -> pd.DataFrame:
    """Build hourly execution-friction proxies from Binance 1h kline summaries.

    Binance public spot archives do not provide historical depth snapshots. The
    fields emitted here intentionally preserve the existing orderbook_hourly
    schema, but values are proxy estimates derived from 1h OHLCV/taker-flow
    summaries rather than true L1/L10/L20 book state.
    """
    if ":" in str(symbol or ""):
        return pd.DataFrame()
    start_ts = pd.to_datetime(since_ms, unit="ms", utc=True)
    end_ts = pd.to_datetime(until_ms - 1, unit="ms", utc=True)
    compact = _normalize_spot_symbol(symbol).replace("/", "")
    parts: list[pd.DataFrame] = []
    for granularity, stamp, _ in _iter_binance_public_aggtrade_archives(
        symbol, start_ts, end_ts
    ):
        url = _binance_public_kline_url(
            compact,
            granularity,
            stamp,
            interval="1h",
            data_base=data_base,
        )
        part = _read_binance_public_kline_archive(
            url, since_ms, until_ms, interval="1h"
        )
        if part is not None and not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    bars = pd.concat(parts).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    mid = pd.to_numeric(bars["close"], errors="coerce")
    volume = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
    quote_volume = pd.to_numeric(bars["quote_volume"], errors="coerce").fillna(0.0)
    trade_count = pd.to_numeric(bars["trade_count"], errors="coerce").fillna(0.0)
    buy_qty = pd.to_numeric(bars["taker_buy_base"], errors="coerce").fillna(0.0)
    buy_notional = pd.to_numeric(bars["taker_buy_quote"], errors="coerce").fillna(0.0)
    sell_qty = (volume - buy_qty).clip(lower=0.0)
    sell_notional = (quote_volume - buy_notional).clip(lower=0.0)
    eps = 1e-9
    imbalance = ((buy_qty - sell_qty) / (volume + eps)).clip(-1.0, 1.0)
    mean_trade_qty = (volume / trade_count.replace(0.0, np.nan)).fillna(0.0)
    hl_spread_bps = (
        (
            (
                pd.to_numeric(bars["high"], errors="coerce")
                - pd.to_numeric(bars["low"], errors="coerce")
            )
            / (mid.abs() + eps)
        )
        * 1e4
    ).fillna(0.0)
    spread_bps = (
        1.0 + 0.20 * hl_spread_bps + 0.05 * np.sqrt(trade_count.clip(lower=0.0))
    ).clip(lower=1.0, upper=35.0)
    half_spread = spread_bps / 2e4
    base_depth = np.maximum(
        volume / 6.0,
        np.maximum(mean_trade_qty * 12.0, trade_count * mean_trade_qty * 0.15),
    )
    side_skew = (0.85 * imbalance).clip(-0.9, 0.9)
    cum_bid_qty_l20 = (base_depth * (1.0 + side_skew)).clip(lower=0.0)
    cum_ask_qty_l20 = (base_depth * (1.0 - side_skew)).clip(lower=0.0)
    cum_bid_qty_l10 = (cum_bid_qty_l20 * 0.55).clip(lower=0.0)
    cum_ask_qty_l10 = (cum_ask_qty_l20 * 0.55).clip(lower=0.0)
    bid_qty_1 = np.maximum(
        mean_trade_qty * (1.0 + 0.50 * imbalance), cum_bid_qty_l10 / 10.0
    )
    ask_qty_1 = np.maximum(
        mean_trade_qty * (1.0 - 0.50 * imbalance), cum_ask_qty_l10 / 10.0
    )

    out = pd.DataFrame(index=bars.index)
    out["best_bid"] = (mid * (1.0 - half_spread)).astype(np.float32)
    out["best_ask"] = (mid * (1.0 + half_spread)).astype(np.float32)
    out["mid"] = mid.astype(np.float32)
    out["bid_qty_1"] = pd.Series(bid_qty_1, index=bars.index).astype(np.float32)
    out["ask_qty_1"] = pd.Series(ask_qty_1, index=bars.index).astype(np.float32)
    out["cum_bid_qty_l10"] = pd.Series(cum_bid_qty_l10, index=bars.index).astype(
        np.float32
    )
    out["cum_ask_qty_l10"] = pd.Series(cum_ask_qty_l10, index=bars.index).astype(
        np.float32
    )
    out["cum_bid_qty_l20"] = pd.Series(cum_bid_qty_l20, index=bars.index).astype(
        np.float32
    )
    out["cum_ask_qty_l20"] = pd.Series(cum_ask_qty_l20, index=bars.index).astype(
        np.float32
    )
    out["snapshot_ts"] = bars.index
    out["trade_count_1h"] = trade_count.astype(np.int32)
    out["buy_qty_1h"] = buy_qty.astype(np.float32)
    out["sell_qty_1h"] = sell_qty.astype(np.float32)
    out["notional_1h"] = quote_volume.astype(np.float32)
    out["buy_notional_1h"] = buy_notional.astype(np.float32)
    out["sell_notional_1h"] = sell_notional.astype(np.float32)
    out["vwap_1h"] = (
        (quote_volume / volume.replace(0.0, np.nan)).fillna(mid).astype(np.float32)
    )
    out["mean_trade_qty_1h"] = mean_trade_qty.astype(np.float32)
    out["signed_flow_imbalance_1h"] = imbalance.astype(np.float32)
    out["source"] = "kline_summary"
    return out.replace([np.inf, -np.inf], np.nan)


def _compute_missing_hourly_ranges(
    existing_index,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_hour = pd.Timestamp(start_ts)
    end_hour = pd.Timestamp(end_ts)
    if start_hour.tzinfo is None:
        start_hour = start_hour.tz_localize("UTC")
    else:
        start_hour = start_hour.tz_convert("UTC")
    if end_hour.tzinfo is None:
        end_hour = end_hour.tz_localize("UTC")
    else:
        end_hour = end_hour.tz_convert("UTC")
    start_hour = start_hour.floor("h")
    end_hour = end_hour.floor("h")
    if end_hour < start_hour:
        return []

    expected = pd.date_range(start=start_hour, end=end_hour, freq="1h", tz="UTC")
    if len(expected) == 0:
        return []
    if existing_index is None:
        existing = pd.DatetimeIndex([], tz="UTC")
    else:
        existing = pd.to_datetime(existing_index, utc=True, errors="coerce")
        existing = pd.DatetimeIndex(existing[~pd.isna(existing)]).floor("h").unique()
    missing = expected.difference(existing)
    if len(missing) == 0:
        return []

    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    range_start = missing[0]
    prev = missing[0]
    step = pd.Timedelta(hours=1)
    for ts in missing[1:]:
        if ts - prev > step:
            ranges.append((range_start, prev + step))
            range_start = ts
        prev = ts
    ranges.append((range_start, prev + step))
    return ranges


def _compute_missing_funding_ranges(
    existing_index,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    cadence_hours: int = 8,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return contiguous windows missing native funding timestamps.

    Funding history is sparse by design. Binance spot/perp funding is normally
    published every 8 hours, so treating non-funding hours as missing creates
    thousands of false gaps and expensive no-op API calls.
    """
    start_hour = pd.Timestamp(start_ts)
    end_hour = pd.Timestamp(end_ts)
    if start_hour.tzinfo is None:
        start_hour = start_hour.tz_localize("UTC")
    else:
        start_hour = start_hour.tz_convert("UTC")
    if end_hour.tzinfo is None:
        end_hour = end_hour.tz_localize("UTC")
    else:
        end_hour = end_hour.tz_convert("UTC")
    start_hour = start_hour.floor(f"{int(cadence_hours)}h")
    end_hour = end_hour.floor(f"{int(cadence_hours)}h")
    if end_hour < start_hour:
        return []

    step = pd.Timedelta(hours=int(cadence_hours))
    expected = pd.date_range(start=start_hour, end=end_hour, freq=step, tz="UTC")
    if len(expected) == 0:
        return []
    if existing_index is None:
        return [(start_hour, end_hour + step)]

    existing = pd.to_datetime(existing_index, utc=True, errors="coerce")
    existing = pd.DatetimeIndex(existing[~pd.isna(existing)]).floor(
        f"{int(cadence_hours)}h"
    )
    existing = existing.unique()
    missing = expected.difference(existing)
    if len(missing) == 0:
        return []

    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    range_start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        if ts - prev > step:
            ranges.append((range_start, prev + step))
            range_start = ts
        prev = ts
    ranges.append((range_start, prev + step))
    return ranges


def _build_hourly_orderbook_proxy_from_trades(
    trades: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    min_spread_bps: float = 1.0,
    max_spread_bps: float = 35.0,
) -> pd.DataFrame:
    start_hour = pd.Timestamp(start_ts)
    end_hour = pd.Timestamp(end_ts)
    if start_hour.tzinfo is None:
        start_hour = start_hour.tz_localize("UTC")
    else:
        start_hour = start_hour.tz_convert("UTC")
    if end_hour.tzinfo is None:
        end_hour = end_hour.tz_localize("UTC")
    else:
        end_hour = end_hour.tz_convert("UTC")
    start_hour = start_hour.floor("h")
    end_hour = end_hour.floor("h")
    hour_index = pd.date_range(start=start_hour, end=end_hour, freq="1h", tz="UTC")
    if len(hour_index) == 0:
        return pd.DataFrame()

    if trades is None or trades.empty:
        return pd.DataFrame(index=hour_index)

    tr = trades.copy().sort_index()
    tr.index = pd.to_datetime(tr.index, utc=True, errors="coerce")
    tr = tr[~tr.index.duplicated(keep="last")]
    tr["notional"] = (tr["price"] * tr["amount"]).astype(np.float32)
    tr["buy_qty"] = np.where(tr["side_sign"] > 0, tr["amount"], 0.0).astype(np.float32)
    tr["sell_qty"] = np.where(tr["side_sign"] < 0, tr["amount"], 0.0).astype(np.float32)
    tr["buy_notional"] = np.where(tr["side_sign"] > 0, tr["notional"], 0.0).astype(
        np.float32
    )
    tr["sell_notional"] = np.where(tr["side_sign"] < 0, tr["notional"], 0.0).astype(
        np.float32
    )
    tr["trade_count"] = np.float32(1.0)

    agg = (
        tr.resample("1h", label="right", closed="left")
        .agg(
            last_price=("price", "last"),
            last_trade_ts=("price", lambda s: s.index.max() if len(s) else pd.NaT),
            mean_price=("price", "mean"),
            price_std=("price", "std"),
            amount_sum=("amount", "sum"),
            notional_sum=("notional", "sum"),
            buy_qty=("buy_qty", "sum"),
            sell_qty=("sell_qty", "sum"),
            buy_notional=("buy_notional", "sum"),
            sell_notional=("sell_notional", "sum"),
            trade_count=("trade_count", "sum"),
        )
        .reindex(hour_index)
    )
    agg["last_price"] = agg["last_price"].ffill()
    agg["mean_price"] = agg["mean_price"].fillna(agg["last_price"])
    agg["last_trade_ts"] = pd.to_datetime(
        agg["last_trade_ts"], utc=True, errors="coerce"
    )
    agg["price_std"] = pd.to_numeric(agg["price_std"], errors="coerce").fillna(0.0)
    agg["amount_sum"] = pd.to_numeric(agg["amount_sum"], errors="coerce").fillna(0.0)
    agg["notional_sum"] = pd.to_numeric(agg["notional_sum"], errors="coerce").fillna(
        0.0
    )
    agg["buy_qty"] = pd.to_numeric(agg["buy_qty"], errors="coerce").fillna(0.0)
    agg["sell_qty"] = pd.to_numeric(agg["sell_qty"], errors="coerce").fillna(0.0)
    agg["buy_notional"] = pd.to_numeric(agg["buy_notional"], errors="coerce").fillna(
        0.0
    )
    agg["sell_notional"] = pd.to_numeric(agg["sell_notional"], errors="coerce").fillna(
        0.0
    )
    agg["trade_count"] = pd.to_numeric(agg["trade_count"], errors="coerce").fillna(0.0)

    eps = 1e-9
    mid = pd.to_numeric(agg["last_price"], errors="coerce")
    total_qty = agg["buy_qty"] + agg["sell_qty"]
    imbalance = ((agg["buy_qty"] - agg["sell_qty"]) / (total_qty + eps)).clip(-1.0, 1.0)
    mean_trade_qty = (total_qty / agg["trade_count"].replace(0.0, np.nan)).fillna(0.0)
    price_dispersion_bps = (agg["price_std"] / (mid.abs() + eps) * 1e4).fillna(0.0)
    flow_bps = (
        (
            (agg["buy_notional"] - agg["sell_notional"]).abs()
            / (agg["notional_sum"] + eps)
        )
        * 8.0
    ).fillna(0.0)
    spread_bps = (
        pd.Series(min_spread_bps, index=agg.index, dtype=np.float32)
        + 0.50 * price_dispersion_bps
        + flow_bps
    ).clip(lower=min_spread_bps, upper=max_spread_bps)
    half_spread = spread_bps / 2e4

    base_depth = np.maximum(
        total_qty / 6.0,
        np.maximum(mean_trade_qty * 12.0, agg["trade_count"] * mean_trade_qty * 0.15),
    )
    side_skew = (0.85 * imbalance).clip(-0.9, 0.9)
    cum_bid_qty_l20 = (base_depth * (1.0 + side_skew)).clip(lower=0.0)
    cum_ask_qty_l20 = (base_depth * (1.0 - side_skew)).clip(lower=0.0)
    cum_bid_qty_l10 = (cum_bid_qty_l20 * 0.55).clip(lower=0.0)
    cum_ask_qty_l10 = (cum_ask_qty_l20 * 0.55).clip(lower=0.0)
    bid_qty_1 = np.maximum(
        mean_trade_qty * (1.0 + 0.50 * imbalance), cum_bid_qty_l10 / 10.0
    )
    ask_qty_1 = np.maximum(
        mean_trade_qty * (1.0 - 0.50 * imbalance), cum_ask_qty_l10 / 10.0
    )

    out = pd.DataFrame(index=hour_index)
    out["best_bid"] = (mid * (1.0 - half_spread)).astype(np.float32)
    out["best_ask"] = (mid * (1.0 + half_spread)).astype(np.float32)
    out["mid"] = mid.astype(np.float32)
    out["bid_qty_1"] = pd.Series(bid_qty_1, index=hour_index).astype(np.float32)
    out["ask_qty_1"] = pd.Series(ask_qty_1, index=hour_index).astype(np.float32)
    out["cum_bid_qty_l10"] = pd.Series(cum_bid_qty_l10, index=hour_index).astype(
        np.float32
    )
    out["cum_ask_qty_l10"] = pd.Series(cum_ask_qty_l10, index=hour_index).astype(
        np.float32
    )
    out["cum_bid_qty_l20"] = pd.Series(cum_bid_qty_l20, index=hour_index).astype(
        np.float32
    )
    out["cum_ask_qty_l20"] = pd.Series(cum_ask_qty_l20, index=hour_index).astype(
        np.float32
    )
    out["snapshot_ts"] = agg["last_trade_ts"]
    out["trade_count_1h"] = agg["trade_count"].astype(np.int32)
    out["buy_qty_1h"] = agg["buy_qty"].astype(np.float32)
    out["sell_qty_1h"] = agg["sell_qty"].astype(np.float32)
    out["notional_1h"] = agg["notional_sum"].astype(np.float32)
    out["buy_notional_1h"] = agg["buy_notional"].astype(np.float32)
    out["sell_notional_1h"] = agg["sell_notional"].astype(np.float32)
    out["vwap_1h"] = (
        (agg["notional_sum"] / (agg["amount_sum"].replace(0.0, np.nan) + eps))
        .fillna(mid)
        .astype(np.float32)
    )
    out["mean_trade_qty_1h"] = mean_trade_qty.astype(np.float32)
    out["signed_flow_imbalance_1h"] = imbalance.astype(np.float32)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["mid", "best_bid", "best_ask"], how="all")
    return out


def normalize_orderbook_proxy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep historical and live orderbook proxy rows schema-compatible and finite."""
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out[~out.index.isna()].sort_index()

    if "snapshot_ts" not in out.columns:
        out["snapshot_ts"] = out.index
    else:
        snap = pd.to_datetime(out["snapshot_ts"], utc=True, errors="coerce")
        out["snapshot_ts"] = snap.where(snap.notna(), out.index)
    if "source" not in out.columns:
        out["source"] = "kline_summary"
    else:
        out["source"] = out["source"].fillna("kline_summary").astype(str)

    if "mid" in out.columns:
        mid = pd.to_numeric(out["mid"], errors="coerce")
    elif {"best_bid", "best_ask"}.issubset(out.columns):
        mid = (
            pd.to_numeric(out["best_bid"], errors="coerce")
            + pd.to_numeric(out["best_ask"], errors="coerce")
        ) / 2.0
        out["mid"] = mid
    else:
        mid = pd.Series(0.0, index=out.index, dtype=np.float32)
        out["mid"] = mid

    zero_fill_cols = [
        "trade_count_1h",
        "buy_qty_1h",
        "sell_qty_1h",
        "notional_1h",
        "buy_notional_1h",
        "sell_notional_1h",
        "mean_trade_qty_1h",
        "signed_flow_imbalance_1h",
    ]
    for col in zero_fill_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    if "vwap_1h" not in out.columns:
        out["vwap_1h"] = mid
    out["vwap_1h"] = (
        pd.to_numeric(out["vwap_1h"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(mid)
        .fillna(0.0)
        .astype(np.float32)
    )

    for col in [c for c in out.columns if c not in {"snapshot_ts", "source"}]:
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
    return out


def fetch_hourly_orderbook_proxy(
    exchange,
    symbol: str,
    since_ms: int,
    until_ms: int,
    *,
    limit: int = 1000,
) -> pd.DataFrame:
    exchange_id = str(getattr(exchange, "id", "") or "").lower()
    data_base = (
        BINANCE_PUBLIC_UM_FUTURES_DATA_BASE
        if "usdm" in exchange_id or "future" in exchange_id
        else BINANCE_PUBLIC_SPOT_DATA_BASE
    )
    public_klines = _fetch_orderbook_proxy_from_public_klines(
        symbol,
        int(since_ms),
        int(until_ms),
        data_base=data_base,
    )
    if public_klines is not None and not public_klines.empty:
        return normalize_orderbook_proxy_frame(public_klines)

    # Historical microstructure proxy must stay summary-based. Do not fall back
    # to aggTrades/trade pagination here; that path is too heavy for universe
    # backfills and is not needed for hourly snapshot features.
    return pd.DataFrame()


def build_hourly_orderbook_proxy_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback execution-friction proxy from local hourly OHLCV only.

    This is less informative than Binance public kline summaries because local
    OHLCV does not always include taker-flow or trade-count fields, but it keeps
    orderbook proxy features finite for perps listings whose public archive is
    unavailable or incomplete.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    bars = df.copy()
    if not isinstance(bars.index, pd.DatetimeIndex):
        if "ts" in bars.columns:
            bars.index = pd.to_datetime(bars["ts"], utc=True, errors="coerce")
        else:
            bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    elif bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars.index = bars.index.tz_convert("UTC")
    bars = bars[~bars.index.isna()].sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(bars.columns)):
        return pd.DataFrame()

    mid = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    volume = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
    quote_volume = pd.to_numeric(
        bars.get("quote_volume", volume * mid), errors="coerce"
    ).fillna(volume * mid)
    trade_count = pd.to_numeric(
        bars.get("trade_count", np.sqrt(volume.clip(lower=0.0)) + 1.0),
        errors="coerce",
    ).fillna(1.0)
    buy_qty = pd.to_numeric(bars.get("taker_buy_base", volume * 0.5), errors="coerce")
    buy_qty = buy_qty.fillna(volume * 0.5).clip(lower=0.0)
    buy_notional = pd.to_numeric(
        bars.get("taker_buy_quote", quote_volume * 0.5), errors="coerce"
    )
    buy_notional = buy_notional.fillna(quote_volume * 0.5).clip(lower=0.0)
    sell_qty = (volume - buy_qty).clip(lower=0.0)
    sell_notional = (quote_volume - buy_notional).clip(lower=0.0)

    eps = 1e-9
    imbalance = ((buy_qty - sell_qty) / (volume + eps)).clip(-1.0, 1.0)
    mean_trade_qty = (volume / trade_count.replace(0.0, np.nan)).fillna(0.0)
    hl_spread_bps = (((high - low) / (mid.abs() + eps)) * 1e4).fillna(0.0)
    spread_bps = (
        1.0 + 0.20 * hl_spread_bps + 0.05 * np.sqrt(trade_count.clip(lower=0.0))
    ).clip(lower=1.0, upper=35.0)
    half_spread = spread_bps / 2e4
    base_depth = np.maximum(
        volume / 6.0,
        np.maximum(mean_trade_qty * 12.0, trade_count * mean_trade_qty * 0.15),
    )
    side_skew = (0.85 * imbalance).clip(-0.9, 0.9)
    cum_bid_qty_l20 = (base_depth * (1.0 + side_skew)).clip(lower=0.0)
    cum_ask_qty_l20 = (base_depth * (1.0 - side_skew)).clip(lower=0.0)
    cum_bid_qty_l10 = (cum_bid_qty_l20 * 0.55).clip(lower=0.0)
    cum_ask_qty_l10 = (cum_ask_qty_l20 * 0.55).clip(lower=0.0)
    bid_qty_1 = np.maximum(
        mean_trade_qty * (1.0 + 0.50 * imbalance), cum_bid_qty_l10 / 10.0
    )
    ask_qty_1 = np.maximum(
        mean_trade_qty * (1.0 - 0.50 * imbalance), cum_ask_qty_l10 / 10.0
    )

    out = pd.DataFrame(index=bars.index)
    out["best_bid"] = (mid * (1.0 - half_spread)).astype(np.float32)
    out["best_ask"] = (mid * (1.0 + half_spread)).astype(np.float32)
    out["mid"] = mid.astype(np.float32)
    out["bid_qty_1"] = pd.Series(bid_qty_1, index=bars.index).astype(np.float32)
    out["ask_qty_1"] = pd.Series(ask_qty_1, index=bars.index).astype(np.float32)
    out["cum_bid_qty_l10"] = pd.Series(cum_bid_qty_l10, index=bars.index).astype(np.float32)
    out["cum_ask_qty_l10"] = pd.Series(cum_ask_qty_l10, index=bars.index).astype(np.float32)
    out["cum_bid_qty_l20"] = pd.Series(cum_bid_qty_l20, index=bars.index).astype(np.float32)
    out["cum_ask_qty_l20"] = pd.Series(cum_ask_qty_l20, index=bars.index).astype(np.float32)
    out["snapshot_ts"] = bars.index
    out["trade_count_1h"] = trade_count.astype(np.float32)
    out["buy_qty_1h"] = buy_qty.astype(np.float32)
    out["sell_qty_1h"] = sell_qty.astype(np.float32)
    out["notional_1h"] = quote_volume.astype(np.float32)
    out["buy_notional_1h"] = buy_notional.astype(np.float32)
    out["sell_notional_1h"] = sell_notional.astype(np.float32)
    out["vwap_1h"] = (
        (quote_volume / volume.replace(0.0, np.nan)).fillna(mid).astype(np.float32)
    )
    out["mean_trade_qty_1h"] = mean_trade_qty.astype(np.float32)
    out["signed_flow_imbalance_1h"] = imbalance.astype(np.float32)
    out["source"] = "local_ohlcv_summary"
    return normalize_orderbook_proxy_frame(out.replace([np.inf, -np.inf], np.nan))


@retry_with_backoff(retries=3, backoff_in_seconds=2)
def _fetch_ohlcv_paged(
    exchange,
    symbol,
    since_ms,
    until_ms,
    timeframe="1h",
    limit=1000,
    params: Optional[dict] = None,
):
    # Reduced logging: entry log removed
    out = []
    since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
            params=params or {},
        )
        if not batch:
            break
        for row in batch:
            ts = row[0]
            if ts < since_ms:
                continue
            if ts >= until_ms:
                break
            out.append(row)

        last = batch[-1][0]
        if last >= until_ms - 1:
            break
        since = last + 1
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    if not out:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume"]
        ).set_index(pd.DatetimeIndex([], tz="UTC", name="ts"))

    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    return df


def fetch_ohlcv_all_7d_chunks(
    exchange,
    symbol,
    since_ms,
    timeframe="1h",
    limit=1000,
    params: Optional[dict] = None,
):
    chunk_ms = int(pd.Timedelta(days=7).total_seconds() * 1000)
    now_ms = int(pd.Timestamp.utcnow().value // 10**6)

    start = since_ms
    while start < now_ms:
        end = min(start + chunk_ms, now_ms)
        df = _fetch_ohlcv_paged(
            exchange,
            symbol,
            start,
            end,
            timeframe=timeframe,
            limit=limit,
            params=params,
        )
        if len(df):
            yield df
        start = end
        time.sleep(exchange.rateLimit / 1000)


def _recent_history_floor_ms(env_name: str, default_days: float) -> int:
    days = float(os.getenv(env_name, str(default_days)) or default_days)
    floor_ts = pd.Timestamp.utcnow() - pd.Timedelta(days=max(days, 0.0))
    return int(floor_ts.value // 10**6)


class PartitionedOHLCVStore:
    def __init__(self, root_dir="data", timeframe="1h"):
        self.root_dir = root_dir
        self.timeframe = timeframe
        self.ohlcv_dir = os.path.join(root_dir, "ohlcv")
        os.makedirs(self.ohlcv_dir, exist_ok=True)

    def _get_symbol_dir(self, symbol: str) -> str:
        canonical = _normalize_spot_symbol(symbol)
        for candidate in _symbol_alias_candidates(canonical):
            safe_sym = candidate.replace("/", "_")
            path = os.path.join(self.ohlcv_dir, f"symbol={safe_sym}")
            if os.path.exists(path):
                return path
        safe_sym = canonical.replace("/", "_")
        return os.path.join(self.ohlcv_dir, f"symbol={safe_sym}")

    def _get_meta_path(self, symbol: str) -> str:
        canonical = _normalize_spot_symbol(symbol)
        for candidate in _symbol_alias_candidates(canonical):
            safe_sym = candidate.replace("/", "_")
            path = os.path.join(self.ohlcv_dir, f"{safe_sym}.meta.json")
            if os.path.exists(path):
                return path
        safe_sym = canonical.replace("/", "_")
        return os.path.join(self.ohlcv_dir, f"{safe_sym}.meta.json")

    def _read_meta(self, symbol: str) -> dict:
        path = self._get_meta_path(symbol)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _write_meta(self, symbol: str, meta: dict):
        path = self._get_meta_path(symbol)
        try:
            existing = {}
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        existing = json.load(f)
                except Exception:
                    existing = {}
            existing.update(meta)
            with open(path, "w") as f:
                json.dump(existing, f)
        except Exception as e:
            tprint(f"Error writing meta for {symbol}: {e}")

    def _downcast(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "funding_rate",
            "open_interest",
            "spot_close",
            "spot_open",
            "spot_high",
            "spot_low",
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
        ]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
        return out

    def load(
        self, symbol: str, columns=None, start_ts=None, end_ts=None
    ) -> pd.DataFrame:
        """
        Load data for symbol with optimized file filtering.
        columns: list of columns to read (optimization).
        start_ts: Optional[pd.Timestamp] - inclusive start
        end_ts: Optional[pd.Timestamp] - inclusive end
        """
        sym_dir = self._get_symbol_dir(symbol)
        if not os.path.exists(sym_dir):
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            ).set_index(pd.DatetimeIndex([], tz="UTC", name="ts"))

        try:
            # 1. Gather and filter files by timestamp BEFORE reading
            files_to_read = []
            s_ts_sec = int(start_ts.timestamp()) if start_ts else 0
            e_ts_sec = int(end_ts.timestamp()) if end_ts else 2**63 - 1

            for root, dirs, files in os.walk(sym_dir):
                for f in files:
                    if not f.endswith(".parquet"):
                        continue

                    fpath = os.path.join(root, f)

                    # Parse filename for timestamp range
                    if start_ts is not None or end_ts is not None:
                        try:
                            base = f.replace(".parquet", "")
                            parts = base.split("-")
                            if len(parts) >= 3:
                                f_min = int(parts[-2])
                                f_max = int(parts[-1])

                                # Check overlap: [f_min, f_max] ∩ [s_ts_sec, e_ts_sec]
                                if f_min > e_ts_sec or f_max < s_ts_sec:
                                    continue  # Skip non-overlapping files
                        except (ValueError, IndexError):
                            pass  # Include file if parsing fails

                    files_to_read.append(fpath)

            if not files_to_read:
                return pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                ).set_index(pd.DatetimeIndex([], tz="UTC", name="ts"))

            # 2. Read only filtered files with column selection
            read_cols = None
            if columns:
                read_cols = list(columns)
                if "ts" not in read_cols:
                    read_cols.append("ts")

            # Read multiple parquet files - handle categorical type mismatches
            # This fixes "Unable to merge: Field symbol has incompatible types" error
            # that occurs when different parquet files have different categorical index types
            try:
                # Try with pandas default first
                df = pd.read_parquet(files_to_read, columns=read_cols)

                # Convert any categorical columns to string to avoid downstream issues
                for col in df.columns:
                    if df[col].dtype.name == "category":
                        df[col] = df[col].astype(str)
            except Exception as e:
                # Fallback: read files one by one and concatenate
                dfs = []
                for fpath in files_to_read:
                    try:
                        part_df = pd.read_parquet(fpath, columns=read_cols)
                        # Convert categorical columns to string
                        for col in part_df.columns:
                            if part_df[col].dtype.name == "category":
                                part_df[col] = part_df[col].astype(str)
                        dfs.append(part_df)
                    except Exception:
                        continue

                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                else:
                    df = (
                        pd.DataFrame(columns=read_cols) if read_cols else pd.DataFrame()
                    )

            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                df = df.set_index("ts")
            elif df.index.name == "ts":
                pass

            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            # 4. Final slice to exact range (since files are coarse)
            if start_ts is not None:
                df = df[df.index >= start_ts]
            if end_ts is not None:
                df = df[df.index <= end_ts]

            return self._downcast(df)
        except Exception as e:
            tprint(f"Error loading {symbol}: {e}")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            ).set_index(pd.DatetimeIndex([], tz="UTC", name="ts"))

    def save_partitioned(
        self, symbol: str, df: pd.DataFrame, defer_compact: bool = False
    ):
        # Safely check if df is a valid DataFrame
        try:
            if (
                df is None
                or not isinstance(df, (pd.DataFrame, pd.Series))
                or (hasattr(df, "empty") and df.empty)
            ):
                return
        except Exception:
            return

        df = self._downcast(df)
        df_reset = df.reset_index().rename(columns={"index": "ts"})
        if "ts" not in df_reset.columns:
            df_reset = df.reset_index()
            if df_reset.columns[0] != "ts":
                df_reset.rename(columns={df_reset.columns[0]: "ts"}, inplace=True)

        df_reset["ts"] = pd.to_datetime(df_reset["ts"], utc=True)
        for c in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "funding_rate",
            "open_interest",
            "spot_close",
            "spot_open",
            "spot_high",
            "spot_low",
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
        ]:
            if c in df_reset.columns:
                df_reset[c] = df_reset[c].astype(np.float32)

        df_reset["year"] = df_reset["ts"].dt.year

        sym_dir = self._get_symbol_dir(symbol)

        for year, group in df_reset.groupby("year"):
            part_dir = os.path.join(sym_dir, f"year={year}")
            os.makedirs(part_dir, exist_ok=True)

            ts_min = int(group["ts"].min().value // 10**9)
            ts_max = int(group["ts"].max().value // 10**9)
            fname = f"part-{ts_min}-{ts_max}.parquet"
            fpath = os.path.join(part_dir, fname)

            write_df = group.drop(columns=["year"])
            write_df.to_parquet(
                fpath, index=False, engine="pyarrow", compression="zstd"
            )

            if not defer_compact:
                self.compact_partition(symbol, year)

    def compact_partition(self, symbol: str, year: int):
        sym_dir = self._get_symbol_dir(symbol)
        part_dir = os.path.join(sym_dir, f"year={year}")

        if not os.path.exists(part_dir):
            return

        files = glob.glob(os.path.join(part_dir, "*.parquet"))
        if not files:
            return

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as e:
                tprint(f"Error reading {f} for compaction: {e}")

        if not dfs:
            return

        merged = pd.concat(dfs)
        if "ts" in merged.columns:
            merged["ts"] = pd.to_datetime(merged["ts"], utc=True)
            merged = merged.sort_values("ts").drop_duplicates("ts", keep="last")

        ts_min = int(merged["ts"].min().value // 10**9)
        ts_max = int(merged["ts"].max().value // 10**9)
        new_fname = f"compact-{ts_min}-{ts_max}.parquet"
        new_fpath = os.path.join(part_dir, new_fname)
        temp_fpath = new_fpath + ".tmp"

        try:
            # Atomic write pattern
            merged.to_parquet(
                temp_fpath, index=False, engine="pyarrow", compression="zstd"
            )
            os.replace(temp_fpath, new_fpath)

            # Log cumulative stats
            interval_sec = pd.to_timedelta(self.timeframe).total_seconds()
            ts_min_val = merged["ts"].min().value // 10**9
            ts_max_val = merged["ts"].max().value // 10**9
            duration_sec = ts_max_val - ts_min_val + interval_sec
            days_covered = duration_sec / 86400.0
            avg_rows = len(merged) / days_covered if days_covered > 0 else 0

            ts_min_dt = pd.Timestamp(ts_min_val, unit="s", tz="UTC").strftime(
                "%Y-%m-%d"
            )
            ts_max_dt = pd.Timestamp(ts_max_val, unit="s", tz="UTC").strftime(
                "%Y-%m-%d"
            )
            tprint(
                f"Updated {new_fpath}: {len(merged)} rows, {ts_min_dt} -> {ts_max_dt} ({days_covered:.0f}d, ~{avg_rows:.0f} r/d)"
            )

            for f in files:
                if f != new_fpath:
                    try:
                        os.remove(f)
                    except OSError:
                        pass  # race condition if already deleted

        except Exception as e:
            tprint(f"Error compacting {symbol} {year}: {e}")

    def update_symbol(self, exchange, symbol: str, since_ms: int) -> pd.DataFrame:
        # Ensure locking
        sym_dir = self._get_symbol_dir(symbol)
        os.makedirs(sym_dir, exist_ok=True)
        lock_path = os.path.join(sym_dir, ".lock")

        with FileLock(lock_path):
            # Check metadata first to avoid IO
            meta = self._read_meta(symbol)
            last_ts_ms = meta.get("last_ts_ms", 0)

            if last_ts_ms > 0:
                start_ms = last_ts_ms + 1
            else:
                # Fallback to load index if no meta
                # Here load() without args is fine, but we might want to check just the last file?
                # For simplicity, keep as is, but maybe optimize load(columns=['ts'])
                existing_idx = self.load(symbol, columns=["ts"]).index
                if not existing_idx.empty:
                    last_ts = existing_idx.max()
                    start_ms = int(last_ts.value // 10**6) + 1
                else:
                    start_ms = since_ms

            now_ms = int(pd.Timestamp.utcnow().value // 10**6)

            if start_ms >= now_ms:
                return self.load(symbol)

            start_dt = pd.to_datetime(start_ms, unit="ms", utc=True).strftime(
                "%Y-%m-%d %H:%M"
            )
            tprint(f"FETCH incr: {symbol} from {start_dt}")

            # Progressive fetch and save (defer compaction to end)
            has_new_data = False
            touched_years = set()
            for chunk_df in fetch_ohlcv_all_7d_chunks(
                exchange, symbol, start_ms, timeframe=self.timeframe, limit=1000
            ):
                if not chunk_df.empty:
                    has_new_data = True
                    fresh = self._downcast(chunk_df)
                    self.save_partitioned(symbol, fresh, defer_compact=True)
                    touched_years.update(fresh.index.year.unique())

                    # Update metadata incrementally
                    new_last = fresh.index.max()
                    new_last_ms = int(new_last.value // 10**6)
                    if new_last_ms > last_ts_ms:
                        self._write_meta(symbol, {"last_ts_ms": new_last_ms})
                        last_ts_ms = new_last_ms

            # Single compaction pass per year at the end
            for yr in sorted(touched_years):
                self.compact_partition(symbol, yr)

    def update_symbol_perp(
        self,
        exchange,
        symbol: str,
        since_ms: int,
        spot_exchange=None,
    ) -> pd.DataFrame:
        """
        Update symbol from perp market:
        - OHLCV from perp exchange
        - funding_rate history
        - open_interest history
        - mark/index/premium-index OHLCV where the exchange exposes it
        - spot OHLCV auxiliary columns for spot/perp parity features
        """
        sym_dir = self._get_symbol_dir(symbol)
        os.makedirs(sym_dir, exist_ok=True)
        lock_path = os.path.join(sym_dir, ".lock")

        with FileLock(lock_path):
            meta = self._read_meta(symbol)
            last_ts_ms = meta.get("last_ts_ms", 0)

            if last_ts_ms > 0:
                start_ms = last_ts_ms + 1
            else:
                existing_idx = self.load(symbol, columns=["ts"]).index
                if not existing_idx.empty:
                    start_ms = int(existing_idx.max().value // 10**6) + 1
                else:
                    start_ms = since_ms

            now_ms = int(pd.Timestamp.utcnow().value // 10**6)
            if start_ms >= now_ms:
                return self.load(symbol)

            perp_symbol = _resolve_perp_symbol(exchange, symbol)
            if not perp_symbol:
                raise ValueError(f"No perp symbol found for {symbol}")

            start_dt = pd.to_datetime(start_ms, unit="ms", utc=True).strftime(
                "%Y-%m-%d %H:%M"
            )
            tprint(f"FETCH perp incr: {symbol} ({perp_symbol}) from {start_dt}")

            touched_years = set()
            has_new_data = False
            for chunk_df in fetch_ohlcv_all_7d_chunks(
                exchange, perp_symbol, start_ms, timeframe=self.timeframe, limit=1000
            ):
                if chunk_df.empty:
                    continue
                chunk = chunk_df.sort_index()
                chunk = chunk[~chunk.index.duplicated(keep="last")]
                if chunk.empty:
                    continue
                has_new_data = True

                # Keep funding/OI fetch bounded to this chunk so we can checkpoint incrementally.
                chunk_start_ms = int(chunk.index.min().value // 10**6)
                chunk_end_ms = int(
                    (chunk.index.max() + pd.Timedelta(hours=1)).value // 10**6
                )
                chunk_end_ms = min(chunk_end_ms, now_ms)

                funding = pd.Series(dtype=np.float32)
                funding_floor_ms = _recent_history_floor_ms(
                    "EPM_FUNDING_HISTORY_DAYS", 30.0
                )
                if (
                    hasattr(exchange, "fetch_funding_rate_history")
                    and chunk_end_ms >= funding_floor_ms
                ):
                    funding = _fetch_ccxt_history_paged(
                        exchange.fetch_funding_rate_history,
                        perp_symbol,
                        max(chunk_start_ms, funding_floor_ms),
                        chunk_end_ms,
                        value_keys=["fundingRate", "funding_rate", "rate"],
                        exchange=exchange,
                        limit=1000,
                    )

                oi = pd.Series(dtype=np.float32)
                oi_floor_ms = _recent_history_floor_ms("EPM_OPEN_INTEREST_HISTORY_DAYS", 30.0)
                if (
                    hasattr(exchange, "fetch_open_interest_history")
                    and chunk_end_ms >= oi_floor_ms
                ):
                    oi = _fetch_ccxt_history_paged(
                        exchange.fetch_open_interest_history,
                        perp_symbol,
                        max(chunk_start_ms, oi_floor_ms),
                        chunk_end_ms,
                        value_keys=[
                            "openInterestValue",
                            "sumOpenInterestValue",
                            "openInterestAmount",
                            "openInterest",
                            "sumOpenInterest",
                        ],
                        exchange=exchange,
                        timeframe=self.timeframe,
                        limit=500,
                    )

                def _align_ohlcv(extra_df: pd.DataFrame, prefix: str) -> None:
                    if extra_df is None or extra_df.empty:
                        return
                    extra = extra_df.reindex(chunk.index).ffill()
                    for src_col in ("open", "high", "low", "close", "volume"):
                        if src_col in extra.columns:
                            chunk[f"{prefix}_{src_col}"] = pd.to_numeric(
                                extra[src_col], errors="coerce"
                            ).astype(np.float32)

                for price_name, prefix in (
                    ("mark", "mark"),
                    ("index", "index"),
                    ("premiumIndex", "premium_index"),
                ):
                    try:
                        price_df = _fetch_ohlcv_paged(
                            exchange,
                            perp_symbol,
                            chunk_start_ms,
                            chunk_end_ms,
                            timeframe=self.timeframe,
                            limit=1000,
                            params={"price": price_name},
                        )
                        _align_ohlcv(price_df, prefix)
                    except Exception as exc:
                        tprint(
                            f"WARN perp {price_name} OHLCV fetch failed for {symbol}: {exc}"
                        )

                if spot_exchange is not None:
                    try:
                        spot_symbol = None
                        if "/" in symbol:
                            base, _quote = symbol.split("/", 1)
                            usdc_symbol = f"{base}/USDC"
                            if usdc_symbol in getattr(spot_exchange, "markets", {}):
                                spot_symbol = usdc_symbol
                        if spot_symbol is None and symbol in getattr(
                            spot_exchange, "markets", {}
                        ):
                            spot_symbol = symbol
                        if spot_symbol is None:
                            normalized = _normalize_spot_symbol(symbol)
                            if normalized in getattr(spot_exchange, "markets", {}):
                                spot_symbol = normalized
                        if spot_symbol is not None:
                            spot_df = _fetch_ohlcv_paged(
                                spot_exchange,
                                spot_symbol,
                                chunk_start_ms,
                                chunk_end_ms,
                                timeframe=self.timeframe,
                                limit=1000,
                            )
                            _align_ohlcv(spot_df, "spot")
                    except Exception as exc:
                        tprint(f"WARN spot auxiliary OHLCV fetch failed for {symbol}: {exc}")

                chunk["funding_rate"] = (
                    funding.reindex(chunk.index).ffill().astype(np.float32)
                )
                chunk["open_interest"] = (
                    oi.reindex(chunk.index).ffill().astype(np.float32)
                )
                if "mark_close" in chunk.columns:
                    chunk["mark_price"] = chunk["mark_close"]
                if "index_close" in chunk.columns:
                    chunk["index_price"] = chunk["index_close"]
                if "premium_index_close" in chunk.columns:
                    chunk["premium_index"] = chunk["premium_index_close"]

                fresh = self._downcast(chunk)
                self.save_partitioned(symbol, fresh, defer_compact=True)
                touched_years.update(fresh.index.year.unique())

                # Spot-parity: persist progress chunk-by-chunk.
                new_last_ms = int(fresh.index.max().value // 10**6)
                if new_last_ms > last_ts_ms:
                    self._write_meta(symbol, {"last_ts_ms": new_last_ms})
                    last_ts_ms = new_last_ms

            if not has_new_data:
                return self.load(symbol)

            for yr in sorted(touched_years):
                self.compact_partition(symbol, int(yr))

            return self.load(symbol)


def _feature_meta_path(parquet_path: str) -> str:
    return parquet_path.replace(".parquet", ".meta.json")


def _feature_lock_path(parquet_path: str) -> str:
    return parquet_path + ".lock"


def _atomic_write_parquet(df: pd.DataFrame, parquet_path: str):
    tmp_path = parquet_path + ".tmp"
    df.to_parquet(tmp_path, engine="pyarrow", compression="zstd")
    os.replace(tmp_path, parquet_path)


def _quarantine_corrupt_feature_file(parquet_path: str):
    if not os.path.exists(parquet_path):
        return
    quarantine_path = parquet_path + ".corrupt"
    if os.path.exists(quarantine_path):
        os.remove(quarantine_path)
    os.replace(parquet_path, quarantine_path)
    meta_path = _feature_meta_path(parquet_path)
    if os.path.exists(meta_path):
        os.remove(meta_path)


def _write_feature_metadata(parquet_path: str, symbol: str, index: pd.Index):
    meta_path = _feature_meta_path(parquet_path)
    if len(index) == 0:
        first_ts = last_ts = None
    else:
        first_ts = pd.Timestamp(index[0]).isoformat()
        last_ts = pd.Timestamp(index[-1]).isoformat()

    meta = {
        "version": 1,
        "symbol": symbol,
        "rows": int(len(index)),
        "first_ts": first_ts,
        "last_ts": last_ts,
    }

    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as fp:
        json.dump(meta, fp)
    os.replace(tmp_meta, meta_path)


def _read_feature_metadata(parquet_path: str) -> dict | None:
    meta_path = _feature_meta_path(parquet_path)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r") as fp:
            return json.load(fp)
    except Exception:
        return None


def _infer_feature_bounds_from_file(
    parquet_path: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    try:
        pf = pq.ParquetFile(parquet_path)
    except Exception:
        return None, None

    if pf.num_row_groups == 0:
        return None, None

    index_col = None
    for name in pf.schema.names:
        if name.startswith("__index_level_"):
            index_col = name
            break

    if index_col is None:
        return None, None

    try:
        first_group = pf.read_row_group(0, columns=[index_col])
        last_group = pf.read_row_group(pf.num_row_groups - 1, columns=[index_col])
        first_val = first_group.column(0)[0].as_py()
        last_val = last_group.column(0)[-1].as_py()
        return pd.Timestamp(first_val), pd.Timestamp(last_val)
    except Exception:
        return None, None


def get_feature_bounds(
    parquet_path: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    meta = _read_feature_metadata(parquet_path)
    if meta:
        first_ts = pd.Timestamp(meta["first_ts"]) if meta.get("first_ts") else None
        last_ts = pd.Timestamp(meta["last_ts"]) if meta.get("last_ts") else None
        return first_ts, last_ts

    return _infer_feature_bounds_from_file(parquet_path)


def append_symbol_features(
    parquet_path: str, symbol: str, new_data: pd.DataFrame
) -> int:
    if new_data.empty:
        return 0

    new_data = new_data.sort_index()
    new_data, new_reason = _ensure_feature_frame_index(
        new_data, parquet_path=parquet_path
    )
    if new_reason not in {None, "ts_column_indexed", "recovered_from_metadata"}:
        raise ValueError(
            f"append_symbol_features received invalid index for {symbol}: {new_reason}"
        )
    numeric_cols = [c for c in new_data.columns if c != "__symbol__"]
    new_data[numeric_cols] = new_data[numeric_cols].astype(np.float32)
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    lock_path = _feature_lock_path(parquet_path)

    with FileLock(lock_path):
        existing = None
        if os.path.exists(parquet_path):
            try:
                existing = pd.read_parquet(parquet_path)
            except Exception as e:
                tprint(
                    f"Warning: quarantining unreadable feature file {parquet_path}: {e}"
                )
                _quarantine_corrupt_feature_file(parquet_path)
            else:
                existing, existing_reason = _ensure_feature_frame_index(
                    existing, parquet_path=parquet_path
                )
                if existing_reason not in {
                    None,
                    "ts_column_indexed",
                    "recovered_from_metadata",
                }:
                    tprint(
                        f"Warning: quarantining feature file with invalid index {parquet_path}: "
                        f"{existing_reason}"
                    )
                    existing = None
                    _quarantine_corrupt_feature_file(parquet_path)
                elif "__symbol__" in existing.columns:
                    existing = existing.drop(columns=["__symbol__"])

        incoming_cols = list(new_data.columns)
        all_cols = sorted(
            set(incoming_cols)
            | (set(existing.columns) if existing is not None else set())
        )

        if existing is not None:
            existing_aligned = existing.reindex(columns=all_cols)
            before_rows = len(existing_aligned)

            # Only consider columns present in the incoming batch. Otherwise a partial
            # append would reindex missing columns to NaN and overwrite valid history.
            required_cols = set()
            existing_all_na = existing_aligned.isna().all(axis=0)
            new_all_na = new_data.isna().all(axis=0)
            drop_cols = [
                c
                for c in incoming_cols
                if bool(existing_all_na.get(c, True))
                and bool(new_all_na.get(c, True))
                and c not in required_cols
            ]
            if drop_cols:
                existing_aligned = existing_aligned.drop(
                    columns=drop_cols, errors="ignore"
                )
                new_data = new_data.drop(columns=drop_cols, errors="ignore")

            combined = existing_aligned.reindex(
                existing_aligned.index.union(new_data.index)
            ).sort_index()
            # Incremental feature generation must be additive: preserve existing
            # non-missing cells and only fill new rows, new columns, or NaNs.
            # Rewriting already-populated cells makes reruns expensive and can
            # mask earlier run provenance.
            target = combined.loc[new_data.index, new_data.columns]
            write_mask = target.isna()
            combined.loc[new_data.index, new_data.columns] = target.where(
                ~write_mask,
                new_data,
            )
        else:
            before_rows = 0
            combined = new_data

        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined["__symbol__"] = symbol

        # Ensure all numeric columns are float32 (not float64) to save memory
        numeric_cols = [c for c in combined.columns if c != "__symbol__"]
        combined[numeric_cols] = combined[numeric_cols].astype(np.float32, copy=False)

        _atomic_write_parquet(combined, parquet_path)
        _write_feature_metadata(parquet_path, symbol, combined.index)

        return len(combined) - before_rows


def save_features(
    feats: dict,
    ts: pd.Timestamp,
    root_dir: str,
    min_timestamp_by_symbol: dict[str, pd.Timestamp] | None = None,
    feat_index: pd.Index | None = None,
    feat_columns: list | None = None,
    save_workers: int | None = None,
    replace_existing: bool = False,
):
    """
    Save generated features to disk (Per-Symbol), streaming one symbol at a time.

    Peak memory ≈ 1 symbol × N_features × T rows (~2 MB).
    No temp chunk dirs, no merge step.

    feats: dict of feature_name -> DataFrame(index=t, cols=syms) OR numpy array (T, S).
           When numpy arrays, feat_index and feat_columns must be provided.
    """
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root_dir, "features", ts_str)
    os.makedirs(out_dir, exist_ok=True)

    tprint(f"Saving features to {out_dir}...")

    first_key = list(feats.keys())[0]
    first_val = feats[first_key]

    # Detect whether feats contains DataFrames or numpy arrays
    if isinstance(first_val, pd.DataFrame):
        symbols = list(first_val.columns)
        time_index = first_val.index
        feat_keys = [k for k in feats if hasattr(feats[k], "columns")]
        col_maps = {
            k: {c: j for j, c in enumerate(feats[k].columns)} for k in feat_keys
        }
        arrays = {k: feats[k].values for k in feat_keys}
    else:
        # Numpy array mode — stream symbol-by-symbol to avoid feature stacking spikes.
        assert (
            feat_index is not None and feat_columns is not None
        ), "feat_index and feat_columns required when feats contains numpy arrays"
        symbols = list(feat_columns)
        time_index = feat_index
        feat_keys = [
            k
            for k in feats
            if isinstance(feats[k], np.ndarray) and feats[k].ndim in (1, 2)
        ]
        n_feats = len(feat_keys)
        total = len(symbols)

        tprint(
            f"  Saving {n_feats} features × {total} symbols in streaming symbol mode..."
        )

        worker_count = max(1, int(save_workers or 1))
        max_pending = max(1, worker_count * 2)

        def _prepare_symbol_payload(j: int, sym: str):
            safe_sym = sym.replace("/", "_")
            final_path = os.path.join(out_dir, f"symbol={safe_sym}.parquet")

            cutoff_ts = None
            if min_timestamp_by_symbol:
                cutoff_ts = min_timestamp_by_symbol.get(sym)

            sym_data = {}
            for k in feat_keys:
                arr = feats[k]
                if arr.ndim == 2:
                    sym_data[k] = arr[:, j]
                else:
                    sym_data[k] = arr
            df_sym = pd.DataFrame(sym_data, index=time_index, copy=False)
            df_sym = df_sym.astype(np.float32, copy=False)

            if cutoff_ts is not None:
                # Ensure timezone compatibility between cutoff and index
                if df_sym.index.tz is not None and cutoff_ts.tzinfo is None:
                    cutoff_ts = cutoff_ts.tz_localize(df_sym.index.tz)
                elif df_sym.index.tz is None and cutoff_ts.tzinfo is not None:
                    cutoff_ts = cutoff_ts.tz_localize(None)
                df_sym = df_sym[df_sym.index > cutoff_ts]
                if df_sym.empty:
                    del sym_data
                    del df_sym
                    return None

            del sym_data
            return final_path, sym, df_sym

        def _write_symbol_payload(payload) -> bool:
            if payload is None:
                return False
            final_path, sym, df_sym = payload
            if replace_existing:
                df_out = df_sym.copy()
                df_out["__symbol__"] = sym
                _atomic_write_parquet(df_out, final_path)
                _write_feature_metadata(final_path, sym, df_out.index)
            else:
                append_symbol_features(final_path, sym, df_sym)
            return True

        count = 0
        if worker_count == 1:
            for j, sym in enumerate(symbols):
                payload = _prepare_symbol_payload(j, sym)
                wrote = _write_symbol_payload(payload)
                if payload is not None:
                    _, _, df_sym = payload
                    del df_sym
                if wrote:
                    count += 1

                if count % 25 == 0 or count == total:
                    tprint(
                        f"  Save progress: {count}/{total} symbols ({n_feats} features each)"
                    )
                    _gc.collect()
                if count % 200 == 0:
                    _gc.collect()
        else:
            tprint(
                f"  Save parallelism enabled: workers={worker_count}, max_pending={max_pending}"
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count
            ) as executor:
                pending: dict[concurrent.futures.Future, pd.DataFrame] = {}
                for j, sym in enumerate(symbols):
                    payload = _prepare_symbol_payload(j, sym)
                    if payload is None:
                        continue
                    future = executor.submit(_write_symbol_payload, payload)
                    pending[future] = payload[2]

                    if len(pending) >= max_pending:
                        done, _ = concurrent.futures.wait(
                            list(pending.keys()),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for fut in done:
                            wrote = fut.result()
                            df_sym = pending.pop(fut)
                            del df_sym
                            if wrote:
                                count += 1
                            if count % 25 == 0 or count == total:
                                tprint(
                                    f"  Save progress: {count}/{total} symbols ({n_feats} features each)"
                                )
                                _gc.collect()
                            if count % 200 == 0:
                                _gc.collect()

                for fut in concurrent.futures.as_completed(list(pending.keys())):
                    wrote = fut.result()
                    df_sym = pending.pop(fut)
                    del df_sym
                    if wrote:
                        count += 1
                    if count % 25 == 0 or count == total:
                        tprint(
                            f"  Save progress: {count}/{total} symbols ({n_feats} features each)"
                        )
                        _gc.collect()
                    if count % 200 == 0:
                        _gc.collect()

        tprint(
            f"Feature save complete. {count}/{total} symbols saved ({n_feats} features)."
        )
        return

    total = len(symbols)
    n_feats = len(feat_keys)

    worker_count = max(1, int(save_workers or 1))
    max_pending = max(1, worker_count * 2)

    def _prepare_dataframe_symbol_payload(sym: str):
        cutoff_ts = None
        if min_timestamp_by_symbol:
            cutoff_ts = min_timestamp_by_symbol.get(sym)

        # Build {feat_name: 1-D array} for this symbol
        col_data = {}
        for k in feat_keys:
            j = col_maps[k].get(sym)
            if j is not None:
                col_data[k] = arrays[k][:, j]

        if not col_data:
            return None

        df_sym = pd.DataFrame(col_data, index=time_index)
        df_sym = df_sym.astype(np.float32, copy=False)
        if cutoff_ts is not None:
            # Ensure timezone compatibility between cutoff and index
            if df_sym.index.tz is not None and cutoff_ts.tzinfo is None:
                cutoff_ts = cutoff_ts.tz_localize(df_sym.index.tz)
            elif df_sym.index.tz is None and cutoff_ts.tzinfo is not None:
                cutoff_ts = cutoff_ts.tz_localize(None)
            df_sym = df_sym[df_sym.index > cutoff_ts]
        if df_sym.empty:
            return None

        safe_sym = sym.replace("/", "_")
        final_path = os.path.join(out_dir, f"symbol={safe_sym}.parquet")
        return final_path, sym, df_sym

    def _write_dataframe_symbol_payload(payload) -> bool:
        if payload is None:
            return False
        final_path, sym, df_sym = payload
        if replace_existing:
            df_out = df_sym.copy()
            df_out["__symbol__"] = sym
            _atomic_write_parquet(df_out, final_path)
            _write_feature_metadata(final_path, sym, df_out.index)
        else:
            append_symbol_features(final_path, sym, df_sym)
        return True

    count = 0
    if worker_count == 1:
        for sym in symbols:
            payload = _prepare_dataframe_symbol_payload(sym)
            wrote = _write_dataframe_symbol_payload(payload)
            if payload is not None:
                _, _, df_sym = payload
                del df_sym
            if wrote:
                count += 1

            if count % 50 == 0:
                tprint(f"Saved {count}/{total} symbols ({n_feats} features each)")
            if count % 100 == 0:
                _gc.collect()
    else:
        tprint(
            f"  Save parallelism enabled: workers={worker_count}, max_pending={max_pending}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending: dict[concurrent.futures.Future, pd.DataFrame] = {}
            for sym in symbols:
                payload = _prepare_dataframe_symbol_payload(sym)
                if payload is None:
                    continue
                future = executor.submit(_write_dataframe_symbol_payload, payload)
                pending[future] = payload[2]

                if len(pending) >= max_pending:
                    done, _ = concurrent.futures.wait(
                        list(pending.keys()),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for fut in done:
                        wrote = fut.result()
                        df_sym = pending.pop(fut)
                        del df_sym
                        if wrote:
                            count += 1
                        if count % 50 == 0:
                            tprint(
                                f"Saved {count}/{total} symbols ({n_feats} features each)"
                            )
                        if count % 100 == 0:
                            _gc.collect()

            for fut in concurrent.futures.as_completed(list(pending.keys())):
                wrote = fut.result()
                df_sym = pending.pop(fut)
                del df_sym
                if wrote:
                    count += 1
                if count % 50 == 0:
                    tprint(f"Saved {count}/{total} symbols ({n_feats} features each)")
                if count % 100 == 0:
                    _gc.collect()
    tprint(
        f"Feature save complete. {count}/{total} symbols saved ({n_feats} features)."
    )


def load_features(ts: pd.Timestamp, root_dir: str) -> dict:
    """
    Load features from disk if they exist for this timestamp.

    Expects files matching 'symbol=*.parquet'. Restores the original symbol name
    from the '__symbol__' column if present, enabling support for symbols with
    special characters (e.g. 'BTC/USDT').

    Returns: dict of DataFrames (feature_name -> DataFrame(index=t, cols=syms)) or None.
    """
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    in_dir = os.path.join(root_dir, "features", ts_str)

    if not os.path.exists(in_dir):
        return None

    files = sorted(glob.glob(os.path.join(in_dir, "symbol=*.parquet")))
    if not files:
        return None

    tprint(f"Found {len(files)} feature files in {in_dir}. Loading...")

    # Build Dict[Feat -> Dict[Symbol -> Series]] incrementally to reduce peak memory.
    # Previous implementation stored all symbol DataFrames first, then pivoted, which
    # could double memory pressure on large universes.
    feat_buffers = {}

    start_load = time.time()
    total_files = len(files)
    progress_every = 25 if total_files >= 100 else 10

    for i, fpath in enumerate(files, start=1):
        try:
            fname = os.path.basename(fpath)
            # fname is symbol=XYZ.parquet
            sym = fname.replace("symbol=", "").replace(".parquet", "")
            df = pd.read_parquet(fpath)

            if "__symbol__" in df.columns:
                if not df.empty:
                    real_sym = str(df["__symbol__"].iloc[0])
                    df = df.drop(columns=["__symbol__"])
                else:
                    df = df.drop(columns=["__symbol__"])
                    real_sym = sym
            else:
                # Legacy files without __symbol__: restore slash from underscore
                # e.g. BTC_USDT -> BTC/USDT (first underscore only)
                real_sym = sym.replace("_", "/", 1)

            for k in df.columns:
                if k not in feat_buffers:
                    feat_buffers[k] = {}
                feat_buffers[k][real_sym] = _coerce_feature_values_float32(df[k])

            del df
            if i % progress_every == 0 or i == total_files:
                elapsed = time.time() - start_load
                tprint(
                    f"Feature load progress: {i}/{total_files} files "
                    f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                )
        except Exception as e:
            tprint(f"Error loading {fpath}: {e}")

    # Encourage timely memory reclamation after file ingest loop
    import gc as _gc

    _gc.collect()

    if not feat_buffers:
        return None

    feats_out = {}
    for k, data in feat_buffers.items():
        # Construct DF for this feature: Index=Time, Cols=Symbols
        feats_out[k] = pd.DataFrame(data).sort_index()

    feat_buffers.clear()
    _gc.collect()

    tprint(f"Loaded {len(feats_out)} feature matrices.")
    return feats_out


class LazyFeatureDict:
    def __init__(self, raw_data_buffers, symbol_indices=None):
        self._raw = raw_data_buffers
        self._symbol_indices = symbol_indices or {}
        self._assembled = {}

    def _assemble_key(self, k, log=True):
        if k in self._assembled:
            return self._assembled[k]
        if k in self._raw:
            if log:
                from extreme_price_movements.utils import tprint

                tprint(f"Lazy-assembling DataFrame for '{k}'...")
            data = self._raw.pop(k)
            clean_data = {}
            skipped_symbols = []
            normalized_symbols = []
            for sym, payload in data.items():
                if isinstance(payload, tuple) and len(payload) == 2:
                    idx_vals, val_array = payload
                else:
                    idx_vals = self._symbol_indices.get(sym)
                    val_array = payload
                if idx_vals is None:
                    skipped_symbols.append(f"{sym}:missing_index")
                    continue
                normalized_idx, normalized_vals, reason = _normalize_feature_index(
                    idx_vals,
                    val_array,
                )
                if normalized_idx is None or normalized_vals is None:
                    skipped_symbols.append(f"{sym}:{reason}")
                    continue
                if reason is not None:
                    normalized_symbols.append(f"{sym}:{reason}")
                series_idx = pd.DatetimeIndex(
                    pd.to_datetime(normalized_idx, utc=True, errors="coerce"),
                    tz="UTC",
                )
                series = pd.Series(normalized_vals, index=series_idx, copy=False)
                if not series.index.is_unique:
                    series = series[~series.index.duplicated(keep="last")]
                clean_data[sym] = series
            if skipped_symbols:
                tprint(
                    f"Lazy feature assembly skipped {len(skipped_symbols)} symbols for '{k}' "
                    f"due to invalid indices. Sample: {skipped_symbols[:5]}"
                )
            if normalized_symbols:
                tprint(
                    f"Lazy feature assembly normalized {len(normalized_symbols)} symbols for '{k}'. "
                    f"Sample: {normalized_symbols[:5]}"
                )
            df = pd.DataFrame(clean_data).sort_index() if clean_data else pd.DataFrame()
            self._assembled[k] = df
            return df
        raise KeyError(k)

    def _assemble_many_keys(self, keys, progress_every=25):
        from extreme_price_movements.utils import tprint

        target_keys = [k for k in keys if k in self._raw and k not in self._assembled]
        total = len(target_keys)
        if total == 0:
            return

        tprint(f"Pre-materializing {total} feature matrices in grouped mode...")
        clean_data_by_key: dict[str, dict[str, pd.Series]] = {
            k: {} for k in target_keys
        }
        skipped_by_key: dict[str, list[str]] = {k: [] for k in target_keys}
        normalized_by_key: dict[str, list[str]] = {k: [] for k in target_keys}

        symbols: set[str] = set()
        for k in target_keys:
            raw_payload = self._raw.get(k)
            if isinstance(raw_payload, dict):
                symbols.update(str(sym) for sym in raw_payload.keys())

        for sym in sorted(symbols):
            idx_vals = self._symbol_indices.get(sym)
            for k in target_keys:
                data = self._raw.get(k)
                if not isinstance(data, dict) or sym not in data:
                    continue
                payload = data[sym]
                if isinstance(payload, tuple) and len(payload) == 2:
                    idx_vals_local, val_array = payload
                else:
                    idx_vals_local, val_array = idx_vals, payload
                if idx_vals_local is None:
                    skipped_by_key[k].append(f"{sym}:missing_index")
                    continue
                normalized_idx, normalized_vals, reason = _normalize_feature_index(
                    idx_vals_local,
                    val_array,
                )
                if normalized_idx is None or normalized_vals is None:
                    skipped_by_key[k].append(f"{sym}:{reason}")
                    continue
                if reason is not None:
                    normalized_by_key[k].append(f"{sym}:{reason}")
                series_idx = pd.DatetimeIndex(
                    pd.to_datetime(normalized_idx, utc=True, errors="coerce"),
                    tz="UTC",
                )
                series = pd.Series(normalized_vals, index=series_idx, copy=False)
                if not series.index.is_unique:
                    series = series[~series.index.duplicated(keep="last")]
                clean_data_by_key[k][sym] = series

        for i, k in enumerate(target_keys, start=1):
            df = (
                pd.DataFrame(clean_data_by_key[k]).sort_index()
                if clean_data_by_key[k]
                else pd.DataFrame()
            )
            self._assembled[k] = df
            self._raw.pop(k, None)
            if skipped_by_key[k]:
                tprint(
                    f"Lazy feature assembly skipped {len(skipped_by_key[k])} symbols for '{k}' "
                    f"due to invalid indices. Sample: {skipped_by_key[k][:5]}"
                )
            if normalized_by_key[k]:
                tprint(
                    f"Lazy feature assembly normalized {len(normalized_by_key[k])} symbols for '{k}'. "
                    f"Sample: {normalized_by_key[k][:5]}"
                )
            if i % progress_every == 0 or i == total:
                tprint(
                    f"  Grouped feature materialization progress: {i}/{total} "
                    f"({(100.0 * i / max(1, total)):.1f}%)"
                )

    def __getitem__(self, k):
        return self._assemble_key(k, log=True)

    def __setitem__(self, k, v):
        self._assembled[k] = v
        if k in self._raw:
            self._raw.pop(k)

    def __contains__(self, k):
        return k in self._assembled or k in self._raw

    def keys(self):
        return list(self._assembled.keys()) + list(self._raw.keys())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def values(self):
        for k in self.keys():
            yield self[k]

    def pop(self, k, default=None):
        if k in self._assembled:
            return self._assembled.pop(k)
        if k in self._raw:
            val = self[k]
            del self._assembled[k]
            return val
        if default is not None:
            return default
        raise KeyError(k)

    def copy(self):
        cloned = LazyFeatureDict({})
        cloned._assembled = dict(self._assembled)
        cloned._symbol_indices = dict(self._symbol_indices)
        cloned._raw = {
            k: dict(v) if isinstance(v, dict) else v for k, v in self._raw.items()
        }
        return cloned

    def materialize(self, keys=None, progress_every=25):
        target_keys = (
            list(self.keys()) if keys is None else [k for k in keys if k in self]
        )
        total = len(target_keys)
        if total == 0:
            return
        self._assemble_many_keys(target_keys, progress_every=progress_every)


def load_features_selected(
    ts: pd.Timestamp,
    root_dir: str,
    feature_keys: list[str] | set[str] | tuple[str, ...] | None = None,
    symbols: list[str] | set[str] | tuple[str, ...] | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    allowed_periods=None,
) -> dict:
    """
    Load a subset of features/symbols from disk.

    This avoids loading every cached feature matrix into memory when only a
    narrow key set is required by downstream steps (e.g. label generation).
    """
    if feature_keys is None and symbols is None:
        return load_features(ts, root_dir)

    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    in_dir = os.path.join(root_dir, "features", ts_str)
    if not os.path.exists(in_dir):
        tprint(
            f"load_features_selected: in_dir not found: {in_dir} (cwd={os.getcwd()})"
        )
        return None

    feature_set = set(feature_keys) if feature_keys else None
    symbol_set = (
        {_normalize_spot_symbol(str(sym)) for sym in symbols} if symbols else None
    )
    start_ts = pd.Timestamp(start_ts) if start_ts is not None else None
    end_ts = pd.Timestamp(end_ts) if end_ts is not None else None
    if start_ts is not None and start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)
    if end_ts is not None and end_ts.tzinfo is not None:
        end_ts = end_ts.tz_localize(None)
    normalized_periods = _normalize_allowed_periods(allowed_periods)
    parquet_filters = _build_parquet_ts_filters(
        start_ts=start_ts,
        end_ts=end_ts,
        allowed_periods=normalized_periods,
    )
    if symbol_set is not None:
        files = []
        seen = set()
        for sym in symbol_set:
            for candidate in _symbol_alias_candidates(sym):
                safe_sym = candidate.replace("/", "_")
                fpath = os.path.join(in_dir, f"symbol={safe_sym}.parquet")
                if os.path.exists(fpath) and fpath not in seen:
                    files.append(fpath)
                    seen.add(fpath)
        if not files:
            tprint(
                f"load_features_selected: no requested symbol parquet files found in {in_dir} "
                f"for {sorted(symbol_set)}"
            )
            return None
    else:
        files = sorted(glob.glob(os.path.join(in_dir, "symbol=*.parquet")))
        if not files:
            tprint(
                f"load_features_selected: no symbol=*.parquet files found in {in_dir}"
            )
            return None

    feat_buffers: dict[str, dict[str, np.ndarray]] = {}
    symbol_indices: dict[str, np.ndarray] = {}

    tprint(
        f"Found {len(files)} feature files in {in_dir}. "
        f"Selective load: keys={len(feature_set) if feature_set else 'ALL'}, "
        f"symbols={len(symbol_set) if symbol_set else 'ALL'}"
    )

    start_load = time.time()
    total_files = len(files)
    progress_every = 25 if total_files >= 100 else 10

    for i, fpath in enumerate(files, start=1):
        try:
            fname = os.path.basename(fpath)
            sym_guess = _normalize_spot_symbol(
                fname.replace("symbol=", "").replace(".parquet", "")
            )
            if symbol_set is not None and sym_guess not in symbol_set:
                if i % progress_every == 0 or i == total_files:
                    elapsed = time.time() - start_load
                    tprint(
                        f"Selective feature load progress: {i}/{total_files} files "
                        f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                    )
                continue

            schema_names = set(pq.ParquetFile(fpath).schema.names)
            cols_to_read = []
            if "__symbol__" in schema_names:
                cols_to_read.append("__symbol__")
            if "ts" in schema_names:
                cols_to_read.append("ts")
            if feature_set is None:
                cols_to_read.extend(
                    [
                        c
                        for c in schema_names
                        if c not in {"__symbol__", "ts"}
                        and not c.startswith("__index_level_")
                    ]
                )
            else:
                cols_to_read.extend([c for c in feature_set if c in schema_names])

            if not cols_to_read or (
                len(cols_to_read) == 1 and cols_to_read[0] == "__symbol__"
            ):
                if i % progress_every == 0 or i == total_files:
                    elapsed = time.time() - start_load
                    tprint(
                        f"Selective feature load progress: {i}/{total_files} files "
                        f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                    )
                continue

            read_kwargs = {"columns": cols_to_read}
            if parquet_filters is not None and "ts" in schema_names:
                read_kwargs["filters"] = parquet_filters
            try:
                df = pd.read_parquet(fpath, **read_kwargs)
            except Exception as e:
                if parquet_filters is not None and "ts" in schema_names:
                    tprint(
                        f"Parquet ts pushdown failed for {fpath}: {e}. Falling back to post-load slicing."
                    )
                    read_kwargs.pop("filters", None)
                    df = pd.read_parquet(fpath, **read_kwargs)
                else:
                    raise
            df, index_reason = _ensure_feature_frame_index(df, parquet_path=fpath)
            if index_reason == "invalid_ts_column":
                tprint(f"Skipping feature file {fpath}: invalid ts column")
                continue
            _idx = df.index
            if isinstance(_idx, pd.DatetimeIndex) and _idx.tz is not None:
                _s = (
                    start_ts.tz_localize(_idx.tz)
                    if start_ts is not None and start_ts.tzinfo is None
                    else start_ts
                )
                _e = (
                    end_ts.tz_localize(_idx.tz)
                    if end_ts is not None and end_ts.tzinfo is None
                    else end_ts
                )
            else:
                _s = start_ts
                _e = end_ts
            if _s is not None and parquet_filters is None:
                df = df[df.index >= _s]
            if _e is not None and parquet_filters is None:
                df = df[df.index <= _e]
            if normalized_periods and parquet_filters is None:
                df = _apply_allowed_periods_mask(df, normalized_periods)
            if df.empty:
                if i % progress_every == 0 or i == total_files:
                    elapsed = time.time() - start_load
                    tprint(
                        f"Selective feature load progress: {i}/{total_files} files "
                        f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                    )
                continue
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep="last")]

            if "__symbol__" in df.columns:
                if not df.empty:
                    real_sym = _normalize_spot_symbol(str(df["__symbol__"].iloc[0]))
                else:
                    real_sym = sym_guess
                df = df.drop(columns=["__symbol__"])
            else:
                real_sym = sym_guess

            if symbol_set is not None and real_sym not in symbol_set:
                if i % progress_every == 0 or i == total_files:
                    elapsed = time.time() - start_load
                    tprint(
                        f"Selective feature load progress: {i}/{total_files} files "
                        f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                    )
                continue

            normalized_idx, _, index_reason = _normalize_feature_index(df.index.values)
            if normalized_idx is None:
                tprint(
                    f"Skipping feature file {fpath} for symbol {real_sym}: invalid index ({index_reason})"
                )
                continue
            if index_reason is not None:
                tprint(
                    f"Normalized feature index for symbol {real_sym} in {fpath}: {index_reason}"
                )
            df.index = normalized_idx
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep="last")]
            idx_vals = df.index.to_numpy(copy=False)
            symbol_indices[real_sym] = idx_vals
            for k in df.columns:
                if feature_set is not None and k not in feature_set:
                    continue
                if k not in feat_buffers:
                    feat_buffers[k] = {}
                feat_buffers[k][real_sym] = _coerce_feature_values_float32(df[k])

            del df
            if i % progress_every == 0 or i == total_files:
                elapsed = time.time() - start_load
                tprint(
                    f"Selective feature load progress: {i}/{total_files} files "
                    f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                )
        except Exception as e:
            tprint(f"Error loading {fpath}: {e}")

    if not feat_buffers:
        return None

    tprint(
        f"Loaded raw arrays for {len(feat_buffers)} features. Returning LazyFeatureDict proxy."
    )
    return LazyFeatureDict(feat_buffers, symbol_indices=symbol_indices)


def check_data_health(df: pd.DataFrame, timeframe="1h") -> dict:
    if df.empty:
        return {"status": "empty", "completeness": 0.0, "missing_count": 0}

    start = df.index.min()
    end = df.index.max()

    if timeframe == "1h":
        freq = "h"
    else:
        freq = timeframe

    full_idx = pd.date_range(start, end, freq=freq, tz="UTC")
    expected_rows = len(full_idx)
    actual_rows = len(df)

    missing = full_idx.difference(df.index)
    missing_count = len(missing)
    completeness = actual_rows / expected_rows if expected_rows > 0 else 0.0

    return {
        "status": "ok" if missing_count == 0 else "gaps",
        "completeness": completeness,
        "missing_count": missing_count,
        "first_missing": missing[0].isoformat() if missing_count > 0 else None,
        "last_missing": missing[-1].isoformat() if missing_count > 0 else None,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
    }


def to_panel(dfs_by_symbol: dict[str, pd.DataFrame]):
    keys = ["open", "high", "low", "close", "volume"]
    extra_keys = sorted(
        {
            col
            for _sym, df in dfs_by_symbol.items()
            for col in df.columns
            if col not in keys
        }
    )
    keys.extend(extra_keys)
    panel = {}
    for k in keys:
        cols = [
            df[k].rename(sym) for sym, df in dfs_by_symbol.items() if k in df.columns
        ]
        if not cols:
            continue
        panel[k] = pd.concat(cols, axis=1).sort_index()
    return panel


OHLCVStore = PartitionedOHLCVStore


def get_feature_path(root_dir: str, ts: pd.Timestamp, symbol: str) -> str:
    """
    Returns the expected file path for a symbol's features at a given timestamp.
    """
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    safe_sym = symbol.replace("/", "_")
    return os.path.join(root_dir, "features", ts_str, f"symbol={safe_sym}.parquet")


def save_artifact_df(
    df: pd.DataFrame, root_dir: str, run_id: str, category: str, name: str
):
    """
    Save a DataFrame as an artifact for a specific run.
    Path: root_dir/artifacts/{run_id}/{category}/{name}.parquet
    """
    out_dir = os.path.join(root_dir, "artifacts", run_id, category)
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"{name}.parquet")
    tprint(f"Saving artifact: {fpath}")
    df.to_parquet(fpath, engine="pyarrow", compression="zstd")


def read_parquet_projected(path: Union[str, os.PathLike], columns: List[str]) -> pd.DataFrame:
    """Read only columns that exist in a parquet file."""
    fpath = os.fspath(path)
    try:
        schema_cols = set(pq.ParquetFile(fpath).schema.names)
        read_columns = [c for c in columns if c in schema_cols]
    except Exception:
        read_columns = list(columns)
    return pd.read_parquet(fpath, columns=read_columns)


def load_artifact_manifest(root_dir: str, run_id: str, category: str) -> dict | None:
    """Load the JSON manifest for a saved artifact category if present."""
    if category == "labels":
        fpath = os.path.join(
            root_dir, "artifacts", run_id, category, "labels_manifest.json"
        )
    else:
        fpath = os.path.join(root_dir, "artifacts", run_id, category, "manifest.json")
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def load_artifact_df(
    root_dir: str,
    run_id: str,
    category: str,
    name: str,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load an artifact DataFrame. Returns None if not found.
    """
    fpath = os.path.join(root_dir, "artifacts", run_id, category, f"{name}.parquet")
    if os.path.exists(fpath):
        tprint(f"Loading artifact: {fpath}")
        if columns is not None:
            df = read_parquet_projected(fpath, columns)
        else:
            df = pd.read_parquet(fpath)

        # Normalize: ensure 'ts' and 'symbol' exist if their dunder versions do.
        # This fixes inconsistencies between training.py and pipeline_steps.py.
        if "__ts__" in df.columns and "ts" not in df.columns:
            df["ts"] = df["__ts__"]
        if "__symbol__" in df.columns and "symbol" not in df.columns:
            df["symbol"] = df["__symbol__"]

        # Downcast floats to float32 to save memory
        for col in df.select_dtypes(include=[np.float64]).columns:
            df[col] = df[col].astype(np.float32)
        return df
    return None
