from __future__ import annotations

import concurrent.futures
import fcntl
import gc as _gc
import glob
import hashlib
import json
import os
import re
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
_KRAKEN_FUNDING_EXPORT_CACHE: dict[tuple[str, str], pd.Series] = {}
LIVE_LATEST_FEATURE_MATRIX_VERSION = 1


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
    if "/" in canonical:
        base, quote = canonical.split("/", 1)
        # Perp feature caches are written as e.g. symbol=KAITO_USD:USD.parquet
        # for KAITO/USD:USD.  The spot-normalized aliases below collapse this to
        # KAITO_USDUSD and miss the authoritative selected-feature cache.
        if ":" in quote:
            candidates.append(f"{base}_{quote}")
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
    inclusive_end: bool = False,
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
        end_op = "<=" if inclusive_end else "<"
        filters.append(
            [
                ("ts", ">=", period_start.to_pydatetime()),
                ("ts", end_op, period_end.to_pydatetime()),
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


def exchange_data_component(exchange_id: Optional[str] = None, market_mode: str = "spot") -> str:
    raw = str(exchange_id or _configured_exchange_id() or "binance").strip().lower()
    raw = raw.replace("-", "_")
    mode = str(market_mode or "spot").strip().lower()
    if raw in {"okx", "okex"}:
        return "okx"
    if raw in {"krakenfutures", "kraken_futures"}:
        return "krakenfutures"
    if raw == "kraken":
        return "krakenfutures" if mode in {"perp", "perps", "future", "futures", "swap"} else "kraken"
    if raw in {"binanceusdm", "binance_usdm", "binance-futures"}:
        return "binanceusdm"
    if raw == "binance":
        return "binanceusdm" if mode in {"perp", "perps", "future", "futures", "swap"} else "binance"
    return re.sub(r"[^a-z0-9_]+", "_", raw).strip("_") or "exchange"


def exchange_data_root(
    data_root: str,
    exchange_id: Optional[str] = None,
    market_mode: str = "spot",
) -> str:
    return os.path.join(
        str(data_root),
        "exchanges",
        exchange_data_component(exchange_id, market_mode),
    )


def use_exchange_scoped_data(cfg: Optional[Dict[str, Any]] = None) -> bool:
    cfg = cfg or {}
    raw = os.environ.get(
        "EPM_EXCHANGE_SCOPED_DATA",
        str(cfg.get("exchange_scoped_data", True)),
    )
    return str(raw).strip().lower() not in {"0", "false", "no", "n", "off"}


def scoped_data_root(cfg: Dict[str, Any]) -> str:
    root = str(cfg.get("data_root", "data"))
    if not use_exchange_scoped_data(cfg):
        return root
    component = exchange_data_component(
        cfg.get("exchange_id") or cfg.get("exchange"),
        cfg.get("market_mode") or ("perps" if cfg.get("use_perps") else "spot"),
    )
    norm_root = os.path.normpath(root)
    parts = norm_root.split(os.sep)
    if len(parts) >= 2 and parts[-2] == "exchanges" and parts[-1] == component:
        return root
    return os.path.join(root, "exchanges", component)


def make_ohlcv_store(cfg: Dict[str, Any], *, timeframe: Optional[str] = None):
    return PartitionedOHLCVStore(
        root_dir=scoped_data_root(cfg),
        timeframe=timeframe or cfg.get("timeframe", "1h"),
    )


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


def _fetch_kraken_futures_historical_funding_rates(
    exchange,
    perp_symbol: str,
    since_ms: int,
    until_ms: int,
) -> pd.Series:
    product_id = _kraken_futures_product_id(exchange, perp_symbol)
    if not product_id:
        return pd.Series(dtype=np.float32)
    export_funding = _fetch_kraken_futures_exported_funding_rates(
        product_id, since_ms, until_ms
    )
    url = "https://futures.kraken.com/derivatives/api/v3/historical-funding-rates"
    headers = {"User-Agent": _ARCHIVE_USER_AGENT}
    try:
        response = _public_data_session().get(
            url,
            params={"symbol": product_id},
            timeout=30,
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        tprint(f"WARN Kraken historical funding fetch failed for {perp_symbol}: {exc}")
        return export_funding

    rows: list[tuple[pd.Timestamp, float]] = []
    for item in payload.get("rates", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        ts = pd.to_datetime(item.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        ts_ms = int(ts.value // 10**6)
        if ts_ms < int(since_ms) or ts_ms >= int(until_ms):
            continue
        value = item.get("relativeFundingRate", item.get("fundingRate"))
        try:
            rate = float(value)
        except Exception:
            continue
        if np.isfinite(rate):
            rows.append((ts.floor("h"), rate))
    if not rows:
        return export_funding
    df = pd.DataFrame(rows, columns=["ts", "value"])
    api_funding = df.groupby("ts")["value"].last().sort_index().astype(np.float32)
    if export_funding.empty:
        return api_funding
    combined = pd.concat([export_funding, api_funding])
    return combined.groupby(level=0).last().sort_index().astype(np.float32)


def _fetch_kraken_futures_exported_funding_rates(
    product_id: str,
    since_ms: int,
    until_ms: int,
) -> pd.Series:
    export_path = os.getenv("EPM_KRAKEN_FUNDING_EXPORT_ZIP", "").strip()
    if not export_path:
        export_path = os.path.join(
            os.getcwd(),
            "data_perp",
            "exchanges",
            "krakenfutures",
            "reference",
            "kraken_funding_rates_export_20260216.zip",
        )
    if not export_path or not os.path.exists(export_path):
        return pd.Series(dtype=np.float32)

    product_id = str(product_id or "").strip().upper()
    if not product_id:
        return pd.Series(dtype=np.float32)
    cache_key = (os.path.abspath(export_path), product_id)
    cached = _KRAKEN_FUNDING_EXPORT_CACHE.get(cache_key)
    if cached is None:
        member = f"exports/{product_id}.csv"
        try:
            with zipfile.ZipFile(export_path) as zf:
                if member not in set(zf.namelist()):
                    _KRAKEN_FUNDING_EXPORT_CACHE[cache_key] = pd.Series(dtype=np.float32)
                    return _KRAKEN_FUNDING_EXPORT_CACHE[cache_key]
                with zf.open(member) as fp:
                    raw = pd.read_csv(fp, usecols=["timestamp", "relative_rate"])
        except Exception as exc:
            tprint(f"WARN Kraken funding export read failed for {product_id}: {exc}")
            _KRAKEN_FUNDING_EXPORT_CACHE[cache_key] = pd.Series(dtype=np.float32)
            return _KRAKEN_FUNDING_EXPORT_CACHE[cache_key]
        ts = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dt.floor("h")
        values = pd.to_numeric(raw["relative_rate"], errors="coerce")
        valid = ts.notna() & np.isfinite(values.to_numpy(dtype=np.float64, copy=False))
        if not bool(valid.any()):
            cached = pd.Series(dtype=np.float32)
        else:
            cached = (
                pd.DataFrame({"ts": ts[valid], "value": values[valid]})
                .groupby("ts")["value"]
                .last()
                .sort_index()
                .astype(np.float32)
            )
        _KRAKEN_FUNDING_EXPORT_CACHE[cache_key] = cached
    if cached.empty:
        return pd.Series(dtype=np.float32)
    start = pd.to_datetime(int(since_ms), unit="ms", utc=True)
    end = pd.to_datetime(int(until_ms), unit="ms", utc=True)
    return cached[(cached.index >= start) & (cached.index < end)].astype(np.float32)


def _kraken_futures_product_id(exchange, perp_symbol: str) -> str:
    try:
        market = exchange.market(perp_symbol)
        market_id = str(market.get("id") or "").strip()
        if market_id:
            return market_id
    except Exception:
        pass
    text = str(perp_symbol or "").strip()
    if "/" in text:
        base, quote = text.split("/", 1)
        quote = quote.split(":", 1)[0]
        base = "XBT" if base.upper() == "BTC" else base.upper()
        return f"PF_{base}{quote.upper()}"
    return text.replace("/", "").replace(":", "").upper()


def _drop_suspicious_zero_volume_carry_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "volume" not in df.columns:
        return df
    volume = pd.to_numeric(df["volume"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    zero_no_trade = volume.eq(0.0) & open_.eq(close)
    invalid_zero = volume.isna() | volume.lt(0.0) | (volume.eq(0.0) & ~open_.eq(close))
    prev_linked = zero_no_trade & zero_no_trade.shift(1, fill_value=False) & close.shift(1).eq(open_)
    next_linked = zero_no_trade & zero_no_trade.shift(-1, fill_value=False) & close.eq(open_.shift(-1))
    suspicious = invalid_zero | (zero_no_trade & (prev_linked | next_linked))
    return df.loc[~suspicious]


def _fetch_kraken_futures_chart_ohlcv(
    exchange,
    perp_symbol: str,
    tick_type: str,
    since_ms: int,
    until_ms: int,
    *,
    timeframe: str = "1h",
) -> pd.DataFrame:
    product_id = _kraken_futures_product_id(exchange, perp_symbol)
    if not product_id:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame_ms = max(_timeframe_ms(timeframe), 1)
    frame_seconds = max(1, int(frame_ms // 1000))
    cursor_s = max(0, int(since_ms // 1000))
    until_s = max(cursor_s, int((until_ms + 999) // 1000))
    # The charts endpoint caps responses around 2,000 candles. Keep windows
    # below that cap and use explicit from/to seconds, which Kraken documents
    # through the Charts API family and accepts on the public endpoint.
    chunk_seconds = max(frame_seconds, frame_seconds * 1800)
    url = f"https://futures.kraken.com/api/charts/v1/{tick_type}/{product_id}/{timeframe}"
    headers = {"User-Agent": _ARCHIVE_USER_AGENT}
    rows: list[dict[str, Any]] = []
    while cursor_s < until_s:
        end_s = min(cursor_s + chunk_seconds, until_s)
        try:
            response = _public_data_session().get(
                url,
                params={"from": cursor_s, "to": end_s},
                timeout=30,
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            tprint(
                f"WARN Kraken {tick_type} chart fetch failed for {perp_symbol}: {exc}"
            )
            break
        candles = payload.get("candles", []) if isinstance(payload, dict) else []
        max_seen_s = cursor_s
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            ts_raw = candle.get("time")
            try:
                ts_ms = int(float(ts_raw))
            except Exception:
                continue
            if ts_ms < int(since_ms) or ts_ms >= int(until_ms):
                continue
            ts = pd.to_datetime(ts_ms, unit="ms", utc=True).floor("h")
            row: dict[str, Any] = {"ts": ts}
            ok = True
            for col in ("open", "high", "low", "close"):
                try:
                    row[col] = float(candle.get(col))
                except Exception:
                    ok = False
                    break
            if ok:
                try:
                    row["volume"] = float(candle.get("volume"))
                except Exception:
                    if row["open"] == row["close"]:
                        row["volume"] = 0.0
                    else:
                        ok = False
            if ok and (row["volume"] < 0.0 or (row["volume"] == 0.0 and row["open"] != row["close"])):
                ok = False
            if ok:
                rows.append(row)
                max_seen_s = max(max_seen_s, int(ts_ms // 1000))
        if max_seen_s <= cursor_s:
            cursor_s = end_s + frame_seconds
        else:
            cursor_s = max_seen_s + frame_seconds
        time.sleep(float(getattr(exchange, "rateLimit", 100)) / 1000.0)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).drop_duplicates("ts").set_index("ts").sort_index()
    df = _drop_suspicious_zero_volume_carry_rows(df)
    return df.astype(np.float32)


def _coerce_kraken_chart_timestamp_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            raw = float(value)
            if not np.isfinite(raw):
                return None
            if raw > 10_000_000_000:
                return int(raw)
            return int(raw * 1000)
        text = str(value).strip()
        if not text:
            return None
        if re.fullmatch(r"-?\d+(\.\d+)?", text):
            raw = float(text)
            if raw > 10_000_000_000:
                return int(raw)
            return int(raw * 1000)
        ts = pd.Timestamp(text, tz="UTC")
        return int(ts.value // 10**6)
    except Exception:
        return None


def _extract_kraken_chart_rows(payload: Any) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    timestamp_keys = (
        "time",
        "timestamp",
        "date",
        "datetime",
        "ts",
        "x",
        "bucket",
    )
    value_keys = (
        "openInterest",
        "open_interest",
        "openInterestValue",
        "value",
        "y",
        "close",
    )

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            ts_values = node.get("timestamp")
            data_values = node.get("data")
            if isinstance(ts_values, list) and isinstance(data_values, list):
                for ts_raw, data_raw in zip(ts_values, data_values):
                    ts = _coerce_kraken_chart_timestamp_ms(ts_raw)
                    val = None
                    if isinstance(data_raw, (list, tuple)):
                        for item in reversed(data_raw):
                            try:
                                candidate = float(item)
                            except Exception:
                                continue
                            if np.isfinite(candidate):
                                val = candidate
                                break
                    else:
                        try:
                            candidate = float(data_raw)
                            if np.isfinite(candidate):
                                val = candidate
                        except Exception:
                            val = None
                    if ts is not None and val is not None:
                        rows.append((ts, val))
                return
            ts = None
            val = None
            for key in timestamp_keys:
                if key in node:
                    ts = _coerce_kraken_chart_timestamp_ms(node.get(key))
                    if ts is not None:
                        break
            for key in value_keys:
                if key in node:
                    try:
                        val = float(node.get(key))
                    except Exception:
                        val = None
                    if val is not None and np.isfinite(val):
                        break
            if ts is not None and val is not None and np.isfinite(val):
                rows.append((ts, val))
                return
            for value in node.values():
                visit(value)
            return
        if isinstance(node, (list, tuple)):
            if len(node) >= 2:
                ts = _coerce_kraken_chart_timestamp_ms(node[0])
                val = None
                for item in node[1:]:
                    try:
                        candidate = float(item)
                    except Exception:
                        continue
                    if np.isfinite(candidate):
                        val = candidate
                        break
                if ts is not None and val is not None:
                    rows.append((ts, val))
                    return
            for item in node:
                visit(item)

    visit(payload)
    return rows


def _fetch_kraken_futures_open_interest_analytics(
    exchange,
    perp_symbol: str,
    since_ms: int,
    until_ms: int,
    *,
    timeframe: str = "1h",
) -> pd.Series:
    product_id = _kraken_futures_product_id(exchange, perp_symbol)
    if not product_id:
        return pd.Series(dtype=np.float32)
    base_url = (
        "https://futures.kraken.com/api/charts/v1/analytics/"
        f"{product_id}/open-interest"
    )
    frame_ms = _timeframe_ms(timeframe)
    interval_seconds = max(60, int(frame_ms // 1000))
    headers = {"User-Agent": _ARCHIVE_USER_AGENT}
    cursor_ms = int(max(0, since_ms))
    rows: list[tuple[int, float]] = []
    empty_advance_ms = 2000 * frame_ms
    while cursor_ms < int(until_ms):
        params = {
            "since": int(cursor_ms // 1000),
            "interval": interval_seconds,
        }
        response = None
        for attempt in range(4):
            try:
                response = _public_data_session().get(
                    base_url,
                    params=params,
                    timeout=30,
                    headers=headers,
                )
                if response.status_code == 429:
                    time.sleep(2.0 + attempt)
                    continue
                response.raise_for_status()
                break
            except Exception as exc:
                if attempt >= 3:
                    tprint(
                        f"WARN Kraken OI analytics fetch failed for {perp_symbol} "
                        f"({product_id}): {exc}"
                    )
                    response = None
                    break
                time.sleep(1.0 + attempt)
        if response is None:
            break
        try:
            payload = response.json()
        except Exception as exc:
            tprint(
                f"WARN Kraken OI analytics JSON parse failed for {perp_symbol} "
                f"({product_id}): {exc}"
            )
            break
        batch_rows = [
            (ts, val)
            for ts, val in _extract_kraken_chart_rows(payload)
            if since_ms <= ts < until_ms and np.isfinite(val)
        ]
        if batch_rows:
            rows.extend(batch_rows)
            max_seen = max(ts for ts, _ in batch_rows)
            next_cursor = max_seen + frame_ms
        else:
            next_cursor = cursor_ms + empty_advance_ms
        result = payload.get("result") if isinstance(payload, dict) else None
        more = bool(result.get("more")) if isinstance(result, dict) else True
        if not more and not batch_rows:
            break
        if next_cursor <= cursor_ms:
            next_cursor = cursor_ms + frame_ms
        cursor_ms = next_cursor
        time.sleep(float(getattr(exchange, "rateLimit", 100)) / 1000.0)
    if not rows:
        return pd.Series(dtype=np.float32)
    df = pd.DataFrame(rows, columns=["ts", "value"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("h")
    return df.groupby("ts")["value"].last().sort_index().astype(np.float32)


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


def _kraken_charts_resolution(timeframe: str) -> str:
    text = str(timeframe or "1h").strip().lower()
    if text.endswith("m"):
        minutes = int(float(text[:-1] or 1))
        return f"{minutes}m"
    if text.endswith("h"):
        hours = int(float(text[:-1] or 1))
        return f"{hours}h"
    if text.endswith("d"):
        days = int(float(text[:-1] or 1))
        return f"{days}d"
    try:
        minutes = int(float(text))
        return f"{minutes}m"
    except Exception:
        return "1h"


def _fetch_kraken_futures_charts_ohlcv(
    exchange,
    symbol: str,
    since_ms: int,
    until_ms: int,
    *,
    timeframe: str = "1h",
    tick_type: str = "trade",
) -> pd.DataFrame:
    product_id = _kraken_futures_product_id(exchange, symbol)
    if not product_id:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume"]
        ).set_index(pd.DatetimeIndex([], tz="UTC", name="ts"))
    tick = str(tick_type or "trade")
    resolution = _kraken_charts_resolution(timeframe)
    url = (
        "https://futures.kraken.com/api/charts/v1/"
        f"{tick}/{product_id}/{resolution}"
    )
    params = {
        "from": int(max(0, since_ms) // 1000),
        "to": int(max(0, until_ms) // 1000),
    }
    response = _public_data_session().get(
        url,
        params=params,
        timeout=60,
        headers={"User-Agent": _ARCHIVE_USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    candles = payload.get("candles") if isinstance(payload, dict) else None
    if not isinstance(candles, list) or not candles:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume"]
        ).set_index(pd.DatetimeIndex([], tz="UTC", name="ts"))

    rows = []
    for candle in candles:
        if not isinstance(candle, dict):
            continue
        ts = _coerce_kraken_chart_timestamp_ms(candle.get("time"))
        if ts is None or ts < since_ms or ts >= until_ms:
            continue
        try:
            open_ = float(candle.get("open"))
            high = float(candle.get("high"))
            low = float(candle.get("low"))
            close = float(candle.get("close"))
            try:
                volume = float(candle.get("volume"))
            except Exception:
                if open_ == close:
                    volume = 0.0
                else:
                    continue
            if volume < 0.0 or (volume == 0.0 and open_ != close):
                continue
            rows.append(
                (
                    ts,
                    open_,
                    high,
                    low,
                    close,
                    volume,
                )
            )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume"]
        ).set_index(pd.DatetimeIndex([], tz="UTC", name="ts"))

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    # Kraken Futures explicitly returns zero-volume 1m carry candles for
    # inactive minutes. Keep them for delayed-entry execution proxies so an
    # illiquid but published minute does not silently fall back to 15m open.
    if _timeframe_ms(timeframe) > 60_000:
        df = _drop_suspicious_zero_volume_carry_rows(df)
    return df.astype(np.float32)


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
    exchange_id = str(getattr(exchange, "id", "") or "").lower()
    if "krakenfutures" in exchange_id:
        price_name = str((params or {}).get("price") or "trade")
        tick_type = "trade" if price_name in {"", "trade", "last"} else price_name
        return _fetch_kraken_futures_charts_ohlcv(
            exchange,
            symbol,
            since_ms,
            until_ms,
            timeframe=timeframe,
            tick_type=tick_type,
        )

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


def _timeframe_ms(timeframe: str) -> int:
    text = str(timeframe or "1h").strip().lower()
    if text.endswith("m"):
        return int(float(text[:-1] or 1) * 60_000)
    if text.endswith("h"):
        return int(float(text[:-1] or 1) * 3_600_000)
    if text.endswith("d"):
        return int(float(text[:-1] or 1) * 86_400_000)
    # Kraken spot CCXT uses numeric-minute timeframes in a few call paths.
    try:
        return int(float(text) * 60_000)
    except Exception:
        return 3_600_000


def _download_backfill_internal_gaps_enabled() -> bool:
    raw = os.getenv("EPM_DOWNLOAD_BACKFILL_INTERNAL_GAPS", "1")
    return str(raw).strip().lower() not in {"0", "false", "no", "n", "off"}


def _download_tail_only_enabled() -> bool:
    raw = os.getenv("EPM_DOWNLOAD_TAIL_ONLY", "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _partition_compact_min_delta_rows() -> int:
    raw = os.getenv("EPM_PARTITION_COMPACT_MIN_DELTA_ROWS", "200")
    try:
        return max(0, int(float(raw)))
    except Exception:
        return 200


def _perp_side_data_enabled() -> bool:
    raw = os.getenv("EPM_PERP_SIDE_DATA_ENABLED", "1")
    return str(raw).strip().lower() not in {"0", "false", "no", "n", "off"}


def _has_internal_time_gaps(index: pd.DatetimeIndex, timeframe: str) -> bool:
    if index is None or len(index) < 2:
        return False
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = idx.dropna().drop_duplicates().sort_values()
    if len(idx) < 2:
        return False
    frame_ms = max(_timeframe_ms(timeframe), 1)
    vals = idx.view("i8") // 1_000_000
    return bool(np.any(np.diff(vals) > int(frame_ms * 1.5)))


def _ohlcv_fetch_profile(exchange, timeframe: str, requested_limit: int) -> tuple[int, int]:
    exchange_id = str(getattr(exchange, "id", "") or "").lower()
    frame_ms = max(_timeframe_ms(timeframe), 1)
    limit = int(requested_limit or 1000)
    if "krakenfutures" in exchange_id:
        # Kraken Futures charts v1 caps responses at 2,000 candles. Keep each
        # outer window within that cap; larger windows silently return only the
        # first 2,000 candles and create deterministic internal gaps.
        limit = 2000
        chunk_ms = max(limit * frame_ms, frame_ms)
        return limit, chunk_ms
    env_days = os.getenv("EPM_OHLCV_CHUNK_DAYS")
    if env_days:
        try:
            return limit, max(int(pd.Timedelta(days=float(env_days)).total_seconds() * 1000), frame_ms)
        except Exception:
            pass
    return limit, int(pd.Timedelta(days=7).total_seconds() * 1000)


def _kraken_spot_ohlcv_floor_ms(exchange, now_ms: int, timeframe: str) -> int | None:
    exchange_id = str(getattr(exchange, "id", "") or "").lower()
    if exchange_id != "kraken":
        return None
    # Kraken spot /public/OHLC returns only the most recent 720 candles,
    # regardless of `since`. Older spot backfill needs another data source.
    return int(now_ms) - (720 * _timeframe_ms(timeframe))


def fetch_ohlcv_all_7d_chunks(
    exchange,
    symbol,
    since_ms,
    timeframe="1h",
    limit=1000,
    params: Optional[dict] = None,
):
    limit, chunk_ms = _ohlcv_fetch_profile(exchange, timeframe, limit)
    now_ms = int(pd.Timestamp.utcnow().value // 10**6)

    start = int(since_ms)
    history_days_raw = str(os.getenv("EPM_OHLCV_HISTORY_DAYS", "")).strip()
    if history_days_raw:
        try:
            floor_ms = _recent_history_floor_ms("EPM_OHLCV_HISTORY_DAYS", float(history_days_raw))
            start = max(start, int(floor_ms))
        except Exception:
            pass
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


def _perp_auxiliary_backfill_enabled() -> bool:
    raw = os.getenv("EPM_PERP_BACKFILL_MISSING_AUX", "1")
    return str(raw).strip().lower() not in {"0", "false", "no", "n", "off"}


def _perp_side_history_days(default_days: float = 3650.0) -> float:
    raw = os.getenv(
        "EPM_PERP_SIDE_HISTORY_DAYS",
        os.getenv("EPM_PERP_AUX_HISTORY_DAYS", str(default_days)),
    )
    try:
        return max(float(raw), 0.0)
    except Exception:
        return float(default_days)


def _has_sparse_perp_auxiliary_data(
    frame: pd.DataFrame,
    *,
    since_ms: int,
    timeframe: str,
) -> bool:
    """Return True when OHLCV timestamps exist but perp auxiliary data is absent.

    Incremental downloads used to consider a symbol complete once close candles
    were present. That leaves mark/OI/funding/spot columns permanently empty if
    side-data fetching was added later or failed transiently, which then leaks
    into trained feature contracts. Treat those rows as incomplete so forced or
    completeness-checked downloads revisit the historical chunks.
    """
    if frame is None or frame.empty:
        return False
    if not _perp_auxiliary_backfill_enabled() or not _perp_side_data_enabled():
        return False
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx, utc=True, errors="coerce")
        except Exception:
            return False
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    since_ts = pd.to_datetime(int(since_ms), unit="ms", utc=True)
    window_mask = idx >= since_ts
    if not bool(np.any(window_mask)):
        return False
    check_mask = np.asarray(window_mask, dtype=bool)
    close_mask = np.ones(len(frame), dtype=bool)
    if "close" in frame.columns:
        close_vals = pd.to_numeric(frame["close"], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        close_mask = np.isfinite(close_vals)
    check_mask = check_mask & close_mask
    if not bool(np.any(check_mask)):
        return False
    min_coverage = float(os.getenv("EPM_PERP_AUX_MIN_COVERAGE", "0.80") or 0.80)
    min_coverage = float(np.clip(min_coverage, 0.0, 1.0))

    idx_ms = idx.view("i8") // 1_000_000
    required_groups = (
        (
            "mark",
            ("mark_close", "mark_price"),
            _recent_history_floor_ms("EPM_MARK_HISTORY_DAYS", _perp_side_history_days()),
        ),
        (
            "spot",
            ("spot_close",),
            _recent_history_floor_ms("EPM_SPOT_HISTORY_DAYS", _perp_side_history_days()),
        ),
        (
            "funding",
            ("funding_rate",),
            _recent_history_floor_ms("EPM_FUNDING_HISTORY_DAYS", _perp_side_history_days()),
        ),
        (
            "open_interest",
            ("open_interest",),
            _recent_history_floor_ms(
                "EPM_OPEN_INTEREST_HISTORY_DAYS", _perp_side_history_days()
            ),
        ),
    )
    sparse_groups: list[str] = []
    group_coverages: dict[str, float] = {}
    for group_name, cols, floor_ms in required_groups:
        group_check = check_mask.copy()
        if floor_ms is not None:
            group_check &= idx_ms >= int(floor_ms)
        if not bool(np.any(group_check)):
            continue
        present = [col for col in cols if col in frame.columns]
        if not present:
            sparse_groups.append(group_name)
            group_coverages[group_name] = 0.0
            continue
        finite_union = np.zeros(len(frame), dtype=bool)
        for col in present:
            vals = pd.to_numeric(frame.loc[group_check, col], errors="coerce")
            finite_union[group_check] |= np.isfinite(
                vals.to_numpy(dtype=np.float64, copy=False)
            )
        coverage = float(finite_union[group_check].mean())
        group_coverages[group_name] = coverage
        if coverage < min_coverage:
            sparse_groups.append(group_name)
    if sparse_groups:
        tprint(
            "Detected sparse perp auxiliary data "
            f"groups={sparse_groups} coverage={group_coverages}; "
            "backfilling side-data history."
        )
        return True
    return False


def _perp_auxiliary_missing_ranges(
    frame: pd.DataFrame,
    *,
    since_ms: int,
    timeframe: str,
) -> list[tuple[int, int]]:
    """Return contiguous timestamp ranges where existing OHLCV lacks aux columns."""
    if frame is None or frame.empty:
        return []
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx, utc=True, errors="coerce")
        except Exception:
            return []
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    frame = frame.copy()
    frame.index = idx
    since_ts = pd.to_datetime(int(since_ms), unit="ms", utc=True)
    check = idx >= since_ts
    if "close" in frame.columns:
        close_vals = pd.to_numeric(frame["close"], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        check &= np.isfinite(close_vals)
    if not bool(np.any(check)):
        return []

    idx_ms = idx.view("i8") // 1_000_000
    required_groups = (
        (
            ("mark_close", "mark_price"),
            _recent_history_floor_ms("EPM_MARK_HISTORY_DAYS", _perp_side_history_days()),
        ),
        (
            ("spot_close",),
            _recent_history_floor_ms("EPM_SPOT_HISTORY_DAYS", _perp_side_history_days()),
        ),
        (
            ("funding_rate",),
            _recent_history_floor_ms("EPM_FUNDING_HISTORY_DAYS", _perp_side_history_days()),
        ),
        (
            ("open_interest",),
            _recent_history_floor_ms(
                "EPM_OPEN_INTEREST_HISTORY_DAYS", _perp_side_history_days()
            ),
        ),
    )
    missing = np.zeros(len(frame), dtype=bool)
    for cols, floor_ms in required_groups:
        group_check = check.copy()
        if floor_ms is not None:
            group_check &= idx_ms >= int(floor_ms)
        if not bool(np.any(group_check)):
            continue
        present = [col for col in cols if col in frame.columns]
        if not present:
            missing |= group_check
            continue
        group_finite = np.zeros(len(frame), dtype=bool)
        for col in present:
            vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(
                dtype=np.float64, copy=False
            )
            group_finite |= np.isfinite(vals)
        missing |= group_check & (~group_finite)

    missing_idx = idx[missing]
    if len(missing_idx) == 0:
        return []

    frame_ms = max(_timeframe_ms(timeframe), 1)
    values_ms = missing_idx.view("i8") // 1_000_000
    ranges: list[tuple[int, int]] = []
    start_ms = int(values_ms[0])
    prev_ms = int(values_ms[0])
    # Merge valid islands inside a Kraken charts-sized request. Sparse aux data
    # often alternates between available and missing hourly rows; treating each
    # island as a separate backfill range makes the official endpoints the
    # bottleneck without improving parity.
    merge_gap_ms = max(frame_ms * 1800, frame_ms)
    for raw_ms in values_ms[1:]:
        current_ms = int(raw_ms)
        if current_ms - prev_ms <= merge_gap_ms:
            prev_ms = current_ms
            continue
        ranges.append((max(int(since_ms), start_ms), prev_ms + frame_ms))
        start_ms = current_ms
        prev_ms = current_ms
    ranges.append((max(int(since_ms), start_ms), prev_ms + frame_ms))
    return [(s, e) for s, e in ranges if e > s]


def _enrich_perp_auxiliary_chunk(
    *,
    chunk: pd.DataFrame,
    exchange,
    symbol: str,
    perp_symbol: str,
    spot_exchange,
    timeframe: str,
    now_ms: int,
    side_data_enabled: bool,
    disabled_extra_ohlcv: set[str],
    supports_oi_history: bool,
    exchange_id: str,
) -> pd.DataFrame:
    if chunk is None or chunk.empty:
        return pd.DataFrame()
    chunk = chunk.sort_index().copy()
    chunk = chunk[~chunk.index.duplicated(keep="last")]
    chunk_start_ms = int(chunk.index.min().value // 10**6)
    chunk_end_ms = int((chunk.index.max() + pd.Timedelta(milliseconds=_timeframe_ms(timeframe))).value // 10**6)
    chunk_end_ms = min(chunk_end_ms, now_ms)

    funding = pd.Series(dtype=np.float32)
    funding_floor_ms = _recent_history_floor_ms(
        "EPM_FUNDING_HISTORY_DAYS", _perp_side_history_days()
    )
    if side_data_enabled and "krakenfutures" in exchange_id and chunk_end_ms >= funding_floor_ms:
        funding = _fetch_kraken_futures_historical_funding_rates(
            exchange,
            perp_symbol,
            max(chunk_start_ms, funding_floor_ms),
            chunk_end_ms,
        )
    if (
        funding.empty
        and
        side_data_enabled
        and hasattr(exchange, "fetch_funding_rate_history")
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
    oi_floor_ms = _recent_history_floor_ms(
        "EPM_OPEN_INTEREST_HISTORY_DAYS", _perp_side_history_days()
    )
    if side_data_enabled and "krakenfutures" in exchange_id and chunk_end_ms >= oi_floor_ms:
        oi = _fetch_kraken_futures_open_interest_analytics(
            exchange,
            perp_symbol,
            max(chunk_start_ms, oi_floor_ms),
            chunk_end_ms,
            timeframe=timeframe,
        )
    if (
        oi.empty
        and side_data_enabled
        and supports_oi_history
        and hasattr(exchange, "fetch_open_interest_history")
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
            timeframe=timeframe,
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

    price_sources = (
        (("mark", "mark"), ("spot", "spot"))
        if "krakenfutures" in exchange_id
        else (
            ("mark", "mark"),
            ("index", "index"),
            ("premiumIndex", "premium_index"),
        )
    )
    for price_name, prefix in price_sources:
        if not side_data_enabled:
            break
        if price_name in disabled_extra_ohlcv:
            continue
        try:
            if "krakenfutures" in exchange_id:
                price_df = _fetch_kraken_futures_chart_ohlcv(
                    exchange,
                    perp_symbol,
                    price_name,
                    chunk_start_ms,
                    chunk_end_ms,
                    timeframe=timeframe,
                )
            else:
                price_df = _fetch_ohlcv_paged(
                    exchange,
                    perp_symbol,
                    chunk_start_ms,
                    chunk_end_ms,
                    timeframe=timeframe,
                    limit=1000,
                    params={"price": price_name},
                )
            _align_ohlcv(price_df, prefix)
        except Exception as exc:
            if "Invalid tick type" in str(exc):
                disabled_extra_ohlcv.add(price_name)
            tprint(f"WARN perp {price_name} OHLCV fetch failed for {symbol}: {exc}")

    if side_data_enabled and spot_exchange is not None and "spot_close" not in chunk.columns:
        try:
            spot_floor_ms = _kraken_spot_ohlcv_floor_ms(spot_exchange, now_ms, timeframe)
            if spot_floor_ms is None or chunk_end_ms > spot_floor_ms:
                spot_symbol = None
                if "/" in symbol:
                    base, raw_quote = symbol.split("/", 1)
                    quote = raw_quote.split(":", 1)[0].upper()
                    spot_candidates = []
                    for candidate_quote in ("USDC", "USD", "USDT", quote):
                        candidate = f"{base}/{candidate_quote}"
                        if candidate not in spot_candidates:
                            spot_candidates.append(candidate)
                    for candidate in spot_candidates:
                        if candidate in getattr(spot_exchange, "markets", {}):
                            spot_symbol = candidate
                            break
                if spot_symbol is None and symbol in getattr(spot_exchange, "markets", {}):
                    spot_symbol = symbol
                if spot_symbol is None:
                    normalized = _normalize_spot_symbol(symbol)
                    if normalized in getattr(spot_exchange, "markets", {}):
                        spot_symbol = normalized
                if spot_symbol is not None:
                    spot_chunk_start_ms = chunk_start_ms
                    if spot_floor_ms is not None:
                        spot_chunk_start_ms = max(spot_chunk_start_ms, spot_floor_ms)
                    spot_df = _fetch_ohlcv_paged(
                        spot_exchange,
                        spot_symbol,
                        spot_chunk_start_ms,
                        chunk_end_ms,
                        timeframe=timeframe,
                        limit=1000,
                    )
                    _align_ohlcv(spot_df, "spot")
        except Exception as exc:
            tprint(f"WARN spot auxiliary OHLCV fetch failed for {symbol}: {exc}")

    chunk["funding_rate"] = funding.reindex(chunk.index).ffill().astype(np.float32)
    chunk["open_interest"] = oi.reindex(chunk.index).ffill().astype(np.float32)
    if "mark_close" in chunk.columns:
        chunk["mark_price"] = chunk["mark_close"]
    if "index_close" in chunk.columns:
        chunk["index_price"] = chunk["index_close"]
    if "premium_index_close" in chunk.columns:
        chunk["premium_index"] = chunk["premium_index_close"]
    return chunk


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

    def _latest_stored_ts_ms(self, symbol: str) -> int:
        """Return the latest timestamp visible from partition filenames."""
        sym_dir = self._get_symbol_dir(symbol)
        latest_s = 0
        if not os.path.exists(sym_dir):
            return 0
        for fpath in glob.glob(os.path.join(sym_dir, "year=*", "*.parquet")):
            base = os.path.basename(fpath).replace(".parquet", "")
            parts = base.split("-")
            if len(parts) < 3:
                continue
            try:
                latest_s = max(latest_s, int(parts[-1]))
            except ValueError:
                continue
        return int(latest_s * 1000) if latest_s > 0 else 0

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

            try:
                from extreme_price_movements.kraken_actual_data import (
                    overlay_actual_volume_sidecar,
                )

                df = overlay_actual_volume_sidecar(
                    df,
                    root_dir=self.root_dir,
                    symbol=symbol,
                )
            except Exception as exc:
                tprint(f"WARN actual volume overlay skipped for {symbol}: {exc}")

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

        self._export_open_interest_sidecar(symbol, df_reset)

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

        try:
            last_ts = pd.to_datetime(df_reset["ts"], utc=True).max()
            if pd.notna(last_ts):
                self._write_meta(symbol, {"last_ts_ms": int(last_ts.value // 10**6)})
        except Exception:
            pass

    def _export_open_interest_sidecar(self, symbol: str, df_reset: pd.DataFrame) -> None:
        """Mirror embedded perp OI into the dedicated sidecar used by features/live."""
        if (
            df_reset is None
            or df_reset.empty
            or "ts" not in df_reset.columns
            or "open_interest" not in df_reset.columns
        ):
            return
        try:
            oi = pd.to_numeric(df_reset["open_interest"], errors="coerce")
            mask = np.isfinite(oi.to_numpy(dtype=np.float64, copy=False)) & (oi > 0.0)
            if not bool(np.any(mask)):
                return
            key = str(symbol).replace("/", "_").replace(":", "_")
            sidecar_dir = os.path.join(self.root_dir, "open_interest_hourly")
            os.makedirs(sidecar_dir, exist_ok=True)
            path = os.path.join(sidecar_dir, f"{key}.parquet")
            incoming = pd.DataFrame(
                {
                    "open_interest": oi.loc[mask].astype(np.float32).to_numpy(),
                },
                index=pd.DatetimeIndex(
                    pd.to_datetime(df_reset.loc[mask, "ts"], utc=True),
                    name="ts",
                ).floor("h"),
            )
            incoming = incoming[~incoming.index.duplicated(keep="last")].sort_index()
            if incoming.empty:
                return
            if os.path.exists(path):
                try:
                    existing = pd.read_parquet(path)
                    if not existing.empty:
                        existing.index = pd.to_datetime(
                            existing.index, utc=True, errors="coerce"
                        ).floor("h")
                        if "open_interest" in existing.columns:
                            existing = existing[["open_interest"]]
                            existing["open_interest"] = pd.to_numeric(
                                existing["open_interest"], errors="coerce"
                            ).astype(np.float32)
                            incoming = (
                                pd.concat([existing, incoming])
                                .sort_index()
                                .groupby(level=0)
                                .last()
                            )
                except Exception as exc:
                    tprint(
                        f"WARN open-interest sidecar merge skipped for {symbol}: {exc}"
                    )
            tmp_path = path + ".tmp"
            incoming.to_parquet(tmp_path, engine="pyarrow", compression="zstd")
            os.replace(tmp_path, path)
        except Exception as exc:
            tprint(f"WARN open-interest sidecar export failed for {symbol}: {exc}")

    def compact_partition(self, symbol: str, year: int):
        sym_dir = self._get_symbol_dir(symbol)
        part_dir = os.path.join(sym_dir, f"year={year}")

        if not os.path.exists(part_dir):
            return

        files = glob.glob(os.path.join(part_dir, "*.parquet"))
        if not files:
            return
        compact_files = [
            f for f in files if os.path.basename(f).startswith("compact-")
        ]
        part_files = [
            f for f in files if os.path.basename(f).startswith("part-")
        ]
        min_delta_rows = _partition_compact_min_delta_rows()
        if compact_files and part_files and min_delta_rows > 0:
            delta_rows = 0
            for f in part_files:
                try:
                    delta_rows += int(pq.ParquetFile(f).metadata.num_rows)
                except Exception:
                    # If metadata is unreadable, fall through to full compaction.
                    delta_rows = min_delta_rows
                    break
            if delta_rows < min_delta_rows:
                tprint(
                    f"Deferring compaction for {symbol} {year}: "
                    f"delta_rows={delta_rows} < {min_delta_rows} "
                    f"part_files={len(part_files)}"
                )
                return
        # Compact files are the historical base; part files are incremental
        # updates. Read part files last so duplicate timestamp merges preserve
        # the newest auxiliary backfill values deterministically.
        files = sorted(
            files,
            key=lambda path: (
                0 if os.path.basename(path).startswith("compact-") else 1,
                os.path.getmtime(path),
                path,
            ),
        )

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
            frame_ns = max(_timeframe_ms(self.timeframe), 1) * 1_000_000
            ts_ns = merged["ts"].astype("int64")
            aligned = (ts_ns % frame_ns) == 0
            dropped = int((~aligned).sum())
            if dropped:
                tprint(
                    f"Dropping {dropped} off-{self.timeframe}-grid rows during "
                    f"compaction for {symbol} {year}"
                )
                merged = merged.loc[aligned].copy()
            if merged.empty:
                return
            # Incremental auxiliary backfills write full rows with newly
            # populated columns. Merge duplicate timestamps column-wise so an
            # older OHLCV row cannot discard newer sparse auxiliary values.
            merged = (
                merged.sort_values("ts", kind="mergesort")
                .groupby("ts", as_index=False, sort=True)
                .last()
            )
            aux_cols = ["funding_rate", "open_interest"]
            present_aux_cols = [col for col in aux_cols if col in merged.columns]
            if present_aux_cols:
                merged[present_aux_cols] = merged[present_aux_cols].ffill()

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
            existing_idx = pd.DatetimeIndex([])

            if _download_tail_only_enabled() and last_ts_ms > 0:
                latest_stored_ms = self._latest_stored_ts_ms(symbol)
                if latest_stored_ms > last_ts_ms:
                    self._write_meta(symbol, {"last_ts_ms": int(latest_stored_ms)})
                    last_ts_ms = int(latest_stored_ms)
                start_ms = int(last_ts_ms) + 1
            elif last_ts_ms > 0:
                existing_frame = self.load(symbol)
                existing_idx = existing_frame.index
                if _has_sparse_perp_auxiliary_data(
                    existing_frame,
                    since_ms=since_ms,
                    timeframe=self.timeframe,
                ):
                    auxiliary_only_ranges = _perp_auxiliary_missing_ranges(
                        existing_frame,
                        since_ms=since_ms,
                        timeframe=self.timeframe,
                    )
                    tprint(
                        f"Detected incomplete perp auxiliary data for {symbol}; "
                        f"backfilling {len(auxiliary_only_ranges)} missing aux range(s)"
                    )
                    start_ms = last_ts_ms + 1
                elif (
                    _download_backfill_internal_gaps_enabled()
                    and not existing_idx.empty
                    and _has_internal_time_gaps(existing_idx, self.timeframe)
                ):
                    tprint(
                        f"Detected internal OHLCV gaps for {symbol}; "
                        "backfilling from requested start instead of last_ts_ms"
                    )
                    start_ms = since_ms
                else:
                    start_ms = last_ts_ms + 1
            else:
                # Fallback to load index if no meta
                # Here load() without args is fine, but we might want to check just the last file?
                # For simplicity, keep as is, but maybe optimize load(columns=['ts'])
                if existing_idx.empty:
                    existing_idx = self.load(symbol, columns=["ts"]).index
                if (
                    _download_backfill_internal_gaps_enabled()
                    and not existing_idx.empty
                    and _has_internal_time_gaps(existing_idx, self.timeframe)
                ):
                    tprint(
                        f"Detected internal OHLCV gaps for {symbol}; "
                        "backfilling from requested start"
                    )
                    start_ms = since_ms
                elif not existing_idx.empty:
                    last_ts = existing_idx.max()
                    start_ms = int(last_ts.value // 10**6) + 1
                    if int(existing_idx.min().value // 10**6) > since_ms:
                        start_ms = since_ms
                else:
                    start_ms = since_ms

            now_ms = int(pd.Timestamp.utcnow().value // 10**6)
            spot_floor_ms = _kraken_spot_ohlcv_floor_ms(exchange, now_ms, self.timeframe)
            if spot_floor_ms is not None and start_ms < spot_floor_ms:
                floor_dt = pd.to_datetime(spot_floor_ms, unit="ms", utc=True).strftime(
                    "%Y-%m-%d %H:%M"
                )
                tprint(
                    "Kraken spot OHLC is limited to the most recent 720 candles; "
                    f"clamping {symbol} spot start to {floor_dt}"
                )
                start_ms = spot_floor_ms

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
            existing_idx = pd.DatetimeIndex([])
            auxiliary_only_ranges: list[tuple[int, int]] = []

            if _download_tail_only_enabled() and last_ts_ms > 0:
                latest_stored_ms = self._latest_stored_ts_ms(symbol)
                if latest_stored_ms > last_ts_ms:
                    self._write_meta(symbol, {"last_ts_ms": int(latest_stored_ms)})
                    last_ts_ms = int(latest_stored_ms)
                start_ms = int(last_ts_ms) + 1
            elif last_ts_ms > 0:
                existing_frame = self.load(symbol)
                existing_idx = existing_frame.index
                has_internal_ohlcv_gaps = (
                    _download_backfill_internal_gaps_enabled()
                    and not existing_idx.empty
                    and _has_internal_time_gaps(existing_idx, self.timeframe)
                )
                has_sparse_auxiliary = not existing_idx.empty and _has_sparse_perp_auxiliary_data(
                    existing_frame,
                    since_ms=since_ms,
                    timeframe=self.timeframe,
                )
                if has_sparse_auxiliary:
                    auxiliary_only_ranges = _perp_auxiliary_missing_ranges(
                        existing_frame,
                        since_ms=since_ms,
                        timeframe=self.timeframe,
                    )
                    tprint(
                        f"Detected incomplete perp auxiliary data for {symbol}; "
                        f"backfilling {len(auxiliary_only_ranges)} missing aux range(s)"
                    )
                if has_internal_ohlcv_gaps:
                    tprint(
                        f"Detected internal perp OHLCV gaps for {symbol}; "
                        "backfilling from requested start instead of last_ts_ms"
                    )
                    start_ms = since_ms
                elif has_sparse_auxiliary:
                    start_ms = last_ts_ms + 1
                elif (
                    not existing_idx.empty
                    and int(existing_idx.min().value // 10**6) > since_ms
                ):
                    start_ms = since_ms
                else:
                    start_ms = last_ts_ms + 1
            else:
                if existing_idx.empty:
                    existing_frame = self.load(symbol)
                    existing_idx = existing_frame.index
                if (
                    _download_backfill_internal_gaps_enabled()
                    and not existing_idx.empty
                    and _has_internal_time_gaps(existing_idx, self.timeframe)
                ):
                    tprint(
                        f"Detected internal perp OHLCV gaps for {symbol}; "
                        "backfilling from requested start"
                    )
                    start_ms = since_ms
                elif (
                    not existing_idx.empty
                    and _has_sparse_perp_auxiliary_data(
                        existing_frame,
                        since_ms=since_ms,
                        timeframe=self.timeframe,
                    )
                ):
                    auxiliary_only_ranges = _perp_auxiliary_missing_ranges(
                        existing_frame,
                        since_ms=since_ms,
                        timeframe=self.timeframe,
                    )
                    tprint(
                        f"Detected incomplete perp auxiliary data for {symbol}; "
                        f"backfilling {len(auxiliary_only_ranges)} missing aux range(s)"
                    )
                    start_ms = int(existing_idx.max().value // 10**6) + 1
                elif not existing_idx.empty:
                    start_ms = int(existing_idx.max().value // 10**6) + 1
                    if int(existing_idx.min().value // 10**6) > since_ms:
                        start_ms = since_ms
                else:
                    start_ms = since_ms

            now_ms = int(pd.Timestamp.utcnow().value // 10**6)

            perp_symbol = _resolve_perp_symbol(exchange, symbol)
            if not perp_symbol:
                raise ValueError(f"No perp symbol found for {symbol}")

            touched_years = set()
            has_new_data = False
            side_data_enabled = _perp_side_data_enabled()
            exchange_has = getattr(exchange, "has", {}) or {}
            supports_oi_history = bool(exchange_has.get("fetchOpenInterestHistory"))
            disabled_extra_ohlcv: set[str] = set()
            exchange_id = str(getattr(exchange, "id", "") or "").lower()
            if "kraken" in exchange_id:
                # Kraken Futures rejects native index/premium OHLCV tick types for
                # many USD swaps. Avoid retrying known-bad native reference ticks
                # on every symbol/chunk; spot/index gaps remain explicit missing
                # source data for strict feature contracts.
                disabled_extra_ohlcv.update({"index", "premiumIndex"})
            if auxiliary_only_ranges:
                tprint(
                    f"FETCH perp aux gaps: {symbol} ({perp_symbol}) "
                    f"ranges={len(auxiliary_only_ranges)}"
                )
                for range_start_ms, range_end_ms in auxiliary_only_ranges:
                    range_start = pd.to_datetime(range_start_ms, unit="ms", utc=True)
                    range_end = pd.to_datetime(range_end_ms, unit="ms", utc=True)
                    existing_chunk = existing_frame.loc[
                        (existing_frame.index >= range_start)
                        & (existing_frame.index < range_end)
                    ].copy()
                    if existing_chunk.empty:
                        continue
                    chunk = _enrich_perp_auxiliary_chunk(
                        chunk=existing_chunk,
                        exchange=exchange,
                        symbol=symbol,
                        perp_symbol=perp_symbol,
                        spot_exchange=spot_exchange,
                        timeframe=self.timeframe,
                        now_ms=now_ms,
                        side_data_enabled=side_data_enabled,
                        disabled_extra_ohlcv=disabled_extra_ohlcv,
                        supports_oi_history=supports_oi_history,
                        exchange_id=exchange_id,
                    )
                    if chunk.empty:
                        continue
                    fresh = self._downcast(chunk)
                    self.save_partitioned(symbol, fresh, defer_compact=True)
                    touched_years.update(fresh.index.year.unique())
                    has_new_data = True
                for yr in sorted(touched_years):
                    self.compact_partition(symbol, int(yr))
                if start_ms >= now_ms:
                    return self.load(symbol)

            if start_ms >= now_ms:
                return self.load(symbol)

            start_dt = pd.to_datetime(start_ms, unit="ms", utc=True).strftime(
                "%Y-%m-%d %H:%M"
            )
            tprint(f"FETCH perp incr: {symbol} ({perp_symbol}) from {start_dt}")
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
                    "EPM_FUNDING_HISTORY_DAYS", _perp_side_history_days()
                )
                if (
                    side_data_enabled
                    and "krakenfutures" in exchange_id
                    and chunk_end_ms >= funding_floor_ms
                ):
                    funding = _fetch_kraken_futures_historical_funding_rates(
                        exchange,
                        perp_symbol,
                        max(chunk_start_ms, funding_floor_ms),
                        chunk_end_ms,
                    )
                if (
                    funding.empty
                    and
                    side_data_enabled
                    and
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
                oi_floor_ms = _recent_history_floor_ms(
                    "EPM_OPEN_INTEREST_HISTORY_DAYS", _perp_side_history_days()
                )
                if side_data_enabled and "krakenfutures" in exchange_id and chunk_end_ms >= oi_floor_ms:
                    oi = _fetch_kraken_futures_open_interest_analytics(
                        exchange,
                        perp_symbol,
                        max(chunk_start_ms, oi_floor_ms),
                        chunk_end_ms,
                        timeframe=self.timeframe,
                    )
                if (
                    oi.empty
                    and
                    side_data_enabled
                    and
                    supports_oi_history
                    and hasattr(exchange, "fetch_open_interest_history")
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

                price_sources = (
                    (("mark", "mark"), ("spot", "spot"))
                    if "krakenfutures" in exchange_id
                    else (
                        ("mark", "mark"),
                        ("index", "index"),
                        ("premiumIndex", "premium_index"),
                    )
                )
                for price_name, prefix in price_sources:
                    if not side_data_enabled:
                        break
                    if price_name in disabled_extra_ohlcv:
                        continue
                    try:
                        if "krakenfutures" in exchange_id:
                            price_df = _fetch_kraken_futures_chart_ohlcv(
                                exchange,
                                perp_symbol,
                                price_name,
                                chunk_start_ms,
                                chunk_end_ms,
                                timeframe=self.timeframe,
                            )
                        else:
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
                        if "Invalid tick type" in str(exc):
                            disabled_extra_ohlcv.add(price_name)
                        tprint(
                            f"WARN perp {price_name} OHLCV fetch failed for {symbol}: {exc}"
                        )

                if (
                    side_data_enabled
                    and spot_exchange is not None
                    and "spot_close" not in chunk.columns
                ):
                    try:
                        spot_floor_ms = _kraken_spot_ohlcv_floor_ms(
                            spot_exchange, now_ms, self.timeframe
                        )
                        if spot_floor_ms is None or chunk_end_ms > spot_floor_ms:
                            spot_symbol = None
                            if "/" in symbol:
                                base, raw_quote = symbol.split("/", 1)
                                quote = raw_quote.split(":", 1)[0].upper()
                                spot_candidates = []
                                for candidate_quote in ("USDC", "USD", "USDT", quote):
                                    candidate = f"{base}/{candidate_quote}"
                                    if candidate not in spot_candidates:
                                        spot_candidates.append(candidate)
                                for candidate in spot_candidates:
                                    if candidate in getattr(spot_exchange, "markets", {}):
                                        spot_symbol = candidate
                                        break
                            if spot_symbol is None and symbol in getattr(
                                spot_exchange, "markets", {}
                            ):
                                spot_symbol = symbol
                            if spot_symbol is None:
                                normalized = _normalize_spot_symbol(symbol)
                                if normalized in getattr(spot_exchange, "markets", {}):
                                    spot_symbol = normalized
                            if spot_symbol is not None:
                                spot_chunk_start_ms = chunk_start_ms
                                if spot_floor_ms is not None:
                                    spot_chunk_start_ms = max(spot_chunk_start_ms, spot_floor_ms)
                                spot_df = _fetch_ohlcv_paged(
                                    spot_exchange,
                                    spot_symbol,
                                    spot_chunk_start_ms,
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


def _feature_delta_dir(parquet_path: str) -> str:
    return parquet_path + ".deltas"


def _feature_delta_duckdb_path(parquet_path: str) -> str:
    return parquet_path + ".deltas.duckdb"


def _feature_delta_append_enabled() -> bool:
    return os.getenv("EPM_FEATURE_DELTA_APPEND", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _feature_delta_compact_rows() -> int:
    try:
        return max(1, int(os.getenv("EPM_FEATURE_DELTA_COMPACT_ROWS", "200")))
    except Exception:
        return 200


_DUCKDB_IMPORT_CACHE: Any | None = None
_DUCKDB_IMPORT_ATTEMPTED = False
_DUCKDB_UNAVAILABLE_LOGGED = False


def _feature_delta_duckdb_enabled() -> bool:
    return os.getenv("EPM_FEATURE_DELTA_DUCKDB", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _feature_delta_trusted_cutoff_append_enabled() -> bool:
    return os.getenv("EPM_FEATURE_DELTA_TRUST_CUTOFF_APPEND", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _feature_delta_duckdb_module():
    global _DUCKDB_IMPORT_ATTEMPTED, _DUCKDB_IMPORT_CACHE, _DUCKDB_UNAVAILABLE_LOGGED
    if _DUCKDB_IMPORT_ATTEMPTED:
        return _DUCKDB_IMPORT_CACHE
    _DUCKDB_IMPORT_ATTEMPTED = True
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        _DUCKDB_IMPORT_CACHE = None
        if _feature_delta_duckdb_enabled() and not _DUCKDB_UNAVAILABLE_LOGGED:
            tprint(
                "Feature delta DuckDB buffer unavailable; falling back to "
                f"parquet delta parts ({type(exc).__name__}: {exc})"
            )
            _DUCKDB_UNAVAILABLE_LOGGED = True
        return None
    _DUCKDB_IMPORT_CACHE = duckdb
    return duckdb


def _feature_delta_compression() -> str | None:
    value = os.getenv("EPM_FEATURE_DELTA_COMPRESSION", "snappy").strip().lower()
    if value in {"", "none", "null", "uncompressed", "off"}:
        return None
    return value


def _atomic_write_parquet(
    df: pd.DataFrame,
    parquet_path: str,
    *,
    compression: str = "zstd",
):
    tmp_path = parquet_path + ".tmp"
    df.to_parquet(tmp_path, engine="pyarrow", compression=compression)
    os.replace(tmp_path, parquet_path)


def _list_feature_delta_parts(parquet_path: str) -> list[str]:
    delta_dir = _feature_delta_dir(parquet_path)
    if not os.path.isdir(delta_dir):
        return []
    return sorted(glob.glob(os.path.join(delta_dir, "part-*.parquet")))


def _clear_feature_deltas(parquet_path: str) -> None:
    delta_dir = _feature_delta_dir(parquet_path)
    if os.path.isdir(delta_dir):
        shutil.rmtree(delta_dir)
    duckdb_path = _feature_delta_duckdb_path(parquet_path)
    for path in glob.glob(duckdb_path + "*"):
        try:
            os.remove(path)
        except OSError:
            pass


def _duckdb_table_columns(con: Any) -> list[str]:
    try:
        rows = con.execute("PRAGMA table_info('feature_deltas')").fetchall()
    except Exception:
        return []
    return [str(row[1]) for row in rows]


def _duckdb_quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _duckdb_where_from_parquet_filters(filters, table_cols: set[str]) -> tuple[str, list]:
    if not filters or "ts" not in table_cols:
        return "", []

    # PyArrow accepts either [(col, op, value), ...] for one conjunction or
    # [[...], [...]] for OR-of-AND predicates.  Normalize to the latter.
    if filters and isinstance(filters[0], tuple):
        filter_groups = [filters]
    else:
        filter_groups = filters

    allowed_ops = {"=", "==", "!=", "<", "<=", ">", ">="}
    clauses: list[str] = []
    params: list = []
    ts_ident = _duckdb_quote_ident("ts")

    for group in filter_groups:
        subclauses: list[str] = []
        subparams: list = []
        for item in group or []:
            if not isinstance(item, tuple) or len(item) != 3:
                continue
            col, op, value = item
            if str(col) != "ts":
                continue
            op_s = "==" if str(op) == "=" else str(op)
            if op_s not in allowed_ops:
                continue
            try:
                value = pd.Timestamp(value)
                if value.tzinfo is None:
                    value = value.tz_localize("UTC")
                else:
                    value = value.tz_convert("UTC")
                value = value.to_pydatetime()
            except Exception:
                pass
            sql_op = "=" if op_s == "==" else op_s
            subclauses.append(f"{ts_ident} {sql_op} ?")
            subparams.append(value)
        if subclauses:
            clauses.append("(" + " AND ".join(subclauses) + ")")
            params.extend(subparams)

    if not clauses:
        return "", []
    return " WHERE " + " OR ".join(clauses), params


def _normalise_delta_storage_frame(new_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = new_data.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        first_col = str(out.columns[0])
        if first_col != "ts":
            out = out.rename(columns={out.columns[0]: "ts"})
    elif "ts" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name or "index": "ts"})
    if "ts" not in out.columns:
        out.insert(0, "ts", pd.to_datetime(new_data.index, utc=True, errors="coerce"))
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out[~out["ts"].isna()].copy()
    out["__symbol__"] = symbol
    return out


def _read_feature_delta_duckdb(
    parquet_path: str,
    columns: list[str] | None = None,
    filters=None,
) -> pd.DataFrame:
    duckdb = _feature_delta_duckdb_module()
    db_path = _feature_delta_duckdb_path(parquet_path)
    if duckdb is None or not os.path.exists(db_path):
        return pd.DataFrame()
    con = None
    try:
        con = duckdb.connect(db_path, read_only=True)
        table_cols = _duckdb_table_columns(con)
        if not table_cols:
            return pd.DataFrame()
        requested = list(columns) if columns is not None else None
        if requested is None:
            select_cols = table_cols
        else:
            select_cols = [c for c in requested if c in table_cols]
            if "ts" in table_cols and "ts" not in select_cols:
                select_cols.insert(0, "ts")
            if (
                "__symbol__" in table_cols
                and "__symbol__" in requested
                and "__symbol__" not in select_cols
            ):
                select_cols.append("__symbol__")
            if not select_cols:
                return pd.DataFrame()
        quoted_cols = ", ".join(_duckdb_quote_ident(c) for c in select_cols)
        where_sql, params = _duckdb_where_from_parquet_filters(
            filters,
            set(table_cols),
        )
        query = f"SELECT {quoted_cols} FROM feature_deltas{where_sql}"
        df = con.execute(query, params).fetchdf()
        if "ts" in df.columns:
            ts_index = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.loc[~ts_index.isna()].copy()
            ts_index = ts_index[~ts_index.isna()]
            df = df.drop(columns=["ts"])
            df.index = pd.DatetimeIndex(ts_index, name="ts")
        if requested is not None:
            known = set(table_cols)
            # ``ts`` is represented as the DatetimeIndex after the DuckDB read.
            # Do not re-add it as an all-NaN payload column when callers request
            # it for index reconstruction; that makes downstream index
            # validation discard otherwise valid delta rows.
            payload_requested = [col for col in requested if col != "ts"]
            for col in payload_requested:
                if col not in df.columns and col in known:
                    df[col] = np.nan
            ordered = [c for c in payload_requested if c in df.columns]
            df = df[ordered]
        return df
    except Exception as exc:
        tprint(f"Warning: failed to read feature DuckDB delta {db_path}: {exc}")
        return pd.DataFrame()
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _write_feature_delta_duckdb(parquet_path: str, symbol: str, new_data: pd.DataFrame) -> int:
    if not _feature_delta_duckdb_enabled():
        return 0
    duckdb = _feature_delta_duckdb_module()
    if duckdb is None:
        return 0
    db_path = _feature_delta_duckdb_path(parquet_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    out = _normalise_delta_storage_frame(new_data, symbol)
    if out.empty:
        return 0
    con = None
    try:
        con = duckdb.connect(db_path)
        con.register("incoming_delta", out)
        table_exists = bool(
            con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'feature_deltas'"
            ).fetchone()[0]
        )
        if not table_exists:
            con.execute("CREATE TABLE feature_deltas AS SELECT * FROM incoming_delta")
        else:
            table_cols = _duckdb_table_columns(con)
            for col in out.columns:
                if col in table_cols:
                    continue
                if col in {"ts", "__symbol__"}:
                    col_type = "VARCHAR" if col == "__symbol__" else "TIMESTAMPTZ"
                else:
                    col_type = "FLOAT"
                con.execute(
                    f"ALTER TABLE feature_deltas ADD COLUMN {_duckdb_quote_ident(col)} {col_type}"
                )
            con.execute("INSERT INTO feature_deltas BY NAME SELECT * FROM incoming_delta")
        return int(len(out))
    except Exception as exc:
        tprint(
            "Warning: feature DuckDB delta append failed; falling back to "
            f"parquet delta part for {symbol}: {type(exc).__name__}: {exc}"
        )
        try:
            if con is not None:
                con.close()
        finally:
            return 0
    finally:
        if con is not None:
            try:
                con.unregister("incoming_delta")
            except Exception:
                pass
            try:
                con.close()
            except Exception:
                pass


def _should_skip_existing_tail_check(parquet_path: str, new_data: pd.DataFrame) -> bool:
    """Return true only when incoming rows are strictly after stored rows."""
    if new_data.empty:
        return False
    _, last_ts = get_feature_bounds(parquet_path)
    if last_ts is None:
        return True
    incoming_first = pd.Timestamp(new_data.index.min())
    stored_last = pd.Timestamp(last_ts)
    if incoming_first.tzinfo is None and stored_last.tzinfo is not None:
        incoming_first = incoming_first.tz_localize(stored_last.tz)
    elif incoming_first.tzinfo is not None and stored_last.tzinfo is None:
        incoming_first = incoming_first.tz_convert("UTC").tz_localize(None)
    return incoming_first > stored_last


def _slice_feature_save_index_for_cutoff(
    time_index: pd.Index,
    cutoff_ts: pd.Timestamp | None,
) -> tuple[pd.Index, np.ndarray | slice]:
    """Return the rows that need saving before materializing symbol payloads."""
    if cutoff_ts is None:
        return time_index, slice(None)
    cutoff = pd.Timestamp(cutoff_ts)
    idx = time_index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(idx.tz)
        elif idx.tz is None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    mask = np.asarray(idx > cutoff, dtype=bool)
    return idx[mask], mask


def _feature_delta_duckdb_row_count(parquet_path: str) -> int:
    duckdb = _feature_delta_duckdb_module()
    db_path = _feature_delta_duckdb_path(parquet_path)
    if duckdb is None or not os.path.exists(db_path):
        return 0
    con = None
    try:
        con = duckdb.connect(db_path, read_only=True)
        return int(con.execute("SELECT COUNT(*) FROM feature_deltas").fetchone()[0])
    except Exception:
        return 0
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _feature_delta_duckdb_columns(parquet_path: str) -> set[str]:
    duckdb = _feature_delta_duckdb_module()
    db_path = _feature_delta_duckdb_path(parquet_path)
    if duckdb is None or not os.path.exists(db_path):
        return set()
    con = None
    try:
        con = duckdb.connect(db_path, read_only=True)
        return set(_duckdb_table_columns(con))
    except Exception:
        return set()
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _feature_schema_names(parquet_path: str) -> set[str]:
    names: set[str] = set()
    paths = []
    if os.path.exists(parquet_path):
        paths.append(parquet_path)
    paths.extend(_list_feature_delta_parts(parquet_path))
    for path in paths:
        try:
            names.update(pq.ParquetFile(path).schema.names)
        except Exception:
            continue
    names.update(_feature_delta_duckdb_columns(parquet_path))
    return names


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


def _write_feature_metadata_values(
    parquet_path: str,
    symbol: str,
    rows: int,
    first_ts: pd.Timestamp | None,
    last_ts: pd.Timestamp | None,
):
    meta_path = _feature_meta_path(parquet_path)
    meta = {
        "version": 2,
        "symbol": symbol,
        "rows": int(rows),
        "first_ts": pd.Timestamp(first_ts).isoformat() if first_ts is not None else None,
        "last_ts": pd.Timestamp(last_ts).isoformat() if last_ts is not None else None,
        "delta_parts": len(_list_feature_delta_parts(parquet_path)),
        "delta_rows": _feature_delta_row_count(parquet_path),
        "delta_duckdb_rows": _feature_delta_duckdb_row_count(parquet_path),
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


def _read_feature_part(
    parquet_path: str,
    columns: list[str] | None = None,
    filters=None,
) -> pd.DataFrame:
    schema_names = _feature_schema_names(parquet_path)
    part_paths = []
    if os.path.exists(parquet_path):
        part_paths.append(parquet_path)
    part_paths.extend(_list_feature_delta_parts(parquet_path))
    frames: list[pd.DataFrame] = []
    requested = list(columns) if columns is not None else None

    for part_path in part_paths:
        try:
            part_schema = set(pq.ParquetFile(part_path).schema.names)
        except Exception:
            continue
        if requested is None:
            read_cols = None
        else:
            read_cols = [c for c in requested if c in part_schema]
            if not read_cols:
                continue
        read_kwargs = {}
        if read_cols is not None:
            read_kwargs["columns"] = read_cols
        if filters is not None and "ts" in part_schema:
            read_kwargs["filters"] = filters
        try:
            frame = pd.read_parquet(part_path, **read_kwargs)
        except Exception:
            if filters is not None and "ts" in part_schema:
                read_kwargs.pop("filters", None)
                frame = pd.read_parquet(part_path, **read_kwargs)
            else:
                raise
        if requested is not None:
            missing = [c for c in requested if c not in frame.columns and c in schema_names]
            for col in missing:
                frame[col] = np.nan
            ordered = [c for c in requested if c in frame.columns]
            frame = frame[ordered]
        frames.append(frame)

    duckdb_frame = _read_feature_delta_duckdb(
        parquet_path,
        columns=columns,
        filters=filters,
    )
    if not duckdb_frame.empty:
        frames.append(duckdb_frame)

    if not frames:
        return pd.DataFrame()

    frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()

    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, axis=0, copy=False)


def _merge_duplicate_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.index.is_unique:
        return df
    # Sparse delta rows may update only a subset of columns for an existing
    # timestamp. Coalesce duplicate timestamps column-wise using the first
    # non-null value so append-only deltas can fill missing cells without
    # silently replacing an existing base feature value for historical rows.
    merged = df.groupby(level=0, sort=True).first()
    return merged.sort_index()


def read_symbol_features(
    parquet_path: str,
    columns: list[str] | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    allowed_periods=None,
) -> pd.DataFrame:
    filters = _build_parquet_ts_filters(
        start_ts=start_ts,
        end_ts=end_ts,
        allowed_periods=allowed_periods,
        inclusive_end=not bool(allowed_periods),
    )
    df = _read_feature_part(parquet_path, columns=columns, filters=filters)
    if df.empty:
        return df

    df, index_reason = _ensure_feature_frame_index(df, parquet_path=parquet_path)
    if index_reason == "invalid_ts_column":
        return pd.DataFrame(columns=df.columns)

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        _s = (
            pd.Timestamp(start_ts).tz_localize(idx.tz)
            if start_ts is not None and pd.Timestamp(start_ts).tzinfo is None
            else start_ts
        )
        _e = (
            pd.Timestamp(end_ts).tz_localize(idx.tz)
            if end_ts is not None and pd.Timestamp(end_ts).tzinfo is None
            else end_ts
        )
    else:
        _s = start_ts
        _e = end_ts
    if _s is not None:
        df = df[df.index >= _s]
    if _e is not None:
        df = df[df.index <= _e]
    if allowed_periods:
        df = _apply_allowed_periods_mask(df, allowed_periods)
    df = _merge_duplicate_feature_rows(df)
    return df.sort_index()


def _feature_delta_row_count(parquet_path: str) -> int:
    total = 0
    for part_path in _list_feature_delta_parts(parquet_path):
        try:
            total += int(pq.ParquetFile(part_path).metadata.num_rows)
        except Exception:
            continue
    total += _feature_delta_duckdb_row_count(parquet_path)
    return total


def _write_feature_delta_part(parquet_path: str, symbol: str, new_data: pd.DataFrame) -> int:
    duckdb_written = _write_feature_delta_duckdb(parquet_path, symbol, new_data)
    if duckdb_written > 0:
        old_first, old_last = get_feature_bounds(parquet_path)
        new_first = pd.Timestamp(new_data.index.min())
        new_last = pd.Timestamp(new_data.index.max())
        first_ts = min([x for x in [old_first, new_first] if x is not None])
        last_ts = max([x for x in [old_last, new_last] if x is not None])
        old_rows = int((_read_feature_metadata(parquet_path) or {}).get("rows") or 0)
        _write_feature_metadata_values(
            parquet_path,
            symbol,
            rows=old_rows + duckdb_written,
            first_ts=first_ts,
            last_ts=last_ts,
        )
        return duckdb_written

    delta_dir = _feature_delta_dir(parquet_path)
    os.makedirs(delta_dir, exist_ok=True)
    part_name = f"part-{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%S%f')}-{time.time_ns()}.parquet"
    part_path = os.path.join(delta_dir, part_name)
    out = new_data.copy()
    out["__symbol__"] = symbol
    _atomic_write_parquet(out, part_path, compression=_feature_delta_compression())
    old_first, old_last = get_feature_bounds(parquet_path)
    new_first = pd.Timestamp(out.index.min())
    new_last = pd.Timestamp(out.index.max())
    first_ts = min([x for x in [old_first, new_first] if x is not None])
    last_ts = max([x for x in [old_last, new_last] if x is not None])
    old_rows = int((_read_feature_metadata(parquet_path) or {}).get("rows") or 0)
    _write_feature_metadata_values(
        parquet_path,
        symbol,
        rows=old_rows + len(out),
        first_ts=first_ts,
        last_ts=last_ts,
    )
    return len(out)


def compact_symbol_feature_deltas(parquet_path: str, symbol: str) -> int:
    parts = _list_feature_delta_parts(parquet_path)
    if not parts:
        return 0
    combined = read_symbol_features(parquet_path)
    if combined.empty:
        return 0
    numeric_cols = [c for c in combined.columns if c != "__symbol__"]
    combined[numeric_cols] = combined[numeric_cols].astype(np.float32, copy=False)
    combined["__symbol__"] = symbol
    _atomic_write_parquet(combined, parquet_path)
    _clear_feature_deltas(parquet_path)
    _write_feature_metadata(parquet_path, symbol, combined.index)
    return len(combined)


def append_symbol_features(
    parquet_path: str,
    symbol: str,
    new_data: pd.DataFrame,
    overwrite_columns: set[str] | None = None,
    skip_existing_tail_check: bool = False,
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
        overwrite_columns = set(str(c) for c in (overwrite_columns or set()))
        if (
            _feature_delta_append_enabled()
            and not overwrite_columns
            and os.path.exists(parquet_path)
        ):
            delta_data = new_data.drop(
                columns=[
                    c
                    for c in new_data.columns
                    if c != "__symbol__" and bool(new_data[c].isna().all())
                ],
                errors="ignore",
            )
            if not delta_data.empty and any(c != "__symbol__" for c in delta_data.columns):
                if not skip_existing_tail_check:
                    # Idempotence for interrupted reruns: if the requested tail rows
                    # are already present in base or delta storage, do not append
                    # duplicate rows. Keep this range read narrow; incremental tails
                    # are normally only a few rows per symbol.
                    incoming_value_cols = [c for c in delta_data.columns if c != "__symbol__"]
                    try:
                        existing_tail = read_symbol_features(
                            parquet_path,
                            columns=incoming_value_cols,
                            start_ts=pd.Timestamp(delta_data.index.min()),
                            end_ts=pd.Timestamp(delta_data.index.max()),
                        )
                    except Exception:
                        existing_tail = pd.DataFrame()
                    if not existing_tail.empty:
                        common_idx = delta_data.index.intersection(existing_tail.index)
                        if len(common_idx) > 0:
                            existing_aligned = existing_tail.reindex(
                                index=common_idx,
                                columns=incoming_value_cols,
                            )
                            populated_cells = existing_aligned.notna()
                            if bool(populated_cells.any().any()):
                                for col in incoming_value_cols:
                                    if col not in delta_data.columns:
                                        continue
                                    populated_idx = populated_cells.index[
                                        populated_cells[col].to_numpy(dtype=bool)
                                    ]
                                    if len(populated_idx) > 0:
                                        delta_data.loc[populated_idx, col] = np.nan
                                value_cols = [
                                    c for c in delta_data.columns if c != "__symbol__"
                                ]
                                if value_cols:
                                    delta_data = delta_data.loc[
                                        ~delta_data[value_cols].isna().all(axis=1)
                                    ]
                if delta_data.empty:
                    return 0
                append_start = time.time()
                written_rows = _write_feature_delta_part(parquet_path, symbol, delta_data)
                append_elapsed = time.time() - append_start
                if append_elapsed >= 5.0:
                    tprint(
                        "Feature delta append slow: "
                        f"symbol={symbol} rows={len(delta_data)} cols={len(delta_data.columns)} "
                        f"elapsed={append_elapsed:.1f}s path={os.path.basename(parquet_path)}"
                    )
                if (
                    not _feature_delta_duckdb_enabled()
                    and _feature_delta_row_count(parquet_path) >= _feature_delta_compact_rows()
                ):
                    compact_symbol_feature_deltas(parquet_path, symbol)
                return written_rows
            return 0

        existing = None
        if os.path.exists(parquet_path):
            try:
                existing = read_symbol_features(parquet_path)
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
                incoming_cols = list(new_data.columns)

            combined = existing_aligned.reindex(
                existing_aligned.index.union(new_data.index)
            ).sort_index()
            # Incremental feature generation must be additive: preserve existing
            # non-missing cells and only fill new rows, new columns, or NaNs.
            # Rewriting already-populated cells makes reruns expensive and can
            # mask earlier run provenance.
            overwrite_cols = [
                c for c in incoming_cols if c in overwrite_columns and c in new_data.columns
            ]
            fill_cols = [c for c in incoming_cols if c not in overwrite_columns]
            if fill_cols:
                target = combined.loc[new_data.index, fill_cols]
                write_mask = target.isna()
                combined.loc[new_data.index, fill_cols] = target.where(
                    ~write_mask,
                    new_data[fill_cols],
                )
            if overwrite_cols:
                combined.loc[new_data.index, overwrite_cols] = new_data[overwrite_cols]
        else:
            before_rows = 0
            combined = new_data

        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined["__symbol__"] = symbol

        # Ensure all numeric columns are float32 (not float64) to save memory
        numeric_cols = [c for c in combined.columns if c != "__symbol__"]
        combined[numeric_cols] = combined[numeric_cols].astype(np.float32, copy=False)

        _atomic_write_parquet(combined, parquet_path)
        _clear_feature_deltas(parquet_path)
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
    overwrite_columns: set[str] | list[str] | tuple[str, ...] | None = None,
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
    overwrite_columns = set(str(c) for c in (overwrite_columns or set()))

    tprint(f"Saving features to {out_dir}...")
    trust_cutoff_append = (
        min_timestamp_by_symbol is not None
        and not overwrite_columns
        and _feature_delta_trusted_cutoff_append_enabled()
    )
    if trust_cutoff_append:
        tprint(
            "  Feature delta trusted cutoff append enabled: skipping per-symbol "
            "existing-tail reads for generated post-cutoff rows"
        )

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
        if (
            worker_count > 1
            and _feature_delta_append_enabled()
            and _feature_delta_duckdb_enabled()
            and not replace_existing
        ):
            # DuckDB is reliable as an append store here, but concurrent
            # per-symbol writers can spend minutes CPU-bound during executor
            # shutdown on macOS. Keep the append path deterministic; the data
            # preparation remains streaming and the writes are tiny tail deltas.
            tprint(
                "  DuckDB feature deltas enabled: forcing single-writer save path "
                f"(requested workers={worker_count})"
            )
            worker_count = 1
        max_pending = max(1, worker_count * 2)

        def _prepare_symbol_payload(j: int, sym: str):
            safe_sym = sym.replace("/", "_")
            final_path = os.path.join(out_dir, f"symbol={safe_sym}.parquet")

            cutoff_ts = None
            if min_timestamp_by_symbol:
                cutoff_ts = min_timestamp_by_symbol.get(sym)

            selected_index, row_selector = _slice_feature_save_index_for_cutoff(
                time_index,
                cutoff_ts,
            )
            if len(selected_index) == 0:
                return None

            sym_data = {}
            for k in feat_keys:
                arr = feats[k]
                if arr.ndim == 2:
                    sym_data[k] = arr[row_selector, j]
                else:
                    sym_data[k] = arr[row_selector]
            df_sym = pd.DataFrame(sym_data, index=selected_index, copy=False)
            df_sym = df_sym.astype(np.float32, copy=False)

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
                _clear_feature_deltas(final_path)
                _write_feature_metadata(final_path, sym, df_out.index)
            else:
                skip_existing_tail_check = trust_cutoff_append and _should_skip_existing_tail_check(
                    final_path,
                    df_sym,
                )
                append_symbol_features(
                    final_path,
                    sym,
                    df_sym,
                    overwrite_columns=overwrite_columns,
                    skip_existing_tail_check=skip_existing_tail_check,
                )
            return True

        count = 0
        if worker_count == 1:
            last_progress_log = time.time()
            for j, sym in enumerate(symbols):
                if time.time() - last_progress_log >= 30.0:
                    tprint(
                        f"  Save progress: preparing symbol {j + 1}/{total} "
                        f"({sym}, {n_feats} features each)"
                    )
                    last_progress_log = time.time()
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
                    last_progress_log = time.time()
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
    if (
        worker_count > 1
        and _feature_delta_append_enabled()
        and _feature_delta_duckdb_enabled()
        and not replace_existing
    ):
        # DuckDB is reliable as an append store here, but concurrent per-symbol
        # writers can spend minutes CPU-bound during executor shutdown on macOS.
        # Keep the append path deterministic; the data preparation remains
        # streaming and the writes are tiny tail deltas.
        tprint(
            "  DuckDB feature deltas enabled: forcing single-writer save path "
            f"(requested workers={worker_count})"
        )
        worker_count = 1
    max_pending = max(1, worker_count * 2)

    def _prepare_dataframe_symbol_payload(sym: str):
        cutoff_ts = None
        if min_timestamp_by_symbol:
            cutoff_ts = min_timestamp_by_symbol.get(sym)
        selected_index, row_selector = _slice_feature_save_index_for_cutoff(
            time_index,
            cutoff_ts,
        )
        if len(selected_index) == 0:
            return None

        # Build {feat_name: 1-D array} for this symbol
        col_data = {}
        for k in feat_keys:
            j = col_maps[k].get(sym)
            if j is not None:
                col_data[k] = arrays[k][row_selector, j]

        if not col_data:
            return None

        df_sym = pd.DataFrame(col_data, index=selected_index)
        df_sym = df_sym.astype(np.float32, copy=False)

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
            _clear_feature_deltas(final_path)
            _write_feature_metadata(final_path, sym, df_out.index)
        else:
            skip_existing_tail_check = trust_cutoff_append and _should_skip_existing_tail_check(
                final_path,
                df_sym,
            )
            append_symbol_features(
                final_path,
                sym,
                df_sym,
                overwrite_columns=overwrite_columns,
                skip_existing_tail_check=skip_existing_tail_check,
            )
        return True

    count = 0
    if worker_count == 1:
        last_progress_log = time.time()
        for j, sym in enumerate(symbols):
            if time.time() - last_progress_log >= 30.0:
                tprint(
                    f"  Save progress: preparing symbol {j + 1}/{total} "
                    f"({sym}, {n_feats} features each)"
                )
                last_progress_log = time.time()
            payload = _prepare_dataframe_symbol_payload(sym)
            wrote = _write_dataframe_symbol_payload(payload)
            if payload is not None:
                _, _, df_sym = payload
                del df_sym
            if wrote:
                count += 1

            if count % 25 == 0 or count == total:
                tprint(f"Saved {count}/{total} symbols ({n_feats} features each)")
                last_progress_log = time.time()
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
            df = read_symbol_features(fpath)

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

    def has_raw_key(self, k):
        return k in self._raw or k in self._assembled

    def raw_symbols_for_key(self, k):
        if k in self._assembled:
            frame = self._assembled.get(k)
            return set(frame.columns) if isinstance(frame, pd.DataFrame) else set()
        payload = self._raw.get(k)
        if isinstance(payload, dict):
            return set(str(sym) for sym in payload.keys())
        return set()

    def latest_values_at(self, k, symbols, ts, *, stale_sensitive=False):
        """Return feature values for symbols at or before ts without wide assembly."""
        ts_utc = pd.Timestamp(ts)
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.tz_localize("UTC")
        else:
            ts_utc = ts_utc.tz_convert("UTC")
        symbol_index = pd.Index([str(sym) for sym in symbols], name="symbol")
        out = pd.Series(np.nan, index=symbol_index, dtype=np.float32)
        if k in self._assembled:
            frame = self._assembled.get(k)
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return out
            available = [sym for sym in symbol_index if sym in frame.columns]
            if not available:
                return out
            idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
            positions = np.flatnonzero(idx <= ts_utc)
            if positions.size == 0:
                return out
            pos = int(positions[-1])
            latest_ts = idx[pos]
            if stale_sensitive and (pd.isna(latest_ts) or pd.Timestamp(latest_ts) < ts_utc):
                return out
            out.loc[available] = frame.iloc[pos].reindex(available).to_numpy(
                dtype=np.float32, copy=False
            )
            return out
        payload = self._raw.get(k)
        if not isinstance(payload, dict):
            return out
        for sym in symbol_index:
            item = payload.get(sym)
            if item is None:
                continue
            if isinstance(item, tuple) and len(item) == 2:
                idx_vals, val_array = item
            else:
                idx_vals = self._symbol_indices.get(sym)
                val_array = item
            if idx_vals is None:
                continue
            normalized_idx, normalized_vals, _ = _normalize_feature_index(
                idx_vals,
                val_array,
            )
            if normalized_idx is None or normalized_vals is None:
                continue
            idx = pd.to_datetime(normalized_idx, utc=True, errors="coerce")
            positions = np.flatnonzero(idx <= ts_utc)
            if positions.size == 0:
                continue
            pos = int(positions[-1])
            latest_ts = idx[pos]
            if stale_sensitive and (pd.isna(latest_ts) or pd.Timestamp(latest_ts) < ts_utc):
                continue
            try:
                out.at[sym] = np.float32(normalized_vals[pos])
            except Exception:
                out.at[sym] = np.nan
        return out

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
    # Preserve timezone-aware bounds for parquet/DuckDB pushdown.  Feature
    # parquet files use a UTC ``ts`` index column; stripping tz here makes
    # pyarrow reject the filter and forces a full-history fallback read.
    start_ts = pd.Timestamp(start_ts) if start_ts is not None else None
    end_ts = pd.Timestamp(end_ts) if end_ts is not None else None
    normalized_periods = _normalize_allowed_periods(allowed_periods)
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

    def _read_one_selected_feature_file(fpath: str):
        try:
            fname = os.path.basename(fpath)
            sym_guess = _normalize_spot_symbol(
                fname.replace("symbol=", "").replace(".parquet", "")
            )
            if symbol_set is not None and sym_guess not in symbol_set:
                return None

            schema_names = _feature_schema_names(fpath)
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
                return None

            df = read_symbol_features(
                fpath,
                columns=cols_to_read,
                start_ts=start_ts,
                end_ts=end_ts,
                allowed_periods=normalized_periods,
            )
            df, index_reason = _ensure_feature_frame_index(df, parquet_path=fpath)
            if index_reason == "invalid_ts_column":
                return ("skip", f"Skipping feature file {fpath}: invalid ts column")
            if df.empty:
                return None
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
                return None

            normalized_idx, _, index_reason = _normalize_feature_index(df.index.values)
            if normalized_idx is None:
                return (
                    "skip",
                    f"Skipping feature file {fpath} for symbol {real_sym}: invalid index ({index_reason})",
                )
            messages = []
            if index_reason is not None:
                messages.append(
                    f"Normalized feature index for symbol {real_sym} in {fpath}: {index_reason}"
                )
            df.index = normalized_idx
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep="last")]
            idx_vals = df.index.to_numpy(copy=False)
            feature_values: dict[str, np.ndarray] = {}
            for k in df.columns:
                if feature_set is not None and k not in feature_set:
                    continue
                feature_values[str(k)] = _coerce_feature_values_float32(df[k])

            del df
            if not feature_values:
                return None
            return ("ok", real_sym, idx_vals, feature_values, messages)
        except Exception as e:
            return ("skip", f"Error loading {fpath}: {e}")

    def _ingest_selected_feature_result(result) -> None:
        if result is None:
            return
        if not isinstance(result, tuple) or not result:
            return
        if result[0] == "skip":
            if len(result) > 1:
                tprint(str(result[1]))
            return
        if result[0] != "ok" or len(result) < 5:
            return
        _, real_sym, idx_vals, feature_values, messages = result
        for message in messages or []:
            tprint(str(message))
        symbol_indices[str(real_sym)] = idx_vals
        for k, values in feature_values.items():
            if k not in feat_buffers:
                feat_buffers[k] = {}
            feat_buffers[k][str(real_sym)] = values

    try:
        max_workers = int(os.getenv("EPM_FEATURE_SELECTED_LOAD_WORKERS", "8") or "8")
    except Exception:
        max_workers = 8
    max_workers = max(1, min(max_workers, total_files))
    parallel_enabled = (
        max_workers > 1
        and total_files >= 16
        and feature_set is not None
        and str(os.getenv("EPM_FEATURE_SELECTED_LOAD_PARALLEL", "0")).strip().lower()
        not in {"0", "false", "no", "off"}
    )
    if parallel_enabled:
        tprint(
            "Selective feature load parallel reader enabled: "
            f"workers={max_workers} files={total_files}"
        )
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(_read_one_selected_feature_file, fpath): fpath
                for fpath in files
            }
            for future in concurrent.futures.as_completed(future_to_path):
                try:
                    _ingest_selected_feature_result(future.result())
                except Exception as exc:
                    tprint(
                        f"Error loading {future_to_path.get(future, '<unknown>')}: {exc}"
                    )
                completed += 1
                if completed % progress_every == 0 or completed == total_files:
                    elapsed = time.time() - start_load
                    tprint(
                        f"Selective feature load progress: {completed}/{total_files} files "
                        f"({(completed / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                    )
    else:
        for i, fpath in enumerate(files, start=1):
            _ingest_selected_feature_result(_read_one_selected_feature_file(fpath))
            if i % progress_every == 0 or i == total_files:
                elapsed = time.time() - start_load
                tprint(
                    f"Selective feature load progress: {i}/{total_files} files "
                    f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                )

    if not feat_buffers:
        return None

    tprint(
        f"Loaded raw arrays for {len(feat_buffers)} features. Returning LazyFeatureDict proxy."
    )
    return LazyFeatureDict(feat_buffers, symbol_indices=symbol_indices)


def _live_latest_feature_matrix_token(end_ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(end_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _live_latest_feature_matrix_paths(
    ts: pd.Timestamp,
    root_dir: str,
    end_ts: pd.Timestamp,
) -> tuple[str, str]:
    ts_str = pd.Timestamp(ts).strftime("%Y%m%d_%H%M%S")
    token = _live_latest_feature_matrix_token(end_ts)
    out_dir = os.path.join(root_dir, "features", ts_str, "_live_latest_matrix")
    return (
        os.path.join(out_dir, f"matrix_{token}.parquet"),
        os.path.join(out_dir, f"matrix_{token}.meta.json"),
    )


def _coerce_live_latest_matrix_index(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix is None or matrix.empty:
        return pd.DataFrame()
    out = matrix.copy()
    for col in ("symbol", "__symbol__"):
        if col in out.columns:
            out = out.set_index(col)
            break
    out.index = pd.Index([_normalize_spot_symbol(str(s)) for s in out.index], name="symbol")
    if not out.index.is_unique:
        out = out[~out.index.duplicated(keep="last")]
    drop_cols = [c for c in out.columns if str(c).startswith("__index_level_")]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def _latest_feature_matrix_from_mapping(
    feats: dict,
    *,
    symbols: list[str] | None,
    end_ts: pd.Timestamp,
    feature_keys: set[str] | None = None,
    feat_index: pd.Index | None = None,
    feat_columns: list | None = None,
) -> pd.DataFrame:
    if not feats:
        return pd.DataFrame()
    symbol_list = [_normalize_spot_symbol(str(s)) for s in (symbols or feat_columns or [])]
    if not symbol_list:
        first_val = next(iter(feats.values()))
        if isinstance(first_val, pd.DataFrame):
            symbol_list = [_normalize_spot_symbol(str(s)) for s in first_val.columns]
    if not symbol_list:
        return pd.DataFrame()

    end_ts = pd.Timestamp(end_ts)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    rows: dict[str, dict[str, float]] = {sym: {} for sym in symbol_list}
    wanted = {str(k) for k in feature_keys} if feature_keys else None

    array_row_idx: int | None = None
    array_columns = [_normalize_spot_symbol(str(s)) for s in (feat_columns or [])]
    array_col_pos = {sym: i for i, sym in enumerate(array_columns)}
    if feat_index is not None and len(feat_index) > 0:
        idx = pd.DatetimeIndex(pd.to_datetime(feat_index, utc=True))
        eligible = np.flatnonzero(idx <= end_ts)
        if eligible.size:
            array_row_idx = int(eligible[-1])

    for key, value in feats.items():
        key_s = str(key)
        if wanted is not None and key_s not in wanted:
            continue
        try:
            if isinstance(value, pd.DataFrame):
                df = value
                if df.empty:
                    continue
                idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
                eligible = np.flatnonzero(idx <= end_ts)
                if eligible.size == 0:
                    continue
                row = df.iloc[int(eligible[-1])]
                for sym in symbol_list:
                    if sym in row.index:
                        rows[sym][key_s] = float(row.loc[sym])
                continue
            if isinstance(value, np.ndarray):
                if array_row_idx is None or not array_columns:
                    continue
                arr = value
                if arr.ndim == 1:
                    val = float(arr[array_row_idx])
                    for sym in symbol_list:
                        rows[sym][key_s] = val
                elif arr.ndim == 2:
                    for sym in symbol_list:
                        j = array_col_pos.get(sym)
                        if j is None:
                            continue
                        rows[sym][key_s] = float(arr[array_row_idx, j])
        except Exception:
            continue

    matrix = pd.DataFrame.from_dict(rows, orient="index")
    if matrix.empty:
        return matrix
    matrix.index.name = "symbol"
    return matrix.astype(np.float32, copy=False)


def write_live_latest_feature_matrix(
    feats: dict,
    ts: pd.Timestamp,
    root_dir: str,
    *,
    end_ts: pd.Timestamp,
    symbols: list[str] | None = None,
    feature_keys: set[str] | list[str] | tuple[str, ...] | None = None,
    feat_index: pd.Index | None = None,
    feat_columns: list | None = None,
    merge_existing: bool = True,
) -> None:
    """Persist one live-hour feature matrix for fast inference loading.

    The normal feature store remains symbol-partitioned for training.  This
    sidecar is intentionally denormalized as symbols x feature keys so live
    inference can load one compact parquet file for the latest decision hour.
    """
    try:
        feature_key_set = {str(k) for k in feature_keys} if feature_keys else None
        matrix = _latest_feature_matrix_from_mapping(
            feats,
            symbols=symbols,
            end_ts=end_ts,
            feature_keys=feature_key_set,
            feat_index=feat_index,
            feat_columns=feat_columns,
        )
        if matrix.empty:
            return
        data_path, meta_path = _live_latest_feature_matrix_paths(ts, root_dir, end_ts)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        if merge_existing and os.path.exists(data_path):
            try:
                existing = _coerce_live_latest_matrix_index(pd.read_parquet(data_path))
            except Exception:
                existing = pd.DataFrame()
            if not existing.empty:
                merged_index = existing.index.union(matrix.index)
                merged = existing.reindex(merged_index)
                for col in matrix.columns:
                    incoming = matrix[col].reindex(merged_index)
                    if col in merged.columns:
                        merged[col] = incoming.combine_first(merged[col])
                    else:
                        merged[col] = incoming
                matrix = merged
        matrix = matrix.sort_index(axis=0).sort_index(axis=1).astype(
            np.float32, copy=False
        )
        tmp_data = data_path + ".tmp"
        tmp_meta = meta_path + ".tmp"
        matrix.to_parquet(tmp_data, engine="pyarrow", compression="zstd")
        os.replace(tmp_data, data_path)
        meta = {
            "version": LIVE_LATEST_FEATURE_MATRIX_VERSION,
            "run_id": pd.Timestamp(ts).strftime("%Y%m%d_%H%M%S"),
            "end_ts": pd.Timestamp(end_ts).isoformat(),
            "rows": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "feature_names_hash": hashlib.sha256(
                json.dumps(
                    sorted(str(c) for c in matrix.columns),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, sort_keys=True)
        os.replace(tmp_meta, meta_path)
        tprint(
            "Persisted live latest feature matrix sidecar: "
            f"symbols={matrix.shape[0]} features={matrix.shape[1]} "
            f"end_ts={pd.Timestamp(end_ts)}"
        )
    except Exception as exc:
        tprint(f"Warning: failed to persist live latest feature matrix sidecar: {exc}")


def load_live_latest_feature_matrix(
    ts: pd.Timestamp,
    root_dir: str,
    *,
    end_ts: pd.Timestamp,
    feature_keys: list[str] | set[str] | tuple[str, ...] | None = None,
    symbols: list[str] | set[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame | None:
    data_path, meta_path = _live_latest_feature_matrix_paths(ts, root_dir, end_ts)
    if not (os.path.exists(data_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if int(meta.get("version", -1)) != LIVE_LATEST_FEATURE_MATRIX_VERSION:
            return None
        if pd.Timestamp(meta.get("end_ts")) != pd.Timestamp(end_ts):
            return None
        matrix = _coerce_live_latest_matrix_index(pd.read_parquet(data_path))
        if matrix.empty:
            return None
        if feature_keys is not None:
            wanted = [str(k) for k in feature_keys if str(k)]
            missing = sorted(set(wanted) - set(str(c) for c in matrix.columns))
            if missing:
                return None
            matrix = matrix.loc[:, wanted]
        if symbols is not None:
            symbol_index = [_normalize_spot_symbol(str(s)) for s in symbols]
            matrix = matrix.reindex(symbol_index)
        return matrix.astype(np.float32, copy=False)
    except Exception:
        return None


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
