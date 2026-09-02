#!/usr/bin/env python3
"""Download Kraken spot and perpetual data for dual-listed USD quote symbols.

The script intentionally reuses the project data-store primitives:
- PartitionedOHLCVStore.update_symbol_perp for perp OHLCV/funding/OI/mark/index.
- PartitionedOHLCVStore.update_symbol for spot OHLCV.
- build_hourly_orderbook_proxy_from_ohlcv + normalize_orderbook_proxy_frame for
  the existing hourly orderbook-proxy schema.

It is incremental: existing OHLCV metadata determines the next fetch point, and
funding/orderbook proxy files are merged by timestamp.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import zipfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    _load_local_env_if_present,
    build_hourly_orderbook_proxy_from_ohlcv,
    make_perp_exchange,
    make_spot_exchange,
    normalize_orderbook_proxy_frame,
)
from extreme_price_movements.utils import tprint


PERP_USD_QUOTES = ("USD",)
SPOT_QUOTE_PRIORITY = ("USD", "USDC", "USDT")
ORDERBOOK_PROXY_COLUMNS = [
    "best_bid",
    "best_ask",
    "mid",
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
    "source",
]
FUNDING_COLUMNS = [
    "funding_rate",
    "index_price",
    "mark_price",
    "open_interest",
    "premium_index",
]
REQUIRED_REFERENCE_PRICE_TICKS = ("mark", "index", "premiumIndex")
KRAKEN_OHLCVT_DRIVE_FILE_ID = "1ptNqWYidLkhb2VAKuLCxmp2OXEfGO-AP"
KRAKEN_OHLCVT_DOWNLOAD_URL = (
    "https://drive.google.com/uc?export=download&id="
    f"{KRAKEN_OHLCVT_DRIVE_FILE_ID}"
)
KRAKEN_OHLCVT_SUPPORT_URL = (
    "https://support.kraken.com/articles/"
    "360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data"
)
KRAKEN_OHLCVT_QUARTERLY_FOLDER_URL = (
    "https://drive.google.com/drive/folders/15RSlNuW_h0kVM8or8McOGOMfHeBFvFGI"
)
KRAKEN_OHLCVT_QUARTERLY_FILE_IDS = {
    "Kraken_OHLCVT_Q1_2023.zip": "17ghRNMQGK0Is7_by784qGzP1eCUokI2V",
    "Kraken_OHLCVT_Q2_2023.zip": "1QGRW_Qg9H2pC2dBTk0b6vlGi93AFiZfI",
    "Kraken_OHLCVT_Q3_2023.zip": "1gE9XyED-bm4ks1PZomDnlpt-f_r9nWu6",
    "Kraken_OHLCVT_Q4_2023.zip": "1c3HQ0-YMvhAuGwo-f4BKAdhkG8Cj6jxx",
    "Kraken_OHLCVT_Q1_2024.zip": "1JkH3c13madqdpF-dzXoseX_sYY1E2iHx",
    "Kraken_OHLCVT_Q2_2024.zip": "1nb0vaPClwYoAGnWjYXkjrBEPQC58lmPN",
    "Kraken_OHLCVT_Q3_2024.zip": "1_GQZ7gqQ9BcIEIA_L8zPwfXTUjxIKEIk",
    "Kraken_OHLCVT_Q4_2024.zip": "1fCJPY1SwJa6py-Dln-Q7S349lBXyH0Dl",
    "Kraken_OHLCVT_Q1_2025.zip": "1dXJummu2qF5J6UC4rQh0T0XmriqngONG",
    "Kraken_OHLCVT_Q2_2025.zip": "1THrQiXsMSyhGb4DmUPCbivAKXoI8rxEG",
    "Kraken_OHLCVT_Q3_2025.zip": "1N6fg5ceXx9iQHEGHyvqUUlgo3NPsRpT7",
    "Kraken_OHLCVT_Q4_2025.zip": "1QbPHLP0TTGo-lqwKn8M-_Xo_oexXlEnB",
    "Kraken_OHLCVT_Q1_2026.zip": "15QxEf_-rRS-Yt7uERCI41HMcQQPKzSHq",
}


@dataclass(frozen=True)
class DualMarketSymbol:
    base: str
    perp_symbol: str
    spot_symbol: str
    perp_quote: str
    spot_quote: str
    settle: str


def _truthy(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "disabled", "delisted"}


def _market_tradeable(market: dict[str, Any]) -> bool:
    info = market.get("info") if isinstance(market.get("info"), dict) else {}
    if market.get("active") is False:
        return False
    for key in ("tradeable", "tradable", "active", "isTrading"):
        if key in info and not _truthy(info.get(key)):
            return False
    status = str(info.get("status") or info.get("marketStatus") or "").strip().lower()
    if status and status not in {"online", "open", "trading", "enabled"}:
        return False
    return True


def _quote_rank(quote: str | None) -> tuple[int, str]:
    q = str(quote or "").upper()
    try:
        return (SPOT_QUOTE_PRIORITY.index(q), q)
    except ValueError:
        return (len(SPOT_QUOTE_PRIORITY), q)


def _choose_by_quote(markets: list[dict[str, Any]], *, settle: bool = False) -> dict[str, Any]:
    def key(market: dict[str, Any]) -> tuple[tuple[int, str], tuple[int, str], str]:
        quote_key = _quote_rank(market.get("quote"))
        settle_key = _quote_rank(market.get("settle")) if settle else (0, "")
        return quote_key, settle_key, str(market.get("symbol") or "")

    return sorted(markets, key=key)[0]


def _dual_kraken_universe(perp_exchange: Any, spot_exchange: Any) -> list[DualMarketSymbol]:
    perp_by_base: dict[str, list[dict[str, Any]]] = {}
    for market in (getattr(perp_exchange, "markets", {}) or {}).values():
        if not isinstance(market, dict) or not _market_tradeable(market):
            continue
        if not bool(market.get("swap")):
            continue
        if not (bool(market.get("linear")) or bool(market.get("inverse")) is False):
            continue
        quote = str(market.get("quote") or "").upper()
        settle = str(market.get("settle") or "").upper()
        if quote not in PERP_USD_QUOTES and settle not in PERP_USD_QUOTES:
            continue
        base = str(market.get("base") or "").upper()
        if base:
            perp_by_base.setdefault(base, []).append(market)

    spot_by_base: dict[str, list[dict[str, Any]]] = {}
    for market in (getattr(spot_exchange, "markets", {}) or {}).values():
        if not isinstance(market, dict) or not _market_tradeable(market):
            continue
        if not (bool(market.get("spot")) or str(market.get("type") or "").lower() == "spot"):
            continue
        quote = str(market.get("quote") or "").upper()
        if quote not in SPOT_QUOTE_PRIORITY:
            continue
        base = str(market.get("base") or "").upper()
        if base:
            spot_by_base.setdefault(base, []).append(market)

    # Do not use the shared Binance universe exclusion helper here: it also
    # filters quotes and would drop Kraken's USD-denominated listings.
    allowed_bases = set(perp_by_base).intersection(spot_by_base)
    out: list[DualMarketSymbol] = []
    for base in sorted(allowed_bases):
        perp = _choose_by_quote(perp_by_base[base], settle=True)
        spot = _choose_by_quote(spot_by_base[base])
        out.append(
            DualMarketSymbol(
                base=base,
                perp_symbol=str(perp["symbol"]),
                spot_symbol=str(spot["symbol"]),
                perp_quote=str(perp.get("quote") or "").upper(),
                spot_quote=str(spot.get("quote") or "").upper(),
                settle=str(perp.get("settle") or "").upper(),
            )
        )
    return out


def _reference_tick_available(
    exchange: Any,
    symbol: str,
    price_tick: str,
    *,
    since_ms: int,
) -> tuple[bool, str]:
    try:
        rows = exchange.fetch_ohlcv(
            symbol,
            timeframe="1h",
            since=int(since_ms),
            limit=3,
            params={"price": str(price_tick)},
        )
    except Exception as exc:
        return False, str(exc)
    return bool(rows), "ok" if rows else "empty"


def _filter_reference_tick_available(
    exchange: Any,
    symbols: list[DualMarketSymbol],
    *,
    lookback_hours: int,
) -> tuple[list[DualMarketSymbol], list[dict[str, Any]]]:
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=int(lookback_hours))
    since_ms = int(since.value // 10**6)
    kept: list[DualMarketSymbol] = []
    audit: list[dict[str, Any]] = []
    for row in symbols:
        tick_status: dict[str, str] = {}
        available = True
        for tick in REQUIRED_REFERENCE_PRICE_TICKS:
            ok, reason = _reference_tick_available(
                exchange,
                row.perp_symbol,
                tick,
                since_ms=since_ms,
            )
            tick_status[tick] = reason
            available = available and ok
            time.sleep(float(getattr(exchange, "rateLimit", 1000) or 1000) / 1000.0)
        entry = {
            **asdict(row),
            "reference_tick_available": bool(available),
            "reference_tick_status": tick_status,
            "reference_tick_probe_since": since.isoformat(),
        }
        audit.append(entry)
        if available:
            kept.append(row)
        else:
            tprint(
                "Skipping reference-incomplete symbol "
                f"{row.base}: {tick_status}"
            )
    return kept, audit


def _safe_symbol_key(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _merge_write_parquet(path: Path, frame: pd.DataFrame) -> tuple[int, str]:
    if frame is None or frame.empty:
        return 0, "empty"
    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out[~out.index.isna()].sort_index()
    if path.exists():
        try:
            old = pd.read_parquet(path)
            if not old.empty:
                if not isinstance(old.index, pd.DatetimeIndex):
                    old.index = pd.to_datetime(old.index, utc=True, errors="coerce")
                elif old.index.tz is None:
                    old.index = old.index.tz_localize("UTC")
                else:
                    old.index = old.index.tz_convert("UTC")
                out = pd.concat([old, out], axis=0)
        except Exception as exc:
            tprint(f"WARN could not read existing {path}: {exc}")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(tmp, compression="zstd")
    tmp.replace(path)
    if out.empty:
        return 0, "empty"
    return len(out), f"{out.index.min().isoformat()} -> {out.index.max().isoformat()}"


def _download_google_drive_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    session = requests.Session()
    response = session.get(url, stream=True, timeout=60)
    token = None
    for key, value in session.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token:
        response.close()
        response = session.get(
            url,
            params={"confirm": token},
            stream=True,
            timeout=60,
        )
    response.raise_for_status()
    content_type = str(response.headers.get("content-type") or "").lower()
    if "text/html" in content_type:
        text = response.text
        action_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', text)
        if action_match is None:
            action_match = re.search(r'<form[^>]+action="([^"]+)"', text)
        inputs = dict(
            (html.unescape(name), html.unescape(value))
            for name, value in re.findall(
                r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', text
            )
        )
        if action_match is None or not inputs:
            raise RuntimeError("Google Drive returned an HTML page without a download form")
        response.close()
        response = session.get(
            urljoin(url, html.unescape(action_match.group(1))),
            params=inputs,
            stream=True,
            timeout=60,
        )
        response.raise_for_status()
    with tmp.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024 * 8):
            if chunk:
                handle.write(chunk)
    with tmp.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"PK\x03\x04":
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded Kraken OHLCVT payload is not a ZIP file")
    tmp.replace(path)


def _norm_archive_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _build_ohlcvt_member_index(
    archives: list[tuple[str, zipfile.ZipFile]],
) -> dict[str, list[tuple[int, str]]]:
    out: dict[str, list[tuple[int, str]]] = {}
    for archive_idx, (_label, archive) in enumerate(archives):
        for member in archive.namelist():
            if member.endswith("/") or not member.lower().endswith(".csv"):
                continue
            if "__MACOSX/" in member or Path(member).name.startswith("._"):
                continue
            stem = Path(member).stem
            parts = re.split(r"[_\-.]", stem)
            if len(parts) < 2:
                continue
            interval = parts[-1]
            if interval != "60":
                continue
            pair_key = _norm_archive_key("".join(parts[:-1]))
            if pair_key:
                out.setdefault(pair_key, []).append((archive_idx, member))
    return out


def _quarter_window(year: int, quarter: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_month = ((int(quarter) - 1) * 3) + 1
    start = pd.Timestamp(year=int(year), month=start_month, day=1, tz="UTC")
    end = start + pd.offsets.QuarterEnd(startingMonth=start_month + 2)
    end = pd.Timestamp(end).tz_convert("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return start, end


def _quarterly_archive_names_for_range(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[str]:
    out: list[str] = []
    for name in sorted(KRAKEN_OHLCVT_QUARTERLY_FILE_IDS):
        match = re.search(r"_Q([1-4])_(\d{4})\.zip$", name)
        if not match:
            continue
        quarter = int(match.group(1))
        year = int(match.group(2))
        q_start, q_end = _quarter_window(year, quarter)
        if q_end >= start_ts and q_start <= end_ts:
            out.append(name)
    return out


def _download_quarterly_ohlcvt_archives(
    *,
    archive_dir: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    no_download: bool,
) -> list[Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in _quarterly_archive_names_for_range(start_ts, end_ts):
        path = archive_dir / name
        if not path.exists():
            if no_download:
                raise FileNotFoundError(f"Kraken OHLCVT quarterly archive not found: {path}")
            file_id = KRAKEN_OHLCVT_QUARTERLY_FILE_IDS[name]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            tprint(f"Downloading Kraken quarterly OHLCVT archive {name} to {path}")
            _download_google_drive_file(url, path)
        paths.append(path)
    return paths


def _build_single_archive_member_index(archive: zipfile.ZipFile) -> dict[str, str]:
    out: dict[str, str] = {}
    for member in archive.namelist():
        if member.endswith("/") or not member.lower().endswith(".csv"):
            continue
        if "__MACOSX/" in member or Path(member).name.startswith("._"):
            continue
        stem = Path(member).stem
        parts = re.split(r"[_\-.]", stem)
        if len(parts) < 2:
            continue
        interval = parts[-1]
        if interval != "60":
            continue
        pair_key = _norm_archive_key("".join(parts[:-1]))
        if pair_key and pair_key not in out:
            out[pair_key] = member
    return out


def _archive_pair_candidates(symbol: str, spot_exchange: Any) -> list[str]:
    base = ""
    quote = ""
    if "/" in symbol:
        base, raw_quote = symbol.split("/", 1)
        quote = raw_quote.split(":", 1)[0]
    candidates: list[str] = []
    try:
        market = spot_exchange.market(symbol)
        for value in (
            market.get("id"),
            market.get("wsId"),
            market.get("altname"),
            (market.get("info") or {}).get("altname"),
            (market.get("info") or {}).get("wsname"),
        ):
            if value:
                candidates.append(str(value))
    except Exception:
        pass
    if base and quote:
        base_aliases = [base]
        quote_aliases = [quote]
        if base.upper() == "BTC":
            base_aliases.extend(["XBT", "XXBT"])
        if quote.upper() == "USD":
            quote_aliases.extend(["ZUSD"])
        for b in base_aliases:
            for q in quote_aliases:
                candidates.append(f"{b}{q}")
                candidates.append(f"{b}_{q}")
                candidates.append(f"X{b}Z{q}")
    normalized: list[str] = []
    for candidate in candidates:
        key = _norm_archive_key(candidate)
        if key and key not in normalized:
            normalized.append(key)
    return normalized


def _import_spot_ohlcvt_from_archive(
    *,
    archives: list[tuple[str, zipfile.ZipFile]],
    member_index: dict[str, list[tuple[int, str]]],
    store: PartitionedOHLCVStore,
    spot_exchange: Any,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[int, str, str]:
    members: list[tuple[int, str]] | None = None
    for key in _archive_pair_candidates(symbol, spot_exchange):
        members = member_index.get(key)
        if members:
            break
    if not members:
        raise FileNotFoundError(f"No Kraken OHLCVT 60-minute CSV found for {symbol}")

    frames: list[pd.DataFrame] = []
    member_labels: list[str] = []
    for archive_idx, member in members:
        label, archive = archives[archive_idx]
        with archive.open(member) as handle:
            part = pd.read_csv(
                handle,
                header=None,
                names=["ts", "open", "high", "low", "close", "volume", "trades"],
                usecols=[0, 1, 2, 3, 4, 5, 6],
            )
        if not part.empty:
            frames.append(part)
            member_labels.append(f"{label}:{member}")
    if not frames:
        return 0, "empty", ",".join(member_labels)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return 0, "empty", ",".join(member_labels)

    ts_unit = "ms" if pd.to_numeric(df["ts"], errors="coerce").max() > 10**11 else "s"
    df["ts"] = pd.to_datetime(df["ts"], unit=ts_unit, utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
    if df.empty:
        return 0, "empty", ",".join(member_labels)
    for col in ("open", "high", "low", "close", "volume", "trades"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    hourly = df.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "trades": "sum",
        }
    )
    full_index = pd.date_range(
        start=start_ts.floor("h"),
        end=end_ts.floor("h"),
        freq="1h",
        tz="UTC",
    )
    hourly = hourly.reindex(full_index)
    hourly["close"] = hourly["close"].ffill()
    for col in ("open", "high", "low"):
        hourly[col] = hourly[col].fillna(hourly["close"])
    hourly["volume"] = hourly["volume"].fillna(0.0)
    hourly["trades"] = hourly["trades"].fillna(0.0)
    hourly = hourly.dropna(subset=["close"])
    hourly.index.name = "ts"
    if hourly.empty:
        return 0, "empty", ",".join(member_labels)

    store.save_partitioned(symbol, hourly, defer_compact=True)
    for year in sorted(set(int(y) for y in hourly.index.year)):
        store.compact_partition(symbol, year)
    store._write_meta(
        symbol,
        {
            "last_ts_ms": int(hourly.index.max().value // 10**6),
            "source": "kraken_ohlcvt_csv",
            "archive_members": member_labels,
            "source_url": KRAKEN_OHLCVT_SUPPORT_URL,
        },
    )
    return (
        len(hourly),
        f"{hourly.index.min().isoformat()} -> {hourly.index.max().isoformat()}",
        ",".join(member_labels),
    )


def _write_orderbook_proxy(
    *,
    store: PartitionedOHLCVStore,
    root: Path,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[int, str]:
    ohlcv = store.load(symbol, start_ts=start_ts, end_ts=end_ts)
    proxy = build_hourly_orderbook_proxy_from_ohlcv(ohlcv)
    if proxy is None or proxy.empty:
        return 0, "empty"
    proxy = normalize_orderbook_proxy_frame(proxy)
    for col in ORDERBOOK_PROXY_COLUMNS:
        if col not in proxy.columns:
            proxy[col] = 0.0 if col not in {"snapshot_ts", "source"} else (
                proxy.index if col == "snapshot_ts" else "local_ohlcv_summary"
            )
    proxy = proxy[ORDERBOOK_PROXY_COLUMNS]
    return _merge_write_parquet(root / "orderbook_hourly" / f"{_safe_symbol_key(symbol)}.parquet", proxy)


def _write_funding_hourly(
    *,
    store: PartitionedOHLCVStore,
    root: Path,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    output_symbol: str | None = None,
) -> tuple[int, str]:
    df = store.load(symbol, columns=["ts", *FUNDING_COLUMNS], start_ts=start_ts, end_ts=end_ts)
    if df is None or df.empty:
        return 0, "empty"
    cols = [col for col in FUNDING_COLUMNS if col in df.columns]
    if not cols:
        return 0, "empty"
    out = df[cols].copy()
    for col in FUNDING_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[FUNDING_COLUMNS]
    key_symbol = output_symbol or symbol
    return _merge_write_parquet(root / "funding_hourly" / f"{_safe_symbol_key(key_symbol)}.parquet", out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=float, default=365 * 4)
    parser.add_argument("--perp-root", default="data_perp/exchanges/krakenfutures")
    parser.add_argument("--spot-root", default="data_spot/exchanges/kraken")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--symbols", default="", help="Comma-separated base allowlist.")
    parser.add_argument(
        "--partition-count",
        type=int,
        default=1,
        help="Split the resolved universe into this many deterministic partitions.",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        default=0,
        help="Zero-based partition id to process when --partition-count > 1.",
    )
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="Process the selected symbol list in reverse order.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-perps", action="store_true")
    parser.add_argument("--skip-spot", action="store_true")
    parser.add_argument(
        "--perp-ohlcv-only",
        action="store_true",
        help="Refresh native perp OHLCV only; skip funding/OI/orderbook sidecar writes.",
    )
    parser.add_argument(
        "--spot-source",
        choices=("ohlcvt_csv", "rest"),
        default="ohlcvt_csv",
        help=(
            "Spot OHLC source. ohlcvt_csv imports Kraken's downloadable "
            "60-minute OHLCVT archive; rest uses Kraken /public/OHLC."
        ),
    )
    parser.add_argument(
        "--spot-ohlcvt-zip",
        default="data_spot/exchanges/kraken/raw/Kraken_OHLCVT.zip",
        help="Local path for Kraken's complete OHLCVT ZIP archive.",
    )
    parser.add_argument(
        "--spot-ohlcvt-mode",
        choices=("complete", "quarterly"),
        default="complete",
        help=(
            "Use Kraken's complete all-history OHLCVT ZIP or the smaller "
            "quarterly update ZIPs. Quarterly archives currently start in 2023."
        ),
    )
    parser.add_argument(
        "--spot-ohlcvt-quarterly-dir",
        default="data_spot/exchanges/kraken/raw/ohlcvt_quarterly",
        help="Directory for Kraken quarterly OHLCVT ZIP archives.",
    )
    parser.add_argument(
        "--spot-ohlcvt-url",
        default=KRAKEN_OHLCVT_DOWNLOAD_URL,
        help="Download URL for Kraken's complete OHLCVT ZIP archive.",
    )
    parser.add_argument(
        "--no-spot-ohlcvt-download",
        action="store_true",
        help="Require --spot-ohlcvt-zip to already exist instead of downloading it.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument(
        "--require-reference-ticks",
        action="store_true",
        help=(
            "Only download symbols where Kraken Futures exposes mark, index, "
            "and premiumIndex OHLCV ticks."
        ),
    )
    parser.add_argument(
        "--reference-probe-lookback-hours",
        type=int,
        default=48,
        help="Lookback window used when probing reference tick availability.",
    )
    parser.add_argument(
        "--rate-limit-ms",
        type=int,
        default=1500,
        help="Minimum ccxt per-request delay applied to both Kraken clients.",
    )
    args = parser.parse_args()

    _load_local_env_if_present()
    os.environ["EPM_EXCHANGE"] = "kraken"
    os.environ.setdefault("EPM_FUNDING_HISTORY_DAYS", str(max(args.lookback_days + 2, 0)))
    os.environ.setdefault("EPM_OPEN_INTEREST_HISTORY_DAYS", str(max(args.lookback_days + 2, 0)))

    perp_exchange = make_perp_exchange()
    spot_exchange = make_spot_exchange()
    min_rate_limit = max(int(args.rate_limit_ms), 0)
    perp_exchange.rateLimit = max(int(getattr(perp_exchange, "rateLimit", 0) or 0), min_rate_limit)
    spot_exchange.rateLimit = max(int(getattr(spot_exchange, "rateLimit", 0) or 0), min_rate_limit)
    universe = _dual_kraken_universe(perp_exchange, spot_exchange)
    if args.symbols.strip():
        allowed = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        universe = [item for item in universe if item.base in allowed]
    availability_audit: list[dict[str, Any]] = []
    if args.require_reference_ticks:
        before_count = len(universe)
        universe, availability_audit = _filter_reference_tick_available(
            perp_exchange,
            universe,
            lookback_hours=int(args.reference_probe_lookback_hours),
        )
        tprint(
            "Reference tick availability filter: "
            f"{len(universe)}/{before_count} symbols kept"
        )
    if args.max_symbols and args.max_symbols > 0:
        universe = universe[: args.max_symbols]
    partition_count = max(1, int(args.partition_count))
    partition_id = int(args.partition_id)
    if partition_id < 0 or partition_id >= partition_count:
        raise ValueError(
            f"--partition-id must be in [0, {partition_count - 1}], got {partition_id}"
        )
    full_universe_count = len(universe)
    if partition_count > 1:
        universe = [
            item
            for idx, item in enumerate(universe)
            if idx % partition_count == partition_id
        ]
        tprint(
            "Universe partition selected: "
            f"partition_id={partition_id}/{partition_count} "
            f"symbols={len(universe)}/{full_universe_count}"
        )
    if args.reverse_order:
        universe = list(reversed(universe))
        tprint("Universe order reversed for this worker.")

    perp_root = Path(args.perp_root)
    spot_root = Path(args.spot_root)
    start_ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=float(args.lookback_days))).floor("1h")
    end_ts = pd.Timestamp.now(tz="UTC").floor("1h")

    manifest = {
        "script": Path(__file__).name,
        "lookback_days": float(args.lookback_days),
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "perp_root": str(perp_root),
        "spot_root": str(spot_root),
        "require_reference_ticks": bool(args.require_reference_ticks),
        "partition_count": partition_count,
        "partition_id": partition_id,
        "reverse_order": bool(args.reverse_order),
        "full_universe_count": full_universe_count,
        "symbols": [asdict(item) for item in universe],
    }
    if availability_audit:
        manifest["reference_tick_availability"] = availability_audit
    tprint(
        "Kraken dual-market USD-swap/spot universe: "
        f"{len(universe)} symbols, start={start_ts}, end={end_ts}"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "symbols"}, indent=2))
    if args.dry_run:
        print(json.dumps({"symbols": manifest["symbols"][:50], "symbol_count": len(universe)}, indent=2))
        return 0

    perp_store = PartitionedOHLCVStore(root_dir=str(perp_root), timeframe="1h")
    spot_store = PartitionedOHLCVStore(root_dir=str(spot_root), timeframe="1h")
    spot_archives: list[tuple[str, zipfile.ZipFile]] = []
    spot_archive_member_index: dict[str, list[tuple[int, str]]] = {}
    if not args.skip_spot and args.spot_source == "ohlcvt_csv":
        if args.spot_ohlcvt_mode == "quarterly":
            archive_paths = _download_quarterly_ohlcvt_archives(
                archive_dir=Path(args.spot_ohlcvt_quarterly_dir),
                start_ts=start_ts,
                end_ts=end_ts,
                no_download=bool(args.no_spot_ohlcvt_download),
            )
        else:
            archive_path = Path(args.spot_ohlcvt_zip)
            if not archive_path.exists():
                if args.no_spot_ohlcvt_download:
                    raise FileNotFoundError(
                        f"Kraken OHLCVT archive not found: {archive_path}"
                    )
                tprint(f"Downloading Kraken OHLCVT archive to {archive_path}")
                _download_google_drive_file(str(args.spot_ohlcvt_url), archive_path)
            try:
                zipfile.ZipFile(archive_path).close()
            except zipfile.BadZipFile:
                if args.no_spot_ohlcvt_download:
                    raise
                tprint(
                    f"Existing Kraken OHLCVT archive is invalid; redownloading {archive_path}"
                )
                archive_path.unlink(missing_ok=True)
                _download_google_drive_file(str(args.spot_ohlcvt_url), archive_path)
            archive_paths = [archive_path]
        spot_archives = [(path.name, zipfile.ZipFile(path)) for path in archive_paths]
        spot_archive_member_index = _build_ohlcvt_member_index(spot_archives)
        tprint(
            "Kraken OHLCVT archive ready: "
            f"{len(spot_archive_member_index)} hourly pair files indexed"
        )
    (perp_root / "manifests").mkdir(parents=True, exist_ok=True)
    (spot_root / "manifests").mkdir(parents=True, exist_ok=True)
    manifest_name = (
        "kraken_dual_market_universe_latest.json"
        if partition_count == 1
        else f"kraken_dual_market_universe_p{partition_id}_of_{partition_count}.json"
    )
    (perp_root / "manifests" / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (spot_root / "manifests" / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stats = {
        "perp_ohlcv_ok": 0,
        "spot_ohlcv_ok": 0,
        "perp_orderbook_ok": 0,
        "spot_orderbook_ok": 0,
        "perp_funding_ok": 0,
        "failures": [],
    }
    for i, item in enumerate(universe, start=1):
        tprint(
            f"[{i:04d}/{len(universe):04d}] {item.base}: "
            f"perp={item.perp_symbol} spot={item.spot_symbol}"
        )
        if not args.skip_perps:
            try:
                old_side_data_env = os.environ.get("EPM_PERP_SIDE_DATA_ENABLED")
                if args.perp_ohlcv_only:
                    os.environ["EPM_PERP_SIDE_DATA_ENABLED"] = "0"
                perp_store.update_symbol_perp(
                    perp_exchange,
                    item.perp_symbol,
                    int(start_ts.value // 10**6),
                    spot_exchange=spot_exchange,
                )
                if args.perp_ohlcv_only:
                    if old_side_data_env is None:
                        os.environ.pop("EPM_PERP_SIDE_DATA_ENABLED", None)
                    else:
                        os.environ["EPM_PERP_SIDE_DATA_ENABLED"] = old_side_data_env
                stats["perp_ohlcv_ok"] += 1
                if args.perp_ohlcv_only:
                    tprint("  perps ok native_ohlcv_only")
                else:
                    rows, span = _write_funding_hourly(
                        store=perp_store,
                        root=perp_root,
                        symbol=item.perp_symbol,
                        start_ts=start_ts,
                        end_ts=end_ts,
                    )
                    spot_funding_rows, spot_funding_span = _write_funding_hourly(
                        store=perp_store,
                        root=spot_root,
                        symbol=item.perp_symbol,
                        output_symbol=item.spot_symbol,
                        start_ts=start_ts,
                        end_ts=end_ts,
                    )
                    if rows:
                        stats["perp_funding_ok"] += 1
                    ob_rows, ob_span = _write_orderbook_proxy(
                        store=perp_store,
                        root=perp_root,
                        symbol=item.perp_symbol,
                        start_ts=start_ts,
                        end_ts=end_ts,
                    )
                    if ob_rows:
                        stats["perp_orderbook_ok"] += 1
                    tprint(
                        f"  perps ok funding_rows={rows} ({span}) "
                        f"spot_funding_rows={spot_funding_rows} ({spot_funding_span}) "
                        f"orderbook_rows={ob_rows} ({ob_span})"
                    )
            except Exception as exc:
                if args.perp_ohlcv_only:
                    if old_side_data_env is None:
                        os.environ.pop("EPM_PERP_SIDE_DATA_ENABLED", None)
                    else:
                        os.environ["EPM_PERP_SIDE_DATA_ENABLED"] = old_side_data_env
                msg = f"perp {item.perp_symbol}: {exc.__class__.__name__}: {exc}"
                stats["failures"].append(msg)
                tprint(f"  FAIL {msg}")
        if not args.skip_spot:
            try:
                if args.spot_source == "ohlcvt_csv":
                    if not spot_archives:
                        raise RuntimeError("Kraken OHLCVT archive was not opened")
                    rows, span, member = _import_spot_ohlcvt_from_archive(
                        archives=spot_archives,
                        member_index=spot_archive_member_index,
                        store=spot_store,
                        spot_exchange=spot_exchange,
                        symbol=item.spot_symbol,
                        start_ts=start_ts,
                        end_ts=end_ts,
                    )
                    tprint(
                        f"  spot csv ok rows={rows} ({span}) member={member}"
                    )
                else:
                    spot_store.update_symbol(
                        spot_exchange,
                        item.spot_symbol,
                        int(start_ts.value // 10**6),
                    )
                stats["spot_ohlcv_ok"] += 1
                ob_rows, ob_span = _write_orderbook_proxy(
                    store=spot_store,
                    root=spot_root,
                    symbol=item.spot_symbol,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
                if ob_rows:
                    stats["spot_orderbook_ok"] += 1
                tprint(f"  spot ok orderbook_rows={ob_rows} ({ob_span})")
            except Exception as exc:
                msg = f"spot {item.spot_symbol}: {exc.__class__.__name__}: {exc}"
                stats["failures"].append(msg)
                tprint(f"  FAIL {msg}")
        time.sleep(max(float(args.sleep_seconds), 0.0))

    manifest["completed_ts"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["stats"] = stats
    (perp_root / "manifests" / "kraken_dual_market_download_latest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (spot_root / "manifests" / "kraken_dual_market_download_latest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    for _label, archive in spot_archives:
        archive.close()
    return 0 if not stats["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
