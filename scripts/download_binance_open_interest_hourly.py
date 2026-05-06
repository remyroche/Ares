#!/usr/bin/env python3
"""Download hourly Binance USD(S)-M futures open-interest history.

The Binance open-interest history endpoint exposes only the latest 30 days and
has a max limit of 500 rows, so this script paginates by timestamp to recover
the full available hourly range.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from extreme_price_movements.universe import (
    HARDCODED_EXCLUDED_SYMBOLS,
    fetch_binance_cross_margin_pairs,
    margin_pairs_to_spot_symbols,
)
from extreme_price_movements.utils import tprint

SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
PERIOD = "1h"
MAX_LIMIT = 500
MAX_LOOKBACK_HOURS = (30 * 24) - 2
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_SLEEP_SECONDS = 0.08
MAX_RETRIES = 5
HOUR_MS = 60 * 60 * 1000


@dataclass
class SymbolResult:
    symbol: str
    base_asset: str
    quote_asset: str
    rows: int
    first_ts: str | None
    last_ts: str | None
    output_path: str | None
    status: str
    error: str | None = None


def _request_json(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    url = f"{base_url}{path}"
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code in {418, 429}:
                retry_after = response.headers.get("Retry-After")
                sleep_s = (
                    float(retry_after) if retry_after else min(60.0, attempt * 5.0)
                )
                tprint(
                    f"Rate limited on {path}; sleeping {sleep_s:.1f}s "
                    f"(attempt={attempt}/{MAX_RETRIES})."
                )
                time.sleep(sleep_s)
                continue
            if response.status_code >= 400:
                raise requests.HTTPError(
                    f"{response.status_code} for {url}: {response.text[:300]}",
                    response=response,
                )
            return response.json()
        except Exception as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            sleep_s = min(30.0, 2.0**attempt)
            tprint(
                f"Request failed for {path}; retrying in {sleep_s:.1f}s "
                f"(attempt={attempt}/{MAX_RETRIES}): {exc}"
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _fetch_perp_symbols() -> list[dict[str, str]]:
    raw = _request_json(FUTURES_BASE_URL, "/fapi/v1/exchangeInfo")
    symbols: list[dict[str, str]] = []
    for row in raw.get("symbols", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("contractType", "")).upper() != "PERPETUAL":
            continue
        if str(row.get("status", "")).upper() != "TRADING":
            continue
        quote = str(row.get("quoteAsset", "")).upper()
        if quote not in {"USDT", "USDC"}:
            continue
        base = str(row.get("baseAsset", "")).upper()
        symbol = str(row.get("symbol", "")).upper()
        if not base or not symbol:
            continue
        symbols.append({"symbol": symbol, "base": base, "quote": quote})
    return sorted(
        symbols, key=lambda item: (item["base"], item["quote"], item["symbol"])
    )


def _fetch_margin_bases() -> set[str]:
    pairs = fetch_binance_cross_margin_pairs()
    margin_symbols = margin_pairs_to_spot_symbols(pairs, quotes=("USDT", "USDC"))
    cleaned = {
        str(sym).upper().strip()
        for sym in margin_symbols
        if "/" in str(sym)
        and str(sym).upper().strip() not in HARDCODED_EXCLUDED_SYMBOLS
    }
    return {sym.split("/", 1)[0] for sym in cleaned if "/" in sym}


def _eligible_symbols() -> list[dict[str, str]]:
    margin_bases = _fetch_margin_bases()
    perps = _fetch_perp_symbols()
    eligible = [row for row in perps if row["base"] in margin_bases]
    tprint(
        "Eligible Binance margin+perp symbols: "
        f"{len(eligible)} perps across {len(margin_bases)} margin bases."
    )
    return eligible


def _binance_futures_server_time_ms() -> int:
    raw = _request_json(FUTURES_BASE_URL, "/fapi/v1/time")
    server_time = int(raw.get("serverTime", 0))
    if server_time <= 0:
        raise RuntimeError(f"Unexpected Binance server time response: {raw}")
    return server_time


def _open_interest_page(
    symbol: str, start_ms: int, end_ms: int
) -> list[dict[str, Any]]:
    raw = _request_json(
        FUTURES_BASE_URL,
        "/futures/data/openInterestHist",
        params={
            "symbol": symbol,
            "period": PERIOD,
            "limit": MAX_LIMIT,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    )
    if not isinstance(raw, list):
        raise RuntimeError(f"Unexpected openInterestHist response for {symbol}: {raw}")
    return [row for row in raw if isinstance(row, dict)]


def _download_symbol(
    row: dict[str, str],
    output_dir: Path,
    start_ms: int,
    end_ms: int,
) -> SymbolResult:
    symbol = row["symbol"]
    records: list[dict[str, Any]] = []
    try:
        cursor_end = end_ms
        while cursor_end >= start_ms:
            page = _open_interest_page(symbol, start_ms, cursor_end)
            if not page:
                break
            page = sorted(page, key=lambda item: int(item.get("timestamp", 0)))
            records.extend(page)
            first_ts = int(page[0].get("timestamp", 0))
            next_cursor_end = first_ts - 1
            if next_cursor_end >= cursor_end:
                break
            cursor_end = next_cursor_end
            if len(page) < MAX_LIMIT:
                break
            time.sleep(REQUEST_SLEEP_SECONDS)

        if not records:
            return SymbolResult(
                symbol=symbol,
                base_asset=row["base"],
                quote_asset=row["quote"],
                rows=0,
                first_ts=None,
                last_ts=None,
                output_path=None,
                status="empty",
            )

        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["symbol"] = symbol
        df["base_asset"] = row["base"]
        df["quote_asset"] = row["quote"]
        for col in ("sumOpenInterest", "sumOpenInterestValue"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.rename(
            columns={
                "sumOpenInterest": "sum_open_interest",
                "sumOpenInterestValue": "sum_open_interest_value",
            }
        )
        df = df[
            [
                "timestamp",
                "symbol",
                "base_asset",
                "quote_asset",
                "sum_open_interest",
                "sum_open_interest_value",
            ]
        ]

        output_path = output_dir / f"{symbol}.csv"
        df.to_csv(output_path, index=False)
        return SymbolResult(
            symbol=symbol,
            base_asset=row["base"],
            quote_asset=row["quote"],
            rows=int(len(df)),
            first_ts=df["timestamp"].iloc[0].isoformat(),
            last_ts=df["timestamp"].iloc[-1].isoformat(),
            output_path=str(output_path),
            status="ok",
        )
    except Exception as exc:
        return SymbolResult(
            symbol=symbol,
            base_asset=row["base"],
            quote_asset=row["quote"],
            rows=0,
            first_ts=None,
            last_ts=None,
            output_path=None,
            status="failed",
            error=str(exc),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download hourly Binance open-interest history for margin+perp symbols."
    )
    parser.add_argument(
        "--output-dir",
        default="data/open_interest_hourly",
        help="Output directory for symbol CSVs and manifest.",
    )
    parser.add_argument(
        "--limit-symbols",
        type=int,
        default=0,
        help="Optional cap for smoke runs; 0 means all eligible symbols.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    server_time_ms = _binance_futures_server_time_ms()
    end_ts = pd.to_datetime(server_time_ms, unit="ms", utc=True).floor("h")
    start_ts = end_ts - pd.Timedelta(hours=MAX_LOOKBACK_HOURS)
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    symbols = _eligible_symbols()
    if args.limit_symbols and args.limit_symbols > 0:
        symbols = symbols[: args.limit_symbols]
        tprint(f"Limiting run to first {len(symbols)} symbols.")

    tprint(
        "Downloading Binance open-interest history "
        f"period={PERIOD}, start={start_ts.isoformat()}, end={end_ts.isoformat()}, "
        f"symbols={len(symbols)}."
    )

    results: list[SymbolResult] = []
    for idx, symbol_row in enumerate(symbols, start=1):
        result = _download_symbol(symbol_row, output_dir, start_ms, end_ms)
        results.append(result)
        tprint(
            f"Open interest {idx}/{len(symbols)} {result.symbol}: "
            f"status={result.status}, rows={result.rows}."
        )
        time.sleep(REQUEST_SLEEP_SECONDS)

    manifest = {
        "period": PERIOD,
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "max_lookback_hours": MAX_LOOKBACK_HOURS,
        "symbols_requested": len(symbols),
        "symbols_ok": sum(1 for item in results if item.status == "ok"),
        "symbols_empty": sum(1 for item in results if item.status == "empty"),
        "symbols_failed": sum(1 for item in results if item.status == "failed"),
        "total_rows": sum(item.rows for item in results),
        "results": [asdict(item) for item in results],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    tprint(
        "Open-interest download complete: "
        f"ok={manifest['symbols_ok']}, empty={manifest['symbols_empty']}, "
        f"failed={manifest['symbols_failed']}, rows={manifest['total_rows']}, "
        f"manifest={manifest_path}."
    )
    return 0 if manifest["symbols_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
