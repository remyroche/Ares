#!/usr/bin/env python3
"""Materialise compact, causal BTC 1-minute OHLCV benchmark recaps.

The official Binance Vision daily archive is used only as a benchmark-price
source. ZIP payloads are read inside a temporary directory and never retained;
the persistent artifact is a small per-minute close/return recap with explicit
candle-close availability timestamps. This is not a trade or order-book feed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import ssl
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import certifi


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ARCHIVE = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{date}.zip"
RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verified_ssl_context() -> ssl.SSLContext:
    """Use the bundled CA store when the local Python lacks system roots."""
    return ssl.create_default_context(cafile=certifi.where())


def _read_day(day: pd.Timestamp, *, timeout_seconds: int) -> tuple[pd.DataFrame, dict[str, object]]:
    date = day.strftime("%Y-%m-%d")
    url = ARCHIVE.format(date=date)
    request = urllib.request.Request(url, headers={"User-Agent": "Ares-liquidity-transition/1"})
    with urllib.request.urlopen(  # nosec B310: fixed public archive
        request,
        timeout=timeout_seconds,
        context=_verified_ssl_context(),
    ) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in BTC archive {date}, found {names}")
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, header=None, usecols=list(range(12)))
    # Binance Vision currently emits a header, while historical archives may
    # not.  Detect it explicitly before applying the invariant 1,440-row
    # daily-candle contract.
    if not frame.empty and str(frame.iloc[0, 0]).strip().lower() in {"open_time", "opentime"}:
        frame = frame.iloc[1:].reset_index(drop=True)
    frame.columns = RAW_COLUMNS
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
    frame["close_time"] = pd.to_numeric(frame["close_time"], errors="coerce")
    frame["candle_open_ts"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True, errors="coerce")
    frame["available_ts"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True, errors="coerce") + pd.Timedelta(milliseconds=1)
    frame["btc_close"] = pd.to_numeric(frame["close"], errors="coerce")
    expected = pd.date_range(day, day + pd.Timedelta(days=1), freq="min", inclusive="left")
    if len(frame) != len(expected) or not frame["candle_open_ts"].equals(pd.Series(expected)):
        raise ValueError(f"BTC archive {date} is not a complete contiguous 1-minute day")
    if frame["btc_close"].isna().any() or frame["btc_close"].le(0.0).any():
        raise ValueError(f"BTC archive {date} has invalid close")
    return frame.loc[:, ["candle_open_ts", "available_ts", "btc_close"]], {
        "date": date, "url": url, "bytes": len(payload), "sha256": _sha256_bytes(payload), "rows": int(len(frame)),
    }


def _add_returns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("available_ts", kind="stable").reset_index(drop=True).copy()
    for lookback in (1, 5, 15):
        prior = out["btc_close"].shift(lookback)
        contiguous = out["available_ts"].shift(lookback).eq(out["available_ts"] - pd.Timedelta(minutes=lookback))
        out[f"btc_ret_{lookback}m"] = (out["btc_close"] / prior - 1.0).where(contiguous)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", required=True, help="UTC YYYY-MM-DD daily recaps")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()
    dates = sorted({pd.Timestamp(value, tz="UTC").normalize() for value in args.dates})
    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, object]] = []
    # The temporary directory proves that downloaded archives never enter the
    # project data tree; the in-memory ZIP reader remains bounded to one day.
    with tempfile.TemporaryDirectory(prefix="ares_btc_ohlcv_"):
        for day in dates:
            frame, receipt = _read_day(day, timeout_seconds=int(args.timeout_seconds))
            frames.append(frame)
            receipts.append(receipt)
    output = _add_returns(pd.concat(frames, ignore_index=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = args.out.with_name(f".{args.out.name}.partial")
    output.to_parquet(staged, index=False)
    staged.replace(args.out)
    manifest = {
        "schema": "ares.btc_benchmark_ohlcv_recap.v1",
        "source": "official Binance Vision BTCUSDT 1-minute OHLCV daily archives",
        "retention": "only compact OHLCV close/return recaps retained; downloaded ZIP archives are temporary and deleted before output is committed",
        "availability": "candle close is first available at close_time plus 1 millisecond",
        "dates": receipts,
        "rows": int(len(output)),
        "output": str(args.out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
