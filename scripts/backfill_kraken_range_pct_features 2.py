#!/usr/bin/env python3
"""Backfill Kraken perp raw range percentage feature columns.

The original Kraken perp feature run encoded missing rolling windows as 0.0 for
range_12h_pct/range_16h_pct/range_24h_pct. That makes sparse or newly listed
perps look near-constant. This script recomputes those columns from raw OHLCV
and preserves missing/pre-listing windows as NaN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RANGE_COLUMNS = {
    12: "range_12h_pct",
    16: "range_16h_pct",
    24: "range_24h_pct",
}


def _feature_symbol(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("symbol="):
        raise ValueError(f"Unexpected feature filename: {path.name}")
    return stem.removeprefix("symbol=")


def _load_raw_ohlcv(ohlcv_root: Path, symbol: str) -> pd.DataFrame:
    symbol_dir = ohlcv_root / f"symbol={symbol}"
    files = sorted(symbol_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(symbol_dir)
    frames = [pd.read_parquet(path) for path in files]
    raw = pd.concat(frames, ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["ts"])
    raw = raw.sort_values("ts")
    raw["ts_hour"] = raw["ts"].dt.floor("h")
    hourly = (
        raw.groupby("ts_hour", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .astype("float32")
    )
    return hourly


def _compute_range_columns(raw_hourly: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    aligned = raw_hourly.reindex(index)
    out = pd.DataFrame(index=index)
    close = aligned["close"].where(aligned["close"] > 0.0)
    valid_base = (
        np.isfinite(aligned["high"])
        & np.isfinite(aligned["low"])
        & np.isfinite(close)
    )
    for window, column in RANGE_COLUMNS.items():
        high = aligned["high"].rolling(window, min_periods=window).max()
        low = aligned["low"].rolling(window, min_periods=window).min()
        valid = valid_base & np.isfinite(high) & np.isfinite(low)
        values = ((high - low) / (close + 1e-12)).replace([np.inf, -np.inf], np.nan)
        out[column] = values.where(valid).astype("float32")
    return out


def _stats(values: pd.Series) -> dict[str, float | int]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "nan": int(values.isna().sum()),
        "zero_frac": float((finite == 0.0).mean()),
        "std": float(finite.std()),
        "p05": float(finite.quantile(0.05)),
        "p50": float(finite.quantile(0.50)),
        "p95": float(finite.quantile(0.95)),
        "max": float(finite.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-root",
        default="/Users/remyroche/Documents/Ares/data_perp/features/20260520_004500",
    )
    parser.add_argument(
        "--ohlcv-root",
        default="/Users/remyroche/Documents/Ares/data_perp/exchanges/krakenfutures/ohlcv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    features_root = Path(args.features_root)
    ohlcv_root = Path(args.ohlcv_root)
    feature_files = sorted(features_root.glob("symbol=*.parquet"))
    if not feature_files:
        raise FileNotFoundError(features_root)

    updated = 0
    missing = []
    for feature_path in feature_files:
        symbol = _feature_symbol(feature_path)
        try:
            raw_hourly = _load_raw_ohlcv(ohlcv_root, symbol)
        except FileNotFoundError:
            missing.append(symbol)
            continue

        df = pd.read_parquet(feature_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts" not in df.columns:
                raise ValueError(f"{feature_path} has no DatetimeIndex or ts column")
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.set_index("ts")

        range_df = _compute_range_columns(raw_hourly, df.index)
        for column in RANGE_COLUMNS.values():
            df[column] = range_df[column]

        if not args.dry_run:
            df.to_parquet(feature_path)
        updated += 1

        if updated <= 5 or updated % 25 == 0:
            stat = _stats(df["range_12h_pct"])
            print(f"{updated:03d}/{len(feature_files)} {symbol} range_12h_pct={stat}")

    print(f"updated={updated} missing_ohlcv={len(missing)} dry_run={args.dry_run}")
    if missing:
        print("missing symbols:", ",".join(missing[:50]))


if __name__ == "__main__":
    main()
