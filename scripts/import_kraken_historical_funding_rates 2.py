#!/usr/bin/env python3
"""Import Kraken's official historical funding-rate export into sidecar parquet files."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from extreme_price_movements.utils import tprint


DEFAULT_URL = (
    "https://assets-cms.kraken.com/files/51n36hrp/facade/"
    "4b70936c1227e4ae5514cba8bf41a5561cf13bd7.zip?dl="
)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100_000_000:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(path)


def _product_to_symbol(product_id: str) -> str:
    text = str(product_id or "").strip().upper()
    match = re.match(r"^(PF|PI)_?([A-Z0-9]+)USD$", text)
    if not match:
        return ""
    base = match.group(2)
    if base == "XBT":
        base = "BTC"
    return f"{base}/USD:USD"


def _symbol_to_filename(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _read_funding_csv(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as handle:
        df = pd.read_csv(handle)
    if df.empty or "timestamp" not in df.columns or "relative_rate" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=pd.to_datetime(df["timestamp"], utc=True, errors="coerce"))
    out = out[~out.index.isna()]
    out["funding_rate"] = pd.to_numeric(df["relative_rate"], errors="coerce").to_numpy()
    out = out.dropna(subset=["funding_rate"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.astype({"funding_rate": "float32"})


def _merge_sidecar(path: Path, funding: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        try:
            old = pd.read_parquet(path)
            old.index = pd.to_datetime(old.index, utc=True, errors="coerce")
            old = old[~old.index.isna()].sort_index()
        except Exception:
            old = pd.DataFrame()
    else:
        old = pd.DataFrame()
    if old.empty:
        merged = funding.copy()
    else:
        all_index = old.index.union(funding.index).sort_values()
        merged = old.reindex(all_index)
        merged["funding_rate"] = funding["funding_rate"].reindex(all_index).combine_first(
            merged.get("funding_rate")
        )
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="data_perp/exchanges/krakenfutures/raw/funding_rates/kraken_historical_funding_rates.zip")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--funding-dir", default="data_perp/exchanges/krakenfutures/funding_hourly")
    parser.add_argument("--manifest", default="data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_universe_latest.json")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not args.no_download:
        _download(args.url, zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    allowed: set[str] = set()
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        allowed = {
            str(item.get("perp_symbol", "")).strip()
            for item in manifest.get("symbols", [])
            if item.get("perp_symbol")
        }

    funding_dir = Path(args.funding_dir)
    funding_dir.mkdir(parents=True, exist_ok=True)
    imported = 0
    skipped = 0
    rows_total = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.startswith("exports/") or not member.endswith(".csv"):
                continue
            product_id = Path(member).stem
            symbol = _product_to_symbol(product_id)
            if not symbol or (allowed and symbol not in allowed):
                skipped += 1
                continue
            funding = _read_funding_csv(archive, member)
            if funding.empty:
                skipped += 1
                continue
            out_path = funding_dir / f"{_symbol_to_filename(symbol)}.parquet"
            merged = _merge_sidecar(out_path, funding)
            merged.to_parquet(out_path, compression="zstd")
            imported += 1
            rows_total += len(funding)
            if imported % 25 == 0:
                tprint(f"Imported Kraken funding CSVs: {imported}")

    tprint(
        f"Kraken funding import complete: imported={imported} skipped={skipped} "
        f"rows={rows_total:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
