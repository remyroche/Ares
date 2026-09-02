#!/usr/bin/env python3
"""Create the immutable exact-minute request for the remaining Aug-2022 gap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/aug2022_inverse_pi_gap_download_request_20260730_v1"
START = pd.Timestamp("2022-08-01T12:00:00Z")
END = pd.Timestamp("2022-08-30T12:00:00Z")
PRODUCTS = {"BTC/USD:BTC": "PI_XBTUSD", "ETH/USD:ETH": "PI_ETHUSD", "LTC/USD:LTC": "PI_LTCUSD", "XRP/USD:XRP": "PI_XRPUSD", "BCH/USD:BCH": "PI_BCHUSD"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output_dir: Path) -> dict:
    if output_dir.exists(): raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    request = pd.DataFrame({"timestamp": [START] * len(PRODUCTS), "symbol": list(PRODUCTS), "product_id": list(PRODUCTS.values())})
    path = output_dir / "download_request.parquet"; request.to_parquet(path, index=False, compression="zstd")
    manifest = {"schema": "aug2022_inverse_pi_exact1m_gap_download_request_v1", "status": "REQUEST_MATERIALIZED", "promotion_eligible": False, "population_separation": "inverse PI separate research lineage; no later PF taxonomy equivalence", "products": PRODUCTS, "required_window": {"start": START.isoformat(), "end_exclusive": END.isoformat(), "minutes": int((END - START) / pd.Timedelta(minutes=1))}, "downloader_contract": "canonical_kraken_execution_1m_immutable_append_missing_v1", "output": {"path": str(path), "rows": len(request), "sha256": sha256(path)}}
    manifest_path = output_dir / "manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"); (output_dir / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(); print(json.dumps(run(args.output_dir), sort_keys=True))
