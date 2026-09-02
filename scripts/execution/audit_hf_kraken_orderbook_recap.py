#!/usr/bin/env python3
"""Audit the compact-only retention and source contract of HF Kraken recaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RAW_COLUMNS = {"bids_json", "asks_json", "trade", "trades_json"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Defaults to <root>/contract_audit.json")
    args = parser.parse_args()

    surfaces = sorted(args.root.rglob("surface.parquet"))
    if not surfaces:
        raise FileNotFoundError(f"no compact surface partitions below {args.root}")
    audits = sorted(args.root.rglob("source_recap_audit.json"))
    partitions: list[dict[str, object]] = []
    retained_raw_columns: set[str] = set()
    all_symbols: set[str] = set()
    rows = 0
    valid_rows = 0
    label_coverage: dict[str, list[float]] = {}
    for path in surfaces:
        frame = pd.read_parquet(path)
        retained_raw_columns.update(RAW_COLUMNS.intersection(frame.columns))
        rows += len(frame)
        valid_rows += int(frame.get("book_valid", pd.Series(False, index=frame.index)).fillna(False).sum())
        all_symbols.update(frame.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
        for column in (name for name in frame.columns if name.startswith("spread_bps_future_")):
            label_coverage.setdefault(column, []).append(float(frame[column].notna().mean()))
        partitions.append({
            "surface": str(path), "rows": int(len(frame)),
            "symbols": int(frame.get("symbol", pd.Series(dtype=str)).nunique()),
            "source_markets": sorted(frame.get("source_market", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        })
    source_bytes = 0
    source_files = 0
    raw_retention_violations: list[str] = []
    for path in audits:
        receipt = json.loads(path.read_text())
        if receipt.get("source_market") != "spot":
            raw_retention_violations.append(f"{path}: source_market is not explicit spot")
        for source in receipt.get("sources", []):
            source_files += 1
            source_bytes += int(source.get("source_size", 0))
            if source.get("raw_payload_retained") is not False or source.get("raw_payload_discarded_after_recap") is not True:
                raw_retention_violations.append(f"{path}: source retention receipt failed")
    # No source payload path is valid in this contract.  This makes accidental
    # persistence visible even when raw files do not use a conventional suffix.
    prohibited_paths = [
        str(path) for path in args.root.rglob("*") if path.is_file()
        and any(token in path.name.lower() for token in ("raw_book", "book_snapshot", "trades", "crypto_trade"))
    ]
    report = {
        "schema": "ares.hf_kraken_orderbook_recap_audit.v1",
        "root": str(args.root),
        "source_contract": "Abraxasccs/kraken-market-data spot book snapshots; futures unavailable in this dataset",
        "retention_contract": "compact order-book recap only; no individual trades; source payloads streamed then discarded",
        "surface_partitions": len(surfaces), "source_audits": len(audits),
        "rows": rows, "valid_book_rows": valid_rows,
        "valid_book_fraction": float(valid_rows / rows) if rows else 0.0,
        "symbols": len(all_symbols), "source_files_streamed": source_files,
        "raw_payload_bytes_discarded_in_memory": source_bytes,
        "raw_columns_persisted": sorted(retained_raw_columns),
        "prohibited_raw_like_files": prohibited_paths,
        "source_retention_violations": raw_retention_violations,
        "mean_label_coverage": {key: float(sum(values) / len(values)) for key, values in label_coverage.items()},
        "partitions": partitions,
    }
    out = args.out or args.root / "contract_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
