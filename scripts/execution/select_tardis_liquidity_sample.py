#!/usr/bin/env python3
"""Select a manifest-bounded chronological L2 research sample without inference.

Selection is explicit by UTC sample date and exact Tardis symbol.  It keeps
unavailable rows in the audit so coverage is measured rather than silently
changed, while the downloader naturally skips them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--dataset-symbols", nargs="+", required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.manifest)
    frame["sample_date"] = pd.to_datetime(frame["sample_date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    selected_dates = set(args.dates)
    selected_symbols = set(args.dataset_symbols)
    selected = frame.loc[
        frame["sample_date"].isin(selected_dates) & frame["dataset_symbol"].isin(selected_symbols)
    ].copy()
    expected = len(selected_dates) * len(selected_symbols)
    coverage = selected.groupby("data_type", dropna=False).agg(
        rows=("dataset_symbol", "size"), pending_or_available=("status", lambda values: int(values.isin(["pending", "available", "downloaded"]).sum())),
        unavailable=("status", lambda values: int(values.str.startswith("unavailable", na=False).sum())),
    ).reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.out, index=False)
    receipt = {
        "schema": "ares.tardis_liquidity_sample.v1", "source_manifest": str(args.manifest),
        "rows": int(len(selected)), "expected_symbol_dates_per_type": expected,
        "dates": sorted(selected_dates), "dataset_symbols": sorted(selected_symbols),
        "coverage": coverage.to_dict("records"),
    }
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
