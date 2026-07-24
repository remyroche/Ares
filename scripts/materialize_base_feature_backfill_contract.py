#!/usr/bin/env python3
"""Write the exact selected base-input/symbol contract for incremental repair.

This keeps an historical rescore repair narrow: only model-selected raw inputs
and only the symbols that actually entered the scorer's canonical universe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-columns", type=Path, required=True)
    parser.add_argument("--scored-frame", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    columns = json.loads(args.base_columns.read_text(encoding="utf-8"))
    feature_by_side = dict(columns.get("feature_names_by_side", {}) or {})
    selected = sorted({str(name) for names in feature_by_side.values() for name in names})
    if not selected:
        raise ValueError("Base columns artifact has no selected features")
    rows = pd.read_parquet(args.scored_frame, columns=["__symbol__"])
    symbols = sorted(rows["__symbol__"].astype(str).unique())
    if not symbols:
        raise ValueError("Scored frame has no symbols")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys_path = args.output_dir / "base_selected_feature_keys.txt"
    symbols_path = args.output_dir / "base_scoring_symbols.txt"
    keys_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    symbols_path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    manifest = {
        "schema": "base_feature_backfill_contract_v1",
        "base_columns": str(args.base_columns),
        "scored_frame": str(args.scored_frame),
        "selected_feature_count": len(selected),
        "symbol_count": len(symbols),
        "keys_path": str(keys_path),
        "symbols_path": str(symbols_path),
        "contract_hash": columns.get("feature_contract_hash"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
