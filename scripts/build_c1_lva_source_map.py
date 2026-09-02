#!/usr/bin/env python3
"""Seal a target-free C1-LVA source-map manifest from immutable candidate rows.

The manifest is deliberately upstream-only: it derives only the sorted source
symbol keys used by the public 15-minute refresher and persists the source
panel hash. It has no labels, model, admission, portfolio, exchange, or order
authority. Immutable output prevents a later universe change from silently
altering a C1 append-state bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"source-map output must be immutable: {output}")
    columns = set(pd.read_parquet(source, columns=None).columns)
    if "__symbol__" not in columns:
        raise KeyError("C1 source panel lacks __symbol__")
    prohibited = [
        column for column in columns
        if any(token in str(column).lower() for token in ("outcome", "label", "policy_net", "exact_net", "gross_bps"))
    ]
    if prohibited:
        raise ValueError(f"C1 source-map input contains outcome-derived fields: {prohibited[:4]}")
    symbols = sorted(pd.read_parquet(source, columns=["__symbol__"])["__symbol__"].astype(str).drop_duplicates())
    if not symbols:
        raise ValueError("C1 source-map input has no symbols")
    payload = {
        "schema": "strict_r3_c1_lva_source_map_v1",
        "scope": "target-free C1-LVA public-source refresh manifest; no labels, model, admission, portfolio, exchange, or order authority",
        "source": {"path": str(source), "sha256": _sha256(source)},
        "source_map": {symbol: symbol for symbol in symbols},
        "symbols": len(symbols),
        "causality": "Symbol membership is frozen from an immutable target-free C1 source panel; a source refresh may append only completed public bars.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
