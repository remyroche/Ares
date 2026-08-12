#!/usr/bin/env python3
"""Train one schema-v2 monthly bundle from a strict prequential ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    load_geometry_bundle,
    persist_monthly_bundle,
    train_monthly_bundle,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields(path: Path, key: str, side: str) -> list[str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [str(value) for value in payload]
    if key == "base_fields" and "base_fields_by_side" in payload:
        return [str(value) for value in payload["base_fields_by_side"][side]]
    if key == "context_fields" and "severe_context_fields" in payload:
        return [str(value) for value in payload["severe_context_fields"]]
    return [str(value) for value in payload.get(key, payload.get("fields", []))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--geometry-bundle", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--context-contract", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    ledger = pd.read_parquet(args.prequential_ledger)
    ledger = ledger.loc[ledger["side_name"].astype(str).str.lower().eq(args.side)].copy()
    geometry = load_geometry_bundle(args.geometry_bundle)
    bundle = train_monthly_bundle(
        cutoff=args.cutoff,
        training_ledger=ledger,
        frozen_geometry=geometry,
        base_fields=_fields(args.base_contract, "base_fields", args.side),
        context_fields=_fields(args.context_contract, "context_fields", args.side),
        source_hashes={
            "prequential_ledger": _sha(args.prequential_ledger),
            "geometry_manifest": _sha(args.geometry_bundle / "run_manifest.json"),
            "base_contract": _sha(args.base_contract),
            "context_contract": _sha(args.context_contract),
        },
    )
    manifest = persist_monthly_bundle(bundle, args.out_dir)
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}, default=str))


if __name__ == "__main__":
    main()
