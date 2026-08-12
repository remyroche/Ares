#!/usr/bin/env python3
"""Fit one canonical three-month Local Distribution Forest Proxy bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_n5_canonical import (  # noqa: E402
    persist_canonical_n5_bundle,
    train_canonical_n5_bundle,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-training-ledger", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    ledger = pd.read_parquet(args.causal_training_ledger)
    bundle = train_canonical_n5_bundle(ledger, cutoff=args.cutoff)
    manifest = persist_canonical_n5_bundle(
        bundle,
        args.out_dir,
        source_hashes={"causal_training_ledger": _sha(args.causal_training_ledger)},
    )
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
