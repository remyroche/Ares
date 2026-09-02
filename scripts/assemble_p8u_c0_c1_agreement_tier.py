#!/usr/bin/env python3
"""Assemble a hash-bound target-free C0/C1 agreement-tier score commit.

The assembler has no candidate-generation, label, portfolio, exchange, or
order-submission authority.  It accepts two independently produced mapped-EV
score panels for exactly the same Router50 population, validates that they are
target-free, and materialises the versioned user-directed priority route:

``both-admitted -> C0-only -> C1-only``.

The raw mapped BCF EV is kept distinct from the dominant portfolio ordering
key so downstream execution arithmetic cannot accidentally consume the tier
offset as expected economics.
"""

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

from extreme_price_movements.inference.p8u_c0_c1_agreement_tier import (
    SCHEMA,
    TIER_OFFSET_BPS,
    UNPAIRED_ORDER_C0_THEN_C1,
    UNPAIRED_ORDER_HIGHEST_RAW_BCF,
    select_c0_c1_agreement_tiers,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-scores", type=Path, required=True)
    parser.add_argument("--c1-scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission-floor-bps", type=float, default=50.0)
    parser.add_argument(
        "--unpaired-order",
        choices=("highest_raw_bcf", "c0_then_c1", "c1_then_c0"),
        default=UNPAIRED_ORDER_C0_THEN_C1,
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("agreement-tier output must be immutable")
    c0_path = args.c0_scores.resolve()
    c1_path = args.c1_scores.resolve()
    selected = select_c0_c1_agreement_tiers(
        c0_scores=pd.read_parquet(c0_path),
        c1_scores=pd.read_parquet(c1_path),
        admission_floor_bps=float(args.admission_floor_bps),
        unpaired_order=str(args.unpaired_order),
    )
    out.mkdir(parents=True, exist_ok=False)
    output_path = out / "agreement_tier_target_free_scores.parquet"
    selected.to_parquet(output_path, index=False, compression="zstd")
    counts = selected["admission_provenance"].value_counts().to_dict()
    manifest = {
        "schema": "strict_r3_p8u_c0_c1_agreement_tier_assembly_v1",
        "selection_schema": SCHEMA,
        "scope": "target-free inference assembly only; no labels, policy, portfolio, exchange I/O, or order submission",
        "admission_floor_bps": float(args.admission_floor_bps),
        "selection": (
            "both-admitted -> C0-only -> C1-only"
            if str(args.unpaired_order) == UNPAIRED_ORDER_C0_THEN_C1
            else "both-admitted -> unpaired by selected raw BCF EV; C0 coordinate in both/C0-only and C1 coordinate in C1-only"
        ),
        "unpaired_order": str(args.unpaired_order),
        "portfolio_ordering": {
            "field": "portfolio_order_priority_bps",
            "formula": "auction_priority_bps + 10000 * portfolio_tier",
            "tier_offset_bps": TIER_OFFSET_BPS,
            "raw_execution_ev_field": "auction_priority_bps",
        },
        "inputs": {
            "c0_scores": {"path": str(c0_path), "sha256": _sha256(c0_path)},
            "c1_scores": {"path": str(c1_path), "sha256": _sha256(c1_path)},
        },
        "output": {
            "path": output_path.name,
            "sha256": _sha256(output_path),
            "rows": int(len(selected)),
            "provenance_counts": {str(key): int(value) for key, value in counts.items()},
        },
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(out)


if __name__ == "__main__":
    main()
