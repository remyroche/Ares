#!/usr/bin/env python3
"""Materialise grouped chronological transport-MDA for a sealed funnel run.

This is a post-funnel development-only sidecar.  It never changes the source
artifact and it refuses an unsealed/mismatched result rather than treating a
partial prediction file as evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_reasoning_grouped_mda import (  # noqa: E402
    GroupedMDAConfig,
    LeafReasoningGroupedMDAError,
    materialize_leaf_reasoning_grouped_mda,
    write_immutable_leaf_reasoning_grouped_mda,
)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--funnel-root", required=True, type=Path, help="sealed immutable L/H/C/S funnel output")
    parser.add_argument("--ledger", required=True, type=Path, help="same strict base-to-meta parquet ledger used by that funnel")
    parser.add_argument("--cluster-root", type=Path, help="required for sealed C funnels: immutable candidate-cluster feature root joined by full strict identity")
    parser.add_argument("--output-dir", required=True, type=Path, help="new immutable grouped-MDA sidecar directory")
    parser.add_argument("--repeats", type=int, default=3, help="pre-declared real joint-permutation repeats (minimum 2)")
    parser.add_argument("--phantom-draws", type=int, default=8, help="pre-declared same-dimensional shadow draws (minimum 8)")
    parser.add_argument("--top-fraction", type=float, default=.10, help="pooled global tail used by MDA")
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    try:
        if args.ledger.suffix.lower() != ".parquet":
            raise ValueError("--ledger must be parquet: grouped MDA reads projected fields one transport/arm at a time")
        result = materialize_leaf_reasoning_grouped_mda(
            args.ledger,
            funnel_root=args.funnel_root,
            cluster_root=args.cluster_root,
            config=GroupedMDAConfig(
                repeats=args.repeats,
                phantom_draws=args.phantom_draws,
                top_fraction=args.top_fraction,
                random_seed=args.seed,
            ),
        )
        output = write_immutable_leaf_reasoning_grouped_mda(result, args.output_dir)
    except (LeafReasoningGroupedMDAError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
