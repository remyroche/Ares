#!/usr/bin/env python3
"""Build compact G2/G3 strict-OOF base-reasoning features; never trains a model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_reasoning_representation import (
    BaseReasoningRepresentationConfig,
    build_base_reasoning_representation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_index", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--prediction-shards-root", type=Path, default=None)
    parser.add_argument("--batch-rows", type=int, default=50_000)
    parser.add_argument("--max-bundle-components", type=int, default=16)
    args = parser.parse_args()
    result = build_base_reasoning_representation(
        args.artifact_index, args.destination,
        prediction_shards_root=args.prediction_shards_root,
        config=BaseReasoningRepresentationConfig(batch_rows=args.batch_rows, max_bundle_components=args.max_bundle_components),
    )
    print(f"wrote {result.row_count} strict-OOF compact rows to {result.artifact_dir}")


if __name__ == "__main__":
    main()
