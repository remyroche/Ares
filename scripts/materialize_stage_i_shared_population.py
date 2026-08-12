#!/usr/bin/env python3
"""Freeze the common valid population for R3/scalar/ordinal Stage-I finalists."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_shared_population import SharedPopulationSpec, materialize_shared_population


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-base-selection-dir", type=Path, required=True)
    parser.add_argument("--scalar-winner-dir", type=Path, required=True)
    parser.add_argument("--ordinal-winner-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_shared_population(SharedPopulationSpec(
        r3_base_selection_dir=args.r3_base_selection_dir,
        scalar_winner_dir=args.scalar_winner_dir,
        ordinal_winner_dir=args.ordinal_winner_dir,
        output_dir=args.output_dir,
    )), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
