#!/usr/bin/env python3
"""Publish a corrected Stage-I selector without recomputing causal features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_selector_timestamp_upgrade import (
    upgrade_stage_i_selector_timestamp_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = upgrade_stage_i_selector_timestamp_contract(
        args.source_dir, args.destination_dir, resume=args.resume
    )
    print(
        json.dumps(
            {
                "status": manifest.get("status"),
                "rows": manifest.get("rows"),
                "destination": str(args.destination_dir),
                "timestamp_contract": manifest.get("timestamp_contract"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
