#!/usr/bin/env python3
"""Create the metadata-only Pack-B binding for a completed static store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.packb_static_point_feature_loader import (
    _bound_store_scan_manifest_sha256,
    write_packb_store_inventory_binding,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-store-dir",
        type=Path,
        default=ROOT / "data_perp/features/20260711_070000",
    )
    args = parser.parse_args()
    path = write_packb_store_inventory_binding(args.feature_store_dir)
    print(f"bound={path}")
    print(f"sha256={_bound_store_scan_manifest_sha256(args.feature_store_dir)}")


if __name__ == "__main__":
    main()
