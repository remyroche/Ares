#!/usr/bin/env python3
"""Seal a reusable strict leaf-reasoning event store from an input spool."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_event_store import (  # noqa: E402
    StrictEventStoreConfig,
    build_strict_event_store,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-spool-dir", type=Path, required=True, help="completed StrictOOFFamilyInputSpool root")
    parser.add_argument("--output-dir", type=Path, required=True, help="new immutable event-store root")
    parser.add_argument("--max-rows-per-part", type=int, default=500_000)
    parser.add_argument("--compression", choices=("zstd", "snappy", "gzip", "none"), default="zstd")
    args = parser.parse_args()
    store = build_strict_event_store(
        args.input_spool_dir, args.output_dir,
        config=StrictEventStoreConfig(compression=args.compression, max_rows_per_part=args.max_rows_per_part),
    )
    print(store.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
