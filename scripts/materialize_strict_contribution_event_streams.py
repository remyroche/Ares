#!/usr/bin/env python3
"""Build immutable paired, time-ordered contribution event streams."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_contribution_event_stream import (  # noqa: E402
    build_strict_contribution_event_streams,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-store", type=Path, required=True, help="sealed canonical strict event store")
    parser.add_argument("--output-dir", type=Path, required=True, help="new immutable contribution-event-stream root")
    parser.add_argument("--max-output-gb", type=float, default=6.0, help="hard physical-output budget; the temporary build is deleted if exceeded")
    parser.add_argument("--minimum-free-gb", type=float, default=12.0, help="disk space kept free throughout the bounded build")
    args = parser.parse_args()
    print(build_strict_contribution_event_streams(
        args.event_store, args.output_dir,
        max_output_bytes=int(args.max_output_gb * 1024**3),
        minimum_free_bytes=int(args.minimum_free_gb * 1024**3),
    ).root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
