#!/usr/bin/env python3
"""Assemble the immutable strict base-to-meta leaf-reasoning ledger."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_reasoning_meta_ledger import (  # noqa: E402
    assemble_leaf_reasoning_meta_ledger_pairs,
    write_immutable_meta_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-root", type=Path, action="append", required=True,
        help="completed strict source; repeat once per compact source in matching order",
    )
    parser.add_argument(
        "--compact-root", type=Path, action="append", required=True,
        help="completed compact source; repeat once per strict source in matching order",
    )
    parser.add_argument(
        "--health-root", type=Path, default=None,
        help="completed causal H1--H5 root; must cover precisely the combined strict candidate ledger",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.strict_root) != len(args.compact_root):
        parser.error("--strict-root and --compact-root must occur equally often")
    result = assemble_leaf_reasoning_meta_ledger_pairs(
        zip(args.strict_root, args.compact_root), health_root=args.health_root,
    )
    print(write_immutable_meta_ledger(result, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
