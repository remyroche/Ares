#!/usr/bin/env python3
"""Assemble the sealed A/L/H/C/S leaf-reasoning terminal report.

This CLI is intentionally only a hash/provenance consumer.  It cannot train,
score, tune, cluster, or open final OOS.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from extreme_price_movements.leaf_reasoning_portability_report import (
    assemble_feature_leaf_reasoning_portability_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a", required=True, type=Path, help="sealed Stage-A artifact root or manifest")
    parser.add_argument("--stage-l", required=True, type=Path, help="sealed Stage-L artifact root or manifest")
    parser.add_argument("--stage-h", required=True, type=Path, help="sealed Stage-H artifact root or manifest")
    parser.add_argument("--stage-c", required=True, type=Path, help="sealed Stage-C artifact root or manifest")
    parser.add_argument("--stage-s", required=True, type=Path, help="sealed Stage-S artifact root or manifest")
    parser.add_argument("--output", required=True, type=Path, help="new immutable terminal-report directory")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = assemble_feature_leaf_reasoning_portability_report(
        {"A": args.stage_a, "L": args.stage_l, "H": args.stage_h, "C": args.stage_c, "S": args.stage_s},
        args.output,
    )
    print(result.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public assembler tests
    raise SystemExit(main())
