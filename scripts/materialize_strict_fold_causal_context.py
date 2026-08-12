#!/usr/bin/env python3
"""Materialise the full July-2023--November-2024 strict causal context sidecar.

This command deliberately reuses the existing OOF market-regime materialiser;
it only supplies an exact outcome-free strict candidate population and records
the H1--H5 contract.  It does not materialise any model, health, or policy
artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_leaf_health_prerequisites import (  # noqa: E402
    DEFAULT_HEALTH_CONTEXT_COLUMNS,
    STRICT_CONTEXT_END_EXCLUSIVE_UTC,
    STRICT_CONTEXT_START_UTC,
    materialize_strict_fold_causal_context,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-root", type=Path, action="append", required=True)
    parser.add_argument("--panel", type=Path, required=True, help="existing segmented causal multiview hourly panel")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency", choices=("month", "quarter"), default="quarter")
    parser.add_argument("--purge-hours", type=int, default=12)
    parser.add_argument("--max-features-per-view", type=int, default=20)
    parser.add_argument("--max-lag-hours", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--health-context-column", action="append", default=None,
        help="fixed H3/H4/H5 causal context field; defaults to the predeclared compact ten-field contract",
    )
    args = parser.parse_args()
    fields = tuple(args.health_context_column or DEFAULT_HEALTH_CONTEXT_COLUMNS)
    print(materialize_strict_fold_causal_context(
        args.strict_root,
        args.output_dir,
        panel_path=args.panel,
        start_utc=STRICT_CONTEXT_START_UTC,
        end_exclusive_utc=STRICT_CONTEXT_END_EXCLUSIVE_UTC,
        frequency=args.frequency,
        purge_hours=args.purge_hours,
        max_features_per_view=args.max_features_per_view,
        max_lag_hours=args.max_lag_hours,
        seed=args.seed,
        health_context_columns=fields,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
