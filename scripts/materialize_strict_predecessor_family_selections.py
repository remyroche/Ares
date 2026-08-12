#!/usr/bin/env python3
"""Freeze H3/H4/H5 family selections from strictly predecessor-resolved rows."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_leaf_health_prerequisites import (  # noqa: E402
    PredecessorFamilySelectionConfig,
    materialize_strict_predecessor_family_selections,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--strict-root", type=Path, action="append", help="completed predecessor strict root; repeat for disjoint roots (legacy direct reader)")
    source.add_argument("--event-store", type=Path, help="sealed reusable strict event store; reads only cutoff-eligible inner-OOF family parts")
    parser.add_argument("--selection-cutoff-utc", required=True, help="only label_available_ts strictly before this timestamp may participate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=24)
    parser.add_argument("--min-independent-timestamps", type=int, default=12)
    parser.add_argument("--min-trading-days", type=int, default=3)
    parser.add_argument("--min-symbols", type=int, default=3)
    parser.add_argument("--max-context-families-per-scope", type=int, default=12)
    parser.add_argument("--max-covariance-families-per-scope", type=int, default=8)
    parser.add_argument("--max-relationship-families-per-scope", type=int, default=12)
    args = parser.parse_args()
    config = replace(
        PredecessorFamilySelectionConfig(),
        min_rows=args.min_rows,
        min_independent_timestamps=args.min_independent_timestamps,
        min_trading_days=args.min_trading_days,
        min_symbols=args.min_symbols,
        max_context_families_per_scope=args.max_context_families_per_scope,
        max_covariance_families_per_scope=args.max_covariance_families_per_scope,
        max_relationship_families_per_scope=args.max_relationship_families_per_scope,
    )
    print(materialize_strict_predecessor_family_selections(
        args.strict_root,
        args.output_dir,
        selection_cutoff_utc=args.selection_cutoff_utc,
        config=config,
        event_store=args.event_store,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
