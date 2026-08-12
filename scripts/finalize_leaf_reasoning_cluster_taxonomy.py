#!/usr/bin/env python3
"""Issue C5/C6 only after immutable C1--C4 and C5-prefix development evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_reasoning_cluster_materializer import (  # noqa: E402
    LeafReasoningClusterFinalizationConfig,
    finalize_leaf_reasoning_cluster_taxonomy,
)


def _table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"{path} must be parquet or CSV")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path, help="completed C1--C4 threshold-sweep candidate root")
    parser.add_argument("--development-metrics", required=True, type=Path, help="immutable C1--C4 meta metrics with top-5/top-10 lifts")
    parser.add_argument("--grouped-mda", required=True, type=Path, help="immutable C1--C4 grouped chronological transport MDA")
    parser.add_argument("--c5-prefix-mda", required=True, type=Path, help="immutable C5 prefix grouped chronological transport MDA")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--c5-top-decile-coverage", type=float, default=.95)
    parser.add_argument("--c5-min-portability", type=float, default=.50)
    parser.add_argument("--c6-soft-cap", type=int, default=12)
    parser.add_argument("--minimum-positive-environment-rate", type=float, default=.70)
    parser.add_argument("--max-worst-month-net-drop-bps", type=float)
    args = parser.parse_args()
    config = LeafReasoningClusterFinalizationConfig(
        c5_top_decile_coverage_target=args.c5_top_decile_coverage,
        c5_min_portability=args.c5_min_portability,
        c6_soft_cap=args.c6_soft_cap,
        minimum_positive_environment_rate=args.minimum_positive_environment_rate,
        max_worst_month_net_drop_bps=args.max_worst_month_net_drop_bps,
    )
    print(finalize_leaf_reasoning_cluster_taxonomy(
        args.candidate_root,
        _table(args.development_metrics),
        _table(args.grouped_mda),
        _table(args.c5_prefix_mda),
        args.output_dir,
        config=config,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
