#!/usr/bin/env python3
"""Materialise strict-OOF C1--C6 candidate cluster features and taxonomy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_reasoning_cluster_materializer import (  # noqa: E402
    LeafReasoningClusterMaterializerConfig,
    materialize_leaf_reasoning_cluster_candidates,
)


def _groups(path: Path | None) -> dict[str, list[str]] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(fields, list) for key, fields in value.items()):
        raise ValueError("--upstream-feature-groups must be a JSON feature-group object")
    return {key: [str(field) for field in fields] for key, fields in value.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-root", type=Path, required=True)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-feature-groups", type=Path, help="immutable ledger meta_feature_groups.json; C0 becomes L0+L2+L3")
    parser.add_argument("--linkage", choices=("average", "complete"), default="average")
    parser.add_argument("--max-rule-instances-per-cell", type=int, default=48)
    parser.add_argument("--max-clusters-per-arm", type=int, default=20)
    parser.add_argument("--c5-top-decile-coverage", type=float, default=.95)
    parser.add_argument("--c5-min-portability", type=float, default=.50)
    parser.add_argument("--c6-soft-cap", type=int, default=12)
    args = parser.parse_args()
    config = LeafReasoningClusterMaterializerConfig(
        linkage=args.linkage,
        max_rule_instances_per_cell=args.max_rule_instances_per_cell,
        max_clusters_per_arm=args.max_clusters_per_arm,
        c5_top_decile_coverage_target=args.c5_top_decile_coverage,
        c5_min_portability=args.c5_min_portability,
        c6_soft_cap=args.c6_soft_cap,
    )
    result = materialize_leaf_reasoning_cluster_candidates(
        args.strict_root, args.compact_root, args.output_dir,
        config=config, upstream_feature_groups=_groups(args.upstream_feature_groups),
    )
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
