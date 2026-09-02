#!/usr/bin/env python3
"""Create a read-only C0--C8 Stage-C feature-group readiness artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_c_feature_group_readiness import AuditInputs, run


PANEL_ROOT = ROOT / "data_perp/artifacts/stage_c_continuation_feature_panel_20260731_v2"
STAGE1_ROOT = ROOT / "data_perp/artifacts/stage_c_conditional_retention_ablation_20260731_v4"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/stage_c_feature_group_readiness_20260801_v1"


def _inputs(panel_root: Path, stage1_root: Path) -> AuditInputs:
    return AuditInputs(
        feature_panel=panel_root / "stage_c_candidate_population.parquet",
        panel_groups=panel_root / "retention_feature_groups.json",
        panel_lineage=panel_root / "feature_source_lineage.parquet",
        panel_coverage=panel_root / "feature_coverage_by_month_side_symbol.parquet",
        stage1_identity=stage1_root / "retention_evaluation_candidate_ids.parquet",
        stage1_stability=stage1_root / "retention_feature_stability.parquet",
        stage1_manifest=stage1_root / "run_manifest.json",
        stage1_results=stage1_root / "retention_conditional_results.parquet",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--stage1-root", type=Path, default=STAGE1_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(inputs=_inputs(args.panel_root, args.stage1_root), output=args.output), indent=2, default=str))


if __name__ == "__main__":
    main()
