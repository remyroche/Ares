#!/usr/bin/env python3
"""Materialize the exact-H12 economic headroom/ranking bottleneck diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.economic_headroom_diagnostic import build_diagnostic


ARTIFACTS = ROOT / "data_perp" / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "exact_h12_economic_headroom_diagnostic_20260801_v1")
    args = parser.parse_args()
    manifest = build_diagnostic(
        primary_path=ARTIFACTS / "root_cause_exact_h12_execution_target_pack_20260801_v3" / "primary_labels.parquet",
        target_metrics_path=ARTIFACTS / "exact_h12_target_purity_ablation_20260731_v11" / "target_ablation_metrics.csv",
        target_results_path=ARTIFACTS / "exact_h12_target_purity_ablation_20260731_v11" / "target_ablation_results.parquet",
        policy_summary_path=ARTIFACTS / "controlled_target_supportive_ablation_20260801_v2" / "target_supportive_policy_summary.parquet",
        output_dir=args.output_dir,
    )
    print(f"status={manifest['status']} diagnosis={manifest['report']['diagnosis']}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
