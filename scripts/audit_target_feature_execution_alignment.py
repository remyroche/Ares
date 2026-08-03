#!/usr/bin/env python3
"""Materialise the canonical target/feature/execution alignment audit pack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.target_feature_execution_alignment import build_alignment_audit


ARTIFACTS = ROOT / "data_perp" / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "target_alignment" / "alignment_audit_20260801_v2")
    parser.add_argument("--target-pack", type=Path, default=ARTIFACTS / "root_cause_exact_h12_execution_target_pack_20260801_v3")
    parser.add_argument("--supportive-run", type=Path, default=ARTIFACTS / "controlled_target_supportive_ablation_20260801_v2")
    parser.add_argument("--feature-audit", type=Path, default=ARTIFACTS / "base_execution_feature_lineage_audit_20260801_v1")
    parser.add_argument("--target-metrics", type=Path, default=ARTIFACTS / "exact_h12_target_purity_ablation_20260731_v11" / "target_ablation_metrics.csv")
    parser.add_argument("--include-legacy-blockers", action="store_true", help="Retain historical native-L2 blocker from the older audit contract")
    args = parser.parse_args()
    target_pack = args.target_pack
    supportive = args.supportive_run
    feature = args.feature_audit
    result = build_alignment_audit(
        contract_path=target_pack / "execution_target_contract.json",
        primary_path=target_pack / "primary_labels.parquet",
        supportive_path=target_pack / "supportive_labels.parquet",
        dictionary_path=target_pack / "label_dictionary.parquet",
        support_report_path=target_pack / "support_report.parquet",
        feature_manifest_path=feature / "feature_eligibility_manifest.parquet",
        prediction_lineage_path=feature / "prediction_lineage_audit.parquet",
        fold_manifest_path=supportive / "fold_manifest.parquet",
        oof_manifest_path=supportive / "oof_prediction_manifest.parquet",
        candidate_oof_path=supportive / "oof_target_supportive_predictions.parquet",
        policy_summary_path=supportive / "target_supportive_policy_summary.parquet",
        target_metrics_path=args.target_metrics,
        monthly_side_path=supportive / "target_supportive_monthly_side_metrics.parquet",
        calibration_path=supportive / "target_supportive_score_calibration.parquet",
        output_dir=args.output_dir,
        include_legacy_blockers=args.include_legacy_blockers,
    )
    print(f"status={result['correctness']['status']} passed={result['correctness']['passed_checks']} failed={result['correctness']['failed_checks']}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
