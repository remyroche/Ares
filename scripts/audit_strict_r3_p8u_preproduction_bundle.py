#!/usr/bin/env python3
"""Audit a P8U staging bundle and publish its automatic feature plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_production_contract import (  # noqa: E402
    P8UPreproductionBundle,
    sha256_file,
    write_feature_plan,
)


def _exclusive_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"immutable audit already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-panel",
        type=Path,
        action="append",
        default=[],
        help="Parquet point-in-time feature panel to schema-audit (repeatable).",
    )
    args = parser.parse_args()
    bundle = P8UPreproductionBundle.load(args.bundle, root=ROOT)
    observed = bundle.verify_artifacts()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    plan = write_feature_plan(bundle, args.out_dir / "required_feature_plan.json")
    panel_coverage: list[dict[str, object]] = []
    for panel in args.feature_panel:
        schema = pq.ParquetFile(panel).schema_arrow
        coverage = bundle.feature_coverage(schema.names)
        panel_coverage.append({
            "path": str(panel),
            "sha256": sha256_file(panel),
            **coverage.as_dict(),
        })
    # A live bundle must prove the active materialiser can generate the
    # complete union.  Historical Router/Base-only panels are useful evidence
    # but deliberately cannot satisfy this condition on their own.
    feature_materialisation_ready = bool(panel_coverage) and all(
        bool(item["complete"]) for item in panel_coverage
    )
    readiness_error = ""
    try:
        bundle.assert_submission_allowed()
        submission_allowed = True
    except PermissionError as exc:
        submission_allowed = False
        readiness_error = str(exc)
    report = {
        "schema": "strict_r3_p8u_preproduction_audit_v1",
        "status": "pass_preproduction_only",
        "bundle": str(bundle.path.relative_to(ROOT)),
        "bundle_sha256": sha256_file(bundle.path),
        "verified_artifact_hashes": observed,
        "router_first": {
            "fraction": 0.50,
            "required_before": ["Base", "Under", "BCF", "Current", "MC1"],
            "downstream_subset_guard": "available in P8UPreproductionBundle route boundary",
        },
        "feature_plan": {
            "path": "required_feature_plan.json",
            "full_union_count": plan["counts"]["full_union"],
            "routed_union_count": plan["counts"]["routed_union"],
        },
        "feature_materialisation": {
            "ready": feature_materialisation_ready,
            "panel_coverage": panel_coverage,
            "fail_closed_rule": "every active point-in-time panel must contain the complete sealed Router/Base/Under union before Router scoring",
        },
        "submission_allowed": submission_allowed,
        "submission_block_reason": readiness_error,
    }
    _exclusive_json(args.out_dir / "correctness_report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
