#!/usr/bin/env python3
"""Audit P8U 175-field feature coverage and warm-state readiness.

This inspection is offline and read-only.  It proves that the sealed feature
union is physically materialised in a target-free causal panel and records
whether an exact same-plan transform-state bundle has been bootstrapped.  It
does not create, mutate, or invoke the live trader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_production_contract import (  # noqa: E402
    P8UPreproductionBundle,
)
from extreme_price_movements.inference.p8u_warm_feature_state import (  # noqa: E402
    P8UWarmFeatureConfig,
    atomic_json,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--full-causal-panel", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable P8U warm-readiness output exists: {args.out_dir}")
    config = P8UWarmFeatureConfig.load(args.config, root=ROOT)
    bundle = P8UPreproductionBundle.load(args.bundle, root=ROOT)
    bundle.verify_artifacts()
    plan = bundle.feature_plan()
    if tuple(plan.full_union) != tuple(config.feature_plan):
        raise ValueError("warm worker and sealed P8U bundle disagree on feature union/order")
    panel_columns = pq.ParquetFile(args.full_causal_panel).schema_arrow.names
    coverage = bundle.assert_feature_coverage(panel_columns)
    frame = pd.read_parquet(args.full_causal_panel, columns=list(config.feature_plan))
    rows = []
    for feature in config.feature_plan:
        values = pd.to_numeric(frame[feature], errors="coerce")
        rows.append({
            "feature": feature,
            "rows": int(len(values)),
            "finite_rows": int(values.notna().sum()),
            "finite_fraction": float(values.notna().mean()),
            "physical_column_present": True,
        })
    field_coverage = pd.DataFrame(rows)
    zero_finite = field_coverage.loc[field_coverage["finite_rows"].eq(0), "feature"].astype(str).tolist()
    state_status = "bootstrap_required"
    state_message = "no initial same-plan state bundle is configured"
    try:
        state_bundle = config.require_state_bundle()
    except (ValueError, FileNotFoundError) as exc:
        state_bundle = None
        state_message = str(exc)
    else:
        state_status = "same_plan_bundle_ready"
        state_message = str(state_bundle)
    args.out_dir.mkdir(parents=True)
    field_coverage.to_parquet(args.out_dir / "feature_coverage_by_field.parquet", index=False, compression="zstd")
    report = {
        "schema": "strict_r3_p8u_warm_feature_readiness_v1",
        "status": "physical_features_ready_state_bootstrap_required" if state_bundle is None else "ready_for_warm_worker",
        "scope": "offline, target-free physical coverage and state-contract readiness only",
        "bundle": str(args.bundle),
        "bundle_sha256": sha256_file(args.bundle),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "feature_union_sha256": config.feature_union_sha256,
        "full_causal_panel": str(args.full_causal_panel),
        "full_causal_panel_sha256": sha256_file(args.full_causal_panel),
        "feature_counts": {
            "router": len(plan.router_features),
            "base": len(plan.base_features),
            "under": len(plan.under_features),
            "union": len(plan.full_union),
        },
        "physical_coverage": coverage.as_dict(),
        "fields_with_zero_finite_values": zero_finite,
        "state_status": state_status,
        "state_message": state_message,
        "required_next_action": (
            "run bootstrap_strict_r3_p8u_warm_feature_state.py on a full, same-contract, target-free raw source panel and full-causal reference; it must pass all 175 field comparisons before this worker may advance state"
            if state_bundle is None else None
        ),
        "order_submission": False,
        "exchange_io": False,
        "outcome_columns_consumed": [],
    }
    atomic_json(args.out_dir / "readiness_report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
