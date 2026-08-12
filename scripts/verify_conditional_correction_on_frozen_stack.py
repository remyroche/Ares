#!/usr/bin/env python3
"""Verify the completed anchored conditional-correction artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact
    manifest = json.loads((root / "run_manifest.json").read_text())
    correctness = json.loads((root / "correctness_test_report.json").read_text())
    contract = json.loads((root / "feature_contract.json").read_text())
    predictions = pd.read_parquet(root / "conditional_oos_predictions.parquet", columns=["candidate_id", "month"])
    metrics = pd.read_parquet(root / "conditional_metrics.parquet")
    checks = {
        "manifest_complete": manifest.get("status") == "complete",
        "correctness_passed": correctness.get("status") == "passed",
        "prediction_ids_unique": bool(predictions["candidate_id"].is_unique),
        "prediction_row_count": int(len(predictions)) == int(manifest.get("rows", -1)),
        "metrics_have_all_arms": set(manifest.get("arms", [])) <= set(metrics["score"].unique()),
        "pooled_global_present": bool((metrics["period"] == "pooled").any()),
        "no_forbidden_features": not any(
            field in set(contract.get("forbidden_outcome_fields", []))
            for group in ("head_score_fields", "condition_fields", "causal_context_fields")
            for field in contract.get(group, [])
        ),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "rows": int(len(predictions)),
        "months": sorted(predictions["month"].dropna().unique().tolist()),
        "arms": manifest.get("arms", []),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
