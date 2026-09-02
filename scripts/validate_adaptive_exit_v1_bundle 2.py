#!/usr/bin/env python3
"""Validate the sealed Adaptive Exit V1 serialization and frozen authority."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.adaptive_exit_v1 import AdaptiveExitV1Bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    bundle = AdaptiveExitV1Bundle.load(args.bundle_dir)
    reference = pd.read_parquet(args.bundle_dir / "serialization_parity_reference.parquet")
    observed = bundle.score(reference, decision_ts=bundle.manifest["activation_ts"])
    expected = reference.set_index("candidate_id")
    actual = observed.set_index("candidate_id")
    comparisons = {
        "adaptive_exit_f1_prediction": 1e-6,
        "adaptive_exit_f4_prediction": 1e-6,
        "adaptive_exit_selected_activation_atr": 1e-6,
    }
    errors = {}
    for field, tolerance in comparisons.items():
        delta = np.abs(
            pd.to_numeric(actual[field], errors="raise").to_numpy(float)
            - pd.to_numeric(expected[field], errors="raise").to_numpy(float)
        )
        errors[field] = float(delta.max(initial=0.0))
        if errors[field] > tolerance:
            raise AssertionError(f"adaptive-exit serialization parity failed: {field}")
    policy = bundle.manifest["policy"]
    checks = {
        "long_only": bundle.manifest["side"] == "long",
        "activation_only": policy["authority"] == "trailing_activation_only",
        "next_15m_effective": policy["effective_from"] == "next_15m_bar",
        "completed_hour_clock": policy["decision_clock"] == "completed_hourly_bar",
        "timeout_h12": int(policy["timeout_hours"]) == 12,
        "cost_once_100bps": float(policy["round_trip_cost_bps"]) == 100.0,
        "research_canonical": bundle.manifest["research_canonical"] is True,
        "live_not_yet_promoted": bundle.manifest["live_canonical"] is False,
    }
    if not all(checks.values()):
        raise AssertionError(f"adaptive-exit authority check failed: {checks}")
    report = {
        "schema": "adaptive_exit_v1_correctness_report_v1",
        "bundle_id": bundle.manifest["bundle_id"],
        "reference_rows": len(reference), "max_absolute_errors": errors,
        "checks": checks, "passed": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
