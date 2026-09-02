#!/usr/bin/env python3
"""Finalize the light reporting phase of the matched MC1/adaptive replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_strict_r3_mc1_adaptive_exit import (
    DEFAULT_BUNDLE, DEFAULT_MAPPERS, DEFAULT_POLICY, _period_rows, _sha,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--mappers", type=Path, default=DEFAULT_MAPPERS)
    args = p.parse_args()
    baseline_decisions = pd.read_parquet(args.run_dir / "baseline_portfolio_decisions.parquet")
    adaptive_decisions = pd.read_parquet(args.run_dir / "adaptive_portfolio_decisions.parquet")
    baseline_equity = pd.read_parquet(args.run_dir / "baseline_equity.parquet")
    adaptive_equity = pd.read_parquet(args.run_dir / "adaptive_equity.parquet")
    support = pd.read_parquet(args.run_dir / "adaptive_support_audit.parquet")
    metrics = pd.DataFrame(
        _period_rows("SimplePolicyOptimiser baseline", baseline_decisions, baseline_equity)
        + _period_rows("SimplePolicyOptimiser + Adaptive Exit V1", adaptive_decisions, adaptive_equity)
    )
    controls = metrics[metrics.arm.eq("SimplePolicyOptimiser baseline")].drop(columns="arm")
    delta = metrics.merge(
        controls, on=["period", "start", "end_exclusive"],
        suffixes=("", "__baseline"), validate="many_to_one",
    )
    numeric = [c for c in metrics if c not in {"arm", "period", "start", "end_exclusive"}]
    for field in numeric:
        if pd.api.types.is_numeric_dtype(delta[field]):
            delta[f"delta_{field}"] = delta[field] - delta[f"{field}__baseline"]
    metrics.to_parquet(args.run_dir / "period_metrics.parquet", index=False)
    metrics.to_csv(args.run_dir / "period_metrics.csv", index=False)
    delta.to_parquet(args.run_dir / "period_metrics_with_delta.parquet", index=False)
    delta.to_csv(args.run_dir / "period_metrics_with_delta.csv", index=False)
    manifest = {
        "schema": "strict_r3_mc1_adaptive_exit_source_aligned_replay_v1",
        "admission": "frozen MC1_d2 expected net >= +50 bps",
        "auction": "frozen strict-R3 final_score",
        "exit_baseline": "frozen SimplePolicyOptimiser",
        "adaptive_exit_role": "activation_only_overlay_on_simple_policy_optimiser",
        "adaptive_controller": "F4_disagreement_abstain_p80",
        "unsupported_historical_action": "preserve SimplePolicyOptimiser outcome exactly",
        "adaptive_candidate_support_rows": int(support.adaptive_exit_historical_supported.sum()),
        "adaptive_candidate_population_rows": int(len(support)),
        "adaptive_candidate_support_fraction": float(support.adaptive_exit_historical_supported.mean()),
        "adaptive_bundle_manifest_sha256": _sha(args.bundle_dir / "run_manifest.json"),
        "adaptive_model_sha256": _sha(args.bundle_dir / "adaptive_exit_v1.joblib"),
        "policy_sha256": _sha(args.policy_json),
        "mapper_source_sha256": _sha(args.mappers),
        "cost_bps": 100.0,
        "portfolio": "long-only; 7x leverage; 10% margin slots; 80% margin cap; 8 concurrent; 2 new entries/hour",
    }
    (args.run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
