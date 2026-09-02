#!/usr/bin/env python3
"""Run the existing downstream stack on the frozen weighted-rank B/E/T source.

This wrapper changes neither the consensus, MC1, dual-admission, portfolio,
cost, nor execution implementation.  It only constrains the score horizon to
the earliest period supported by the frozen router/new feature universe and
records the explicit weighted-rank upstream in the result lineage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_strict_r3_enhanced_base_live_stack_challenger as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--target-free-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--current-mc1", type=Path, required=True)
    parser.add_argument("--bcf-mc1", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--expected-upstream-coordinate", required=True,
        help="exact source-manifest upstream coordinate expected by this replay",
    )
    parser.add_argument("--score-start", default="2026-01-01")
    parser.add_argument("--score-end", default="2026-07-01")
    parser.add_argument(
        "--meta-resolved-train-months", type=int, default=4,
        help=(
            "number of fully resolved calendar months used by each consensus "
            "fit before its separate 28-day embargo"
        ),
    )
    parser.add_argument(
        "--mc1-train-months", type=int, default=4,
        help="strict preceding calendar months used by each family MC1 fit",
    )
    parser.add_argument(
        "--mc1-threshold-bps", type=float, default=50.0,
        help=(
            "dual current/BCF MC1 admission threshold in common bps; this "
            "changes only the final admission/portfolio gate, never an MC1 fit"
        ),
    )
    parser.add_argument("--evaluation-start", default="2026-01-01")
    parser.add_argument("--evaluation-end", default="2026-07-01")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.meta_resolved_train_months < 1 or args.mc1_train_months < 1:
        raise ValueError("train-month parameters must be positive")
    if not np.isfinite(float(args.mc1_threshold_bps)):
        raise ValueError("mc1 threshold must be finite")
    source_manifest = json.loads((args.target_free_root / "run_manifest.json").read_text())
    upstream = source_manifest.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("coordinate") != args.expected_upstream_coordinate:
        raise AssertionError("target-free source upstream coordinate does not match the declared replay contract")
    start, end = core._utc(args.score_start), core._utc(args.score_end)
    core.SCORE_MONTHS = tuple(pd.date_range(start, end, freq="MS", tz="UTC"))
    # `_score_fold` defines the supervised fit as `[month - N months,
    # month - 28 days)`.  Add one calendar month so the retained interval has
    # four complete resolved months *before* the reserve, rather than silently
    # treating the reserve as part of the requested training horizon.
    core.META_TRAIN_MONTHS = int(args.meta_resolved_train_months) + 1
    core.MAP_TRAIN_MONTHS = int(args.meta_resolved_train_months)
    core.MC1_TRAIN_MONTHS = int(args.mc1_train_months)
    core.MC1_THRESHOLD_BPS = float(args.mc1_threshold_bps)
    evaluation_start, evaluation_end = core._utc(args.evaluation_start), core._utc(args.evaluation_end)
    if evaluation_start >= evaluation_end:
        raise ValueError("evaluation-start must precede evaluation-end")
    core.EVALUATION_PERIODS = {
        "strict_held": (evaluation_start, evaluation_end),
    }
    paths = core.Paths(
        raw_ledger=args.raw_ledger, direct_root=args.target_free_root,
        policy_root=args.policy_root, current_mc1=args.current_mc1,
        bcf_mc1=args.bcf_mc1, bundle_root=args.bundle_root,
    )
    result = core.run(
        paths, args.out, label_spec=core.POLICY_CONVERSION_LABEL_SPECS["residual_actual_100_30_90"],
        shared_target_free_root=args.target_free_root,
        score_architecture="base_consensus_correctness", pairwise_mode="none",
        integration_spec=core.BPS_INTEGRATION_SPECS["rank_75_25"],
        feature_contract="current", trust_arm="generic_correctness",
    )
    manifest_path = result / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "schema": "strict_r3_hpo_threeway_downstream_validation_v1",
        "enhanced_base": args.expected_upstream_coordinate,
        "upstream_override": upstream,
        "score_months": [f"{month:%Y-%m}" for month in core.SCORE_MONTHS],
        "evaluation_scope": "strict base-grid held period: four fully resolved consensus-training months plus a separate 28-day embargo and strict prior-only MC1 fitting; research-only, not untouched promotion evidence",
        "temporal_schedule": {
            "meta_resolved_train_months": int(args.meta_resolved_train_months),
            "meta_calendar_span_months_including_embargo": int(core.META_TRAIN_MONTHS),
            "meta_embargo_days": int(core.RESERVE_DAYS),
            "mc1_train_months": int(args.mc1_train_months),
            "mc1_admission_threshold_bps": float(args.mc1_threshold_bps),
            "held_score_start": start.isoformat(),
            "held_score_end_inclusive_month": end.isoformat(),
            "evaluation_start": evaluation_start.isoformat(),
            "evaluation_end_exclusive": evaluation_end.isoformat(),
        },
        "base_feature_contract_note": "the frozen 120-field meta contract is unchanged; only its target-free upstream B/E/T coordinates were replaced and all downstream supervised consumers were refit prequentially",
    })
    causality = dict(manifest.get("causality", {}))
    causality["base"] = args.expected_upstream_coordinate
    admission_contract = (
        "both retrained family maps >= "
        f"{float(args.mc1_threshold_bps):g} bps; BCF-like mapped EV priority"
    )
    causality["admission"] = admission_contract
    manifest["causality"] = causality
    manifest["admission"] = admission_contract
    manifest["admission_contract"] = {
        "dual_current_bcf_threshold_bps": float(args.mc1_threshold_bps),
        "priority": "bcf_mc1_expected_bps",
        "fit_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out": str(result), "upstream": upstream}), flush=True)


if __name__ == "__main__":
    main()
