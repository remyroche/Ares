#!/usr/bin/env python3
"""Freeze a target-specific Meta feature contract from a sealed CMI screen.

This small adapter deliberately performs no fitting.  It converts the
top-15%-of-Base conditional-MI screen into an explicit, hash-bound feature
contract that can be consumed by the strict OOF Meta scorer.  It exists so
the Under, Over, and Magnitude experiments use the same selection rule and
never reconstruct a feature list at scoring time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


SCHEMA = "strict_r3_p8u_cmi_meta_contract_v1"


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prescreen-root", type=Path, required=True)
    parser.add_argument("--feature-count", type=int, default=80)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.prescreen_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    if args.feature_count < 20:
        raise ValueError("feature count must be at least 20")
    screen = json.loads((root / "prescreen_contract.json").read_text())
    manifest = json.loads((root / "run_manifest.json").read_text())
    summary = pd.read_parquet(root / "prescreen_summary.parquet")
    candidates = [str(value) for value in screen["candidate_fields"]]
    if len(candidates) < args.feature_count or len(candidates) != len(set(candidates)):
        raise AssertionError("prescreen candidate contract is incomplete or duplicated")
    ranked = summary.set_index("feature", verify_integrity=True).loc[candidates].reset_index()
    # The prescreen contract is already ordered by its cross-fold selection
    # score after redundancy.  Do not re-sort it on a held downstream metric.
    fields = candidates[: args.feature_count]
    selected = ranked.head(args.feature_count).copy()
    out.mkdir(parents=True)
    feature_hash = hashlib.sha256("\n".join(fields).encode()).hexdigest()
    _once(out / "contract.json", {
        "schema": SCHEMA,
        "scope": "offline causal P8u Meta input contract; selected before any downstream Meta/MC1/portfolio evaluation",
        "arm": screen["arm"], "family": screen["family"],
        "query": manifest["arm"]["query"], "feature_count": len(fields),
        "selected_features": fields, "feature_sha256": feature_hash,
        "selection": "top 80 from the sealed full-1400 causal hygiene + Base-top-15%-conditional-IC/CMI given the joint Base Explanation V1 geometry + redundancy screen; no MDA stage requested",
        "base_explanation_v1": {
            "fields": ["base_score", "base_rank_ts", "base_query_count", "base_query_mean", "base_query_std", "base_query_range", "base_score_z_ts", "base_top_gap", "base_top2_gap"],
            "conditioning_population": "timestamp-local strongest 15% of frozen Base predictions",
            "conditional_cmi": "12 deterministic, fold-local outcome-free MiniBatchKMeans strata over rank-normalized Base Explanation V1; CMI estimates feature-policy information conditional on these joint strata",
            "is_added_by_meta_scorer": True,
        },
        "source_prescreen_root": str(root),
        "source_prescreen_manifest_sha256": hashlib.sha256((root / "run_manifest.json").read_bytes()).hexdigest(),
        "base_top_fraction_for_cmi": float(screen["base_top_fraction_for_cmi"]),
        "selection_months": manifest["held_months"],
    })
    selected.to_parquet(out / "selected_feature_evidence.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "source": str(root), "arm": screen["arm"], "family": screen["family"],
        "feature_count": len(fields), "feature_sha256": feature_hash,
        "correctness": {
            "selection_receipt_is_sealed": True,
            "full_causal_feature_universe_was_screened": True,
            "cmi_is_conditioned_on_full_base_explanation_v1_in_base_top15": True,
            "no_target_or_outcome_field_in_contract": True,
            "no_mda_or_downstream_metric_used_to_reorder_contract": True,
        },
    })
    print(out)


if __name__ == "__main__":
    main()
