#!/usr/bin/env python3
"""Mechanical correctness audit for the transport-supervised archetype run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data_perp/artifacts/transport_supervised_archetypes_20260803_v1"


def run() -> None:
    required = (
        "archetype_rule_candidates.parquet", "archetype_rule_alignment.parquet",
        "archetype_consensus_definitions.json", "archetype_soft_memberships_oof.parquet",
        "archetype_support_by_environment.parquet", "archetype_conditional_effects.parquet",
        "archetype_era_shortcut_audit.parquet", "archetype_transport_mda.parquet",
        "archetype_role_classification.yaml", "archetype_nested_count_ablation.parquet",
        "archetype_final_manifest_long.json", "archetype_final_manifest_short.json",
        "TRANSPORT_SUPERVISED_ARCHETYPE_REPORT.md", "archetype_feature_scalers.parquet",
    )
    checks: dict[str, bool] = {f"artifact_exists__{name}": (ARTIFACT / name).exists() for name in required}
    manifest = json.loads((ARTIFACT / "archetype_discovery_manifest.json").read_text())
    checks["all_configured_meta_features_screened"] = int(manifest["all_configured_meta_features"]) == 587
    coverage = pd.read_parquet(ARTIFACT / "meta_feature_coverage.parquet")
    checks["coverage_gate_applied"] = bool((coverage.loc[coverage.usable, "coverage"] >= .90).all())
    setup = pd.read_parquet(ARTIFACT / "archetype_setup_baseline_oof.parquet")
    maximum_history = pd.to_datetime(setup.max_history_decision_ts, utc=True)
    minimum_held = pd.to_datetime(setup.min_held_decision_ts, utc=True)
    checks["setup_oof_label_embargo"] = bool((maximum_history < minimum_held - pd.Timedelta(hours=13)).all())
    scaler = pd.read_parquet(ARTIFACT / "archetype_feature_scalers.parquet")
    checks["fold_scaler_lineage_complete"] = bool({"fold", "side_name", "head", "feature", "center", "scale"}.issubset(scaler.columns) and np.isfinite(scaler[["center", "scale"]].to_numpy(float)).all() and (scaler.scale.to_numpy(float) > 0).all())
    membership = pd.read_parquet(ARTIFACT / "archetype_soft_memberships_oof.parquet")
    columns = [name for name in membership if name.startswith("frozen_d2__")]
    matrix = membership.loc[:, columns].to_numpy(float)
    checks["membership_is_finite_and_bounded"] = bool(np.isfinite(matrix).all() and (matrix >= 0).all() and (matrix <= 1).all())
    checks["membership_not_forced_simplex"] = bool(np.any(matrix.sum(axis=1) > 1.05))
    dictionary = pd.read_parquet(ARTIFACT / "archetype_oof_membership_dictionary.parquet")
    checks["frozen_catalogue_is_prior_to_test"] = bool(all(max(folds) < 3 for folds in dictionary.loc[dictionary.definition_version.eq("frozen_d2"), "source_discovery_folds"]))
    terminal = json.loads((ARTIFACT / "archetype_terminal_decision.json").read_text())
    checks["no_unsupported_promotion"] = terminal["decision"] == "NO_TRANSPORT_ARCHETYPE_ADVANCES"
    result = {"schema": "transport_archetype_correctness_v1", "all_passed": bool(all(checks.values())), "checks": checks, "promotion_decision": terminal["decision"]}
    (ARTIFACT / "correctness_test_report.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"archetype correctness audit failed: {failed}")
    print(json.dumps(result))


if __name__ == "__main__":
    run()
