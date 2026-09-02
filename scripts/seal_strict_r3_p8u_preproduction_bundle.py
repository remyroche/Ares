#!/usr/bin/env python3
"""Seal the selected P8U artefacts into a hash-bound preproduction bundle.

The resulting receipt is deliberately non-executable.  It binds the complete
Router50 → Base → Under → dual-MC1 package, policy, feature runtime, and
operator-state tooling by content hash.  A subsequent live-promotion receipt
must still prove a same-contract warm state and exact-exit execution parity;
this command never grants exchange authority.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_production_contract import (  # noqa: E402
    SCHEMA,
    artifact_hash,
    sha256_file,
)


DEFAULTS = {
    "canonical_contract": "config/strict_r3_p8u_routed_f72_underf120_research_canonical_20260829_v7.json",
    "router_contract": "data_perp/artifacts/strict_r3_p8u_router_oof_apr25_jul26_successorlabels_20260828_v1/run_contract.json",
    "base_feature_contract": "data_perp/artifacts/strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3/selection.json",
    "base_hpo_receipt": "data_perp/artifacts/strict_r3_p8u_precision_preservation_hpo_raw_cat_20260827_v2/run_manifest.json",
    "under_feature_contract": "data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_selection_20260828_v2/contracts/under_f120.json",
    "under_score_receipt": "data_perp/artifacts/strict_r3_p8u_under_f120_score_bridge_aug25_aug27_20260828_v1/run_manifest.json",
    "model_package": "data_perp/artifacts/strict_r3_p8u_inference_model_bundle_cutoff_20260828_v1",
    "router_model": "data_perp/artifacts/strict_r3_p8u_inference_model_bundle_cutoff_20260828_v1/models/router_model",
    "base_model": "data_perp/artifacts/strict_r3_p8u_inference_model_bundle_cutoff_20260828_v1/models/base_model",
    "under_model": "data_perp/artifacts/strict_r3_p8u_inference_model_bundle_cutoff_20260828_v1/models/under_model",
    # Seal the complete month-indexed MC1 repository, not only its current
    # latest BCF/Current leaves.  The selector is allowed to choose *only*
    # an exact monthly vintage, so its index must be as immutable as the
    # individual model packages it describes.
    "mc1_package_root": "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_sixmonth_aug25_aug26_20260828_v4",
    "bcf_mc1_model": "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_sixmonth_aug25_aug26_20260828_v4/mc1_packages/family=bcf/month=2026-08",
    "current_mc1_model": "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_sixmonth_aug25_aug26_20260828_v4/mc1_packages/family=current/month=2026-08",
    "mc1_receipt": "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_sixmonth_aug25_aug26_20260828_v4/run_manifest.json",
    "mc1_correctness_receipt": "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_sixmonth_aug25_aug26_20260828_v4/correctness_report.json",
    "mc1_inference_config": "config/strict_r3_p8u_dual_mc1_sixmonth_inference_20260828_v2.json",
    "mc1_selector_runtime": "extreme_price_movements/inference/p8u_mc1_selector.py",
    "policy": "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json",
    "portfolio_engine": "extreme_price_movements/portfolio_policy_replay.py",
    "incremental_feature_runtime": "scripts/materialize_strict_r3_forward_features_incremental_v13.py",
    "production_guard_runtime": "extreme_price_movements/inference/p8u_production_contract.py",
    "model_package_runtime": "extreme_price_movements/inference/p8u_model_package.py",
    "mc1_package_runtime": "extreme_price_movements/inference/p8u_mc1_inference_package.py",
    "sealed_inference_runtime": "extreme_price_movements/inference/p8u_sealed_inference_stack.py",
    # These three modules implement the Router-first staged inference path.
    # Seal them alongside the scorer so a caller cannot hash-check model files
    # while substituting an unreviewed direct or vector feature handoff.
    "router_first_vector_runtime": "extreme_price_movements/inference/p8u_router_first_vectorized.py",
    "direct_timestamp_runtime": "extreme_price_movements/inference/p8u_single_timestamp_graph.py",
    "direct_state_forward_runtime": "extreme_price_movements/inference/p8u_direct_state_forward.py",
    "direct_state_forward_runner": "scripts/forward_strict_r3_p8u_direct_timestamp_state.py",
    "regular_state_forward_runtime": "extreme_price_movements/inference/p8u_regular_state_forward.py",
    "regular_state_forward_runner": "scripts/forward_strict_r3_p8u_regular_feature_state.py",
    "staged_timestamp_runtime": "extreme_price_movements/inference/p8u_staged_timestamp_executor.py",
    "staged_timestamp_runner": "scripts/run_strict_r3_p8u_staged_timestamp_score.py",
    # The stateful successor composes the regular one-row state transition and
    # staged scorer.  It is separately hash-bound so a caller cannot retain
    # the stateful parity receipt while substituting the legacy batch wrapper.
    "stateful_single_timestamp_runtime": "extreme_price_movements/inference/p8u_stateful_single_timestamp_executor.py",
    "stateful_single_timestamp_runner": "scripts/run_strict_r3_p8u_stateful_single_timestamp_executor.py",
    "regular_feature_state_snapshot_runtime": "scripts/materialize_strict_r3_p8u_regular_feature_state_snapshot.py",
    "target_free_candidate_runtime": "scripts/materialize_strict_r3_p8u_target_free_candidates.py",
    "canonical_warm_runtime": "extreme_price_movements/inference/p8u_canonical_warm_runtime.py",
    "canonical_warm_daemon": "scripts/run_strict_r3_p8u_canonical_warm_daemon.py",
    "canonical_single_timestamp_runtime": "extreme_price_movements/inference/p8u_canonical_single_timestamp_runtime.py",
    "canonical_single_timestamp_runner": "scripts/run_strict_r3_p8u_canonical_single_timestamp_executor.py",
    "canonical_state_sealer": "scripts/seal_strict_r3_p8u_canonical_state_bundle.py",
    "source_anchor_sealer": "scripts/seal_strict_r3_p8u_source_anchored_reference.py",
    "canonical_feature_adapter": "extreme_price_movements/inference/p8u_canonical_feature_adapter.py",
    "canonical_feature_state_runtime": "extreme_price_movements/inference/p8u_warm_feature_state.py",
    "canonical_feature_engine": "extreme_price_movements/features.py",
    "canonical_fast_functions": "extreme_price_movements/fast_funcs.py",
    "canonical_oi_feature_engine": "extreme_price_movements/features_oi.py",
    "canonical_feature_config": "extreme_price_movements/config.py",
    "canonical_generation_dependencies": "scripts/run_tp6_sl4_exact170_canonical_consensus.py",
    "warm_feature_config": "config/strict_r3_p8u_canonical_warm_feature_worker_20260829_v6_sourceanchored.json",
    "state_bundle": "data_perp/artifacts/strict_r3_p8u_canonical_warm_state_bundle_20260829_v4_sourceanchored",
    "warm_feature_parity_summary": "data_perp/artifacts/strict_r3_p8u_canonical_stateful_tail1536_sourceanchored_t12_20260829_v1/parity_summary.json",
    # The direct-state + regular-vector composition is a distinct inference
    # contract.  Bind its all-175-field target-free score-parity receipt so a
    # later caller cannot claim equivalence from a feature-only check.
    "staged_score_parity_receipt": "data_perp/artifacts/strict_r3_p8u_staged_full_score_parity_20260829_t14_t16_v3_full_router_gate/correctness_report.json",
    "stateful_score_parity_receipt": "data_perp/artifacts/strict_r3_p8u_stateful_single_timestamp_executor_20260830_t14_t16_v2_asof_full_parity_audit/correctness_report.json",
    # All-valid-path evidence for the frozen rich exit state machine.  It is
    # historical threshold-fill parity, not a claim about exchange fill
    # quality.
    "exact_1m_exit_parity_receipt": "data_perp/artifacts/strict_r3_p8u_exact1m_rich_policy_dual50_aug01_27_20260829_v4_full_live_state_parity/run_manifest.json",
    # The deterministic no-order adapter is bound separately from the score
    # runtime so hash-valid scores cannot be coupled with a substituted auction.
    "execution_portfolio_adapter_contract": "config/strict_r3_p8u_execution_portfolio_adapter_preproduction_20260830_v3_multicandle_parity.json",
    "execution_portfolio_adapter_runtime": "extreme_price_movements/inference/p8u_execution_portfolio_adapter.py",
    "execution_portfolio_adapter_runner": "scripts/prepare_strict_r3_p8u_execution_intent.py",
}


TREE_ROLES = frozenset({
    "model_package",
    "router_model",
    "base_model",
    "under_model",
    "mc1_package_root",
    "bcf_mc1_model",
    "current_mc1_model",
    "state_bundle",
})


def _entry(relative: str, *, root: Path, artifact_type: str) -> dict[str, str]:
    path = (root / relative).resolve()
    if root not in path.parents or not path.exists():
        raise FileNotFoundError(path)
    if artifact_type == "file" and not path.is_file():
        raise ValueError(f"expected file for P8U artefact: {path}")
    if artifact_type == "tree" and not path.is_dir():
        raise ValueError(f"expected directory for P8U artefact: {path}")
    return {"path": relative, "type": artifact_type, "sha256": artifact_hash(path, artifact_type)}


def _write_exclusive(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"immutable P8U bundle already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_parity_evidence(artifacts: dict[str, dict[str, str]], *, root: Path) -> None:
    """Refuse a bundle which merely names, rather than proves, its key gates."""
    score = json.loads((root / artifacts["staged_score_parity_receipt"]["path"]).read_text(encoding="utf-8"))
    if score.get("status") != "pass" or int(score.get("mismatch_cells", -1)) != 0:
        raise ValueError("staged score parity receipt is not a zero-mismatch pass")
    stateful_score = json.loads(
        (root / artifacts["stateful_score_parity_receipt"]["path"]).read_text(encoding="utf-8")
    )
    if stateful_score.get("status") != "pass" or int(stateful_score.get("mismatch_cells", -1)) != 0:
        raise ValueError("stateful single-timestamp score parity receipt is not a zero-mismatch pass")
    exact = json.loads((root / artifacts["exact_1m_exit_parity_receipt"]["path"]).read_text(encoding="utf-8"))
    oracle = exact.get("oracle_equivalence")
    if exact.get("status") != "complete" or not isinstance(oracle, dict):
        raise ValueError("exact-1m exit receipt is incomplete")
    if int(oracle.get("oracle_equivalence_rows", 0)) <= 0 or int(oracle.get("live_state_machine_equivalence_rows", 0)) <= 0:
        raise ValueError("exact-1m exit receipt lacks live-state-machine parity")
    adapter = json.loads((root / artifacts["execution_portfolio_adapter_contract"]["path"]).read_text(encoding="utf-8"))
    if adapter.get("order_submission") is not False or adapter.get("status") != "preproduction_no_order":
        raise ValueError("execution adapter contract must remain preproduction no-order")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=default)
    args = parser.parse_args()
    root = ROOT.resolve()
    artifacts = {
        name: _entry(
            str(getattr(args, name)),
            root=root,
            artifact_type="tree" if name in TREE_ROLES else "file",
        )
        for name in DEFAULTS
    }
    _validate_parity_evidence(artifacts, root=root)
    payload = {
        "schema": SCHEMA,
        "side": "long",
        "source": "P8U Router50 + F72 Base + Under F120 + dual six-month MC1 source-anchored research contract v9",
        "routing": {
            "fraction": 0.50,
            "scope": "full point-in-time candidate universe",
            "rule": "Router50 must run before Base, Under, BCF/Current, and MC1",
        },
        "artifacts": artifacts,
        "runtime": {
            "order_submission": False,
            "promotion_status": "blocked_preproduction",
            "blockers": [
                "untouched prospective forward evidence is still required before any promotion",
                "the current MC1 package ends in August; a causally trained September vintage must be sealed before September trading",
                "this bundle and its portfolio adapter intentionally have no exchange or order-submission authority",
            ],
            "feature_execution": {
                "requested_feature_keys": "automatic union from hash-bound Router/Base/Under feature contracts",
                "state_contract": "complete canonical transform state advances transactionally from exactly one full-universe primitive source row; a 1,536-hour decision-time panel is prohibited",
                "full_history_recomputation_allowed": False,
                "single_timestamp_runtime": "the hash-bound stateful executor advances the native direct state and regular state-only projection from exactly one target-free full-universe source row, then composes the full 175-field Router/Base/Under matrix before any score exists; abandoned private state attempts remain preserved and cannot be overwritten",
                "hvn_lvn_scheduling": "single-worker only for the one-row transaction; this changes scheduling, not the per-symbol deterministic calculation",
                "state_parity_status": "pass: three consecutive target-free appended candles; all 175 features and Router/Base/Under/MC1 scores had zero mismatched cells in both staged and stateful executor evidence",
                "exact_exit_parity_status": "pass: 2,050 valid exact one-minute paths equal the completed-one-minute live policy state machine; historical threshold-fill proxy only",
                "execution_adapter_status": "sealed deterministic no-order portfolio-intent adapter; exchange/account gateway remains outside this contract",
                "state_full_universe_symbols": 160,
                "feature_coverage_requirement": "the active point-in-time panel must contain every member of the sealed Router/Base/Under union; missing fields fail before Router scoring",
                "routing_execution": "score Router on the complete point-in-time universe; send only exact Router50 identities to Base, Under, BCF/Current and MC1",
            },
        },
    }
    output = args.out.resolve()
    if root not in output.parents:
        raise ValueError("bundle output escapes repository root")
    _write_exclusive(output, payload)
    print(json.dumps({
        "status": "sealed_preproduction_only",
        "bundle": str(output.relative_to(root)),
        "bundle_sha256": sha256_file(output),
        "artifact_count": len(artifacts),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
