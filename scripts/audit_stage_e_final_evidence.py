#!/usr/bin/env python3
"""Independent fail-closed verifier for the final Stage-E evidence pack."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "STAGE_E_INDEPENDENT_COMPLETION_AUDIT_20260731.md"
ALLOWED = {
    "STAGE_D_PASS_REVOKED_TARGET_PROXY_OR_CAUSAL_DEFECT",
    "CLEAR_EVENT_ACTION_SIGNAL_VALID_BUT_FRAGILE",
    "CLEAR_EVENT_ACTION_SIGNAL_RECONFIRMED",
    "CLEAR_EVENT_ACTION_OVERLAY_ADVANCES",
}
REQUIRED_TESTS = {
    "test_estimated_exit_net_uses_decision_time_price_only",
    "test_estimated_exit_net_does_not_equal_realised_next_fill_by_construction",
    "test_a0_independent_recomputation_matches_sealed_features",
    "test_no_target_column_used_to_reconstruct_a0",
    "test_minimal_ablation_rows_match_m0",
    "test_minimal_ablation_folds_match_m0",
    "test_feature_group_deletion_is_train_only",
    "test_conditional_permutation_preserves_day_and_side_structure",
    "test_frozen_model_is_not_refit_on_second_oos",
    "test_second_oos_was_not_used_by_prior_stages",
    "test_latency_replay_keeps_model_decisions_frozen",
    "test_slippage_replay_applies_incremental_cost_once",
    "test_full_overlay_keeps_entry_population_identical",
    "test_non_clear_candidates_follow_frozen_policy",
    "test_only_first_clear_action_is_changed",
    "test_no_portfolio_or_sizing_logic_is_invoked",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identical_dirs(a: Path, b: Path) -> bool:
    left = {p.name: sha(p) for p in a.iterdir() if p.is_file()}
    right = {p.name: sha(p) for p in b.iterdir() if p.is_file()}
    return left == right


def main() -> None:
    final = json.loads((ROOT / "run_manifest.json").read_text())
    correctness = json.loads((ROOT / "correctness_test_report.json").read_text())
    checks: dict[str, bool] = {}

    for rel, expected in final["canonical_evidence_manifests_sha256"].items():
        path = ROOT / rel
        checks[f"hash:{rel}"] = path.is_file() and sha(path) == expected
    for rel, expected in final["code_sha256"].items():
        checks[f"hash:{rel}"] = sha(ROOT / rel) == expected
    for rel, expected in final["final_outputs_sha256"].items():
        checks[f"hash:{rel}"] = sha(ROOT / rel) == expected

    required_artifacts = [
        "stage_e_a0_feature_inventory.parquet", "stage_e_a0_target_proximity_audit.md",
        "stage_e_independent_feature_recomputation.json", "stage_e_minimal_ablation_results.parquet",
        "stage_e_leave_group_out_results.parquet", "stage_e_conditional_permutation_results.parquet",
        "stage_e_execution_sensitivity.parquet", "stage_e_execution_sensitivity_summary.md",
        "stage_e_second_oos_manifest.json", "stage_e_second_oos_results.parquet",
        "stage_e_second_oos_bootstrap.parquet", "stage_e_full_candidate_overlay.parquet",
        "stage_e_full_candidate_waterfall.parquet",
    ]
    roots = [
        ROOT / "data_perp/artifacts/stage_e_a0_causal_sufficiency_20260731_v3",
        ROOT / "data_perp/artifacts/stage_e_minimal_information_diagnostics_20260731_v1",
        ROOT / "data_perp/artifacts/stage_e_execution_sensitivity_20260731_v4",
        ROOT / "data_perp/artifacts/stage_e_second_oos_readiness_20260731_v1",
        ROOT / "data_perp/artifacts/stage_e_full_candidate_overlay_20260731_v1",
    ]
    found = {p.name for root in roots for p in root.iterdir() if p.is_file()}
    checks["all_13_stage_artifacts_exist"] = set(required_artifacts).issubset(found)
    checks["final_three_deliverables_exist"] = all((ROOT / p).is_file() for p in (
        "STAGE_E_FINAL_REPORT.md", "correctness_test_report.json", "run_manifest.json"))

    e1 = json.loads((roots[0] / "run_manifest.json").read_text())
    recompute = json.loads((roots[0] / "stage_e_independent_feature_recomputation.json").read_text())
    checks["e1_terminal_is_causal_revocation"] = e1["status"] == final["terminal_decision"]
    checks["unavailable_selected_cost_fails_reconstruction"] = (
        not recompute["passed"] and "known_row_cost_bps" in recompute["selected_failures"]
    )
    checks["cost_is_outcome_derived"] = "label_execution_cost_return" in e1["realised_cost_target_proximity"]["source_field"]

    e5 = json.loads((roots[3] / "stage_e_second_oos_manifest.json").read_text())
    checks["e5_not_run_without_refit_or_results"] = (
        e5["status"] == "NOT_RUN_FROZEN_MODEL_ARTIFACT_UNAVAILABLE"
        and not e5["model_refit"] and not e5["results_opened"]
        and pd.read_parquet(roots[3] / "stage_e_second_oos_results.parquet").empty
        and pd.read_parquet(roots[3] / "stage_e_second_oos_bootstrap.parquet").empty
    )

    overlay = pd.read_parquet(roots[4] / "stage_e_full_candidate_overlay.parquet")
    checks["overlay_identity_and_arithmetic"] = (
        len(overlay) == overlay.candidate_id.nunique() == 132248
        and (overlay.p1_net_bps < 0).mean() > 0
        and abs(overlay.incremental_net_bps.mean() - 32.53903708804115) < 1e-9
        and abs(overlay.p1_net_bps.mean() - (-152.57341730245963)) < 1e-9
    )
    report = (ROOT / "STAGE_E_FINAL_REPORT.md").read_text()
    present_decisions = {value for value in ALLOWED if value in report}
    checks["exactly_one_allowed_terminal_decision_in_report"] = present_decisions == {final["terminal_decision"]}
    checks["all_required_named_tests_recorded_pass"] = (
        correctness["passed"] == correctness["collected"] == 28
        and REQUIRED_TESTS == set(correctness["required_named_tests"])
        and all(value.startswith("passed") for value in correctness["required_named_tests"].values())
    )
    checks["e1_reproducible"] = identical_dirs(roots[0], ROOT / "data_perp/artifacts/stage_e_a0_causal_sufficiency_20260731_v4")
    checks["e2_e3_reproducible"] = identical_dirs(roots[1], ROOT / "data_perp/artifacts/stage_e_minimal_information_diagnostics_20260731_v2")
    checks["e4_reproducible"] = identical_dirs(roots[2], ROOT / "data_perp/artifacts/stage_e_execution_sensitivity_20260731_v5")
    checks["e6_reproducible"] = identical_dirs(roots[4], ROOT / "data_perp/artifacts/stage_e_full_candidate_overlay_20260731_v2")

    failed = [name for name, passed in checks.items() if not passed]
    lines = [
        "# Independent Stage-E completion audit", "",
        f"Result: **{'PASS' if not failed else 'FAIL'}**.", "",
        f"Checks passed: **{sum(checks.values())}/{len(checks)}**.", "",
        "The audit independently verifies required artifacts, recorded hashes, causal-failure semantics, empty E5 blocked outputs, complete-population overlay arithmetic, the single allowed terminal decision, required named tests, and byte-identical companion runs.", "",
        "## Checks", "",
        *[f"- [{'x' if passed else ' '}] `{name}`" for name, passed in checks.items()], "",
        f"Failures: `{failed}`.", "",
    ]
    OUT.write_text("\n".join(lines))
    if failed:
        raise SystemExit(f"Stage-E completion audit failed: {failed}")
    print(json.dumps({"status": "PASS", "checks": len(checks), "report": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
