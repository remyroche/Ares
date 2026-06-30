import json
from pathlib import Path

from scripts.audit_size_action_live_materialization import audit_materialization


ARM = "C3em_bagged_safety_c3ed_or_high_value_zero_classifier_expanded_union_gate"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _minimal_research_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "manifest.json",
        {
            "generated_by": "run_exact_state_size_action_learning",
            "policy_variant": "refit_bar4_strategy_bar2",
            "selected_arms": [ARM],
            "outputs": {"schedules": str(run_dir / "size_action_schedules.csv")},
        },
    )
    for name in [
        "size_action_promotion_summary.csv",
        "size_action_replay_vs_label_audit.csv",
        "size_action_action_quality.csv",
        "size_action_schedules.csv",
        "size_action_gate_thresholds.csv",
        "size_action_selected_features.csv",
        "size_action_exact_panel.csv",
    ]:
        (run_dir / name).write_text("placeholder\n")


def _freeze_manifest(path: Path, run_dir: Path) -> None:
    _write_json(
        path,
        {
            "arm": ARM,
            "run_dir": str(run_dir),
            "gate_status": {
                "research_ready": True,
                "production_ready": False,
                "production_blockers": [
                    "true_prospective_frozen_dual_scoring_not_completed",
                    "live_inference_materialization_not_verified",
                ],
            },
            "source_manifest": {"policy_variant": "refit_bar4_strategy_bar2"},
        },
    )


def test_replay_only_freeze_fails_live_materialization(tmp_path: Path) -> None:
    run_dir = tmp_path / "research_run"
    _minimal_research_run(run_dir)
    freeze = tmp_path / "size_action_freeze_manifest.json"
    _freeze_manifest(freeze, run_dir)

    audit = audit_materialization(freeze)

    assert audit["research_ready"] is True
    assert audit["live_materialized"] is False
    assert audit["production_ready"] is False
    assert "live_scorer_manifest_missing" in audit["blockers"]
    assert "live_scorer_file_missing:model_bundle" in audit["blockers"]
    assert "source_run_outputs_are_replay_artifacts_not_live_scorer_artifacts" in audit["warnings"]


def test_complete_fail_closed_scorer_bundle_passes_materialization(tmp_path: Path) -> None:
    run_dir = tmp_path / "research_run"
    _minimal_research_run(run_dir)
    freeze = tmp_path / "size_action_freeze_manifest.json"
    _freeze_manifest(freeze, run_dir)
    scorer_dir = tmp_path / "live_scorer"
    scorer_dir.mkdir()
    _write_json(
        scorer_dir / "size_action_live_scorer_manifest.json",
        {
            "generated_by": "materialize_size_action_live_scorer",
            "mode": "live",
            "coverage": "full_arm",
            "arm": ARM,
            "run_dir": str(run_dir),
            "policy_variant": "refit_bar4_strategy_bar2",
            "feature_columns": ["strategy_rank_q90", "wallet"],
            "model_artifacts": ["size_action_live_scorer.joblib"],
            "missing_components": [],
            "imputation_policy": {"strategy_rank_q90": 0.0, "wallet": 0.0},
            "fail_closed": True,
            "score_contract": {"missing_required_feature": "reject"},
        },
    )
    for name in [
        "size_action_live_scorer.joblib",
        "size_action_live_feature_contract.json",
        "size_action_live_imputation.json",
        "size_action_live_policy_contract.json",
    ]:
        (scorer_dir / name).write_text("placeholder\n")

    audit = audit_materialization(freeze, scorer_bundle_dir=scorer_dir)

    assert audit["research_ready"] is True
    assert audit["live_materialized"] is True
    assert audit["production_ready"] is True
    assert audit["blockers"] == []


def test_partial_component_scorer_bundle_does_not_pass_full_arm_materialization(tmp_path: Path) -> None:
    run_dir = tmp_path / "research_run"
    _minimal_research_run(run_dir)
    freeze = tmp_path / "size_action_freeze_manifest.json"
    _freeze_manifest(freeze, run_dir)
    scorer_dir = tmp_path / "live_scorer"
    scorer_dir.mkdir()
    _write_json(
        scorer_dir / "size_action_live_scorer_manifest.json",
        {
            "generated_by": "materialize_size_action_live_scorer",
            "mode": "live",
            "coverage": "partial_component",
            "component": "high_value_zero_cut_classifier_secondary",
            "missing_components": ["primary_c3ed_union_component"],
            "arm": ARM,
            "run_dir": str(run_dir),
            "policy_variant": "refit_bar4_strategy_bar2",
            "feature_columns": ["strategy_rank_q90", "wallet"],
            "model_artifacts": ["size_action_live_scorer.joblib"],
            "imputation_policy": {"strategy_rank_q90": 0.0, "wallet": 0.0},
            "fail_closed": True,
            "score_contract": {"missing_required_feature": "reject"},
        },
    )
    for name in [
        "size_action_live_scorer.joblib",
        "size_action_live_feature_contract.json",
        "size_action_live_imputation.json",
        "size_action_live_policy_contract.json",
    ]:
        (scorer_dir / name).write_text("placeholder\n")

    audit = audit_materialization(freeze, scorer_bundle_dir=scorer_dir)

    assert audit["research_ready"] is True
    assert audit["live_materialized"] is False
    assert audit["production_ready"] is False
    assert "live_scorer_not_full_arm_coverage" in audit["blockers"]
    assert "live_scorer_missing_components" in audit["blockers"]
