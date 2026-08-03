from pathlib import Path

from scripts.audit_2022_2026_regime_objective_completion import DEFAULT_OUTPUT, build_audit


def test_empty_artifact_root_reports_missing_without_claiming_completion(tmp_path: Path) -> None:
    audit, todo, summary = build_audit(tmp_path)
    # The BRL implementation is source evidence, even when the supplied
    # artifact root is empty; it must still remain explicitly incomplete.
    assert set(audit.status).issubset({"missing", "incomplete"})
    assert "bayesian_rule_list_method" in set(audit.requirement)
    assert len(todo) >= 5
    assert audit.requirement.is_unique
    stale = " ".join(todo.next_work).lower()
    assert "run the optional bayesian rule-list" not in stale
    assert "materialize a causal regime-feature inventory" not in stale
    assert "extend feature, covariance and interaction-shift testing" not in stale
    assert summary["proved_is_not_promotion"] is True


def test_v20_default_binds_timestamp_book_risk_but_keeps_forward_economics_incomplete() -> None:
    assert DEFAULT_OUTPUT.name.endswith("_v20")
    audit, _, _ = build_audit(Path("data_perp/artifacts"))
    statuses = audit.set_index("requirement").status
    assert statuses["direct_model_failure_incremental_value_learning"] == "proved"
    assert statuses["context_incremental_value_beyond_score_only"] == "proved"
    assert statuses["timestamp_level_book_risk_calibration"] == "proved"
    assert statuses["timestamp_level_expected_downside_broadcast"] == "proved"
    assert statuses["frozen_2026_failure_incremental_economics_application"] == "incomplete"
    assert statuses["december_2025_final12_frozen_context_reproduction"] == "incomplete"
