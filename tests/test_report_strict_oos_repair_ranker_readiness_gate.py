import pandas as pd

from scripts.report_strict_oos_repair_ranker_readiness_gate import build_readiness_decision


def _runner_manifest(**overrides):
    manifest = {
        "scope": "strict_oos_repair_ranker_frozen_profile_run",
        "history_periods": ["2026-04", "2026-05", "2026-06"],
        "profile_count": 1,
        "reference_consistency": {
            "enabled": True,
            "status": "checked",
            "passes": True,
            "rows_checked": 2,
        },
        "validation_manifest": {
            "promotion_allowed_count": 1,
            "status_counts": {"passes_frozen_validation": 1},
            "validation_periods": ["2026-07"],
        },
    }
    manifest.update(overrides)
    return manifest


def _validation_row(**overrides):
    row = {
        "profile_name": "candidate",
        "validation_status": "passes_frozen_validation",
        "missing_periods": "",
        "promotion_allowed": True,
    }
    row.update(overrides)
    return row


def test_readiness_gate_allows_clean_frozen_validation():
    decision = build_readiness_decision(
        _runner_manifest(),
        pd.DataFrame([_validation_row()]),
    )

    assert decision["ready_for_training_integration"] is True
    assert decision["decision"] == "ready"
    assert decision["block_reasons"] == []


def test_readiness_gate_blocks_missing_validation_periods():
    decision = build_readiness_decision(
        _runner_manifest(
            validation_manifest={
                "promotion_allowed_count": 0,
                "status_counts": {"fails_frozen_validation": 1},
                "validation_periods": ["2026-07"],
            }
        ),
        pd.DataFrame(
            [
                _validation_row(
                    validation_status="fails_frozen_validation",
                    missing_periods="2026-07",
                    promotion_allowed=False,
                )
            ]
        ),
    )

    assert decision["ready_for_training_integration"] is False
    assert "missing_validation_periods" in decision["block_reasons"]
    assert "no_profiles_passed_frozen_validation" in decision["block_reasons"]


def test_readiness_gate_blocks_reference_mismatch():
    decision = build_readiness_decision(
        _runner_manifest(reference_consistency={"enabled": True, "status": "checked", "passes": False}),
        pd.DataFrame([_validation_row()]),
    )

    assert decision["ready_for_training_integration"] is False
    assert "reference_consistency_failed" in decision["block_reasons"]


def test_readiness_gate_blocks_retrospective_only_profile():
    decision = build_readiness_decision(
        _runner_manifest(
            validation_manifest={
                "promotion_allowed_count": 0,
                "status_counts": {"passes_guards_but_retrospective_only": 1},
                "validation_periods": ["2026-06"],
            }
        ),
        pd.DataFrame([_validation_row(validation_status="passes_guards_but_retrospective_only", promotion_allowed=False)]),
    )

    assert decision["ready_for_training_integration"] is False
    assert "retrospective_only_profile_present" in decision["block_reasons"]
