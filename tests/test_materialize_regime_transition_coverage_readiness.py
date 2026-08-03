import pandas as pd
import pytest

from scripts.materialize_regime_transition_coverage_readiness import (
    HourCoverage,
    _validate_candidate_identity,
    assign_reconstruction_reasons,
    build_coverage,
    coverage_metrics,
    event_source_hours,
)


def _event() -> dict[str, object]:
    return {
        "event_id": "event_one",
        "anchor_source_utc": pd.Timestamp("2025-05-03 01:00", tz="UTC"),
        "anchor_decision_utc": pd.Timestamp("2025-05-03 02:00", tz="UTC"),
        "transition_start_utc": pd.Timestamp("2025-05-03 01:00", tz="UTC"),
        "transition_end_utc": pd.Timestamp("2025-05-03 03:00", tz="UTC"),
        "robust_pre_post_shift": 1.5,
        "economic_failure_event_within_6h": None,
    }


def _source(name: str, *, shift: int = 0, missing_last: bool = False) -> HourCoverage:
    index = pd.date_range("2025-05-03 01:00", periods=3, freq="h", tz="UTC")
    if shift:
        index = index + pd.Timedelta(hours=shift)
    if missing_last:
        index = index[:-1]
    return HourCoverage(
        name=name,
        counts=pd.Series([10] * len(index), index=index, dtype="int64"),
        decision_shift_hours=shift,
        identity_complete=True,
        lineage="test",
    )


def test_event_interval_and_decision_axis_are_explicit() -> None:
    event = _event()
    assert list(event_source_hours(event)) == list(
        pd.date_range("2025-05-03 01:00", periods=3, freq="h", tz="UTC")
    )
    assert list(event_source_hours(event, decision_shift_hours=1)) == list(
        pd.date_range("2025-05-03 02:00", periods=3, freq="h", tz="UTC")
    )
    metrics = coverage_metrics(event, _source("health", shift=1))
    assert metrics["covered_hours"] == 3
    assert metrics["full_coverage"] is True


def test_reason_codes_do_not_treat_partial_coverage_as_valid() -> None:
    event = _event()
    sources = {
        name: _source(name, missing_last=name == "raw_1m_execution_path")
        for name in (
            "active_probability_grouped_oof",
            "historical_score_oof",
            "current_score_oof",
            "raw_1m_execution_path",
            "replay_price_path",
            "deployed_policy_geometry",
            "historical_health_context",
            "current_model_health",
            "legacy_score_context",
        )
    }
    coverage = build_coverage(pd.DataFrame([event]), sources, pd.DataFrame())
    row = coverage.iloc[0]
    assert not bool(row["raw_1m_execution_path_full"])
    assert "MISSING_RAW_1M_EXECUTION_PATH" in row["reconstruction_reason_codes"]
    assert not bool(row["archival_common_valid_full"])


def test_reason_assignment_names_current_lineage_gap() -> None:
    row = {
        "active_probability_grouped_oof_full": True,
        "candidate_score_identity_exact_full": True,
        "raw_1m_execution_path_full": True,
        "replay_price_path_full": True,
        "deployed_policy_geometry_full": True,
        "health_any_lineage_full": True,
        "current_score_lineage_full": False,
        "current_model_health_full": False,
        "historical_failure_episode_within_6h": False,
    }
    assert assign_reconstruction_reasons(row) == [
        "CURRENT_SCORE_LINEAGE_NOT_MATERIALIZED",
        "CURRENT_HEALTH_INPUTS_NOT_MATERIALIZED",
        "NO_NATIVE_ECONOMIC_FAILURE_LINK_WITHIN_6H",
    ]


def test_exact_candidate_identity_rejects_duplicate_time_symbol_side_id(tmp_path) -> None:
    path = tmp_path / "duplicate_identity.parquet"
    pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
            "__symbol__": ["AAA/USD:USD"] * 2,
            "side_name": ["long"] * 2,
            "candidate_id": ["same"] * 2,
        }
    ).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate candidate identities"):
        _validate_candidate_identity(path)
