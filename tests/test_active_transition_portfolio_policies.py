import numpy as np
import pandas as pd
import pytest

from scripts.ablate_active_transition_portfolio_policies import (
    assert_economic_source_is_valid,
    select_policy_arm,
)


def _tier() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mapped_score": [0.10, 0.09, 0.08, 0.07],
            "active_transition_probability_oof": [1.0, 0.0, 0.0, 0.0],
            "selected_global_top10": [True, True, False, False],
            "__ts__": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "__symbol__": ["A", "B", "C", "D"],
            "side_name": ["long"] * 4,
            "candidate_id": ["a", "b", "c", "d"],
        }
    )


def test_trust_discount_reselects_same_global_count() -> None:
    selected = select_policy_arm(_tier(), policy="trust_discount", value=1.0)
    assert len(selected) == 2
    assert set(selected["candidate_id"]) == {"b", "c"}


def test_threshold_increase_only_removes_frozen_book_rows() -> None:
    selected = select_policy_arm(_tier(), policy="threshold_increase", value=0.02)
    assert set(selected["candidate_id"]) == {"b"}


def test_exposure_reduction_keeps_book_and_scales_risk() -> None:
    selected = select_policy_arm(_tier(), policy="exposure_reduction", value=0.5)
    assert set(selected["candidate_id"]) == {"a", "b"}
    values = selected.set_index("candidate_id")["portfolio_size_multiplier"]
    assert np.isclose(values["a"], 0.5)
    assert np.isclose(values["b"], 1.0)


def test_explicit_economic_invalidation_is_rejected(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    assignments = artifact / "assignments.parquet"
    assignments.touch()
    (artifact / "ECONOMIC_INVALIDATION.json").write_text(
        '{"economic_status":"invalidated_target_lineage_mismatch",'
        '"reason":{"target":"wrong simulator"}}\n'
    )

    with pytest.raises(ValueError, match="invalidated_target_lineage_mismatch"):
        assert_economic_source_is_valid(assignments)
