from __future__ import annotations

import pandas as pd
import pytest

from scripts.materialize_historical_exact_transition_context_continuation import (
    HistoricalTransitionContinuationError,
    SOURCE_FAMILY,
    assert_frozen_prefix_feature_parity,
    build_continuation_sidecar,
)


FIELDS = (
    "market_pressure",
    "transition_new__breadth__delta_3h",
    "state_context__state_age_hours",
)


def _panel() -> pd.DataFrame:
    source = pd.date_range("2022-12-30T00:00:00Z", periods=4, freq="h")
    return pd.DataFrame({
        "source_utc": source,
        "execution_decision_utc": source + pd.Timedelta(hours=1),
        "market_pressure": [1.0, 2.0, 3.0, 4.0],
        "transition_new__breadth__delta_3h": [0.1, 0.2, 0.3, 0.4],
        "state_context__state_age_hours": [10.0, 11.0, 12.0, 13.0],
        "target__onset_within_12h": [0.0, 0.0, 1.0, 1.0],
    })


def _candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "__ts__": pd.date_range("2022-12-30T00:00:00Z", periods=4, freq="h"),
        "__symbol__": ["A", "B", "C", "D"],
        "side_name": ["long", "short", "long", "short"],
        "candidate_id": ["a", "b", "c", "d"],
    })


def test_prefix_parity_treats_matching_nans_as_equal() -> None:
    frozen = _panel()
    rebuilt = _panel()
    frozen.loc[1, "market_pressure"] = float("nan")
    rebuilt.loc[1, "market_pressure"] = float("nan")

    report = assert_frozen_prefix_feature_parity(frozen, rebuilt, fields=FIELDS)

    assert report["status"] == "EXACT_FEATURE_PARITY"
    assert report["frozen_rows"] == 4
    assert report["feature_count"] == len(FIELDS)


def test_prefix_parity_fails_on_state_history_difference() -> None:
    rebuilt = _panel()
    rebuilt.loc[3, "state_context__state_age_hours"] = 1.0

    with pytest.raises(HistoricalTransitionContinuationError, match="state_context__state_age_hours"):
        assert_frozen_prefix_feature_parity(_panel(), rebuilt, fields=FIELDS)


def test_continuation_sidecar_requires_exact_full_coverage_and_excludes_targets() -> None:
    result = build_continuation_sidecar(
        _candidates(), _panel(), fields=FIELDS,
        expected_rows=4, expected_covered_rows=4,
    )

    assert result["transition_context_available"].all()
    assert result["source_family"].eq(SOURCE_FAMILY).all()
    assert "target__onset_within_12h" not in result
    assert result["state_context__state_age_hours"].tolist() == [10.0, 11.0, 12.0, 13.0]


def test_continuation_sidecar_fails_when_a_candidate_has_no_exact_hour() -> None:
    candidates = _candidates()
    candidates.loc[3, "__ts__"] += pd.Timedelta(hours=1)

    with pytest.raises(HistoricalTransitionContinuationError, match="covered rows"):
        build_continuation_sidecar(
            candidates, _panel(), fields=FIELDS,
            expected_rows=4, expected_covered_rows=4,
        )
