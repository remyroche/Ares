from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_historical_current_common_transition_geometry import (
    CANONICAL_FEATURES,
    CommonGeometryError,
    RAW_FIELDS,
    attach_historical_candidates,
    build_historical_hourly_state,
    canonical_feature_columns,
    project_current_v4_context,
)


def _raw_rows() -> pd.DataFrame:
    rows = []
    for timestamp, long_value, short_value in [
        ("2023-01-01T00:00:00Z", 1.0, 3.0),
        ("2023-01-01T01:00:00Z", 2.0, 4.0),
        ("2023-01-01T03:00:00Z", 6.0, 8.0),
    ]:
        for side, value in (("long", long_value), ("short", short_value)):
            for duplicate in (0, 1):
                row = {"__ts__": pd.Timestamp(timestamp), "side_name": side}
                row.update({field: value + duplicate for field in RAW_FIELDS})
                rows.append(row)
    return pd.DataFrame(rows)


def test_contract_contains_only_nine_field_canonical_geometry() -> None:
    assert len(RAW_FIELDS) == 9
    assert len(CANONICAL_FEATURES) == 90
    assert canonical_feature_columns() == CANONICAL_FEATURES
    assert all("dae" not in name.lower() and "gmm" not in name.lower() for name in CANONICAL_FEATURES)
    assert all("target" not in name.lower() and "future" not in name.lower() for name in CANONICAL_FEATURES)


def test_historical_geometry_uses_side_median_iqr_and_exact_past_timestamp() -> None:
    result = build_historical_hourly_state(_raw_rows())
    field = RAW_FIELDS[0]
    at_one = result.loc[result["signal_context_utc"].eq(pd.Timestamp("2023-01-01T01:00:00Z"))].iloc[0]
    assert at_one[f"context__state_mean__median__{field}"] == pytest.approx(3.5)
    assert at_one[f"context__state_long_short_gap__median__{field}"] == pytest.approx(-2.0)
    assert at_one[f"context__state_mean__iqr__{field}"] == pytest.approx(0.5)
    assert at_one[f"context__past_delta_1h__median__{field}"] == pytest.approx(1.0)
    at_three = result.loc[result["signal_context_utc"].eq(pd.Timestamp("2023-01-01T03:00:00Z"))].iloc[0]
    assert np.isnan(at_three[f"context__past_delta_1h__median__{field}"])
    assert at_three[f"context__past_delta_3h__median__{field}"] == pytest.approx(5.0)


def test_candidate_attachment_is_signal_to_decision_and_preserves_identity() -> None:
    raw = _raw_rows()
    hourly = build_historical_hourly_state(raw)
    labels = pd.DataFrame({
        "__ts__": [pd.Timestamp("2023-01-01T00:00:00Z")],
        "__symbol__": ["A"], "side_name": ["long"], "candidate_id": ["one"],
    })
    stage = pd.DataFrame({
        "signal_timestamp": labels["__ts__"], "decision_timestamp": labels["__ts__"] + pd.Timedelta(hours=1),
        "symbol": ["A"], "side_name": ["long"], "candidate_id": ["one"],
        "source_shard_path": ["unused"], "source_shard_sha256": ["unused"], "source_row_number": [0],
    })
    result = attach_historical_candidates(labels, stage, hourly)
    assert result["__decision_ts__"].iloc[0] == pd.Timestamp("2023-01-01T01:00:00Z")
    assert result["candidate_id"].tolist() == ["one"]
    assert result["common_transition_context_available"].tolist() == [True]


def test_current_projection_rejects_inconsistent_same_timestamp() -> None:
    first = {"signal_context_utc": pd.Timestamp("2026-06-01T00:00:00Z"), "context_available": True}
    first.update({name: 1.0 for name in CANONICAL_FEATURES})
    second = dict(first)
    second[CANONICAL_FEATURES[0]] = 2.0
    with pytest.raises(CommonGeometryError, match="inconsistent"):
        project_current_v4_context(pd.DataFrame([first, second]), CANONICAL_FEATURES)


def test_current_projection_keeps_only_common_fields_and_unavailable_nulls() -> None:
    row = {"signal_context_utc": pd.Timestamp("2026-06-01T00:00:00Z"), "context_available": False}
    row.update({name: np.nan for name in CANONICAL_FEATURES})
    result = project_current_v4_context(pd.DataFrame([row]), CANONICAL_FEATURES)
    assert result.columns.tolist() == ["signal_context_utc", "common_transition_context_available", *CANONICAL_FEATURES]
    assert result["common_transition_context_available"].tolist() == [False]
