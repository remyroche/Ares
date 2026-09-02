from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_staged_timestamp_executor import (
    DIRECT_EXPENSIVE_FEATURES,
    P8UStagedTimestampExecutor,
    P8UVectorStateSpec,
)


def _candidates() -> pd.DataFrame:
    stamp = pd.Timestamp("2026-08-29T13:00:00Z")
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "__decision_ts__": [stamp + pd.Timedelta(hours=1)] * 4,
        "side_name": ["long"] * 4,
        "__symbol__": ["A", "B", "C", "D"],
        "__ts__": [stamp] * 4,
    })


def test_route_candidates_retains_only_explicit_router50_identities() -> None:
    candidates = _candidates()
    routed_population = candidates.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].copy()
    routed_population["router_score"] = [.9, .8, .7, .6]
    routed_population["router50_eligible"] = [True, True, False, False]
    routed_population["router_fraction"] = .5
    routed_population["router_timestamp_ordinal"] = [1, 2, 3, 4]
    routed_population["router_timestamp_count"] = 4
    result = P8UStagedTimestampExecutor._route_candidates(candidates, routed_population)
    assert result["candidate_id"].tolist() == ["a", "b"]
    assert result["router50_eligible"].all()


def test_only_the_approved_expensive_fields_use_direct_state() -> None:
    assert DIRECT_EXPENSIVE_FEATURES == (
        "price_rv_7d_robust_z",
        "price_rv_15d_robust_z",
        "liquidity_ratio_peer_resid",
        "ob_depth_l20_to_qv_24h",
    )


def test_regular_vector_state_requires_the_proven_tail_and_components(tmp_path) -> None:
    with pytest.raises(ValueError, match="1,536-hour"):
        P8UVectorStateSpec(
            bootstrap_state_root=tmp_path,
            state_scope="test",
            state_components=("raw",),
            tail_hours=1535,
        )
    with pytest.raises(ValueError, match="state components"):
        P8UVectorStateSpec(
            bootstrap_state_root=tmp_path,
            state_scope="test",
            state_components=(),
        )


def test_regular_vector_state_defaults_to_a_sealed_bootstrap() -> None:
    assert not P8UVectorStateSpec(
        bootstrap_state_root=Path("/tmp"),
        state_scope="test",
        state_components=("raw",),
    ).allow_unsealed_bootstrap


def test_route_candidates_fails_when_router_provenance_cannot_cover_source_identity() -> None:
    candidates = _candidates()
    routed_population = candidates.loc[:1, ["candidate_id", "__decision_ts__", "side_name"]].copy()
    routed_population["router_score"] = [.9, .8]
    routed_population["router50_eligible"] = [True, True]
    routed_population["router_fraction"] = .5
    routed_population["router_timestamp_ordinal"] = [1, 2]
    routed_population["router_timestamp_count"] = 4
    with pytest.raises(AssertionError, match="cover every scored source identity"):
        P8UStagedTimestampExecutor._route_candidates(candidates, routed_population)


def test_finite_input_coverage_rejects_an_entirely_missing_live_field() -> None:
    matrix = pd.DataFrame({"available": [1.0, 2.0], "missing": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="wholly unavailable fields"):
        P8UStagedTimestampExecutor._finite_input_coverage(
            matrix,
            fields=("available", "missing"),
            stage="router_complete_universe",
        )


def test_finite_input_coverage_records_row_local_missingness_for_model_imputation() -> None:
    matrix = pd.DataFrame({"field": [1.0, np.nan, 3.0]})
    audit = P8UStagedTimestampExecutor._finite_input_coverage(
        matrix,
        fields=("field",),
        stage="router_complete_universe",
    )
    assert audit["wholly_unavailable_fields"] == []
    assert audit["minimum_field_coverage"] == pytest.approx(2.0 / 3.0)
