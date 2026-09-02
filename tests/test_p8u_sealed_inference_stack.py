from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from extreme_price_movements.inference.p8u_model_package import timestamp_desc_rank
from extreme_price_movements.inference.p8u_production_contract import (
    IDENTITY_COLUMNS,
    P8UFeaturePlan,
    P8URouterFirstBoundary,
)
from extreme_price_movements.inference.p8u_sealed_inference_stack import (
    P8USealedInferenceStack,
    _strict_shift,
)


def _scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    identities = {
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2026-08-02T01:00:00Z"] * 2, utc=True),
        "side_name": ["long", "long"],
    }
    base = pd.DataFrame({**identities, "base_score": [2.0, 1.0], "base_rank_ts": [.75, .25]})
    under = pd.DataFrame({**identities, "under_raw_score": [1.0, 2.0], "under_rank_ts": [.25, .75]})
    return base, under


def test_p8u_coordinates_match_selected_base_under_contract() -> None:
    base, under = _scores()
    current, bcf = P8USealedInferenceStack._assemble_coordinates(base, under)
    assert np.allclose(current["upstream"], [.625, .375])
    assert np.allclose(current["final_score"], current["upstream"])
    assert np.allclose(current["correctness_rank"], [.5, .5])
    assert np.allclose(bcf["final_score"], [.75, .25])
    assert np.allclose(bcf["correctness_rank"], [.5, .5])


def test_shift_state_must_cover_every_decision_day_and_be_causal() -> None:
    decision = pd.Series(pd.to_datetime(["2026-08-02T01:00:00Z"], utc=True))
    state = pd.DataFrame({
        "decision_day": pd.to_datetime(["2026-08-02T00:00:00Z"], utc=True),
        "recent_shift_bps": [10.0],
        "max_policy_label_available_ts": pd.to_datetime(["2026-08-01T23:00:00Z"], utc=True),
    })
    assert np.allclose(_strict_shift(np.array([5.0]), decision, state, family="bcf"), [15.0])
    with pytest.raises(ValueError, match="no causal calibration"):
        _strict_shift(np.array([5.0]), decision + pd.Timedelta(days=1), state, family="bcf")
    state.loc[0, "max_policy_label_available_ts"] = pd.Timestamp("2026-08-02T00:00:00Z")
    with pytest.raises(ValueError, match="not causal"):
        _strict_shift(np.array([5.0]), decision, state, family="bcf")


class _StagedPreproduction:
    """Small hash-bound boundary substitute used to exercise staged wiring."""

    def __init__(self) -> None:
        self.boundary = P8URouterFirstBoundary(
            P8UFeaturePlan(
                router_features=("router_feature",),
                base_features=("base_feature",),
                under_features=("under_feature",),
                full_union=("router_feature", "base_feature", "under_feature"),
                routed_union=("base_feature", "under_feature"),
            )
        )
        self.verifications = 0

    def verify_artifacts(self) -> dict[str, str]:
        self.verifications += 1
        return {}

    def router_first_boundary(self) -> P8URouterFirstBoundary:
        return self.boundary


class _StagedModels:
    def __init__(self) -> None:
        self.router_calls = 0
        self.base_inputs: list[pd.DataFrame] = []
        self.under_inputs: list[pd.DataFrame] = []

    def score_router(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.router_calls += 1
        result = frame.loc[:, list(IDENTITY_COLUMNS)].copy()
        result["router_score"] = frame["router_feature"].to_numpy(float)
        return result

    def score_base(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.base_inputs.append(frame.copy())
        result = frame.loc[:, list(IDENTITY_COLUMNS)].copy()
        result["base_score"] = frame["base_feature"].to_numpy(float)
        result["base_rank_ts"] = timestamp_desc_rank(result, "base_score")
        return result

    def score_under(self, frame: pd.DataFrame, _base: pd.DataFrame) -> pd.DataFrame:
        self.under_inputs.append(frame.copy())
        result = frame.loc[:, list(IDENTITY_COLUMNS)].copy()
        result["under_raw_score"] = frame["under_feature"].to_numpy(float)
        result["under_rank_ts"] = timestamp_desc_rank(result, "under_raw_score")
        return result


class _StaticMapper:
    def predict_static(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), 75.0, dtype=float)


class _StagedSelector:
    def __init__(self) -> None:
        state = pd.DataFrame({
            "decision_day": pd.to_datetime(["2026-08-02T00:00:00Z"], utc=True),
            "recent_shift_bps": [0.0],
            "max_policy_label_available_ts": pd.to_datetime(["2026-08-01T23:00:00Z"], utc=True),
        })
        package = _StaticMapper()
        family = SimpleNamespace(package=package, shift_state=state)
        self.vintage = SimpleNamespace(current=family, bcf=family)

    def select(self, _timestamp: object) -> SimpleNamespace:
        return self.vintage


def _staged_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    decision = pd.Timestamp("2026-08-02T01:00:00Z")
    router = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "__decision_ts__": [decision] * 4,
        "side_name": ["long"] * 4,
        "router_feature": [4.0, 3.0, 2.0, 1.0],
    })
    routed = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": [decision] * 2,
        "__symbol__": ["A/USD:USD", "B/USD:USD"],
        "side_name": ["long"] * 2,
        "base_feature": [2.0, 1.0],
        "under_feature": [1.0, 2.0],
    })
    return router, routed


def test_staged_scoring_only_materialises_exact_router50_and_scores_router_once() -> None:
    preproduction = _StagedPreproduction()
    models = _StagedModels()
    stack = P8USealedInferenceStack(
        preproduction=preproduction, models=models, mc1_selector=_StagedSelector()
    )
    router, routed = _staged_inputs()
    _boundary, _inputs, router_scores, routed_population = stack.route_staged(router)
    assert models.router_calls == 1
    assert routed_population.loc[routed_population["router50_eligible"], "candidate_id"].tolist() == ["a", "b"]

    decision = stack.score_staged(
        router_features=router, routed_features=routed, router_scores=router_scores
    )
    assert models.router_calls == 1, "staged handoff must not score Router twice"
    assert [set(frame["candidate_id"]) for frame in models.base_inputs] == [{"a", "b"}]
    assert [set(frame["candidate_id"]) for frame in models.under_inputs] == [{"a", "b"}]
    assert len(decision.router_population) == 4
    assert set(decision.router_population.loc[decision.router_population["router50_eligible"], "candidate_id"]) == {"a", "b"}
    assert set(decision.routed_scores["candidate_id"]) == {"a", "b"}
    assert decision.routed_scores["dual_mc1_admitted"].all()


def test_staged_coordinates_retain_exact_router50_symbol_identities() -> None:
    preproduction = _StagedPreproduction()
    models = _StagedModels()
    stack = P8USealedInferenceStack(
        preproduction=preproduction, models=models, mc1_selector=_StagedSelector()
    )
    router, routed = _staged_inputs()
    _boundary, _inputs, router_scores, _population = stack.route_staged(router)
    coordinates = stack.score_staged_coordinates(
        router_features=router, routed_features=routed, router_scores=router_scores,
    )
    assert set(coordinates.current_coordinates["candidate_id"]) == {"a", "b"}
    assert coordinates.current_coordinates["__symbol__"].tolist() == ["A/USD:USD", "B/USD:USD"]
    assert np.allclose(coordinates.bcf_coordinates["final_score"], [.75, .25])


def test_c0_mapping_reuses_precomputed_coordinates_without_rescoring_models() -> None:
    preproduction = _StagedPreproduction()
    models = _StagedModels()
    stack = P8USealedInferenceStack(
        preproduction=preproduction, models=models, mc1_selector=_StagedSelector()
    )
    router, routed = _staged_inputs()
    _boundary, _inputs, router_scores, _population = stack.route_staged(router)
    coordinates = stack.score_staged_coordinates(
        router_features=router, routed_features=routed, router_scores=router_scores,
    )
    base_calls, under_calls = len(models.base_inputs), len(models.under_inputs)
    c0 = stack.map_c0_coordinates(coordinates)
    assert len(models.base_inputs) == base_calls
    assert len(models.under_inputs) == under_calls
    assert c0["dual_mc1_admitted"].all()
    assert c0["__symbol__"].tolist() == ["A/USD:USD", "B/USD:USD"]


def test_staged_scoring_rejects_any_base_under_identity_outside_router50() -> None:
    preproduction = _StagedPreproduction()
    models = _StagedModels()
    stack = P8USealedInferenceStack(
        preproduction=preproduction, models=models, mc1_selector=_StagedSelector()
    )
    router, routed = _staged_inputs()
    invalid = pd.concat(
        [
            routed,
            pd.DataFrame({
                "candidate_id": ["c"],
                "__decision_ts__": [router["__decision_ts__"].iloc[0]],
                "side_name": ["long"],
                "base_feature": [0.0],
                "under_feature": [0.0],
            }),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="does not equal Router50"):
        stack.score_staged(router_features=router, routed_features=invalid)


def test_staged_handoff_retains_shared_router_base_feature_without_merge_suffix() -> None:
    """A Router/Base shared physical field must reach downstream unchanged."""

    decision = pd.Timestamp("2026-08-02T01:00:00Z")
    routed_population = pd.DataFrame({
        "candidate_id": ["a"],
        "__decision_ts__": [decision],
        "side_name": ["long"],
        "router_score": [.9],
        "router50_eligible": [True],
        "router_fraction": [.5],
        "router_timestamp_ordinal": [1],
        "router_timestamp_count": [2],
        "shared_feature": [10.0],
    })
    staged_features = pd.DataFrame({
        "candidate_id": ["a"],
        "__decision_ts__": [decision],
        "side_name": ["long"],
        "shared_feature": [11.0],
        "base_only_feature": [12.0],
    })
    result = P8USealedInferenceStack._exact_routed_matrix(
        routed_population, staged_features
    )
    assert "shared_feature" in result.columns
    assert not any(name.endswith(("_x", "_y")) for name in result.columns)
    assert result.loc[0, "shared_feature"] == 11.0
