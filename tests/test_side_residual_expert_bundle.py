from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.inference.side_residual_expert import (
    SideResidualExpertBundle,
)


class _Map:
    def __init__(self):
        self.global_model = self
        self.local_models = {}
        self.local_weights = {}
        self.rank_reference = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
        self.rank_blend = 1.0
        self.monotonic_refinement_slope = 0.0
        self.refinement_score_min = 0.0
        self.refinement_score_max = 1.0

    def predict(self, values):
        return np.asarray(values)


class _Model:
    def predict(self, values):
        return values[:, 1]


def _bundle() -> SideResidualExpertBundle:
    return SideResidualExpertBundle(
        {
            "schema": "side_base_residual_expert_inference_v2",
            "backbone_score_col": "score",
            "feature_contract": {
                "long": ["score", "long_context"],
                "short": ["score", "short_context"],
            },
            "alpha_by_side": {"long": 0.5, "short": 0.25},
            "baseline_ev_map": _Map(),
            "corrected_ev_map": _Map(),
            "residual_models": {"long": _Model(), "short": _Model()},
            "model_params_by_side": {"long": {}, "short": {}},
            "round_trip_cost": 0.01,
        }
    )


def test_side_residual_expert_scores_only_complete_rows():
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short", "long"],
            "archetype_policy_key": ["long__mixed", "default", "mixed"],
            "score": [0.2, 0.4, 0.6],
            "long_context": [0.1, 9.0, np.nan],
            "short_context": [9.0, -0.2, 9.0],
        }
    )
    result = _bundle().transform(frame)
    assert result["meta_residual_expert_complete_case"].tolist() == [True, True, False]
    np.testing.assert_allclose(
        result.loc[:1, "score_base_ev_residual_expert"], [0.25, 0.35]
    )
    assert np.isnan(result.loc[2, "score_base_ev_residual_expert"])


def test_side_residual_expert_preserves_duplicate_batch_indexes():
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short", "long"],
            "archetype_policy_key": ["mixed", "default", "mixed"],
            "score": [0.2, 0.4, 0.6],
            "long_context": [0.1, 9.0, 0.2],
            "short_context": [9.0, -0.2, 9.0],
        },
        index=[0, 0, 0],
    )

    result = _bundle().transform(frame)

    assert len(result) == len(frame)
    np.testing.assert_allclose(
        result["score_base_ev_residual_expert"].to_numpy(),
        [0.25, 0.35, 0.70],
    )


def test_side_residual_expert_requires_only_observed_side_contract():
    long_frame = pd.DataFrame(
        {
            "side_name": ["long"],
            "archetype_policy_key": ["mixed"],
            "score": [0.2],
            "long_context": [0.1],
        }
    )
    short_frame = pd.DataFrame(
        {
            "side_name": ["short"],
            "archetype_policy_key": ["default"],
            "score": [0.4],
            "short_context": [-0.2],
        }
    )

    assert _bundle().transform(long_frame)[
        "meta_residual_expert_complete_case"
    ].tolist() == [True]
    assert _bundle().transform(short_frame)[
        "meta_residual_expert_complete_case"
    ].tolist() == [True]


def test_side_residual_expert_requires_both_side_contracts():
    bundle = _bundle()
    payload = dict(bundle.payload)
    payload["feature_contract"] = {"long": ["score"]}
    try:
        SideResidualExpertBundle(payload).validate_contract()
    except ValueError as exc:
        assert "side-local feature_contract" in str(exc)
    else:
        raise AssertionError("missing short contract should fail validation")


def test_side_routed_residual_expert_uses_each_frozen_route() -> None:
    long_payload = dict(_bundle().payload)
    short_payload = dict(_bundle().payload)
    long_payload["alpha_by_side"] = {"long": 0.5, "short": 0.5}
    short_payload["alpha_by_side"] = {"long": 0.25, "short": 0.25}
    routed = SideResidualExpertBundle(
        {
            "schema": "side_routed_side_residual_expert_v1",
            "routes": {"long": long_payload, "short": short_payload},
        }
    )
    routed.validate_contract()
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "archetype_policy_key": ["mixed", "default"],
            "score": [0.2, 0.4],
            "long_context": [0.1, 9.0],
            "short_context": [9.0, -0.2],
        }
    )

    result = routed.transform(frame)

    assert result["meta_residual_expert_complete_case"].tolist() == [True, True]
    np.testing.assert_allclose(
        result["score_base_ev_residual_expert"].to_numpy(),
        [0.25, 0.35],
    )
    assert routed.feature_contract("long") == ["score", "long_context"]
    assert routed.feature_contract("short") == ["score", "short_context"]
