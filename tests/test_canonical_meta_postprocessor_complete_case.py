from types import SimpleNamespace

import numpy as np
import pandas as pd

from extreme_price_movements.inference.canonical_meta_postprocessor import (
    CanonicalMetaPostprocessor,
    V9TailPostprocessor,
    V9_TAIL_HIER_EV_POLICY_ID,
    _add_post_v9_reliability_features,
)


def test_complete_case_report_vectorized_contract_preserves_missing_names():
    predecessor = SimpleNamespace(
        raw_selected_features=["observable_a"],
        residual_representation_state=None,
        shock_overlay_state=None,
        residual_recognizer=None,
    )
    residual_state = SimpleNamespace(
        local_models={}, side_models={}, market_model=None, config=SimpleNamespace()
    )
    postprocessor = CanonicalMetaPostprocessor(predecessor, residual_state, {"effects": []})
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "archetype_policy_key": ["a", "b"],
            "observable_a": [1.0, np.nan],
            "score": [0.5, 0.6],
            "score_base": [0.4, 0.4],
            "score_meta_base_soft_label": [0.7, 0.8],
        }
    )

    report = postprocessor.complete_case_report(frame)

    assert report["complete_case"].tolist() == [True, False]
    assert report.loc[0, "missing_feature_count"] == 0
    assert report.loc[1, "missing_feature_count"] == 1
    assert report.loc[1, "missing_features"] == "observable_a"


def test_post_v9_reliability_features_match_training_definition():
    rank = pd.Series([0.2, 0.9], dtype=np.float32)
    out = _add_post_v9_reliability_features(pd.DataFrame(index=rank.index), rank)

    np.testing.assert_allclose(out["hit_probability"], rank)
    np.testing.assert_allclose(out["policy_parent_rank"], rank)
    np.testing.assert_allclose(
        out["meta_parent_rank_uncertainty_p1mp"], [0.16, 0.09], atol=1e-7
    )
    np.testing.assert_allclose(
        out["meta_hit_probability_uncertainty_p1mp"], [0.16, 0.09], atol=1e-7
    )
    np.testing.assert_allclose(
        out["meta_parent_rank_margin_top10"], [-0.7, 0.0], atol=1e-7
    )


def test_complete_case_does_not_require_post_v9_derived_features():
    predecessor = SimpleNamespace(
        raw_selected_features=["observable_a"],
        residual_representation_state=None,
        shock_overlay_state=None,
        residual_recognizer=None,
    )
    residual_model = SimpleNamespace(
        feature_columns=["meta_parent_rank_uncertainty_p1mp"]
    )
    residual_state = SimpleNamespace(
        local_models={"long|a": residual_model},
        side_models={},
        market_model=None,
        config=SimpleNamespace(allow_side_fallback=False),
    )
    postprocessor = CanonicalMetaPostprocessor(
        predecessor,
        residual_state,
        {
            "effects": [
                {
                    "side_name": "long",
                    "archetype_policy_key": "a",
                    "feature_cols": ["meta_hit_probability_local_top10_margin"],
                }
            ]
        },
    )
    frame = pd.DataFrame(
        {
            "side_name": ["long"],
            "archetype_policy_key": ["a"],
            "observable_a": [1.0],
            "score": [0.5],
            "score_base": [0.4],
            "score_meta_base_soft_label": [0.7],
        }
    )

    assert postprocessor.complete_case_report(frame)["complete_case"].tolist() == [
        True
    ]


def test_v9_only_keeps_hierarchical_ev_mapping_without_mlp_effects():
    class Predecessor:
        def predict(self, frame):
            return pd.DataFrame(
                {"historical_rank": frame["rank_input"].to_numpy()},
                index=frame.index,
            )

    class ResidualState:
        local_models = {}
        market_model = None

        def transform_oos(self, frame):
            return pd.DataFrame(index=frame.index)

    artifact = {
        "effects": [
            {
                "side_name": "long",
                "archetype_policy_key": "clean",
                "feature_col": "rank_input",
                "shape": "linear",
                "params": {"slope": 0.5},
            }
        ],
        "expected_ev_mapping": {
            "global": {"x": [0.0, 1.0], "y": [-0.01, 0.02]},
            "local": {
                "long||clean": {
                    "x": [0.0, 1.0],
                    "y": [-0.02, 0.04],
                    "support": 1000,
                    "weight": 1.0,
                }
            },
        },
    }
    postprocessor = V9TailPostprocessor(Predecessor(), ResidualState(), artifact)
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "archetype_policy_key": ["clean", "other"],
            "rank_input": [0.5, 0.5],
        }
    )

    scored = postprocessor.transform(frame)

    np.testing.assert_allclose(scored["score_regime_calibrated"], [0.5, 0.5])
    np.testing.assert_allclose(
        scored["expected_net_ev_after_1pct"], [0.01, 0.005], atol=1e-8
    )
    assert scored["expected_ev_mapping_scope"].tolist() == [
        "side_x_archetype",
        "global_fallback",
    ]
    assert scored["market_state_mlp_score_correction"].eq(0.0).all()
    assert scored["meta_postprocessor_policy_id"].eq(
        V9_TAIL_HIER_EV_POLICY_ID
    ).all()
