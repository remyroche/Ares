from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.regime_ev_calibration import apply_regime_ev_calibration
from extreme_price_movements.supervised_market_state_calibration import (
    expected_ev_rank,
    fit_hierarchical_ev_calibrator,
    predict_hierarchical_ev,
)


def test_live_mapping_emits_explicit_side_archetype_ev() -> None:
    frame = pd.DataFrame(
        {
            "score": [0.8, 0.8],
            "side_name": ["long", "short"],
            "archetype_policy_key": ["state", "state"],
        }
    )
    artifact = {
        "source_score_col": "score",
        "adjusted_score_col": "adjusted",
        "expected_ev_mapping": {
            "global": {"x": [0.0, 1.0], "y": [0.0, 0.0]},
            "local": {
                "long||state": {
                    "x": [0.0, 1.0],
                    "y": [0.02, 0.02],
                    "weight": 1.0,
                    "support": 500,
                },
                "short||state": {
                    "x": [0.0, 1.0],
                    "y": [-0.01, -0.01],
                    "weight": 1.0,
                    "support": 600,
                },
            },
            "rank_reference": [-0.01, 0.02],
            "rank_blend": 1.0,
            "require_side_archetype_curve": True,
        },
    }

    scored = apply_regime_ev_calibration(frame, artifact)

    np.testing.assert_allclose(
        scored["expected_net_ev_after_1pct_side_archetype"], [0.02, -0.01]
    )
    assert scored["expected_ev_side_archetype_curve_applied"].all()
    assert scored["expected_ev_mapping_scope"].eq("side_x_archetype").all()
    assert scored["expected_ev_side_archetype_support"].tolist() == [500, 600]


def test_live_mapping_applies_monotonic_plateau_refinement() -> None:
    frame = pd.DataFrame(
        {
            "score": [0.2, 0.8],
            "side_name": ["long", "long"],
            "archetype_policy_key": ["state", "state"],
        }
    )
    artifact = {
        "source_score_col": "score",
        "adjusted_score_col": "adjusted",
        "expected_ev_mapping": {
            "global": {"x": [0.0, 1.0], "y": [0.01, 0.01]},
            "local": {},
            "rank_reference": [0.0099, 0.0101],
            "rank_blend": 1.0,
            "monotonic_refinement": {
                "enabled": True,
                "slope": 0.0002,
                "score_min": 0.0,
                "score_max": 1.0,
                "centering": 0.5,
            },
        },
    }

    scored = apply_regime_ev_calibration(frame, artifact)

    expected = scored["expected_net_ev_after_1pct"].to_numpy()
    assert expected[1] > expected[0]
    np.testing.assert_allclose(expected, [0.00994, 0.01006], atol=1e-7)


def test_rank_reference_uses_hierarchical_expected_ev_not_global_only() -> None:
    n = 800
    frame = pd.DataFrame(
        {
            "side_name": ["long"] * (n // 2) + ["short"] * (n // 2),
            "archetype_policy_key": ["state"] * n,
        }
    )
    score = np.tile(np.linspace(0.0, 1.0, n // 2), 2)
    realized = np.concatenate(
        [0.03 + 0.01 * score[: n // 2], -0.02 + 0.01 * score[n // 2 :]]
    )
    calibrator = fit_hierarchical_ev_calibrator(
        frame,
        score,
        realized,
        shrink_rows=1.0,
        min_local_rows=100,
        local_weight_cap=1.0,
        rank_blend=1.0,
    )
    mapped = predict_hierarchical_ev(calibrator, frame, score)
    rank = expected_ev_rank(calibrator, mapped, score)

    assert np.nanmedian(rank[: n // 2]) > np.nanmedian(rank[n // 2 :])
    assert np.nanmin(rank[: n // 2]) > np.nanmax(rank[n // 2 :])


def test_blacklist_updates_all_expected_ev_aliases() -> None:
    frame = pd.DataFrame(
        {
            "score": [0.8],
            "side_name": ["long"],
            "archetype_policy_key": ["long_dirtyavoid_sparse_questionable"],
        }
    )
    artifact = {
        "source_score_col": "score",
        "adjusted_score_col": "adjusted",
        "blacklisted_side_archetypes": [
            "long||long_dirtyavoid_sparse_questionable"
        ],
        "expected_ev_mapping": {
            "global": {"x": [0.0, 1.0], "y": [0.01, 0.02]},
            "local": {},
            "rank_reference": [0.01, 0.02],
            "rank_blend": 1.0,
        },
    }

    scored = apply_regime_ev_calibration(frame, artifact)

    assert bool(scored.loc[0, "regime_ev_blacklisted"])
    for column in (
        "expected_net_ev_after_1pct",
        "expected_net_ev_after_1pct_side_archetype",
        "market_state_mlp_expected_net_ev_after_1pct",
    ):
        assert scored.loc[0, column] == -1.0
    assert scored.loc[0, "expected_ev_rank_score"] == 0.0
    assert scored.loc[0, "market_state_mlp_expected_ev_rank_score"] == 0.0
