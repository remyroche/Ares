from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_meta_residual_extreme_local_champion_overlay import (
    FEATURES,
    _adjust_rank,
    _composite,
    _fit_references,
    _local_tail_diagnostics,
)


def test_extreme_overlay_only_changes_parent_top20_band() -> None:
    parent = np.asarray([0.50, 0.79, 0.85, 0.91, 0.99], dtype=np.float32)
    adverse = np.asarray([1.0, 1.0, 0.0, 1.0, 0.5], dtype=np.float32)
    positive = np.asarray([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    adjusted = _adjust_rank(
        parent,
        adverse,
        positive,
        threshold=0.95,
        alpha_down=0.03,
        alpha_up=0.02,
    )

    np.testing.assert_allclose(adjusted[:2], parent[:2])
    assert adjusted[2] > parent[2]
    assert adjusted[3] < parent[3]
    assert adjusted[4] == parent[4]


def test_downside_only_overlay_cannot_add_trades() -> None:
    parent = np.asarray([0.85, 0.89, 0.91, 0.95], dtype=np.float32)
    adjusted = _adjust_rank(
        parent,
        adverse=np.ones(4, dtype=np.float32),
        positive=np.ones(4, dtype=np.float32),
        threshold=0.95,
        alpha_down=0.03,
        alpha_up=0.0,
    )
    assert int((adjusted >= 0.90).sum()) <= int((parent >= 0.90).sum())


def test_feature_basket_is_local_state_only() -> None:
    assert FEATURES
    assert all(name.startswith("resid_event_aegmm_") for name in FEATURES)
    assert not any("market" in name for name in FEATURES)


def test_tail_thresholds_are_fitted_per_side_archetype() -> None:
    feature = FEATURES[0]
    rows = 300
    train = pd.DataFrame(
        {
            "side_name": ["long"] * rows + ["short"] * rows,
            "archetype_policy_key": ["a"] * rows + ["b"] * rows,
            feature: np.concatenate(
                [
                    np.linspace(0.0, 1.0, rows, dtype=np.float32),
                    np.linspace(100.0, 101.0, rows, dtype=np.float32),
                ]
            ),
        }
    )
    catalog = pd.DataFrame(
        [
            {
                "side_name": "long",
                "archetype_policy_key": "a",
                "event": "adverse",
                "feature": feature,
                "direction": 1.0,
                "mi": 1.0,
            },
            {
                "side_name": "short",
                "archetype_policy_key": "b",
                "event": "adverse",
                "feature": feature,
                "direction": 1.0,
                "mi": 1.0,
            },
        ]
    )
    report = _local_tail_diagnostics(
        train,
        catalog,
        {"top_feature_count": 1, "threshold": 0.96},
    )

    assert report["scope"].eq("side_x_archetype").all()
    cutoffs = report.set_index("archetype_policy_key")["local_tail_cutoff"]
    assert cutoffs["a"] < 2.0
    assert cutoffs["b"] > 99.0
    np.testing.assert_allclose(report["local_tail_fraction"], 0.04, atol=0.005)


def test_zero_point_mass_is_not_promoted_to_extreme_tail() -> None:
    feature = FEATURES[0]
    train = pd.DataFrame(
        {
            "side_name": ["short"] * 300,
            "archetype_policy_key": ["mixed"] * 300,
            feature: np.concatenate(
                [np.zeros(240, dtype=np.float32), np.linspace(1.0, 2.0, 60)]
            ),
        }
    )
    catalog = pd.DataFrame(
        [
            {
                "side_name": "short",
                "archetype_policy_key": "mixed",
                "event": "adverse",
                "feature": feature,
                "direction": -1.0,
                "mi": 1.0,
            }
        ]
    )
    references = _fit_references(train, catalog, 1)
    composite = _composite(train, references, "adverse")

    # Direction=-1 makes zero the maximum value, but its 80% point mass gets
    # a midrank of 60%, not a right-tie percentile of 100%.
    assert not np.any(composite >= 0.95)
