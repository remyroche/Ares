import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.residual_leaf_state_discovery import (
    ResidualLeafConfig,
    failure_target,
    feature_cluster_composites,
    fit_shallow_classifier,
    fit_time_leaf_clusters,
    leaf_feature_clusters,
    observable_feature_names,
    stable_binned_mi_screen,
)


def test_causal_rolling_summary_does_not_use_future_rows() -> None:
    from extreme_price_movements.residual_leaf_state_discovery import (
        causal_rolling_summary_features,
    )

    original = pd.DataFrame({"signal": np.arange(12, dtype=np.float32)})
    baseline = causal_rolling_summary_features(original, ["signal"], window=4)
    changed = original.copy()
    changed.loc[8:, "signal"] = 10_000.0
    perturbed = causal_rolling_summary_features(changed, ["signal"], window=4)
    pd.testing.assert_frame_equal(baseline.iloc[:8], perturbed.iloc[:8])
from scripts.run_residual_calendar_leaf_state_discovery import (
    _attach_calendar_target,
)


def test_leaf_state_outputs_are_observable_and_frozen():
    rng = np.random.default_rng(7)
    rows = 500
    frame = pd.DataFrame(
        {
            "market_shock": rng.normal(size=rows).astype(np.float32),
            "breadth": rng.normal(size=rows).astype(np.float32),
            "target_signature_arch__long_x_negative_persistence_prev7d": rng.random(rows),
        }
    )
    config = ResidualLeafConfig(n_estimators=20, feature_cluster_count=3, time_cluster_count=3)
    features = observable_feature_names(frame, config)
    assert features == ["market_shock", "breadth"]
    target = frame["market_shock"].clip(lower=0)
    y, weights, _ = failure_target(target, 0.80)
    x = frame[features].to_numpy(dtype=np.float32)
    model = fit_shallow_classifier(
        x,
        y,
        weights,
        {
            "max_depth": 2,
            "num_leaves": 4,
            "min_child_samples": 10,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "min_split_gain": 0.0,
            "n_estimators": 20,
        },
        7,
    )
    leaves = model.predict(x, pred_leaf=True).astype(np.int16)
    table, mapping = leaf_feature_clusters(model, features, leaves, target.to_numpy(), config)
    composites = feature_cluster_composites(
        leaves, mapping, int(table["feature_cluster"].max()) + 1
    )
    assert composites.shape[0] == rows
    assert np.isfinite(composites).all()
    time_model = fit_time_leaf_clusters(leaves, target.to_numpy(), config)
    probability, state, risk = time_model.transform(leaves[:25])
    assert probability.shape[0] == 25
    assert 2 <= probability.shape[1] <= 3
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-5)
    assert state.shape == risk.shape == (25,)


def test_leaf_paths_support_constant_trees():
    rows = 120
    frame = pd.DataFrame({"constant": np.ones(rows, dtype=np.float32)})
    y = np.r_[np.zeros(rows // 2), np.ones(rows // 2)].astype(np.int8)
    model = fit_shallow_classifier(
        frame.to_numpy(),
        y,
        np.ones(rows, dtype=np.float32),
        {
            "max_depth": 2,
            "num_leaves": 4,
            "min_child_samples": 100,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "min_split_gain": 1.0,
            "n_estimators": 3,
        },
        3,
    )
    leaves = model.predict(frame.to_numpy(), pred_leaf=True).astype(np.int16)
    table, mapping = leaf_feature_clusters(
        model,
        ["constant"],
        leaves,
        y.astype(np.float32),
        ResidualLeafConfig(n_estimators=3, feature_cluster_count=2),
    )
    tree_count = 1 if leaves.ndim == 1 else leaves.shape[1]
    assert len(table) == tree_count
    assert all((tree, 0) in mapping for tree in range(tree_count))
    time_model = fit_time_leaf_clusters(
        leaves, y.astype(np.float32), ResidualLeafConfig(time_cluster_count=3)
    )
    probability, state, risk = time_model.transform(leaves[:10])
    assert probability.shape == (10, 1)
    assert np.all(state == 0)
    assert np.isfinite(risk).all()


def test_calendar_target_marks_only_declared_high_ac_days():
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T12:00:00Z", "2026-01-02T00:00:00Z"]
            ),
            "side_name": ["long"] * 3,
            "archetype_policy_key": ["breakout"] * 3,
        }
    )
    events = pd.DataFrame(
        {
            "day": pd.to_datetime(["2026-01-01"], utc=True),
            "event_severity": [2.5],
        }
    )
    result = _attach_calendar_target(frame, events)
    assert result["__calendar_event"].tolist() == [1, 1, 0]
    assert result["__calendar_event_severity"].tolist() == [2.5, 2.5, 0.0]


def test_stable_binned_mi_screen_prefers_persistent_nonlinear_signal():
    rng = np.random.default_rng(19)
    rows = 900
    signal = rng.normal(size=rows)
    label = (np.abs(signal) > 1.2).astype(np.int8)
    frame = pd.DataFrame(
        {
            "nonlinear_signal": signal,
            "noise_1": rng.normal(size=rows),
            "noise_2": rng.normal(size=rows),
        }
    )
    selected, report = stable_binned_mi_screen(
        frame, label, list(frame), max_features=1
    )
    assert selected == ["nonlinear_signal"]
    assert report.iloc[0]["mi_min"] > 0
