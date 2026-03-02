import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from extreme_price_movements.feature_selection_extreme_events import mdi_feature_selection_v3


def _synthetic_frame(n=420, seed=42):
    rng = np.random.RandomState(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = x1 * x2 + 0.1 * rng.normal(size=n)
    x_noise = rng.normal(size=(n, 6))
    y = 0.8 * x1 + 0.4 * x3 + 0.1 * rng.normal(size=n)
    X = pd.DataFrame(
        np.column_stack([x1, x2, x3, x_noise]),
        columns=["x1", "x2", "x3", "n1", "n2", "n3", "n4", "n5", "n6"],
    )
    return X, y


def test_selector_v3_regression_defaults_and_scores():
    X, y = _synthetic_frame()
    base = ExtraTreesRegressor(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=12,
        random_state=1,
        n_jobs=1,
    )
    res = mdi_feature_selection_v3(
        X,
        y,
        base_model=base,
        n_splits=3,
        purge=1,
        end_features=8,
        min_features=5,
        selector_target="regression",
        selector_head_name="test_reg",
        selector_emit_report=False,
    )
    assert len(res.selected_features) >= 5
    assert res.summary is not None
    assert res.summary.get("selector_top_metric") == "ic_top"
    assert "top30_support" in res.metrics_table.columns
    assert "global_importance" in res.metrics_table.columns
    assert "stability_score" in res.metrics_table.columns
    assert "frequency_score" in res.metrics_table.columns
    assert "interaction_support" in res.metrics_table.columns
    # Ensure top30 attribution is not a direct clone of global importance.
    assert not np.allclose(
        res.metrics_table["top30_support"].to_numpy(float),
        res.metrics_table["global_importance"].to_numpy(float),
    )


def test_selector_v3_interaction_and_hysteresis():
    X, y = _synthetic_frame(seed=7)
    base = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=10,
        random_state=7,
        n_jobs=1,
    )
    prev = ["x1", "x2", "n1", "n2", "n3", "n4"]
    res = mdi_feature_selection_v3(
        X,
        y,
        base_model=base,
        n_splits=3,
        purge=1,
        end_features=8,
        min_features=6,
        selector_target="regression",
        selector_head_name="test_hyst",
        selector_prev_selected=prev,
        selector_min_overlap=0.70,
        selector_hysteresis_margin=0.05,
        selector_interaction_mode="tree_path_lift",
        selector_emit_report=False,
    )
    assert res.interaction_table is not None
    assert {"feature_a", "feature_b", "lift", "final_pair_score"}.issubset(set(res.interaction_table.columns))
    assert res.summary is not None
    assert float(res.summary.get("overlap_after", 0.0)) >= 0.60


def test_selector_v3_interaction_off_works():
    X, y = _synthetic_frame(seed=9)
    base = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=5,
        min_samples_leaf=10,
        random_state=9,
        n_jobs=1,
    )
    res = mdi_feature_selection_v3(
        X,
        y,
        base_model=base,
        n_splits=3,
        purge=1,
        end_features=7,
        min_features=5,
        selector_target="regression",
        selector_interaction_mode="off",
        selector_head_name="test_off",
        selector_emit_report=False,
    )
    assert len(res.selected_features) >= 5
    assert res.interaction_table is not None


def test_training_no_duration_metadata_export():
    training_path = "/Users/remyroche/Documents/Ares/extreme_price_movements/training.py"
    with open(training_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "__duration__" not in src
