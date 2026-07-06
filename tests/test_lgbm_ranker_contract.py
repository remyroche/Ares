from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import extreme_price_movements.lgbm_pipeline as lgbm_pipeline  # noqa: E402
from extreme_price_movements.lgbm_pipeline import (  # noqa: E402
    _fit_lgbm_model,
    _lgbm_ranker_absolute_relevance,
    _lgbm_ranker_absolute_relevance_mode,
    _lgbm_ranker_group_keys,
    _lgbm_ranker_group_order,
    _lgbm_ranker_relevance,
    _lgbm_ranker_relevance_source,
)


def test_lgbm_ranker_timestamp_side_group_keys_separate_long_short() -> None:
    frame = pd.DataFrame({"side": [1, -1, "long", "short"]})
    timestamps = pd.to_datetime(
        [
            "2026-06-01 00:00:00Z",
            "2026-06-01 00:00:00Z",
            "2026-06-01 01:00:00Z",
            "2026-06-01 01:00:00Z",
        ]
    )

    keys = _lgbm_ranker_group_keys(frame, timestamps, mode="timestamp_side")

    assert keys is not None
    assert len(set(keys.astype(str))) == 4
    assert str(keys[0]).endswith("|long")
    assert str(keys[1]).endswith("|short")


def test_lgbm_ranker_group_order_returns_contiguous_group_counts() -> None:
    keys = np.asarray(["b", "a", "b", "a", "c"], dtype=object)

    order, group = _lgbm_ranker_group_order(keys)

    assert keys[order].astype(str).tolist() == ["a", "a", "b", "b", "c"]
    assert group.tolist() == [2, 2, 1]


def test_lgbm_ranker_relevance_is_local_to_group() -> None:
    keys = np.asarray(["t0", "t0", "t0", "t1", "t1", "t1"], dtype=object)
    y = np.asarray([0.1, 0.3, 0.2, 10.0, 30.0, 20.0], dtype=np.float32)

    relevance = _lgbm_ranker_relevance(y, keys, bins=5)

    assert relevance[:3].tolist() == [1, 4, 3]
    assert relevance[3:].tolist() == [1, 4, 3]


def test_s52_ranker_relevance_demotes_dirty_positive_high_mfe() -> None:
    y = np.asarray([0.20, 0.90, 0.10], dtype=np.float32)
    label_context = {
        "__first_touch_capture_net__": np.asarray([0.01, 0.05, -0.02], dtype=np.float32),
        "__first_touch_round_trip_cost__": np.asarray([0.01, 0.01, 0.01], dtype=np.float32),
        "__first_touch_hit__": np.asarray([1, 1, 0], dtype=np.float32),
        "__first_touch_stop__": np.asarray([0, 0, 1], dtype=np.float32),
        "__first_touch_timeout__": np.asarray([0, 0, 0], dtype=np.float32),
        "__first_touch_valid_path__": np.asarray([1, 1, 1], dtype=np.float32),
        "__first_touch_same_bar_both__": np.asarray([0, 0, 0], dtype=np.float32),
        "__first_touch_mae_norm__": np.asarray([0.20, 1.60, 2.00], dtype=np.float32),
        "__first_touch_mfe_norm__": np.asarray([1.10, 4.00, 0.20], dtype=np.float32),
        "__mfe_1r_before_mae_1r__": np.asarray([1, 0, 0], dtype=np.float32),
        "__mae_1r_before_mfe_1r__": np.asarray([0, 1, 1], dtype=np.float32),
    }

    relevance = _lgbm_ranker_relevance_source(y, label_context=label_context, mode="s52_path_order")

    assert relevance is not None
    assert relevance[0] > relevance[1]
    assert relevance[1] > relevance[2]


def test_s52_path_order_relevance_ignores_post_exit_full_path_mae() -> None:
    y = np.asarray([0.50, 0.50], dtype=np.float32)
    label_context = {
        "__first_touch_capture_net__": np.asarray([0.02, 0.025], dtype=np.float32),
        "__first_touch_round_trip_cost__": np.asarray([0.01, 0.01], dtype=np.float32),
        "__first_touch_hit__": np.asarray([1, 1], dtype=np.float32),
        "__first_touch_stop__": np.asarray([0, 0], dtype=np.float32),
        "__first_touch_timeout__": np.asarray([0, 0], dtype=np.float32),
        "__first_touch_valid_path__": np.asarray([1, 1], dtype=np.float32),
        "__first_touch_same_bar_both__": np.asarray([0, 0], dtype=np.float32),
        "__first_touch_mae_norm__": np.asarray([0.25, 0.25], dtype=np.float32),
        "__first_touch_full_path_mae_norm__": np.asarray([0.60, 3.00], dtype=np.float32),
        "__first_touch_mfe_norm__": np.asarray([1.20, 1.20], dtype=np.float32),
        "__mfe_1r_before_mae_1r__": np.asarray([1, 1], dtype=np.float32),
        "__mae_1r_before_mfe_1r__": np.asarray([0, 0], dtype=np.float32),
    }

    relevance = _lgbm_ranker_relevance_source(y, label_context=label_context, mode="s52_path_order")

    assert relevance is not None
    assert relevance[1] > relevance[0]


def test_s52_path_order_relevance_demotes_slow_underwater_path() -> None:
    y = np.asarray([0.50, 0.50], dtype=np.float32)
    label_context = {
        "__first_touch_capture_net__": np.asarray([0.02, 0.02], dtype=np.float32),
        "__first_touch_round_trip_cost__": np.asarray([0.01, 0.01], dtype=np.float32),
        "__first_touch_hit__": np.asarray([1, 1], dtype=np.float32),
        "__first_touch_stop__": np.asarray([0, 0], dtype=np.float32),
        "__first_touch_timeout__": np.asarray([0, 0], dtype=np.float32),
        "__first_touch_valid_path__": np.asarray([1, 1], dtype=np.float32),
        "__first_touch_same_bar_both__": np.asarray([0, 0], dtype=np.float32),
        "__first_touch_mae_norm__": np.asarray([0.25, 0.25], dtype=np.float32),
        "__first_touch_full_path_mae_norm__": np.asarray([0.60, 0.60], dtype=np.float32),
        "__first_touch_mfe_norm__": np.asarray([1.20, 1.20], dtype=np.float32),
        "__mfe_1r_before_mae_1r__": np.asarray([1, 1], dtype=np.float32),
        "__mae_1r_before_mfe_1r__": np.asarray([0, 0], dtype=np.float32),
        "__max_adverse_before_mfe_1r__": np.asarray([0.50, 1.80], dtype=np.float32),
        "__underwater_bars_before_mfe_1r__": np.asarray([4.0, 18.0], dtype=np.float32),
        "__underwater_fraction_before_mfe_1r__": np.asarray([0.20, 0.55], dtype=np.float32),
    }

    relevance = _lgbm_ranker_relevance_source(y, label_context=label_context, mode="s52_path_order")

    assert relevance is not None
    assert relevance[0] > relevance[1]


def test_s52_soft_ordered_ev_preserves_near_clean_path_breadth() -> None:
    y = np.asarray([0.70, 0.90, 0.75, 0.20], dtype=np.float32)
    label_context = {
        "__first_touch_capture_net__": np.asarray([0.006, 0.050, 0.004, -0.002], dtype=np.float32),
        "__first_touch_round_trip_cost__": np.asarray([0.01, 0.01, 0.01, 0.01], dtype=np.float32),
        "__first_touch_hit__": np.asarray([1, 1, 1, 0], dtype=np.float32),
        "__first_touch_stop__": np.asarray([0, 0, 0, 1], dtype=np.float32),
        "__first_touch_timeout__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_valid_path__": np.asarray([1, 1, 1, 1], dtype=np.float32),
        "__first_touch_same_bar_both__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_mae_norm__": np.asarray([0.30, 1.50, 0.40, 0.20], dtype=np.float32),
        "__first_touch_mfe_norm__": np.asarray([1.20, 4.00, 1.20, 0.20], dtype=np.float32),
        "__mfe_1r_before_mae_1r__": np.asarray([1, 0, 1, 0], dtype=np.float32),
        "__mae_1r_before_mfe_1r__": np.asarray([0, 1, 0, 0], dtype=np.float32),
        "__max_adverse_before_mfe_1r__": np.asarray([0.70, 2.50, 1.35, 0.20], dtype=np.float32),
        "__underwater_bars_before_mfe_1r__": np.asarray([5.0, 20.0, 12.0, 2.0], dtype=np.float32),
        "__underwater_fraction_before_mfe_1r__": np.asarray([0.25, 0.60, 0.50, 0.10], dtype=np.float32),
    }

    relevance = _lgbm_ranker_relevance_source(y, label_context=label_context, mode="s52_soft_ordered_ev")

    assert relevance is not None
    assert relevance[0] > relevance[1]
    assert relevance[2] > relevance[1]
    assert relevance[2] > relevance[3]


def test_s52_exec_ordered_ev_requires_positive_net_clean_path() -> None:
    y = np.asarray([0.70, 0.90, 0.75, 0.20], dtype=np.float32)
    label_context = {
        "__first_touch_capture_net__": np.asarray([-0.001, 0.050, 0.004, 0.002], dtype=np.float32),
        "__first_touch_round_trip_cost__": np.asarray([0.01, 0.01, 0.01, 0.01], dtype=np.float32),
        "__first_touch_hit__": np.asarray([1, 1, 1, 1], dtype=np.float32),
        "__first_touch_stop__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_timeout__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_valid_path__": np.asarray([1, 1, 1, 1], dtype=np.float32),
        "__first_touch_same_bar_both__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_mae_norm__": np.asarray([0.30, 1.50, 0.25, 0.25], dtype=np.float32),
        "__first_touch_mfe_norm__": np.asarray([1.20, 4.00, 1.20, 1.20], dtype=np.float32),
        "__mfe_1r_before_mae_1r__": np.asarray([1, 0, 1, 1], dtype=np.float32),
        "__mae_1r_before_mfe_1r__": np.asarray([0, 1, 0, 0], dtype=np.float32),
        "__max_adverse_before_mfe_1r__": np.asarray([0.70, 2.50, 0.40, 1.55], dtype=np.float32),
        "__underwater_bars_before_mfe_1r__": np.asarray([5.0, 20.0, 3.0, 12.0], dtype=np.float32),
        "__underwater_fraction_before_mfe_1r__": np.asarray([0.25, 0.60, 0.15, 0.50], dtype=np.float32),
    }

    relevance = _lgbm_ranker_relevance_source(y, label_context=label_context, mode="s52_exec_ordered_ev")

    assert relevance is not None
    assert relevance[2] > relevance[0]
    assert relevance[2] > relevance[1]
    assert relevance[2] > relevance[3]


def test_s52_soft_exec_ordered_ev_boosts_positive_net_clean_path() -> None:
    y = np.asarray([0.70, 0.70, 0.95, 0.65], dtype=np.float32)
    label_context = {
        "__first_touch_capture_net__": np.asarray([-0.001, 0.004, 0.050, 0.002], dtype=np.float32),
        "__first_touch_round_trip_cost__": np.asarray([0.01, 0.01, 0.01, 0.01], dtype=np.float32),
        "__first_touch_hit__": np.asarray([1, 1, 1, 1], dtype=np.float32),
        "__first_touch_stop__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_timeout__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_valid_path__": np.asarray([1, 1, 1, 1], dtype=np.float32),
        "__first_touch_same_bar_both__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_mae_norm__": np.asarray([0.25, 0.25, 1.50, 0.30], dtype=np.float32),
        "__first_touch_mfe_norm__": np.asarray([1.20, 1.20, 4.00, 1.20], dtype=np.float32),
        "__mfe_1r_before_mae_1r__": np.asarray([1, 1, 0, 1], dtype=np.float32),
        "__mae_1r_before_mfe_1r__": np.asarray([0, 0, 1, 0], dtype=np.float32),
        "__max_adverse_before_mfe_1r__": np.asarray([0.40, 0.40, 2.50, 1.35], dtype=np.float32),
        "__underwater_bars_before_mfe_1r__": np.asarray([3.0, 3.0, 20.0, 12.0], dtype=np.float32),
        "__underwater_fraction_before_mfe_1r__": np.asarray([0.15, 0.15, 0.60, 0.50], dtype=np.float32),
    }

    relevance = _lgbm_ranker_relevance_source(y, label_context=label_context, mode="s52_soft_exec_ordered_ev")

    assert relevance is not None
    assert relevance[1] > relevance[0]
    assert relevance[1] > relevance[2]
    assert relevance[3] > relevance[2]


def test_s52_soft_breadth_ordered_ev_demotes_high_net_underwater_path() -> None:
    y = np.asarray([0.70, 0.95, 0.65, 0.20], dtype=np.float32)
    label_context = {
        "__first_touch_capture_net__": np.asarray([0.004, 0.050, 0.002, -0.002], dtype=np.float32),
        "__first_touch_round_trip_cost__": np.asarray([0.01, 0.01, 0.01, 0.01], dtype=np.float32),
        "__first_touch_hit__": np.asarray([1, 1, 1, 0], dtype=np.float32),
        "__first_touch_stop__": np.asarray([0, 0, 0, 1], dtype=np.float32),
        "__first_touch_timeout__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_valid_path__": np.asarray([1, 1, 1, 1], dtype=np.float32),
        "__first_touch_same_bar_both__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_mae_norm__": np.asarray([0.25, 1.50, 0.25, 0.20], dtype=np.float32),
        "__first_touch_mfe_norm__": np.asarray([1.20, 4.00, 1.20, 0.20], dtype=np.float32),
        "__mfe_1r_before_mae_1r__": np.asarray([1, 0, 1, 0], dtype=np.float32),
        "__mae_1r_before_mfe_1r__": np.asarray([0, 1, 0, 0], dtype=np.float32),
        "__max_adverse_before_mfe_1r__": np.asarray([0.45, 2.50, 1.15, 0.20], dtype=np.float32),
        "__underwater_bars_before_mfe_1r__": np.asarray([3.0, 20.0, 9.0, 2.0], dtype=np.float32),
        "__underwater_fraction_before_mfe_1r__": np.asarray([0.15, 0.60, 0.40, 0.10], dtype=np.float32),
    }

    relevance = _lgbm_ranker_relevance_source(
        y,
        label_context=label_context,
        mode="s52_soft_breadth_ordered_ev",
    )

    assert relevance is not None
    assert relevance[0] > relevance[1]
    assert relevance[2] > relevance[1]
    assert relevance[2] > relevance[3]


def test_s52_firstpass_exec_ev_returns_absolute_dirty_tiers() -> None:
    y = np.asarray([0.70, 0.95, 0.65, 0.90], dtype=np.float32)
    label_context = {
        "__first_touch_capture_net__": np.asarray([0.004, 0.050, -0.001, 0.060], dtype=np.float32),
        "__first_touch_round_trip_cost__": np.asarray([0.01, 0.01, 0.01, 0.01], dtype=np.float32),
        "__first_touch_hit__": np.asarray([1, 1, 1, 1], dtype=np.float32),
        "__first_touch_stop__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_timeout__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_valid_path__": np.asarray([1, 1, 1, 1], dtype=np.float32),
        "__first_touch_same_bar_both__": np.asarray([0, 0, 0, 0], dtype=np.float32),
        "__first_touch_mae_norm__": np.asarray([0.25, 1.50, 0.25, 1.70], dtype=np.float32),
        "__first_touch_mfe_norm__": np.asarray([1.20, 4.00, 1.20, 5.00], dtype=np.float32),
        "__mfe_1r_before_mae_1r__": np.asarray([1, 0, 1, 0], dtype=np.float32),
        "__mae_1r_before_mfe_1r__": np.asarray([0, 1, 0, 1], dtype=np.float32),
        "__max_adverse_before_mfe_1r__": np.asarray([0.45, 2.50, 0.90, 3.00], dtype=np.float32),
        "__underwater_bars_before_mfe_1r__": np.asarray([3.0, 20.0, 7.0, 24.0], dtype=np.float32),
        "__underwater_fraction_before_mfe_1r__": np.asarray([0.15, 0.60, 0.33, 0.75], dtype=np.float32),
    }

    relevance = _lgbm_ranker_relevance_source(
        y,
        label_context=label_context,
        mode="s52_firstpass_exec_ev",
    )

    assert _lgbm_ranker_absolute_relevance_mode("s52_firstpass_exec_ev")
    assert relevance is not None
    rel = _lgbm_ranker_absolute_relevance(relevance)
    assert rel[0] >= 3
    assert rel[2] <= 1
    assert rel[1] == 0
    assert rel[3] == 0


def test_s52_full_path_relevance_demotes_dirty_after_clean_first_touch() -> None:
    y = np.asarray([0.50, 0.50], dtype=np.float32)
    label_context = {
        "__first_touch_capture_net__": np.asarray([0.02, 0.025], dtype=np.float32),
        "__first_touch_round_trip_cost__": np.asarray([0.01, 0.01], dtype=np.float32),
        "__first_touch_hit__": np.asarray([1, 1], dtype=np.float32),
        "__first_touch_stop__": np.asarray([0, 0], dtype=np.float32),
        "__first_touch_timeout__": np.asarray([0, 0], dtype=np.float32),
        "__first_touch_valid_path__": np.asarray([1, 1], dtype=np.float32),
        "__first_touch_same_bar_both__": np.asarray([0, 0], dtype=np.float32),
        "__first_touch_mae_norm__": np.asarray([0.25, 0.25], dtype=np.float32),
        "__first_touch_full_path_mae_norm__": np.asarray([0.60, 3.00], dtype=np.float32),
        "__first_touch_mfe_norm__": np.asarray([1.20, 1.20], dtype=np.float32),
        "__mfe_1r_before_mae_1r__": np.asarray([1, 1], dtype=np.float32),
        "__mae_1r_before_mfe_1r__": np.asarray([0, 0], dtype=np.float32),
    }

    relevance = _lgbm_ranker_relevance_source(y, label_context=label_context, mode="s52_full_path")

    assert relevance is not None
    assert relevance[0] > relevance[1]


def test_lgbm_ranker_fit_accepts_explicit_relevance_override() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.1, 1.1, 0.2, 1.2],
            "side": [1.0] * 6,
        }
    )
    y = np.asarray([0.1, 0.9, 0.1, 0.9, 0.1, 0.9], dtype=np.float32)
    relevance = np.asarray([4.0, 0.0, 4.0, 0.0, 4.0, 0.0], dtype=np.float32)
    groups = np.asarray(["t0", "t0", "t1", "t1", "t2", "t2"], dtype=object)

    old_enabled = lgbm_pipeline.LGBM_RANKER_ENABLED
    old_objectives = set(lgbm_pipeline.LGBM_RANKER_OBJECTIVES)
    try:
        lgbm_pipeline.LGBM_RANKER_ENABLED = True
        lgbm_pipeline.LGBM_RANKER_OBJECTIVES = {"train_base"}
        model = _fit_lgbm_model(
            frame[["x", "side"]],
            y,
            np.ones(len(y), dtype=np.float32),
            classifier=True,
            params={
                "n_estimators": 3,
                "learning_rate": 0.1,
                "num_leaves": 3,
                "min_child_samples": 1,
                "random_state": 7,
                "verbose": -1,
            },
            objective_mode="train_base",
            ranker_train_groups=groups,
            ranker_train_relevance=relevance,
        )
    finally:
        lgbm_pipeline.LGBM_RANKER_ENABLED = old_enabled
        lgbm_pipeline.LGBM_RANKER_OBJECTIVES = old_objectives

    assert getattr(model, "_ares_lgbm_ranker_enabled_", False)
