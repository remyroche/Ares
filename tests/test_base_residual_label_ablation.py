from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.base_residual_label_ablation import (
    FixedWindowCalendar,
    LabelRecipe,
    build_soft_label,
    default_label_recipes,
    label_components,
    rank_mask,
)


def test_calendar_is_exact_four_six_three_three_and_purged() -> None:
    calendar = FixedWindowCalendar.from_first_oos_month("2026-01")
    assert calendar.base_train_start == pd.Timestamp("2025-09-01", tz="UTC")
    assert calendar.base_train_end == pd.Timestamp("2026-01-01", tz="UTC")
    assert calendar.base_oos_end == pd.Timestamp("2026-07-01", tz="UTC")
    assert calendar.meta_train_end == pd.Timestamp("2026-04-01", tz="UTC")
    masks = calendar.masks(
        [
            "2025-09-01T00:00:00Z",
            "2025-12-30T22:59:59Z",
            "2025-12-30T23:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-04-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        ]
    )
    assert masks["base_train"].tolist() == [True, True, False, False, False, False]
    assert masks["base_oos"].tolist() == [False, False, False, True, True, False]
    assert masks["meta_train"].tolist() == [False, False, False, True, False, False]
    assert masks["meta_oos"].tolist() == [False, False, False, False, True, False]


def test_time_and_mae_make_time_aware_label_stronger_only_when_better() -> None:
    frame = pd.DataFrame(
        {
            "__first_touch_target_soft__": [0.5, 0.5],
            "__peak_mfe_atr_12h__": [2.0, 2.0],
            "__mae_before_meaningful_mfe_atr_12h__": [0.1, 1.5],
            "__time_to_first_meaningful_mfe_hours_12h__": [1.0, 11.0],
            "__bars_to_80pct_peak__": [1.0, 11.0],
            "__mfe_before_60m_atr__": [0.5, 0.0],
            "__mfe_2h_over_mfe_12h__": [0.9, 0.1],
            "__adverse_trough_within_60m__": [0.0, 1.0],
            "__adverse_trough_within_120m__": [0.0, 1.0],
            "__future_slope_atr_per_hour_12h__": [0.3, -0.3],
            "__meaningful_mfe_reached_12h__": [1.0, 1.0],
        }
    )
    recipe = LabelRecipe("time", 0.3, 0.1, 0.2, 0.2, 0.1, 0.1)
    soft, hard = build_soft_label(label_components(frame), recipe)
    assert soft[0] > soft[1]
    assert hard[0] >= hard[1]


def test_12h_timeout_stays_below_half_without_meaningful_mfe() -> None:
    frame = pd.DataFrame(
        {
            "__first_touch_target_soft__": [1.0],
            "__peak_mfe_atr_12h__": [0.5],
            "__mae_before_meaningful_mfe_atr_12h__": [0.2],
            "__meaningful_mfe_reached_12h__": [0.0],
        }
    )
    assert float(label_components(frame)["execution_12h"].iloc[0]) < 0.5


def test_baseline_recipe_replays_incumbent_soft_label_exactly() -> None:
    frame = pd.DataFrame({"__first_touch_target_soft__": [0.1, 0.7]})
    components = label_components(frame)
    baseline = default_label_recipes()[0]
    soft, hard = build_soft_label(components, baseline)
    np.testing.assert_allclose(soft, [0.1, 0.7])
    np.testing.assert_array_equal(hard, [0.0, 1.0])


def test_rank_mask_is_deterministic_and_side_local() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-04-01"] * 4, utc=True),
            "side_name": ["long", "long", "short", "short"],
            "__symbol__": ["B", "A", "D", "C"],
        }
    )
    selected = rank_mask(frame, [1.0, 1.0, 2.0, 2.0], fraction=0.5)
    assert selected.tolist() == [False, True, False, True]
