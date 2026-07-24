import numpy as np
import pytest

from extreme_price_movements.path_auxiliary_targets import (
    ALL_SUPPORTIVE_LABEL_COLUMNS,
    FUTURE_SLOPE_ATR_PER_HOUR_CLIP,
    MAE_BEFORE_MEANINGFUL_MFE_ATR_CLIP,
    MIN_USABLE_MFE_ATR,
    MIN_USABLE_MFE_RETURN,
    PEAK_MFE_ATR_CLIP,
    SUPPORTIVE_LABEL_COLUMNS,
    bars_before_price_stops_decreasing_regression_metrics,
    build_path_auxiliary_targets,
    future_slope_atr_per_hour_regression_metrics,
    mae_before_meaningful_mfe_regression_metrics,
    peak_mfe_regression_metrics,
    required_target_columns,
    timing_regression_metrics,
)


def test_requested_supportive_aliases_are_exact_materialized_float32_targets():
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0, 100.0]),
        future_high=np.array(
            [[101.0, 103.0, 104.0, 102.0], [101.0, 102.0, 103.0, 100.0]]
        ),
        future_low=np.array(
            [[99.0, 98.0, 97.0, 100.0], [99.0, 97.0, 96.0, 98.0]]
        ),
        atr_fraction=np.array([0.02, 0.02]),
        side_sign=np.array([1.0, -1.0]),
        bar_minutes=60,
        horizon_hours=4,
    )
    columns = targets.as_columns()

    assert tuple(SUPPORTIVE_LABEL_COLUMNS) == (
        "peak_mfe_12h_atr",
        "time_to_first_meaningful_mfe",
        "mae_before_meaningful_mfe_atr",
        "bars_before_price_stops_decreasing",
        "future_slope_atr_per_hour",
    )
    assert set(ALL_SUPPORTIVE_LABEL_COLUMNS).issubset(columns)
    assert set(ALL_SUPPORTIVE_LABEL_COLUMNS).issubset(required_target_columns())
    assert all(columns[name].dtype == np.float32 for name in ALL_SUPPORTIVE_LABEL_COLUMNS)

    for name in ALL_SUPPORTIVE_LABEL_COLUMNS:
        np.testing.assert_allclose(columns[name][0], columns[name][1], equal_nan=True)

    np.testing.assert_allclose(
        [columns[f"__mfe_ge_{label}atr__"][0] for label in ("0_5", "1", "1_5", "2", "3", "4")],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    )
    assert columns["__bars_above_50pct_peak__"][0] == 3.0
    assert columns["__bars_above_80pct_peak__"][0] == 1.0
    assert columns["__fraction_bars_above_50pct_peak__"][0] == pytest.approx(0.75)
    assert columns["__fraction_bars_above_80pct_peak__"][0] == pytest.approx(0.25)
    assert columns["__mfe_integral_atr_hours__"][0] == pytest.approx(5.0)

    assert columns["__mfe_ratio_at_2h_to_peak__"][0] == pytest.approx(0.75)
    assert columns["__mfe_ratio_at_4h_to_peak__"][0] == 1.0
    assert np.isnan(columns["__mfe_ratio_at_8h_to_peak__"][0])
    np.testing.assert_allclose(
        [columns[f"__peak_within_{hours}h__"][0] for hours in (1, 2, 4)],
        [0.0, 0.0, 1.0],
    )
    assert np.isnan(columns["__peak_within_8h__"][0])
    assert columns["__mfe_2h_over_mfe_12h__"][0] == pytest.approx(0.75)
    assert columns["__peak_mfe_atr_clip_6__"][0] == pytest.approx(2.0)
    assert columns["__reaches_1_5atr_within_12h__"][0] == 1.0
    assert np.isfinite(columns["__path_efficiency_to_1_5atr__"][0])

    np.testing.assert_allclose(
        [
            columns[f"__pre_mfe_mae_ge_{label}atr__"][0]
            for label in ("0_25", "0_5", "0_75", "1", "1_5")
        ],
        [1.0, 1.0, 1.0, 1.0, 0.0],
    )
    assert columns["__meaningful_mfe_before_mae_1atr__"][0] == 0.0
    assert columns["__pre_mfe_underwater_bars__"][0] == 2.0
    assert columns["__pre_mfe_underwater_fraction__"][0] == 1.0

    assert columns["__adverse_trough_atr__"][0] == pytest.approx(1.5)
    assert columns["__adverse_trough_bar__"][0] == 3.0
    assert columns["__adverse_trough_recovery_fraction__"][0] == 1.0
    assert columns["__adverse_trough_recovered_100pct__"][0] == 1.0
    assert columns["__adverse_trough_recovery_100pct_confirmed_2bars__"][0] == 0.0
    assert columns["__bars_from_adverse_trough_to_full_recovery__"][0] == 1.0
    assert columns["__time_from_adverse_trough_to_full_recovery_hours__"][0] == 1.0

    assert columns["__future_slope_2h_atr_per_hour__"][0] == pytest.approx(0.6)
    assert columns["__future_slope_4h_atr_per_hour__"][0] == pytest.approx(1.6 / 3.0)
    assert np.isnan(columns["__future_slope_8h_atr_per_hour__"][0])
    assert columns["__bars_to_peak_mfe__"][0] == 3.0
    assert columns["__mfe_mae_path_efficiency__"][0] == pytest.approx(2.0 / 3.5)
    assert columns["__mfe_integral_path_efficiency__"][0] == pytest.approx(0.625)
    assert columns["__mfe_timing_path_efficiency__"][0] == pytest.approx(0.25)
    assert columns["__mfe_persistence_path_efficiency__"][0] == pytest.approx(0.25)


def test_requested_supportive_labels_preserve_nan_and_censoring_semantics():
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0, np.nan]),
        future_high=np.array([[101.0, 101.0], [101.0, 101.0]]),
        future_low=np.array([[99.0, 98.0], [99.0, 98.0]]),
        atr_fraction=np.array([0.02, 0.02]),
        side_sign=np.array([1.0, 1.0]),
        bar_minutes=60,
        horizon_hours=2,
    )
    columns = targets.as_columns()

    # Valid but unreached meaningful MFE is censored at the horizon; labels
    # requiring a non-zero MFE/trough remain explicitly undefined.
    assert targets.time_to_first_meaningful_mfe_hours_12h[0] == 2.0
    assert columns["__peak_within_1h__"][0] == 1.0
    assert not np.isnan(columns["__time_to_peak_mfe_hours__"][0])
    for name in ALL_SUPPORTIVE_LABEL_COLUMNS:
        assert np.isnan(columns[name][1])


def test_build_targets_is_side_relative_and_uses_full_horizon():
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0, 100.0]),
        future_high=np.array([[101.0, 104.0, 103.0], [101.0, 102.0, 103.0]]),
        future_low=np.array([[99.0, 98.0, 97.0], [99.0, 94.0, 96.0]]),
        atr_fraction=np.array([0.02, 0.02]),
        side_sign=np.array([1.0, -1.0]),
        bar_minutes=60,
        horizon_hours=3,
    )
    np.testing.assert_allclose(targets.peak_mfe_return_12h, [0.04, 0.06])
    np.testing.assert_allclose(targets.peak_mfe_atr_12h, [2.0, 3.0])
    np.testing.assert_allclose(
        targets.time_to_first_meaningful_mfe_hours_12h, [2.0, 2.0]
    )


def test_target_metrics_report_requested_diagnostics():
    true_time = np.log1p(np.array([1.0, 3.0, 6.0, 10.0]))
    pred_time = true_time.copy()
    timing = timing_regression_metrics(true_time, pred_time)
    assert timing["mae_log_time"] == 0.0
    assert timing["mae_hours"] == 0.0
    assert timing["accuracy_meaningful_mfe_by_2h"] == 1.0
    peak = peak_mfe_regression_metrics(
        np.log1p(np.array([0.5, 1.0, 2.0, 4.0])),
        np.log1p(np.array([0.5, 1.0, 2.0, 4.0])),
    )
    assert peak["mae_log_peak_mfe_atr"] == 0.0
    assert peak["top_01pct_realized_peak_mfe_atr"] == pytest.approx(4.0)


def test_constant_predictions_have_zero_not_nan_rank_information():
    target = np.log1p(np.array([1.0, 2.0, 3.0, 4.0]))
    constant = np.full(4, target.mean())
    assert timing_regression_metrics(target, constant)["spearman_ic"] == 0.0
    assert peak_mfe_regression_metrics(target, constant)["spearman_ic"] == 0.0


def test_target_requires_complete_path_and_unreached_mfe_is_capped_at_horizon():
    with pytest.raises(ValueError, match="complete requested horizon"):
        build_path_auxiliary_targets(
            entry_price=np.array([100.0]),
            future_high=np.array([[101.0]]),
            future_low=np.array([[99.0]]),
            atr_fraction=np.array([0.02]),
            side_sign=np.array([1.0]),
            bar_minutes=60,
            horizon_hours=2,
        )
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0]),
        future_high=np.array([[99.0, 99.5]]),
        future_low=np.array([[98.0, 98.5]]),
        atr_fraction=np.array([0.02]),
        side_sign=np.array([1.0]),
        bar_minutes=60,
        horizon_hours=2,
    )
    assert targets.valid[0]
    assert targets.timing_valid[0]
    assert not targets.meaningful_mfe_reached[0]
    assert targets.time_to_first_meaningful_mfe_hours_12h[0] == 2.0


def test_peak_mfe_atr_target_is_robust_to_near_zero_atr_denominator():
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0]),
        future_high=np.array([[110.0, 111.0]]),
        future_low=np.array([[99.0, 99.0]]),
        atr_fraction=np.array([1e-30]),
        side_sign=np.array([1.0]),
        bar_minutes=60,
        horizon_hours=2,
    )
    np.testing.assert_allclose(targets.peak_mfe_atr_12h, [PEAK_MFE_ATR_CLIP])
    np.testing.assert_allclose(
        targets.log1p_peak_mfe_atr_12h, [np.log1p(PEAK_MFE_ATR_CLIP)]
    )


def test_mfe_below_economic_floor_is_zero_and_timing_is_censored():
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0, 100.0]),
        future_high=np.array([[101.4] * 12, [101.9] * 12]),
        future_low=np.array([[99.0] * 12, [99.0] * 12]),
        # Row one is governed by 1.5 ATR = 1.5%; row two by 1.5 ATR = 6%.
        atr_fraction=np.array([0.01, 0.04]),
        side_sign=np.array([1.0, 1.0]),
    )
    assert MIN_USABLE_MFE_ATR == 1.5
    assert MIN_USABLE_MFE_RETURN == 0.015
    np.testing.assert_allclose(targets.peak_mfe_return_12h, [0.0, 0.0])
    np.testing.assert_allclose(targets.peak_mfe_atr_12h, [0.0, 0.0])
    assert targets.timing_valid.all()
    assert not targets.meaningful_mfe_reached.any()
    np.testing.assert_allclose(
        targets.time_to_first_meaningful_mfe_hours_12h, [12.0, 12.0]
    )


def test_meaningful_mfe_floor_uses_max_of_15atr_and_15pct_near_boundary():
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0, 100.0]),
        future_high=np.array([[101.5001] * 12, [103.0001] * 12]),
        future_low=np.array([[99.0] * 12, [99.0] * 12]),
        atr_fraction=np.array([0.01, 0.02]),
        side_sign=np.array([1.0, 1.0]),
    )
    np.testing.assert_allclose(targets.peak_mfe_return_12h, [0.015001, 0.030001])
    np.testing.assert_allclose(targets.peak_mfe_atr_12h, [1.5001, 1.50005])
    np.testing.assert_allclose(
        targets.time_to_first_meaningful_mfe_hours_12h, [1.0, 1.0]
    )


def test_first_meaningful_mfe_time_is_capped_for_non_divisor_bars():
    bars = 15  # ceil(12h * 60 / 50m); the final bar ends at 12.5h.
    high = np.ones((1, bars), dtype=float)
    low = np.ones((1, bars), dtype=float)
    high[0, -1] = 1.02

    targets = build_path_auxiliary_targets(
        entry_price=np.array([1.0]),
        future_high=high,
        future_low=low,
        atr_fraction=np.array([0.01]),
        side_sign=np.array([1.0]),
        bar_minutes=50,
        horizon_hours=12,
    )

    assert targets.meaningful_mfe_reached.tolist() == [True]
    assert targets.time_to_first_meaningful_mfe_hours_12h.tolist() == pytest.approx(
        [12.0]
    )


def test_new_path_shape_targets_are_long_short_symmetric():
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0, 100.0]),
        future_high=np.array(
            [
                [101.0, 102.0, 103.0, 104.0],
                [101.0, 103.0, 104.0, 102.0],
            ]
        ),
        future_low=np.array(
            [
                [99.0, 97.0, 96.0, 98.0],
                [99.0, 98.0, 97.0, 96.0],
            ]
        ),
        atr_fraction=np.array([0.02, 0.02]),
        side_sign=np.array([1.0, -1.0]),
        bar_minutes=60,
        horizon_hours=4,
    )

    assert targets.meaningful_mfe_reached.tolist() == [True, True]
    np.testing.assert_allclose(targets.mae_before_meaningful_mfe_atr_12h, [2.0, 2.0])
    np.testing.assert_allclose(
        targets.bars_before_price_stops_decreasing_12h, [2.0, 2.0]
    )
    np.testing.assert_allclose(
        targets.future_slope_atr_per_hour_12h, [0.4, 0.4]
    )


def test_path_shape_targets_use_full_path_when_meaningful_mfe_is_unreached():
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0]),
        future_high=np.array([[100.5, 101.0, 100.8, 100.7]]),
        future_low=np.array([[99.5, 98.0, 97.0, 98.0]]),
        atr_fraction=np.array([0.02]),
        side_sign=np.array([1.0]),
        bar_minutes=60,
        horizon_hours=4,
    )

    assert not targets.meaningful_mfe_reached[0]
    np.testing.assert_allclose(targets.mae_before_meaningful_mfe_atr_12h, [1.5])
    np.testing.assert_allclose(
        targets.bars_before_price_stops_decreasing_12h, [3.0]
    )
    # The slope remains a pure path-shape target even when the economic floor
    # is unreached: its 0.5 ATR peak reaches 80% on the second bar.
    np.testing.assert_allclose(targets.future_slope_atr_per_hour_12h, [0.2])


def test_turning_point_is_zero_when_entry_is_the_adverse_extreme():
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0]),
        future_high=np.array([[101.0, 102.0, 104.0]]),
        future_low=np.array([[100.5, 100.2, 100.1]]),
        atr_fraction=np.array([0.02]),
        side_sign=np.array([1.0]),
        bar_minutes=60,
        horizon_hours=3,
    )

    assert targets.meaningful_mfe_reached[0]
    np.testing.assert_allclose(targets.mae_before_meaningful_mfe_atr_12h, [0.0])
    np.testing.assert_allclose(
        targets.bars_before_price_stops_decreasing_12h, [0.0]
    )


def test_shape_targets_exclude_partial_irregular_horizon_bar():
    bars = 15  # ceil(12h / 50m); legacy timing remains capped at 12 hours.
    high = np.ones((1, bars), dtype=float)
    low = np.ones((1, bars), dtype=float)
    high[0, -1] = 1.10
    low[0, -2] = 0.90
    targets = build_path_auxiliary_targets(
        entry_price=np.array([1.0]),
        future_high=high,
        future_low=low,
        atr_fraction=np.array([0.01]),
        side_sign=np.array([1.0]),
        bar_minutes=50,
        horizon_hours=12,
    )

    assert targets.time_to_first_meaningful_mfe_hours_12h[0] == pytest.approx(12.0)
    # The new shape targets only use the first 14 complete 50-minute bars,
    # rather than the final 12.5-hour bar used by the legacy timing target.
    assert targets.bars_before_price_stops_decreasing_12h[0] == pytest.approx(14.0)
    assert targets.mae_before_meaningful_mfe_atr_12h[0] == pytest.approx(10.0)
    assert targets.future_slope_atr_per_hour_12h[0] == pytest.approx(0.0)


def test_slope_uses_first_80pct_peak_endpoint():
    targets = build_path_auxiliary_targets(
        entry_price=np.array([100.0]),
        future_high=np.array([[101.0, 103.0]]),
        future_low=np.array([[99.0, 99.0]]),
        atr_fraction=np.array([0.02]),
        side_sign=np.array([1.0]),
        bar_minutes=60,
        horizon_hours=2,
    )

    # Peak MFE is 1.5 ATR. Its 1.2 ATR (80%) level is first reached at bar two.
    assert targets.future_slope_atr_per_hour_12h[0] == pytest.approx(0.6)


def test_slope_clipping_and_invalid_atr_are_explicit():
    clipped = build_path_auxiliary_targets(
        entry_price=np.array([1.0]),
        future_high=np.full((1, 60), 2.0),
        future_low=np.zeros((1, 60)),
        atr_fraction=np.array([0.01]),
        side_sign=np.array([1.0]),
        bar_minutes=1,
        horizon_hours=1,
    )
    assert clipped.mae_before_meaningful_mfe_atr_12h[0] == pytest.approx(
        MAE_BEFORE_MEANINGFUL_MFE_ATR_CLIP
    )
    assert clipped.future_slope_atr_per_hour_12h[0] == pytest.approx(
        FUTURE_SLOPE_ATR_PER_HOUR_CLIP
    )

    invalid = build_path_auxiliary_targets(
        entry_price=np.array([100.0, 100.0]),
        future_high=np.array([[103.0], [103.0]]),
        future_low=np.array([[97.0], [97.0]]),
        atr_fraction=np.array([np.nan, 0.0]),
        side_sign=np.array([1.0, 1.0]),
        bar_minutes=60,
        horizon_hours=1,
    )
    assert not invalid.valid.any()
    assert np.isnan(invalid.mae_before_meaningful_mfe_atr_12h).all()
    assert np.isnan(invalid.bars_before_price_stops_decreasing_12h).all()
    assert np.isnan(invalid.future_slope_atr_per_hour_12h).all()


def test_new_target_regression_metrics_are_target_namespaced():
    depth = np.log1p(np.array([0.0, 1.0, 2.0, 4.0]))
    slope = np.log1p(np.array([0.0, 0.5, 1.0, 2.0]))
    bars = np.log1p(np.array([0.0, 1.0, 4.0, 12.0]))

    depth_metrics = mae_before_meaningful_mfe_regression_metrics(depth, depth)
    slope_metrics = future_slope_atr_per_hour_regression_metrics(slope, slope)
    bars_metrics = bars_before_price_stops_decreasing_regression_metrics(bars, bars)

    assert depth_metrics["mae_before_meaningful_mfe_atr_log_mae"] == 0.0
    assert depth_metrics["mae_before_meaningful_mfe_atr_natural_rmse"] == 0.0
    assert depth_metrics["mae_before_meaningful_mfe_atr_top_01pct_realized"] == pytest.approx(
        4.0
    )
    assert slope_metrics["future_slope_atr_per_hour_natural_huber"] == 0.0
    assert slope_metrics["future_slope_atr_per_hour_top_01pct_realized"] == pytest.approx(
        2.0
    )
    assert bars_metrics["bars_before_price_stops_decreasing_log_mae"] == 0.0
    assert bars_metrics["bars_before_price_stops_decreasing_mae_bars"] == 0.0
    for threshold in (1, 2, 4, 8, 12):
        assert (
            bars_metrics[
                f"bars_before_price_stops_decreasing_accuracy_by_{threshold}_bars"
            ]
            == 1.0
        )
