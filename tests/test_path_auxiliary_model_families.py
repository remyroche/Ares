"""Tests for canonical mixture/survival path auxiliary role contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.path_auxiliary_model_families import (
    HEAD_SPECS,
    ROLE_SPECS_BY_NAME,
    build_role_targets,
    compose_adverse_timing_predictions,
    compose_mae_predictions,
    compose_peak_predictions,
    compose_timing_cdf_predictions,
    conditional_regression_metrics,
    probability_calibration_metrics,
    project_monotone_timing_cdf,
    validate_canonical_auxiliary_labels,
)


def _labels() -> pd.DataFrame:
    """Small canonical-like frame with hit/no-hit and optional trough examples."""

    return pd.DataFrame(
        {
            "__path_auxiliary_target_valid__": [1, 1, 1, 1, 0],
            # This is intentionally distinct from an ATR-only support event.
            "__meaningful_mfe_reached_12h__": [1, 0, 1, 0, np.nan],
            "__mfe_ge_1_5atr__": [1, 1, 1, 0, np.nan],
            "__time_to_first_meaningful_mfe_hours_12h__": [2, 12, 8, 12, np.nan],
            "__peak_mfe_atr_12h__": [3.0, 0.0, 2.0, 0.0, np.nan],
            "__mae_before_meaningful_mfe_atr_12h__": [0.5, 3.0, 1.0, 4.0, np.nan],
            "__mae_until_horizon_if_no_1_5atr__": [np.nan, 3.0, np.nan, 4.0, np.nan],
            "__bars_to_adverse_extreme_before_mfe_12h__": [1.0, 4.0, 2.0, 5.0, np.nan],
            # No adversarial trough on rows 1 and 3: that is a real missing
            # conditional target, not a broken canonical path label.
            "__bars_to_confirmed_adverse_trough__": [4.0, np.nan, 5.0, np.nan, np.nan],
            "__future_slope_atr_per_hour_12h__": [1.0, 0.0, 0.5, 0.1, np.nan],
        }
    )


def test_exactly_five_fixed_heads_and_required_role_families() -> None:
    assert [head.name for head in HEAD_SPECS] == [
        "peak_mfe_12h_atr",
        "time_to_first_meaningful_mfe",
        "mae_before_meaningful_mfe_atr",
        "bars_before_price_stops_decreasing",
        "future_slope_atr_per_hour",
    ]
    assert {
        "peak_mfe_12h_atr.p_hit",
        "peak_mfe_12h_atr.conditional_mean",
        "peak_mfe_12h_atr.conditional_q80",
        "time_to_first_meaningful_mfe.hit_by_2h",
        "time_to_first_meaningful_mfe.hit_by_4h",
        "time_to_first_meaningful_mfe.hit_by_8h",
        "time_to_first_meaningful_mfe.hit_by_12h",
        "mae_before_meaningful_mfe_atr.p_hit",
        "mae_before_meaningful_mfe_atr.if_hit",
        "mae_before_meaningful_mfe_atr.if_no_hit",
        "bars_before_price_stops_decreasing.legacy_adverse_extreme",
        "bars_before_price_stops_decreasing.confirmed_adverse_trough",
        "future_slope_atr_per_hour.diagnostic",
    } == set(ROLE_SPECS_BY_NAME)
    assert ROLE_SPECS_BY_NAME["peak_mfe_12h_atr.conditional_q80"].quantile == 0.80
    assert (
        ROLE_SPECS_BY_NAME["future_slope_atr_per_hour.diagnostic"].deployment_status
        == "diagnostic_only"
    )


def test_role_targets_are_aligned_and_conditionally_masked_without_row_drop() -> None:
    targets = build_role_targets(_labels())

    assert all(len(item.target) == 5 for item in targets.values())
    assert all(len(item.train_mask) == 5 for item in targets.values())
    assert np.array_equal(
        targets["peak_mfe_12h_atr.p_hit"].target[:4], np.array([1.0, 0.0, 1.0, 0.0])
    )
    assert np.flatnonzero(
        targets["peak_mfe_12h_atr.conditional_mean"].train_mask
    ).tolist() == [0, 2]
    assert np.flatnonzero(
        targets["peak_mfe_12h_atr.conditional_q80"].train_mask
    ).tolist() == [0, 2]
    assert np.flatnonzero(
        targets["time_to_first_meaningful_mfe.hit_by_4h"].train_mask
    ).tolist() == [0, 1, 2, 3]
    assert np.array_equal(
        targets["time_to_first_meaningful_mfe.hit_by_4h"].target[:4],
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert np.array_equal(
        targets["time_to_first_meaningful_mfe.hit_by_12h"].target[:4],
        np.array([1.0, 0.0, 1.0, 0.0]),
    )
    assert np.flatnonzero(
        targets["mae_before_meaningful_mfe_atr.if_hit"].train_mask
    ).tolist() == [0, 2]
    assert np.flatnonzero(
        targets["mae_before_meaningful_mfe_atr.if_no_hit"].train_mask
    ).tolist() == [1, 3]
    assert (
        targets["mae_before_meaningful_mfe_atr.if_no_hit"].source_column
        == "__mae_before_meaningful_mfe_atr_12h__"
    )
    assert np.allclose(
        targets["mae_before_meaningful_mfe_atr.if_no_hit"].target[[1, 3]], [3.0, 4.0]
    )
    assert np.flatnonzero(
        targets[
            "bars_before_price_stops_decreasing.confirmed_adverse_trough"
        ].train_mask
    ).tolist() == [0, 2]
    assert np.all(
        np.isnan(targets["mae_before_meaningful_mfe_atr.if_hit"].target[[1, 3, 4]])
    )


def test_meaningful_event_is_required_and_never_replaced_by_atr_only_support_event() -> (
    None
):
    labels = _labels().drop(columns="__meaningful_mfe_reached_12h__")
    with pytest.raises(ValueError, match="meaningful_mfe_reached_12h"):
        build_role_targets(labels, role_names=["peak_mfe_12h_atr.p_hit"])


def test_optional_confirmed_trough_missingness_is_a_training_mask_not_a_label_substitution() -> (
    None
):
    labels = _labels().drop(columns="__bars_to_adverse_extreme_before_mfe_12h__")
    labels["__bars_before_price_stops_decreasing_12h__"] = [1.0, 4.0, 2.0, 5.0, np.nan]
    validate_canonical_auxiliary_labels(
        labels,
        role_names=[
            "bars_before_price_stops_decreasing.legacy_adverse_extreme",
            "bars_before_price_stops_decreasing.confirmed_adverse_trough",
        ],
    )
    roles = build_role_targets(
        labels,
        role_names=[
            "bars_before_price_stops_decreasing.legacy_adverse_extreme",
            "bars_before_price_stops_decreasing.confirmed_adverse_trough",
        ],
    )
    assert (
        roles["bars_before_price_stops_decreasing.legacy_adverse_extreme"].source_column
        == "__bars_before_price_stops_decreasing_12h__"
    )
    assert np.flatnonzero(
        roles["bars_before_price_stops_decreasing.confirmed_adverse_trough"].train_mask
    ).tolist() == [0, 2]


def test_timing_cdf_projection_is_isotonic_and_composition_is_in_natural_hours() -> (
    None
):
    projected = project_monotone_timing_cdf(
        {2: [0.8, 0.1], 4: [0.2, 0.5], 8: [0.7, 0.4], 12: [0.6, 0.9]}
    )
    assert np.allclose(projected[2.0], [0.5, 0.1])
    assert np.allclose(projected[4.0], [0.5, 0.45])
    assert np.allclose(projected[8.0], [0.65, 0.45])
    assert np.allclose(projected[12.0], [0.65, 0.9])
    composed = compose_timing_cdf_predictions(
        {2: [0.8, 0.1], 4: [0.2, 0.5], 8: [0.7, 0.4], 12: [0.6, 0.9]}
    )
    assert np.allclose(composed["p_hit_12h"], [0.65, 0.9])
    assert np.all(composed["expected_censored_time_hours"] <= 12.0)
    assert np.all(composed["expected_censored_time_hours"] >= 0.0)
    never_hits = compose_timing_cdf_predictions(
        {2: [0.0], 4: [0.0], 8: [0.0], 12: [0.0]}
    )
    assert np.allclose(never_hits["expected_censored_time_hours"], [12.0])


def test_natural_unit_mixture_and_adverse_compositions() -> None:
    peak = compose_peak_predictions([0.25, 0.8], [4.0, 2.0], [7.0, 3.0])
    assert np.allclose(peak["expected_peak_mfe_atr"], [1.0, 1.6])
    mae = compose_mae_predictions([0.25, 0.8], [1.0, 2.0], [5.0, 4.0])
    assert np.allclose(mae["expected_mae_atr"], [4.0, 2.4])
    adverse = compose_adverse_timing_predictions([1.0, 3.0], [4.0, 4.0])
    assert np.allclose(adverse["confirmed_minus_legacy_bars"], [3.0, 1.0])


def test_probability_calibration_and_conditional_regression_metrics_are_safe() -> None:
    probability = probability_calibration_metrics(
        [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], n_bins=2
    )
    assert probability["rows"] == 4
    assert probability["brier"] == pytest.approx(0.025)
    assert probability["roc_auc"] == pytest.approx(1.0)
    assert 0.0 <= probability["ece"] <= 1.0
    assert len(probability["calibration_bins"]) == 2

    conditional = conditional_regression_metrics(
        [1.0, 3.0, 2.0, 4.0],
        [1.2, 2.5, 2.2, 5.0],
        [1, 0, 1, 0],
    )
    assert conditional["overall"]["rows"] == 4
    assert conditional["if_hit"]["rows"] == 2
    assert conditional["if_no_hit"]["rows"] == 2
    assert conditional["overall"]["mae"] == pytest.approx(0.475)


def test_invalid_probability_and_negative_target_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        compose_peak_predictions([1.1], [1.0], [2.0])
    labels = _labels()
    labels.loc[0, "__peak_mfe_atr_12h__"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        build_role_targets(labels, role_names=["peak_mfe_12h_atr.conditional_mean"])
