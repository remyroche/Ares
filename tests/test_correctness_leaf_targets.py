import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.performance_regimes.correctness_leaf_targets import (
    aggregate_correctness_periods,
    binary_surprise,
    fit_correctness_scale,
    probability_entropy,
    select_top_base_per_timestamp,
    soft_correctness,
    soft_negative_surprise,
    soft_positive_surprise,
)


def test_soft_correctness_is_zero_centred_and_clipped() -> None:
    scale = fit_correctness_scale(np.linspace(-200., 200., 101))
    y = soft_correctness([-1e6, 0., 1e6], scale)
    assert y[0] == 0. and y[1] == .5 and y[2] == 1.


def test_top_base_cohort_and_period_label_availability() -> None:
    ts = pd.to_datetime(["2024-01-01 00:00Z"] * 3 + ["2024-01-01 01:00Z"] * 3)
    frame = pd.DataFrame({"candidate_id": list("abcdef"), "__ts__": ts, "base": [1., 3., 2., 4., 6., 5.], "y": [0., 1., .5, .2, .4, .6], "label_available_ts": ts + pd.Timedelta(hours=13)})
    selected = select_top_base_per_timestamp(frame, score_column="base", fraction=.05)
    assert frame.loc[selected, "candidate_id"].tolist() == ["b", "e"]
    output = aggregate_correctness_periods(frame.loc[selected], target_column="y", horizon_hours=12)
    assert np.allclose(output.period_correctness_target, .7)
    assert output.period_label_available_ts.min() == pd.Timestamp("2024-01-01 14:00Z")


def test_economic_surprise_targets_use_declared_50_75_bps_boundaries() -> None:
    residual = np.array([-100., -75., -50., 0., 50., 62.5, 75., 100.])
    assert np.allclose(soft_positive_surprise(residual), [0., 0., 0., 0., 0., .5, 1., 1.])
    assert np.allclose(soft_negative_surprise(residual), [1., 1., 0., 0., 0., 0., 0., 0.])
    assert np.array_equal(binary_surprise(residual), [1., 1., 0., 0., 0., 1., 1., 1.])


def test_probability_entropy_is_normalized_and_simplex_safe() -> None:
    entropy = probability_entropy(np.array([[1., 0., 0.], [1 / 3, 1 / 3, 1 / 3]]))
    assert entropy[0] < 1e-5
    assert entropy[1] == pytest.approx(1.0)
