import numpy as np
import pandas as pd
import pytest

from src.training.steps.labeling.generate_weights_per_label import (
    generate_weights_per_label,
)


def test_generate_weights_happy_path_all_factors():
    returns = np.array([0.02, -0.01, 0.03, -0.02])
    t_events = pd.date_range("2020-01-01", periods=4, freq="T")
    consistency = np.array([0.2, 0.5, 0.8, 0.9])
    uniqueness = np.array([1.0, 0.8, 1.2, 1.1])
    vol = np.array([0.01, 0.015, 0.02, 0.025])

    weights = generate_weights_per_label(
        returns=returns,
        t_events=t_events,
        consistency_scores=consistency,
        uniqueness_scores=uniqueness,
        vol_proxy=vol,
        downside_multiplier=1.5,
    )

    assert len(weights) == len(returns)
    assert np.isfinite(weights).all()
    assert weights.mean() == pytest.approx(1.0, rel=1e-6)
    assert weights[2] == pytest.approx(weights.max())


def test_generate_weights_length_mismatch_falls_back_to_baseline():
    returns = np.array([0.1, -0.1, 0.05])
    t_events = pd.date_range("2020-01-01", periods=3, freq="T")

    baseline = generate_weights_per_label(returns=returns, t_events=t_events)
    mismatched = generate_weights_per_label(
        returns=returns,
        t_events=t_events,
        consistency_scores=np.array([0.5, 0.6]),  # wrong length, should be ignored
    )

    np.testing.assert_allclose(mismatched, baseline)


def test_generate_weights_handles_nan_and_zero_vol():
    returns = np.array([np.nan, 0.0, 0.02])
    t_events = pd.date_range("2020-01-01", periods=3, freq="T")
    vol = np.array([0.0, 0.0, 0.0])

    weights = generate_weights_per_label(
        returns=returns, t_events=t_events, vol_proxy=vol
    )

    assert len(weights) == len(returns)
    assert np.isfinite(weights).all()
    assert weights.mean() == pytest.approx(1.0, rel=1e-6)


def test_downside_multiplier_preserved_after_normalization():
    returns = np.array([0.01, -0.01])
    t_events = pd.date_range("2020-01-01", periods=2, freq="T")
    vol = np.array([0.01, 0.02])

    weights = generate_weights_per_label(
        returns=returns,
        t_events=t_events,
        vol_proxy=vol,
        downside_multiplier=2.0,
        mag_compression=1.0,
        exp_cross=1.0,
    )

    assert weights.mean() == pytest.approx(1.0, rel=1e-6)
    assert weights[1] < weights[0]

