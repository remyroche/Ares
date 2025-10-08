import pandas as pd
import numpy as np
import pytest

from src.utils.common_ml.backtesting.turnover import (
    calculate_turnover_metrics,
    apply_market_impact_model,
    reject_high_turnover_configs,
)


def test_calculate_turnover_metrics_basic():
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    positions = pd.Series([0, 1, 1, 0, -1, -1], index=dates, dtype=float)
    returns = pd.Series([0.0, 0.01, -0.005, 0.002, -0.001, 0.0], index=dates)

    metrics = calculate_turnover_metrics(positions, returns)

    assert metrics["turnover_per_period"] == pytest.approx(0.5)
    assert metrics["turnover_annual"] == pytest.approx(126.0)
    assert metrics["avg_holding_period_bars"] == pytest.approx(2.0)
    assert metrics["position_stability"] == pytest.approx(0.5)


def test_apply_market_impact_model_square_root_and_clip():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    returns = pd.Series([0.02, 0.01, -0.005], index=dates)
    positions = pd.Series([0, 100, 50], index=dates, dtype=float)
    volume = pd.Series([1_000_000, 1_000_000, 1_000_000], index=dates, dtype=float)

    adjusted = apply_market_impact_model(returns, positions, volume, impact_coefficient=0.1, max_impact=0.01)

    expected = pd.Series(
        [0.02, 0.009, -0.00570710678],
        index=dates
    )
    assert np.allclose(adjusted.values, expected.values, atol=1e-9)

    # Verify clipping at max impact
    large_positions = pd.Series([0, 2_000_000, 0], index=dates, dtype=float)
    tight_volume = pd.Series([100_000, 100_000, 100_000], index=dates, dtype=float)
    clipped = apply_market_impact_model(
        returns,
        large_positions,
        tight_volume,
        impact_coefficient=0.5,
        max_impact=0.01,
    )
    assert clipped.iloc[1] == pytest.approx(0.0)
    assert clipped.iloc[2] == pytest.approx(-0.015)


def test_reject_high_turnover_configs():
    assert reject_high_turnover_configs({"turnover_annual": 100.0, "sharpe_ratio": 0.5}, max_turnover_annual=50.0)
    assert reject_high_turnover_configs(
        {"turnover_annual": 20.0, "sharpe_ratio": 1.0},
        max_turnover_annual=50.0,
        max_sharpe_to_turnover_ratio=0.2,
    )
    assert not reject_high_turnover_configs(
        {"turnover_annual": 10.0, "sharpe_ratio": 3.0},
        max_turnover_annual=50.0,
        max_sharpe_to_turnover_ratio=0.2,
    )
