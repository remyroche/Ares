from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.causal_exotic_dynamics import (
    FEATURE_COLUMNS,
    feature_metadata_frame,
    materialize_symbol,
)


def _bars(count: int = 420) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC")
    rng = np.random.default_rng(1729)
    returns = rng.normal(0.0, 0.001, count)
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "open": close * .9998,
        "high": close * 1.001,
        "low": close * .999,
        "close": close,
        "volume": rng.lognormal(4.0, .4, count),
    }, index=index)


def test_metadata_covers_exact_feature_contract() -> None:
    metadata = feature_metadata_frame()
    assert metadata.feature_name.tolist() == list(FEATURE_COLUMNS)
    assert metadata.family.nunique() == 5
    assert metadata.minimum_support.ge(32).all()
    assert metadata.causal_availability.eq(
        "last_completed_15m_bar_strictly_before_decision"
    ).all()


def test_future_bar_perturbation_cannot_change_prior_dynamic_state() -> None:
    bars = _bars()
    decisions = pd.DatetimeIndex([
        bars.index[320] + pd.Timedelta(minutes=15),
        bars.index[360] + pd.Timedelta(minutes=15),
        bars.index[400] + pd.Timedelta(minutes=15),
    ])
    baseline = materialize_symbol(bars, decisions)
    changed = bars.copy()
    # Alter only bars strictly after the second decision.  The first two
    # feature rows must remain bit-identical (apart from harmless NaN form).
    changed.loc[changed.index > decisions[1], ["open", "high", "low", "close"]] *= 1.7
    changed.loc[changed.index > decisions[1], "volume"] *= 20.0
    compared = materialize_symbol(changed, decisions)
    for field in FEATURE_COLUMNS:
        left = baseline.loc[:1, field].to_numpy(float)
        right = compared.loc[:1, field].to_numpy(float)
        assert np.allclose(left, right, atol=0.0, rtol=0.0, equal_nan=True), field


def test_decision_uses_strictly_prior_completed_bar() -> None:
    bars = _bars()
    decision = bars.index[350]
    baseline = materialize_symbol(bars, pd.DatetimeIndex([decision]))
    changed = bars.copy()
    # This is the bar opening exactly at decision and must not be consumed.
    changed.loc[decision, ["open", "high", "low", "close"]] *= .1
    changed.loc[decision, "volume"] *= 100.0
    compared = materialize_symbol(changed, pd.DatetimeIndex([decision]))
    for field in FEATURE_COLUMNS:
        assert np.allclose(
            baseline[field].to_numpy(float), compared[field].to_numpy(float),
            atol=0.0, rtol=0.0, equal_nan=True,
        ), field


def test_extended_horizon_contract_is_present_and_prior_only() -> None:
    required = {
        "cp_price_return_24h_atr", "cp_price_page_hinkley_score",
        "sp_return_low_power_share_72h", "sp_return_peak_frequency_8h",
        "wv_return_energy_long_72h", "en_return_sign_entropy_72h",
        "ds_return_std_72h", "ds_return_shift_24h_72h",
    }
    assert required.issubset(FEATURE_COLUMNS)
    bars = _bars(420)
    decision = bars.index[360] + pd.Timedelta(minutes=15)
    baseline = materialize_symbol(bars, pd.DatetimeIndex([decision]))
    altered = bars.copy()
    altered.loc[altered.index > decision, ["open", "high", "low", "close"]] *= 1.5
    compared = materialize_symbol(altered, pd.DatetimeIndex([decision]))
    for field in required:
        assert np.allclose(
            baseline[field].to_numpy(float), compared[field].to_numpy(float),
            atol=0.0, rtol=0.0, equal_nan=True,
        ), field
