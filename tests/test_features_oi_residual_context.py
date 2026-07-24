import numpy as np
import pandas as pd

from extreme_price_movements.features_negative_residuals import (
    add_negative_residual_features,
    compute_short_default_mechanism_context,
)
from extreme_price_movements.features_oi import (
    compute_oi_features,
    compute_residual_market_context_oi_features,
)
from extreme_price_movements.features_residual import add_residual_features


def _panel(rows: int = 900, columns: int = 4):
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    names = ["BTC/USD:USD", "ETH/USD:USD", "AAVE/USD:USD", "SOL/USD:USD"][:columns]
    returns = rng.normal(0.0, 0.01, size=(rows, columns)).astype(np.float32)
    price = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=names)
    oi = pd.DataFrame(1_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, size=(rows, columns)), axis=0)), index=index, columns=names)
    volume = pd.DataFrame(rng.uniform(100.0, 10_000.0, size=(rows, columns)), index=index, columns=names)
    funding = pd.DataFrame(rng.normal(0.0, 0.0001, size=(rows, columns)), index=index, columns=names)
    return oi, price, volume, funding


def test_narrow_residual_context_matches_full_oi_formulas():
    oi, price, volume, funding = _panel()
    full = compute_oi_features(
        oi_native=oi, price=price, quote_volume=volume, funding_rate=funding, bars_per_day=24
    )
    narrow = compute_residual_market_context_oi_features(
        oi_native=oi, price=price, quote_volume=volume, funding_rate=funding, bars_per_day=24
    )
    for name in (
        "asset_short_covering_score",
        "funding_1d_chg_z_90d",
        "price_down_oi_down_4h_rz",
    ):
        np.testing.assert_allclose(
            narrow[name].to_numpy(), full[name].to_numpy(), rtol=1e-5, atol=1e-5, equal_nan=True
        )


def test_narrow_context_retains_residual_dependency():
    oi, price, volume, funding = _panel()
    narrow = compute_residual_market_context_oi_features(
        oi_native=oi, price=price, quote_volume=volume, funding_rate=funding, bars_per_day=24
    )
    add_residual_features(narrow, None, {})
    assert "funding_1d_chg_ts_resid" in narrow
    assert narrow["funding_1d_chg_ts_resid"].notna().to_numpy().any()


def test_short_default_market_helper_matches_panel_composites():
    oi, price, volume, funding = _panel()
    narrow = compute_residual_market_context_oi_features(
        oi_native=oi, price=price, quote_volume=volume, funding_rate=funding, bars_per_day=24
    )
    add_residual_features(narrow, None, {})
    expected = {key: value.copy() for key, value in narrow.items()}
    add_negative_residual_features(
        expected,
        requested_feature_keys=["short_covering_score_market", "funding_confirmed_long_flush"],
        cfg={"feature_bars_per_hour": 1},
    )
    actual = compute_short_default_mechanism_context(
        asset_short_covering_score=narrow["asset_short_covering_score"],
        funding_1d_chg_ts_resid=narrow["funding_1d_chg_ts_resid"],
        price_down_oi_down_4h_rz=narrow["price_down_oi_down_4h_rz"],
    )
    for name, series in actual.items():
        np.testing.assert_allclose(
            series.to_numpy(), expected[name].iloc[:, 0].to_numpy(), rtol=1e-5, atol=1e-5, equal_nan=True
        )
