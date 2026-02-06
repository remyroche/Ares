import pandas as pd
import numpy as np
import pytest
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.config import CFG

def test_feature_causality():
    """
    Verify that features at time t do not change when data at t+1 changes.
    """
    # 1. Create dummy data
    dates = pd.date_range("2021-01-01", periods=1000, freq="h", tz="UTC")
    symbols = ["A", "B", "C"]

    np.random.seed(42)

    def make_df():
        data = np.random.randn(len(dates), len(symbols))
        # Cumulative sum to make it look like price
        price = 100 + np.cumsum(data, axis=0)
        return pd.DataFrame(price, index=dates, columns=symbols)

    close = make_df()
    open_ = close + np.random.randn(*close.shape) * 0.1
    high = np.maximum(close, open_) + np.abs(np.random.randn(*close.shape) * 0.1)
    low = np.minimum(close, open_) - np.abs(np.random.randn(*close.shape) * 0.1)
    volume = np.abs(np.random.randn(*close.shape) * 1000)

    panel = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": pd.DataFrame(volume, index=dates, columns=symbols)
    }

    # 2. Compute baseline features
    cfg = CFG.copy()
    cfg["market_basket"] = symbols

    # Force re-computation (bypass cache if possible, though tests usually run in clean env)
    # The cache relies on data hash, so different data will trigger recompute.

    mkt_base = compute_market_features(panel, symbols)
    mkt_gates_base = add_regime_gates(mkt_base, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    feats_base = compute_features_hourly(panel, mkt_gates_base, cfg)

    # 3. Modify data at t+1
    # Pick a target time t
    t_idx = 500
    t = dates[t_idx]
    t_next = dates[t_idx+1]

    # Modify data AT t+1 (and beyond)
    # We add a huge spike at t+1
    shock = 1000.0

    panel_mod = {k: v.copy() for k, v in panel.items()}
    for k in panel_mod:
        # Modify specifically at t_next
        panel_mod[k].loc[t_next, "A"] += shock

    # 4. Recompute features
    mkt_mod = compute_market_features(panel_mod, symbols)
    mkt_gates_mod = add_regime_gates(mkt_mod, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    feats_mod = compute_features_hourly(panel_mod, mkt_gates_mod, cfg)

    # 5. Verify equality at t
    # Check Market Features
    row_base = mkt_base.loc[t]
    row_mod = mkt_mod.loc[t]

    pd.testing.assert_series_equal(row_base, row_mod, obj="Market Features at t")

    # Check Market Gates
    row_gate_base = mkt_gates_base.loc[t]
    row_gate_mod = mkt_gates_mod.loc[t]
    pd.testing.assert_series_equal(row_gate_base, row_gate_mod, obj="Market Gates at t")

    # Check Hourly Features
    # Pick a feature, e.g. ret24h, atr_pct
    for key in feats_base:
        if key in ["sin_hod", "cos_hod", "sin_dow", "cos_dow"]: continue

        # Check for symbol A (which was modified at t+1)
        val_base = feats_base[key].loc[t, "A"]
        val_mod = feats_mod[key].loc[t, "A"]

        # If val_base is NaN, val_mod should be NaN
        if np.isnan(val_base):
            assert np.isnan(val_mod), f"Feature {key} changed from NaN to {val_mod} at {t}"
        else:
            assert np.isclose(val_base, val_mod, rtol=1e-5), f"Feature {key} changed at {t} (Base: {val_base}, Mod: {val_mod})"

    print("Causality test passed!")

if __name__ == "__main__":
    test_feature_causality()
