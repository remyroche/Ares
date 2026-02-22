import numpy as np
import pandas as pd
# import pytest
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.volume_node_features import hvn_lvn_features_ohlcv

class TestNewIndicators:
    def test_ker(self):
        # Create a simple trend
        close = pd.Series(np.arange(100, dtype=float))
        # KER(10) = (99-89) / sum(|1|) = 10 / 10 = 1.0 (perfect efficiency)
        ker = ff.numba_ker(close, 10)
        assert np.isclose(ker.iloc[-1], 1.0)

        # Create a noisy flat line
        close_noisy = pd.Series([100, 101] * 50, dtype=float)
        # N=4.
        # Change(4) = |100-100| = 0.
        # Sum(|Change(1)|) over 4 bars = 4.
        # KER = 0 / 4 = 0.
        ker_noisy = ff.numba_ker(close_noisy, 4)
        # Check even index > 4 (where price is 100)
        assert np.isclose(ker_noisy.iloc[10], 0.0)

    def test_vortex(self):
        # Uptrend
        high = pd.Series(np.arange(100, dtype=float) + 1)
        low = pd.Series(np.arange(100, dtype=float) - 1)
        close = pd.Series(np.arange(100, dtype=float))

        vi_diff = ff.numba_vortex(high, low, close, 14)
        # In a perfect uptrend, VI+ should be high, VI- low. Diff > 0.
        assert vi_diff.iloc[-1] > 0

    def test_adx(self):
        high = pd.Series(np.arange(100, dtype=float) + 1)
        low = pd.Series(np.arange(100, dtype=float) - 1)
        close = pd.Series(np.arange(100, dtype=float))

        adx, dip, dim = ff.numba_adx(high, low, close, 14)

        # Perfect uptrend: DI+ > DI-, ADX increasing
        assert dip.iloc[-1] > dim.iloc[-1]
        assert adx.iloc[-1] > 50 # Strong trend

    def test_hvn_lvn(self):
        # Create synthetic OHLCV
        idx = pd.date_range("2024-01-01", periods=200, freq="1h")
        close = np.sin(np.linspace(0, 10, 200)) * 100 + 1000
        df = pd.DataFrame({
            "open": close,
            "high": close + 5,
            "low": close - 5,
            "close": close,
            "volume": np.random.rand(200) * 1000 + 100
        }, index=idx)

        # Run function
        feats = hvn_lvn_features_ohlcv(df, vp_lookback=24, vp_bins=10)

        assert not feats.empty
        assert "dist_poc_atr" in feats.columns
        assert not feats["dist_poc_atr"].isnull().all()
