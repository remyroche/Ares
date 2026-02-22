#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.volume_node_features import hvn_lvn_features_ohlcv

def test_ker():
    print("Testing KER...")
    # Create a simple trend
    close = pd.Series(np.arange(100, dtype=float))
    # KER(10) = (99-89) / sum(|1|) = 10 / 10 = 1.0 (perfect efficiency)
    ker = ff.numba_ker(close, 10)
    assert np.isclose(ker.iloc[-1], 1.0), f"Expected 1.0, got {ker.iloc[-1]}"
    print("✓ KER perfect trend test passed")

    # Create a noisy flat line
    close_noisy = pd.Series([100, 101] * 50, dtype=float)
    # N=4.
    # Change(4) = |100-100| = 0.
    # Sum(|Change(1)|) over 4 bars = 4.
    # KER = 0 / 4 = 0.
    ker_noisy = ff.numba_ker(close_noisy, 4)
    # Check even index > 4 (where price is 100)
    assert np.isclose(ker_noisy.iloc[10], 0.0), f"Expected 0.0, got {ker_noisy.iloc[10]}"
    print("✓ KER noisy flat line test passed")

def test_vortex():
    print("Testing Vortex...")
    # Uptrend
    high = pd.Series(np.arange(100, dtype=float) + 1)
    low = pd.Series(np.arange(100, dtype=float) - 1)
    close = pd.Series(np.arange(100, dtype=float))

    vi_diff = ff.numba_vortex(high, low, close, 14)
    # In a perfect uptrend, VI+ should be high, VI- low. Diff > 0.
    assert vi_diff.iloc[-1] > 0, f"Expected > 0, got {vi_diff.iloc[-1]}"
    print("✓ Vortex uptrend test passed")

def test_adx():
    print("Testing ADX...")
    high = pd.Series(np.arange(100, dtype=float) + 1)
    low = pd.Series(np.arange(100, dtype=float) - 1)
    close = pd.Series(np.arange(100, dtype=float))

    adx, dip, dim = ff.numba_adx(high, low, close, 14)

    # Perfect uptrend: DI+ > DI-, ADX increasing
    assert dip.iloc[-1] > dim.iloc[-1], f"DI+ should be > DI-, got DI+={dip.iloc[-1]}, DI-={dim.iloc[-1]}"
    assert adx.iloc[-1] > 50, f"ADX should be > 50 in strong trend, got {adx.iloc[-1]}"
    print("✓ ADX perfect uptrend test passed")

def test_hvn_lvn():
    print("Testing HVN/LVN...")
    # Create synthetic OHLCV
    idx = pd.date_range("2024-01-01", periods=200, freq="1h")
    close = np.sin(np.linspace(0, 10, 200)) * 100 + 1000
    high = close + np.random.uniform(0, 5, 200)
    low = close - np.random.uniform(0, 5, 200)
    open_price = np.random.uniform(low, high, 200)
    volume = np.random.uniform(1000, 5000, 200)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=idx)
    
    # Test HVN/LVN feature generation
    features = hvn_lvn_features_ohlcv(df, vp_lookback=50, vp_bins=15)
    
    # Check that we get some features
    assert len(features.columns) > 0, "No features generated"
    assert len(features) == len(df), "Feature length mismatch"
    
    # Check that features are not all NaN
    for col in features.columns:
        assert not features[col].isna().all(), f"All NaN values in {col}"
    
    print(f"✓ HVN/LVN test passed - generated {len(features.columns)} features")

if __name__ == "__main__":
    print("Running new indicators tests...")
    
    try:
        test_ker()
        test_vortex()
        test_adx()
        test_hvn_lvn()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
