
import numpy as np
import pandas as pd
import pytest
from extreme_price_movements.fast_funcs import numba_pick_by_regime

def pick_by_rv_baseline(fast_df, base_df, slow_df, rv_ratio, fast_thr, slow_thr):
    rr = rv_ratio
    out = base_df.copy()
    out = out.where(~(rr > fast_thr), fast_df)
    out = out.where(~(rr < slow_thr), slow_df)
    return out.astype(np.float32)

def test_numba_pick_by_regime_correctness():
    np.random.seed(42)
    rows = 100
    cols = 10

    index = pd.RangeIndex(rows)
    columns = [f"col_{i}" for i in range(cols)]

    fast_df = pd.DataFrame(np.random.randn(rows, cols).astype(np.float32), index=index, columns=columns)
    base_df = pd.DataFrame(np.random.randn(rows, cols).astype(np.float32), index=index, columns=columns)
    slow_df = pd.DataFrame(np.random.randn(rows, cols).astype(np.float32), index=index, columns=columns)
    rv_ratio = pd.DataFrame(np.random.uniform(0.5, 1.5, size=(rows, cols)).astype(np.float32), index=index, columns=columns)

    # Introduce NaNs in regime
    rv_ratio.iloc[0:10, 0:5] = np.nan

    # Introduce NaNs in data
    fast_df.iloc[10:20, 0:5] = np.nan
    base_df.iloc[20:30, 0:5] = np.nan
    slow_df.iloc[30:40, 0:5] = np.nan

    fast_thr = 1.2
    slow_thr = 0.8

    expected = pick_by_rv_baseline(fast_df, base_df, slow_df, rv_ratio, fast_thr, slow_thr)
    actual = numba_pick_by_regime(fast_df, base_df, slow_df, rv_ratio, fast_thr, slow_thr)

    pd.testing.assert_frame_equal(expected, actual)

def test_numba_pick_by_regime_misaligned():
    # Test strict alignment enforcement
    rows = 50
    cols = 5
    index = pd.RangeIndex(rows)
    columns = [f"c_{i}" for i in range(cols)]

    # Create master aligned
    fast_df = pd.DataFrame(np.random.randn(rows, cols).astype(np.float32), index=index, columns=columns)
    base_df = pd.DataFrame(np.random.randn(rows, cols).astype(np.float32), index=index, columns=columns)
    slow_df = pd.DataFrame(np.random.randn(rows, cols).astype(np.float32), index=index, columns=columns)
    rv_ratio = pd.DataFrame(np.random.uniform(0.5, 1.5, size=(rows, cols)).astype(np.float32), index=index, columns=columns)

    # Misalign fast_df (shift index)
    # fast_df has index 0..49
    # fast_shifted has index 1..50
    fast_shifted = fast_df.shift(1).dropna() # 1..49 (index 0 gone)
    fast_shifted.index = fast_shifted.index + 1 # 2..50
    # Actually just reindex is cleaner

    # Let's say base_df is 0..49.
    # fast_df only has 10..59.
    fast_df_mis = pd.DataFrame(np.random.randn(rows, cols).astype(np.float32), index=pd.RangeIndex(10, 60), columns=columns)

    fast_thr = 1.2
    slow_thr = 0.8

    # Baseline handles alignment automatically (anchored to base_df)
    expected = pick_by_rv_baseline(fast_df_mis, base_df, slow_df, rv_ratio, fast_thr, slow_thr)

    # Optimized should also handle alignment (anchored to base_df)
    actual = numba_pick_by_regime(fast_df_mis, base_df, slow_df, rv_ratio, fast_thr, slow_thr)

    # Check alignment
    pd.testing.assert_frame_equal(expected, actual)

if __name__ == "__main__":
    test_numba_pick_by_regime_correctness()
    test_numba_pick_by_regime_misaligned()
