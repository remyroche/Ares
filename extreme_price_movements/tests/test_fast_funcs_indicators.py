
import numpy as np
import pandas as pd
import pytest
from extreme_price_movements.fast_funcs import numba_rsi, numba_atr, numba_atr_no_norm

def test_numba_rsi_accuracy():
    np.random.seed(42)
    rows = 100
    cols = 2
    data = np.random.randn(rows, cols).cumsum(axis=0) + 100
    df = pd.DataFrame(data, columns=['A', 'B'])

    n = 14
    rsi = numba_rsi(df, n)

    # Simple check: RSI should be between 0 and 100
    assert rsi.min().min() >= 0
    assert rsi.max().max() <= 100

    # Check against pandas_ta or manual calculation if possible.
    # Manual calc for verification:
    delta = df.diff()
    up = delta.clip(lower=0)
    dn = -1 * delta.clip(upper=0)

    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()

    rs = ma_up / ma_dn
    expected_rsi = 100 - (100 / (1 + rs))

    # First n values might differ slightly due to initialization or be NaN
    # Numba implementation usually replicates pandas behavior

    # Fillna to compare
    rsi_filled = rsi.fillna(0)
    expected_filled = expected_rsi.fillna(0).astype(np.float32)

    # Relax tolerance slightly for floating point diffs
    pd.testing.assert_frame_equal(rsi_filled.iloc[n:], expected_filled.iloc[n:], atol=1e-4)

def test_numba_atr_accuracy():
    np.random.seed(42)
    rows = 100
    cols = 2
    close = pd.DataFrame(np.random.randn(rows, cols).cumsum(axis=0) + 100, columns=['A', 'B'])
    high = close + np.abs(np.random.randn(rows, cols))
    low = close - np.abs(np.random.randn(rows, cols))

    n = 14
    atr = numba_atr(high, low, close, n)

    # Manual calc
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1) # This logic is flawed for multi-column, need per-column

    # Correct manual loop
    expected_atr_list = []
    for c in close.columns:
        h_s = high[c]
        l_s = low[c]
        c_s = close[c]
        pc_s = c_s.shift(1)

        tr_s = pd.concat([h_s - l_s, (h_s - pc_s).abs(), (l_s - pc_s).abs()], axis=1).max(axis=1)
        # First TR is H-L
        tr_s.iloc[0] = h_s.iloc[0] - l_s.iloc[0]

        atr_s = tr_s.ewm(alpha=1/n, adjust=False).mean()
        atr_pct_s = atr_s / c_s
        expected_atr_list.append(atr_pct_s)

    expected_atr = pd.concat(expected_atr_list, axis=1)
    expected_atr.columns = close.columns # fix names

    pd.testing.assert_frame_equal(atr, expected_atr.astype(np.float32), atol=1e-4)

def test_numba_atr_no_norm_accuracy():
    np.random.seed(42)
    rows = 100
    cols = 2
    close = pd.DataFrame(np.random.randn(rows, cols).cumsum(axis=0) + 100, columns=['A', 'B'])
    high = close + np.abs(np.random.randn(rows, cols))
    low = close - np.abs(np.random.randn(rows, cols))

    n = 14
    atr_nn = numba_atr_no_norm(high, low, close, n)

    # Correct manual loop
    expected_atr_list = []
    for c in close.columns:
        h_s = high[c]
        l_s = low[c]
        c_s = close[c]
        pc_s = c_s.shift(1)

        tr_s = pd.concat([h_s - l_s, (h_s - pc_s).abs(), (l_s - pc_s).abs()], axis=1).max(axis=1)
        # First TR is H-L
        tr_s.iloc[0] = h_s.iloc[0] - l_s.iloc[0]

        atr_s = tr_s.ewm(alpha=1/n, adjust=False).mean()
        # No normalization
        expected_atr_list.append(atr_s)

    expected_atr = pd.concat(expected_atr_list, axis=1)
    expected_atr.columns = close.columns

    pd.testing.assert_frame_equal(atr_nn, expected_atr.astype(np.float32), atol=1e-4)

if __name__ == "__main__":
    test_numba_rsi_accuracy()
    test_numba_atr_accuracy()
    test_numba_atr_no_norm_accuracy()
