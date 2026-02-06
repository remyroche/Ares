
import numpy as np
import pandas as pd
import pytest
from extreme_price_movements.fast_funcs import numba_grouped_rolling_mean

def test_numba_grouped_rolling_mean_accuracy():
    np.random.seed(42)
    rows = 200
    cols = 3
    data = pd.DataFrame(np.random.randn(rows, cols).astype(np.float32), columns=['A', 'B', 'C'])

    # 4 groups repeating
    groups = pd.Series(np.arange(rows) % 4)

    window = 10

    # Optimized implementation result
    res = numba_grouped_rolling_mean(data, groups, window)

    # Check correctness against naive Pandas GroupBy Rolling
    # Note: Pandas groupby rolling is tricky with multi-columns and index alignment.
    # We simulate it manually per group and column.

    expected = pd.DataFrame(index=data.index, columns=data.columns, dtype=np.float32)

    for g in range(4):
        # Get indices for this group
        indices = groups[groups == g].index

        # Extract subset
        subset = data.loc[indices]

        # Calculate rolling mean on subset
        # min_periods=1 because Numba impl currently starts outputting as soon as it has values?
        # Check logic: "if current_count > 0: output[i] = current_sum / current_count"
        # Yes, min_periods=1.

        # Important: The optimized kernel treats indices as a sequence.
        # "subset" rolling means row i of subset corresponds to i-th occurrence of group g.

        for col in data.columns:
            rolled = subset[col].rolling(window, min_periods=1).mean()
            expected.loc[indices, col] = rolled.astype(np.float32)

    # Verify
    pd.testing.assert_frame_equal(res, expected, atol=1e-5)

if __name__ == "__main__":
    test_numba_grouped_rolling_mean_accuracy()
