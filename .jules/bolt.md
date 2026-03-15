## 2026-03-04 - Parallelizing Numba Rolling VWAP and Fallbacks
Learning: Iterating over DataFrame columns in Python using a Numba JIT function (`ff._numba_rolling_vwap`) inside a loop incurs massive overhead. Furthermore, when parallelizing this function with `prange` inside `_numba_rolling_vwap_parallel`, it is even faster to inline the 1D logic directly into the 2D loop rather than calling the 1D Numba function inside the parallel loop.
Action: Implement `_numba_rolling_vwap_parallel` with the logic fully inlined within the 2D `prange` loop. Crucially, when updating Numba-related fallback imports in `src_utils_numba_funcs.py` for environments without Numba, ensure `prange = range` is set in the `except ImportError` block to prevent `NameError` exceptions.

## 2026-03-04 - Vectorized Run-Length Encoding for Regimes
Learning: Calculating run-length encoding across a DataFrame's columns using `groupby().cumcount()` on each column individually is extremely slow due to Python looping overhead.
Action: A vectorized approach using NumPy's `np.maximum.accumulate` to track the last change index along axis 0 is significantly faster. Instead of grouping, subtract the row index from the last index where a regime change occurred.

## 2026-03-04 - JIT Rolling Argmax Optimization
Learning: `pd.rolling(window).apply(np.argmax)` inside a loop over DataFrame columns is phenomenally slow due to the overhead of setting up thousands of Pandas Series operations.
Action: Replacing this logic with a custom parallelized Numba kernel (`_numba_rolling_bars_since_extreme_parallel`) that computes the distance from the last peak/trough reduced execution time for a 100k x 50 dataset from 22.6s to 0.12s (~190x speedup). Always favor custom JIT kernels over Pandas `apply` in performance-critical paths.

## 2026-03-04 - Fractionally Differentiated Features Parallelization
Learning: Fractional differentiation `_numba_apply_weights` was iteratively applied column-by-column in a Python loop in `features.py`. The convolution approach can be significantly sped up across columns using `@jit(parallel=True)`.
Action: Vectorize `_numba_apply_weights` across columns with `prange` into `_numba_apply_weights_parallel`. Group columns by their calculated `d_use` value and apply `frac_diff_ffd` natively on DataFrame chunks rather than processing column by column.

## 2026-03-04 - ProcessPoolExecutor for Independent Columns
Learning: Functions like `hvn_lvn_features_ohlcv` process single-asset (column-wise) data using NumPy `sliding_window_view` and `np.digitize`. When applied sequentially across a wide panel, they create massive bottlenecks (e.g. 52 seconds for 50 cols). Because they don't share state and release the GIL effectively within NumPy, they are prime candidates for multiprocessing.
Action: Used `concurrent.futures.ProcessPoolExecutor` to farm out column-level work. Using `min(8, multiprocessing.cpu_count())` workers yielded a 3.1x speedup (52.2s -> 16.7s).

## 2026-03-04 - Vectorized On Balance Volume (OBV)
Learning: Iterating over DataFrame rows sequentially using `for i in range(1, len(data)):` and `data['close'].iloc[i] > data['close'].iloc[i - 1]` to calculate running totals like On Balance Volume is extremely slow in Python (taking over 19 seconds for 100k rows).
Action: Replaced the iterative loop with a fully vectorized approach using `np.sign(data['close'].diff())` to determine direction and `vol_adj.cumsum()` to accumulate. This reduced execution time to ~0.012 seconds (over 1500x speedup). Always prefer vectorized operations (`diff`, `sign`, `cumsum`) over sequential row iteration.

## 2026-03-04 - Parallelizing Numba Rolling Functions (Mean, Std)
Learning: Even when using a fast 1D Numba JIT compiled function via `apply_to_matrix(df, func)`, looping sequentially over each column sequentially in Python adds considerable overhead for large DataFrames.
Action: Rewrote core rolling window functions (`_numba_rolling_std_nan_safe` and `_numba_rolling_mean_nan_safe`) as 2D functions running natively across all columns simultaneously using Numba's `@jit(parallel=True)` and `prange`. Updating `apply_to_frame` to intercept and delegate to this new parallel approach provided a transparent optimization with massive speedup for feature generation pipelines without altering any logic in features.py.

## 2026-03-05 - Vectorizing complex Series.rolling().apply() calls
Learning: Using `pd.Series.rolling(window).apply(func, raw=True)` is incredibly slow when `func` relies on custom Python logic or NumPy operations like `polyfit`, since it evaluates the Python function once per row.
Action: Replace `.rolling().apply()` with either pure vectorized Pandas rolling operations (e.g. counting positives and computing proportions using `.sum()` over differences, which was used to optimize `_entropy`) or optimized Numba loops exposed in `fast_funcs.py` (e.g. `numba_rolling_rank_pct` and `apply_to_frame(..., _numba_rolling_slope, ...)`). This avoids loop overhead entirely and pushes computation to optimized C/C++ backends.
