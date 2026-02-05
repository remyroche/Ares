## 2026-01-14 - Layer 2 Optimization: Vectorization & JIT Compilation

**Learning:** Python loops for feature engineering and parameter search are significant bottlenecks in the Layer 2 pipeline. By replacing these with vectorized NumPy operations and Numba JIT-compiled functions, we achieved substantial speedups:
- Feature Selection: 8x faster (5.7s vs ~45s baseline estimate)
- Geometry Search: 7.5x faster (0.98s vs ~7.5s baseline estimate)
- Feature Engineering: 3-5x faster (5.5s vs ~20s baseline estimate)

**Action:** Whenever iterating over large DataFrames or performing grid searches, prioritize vectorization first, then Numba JIT.

**Key Optimizations Implemented:**
1. **Vectorized Feature Selection:** Replaced loop-based correlation/variance calculation with `njit(parallel=True)` implementations.
2. **Vectorized Geometry Search:** Flattened 2D parameter grids into 1D arrays for single-pass JIT-compiled performance scoring.
3. **JIT Feature Engineering:** Implemented rolling mean/std/range and lag features using Numba to bypass Pandas overhead for large datasets.

## 2026-01-19 - Rolling Sum vs Diff Edge Case

**Learning:** When replacing `log_returns.rolling(w).sum()` with `log_price.diff(w)` for performance (80x+ speedup), `diff(w)` yields 1 fewer valid sample at the beginning of the series compared to `rolling(w, min_periods=w)` if `log_returns` was 0-filled at the start. `diff` is strictly structural, while `rolling` includes the 0-filled start in its valid count.

**Action:** Accept the minor data loss for substantial speedups, but verify downstream tests don't rely on the exact count of valid initial samples.

## 2026-01-19 - Rolling Skew/Kurtosis Optimization

**Learning:** Pandas `rolling().skew()` and `.kurt()` are reasonably optimized (likely Cython) and beat naive Numba window-recalculation ($O(N \cdot W)$). However, an online algorithm (Welford's / Sum of Powers) implemented in Numba ($O(N)$) beats Pandas by 4x (Skew) to 12x (Kurtosis).

**Action:** For higher-moment rolling statistics, use Numba-optimized online algorithms, but be careful with numerical stability (ensure data is centered or small like returns) and NaN handling (online algorithms propagate NaNs aggressively). Passing 0-filled data is essential for robustness here.

## 2026-01-20 - Rolling Streak Persistence Optimization

**Learning:** The previous implementation of `_numba_streak_persistence` had $O(N \cdot W)$ complexity because it recalculated streaks for the entire window at every step. This led to linear performance degradation as the window size increased (e.g., 0.27s for window=1000 vs 0.04s for window=100).

**Action:** Implemented an $O(N)$ online algorithm using a circular buffer to incrementally track streaks by handling "leaving" and "entering" elements. This reduced execution time for window=1000 from ~0.27s to ~0.003s (a ~90x speedup), making performance independent of window size. Always look for incremental update opportunities in sliding window calculations.

## 2026-01-20 - Entropy Feature Discrepancy & Optimization

**Learning:** A critical logic discrepancy was found where the optimized Numba path calculated entropy on *prices* (non-stationary) while the Pandas fallback used *returns* (stationary). This invalidated the feature definition. Furthermore, `lempel_ziv_complexity_numba` used an $O(N^2)$ algorithm (Kaspar-Schuster) which is prohibitive for large datasets (N>100k).

**Action:** Always verify that optimized implementations (Numba/Cython) receive the same transformed input (e.g., returns) as the reference implementation. For quadratic algorithms like LZ complexity, enforce a `max_lookback` (e.g., 5000) to cap complexity at $O(N \cdot K)$, trading infinite memory for linear performance scaling.

## 2026-01-23 - Entropy Statistics & Volatility Loop Optimization

**Learning:** `calculate_entropy_statistics_numba` was using a naive sliding window loop recalculating `np.mean` and `np.std` at every step, leading to $O(N \cdot W)$ complexity. Additionally, `vectorized_entropy_features` used a pure Python loop for rolling volatility, which was a massive bottleneck (10s execution time vs <1s optimized).
Also, `fastmath=True` in Numba functions can break `np.isfinite` checks for NaNs in certain environments, leading to incorrect results (e.g. Entropy=0.0).

**Action:**
1. Replaced naive window loops with O(N) online rolling statistics (`_numba_rolling_mean`, `_numba_rolling_std`).
2. Replaced Python loops with Numba-optimized equivalents.
3. Disabled `fastmath` for functions requiring robust NaN handling (`shannon_entropy_numba`).
Result: `calculate_entropy_statistics_numba` speedup >700x (2.11s -> 0.003s). Full feature generation speedup ~10x (10s -> 0.8s).

## 2026-01-27 - Rolling VWAP Optimization

**Learning:** Pandas `rolling()` operations are flexible but can be slow for composite metrics like VWAP (`sum(pv)/sum(v)`), which require two rolling aggregations and intermediate Series. A Numba $O(N)$ implementation using single-pass sliding window sums achieved a ~6.5x speedup. Careful handling of `min_periods=1` logic (accumulating until window is full) and NaN propagation (ignoring missing values in sums, but invalidating result if volume is zero) was required to match Pandas exactly.

**Action:** For composite rolling metrics (e.g., VWAP, correlation), implement single-pass O(N) Numba functions instead of chaining multiple Pandas rolling calls. Verify `min_periods` and NaN behavior against the reference implementation.

## 2026-02-02 - Rolling Price Jump Frequency Optimization

**Learning:** `_numba_price_jump_frequency` was O(N*W) because it recalculated mean and std for the window at every step using `np.mean` and `np.std`. Even with JIT, this is inefficient. Replacing these with incremental O(1) updates (using Welford's or sum/sq_sum tracking) reduced complexity to O(N * (W/k))—we still iterate to count jumps, but the expensive mean/std calculation is gone. This achieved a ~2.7x speedup for typical window sizes.

**Action:** When implementing rolling features that require window statistics (mean, std) for thresholding, always maintain these statistics incrementally (O(1)) rather than recalculating them (O(W)), even if the subsequent logic (like counting outliers) requires iterating the window. Every O(W) reduction inside the main loop counts.

## 2026-02-04 - Rolling Mean/Std Nan-Safe Optimization

**Learning:** Naive sliding window implementations for "nan-safe" rolling statistics ($O(N \cdot W)$) are extremely slow for large windows.
- `_numba_rolling_mean_nan_safe`: Was scanning the window for every element to skip NaNs. Optimized to $O(N)$ by tracking `sum` and `count` incrementally. Speedup: ~10x (2.8s -> 0.28s for N=1M, W=1000).
- `_numba_rolling_std_nan_safe`: Was doing TWO passes over the window (Mean then Variance). Optimized to $O(N)$ using online variance update (tracking `sum`, `sum_sq`, `count`). Speedup: ~230x (4.25s -> 0.018s).

**Action:** Always prefer $O(N)$ incremental updates for rolling statistics, even when handling NaNs requires conditional logic. The speedup is massive for W > 100.

## 2026-02-05 - Rolling Min/Max Optimization

**Learning:** `_numba_rolling_max` and `min` were implemented using a naive $O(N \cdot W)$ sliding window loop. This scales linearly with window size and is very slow for large windows (e.g. 463x slower for W=5000). Replacing this with a monotonic deque implementation reduces complexity to amortized $O(N)$, resulting in constant time execution regardless of window size.

**Action:** Always use monotonic deques for rolling min/max operations instead of naive iteration over the window.

## 2026-02-05 - Rolling Quantile Partitioning vs Sorting

**Learning:** For rolling quantile (and median) calculations in Numba where $O(N)$ online update algorithms (like Two Heaps) are complex to implement correctly with NaNs:
-   `Reuse Buffer + Sort` (sorting the window buffer every step) is $O(N \cdot W \log W)$. It is slow for large W.
-   `Reuse Buffer + Partition` (using `np.partition` for finding kth element) is $O(N \cdot W)$ for each quantile.
-   Even though `np.partition` creates a copy in Numba (currently), it outperformed Allocation+Sort by ~4.5x for W=1000.
-   Reusing the buffer for collection avoids $N$ allocations, which is a massive win for GC and speed.

**Action:** Use `np.partition` instead of `sort` for rolling quantile calculations if exact sorted order isn't required (only k-th element). Ensure to reuse the collection buffer to minimize allocation overhead.
