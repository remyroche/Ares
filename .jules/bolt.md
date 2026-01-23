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
