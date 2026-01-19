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
