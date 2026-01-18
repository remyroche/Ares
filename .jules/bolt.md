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

## 2026-01-16 - Feature Engineering Optimization: Mathematical Equivalence

**Learning:** `log_returns.rolling(w).sum()` is mathematically equivalent to `log_price.diff(w)` (assuming `log_returns = log_price.diff()`). Replacing the rolling window operation with vectorized differencing yields a >10x speedup (24ms -> 2ms for 1M rows) by avoiding iterator overhead.

**Action:** Look for rolling sum operations on differenced series and replace them with direct differencing of the original series where appropriate.

**Key Optimizations Implemented:**
1. **Optimized Rolling Momentum:** Replaced `rolling(w).sum()` on returns with `log_price.diff(w)` in `apply_layer2_price_processing`.
