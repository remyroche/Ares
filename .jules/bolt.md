## 2026-01-14 - Python Rolling Operations Optimization
Learning: Replacing `pandas.Series.rolling().apply()` with Numba-optimized functions yields massive performance gains (2700x speedup observed).
Insight: `rolling().apply()` with a Python function incurs significant overhead due to per-window function calls and lack of vectorization. Numba can compile the rolling logic into optimized machine code, utilizing parallel processing (`prange`) and avoiding the Python interpreter loop.
Action: Whenever encountering `rolling().apply()` with custom Python logic in performance-critical paths, replace it with a Numba `@njit` implementation. Ensure to handle `NaN` propagation explicitly if needed, as Numba bypasses Pandas' built-in `NaN` guarding.
