# Report: compare_tbm_parameters.py Changes

This report compares the evolution of `extreme_price_movements/offline_optimisers/compare_tbm_parameters.py` over the last 3 days.

## Version Reference
*   **Current Version:** Wednesday Morning (Feb 25)
*   **Yesterday's Version:** Tuesday Morning (Feb 24) [Commit: `7e08a0d3e`]
*   **Monday's Version:** Monday Morning (Feb 23) [Commit: `766d1a1b0`]

## 1. Monday Morning vs. Yesterday Morning (Feb 23 -> Feb 24)
**Major Theme: Adoption of Optuna and Parallelization**

The transition from Monday to Tuesday involved a significant architectural shift from simple grid search to an Optuna-based optimization framework, along with performance improvements for data loading.

*   **Optuna Integration:**
    *   Replaced the manual `stage1_grid` loop with `TBMObjective` class and `optuna.create_study()`.
    *   Stage 1 and Stage 2 now use `TPESampler` (implied default) for smarter parameter exploration instead of fixed grids.
*   **Parallel Data Loading:**
    *   Refactored `_read_symbol_parquet_dir` to use `joblib.Parallel` and `delayed` for concurrent file reading, improving startup time.
*   **Expanded Scope:**
    *   Increased Stage 1 symbol subsampling limit from **30** to **150** symbols, allowing for more representative optimization runs.
*   **Code Quality & Constants:**
    *   Replaced hardcoded label integers (`1`, `-1`, `0`) with imported constants `OUT_TP`, `OUT_SL`, `OUT_TO` from `labeling.py`.
    *   Added `PERP_FEATURE_KEYS` handling for perpetual futures support.

## 2. Yesterday Morning vs. Current (Feb 24 -> Feb 25)
**Major Theme: Speed Optimization (Vectorization & Caching)**

The changes since yesterday focus on accelerating the optimization process, particularly for Stage 2 refinements, and adding platform-specific stability controls.

*   **Vectorized SL Sweeps:**
    *   In Stage 2 (`s2_objective`), the code now evaluates a "ladder" of SL multipliers (e.g., `[sl, sl-0.1, sl+0.1, ...]`) for each TP anchor trial.
    *   This allows the optimizer to test multiple Risk/Reward ratios efficiently without re-computing the expensive TP barriers.
*   **TP Anchor Caching:**
    *   Introduced `_TP_ANCHOR_CACHE` to cache the TP barrier dataframe and stats.
    *   Updated `build_barriers` to check this cache when `sl_method == "tp_pct"`, avoiding redundant TP barrier construction during the vectorized SL sweeps.
*   **Parallelism Control:**
    *   Added `--n-jobs` argument to control Optuna worker concurrency.
    *   Implemented logic to auto-cap `n_jobs` to 1 on Apple Silicon (ARM64) to avoid known stability issues (`_is_apple_arm`).
*   **Refinement Scheduling:**
    *   Added `_candidate_pct_schedule_for_stage` helper to allow different candidate percentile sweeps for Stage 3/Refine steps.
*   **Production Alignment:**
    *   Integrated `_apply_prod_aligned_tp_centering` more deeply into the `TBMObjective` loop to ensure candidates align with production TP mechanics.

## Summary
The script has evolved from a linear grid-search tool into a sophisticated, parallelized, and cached Optuna optimizer. Monday's version was a simple iterator; Tuesday's version introduced the Optuna engine; and the current version optimizes that engine with vectorization and caching to handle the increased computational load efficiently.
