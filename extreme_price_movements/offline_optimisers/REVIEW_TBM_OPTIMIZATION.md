# Review of TBM Parameter Optimization in `compare_tbm_parameters.py`

This document outlines the findings from reviewing the current TBM parameter optimization implementation and provides recommendations for moving fully to a robust Optuna-based workflow.

## Executive Summary

The codebase **already uses Optuna** for optimization in `compare_tbm_parameters.py`. However, the integration is currently **broken** due to a critical bug in the objective function, rendering the optimization process ineffective (likely equivalent to random search or worse). Additionally, the file contains significant dead code from the previous grid-search implementation.

## Critical Findings

### 1. Broken Objective Function (`mean_auc` Missing)

The Optuna objective function (`TBMObjective.__call__` and the inline `s2_objective` in `run`) attempts to maximize a metric named `mean_auc`:

```python
# In TBMObjective.__call__
score = float(res.get("mean_auc", 0.0))
return score
```

 However, the `evaluate_config` function **does not return a key named `mean_auc`** in its summary dictionary. It returns:
 - `min_cell_auc`
 - `median_cell_auc`
 - `stage1_score`
 - `stage2_score`
 - `mean_bucket_ic` (implied, though not explicitly named "mean_auc")

**Consequence:** The optimizer receives `0.0` for every trial (unless it hits an exception and returns -1.0). This means Optuna has no signal to learn from and is not optimizing anything.

### 2. Dead Code (Grid Search)

The functions `stage1_grid` and `stage2_grids_from_stage1` are defined but **never called** in the `run` function. The `run` function exclusively uses the Optuna path (`TBMObjective` and `s2_objective`). These functions should be removed to avoid confusion.

### 3. Inefficient Two-Stage Manual Process

The current implementation manually orchestrates a two-stage process:
1.  **Stage 1:** Run a global Optuna study (`optuna_stage1`).
2.  **Selection:** Manually select "winners" based on `promote_stage1`.
3.  **Stage 2:** Run a new, separate Optuna study (`optuna_s2_refine`) for each winner, constrained to a small neighborhood around the winner's parameters.

While this mimics the logic of the old grid search refinement, it is less efficient than allowing a single, longer Optuna study to naturally converge. TPE (Tree-structured Parzen Estimator) is designed to narrow down the search space automatically. The manual restart loses the history (meta-information) built up during Stage 1, effectively resetting the sampler's knowledge for each refinement.

## Diversity & Quality Analysis

**Question:** *Would this setup (single long Optuna study) still allow us to get a set of diverse & high quality geometries?*

**Answer:** Yes, but with a specific post-processing strategy.

Grid search naturally produces diversity by systematically covering the space. Optuna's TPE sampler, by design, converges on the "best" region, potentially reducing diversity (all top trials might cluster around a single optimal geometry).

To guarantee both **Quality** and **Diversity**, we recommend the following **"Generate -> Filter -> Select"** pipeline:

1.  **Generate Pool (Exploration):**
    - Run a single, large Optuna study (e.g., 500-1000 trials).
    - **Crucial:** Use `optuna.samplers.TPESampler(n_startup_trials=100)` to ensure sufficient random exploration before convergence.
    - This creates a large "pool" of candidate geometries, some excellent (converged) and some diverse (exploratory).

2.  **Filter Quality:**
    - Filter the pool to keep only "high quality" candidates.
    - Criterion: `stage2_score >= threshold` (e.g., top 20% or absolute value > 0.5).
    - This ensures any geometry we select is tradeable and performant.

3.  **Select Diverse (Greedy Selection):**
    - Apply the existing `_diverse_subset` function (Jaccard distance on trade labels) to this high-quality pool.
    - **Mechanism:**
        - Pick the single best scorer.
        - Iteratively pick the next best scorer that has `distance > min_distance` from all already selected configs.
        - Repeat until `N` geometries are found.
    - This leverages the existing robust diversity logic but feeds it a much richer, quality-assured pool than the limited grid search ever could.

**Conclusion:** This pipeline is superior to grid search because it focuses compute on promising regions (Quality) while the explicit post-selection step guarantees behavioral variety (Diversity).

## Recommendations

### 1. Fix the Objective Function

**Immediate Action:** Change the objective return value to use a valid metric. `stage2_score` appears to be the most comprehensive metric designed in the file (combining AUC, separation, stability, Sortino, etc.). Alternatively, use `median_cell_auc` if raw discrimination is the priority.

```python
# Recommended fix in TBMObjective.__call__
score = float(res.get("stage2_score", float("-inf")))
return score
```

### 2. Remove Dead Code

Delete the following unused functions:
- `stage1_grid`
- `stage2_grids_from_stage1`

### 3. Consolidate Optimization

Instead of manual stages, consider running a **single, longer Optuna study** with a higher trial count (e.g., 200-500 trials).
- **Why:** TPE needs around 50-100 trials to build a good model of the hyperparameter space. Splitting this into two short runs (100 + 15) disrupts this learning.
- **Diversity:** The current "diversity" selection (`_diverse_subset` using Jaccard distance) happens *after* optimization on the pool of results. This is fine and can remain. A single large pool of 500 trials will provide a better substrate for diversity selection than multiple small pools.

### 4. Enable Parallel Execution

The current implementation uses `n_jobs=1` and in-memory storage.
- **Action:** Use `optuna.storages.RDBStorage` (e.g., SQLite) to allow multiple parallel workers (running the script multiple times pointing to the same DB).
- **Code Change:**
  ```python
  storage_url = "sqlite:///tbm_optimization.db"
  study = optuna.create_study(storage=storage_url, study_name="tbm_study", load_if_exists=True)
  ```

### 5. Parameter Space Refinement

The current search space uses discrete `step` parameters (e.g., `step=0.1` for `k_tp`). This is acceptable if we want human-readable parameters, but Optuna works best with continuous spaces. Consider removing `step` or reducing it significantly (e.g., `0.01`).

### 6. Pruning (Optional but Advanced)

Currently, the objective function evaluates all horizons (2, 4, 8) atomically. Pruning (stopping a trial early) is difficult because intermediate results aren't reported until the end.
- **Optimization:** If simulation is slow, consider reporting the `stage1_score` or `median_cell_auc` of just the *first* horizon (H2) or a subset of symbols as an intermediate step. If H2 performance is terrible, prune the trial before computing H4 and H8.

## Implementation Plan

1.  **Bugfix:** Modify `TBMObjective` to return `res.get("stage2_score", -10.0)`.
2.  **Cleanup:** Delete `stage1_grid` and `stage2_grids_from_stage1`.
3.  **Refactor:** Simplify `run` to execute a single main study.
4.  **Scaling:** Add SQLite storage support for parallel execution.
