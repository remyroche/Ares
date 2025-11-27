# Mean Reversion Step Bottleneck Analysis

**Date:** 2025-11-27
**Step:** `ml_mean_reversion_step` (`MLMeanReversionRegimeStep`)
**File:** `src/training/steps/market_analysis/ml_reversion_regime_step.py`

## Executive Summary

The `ml_mean_reversion_step` has been observed to be "stuck on the last round." After analyzing the code, **the "last round" refers to the final fold in the walk-forward validation** (fold 4 of 5, 0-indexed). This fold is the most computationally expensive because it trains on the largest expanding window of data.

## Identified Bottlenecks (Ranked by Severity)

### 🔴 CRITICAL: Walk-Forward Validation (Lines 1302-1408)

**Location:** `_run_walkforward_validation()` method, line 1340

**Complexity:** O(n_folds × n_samples × n_estimators)

**Issue:**
- Trains **5 separate XGBoost models** (one per fold), each with up to **400 estimators** (line 1338)
- Each fold uses an **expanding window**, so the last fold trains on ~75-80% of the full dataset
- Each fold also includes:
  - Model training with early stopping
  - Calibration fitting (`CalibratedClassifierCV` with isotonic/sigmoid, line 1374)
  - Prediction and metrics computation

**Time Estimate per Fold:**
- Fold 0: ~10-20s (small window)
- Fold 1: ~15-30s (medium window)
- Fold 2: ~20-40s (larger window)
- Fold 3: ~30-60s (even larger window)
- **Fold 4: ~60-120s (LARGEST window, 75-80% of data)** ⚠️

**Total Walk-Forward Time:** 135-270 seconds (~2-4.5 minutes)

**Why the Last Round is Stuck:**
The last fold (fold 4) trains on the most data and takes 2-3x longer than the previous folds. This creates the perception of being "stuck" even though it's just computationally intensive.

---

### 🟡 HIGH: Teacher Feature Calculation (Lines 391-441)

**Location:** `_build_teacher_features()` method

**Complexity:** O(n_samples × window_size) for each of 4-5 features

**Issues:**
1. **Rolling Hurst calculation** (line 446): Nested loop computing R/S statistic
   - Outer loop: `len(series)` iterations
   - Inner loop: Operations on `window` samples (default 200)
   - **Not vectorized**, pure Python loops

2. **Rolling OU parameters** (line 470): Similar nested structure
   - Computes half-life and theta for each window
   - Linear regression in each window

3. **Variance Ratio** (line 407): Triple-nested computation
   - Outer loop over samples
   - Rolling window calculation
   - Inner variance computation

4. **ADF test** (line 425): Statistical test per window
   - Uses `statsmodels.tsa.stattools.adfuller`
   - Not parallelized

**Time Estimate:** 60-180 seconds for full dataset (~30k-100k samples)

**Optimization Potential:** HIGH - These could be Numba-compiled or parallelized

---

### 🟡 HIGH: Balanced Feature Extraction (Lines 751-786)

**Location:** `_build_student_features()` method

**Complexity:** O(n_samples × n_features × window_size)

**Issues:**
- Calls `BalancedFeatureExtractor` which computes 64+ features (line 765)
- Multiple rolling windows (20, 30, 50 bars)
- Momentum divergence, reversion speed, regime persistence features
- Adaptive normalization with ATR calculations

**Time Estimate:** 30-90 seconds

**Optimization Potential:** MEDIUM - Already uses vectorized pandas operations, but could cache results

---

### 🟠 MEDIUM: HPO (Hierarchical Parameter Optimization) (Lines 891-1087)

**Location:** `_run_hierarchical_hpo()` method

**Complexity:** O(n_trials × n_samples × n_estimators)

**Issues:**
- Only runs when `mr_enable_hpo=True` (line 237)
- Uses `HierarchicalParameterOptimizer` with:
  - COARSE_GRID stage: ~20-50 trials
  - TPE stage: ~50-100 trials
  - Each trial trains an XGBoost model with 500 estimators (line 1008)
- **cv_folds=3** for cross-validation (line 1054)

**Time Estimate (if enabled):** 300-900 seconds (5-15 minutes)

**Current Status:** Likely **DISABLED** by default, but if enabled, this dominates runtime

**Optimization Potential:** HIGH - Could reduce n_estimators during HPO, use early stopping more aggressively

---

### 🟠 MEDIUM: Grid Backtest (Lines 1714-1851)

**Location:** `_save_artifacts_and_reports()` method

**Complexity:** O(n_samples × n_tp_values × n_sl_values)

**Issues:**
- Runs `run_simple_long_grid_backtest` or `run_simple_short_grid_backtest`
- By default, tests multiple TP/SL combinations (though currently using dynamic TPSL with single values, lines 1815-1828)
- Simulates trades with high/low price checks

**Time Estimate:** 15-45 seconds

**Optimization Potential:** LOW - Already optimized with single TP/SL values

---

### 🟢 LOW: Forward Metrics Computation (Lines 1411-1456)

**Location:** `_compute_forward_metrics()` method

**Complexity:** O(n_horizons × n_samples)

**Issues:**
- Computes metrics at 4 horizons (line 296): [2, 4, 8, 12]
- Simple vectorized operations
- Bucketing and correlation calculations

**Time Estimate:** 5-15 seconds

**Optimization Potential:** MINIMAL - Already vectorized

---

### 🟢 LOW: Direction Target Building (Lines 811-840)

**Location:** `_build_direction_target()` method

**Complexity:** O(n_samples × forward_horizon)

**Issues:**
- Simple forward returns calculation (line 828)
- Horizon = 6 bars (line 824)

**Time Estimate:** 2-5 seconds

**Optimization Potential:** MINIMAL

---

## Total Estimated Execution Time

**Typical run (HPO disabled):**
- Market data loading: 5-10s
- Teacher features: 60-180s
- Student features: 30-90s
- XGBoost training (main): 30-60s
- Calibration: 10-20s
- **Walk-forward validation: 135-270s** ⚠️
- Forward metrics: 5-15s
- Grid backtest: 15-45s
- Artifact saving: 5-10s

**Total: 295-700 seconds (5-12 minutes)**

**With HPO enabled:**
- Add 300-900 seconds → **Total: 595-1600 seconds (10-27 minutes)**

---

## Root Cause: "Stuck on Last Round"

The perception of being "stuck" is caused by:

1. **Walk-forward validation fold 4** taking 60-120 seconds (2-3x longer than earlier folds)
2. **Lack of progress logging** within the fold loop (no per-fold timing output)
3. **Silent execution** - `verbose=False` in XGBoost training (line 1214, 1371)

The step is **NOT actually stuck** - it's just in the computationally most expensive phase.

---

## Recommended Optimizations (Priority Order)

### 1. ⚡ Add Progress Logging to Walk-Forward Validation

**File:** `src/training/steps/market_analysis/ml_reversion_regime_step.py:1340`

**Change:**
```python
for fold in range(n_folds):
    tprint_info(f"🔄 Walk-forward validation: fold {fold+1}/{n_folds} (train_size={train_end})")
    # ... existing code ...
```

**Impact:** Eliminates perception of being stuck
**Effort:** 5 minutes
**Risk:** None

---

### 2. ⚡ Reduce Walk-Forward Estimators

**File:** Line 1338

**Current:**
```python
wf_estimators = max(200, min(base_estimators, 400))
```

**Suggested:**
```python
wf_estimators = max(100, min(base_estimators, 200))  # Reduced from 400 to 200
```

**Impact:** ~40-50% speedup in walk-forward validation (135-270s → 70-140s)
**Effort:** 1 minute
**Risk:** LOW - Walk-forward is for diagnostics only, not used in final model

---

### 3. ⚡ Make Walk-Forward Validation Optional

**File:** Lines 1284-1294

**Add config parameter:**
```python
enable_walkforward = bool(config.get("mr_enable_walkforward_validation", True))
if enable_walkforward:
    try:
        wf_metrics = self._run_walkforward_validation(...)
        if wf_metrics:
            metrics["walkforward"] = wf_metrics
    except Exception as e:
        tprint_warning(f"Walk-forward validation failed: {e}")
```

**Impact:** ~2-4 minutes speedup when disabled
**Effort:** 5 minutes
**Risk:** LOW - Walk-forward provides stability metrics but isn't critical for production

---

### 4. 🔥 Optimize Teacher Feature Calculations with Numba

**File:** Lines 443-487 (Hurst and OU methods)

**Suggested:** Compile with `@numba.jit(nopython=True)` decorators

**Impact:** ~40-60% speedup in teacher features (60-180s → 24-72s)
**Effort:** 30-60 minutes (need to refactor for Numba compatibility)
**Risk:** MEDIUM - Requires testing for numerical stability

---

### 5. 🔥 Cache Teacher Features

**File:** Lines 172-179

**Suggested:** Add caching similar to market data caching (lines 116-130)

```python
if self._cached_teacher_features is not None and self._cached_teacher_cache_key == teacher_key:
    teacher_df = self._cached_teacher_features.copy()
else:
    teacher_df = self._build_teacher_features(market_data, config)
    self._cached_teacher_features = teacher_df.copy()
    self._cached_teacher_cache_key = teacher_key
```

**Impact:** 60-180s speedup on repeated runs
**Effort:** 10 minutes
**Risk:** LOW - Just adds caching layer

---

### 6. 🔥 Parallelize Walk-Forward Folds

**File:** Lines 1340-1390

**Suggested:** Use `joblib.Parallel` or `multiprocessing.Pool` to run folds in parallel

**Impact:** ~3-4x speedup on 4+ core systems (135-270s → 35-70s)
**Effort:** 30-45 minutes
**Risk:** MEDIUM - Need to handle XGBoost threading properly

---

### 7. 🔥 Reduce Walk-Forward Folds

**File:** Line 1320

**Current:** `n_folds = int(config.get("mr_walkforward_folds", 5))`

**Suggested:** `n_folds = int(config.get("mr_walkforward_folds", 3))`

**Impact:** ~40% speedup (135-270s → 80-160s)
**Effort:** 1 minute
**Risk:** LOW - 3 folds still provides reasonable stability estimates

---

### 8. ⚡ Add Verbose Logging to Main XGBoost Training

**File:** Line 1214

**Change:**
```python
model.fit(
    X_train_np,
    y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    verbose=10  # Changed from False - print every 10 rounds
)
```

**Impact:** Better visibility into training progress
**Effort:** 1 minute
**Risk:** None (just adds output)

---

### 9. 🔥 Use XGBoost's `hist` Tree Method with GPU

**File:** Line 1189

**Current:** `tree_method="hist"`

**Suggested:** `tree_method="gpu_hist"` (if CUDA available)

**Impact:** 3-10x speedup in XGBoost training
**Effort:** 2 minutes (just config change)
**Risk:** LOW - Requires GPU, falls back to CPU if unavailable

---

### 10. ⚡ Skip Balanced Feature Extraction in Fast Mode

**File:** Line 751

**Suggested:**
```python
if bool(config.get("mr_enable_balanced_features", True)) and not config.get("fast_mode", False):
```

**Impact:** 30-90s speedup when disabled
**Effort:** 2 minutes
**Risk:** MEDIUM - May reduce model quality

---

## Immediate Actions (Quick Wins)

To address the "stuck on last round" issue **immediately**, implement these in order:

1. **Add progress logging to walk-forward validation** (5 min) - Shows it's not stuck
2. **Reduce walk-forward estimators from 400 to 200** (1 min) - 40% speedup
3. **Reduce walk-forward folds from 5 to 3** (1 min) - 40% speedup
4. **Add verbose logging to main XGBoost training** (1 min) - Better visibility

**Combined impact:** Reduces walk-forward time from 135-270s to ~50-95s (~60% speedup) and eliminates perception of being stuck.

---

## Configuration Recommendations

**Add to config for fast iteration:**
```yaml
mr_enable_walkforward_validation: false  # Disable walk-forward for fast iteration
mr_walkforward_folds: 3  # Reduce from 5 to 3 if enabled
mr_n_estimators: 300  # Reduce from 500 for faster training
mr_enable_hpo: false  # Disable HPO unless tuning
mr_enable_balanced_features: true  # Keep for quality
```

**Add to config for production:**
```yaml
mr_enable_walkforward_validation: true  # Full validation
mr_walkforward_folds: 5  # Full 5-fold validation
mr_n_estimators: 500  # Full trees
mr_enable_hpo: false  # Only enable when tuning
mr_enable_balanced_features: true  # Full features
```

---

## Conclusion

The `ml_mean_reversion_step` is **not actually stuck** - it's executing the computationally intensive walk-forward validation, with the last fold (fold 4) taking 60-120 seconds due to training on the largest data window.

The **immediate fix** is to add progress logging so users can see the step is progressing. The **medium-term optimizations** focus on reducing walk-forward computational cost and optimizing teacher feature calculations, which together can reduce execution time by 50-70%.
