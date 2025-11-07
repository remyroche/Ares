# Backtesting & Validation Refactoring

## Summary of Changes

This refactoring removes the dedicated `walk_forward_validation` step and strengthens `basic_backtesting_post` with integrated time-series cross-validation. The goal is to ensure robust validation without data leakage while streamlining the pipeline.

## Key Changes

### 1. Removed walk_forward_validation Step

**Files Deleted:**
- `src/training/steps/backtesting/walk_forward_validation_step.py`

**Files Modified:**
- `src/training/steps/backtesting/__init__.py` - Removed import and registration
- `src/launcher/validation_utilities.py` - Removed from step dependencies and order
- `src/launcher/ares_launcher.py` - Removed from BACKTESTING pipeline
- `src/config/training.py` - Removed `enable_walk_forward_validation` flag
- `src/config/multi_output_config.py` - Removed `enable_walk_forward_validation` flag

**Rationale:**
- The walk_forward_validation step was redundant with the CV capabilities already built into optimization stages
- Time-series CV is now integrated directly into basic_backtesting_post for more cohesive validation
- Reduces pipeline complexity while maintaining validation rigor

### 2. Strengthened basic_backtesting_post with Time-Series CV

**Enhanced Features:**
- Added `sklearn.model_selection.TimeSeriesSplit` for proper temporal ordering
- Implemented embargo period (2% default) between train/test to prevent look-ahead bias
- Added `_run_time_series_cv_backtest()` method that:
  - Splits data into N folds (default 5) with proper temporal ordering
  - Applies embargo between train/test boundaries
  - Validates strategy performance across different time periods
  - Aggregates results with mean/std metrics
- Updated report generation to include CV results section

**Configuration:**
```python
self.enable_time_series_cv = True
self.cv_n_splits = 5
self.cv_embargo_pct = 0.02  # 2% embargo
```

**Validation Approach:**
1. Data is split chronologically into N folds
2. Each fold uses only data from previous folds for "training" context
3. Embargo period ensures no overlap between train/test boundaries
4. Strategy is backtested on each fold's test period independently
5. Results are aggregated across all folds (mean ± std)

### 3. HPO Validation Confirmation

**Verified that `final_parameters_optimization` already includes:**

✅ **Cross-Validation Support:**
- `use_cv = config.get('use_cross_validation', True)`
- `cv_folds = config.get('cv_folds', 5)`
- Integrated with `TimeSeriesSplitValidator` from `src/utils/ml_common/validation/cv_utils.py`

✅ **Time-Series CV in Hierarchical HPO:**
- `hierarchical_hpo.py` imports `TimeSeriesSplit` from sklearn
- Config includes `enable_time_series_cv: bool = True`
- Supports purged k-fold with embargo via `PurgedKFoldTime`

✅ **Validation Features:**
- Overfitting detection (`enable_overfitting_detection`)
- Temporal validation (`enable_temporal_validation`)
- Timeframe validation (`enable_timeframe_validation`)
- Data leakage detection via `DataLeakageDetector`

✅ **Optimization Infrastructure:**
- Bayesian TPE optimizer with staged optimization (coarse → fine → TPE)
- Hardware acceleration (M1 GPU/CPU optimization)
- Parallel evaluation with proper CV splits
- Early stopping with patience and threshold

**Key Files:**
- `src/training/steps/backtesting/final_parameters_optimization.py`
- `src/utils/ml_common/optimization/bayesian_tpe_optimizer.py`
- `src/utils/ml_common/optimization/hierarchical_hpo.py`
- `src/utils/ml_common/validation/cv_utils.py`

## Data Leakage Prevention

### 1. **Temporal Split Configuration (NEW!)**

**Added:** `src/utils/versioned_artifacts/temporal_splits.py`

The system now provides **pipeline-level temporal boundaries** through versioned artifacts:

```python
# Create configuration once for the pipeline
config = create_temporal_split_config_for_pipeline(
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="15m",
    data_start=datetime(2020, 1, 1),
    data_end=datetime(2025, 1, 1)
)

# Automatically creates:
# Training:   2020-01-01 to 2023-01-01 (60%) + 30-day embargo
# Validation: 2023-02-01 to 2024-01-01 (20%) + 30-day embargo
# Test:       2024-02-01 to 2025-01-01 (20%)
```

**Usage in Pipeline Steps:**

```python
# In model training steps
training_view = get_data_for_purpose(features_view, 'training', config)
training_data = training_view.materialize()  # Only 2020-2023 data

# In parameter optimization
validation_view = get_data_for_purpose(predictions_view, 'validation', config)
validation_data = validation_view.materialize()  # Only 2023-2024 data (unseen by models)

# In final backtesting
test_view = get_data_for_purpose(predictions_view, 'test', config)
test_data = test_view.materialize()  # Only 2024-2025 data (completely unseen)
```

**Key Features:**
- **Validation at Config Creation:** Ensures no overlap between periods
- **Embargo Enforcement:** Automatic 30-day buffer between periods
- **Saved & Reused:** Config saved to `config/temporal_splits/{symbol}_{exchange}_{timeframe}.json`
- **Integrates with Versioned Artifacts:** Uses efficient view filtering (no full loads)

See `src/utils/versioned_artifacts/TEMPORAL_SPLITS_GUIDE.md` for complete documentation.

### 2. Time-Series CV with Embargo (Within Periods)
The CV implementation in `basic_backtesting_post` ensures robustness **within the test period**:
- **Temporal Ordering:** Test data always comes after training data
- **Embargo Period:** 2% buffer between train/test boundaries to prevent look-ahead
- **No Overlap:** Each fold's test period is completely independent
- **Realistic Walk-Forward:** Mimics real trading where you only have access to past data

### 2. HPO Stage Validation
The `final_parameters_optimization` step already ensures:
- **Nested CV:** Inner CV loops for hyperparameter selection within outer CV folds
- **Purged K-Fold:** Optional purging of samples near boundaries
- **Embargo Logic:** Built into `PurgedKFoldTime` if enabled
- **OOF Generation:** Out-of-fold predictions for unbiased validation

### 3. ML Model Training Separation
The pipeline structure naturally prevents leakage:
```
step15_tactician_specialist_training
   ↓
step16_confidence_calibration
   ↓
step17_final_parameters_optimization (with CV)
   ↓
step19_monte_carlo_validation (uses different period)
   ↓
step20_ab_testing
```

Each stage uses distinct time periods or CV folds, ensuring that:
- Model training data ≠ Parameter optimization data ≠ Final validation data
- Time-series CV ensures no future information leaks into past decisions
- Monte Carlo simulation tests on synthetic variations

## Validation Hierarchy

### Level 1: Training-Time Validation
- **Where:** Model training steps (analyst, tactician, regime models)
- **Method:** Time-series CV during model fitting
- **Purpose:** Prevent model overfitting to training data

### Level 2: Parameter Optimization Validation
- **Where:** `final_parameters_optimization` step
- **Method:** Nested CV with embargo
- **Purpose:** Validate parameter choices across different time periods
- **Already Includes:** Walk-forward validation via nested CV

### Level 3: Strategy Validation
- **Where:** `basic_backtesting_post` step
- **Method:** Time-series CV on full strategy
- **Purpose:** Validate complete strategy performance with optimized parameters
- **New Enhancement:** Integrated walk-forward CV with embargo

### Level 4: Robustness Testing
- **Where:** `monte_carlo_simulation` step
- **Method:** Synthetic data variations
- **Purpose:** Test strategy under different market conditions

## Benefits of This Refactoring

1. **Simplified Pipeline:** One less step to maintain and understand
2. **Integrated Validation:** CV is where it belongs - in the backtesting step
3. **No Functionality Loss:** All validation capabilities preserved
4. **Better Reporting:** CV results integrated into backtest reports
5. **Reduced Redundancy:** Eliminates duplicate validation logic
6. **Maintained Rigor:** All data leakage prevention mechanisms intact

## Configuration Changes Required

### Old Configuration (No longer needed):
```python
"VALIDATION": {
    "enable_walk_forward_validation": True,  # Removed
    ...
}
```

### New Configuration (Already in place):
```python
# In basic_backtesting_post
self.enable_time_series_cv = True  # Enable CV during backtesting
self.cv_n_splits = 5              # Number of CV splits
self.cv_embargo_pct = 0.02        # 2% embargo between folds

# In final_parameters_optimization (already configured)
use_cv = True                      # Enable CV during optimization
cv_folds = 5                       # Number of CV folds
enable_time_series_cv = True       # Use time-series split
```

## Testing Recommendations

1. **Verify CV Results:** Check that cv_results appear in backtest reports
2. **Validate Embargo:** Ensure test periods don't overlap with training
3. **Check Fold Sizes:** Verify each fold has sufficient samples (>50)
4. **Compare Metrics:** CV std should indicate consistency across periods
5. **Monitor Performance:** CV mean should be close to full backtest result

## Migration Notes

### For Existing Workflows:
- Remove any explicit calls to `walk_forward_validation` step
- Update pipeline configurations to remove the step
- CV validation now happens automatically in `basic_backtesting_post`
- Check backtest reports for new CV results section

### For Custom Implementations:
- If you need separate walk-forward validation, use `src/validation/walkforward_validation.WalkForwardValidator`
- The full implementation is still available in `src/validation/` directory
- This refactoring only removes it from the standard backtesting pipeline

## Implementation Status

### ✅ Completed Features

1. **Temporal Splitting System** (`src/utils/versioned_artifacts/temporal_splits.py`)
   - ✅ `TemporalPeriod` class for period definitions
   - ✅ `TemporalSplitConfig` for train/val/test configuration
   - ✅ `get_data_for_purpose()` function for filtering data by period
   - ✅ 1-day embargo between periods
   - ✅ Backward compatible (returns full data if config=None)

2. **Backtesting Integration** (`src/training/steps/backtesting/basic_backtesting_post_step.py`)
   - ✅ Temporal filtering integrated
   - ✅ `backtest_period` config: 'training', 'test', or 'both'
   - ✅ Train/test comparison for overfitting detection
   - ✅ Time-series CV within test period

3. **Training Pipeline Integration** (`src/training/steps/model_training/unified_models_training_step.py`)
   - ✅ Temporal config creation at pipeline start
   - ✅ Training data filtered to training period only
   - ✅ Full dataset preserved for validation period access
   - ✅ HPO uses validation period (Option A implemented)
   - ✅ Backward compatible fallback to 80/20 split if needed

4. **HPO Best Practices** (`src/utils/ml_common/optimization/hierarchical_parameter_optimizer.py`)
   - ✅ Uses `TimeSeriesSplit` for temporal ordering in CV
   - ✅ Supports separate validation period for holdout validation
   - ✅ Falls back to nested CV if validation set not provided

### 📊 Data Flow (Current Implementation)

```
Pipeline Start
    ↓
Create Temporal Split Config
    ├─ Training:   60% of data (2020-2023)
    ├─ Validation: 20% of data (2023-2024)
    └─ Test:       20% of data (2024-2025)
    ↓
unified_models_training_step.py
    ├─ Retrieve full dataset from artifacts
    ├─ Store full dataset for validation access
    ├─ Filter to TRAINING period only
    └─ Pass filtered training data to models
    ↓
HPO (_perform_hierarchical_hpo)
    ├─ X_train: Training period data
    ├─ X_val: Validation period data (separate!)
    ├─ Validates on unseen validation period
    └─ Returns optimal hyperparameters
    ↓
Model Training
    ├─ Trains ONLY on training period
    ├─ Uses optimal params from HPO
    └─ Never sees validation or test data
    ↓
basic_backtesting_post
    ├─ backtest_period='test'
    ├─ Filters to TEST period only
    ├─ Walk-forward CV within test period
    └─ Compares train vs test for overfitting detection
```

### 🔒 Data Leakage Prevention (Active)

| Stage | Data Used | Period | Leakage Risk |
|-------|-----------|--------|--------------|
| **Model Training** | Training period | 2020-2023 (60%) | ✅ LOW - Isolated period |
| **HPO Validation** | Validation period | 2023-2024 (20%) | ✅ LOW - Separate from training |
| **Final Backtesting** | Test period | 2024-2025 (20%) | ✅ LOW - Never seen by models |
| **Embargo** | 1 day buffer | Between all periods | ✅ Active |

### 📝 Key Files Modified

1. **unified_models_training_step.py** (Lines 32-37, 112-189, 676-738)
   - Added temporal splitting imports
   - Integrated temporal config creation
   - Filtered training data to training period
   - Updated HPO to use validation period

2. **basic_backtesting_post_step.py** (Previously modified)
   - Integrated temporal filtering
   - Added train/test comparison
   - Implemented walk-forward CV

3. **temporal_splits.py** (New file, 337 lines)
   - Complete temporal splitting system

### ⚠️ Important Notes

1. **Backward Compatibility**: If temporal config is not available, the system falls back to:
   - 80/20 split within training data for HPO
   - Full dataset for model training (pre-refactor behavior)

2. **Configuration**: Temporal config is created automatically using data boundaries:
   - Percentages: 60% train / 20% validation / 20% test (configurable)
   - Embargo: 1 day between periods (configurable)

3. **Validation Period Purpose**: Used exclusively for HPO, ensuring hyperparameter optimization happens on data completely separate from model training.

## Conclusion

This refactoring maintains all validation capabilities while simplifying the pipeline. The key insight is that walk-forward validation is essentially time-series CV, which is better integrated into the backtesting step itself rather than being a separate pipeline stage. The HPO stages already include comprehensive CV validation, making the dedicated walk-forward step redundant.

**NEW: Temporal splitting now enforces strict train/val/test boundaries at the pipeline level, preventing data leakage.**

**Data Leakage Prevention:** ✅ Enforced via pipeline-level temporal splitting + embargo
**Validation Rigor:** ✅ Enhanced with integrated CV in backtesting + separate validation period
**Pipeline Simplicity:** ✅ Improved by removing redundant step
**HPO Validation:** ✅ Uses separate validation period (Option A implemented)
**Training Isolation:** ✅ Models train ONLY on training period
