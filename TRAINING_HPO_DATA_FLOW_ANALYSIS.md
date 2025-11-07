# Training & HPO Data Flow Analysis

## Executive Summary

**Critical Finding**: The current training pipeline does NOT apply temporal splitting before model training or HPO. This means:
- Models are trained on the entire dataset (including data that should be reserved for validation/testing)
- HPO validates on data that overlaps with what should be test data
- **Data leakage risk**: Models see future data during training/optimization

## Current Data Flow (Problematic)

```
┌─────────────────────────────────────────────────────────────┐
│ Step: unified_models_training_step.py                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ retrieve_training_data()                                     │
│   → Loads FULL dataset from artifacts                       │
│   → Shape: (N samples × M features)                         │
│   → NO temporal filtering applied                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ HPO: _perform_hierarchical_hpo() (Line 552)                 │
│   → Split: 80% train / 20% validation                       │
│   → Uses hierarchical_parameter_optimizer                   │
│   → hierarchical_parameter_optimizer._evaluate_params()     │
│       → If X_val provided: uses holdout validation          │
│       → Otherwise: uses TimeSeriesSplit CV                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Model Training: UnifiedTrainingPipeline                     │
│   → Trains on FULL dataset (or 80% if HPO ran)             │
│   → NO separation of train/val/test                         │
└─────────────────────────────────────────────────────────────┘

❌ PROBLEM: All data used for training, including future periods
❌ PROBLEM: No temporal boundaries enforced
❌ PROBLEM: Temporal splitting system created but NOT integrated
```

## Ideal Data Flow (What Should Happen)

```
┌─────────────────────────────────────────────────────────────┐
│ Step: Create Temporal Split Config (ONCE per pipeline)      │
│   config = create_temporal_split_config_for_pipeline(...)   │
│   → Defines: Training / Validation / Test periods          │
│   → Saved to: config/temporal_splits/{symbol}_{exchange}.json│
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step: unified_models_training_step.py                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ retrieve_training_data()                                     │
│   → Loads FULL dataset from artifacts                       │
│   → Shape: (N samples × M features)                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Apply Temporal Filtering (NEW - NEEDS INTEGRATION)          │
│   training_view = get_data_for_purpose(                     │
│       data, 'training', temporal_config)                    │
│   → Filters to TRAINING period only                         │
│   → Example: 2020-2023 (60% + 1-day embargo)               │
│   → Shape: (N_train samples × M features)                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ HPO: _perform_hierarchical_hpo()                            │
│   → Option A: Nested CV within training period             │
│       → TimeSeriesSplit with embargo                        │
│       → Each fold uses past data for train, future for val │
│   → Option B: Use separate validation period               │
│       → Training period: 2020-2022 (model training)        │
│       → Validation period: 2023-2024 (HPO validation)      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Model Training: UnifiedTrainingPipeline                     │
│   → Trains ONLY on training period data                     │
│   → Uses optimal params from HPO                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Final Validation: basic_backtesting_post                    │
│   test_view = get_data_for_purpose(                         │
│       predictions, 'test', temporal_config)                 │
│   → Uses TEST period (2024-2025 - completely unseen)       │
│   → Walk-forward CV within test period                      │
└─────────────────────────────────────────────────────────────┘

✅ CORRECT: Training uses only training period
✅ CORRECT: HPO validates on separate data (validation period or nested CV)
✅ CORRECT: Final testing on completely unseen test period
```

## Detailed Code Analysis

### 1. **unified_models_training_step.py**

#### Line 102-103: Data Retrieval
```python
# Retrieve training data and targets from artifacts
training_data, analyst_targets, tactician_targets = await self._retrieve_training_data(config, yaml_config)
```
**Issue**: Returns FULL dataset without temporal filtering.

#### Lines 186-225: HPO Execution
```python
# Perform hyperparameter optimization before training
if config.get('enable_hpo', True) and training_data is not None:
    # ...
    # Line 590-596 in _perform_hierarchical_hpo():
    hpo_train_size = int(len(training_data) * 0.8)
    X_train = training_data.iloc[:hpo_train_size]
    X_val = training_data.iloc[hpo_train_size:]
```
**Issue**:
- HPO uses 80/20 split of FULL dataset
- No temporal boundaries applied
- Validation data (20%) could include future test period data

#### Lines 994-1393: _retrieve_training_data()
```python
async def _retrieve_training_data(self, config: Dict[str, Any], yaml_config: Dict[str, Any]) -> tuple:
    # ...loads data from artifacts...
    # NO temporal filtering applied!
    return training_data, analyst_targets, tactician_targets
```
**Issue**: No integration with temporal splitting system.

### 2. **hierarchical_parameter_optimizer.py**

#### Lines 524-732: optimize() method
```python
def optimize(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,  # ← Optional validation set
    y_val: Optional[np.ndarray] = None,
    # ...
):
```
**Current behavior**:
- If `X_val` provided: uses holdout validation (line 1678-1689)
- If `X_val` is None: uses `TimeSeriesSplit` CV (line 1692-1700)

**Issue**: The holdout validation is good, but the split happens at the wrong level (in training step, not pipeline level).

#### Lines 1372-1396: _evaluate_params()
```python
def _evaluate_params(
    self,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    model: Optional[Any]
) -> float:
    """Evaluate a set of parameters using the objective function."""
    try:
        score = self.objective_func(
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model=model,
            cv_folds=self.cv_folds,
            scoring_metric=self.scoring_metric
        )
        return score
```
**Good**: Properly uses CV when validation set not provided.

**Issue**: But the data passed in is not temporally split at the pipeline level.

#### Lines 1662-1704: default_objective_function()
```python
def default_objective_function(
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
    cv_folds: int = 5,
    scoring_metric: str = 'neg_mean_squared_error'
) -> float:
    # ...
    if X_val is not None and y_val is not None:
        # Use holdout validation
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        # ...

    # Use cross-validation
    cv = TimeSeriesSplit(n_splits=cv_folds)  # ← Line 1692
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring=scoring_metric,
        n_jobs=1
    )
    return np.mean(scores)
```
**Good**: Uses `TimeSeriesSplit` for temporal ordering within CV folds.

**Issue**: But this CV is within the data passed to it (which is the full dataset, not just training period).

### 3. **basic_backtesting_post_step.py** (from previous analysis)

Already enhanced with:
- ✅ `get_data_for_purpose()` integration for temporal filtering
- ✅ `backtest_period` config: 'training', 'test', or 'both'
- ✅ Train/test comparison for overfitting detection
- ✅ Time-series CV within test period

**Good**: This step is properly integrated with temporal splitting.

## Recommendations

### Priority 1: Integrate Temporal Splitting into Training Pipeline (CRITICAL)

**File**: `src/training/steps/model_training/unified_models_training_step.py`

**Changes needed**:

1. **Load or create temporal config** (after line 102):
```python
# After retrieving training_data
from src.utils.versioned_artifacts import create_temporal_split_config_for_pipeline

# Create/load temporal split config
temporal_config = create_temporal_split_config_for_pipeline(
    symbol=config['symbol'],
    exchange=config['exchange'],
    timeframe=config['timeframe']
)
```

2. **Filter to training period** (after line 103):
```python
from src.utils.versioned_artifacts import get_data_for_purpose

# Filter to TRAINING period only
training_data = get_data_for_purpose(
    training_data,
    purpose='training',
    config=temporal_config
)

# Also filter targets
if analyst_targets is not None:
    analyst_targets = analyst_targets.loc[training_data.index]
if tactician_targets is not None:
    tactician_targets = tactician_targets.loc[training_data.index]
```

3. **Update HPO to use validation period OR nested CV** (around line 552):

**Option A: Use Validation Period for HPO**
```python
async def _perform_hierarchical_hpo(
    self,
    training_data: pd.DataFrame,  # Already filtered to training period
    targets: pd.Series,
    model_config: Dict[str, Any],
    config_file: str,
    config: Dict[str, Any],
    training_type: str
) -> Dict[str, Any]:
    """HPO using validation period."""

    # Load temporal config
    temporal_config = create_temporal_split_config_for_pipeline(
        symbol=config['symbol'],
        exchange=config['exchange'],
        timeframe=config['timeframe']
    )

    # Get VALIDATION period for HPO
    validation_data = get_data_for_purpose(
        self._get_full_dataset(),  # Need to retrieve full dataset again
        purpose='validation',
        config=temporal_config
    )
    validation_targets = self._get_full_targets().loc[validation_data.index]

    # Now HPO trains on training_data, validates on validation_data
    X_train = training_data
    y_train = targets
    X_val = validation_data
    y_val = validation_targets

    # Run HPO with separate validation period
    result = await asyncio.to_thread(
        self.hpo_orchestrator.run_hpo,
        model_name=model_info['name'],
        model_type=model_info['type'],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,  # ← Separate validation period
        y_val=y_val,
        model_class=model_info['class'],
        is_classification=model_info['is_classification']
    )
```

**Option B: Use Nested CV within Training Period**
```python
async def _perform_hierarchical_hpo(
    self,
    training_data: pd.DataFrame,  # Already filtered to training period
    targets: pd.Series,
    # ...
) -> Dict[str, Any]:
    """HPO using nested CV within training period."""

    # Don't split - let HPO use full training data with nested CV
    X_train = training_data
    y_train = targets
    X_val = None  # ← No validation set = forces TimeSeriesSplit CV
    y_val = None

    # Run HPO with nested CV (hierarchical_parameter_optimizer will use TimeSeriesSplit)
    result = await asyncio.to_thread(
        self.hpo_orchestrator.run_hpo,
        model_name=model_info['name'],
        model_type=model_info['type'],
        X_train=X_train,
        y_train=y_train,
        X_val=None,  # ← Forces nested CV
        y_val=None,
        model_class=model_info['class'],
        is_classification=model_info['is_classification']
    )
```

**Recommendation**: Use **Option A (Validation Period)** for better separation and clearer validation.

### Priority 2: Add Temporal Config to Pipeline Configuration

**File**: Any pipeline configuration or launcher

**Add**:
```python
# At pipeline start
temporal_config = create_temporal_split_config_for_pipeline(
    symbol=config['symbol'],
    exchange=config['exchange'],
    timeframe=config['timeframe'],
    data_start=datetime(2020, 1, 1),  # From data inspection
    data_end=datetime(2025, 1, 1)
)

# Pass to all steps
config['temporal_config'] = temporal_config
```

### Priority 3: Update Documentation

**File**: `REFACTORING_BACKTESTING_VALIDATION.md`

Add section:
```markdown
## Training Data Splitting Implementation Status

### ✅ Implemented:
- Temporal splitting system (`src/utils/versioned_artifacts/temporal_splits.py`)
- `basic_backtesting_post` integration with temporal filtering
- Train/test comparison for overfitting detection

### ⚠️ Pending Integration:
- **Training steps** need temporal filtering before model training
- **HPO** needs to use validation period (not 80/20 split of full data)
- **Pipeline launcher** needs to create/load temporal config

### 📋 Integration Checklist:
- [ ] Add temporal config creation to pipeline launcher
- [ ] Update `unified_models_training_step.py` to filter to training period
- [ ] Update `_perform_hierarchical_hpo()` to use validation period
- [ ] Update `_retrieve_training_data()` to accept temporal config
- [ ] Test end-to-end with temporal boundaries
```

## Summary of Required Changes

### Files to Modify:

1. **src/training/steps/model_training/unified_models_training_step.py**
   - Import temporal splitting functions (lines 15-20)
   - Create/load temporal config in `execute()` (after line 102)
   - Filter training data to training period (after line 103)
   - Update `_perform_hierarchical_hpo()` to use validation period (lines 552-720)
   - Update `_retrieve_training_data()` signature to accept temporal config (line 994)

2. **src/launcher/ares_launcher.py** (or equivalent)
   - Create temporal config at pipeline start
   - Pass temporal config to all training steps

3. **REFACTORING_BACKTESTING_VALIDATION.md**
   - Add implementation status section
   - Document remaining integration work

### Estimated Changes:
- **Lines to add**: ~50-100 lines
- **Lines to modify**: ~20-30 lines
- **New imports**: 2 (temporal splitting functions)
- **Breaking changes**: None (backward compatible via `config=None` default)

## Data Leakage Risk Assessment

### Current Risk: **HIGH** ⚠️

**Reason**: Models train on full dataset including future test data.

**Evidence**:
1. No temporal boundaries in training pipeline
2. HPO validates on 20% of full dataset (could include test period)
3. Models could learn patterns from future data

### After Integration: **LOW** ✅

**Reason**: Strict temporal boundaries enforced at pipeline level.

**Protection**:
1. Training period: 60% (2020-2023) + 1-day embargo
2. Validation period: 20% (2023-2024) + 1-day embargo
3. Test period: 20% (2024-2025) - completely unseen
4. HPO validates on validation period (unseen during model training)
5. Final testing on test period (unseen during both training and HPO)

## Best Practices Compliance

### ✅ Already Following:
- TimeSeriesSplit for temporal ordering in CV
- Embargo periods between splits
- Train/test comparison for overfitting detection

### ⚠️ Needs Implementation:
- Pipeline-level temporal boundaries
- Validation period for HPO (not just 80/20 split)
- Training period isolation for model training

### 📋 Recommended Approach:

**Data Flow**:
```
Full Dataset (2020-2025)
    │
    ├─ Training Period (2020-2023) ──────► Model Training
    │                                      └─ Nested CV for model validation
    │
    ├─ Validation Period (2023-2024) ────► HPO Optimization
    │                                      └─ Find best hyperparameters
    │
    └─ Test Period (2024-2025) ───────────► Final Backtesting
                                           └─ Walk-forward CV for robustness
```

**Key Principle**: Each stage uses a separate, non-overlapping period with embargo buffers.

## Conclusion

The current training pipeline does NOT integrate with the temporal splitting system that was created. This represents a **critical data leakage risk** where models train on data that should be reserved for validation/testing.

**Immediate Action Required**:
1. Integrate temporal filtering into `unified_models_training_step.py`
2. Update HPO to use validation period (not 80/20 split)
3. Test end-to-end to ensure proper data separation

**Expected Outcome**:
- ✅ Models train ONLY on training period (60% historical data)
- ✅ HPO validates ONLY on validation period (20% unseen data)
- ✅ Final testing ONLY on test period (20% completely unseen data)
- ✅ Zero data leakage risk
- ✅ Proper walk-forward validation throughout pipeline
