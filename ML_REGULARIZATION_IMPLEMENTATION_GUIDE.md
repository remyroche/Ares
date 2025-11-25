# ML Model Regularization Implementation Guide

## Overview

This guide documents the improvements to ML model regularization and data handling in the Ares trading system. The changes ensure:

1. **Burn-in Period**: 6-month burn-in period (1/6 of data) for indicator stabilization
2. **Rolling Window Normalization**: Use rolling window min/max instead of global normalization
3. **Temporal Data Splits**: Proper train/val/test splits with embargo periods
4. **Causality**: Ensure calculations at time t use only data available at time t

## Changes Made

### 1. Temporal Splits Infrastructure (`temporal_splits.py`)

#### Added Burn-in Period Support

The `TemporalSplitConfig` class now supports an optional burn-in period:

```python
@dataclass
class TemporalSplitConfig:
    training: TemporalPeriod
    validation: TemporalPeriod
    test: TemporalPeriod
    burnin: Optional[TemporalPeriod] = None  # NEW: Burn-in period
```

#### Updated `create_from_data` Method

Added `burnin_pct` parameter (default: 0.0, recommended: 1/6):

```python
config = TemporalSplitConfig.create_from_data(
    data_start=data_start,
    data_end=data_end,
    train_pct=0.6,
    val_pct=0.2,
    test_pct=0.2,
    embargo_days=1,
    burnin_pct=1/6  # 6 months of 3 years
)
```

The burn-in period structure:
- **Burn-in (1/6)**: Data used for training indicators without generating probabilities
- **[Embargo]**: 1-day buffer
- **Training (60%)**: Model training period
- **[Embargo]**: 1-day buffer
- **Validation (20%)**: HPO and calibration
- **[Embargo]**: 1-day buffer
- **Test (20%)**: Final evaluation

#### Updated Pipeline Helper

```python
def create_temporal_split_config_for_pipeline(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_start: Optional[datetime] = None,
    data_end: Optional[datetime] = None,
    config_path: Optional[Path] = None,
    enable_burnin: bool = True,  # Enable by default for ML models
    burnin_pct: float = 1/6
) -> TemporalSplitConfig
```

#### Added Burn-in Data Access

```python
# Get burn-in data for indicator stabilization
burnin_view = get_data_for_purpose(view, 'burnin', config)
```

### 2. Rolling Window Normalization (`scaling_normalization.py`)

Added three new normalization functions that ensure causality:

#### `rolling_zscore_normalize`

```python
def rolling_zscore_normalize(
    data: Union[pd.DataFrame, pd.Series],
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Apply rolling z-score normalization using only data available at time t.

    At time t, uses data from [t-window, t-1] to compute mean and std.
    This ensures no look-ahead bias.
    """
```

**Example Usage:**
```python
# Instead of global normalization
features_normalized = winsorized_zscore_normalize(features)

# Use rolling window normalization
features_normalized = rolling_zscore_normalize(
    features,
    window=500,  # Use last 500 bars
    min_periods=200
)
```

#### `rolling_winsorized_zscore_normalize`

```python
def rolling_winsorized_zscore_normalize(
    data: Union[pd.DataFrame, pd.Series],
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Apply rolling winsorized z-score normalization.

    Combines rolling window with winsorization to handle outliers
    while ensuring causality.
    """
```

**Example Usage:**
```python
features_normalized = rolling_winsorized_zscore_normalize(
    features,
    window=500,
    min_periods=200,
    lower_quantile=0.01,
    upper_quantile=0.99
)
```

#### `rolling_minmax_normalize`

```python
def rolling_minmax_normalize(
    data: Union[pd.DataFrame, pd.Series],
    window: int,
    min_periods: Optional[int] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Union[pd.DataFrame, pd.Series]:
    """
    Apply rolling min-max normalization.

    Normalizes to [0, 1] based on rolling min/max from past window.
    """
```

**Example Usage:**
```python
probabilities_normalized = rolling_minmax_normalize(
    probabilities,
    window=500,
    feature_range=(0.0, 1.0)
)
```

## Implementation Guide for ML Models

### Models to Update

#### Specialist ML Models (require burn-in + rolling normalization + train/val/test)
1. `hmm_ml_alpha_step`
2. `ml_smc_regime_step`
3. `ml_breakout_bounce_regime_step`
4. `ml_reversion_regime_step`

#### Regime Models (require rolling normalization only)
5. `ml_liquidity_regime_step`
6. `ml_risk_regime_step`
7. `ml_path_regime_step`

### Step-by-Step Implementation for Specialist ML Models

#### Step 1: Create Temporal Split Config with Burn-in

In the `execute` method, create a split config with burn-in enabled:

```python
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    get_data_for_purpose,
)

# In execute method
symbol = str(config.get("symbol", "ETHUSDT"))
exchange = str(config.get("exchange", "binance"))
timeframe = str(config.get("regime_timeframe", "15m"))

# Load market data
market_data, market_source = self.load_market_data_or_fail(...)

# Create temporal split config with 6-month burn-in
split_config = create_temporal_split_config_for_pipeline(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe,
    data_start=market_data.index.min(),
    data_end=market_data.index.max(),
    enable_burnin=True,  # Enable burn-in
    burnin_pct=1/6  # 6 months = 1/6 of 3 years
)
```

#### Step 2: Use Burn-in Data for Feature Generation

Features should be calculated on ALL data (including burn-in) to ensure proper indicator stabilization:

```python
# Calculate features on ALL data (including burn-in)
features_df = self._build_features(market_data, config)

# But filter to training/val/test when creating datasets
train_idx = (features_df.index >= split_config.training.start) & \
            (features_df.index <= split_config.training.effective_end)
val_idx = (features_df.index >= split_config.validation.start) & \
          (features_df.index <= split_config.validation.effective_end)
test_idx = (features_df.index >= split_config.test.start) & \
           (features_df.index <= split_config.test.effective_end)

X_train = features_df.loc[train_idx]
X_val = features_df.loc[val_idx]
X_test = features_df.loc[test_idx]
```

#### Step 3: Replace Global Normalization with Rolling Window

**Before (Global Normalization):**
```python
# OLD: Uses global mean/std across entire dataset
features_normalized = winsorized_zscore_normalize(features_df)
```

**After (Rolling Window Normalization):**
```python
from src.features_common.transforms.scaling_normalization import (
    rolling_winsorized_zscore_normalize,
)

# NEW: Uses rolling window (only past data at each time t)
window_size = int(config.get("normalization_window", 500))  # ~500 bars
features_normalized = rolling_winsorized_zscore_normalize(
    features_df,
    window=window_size,
    min_periods=window_size // 2,  # Allow half window for early data
    lower_quantile=0.01,
    upper_quantile=0.99
)
```

#### Step 4: Save Probabilities with Proper Temporal Alignment

When saving probabilities, ensure they're only calculated for training/val/test periods:

```python
# Create output dataframe
output_df = market_data.copy()

# Add features (calculated on all data for continuity)
for col in features_df.columns:
    output_df[col] = features_df[col]

# Add model predictions ONLY for train/val/test periods
output_df['probability'] = np.nan
output_df.loc[X_train.index, 'probability'] = train_probs
output_df.loc[X_val.index, 'probability'] = val_probs
output_df.loc[X_test.index, 'probability'] = test_probs

# Save with metadata about burn-in period
metadata = {
    'symbol': symbol,
    'exchange': exchange,
    'timeframe': timeframe,
    'burnin_start': split_config.burnin.start if split_config.burnin else None,
    'burnin_end': split_config.burnin.effective_end if split_config.burnin else None,
    'training_start': split_config.training.start,
    'training_end': split_config.training.effective_end,
    'validation_start': split_config.validation.start,
    'validation_end': split_config.validation.effective_end,
    'test_start': split_config.test.start,
    'test_end': split_config.test.effective_end,
}

self._save_artifact(
    data=output_df,
    artifact_name=f"ml_model_outputs_{timeframe}",
    artifact_type="data",
    metadata=metadata
)
```

### Step-by-Step Implementation for Regime Models

Regime models (liquidity, risk, path) only need rolling window normalization updates:

#### Replace Global Normalization

**Before:**
```python
# In _generate_features or similar method
normalized_features = winsorized_zscore_normalize(features)
```

**After:**
```python
from src.features_common.transforms.scaling_normalization import (
    rolling_winsorized_zscore_normalize,
)

# Use rolling window normalization
window_size = int(config.get("normalization_window", 500))
normalized_features = rolling_winsorized_zscore_normalize(
    features,
    window=window_size,
    min_periods=window_size // 2
)
```

## Regularization Best Practices

### 1. HPO Configuration

Ensure models have proper regularization parameters for HPO:

```python
# XGBoost regularization parameters
params = {
    'learning_rate': 0.01,  # Lower learning rate
    'max_depth': 3-5,  # Limit tree depth
    'min_child_weight': 10.0,  # Prevent overfitting
    'subsample': 0.7,  # Row sampling
    'colsample_bytree': 0.6,  # Column sampling
    'gamma': 0.1,  # Minimum loss reduction
    'reg_alpha': 1.0,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
}
```

### 2. Validation on Unseen Data

Always evaluate on test set (unseen data):

```python
# Train on training set
model.fit(X_train, y_train)

# Calibrate on validation set
calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
calibrated.fit(X_val, y_val)

# Evaluate on TEST set (unseen data)
test_predictions = calibrated.predict_proba(X_test)
test_metrics = calculate_metrics(y_test, test_predictions)

# Report test set performance as final evaluation
print(f"Test Set Performance: {test_metrics}")
```

### 3. Cross-Validation

Use walk-forward validation for time-series:

```python
def walk_forward_validation(X, y, n_folds=5):
    """Walk-forward validation respecting temporal order."""
    fold_size = len(X) // n_folds
    metrics = []

    for i in range(n_folds):
        # Expanding window: train on [0, i*fold_size]
        train_end = (i + 1) * fold_size
        val_end = min(train_end + fold_size, len(X))

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]

        # Train and evaluate
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics.append(calculate_metrics(y_val, preds))

    return metrics
```

## Testing Checklist

Before deploying updated models:

- [ ] Verify burn-in period is correctly applied (6 months)
- [ ] Confirm rolling window normalization uses only past data
- [ ] Check that probabilities are only generated for train/val/test periods
- [ ] Validate train/val/test splits with proper embargo periods
- [ ] Test on unseen data (test set) shows acceptable performance
- [ ] Verify no look-ahead bias in feature calculations
- [ ] Confirm metadata includes all temporal period boundaries
- [ ] Check that model performance doesn't degrade on test set vs validation set (sign of overfitting if it does)

## Configuration Parameters

Recommended configuration parameters:

```python
config = {
    # Temporal split parameters
    'train_pct': 0.6,
    'val_pct': 0.2,
    'test_pct': 0.2,
    'embargo_days': 1,
    'burnin_pct': 1/6,  # 6 months

    # Normalization parameters
    'normalization_window': 500,  # Rolling window size
    'normalization_min_periods': 250,  # Minimum periods
    'normalization_lower_quantile': 0.01,  # Winsorization lower
    'normalization_upper_quantile': 0.99,  # Winsorization upper

    # Regularization parameters (XGBoost)
    'learning_rate': 0.01,
    'max_depth': 4,
    'min_child_weight': 10.0,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'gamma': 0.1,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
}
```

## Next Steps

For each ML model listed above:

1. **Update feature generation** to use rolling window normalization
2. **Add temporal split config** with burn-in period
3. **Update train/val/test splits** to use the split config
4. **Modify probability generation** to respect temporal boundaries
5. **Update artifact saving** to include metadata about periods
6. **Add validation** on unseen test data
7. **Document** the changes in the model's docstring

## Example: Updating ml_reversion_regime_step

See the current implementation in `/home/user/Ares/src/training/steps/market_analysis/ml_reversion_regime_step.py` for the existing structure.

Key changes needed:

1. **Line 100-117**: Add temporal split config creation
2. **Line 257-307**: Update teacher feature normalization to use rolling window
3. **Line 486-654**: Update student feature normalization to use rolling window
4. **Line 687-719**: Update train/val/test split to use temporal split config
5. **Line 977-1327**: Update artifact saving to include burn-in metadata

This pattern should be followed for all other ML models.
