# Temporal Data Splitting Guide

## Problem: Data Leakage in ML Pipelines

When training ML models and then validating them, it's critical that:
1. **Training data** is used only for model training
2. **Validation data** is used only for parameter optimization
3. **Test data** is used only for final evaluation

Without proper separation, you risk **data leakage** - using future information to make past predictions.

## Solution: Temporal Split Configuration

The `temporal_splits` module provides a clean interface for managing train/val/test splits with:
- **Temporal ordering**: Test data always comes after training data
- **Embargo periods**: Buffer zones between periods to prevent look-ahead bias
- **Versioned artifacts integration**: Works seamlessly with the existing view system

---

## Quick Start

### 1. Create Temporal Split Configuration

```python
from src.utils.versioned_artifacts import create_temporal_split_config_for_pipeline
from datetime import datetime

# Create configuration for a trading pair
config = create_temporal_split_config_for_pipeline(
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="15m",
    data_start=datetime(2020, 1, 1),
    data_end=datetime(2025, 1, 1)
)

# Config is automatically saved to: config/temporal_splits/ETHUSDT_binance_15m.json
# Future calls will load the saved config to ensure consistency
```

This creates a split like:
```
2020-01-01 ════════════════════════════ 2023-01-01
   Training Period (60%)

                                         [30-day embargo]

                   2023-02-01 ═══════════════ 2024-01-01
                     Validation Period (20%)

                                                [30-day embargo]

                                                        2024-02-01 ════ 2025-01-01
                                                          Test Period (20%)
```

### 2. Use in Training Steps

```python
from src.training.steps.base_step import BaseStep
from src.utils.versioned_artifacts import get_data_for_purpose

class AnalystTrainingStep(BaseStep):
    async def execute(self, config):
        # Get temporal config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        # Get versioned artifact view
        features_view = self.versioned_store.get_view("features_v1")

        # Filter to TRAINING period only
        training_view = get_data_for_purpose(
            features_view,
            purpose='training',
            config=temporal_config
        )

        # Materialize training data
        training_data = training_view.materialize()

        # Train models on training data only
        models = self.train_models(training_data)

        return {'success': True}
```

### 3. Use in Parameter Optimization

```python
class FinalParametersOptimizer(BaseStep):
    async def execute(self, config):
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        # Get predictions from trained models
        predictions_view = self.versioned_store.get_view("ml_predictions")

        # Filter to VALIDATION period only
        validation_view = get_data_for_purpose(
            predictions_view,
            purpose='validation',
            config=temporal_config
        )

        validation_data = validation_view.materialize()

        # Optimize parameters on validation data
        # with nested CV WITHIN the validation period
        best_params = self.optimize_parameters(validation_data)

        return {'success': True}
```

### 4. Use in Final Backtesting

```python
class BasicBacktestingPost(BaseStep):
    async def execute(self, config):
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        # Get predictions with optimized parameters
        predictions_view = self.versioned_store.get_view("final_predictions")

        # Filter to TEST period only
        test_view = get_data_for_purpose(
            predictions_view,
            purpose='test',
            config=temporal_config
        )

        test_data = test_view.materialize()

        # Run walk-forward CV WITHIN the test period
        cv_results = self._run_time_series_cv_backtest(test_data, ...)

        return {'success': True}
```

---

## Data Flow Example

```
All Historical Data (2020-2025)
│
├─ Training Period (2020-01-01 to 2023-01-01)
│  │
│  ├─ Analyst Training ──────────────────┐
│  ├─ Tactician Training ─────────────────┤
│  └─ All ML models trained here ─────────┘
│     [Models see ONLY 2020-2023 data]
│
├─ [EMBARGO: 30 days]
│
├─ Validation Period (2023-02-01 to 2024-01-01)
│  │
│  └─ Final Parameters Optimization
│        └─ Nested CV within this period
│           ├─ Fold 1: Feb-May 2023 train, Jun 2023 test
│           ├─ Fold 2: Feb-Aug 2023 train, Sep 2023 test
│           └─ etc.
│        [Parameters optimized on unseen data]
│
├─ [EMBARGO: 30 days]
│
└─ Test Period (2024-02-01 to 2025-01-01)
   │
   └─ Basic Backtesting Post
         └─ Walk-forward CV within this period
            ├─ Fold 1: Feb-Apr 2024 train, May 2024 test
            ├─ Fold 2: Feb-Jun 2024 train, Jul 2024 test
            └─ etc.
         [Final validation on completely unseen data]
```

---

## Advanced: Custom Period Configuration

```python
from src.utils.versioned_artifacts import TemporalPeriod, TemporalSplitConfig
from datetime import datetime

# Create custom periods
training = TemporalPeriod(
    start=datetime(2020, 1, 1),
    end=datetime(2023, 6, 30),
    embargo_days=30,
    name="training"
)

validation = TemporalPeriod(
    start=datetime(2023, 8, 1),
    end=datetime(2024, 6, 30),
    embargo_days=30,
    name="validation"
)

test = TemporalPeriod(
    start=datetime(2024, 8, 1),
    end=datetime(2025, 1, 1),
    embargo_days=0,  # No embargo needed after test
    name="test"
)

# Create config
custom_config = TemporalSplitConfig(
    training=training,
    validation=validation,
    test=test
)

# Save for reuse
custom_config.save("config/temporal_splits/custom_split.json")
```

---

## Benefits

### 1. **Prevents Data Leakage**
- Guarantees no overlap between train/val/test periods
- Embargo periods prevent look-ahead bias at boundaries
- Validation enforced at config creation time

### 2. **Pipeline Consistency**
- All steps use the same temporal splits (via saved config)
- No confusion about which data to use where
- Reproducible results across runs

### 3. **Integrates with Versioned Artifacts**
- Uses efficient view filtering (no full data loads)
- Works with lazy evaluation
- Composable with other view operations

### 4. **Clear API**
```python
# Simple, obvious usage
training_data = get_data_for_purpose(view, 'training', config)
validation_data = get_data_for_purpose(view, 'validation', config)
test_data = get_data_for_purpose(view, 'test', config)
```

---

## Validation

The system validates:

1. **No overlap between periods**:
   ```python
   # This would raise ValueError
   training = TemporalPeriod(start=..., end=datetime(2023, 6, 30), embargo_days=0)
   validation = TemporalPeriod(start=datetime(2023, 6, 1), ...)  # Overlaps!
   config = TemporalSplitConfig(training, validation, test)  # ❌ Raises error
   ```

2. **Proper DatetimeIndex**:
   ```python
   # This would raise ValueError
   df_without_datetime_index = pd.DataFrame(...)
   view.filter(period_filter)  # ❌ Raises error if not DatetimeIndex
   ```

3. **Period validity**:
   ```python
   # This would raise ValueError
   period = TemporalPeriod(
       start=datetime(2023, 1, 1),
       end=datetime(2022, 1, 1)  # End before start!
   )  # ❌ Raises error
   ```

---

## Migration from Old Approach

### Before (Manual Date Filtering):
```python
# Every step manually filters dates - risk of inconsistency
training_data = df[(df.index >= '2020-01-01') & (df.index <= '2023-01-01')]
# Oops, forgot embargo! Data leakage!
```

### After (Temporal Splits):
```python
# Config enforces consistency and embargo
training_view = get_data_for_purpose(view, 'training', config)
training_data = training_view.materialize()
# ✅ Embargo automatically applied
# ✅ Same splits across all steps
```

---

## Best Practices

1. **Create config once, reuse everywhere**:
   ```python
   # Do this at pipeline start
   config = create_temporal_split_config_for_pipeline(...)
   # Config is saved and reused automatically
   ```

2. **Always use get_data_for_purpose()**:
   ```python
   # ✅ Good
   data = get_data_for_purpose(view, 'training', config)

   # ❌ Bad (manual filtering risks inconsistency)
   data = view.filter(lambda df: df.index <= some_date)
   ```

3. **Use CV only within periods**:
   ```python
   # Get test period data
   test_data = get_data_for_purpose(view, 'test', config)

   # Run CV WITHIN the test period
   cv = TimeSeriesSplit(n_splits=5)
   for train_idx, test_idx in cv.split(test_data):
       # These splits are all WITHIN the test period
       # So they're safe from train/val contamination
       ...
   ```

4. **Document your splits**:
   ```python
   # Configs are saved as JSON - commit to git!
   # This ensures everyone uses the same splits
   git add config/temporal_splits/ETHUSDT_binance_15m.json
   ```

---

## Troubleshooting

### "Period overlaps" error
- Increase embargo_days in the earlier period
- Adjust period boundaries to create more space

### "No data in period" error
- Check that your data actually spans the configured periods
- Use `config.training.start` to verify against your data range

### "Must have DatetimeIndex" error
- Ensure your DataFrame has a DatetimeIndex:
  ```python
  df.index = pd.to_datetime(df.index)
  ```

---

## Summary

The temporal splitting system ensures **zero data leakage** by:
1. Enforcing strict temporal boundaries
2. Adding embargo periods between splits
3. Integrating seamlessly with versioned artifacts
4. Providing a clean, validated API

Use `get_data_for_purpose(view, 'training'|'validation'|'test', config)` in every step to guarantee proper data separation!
