# Summary: Temporal Splits with Backward Compatibility

## What Changed

### 1. Embargo Reduced: 30 days → 1 day
- **Default embargo**: 1 day (configurable)
- Still prevents look-ahead bias at period boundaries
- More practical for trading timeframes

### 2. Backward Compatibility ✅
**No changes required for existing code!**

```python
# OLD CODE (still works)
view = versioned_store.get_view("features")
data = view.materialize()  # Gets full dataset

# NEW CODE (with temporal filtering)
config = create_temporal_split_config_for_pipeline(...)
training_view = get_data_for_purpose(view, 'training', config)
training_data = training_view.materialize()  # Gets only training period
```

**Key point:** `get_data_for_purpose(view)` without config returns full view (backward compatible)

### 3. New 'all' Period for Monte Carlo
```python
# Monte Carlo should use all periods combined
all_data = get_data_for_purpose(view, 'all', config)  # Returns full view
```

### 4. Enhanced basic_backtesting_post

**New Config Options:**
```python
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',

    # NEW OPTIONS:
    'backtest_period': 'training',  # 'training' (default), 'test', or 'both'
    'temporal_config': temporal_config  # Optional TemporalSplitConfig
}
```

**Behavior:**
- **backtest_period='training'** (default): Runs on training data (or full data if no temporal_config)
- **backtest_period='test'**: Runs on test data only
- **backtest_period='both'**: Runs on both, compares performance to detect overfitting

### 5. Overfitting Detection

When `backtest_period='both'`, the step compares training vs test performance:

**New Metrics Added:**
```python
metrics = {
    ...
    'train_test_comparison': {
        'total_return_train': 0.25,
        'total_return_test': 0.18,
        'total_return_degradation': 0.28,  # 28% worse on test

        'sharpe_ratio_train': 1.5,
        'sharpe_ratio_test': 1.1,
        'sharpe_ratio_degradation': 0.27,

        ... (for all key metrics)

        'avg_performance_degradation': 0.25,  # 25% average degradation
        'overfitting_detected': False,  # True if >30% degradation
        'generalization_quality': 'moderate'  # 'poor', 'moderate', or 'good'
    }
}
```

**Report Section:**
```markdown
## 🔍 Training vs Test Performance (Overfitting Detection)

✅ **Good Generalization** - Quality: moderate

| Metric | Training | Test | Degradation |
|--------|----------|------|-------------|
| Total Return | 25.00% | 18.00% | 28.0% |
| Sharpe Ratio | 1.500 | 1.100 | 26.7% |
| Win Rate | 65.00% | 58.00% | 10.8% |
...

**Average Performance Degradation:** 25.0%
```

---

## Usage Examples

### Example 1: Backward Compatible (No Changes Needed)

```python
# Existing code works as before
class MyTrainingStep(BaseStep):
    async def execute(self, config):
        # Gets full dataset (training period by default)
        features_view = self.versioned_store.get_view("features")
        data = features_view.materialize()

        # Train models
        models = self.train(data)
        return {'success': True}
```

### Example 2: Model Training with Temporal Splits

```python
class AnalystTrainingStep(BaseStep):
    async def execute(self, config):
        # Create or load temporal config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe'],
            # Only needed first time:
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2025, 1, 1)
        )

        # Get features
        features_view = self.versioned_store.get_view("features_v1")

        # Filter to training period
        training_view = get_data_for_purpose(
            features_view,
            purpose='training',  # Default, can omit
            config=temporal_config
        )

        training_data = training_view.materialize()

        # Train models on training data ONLY
        models = self.train_models(training_data)

        return {'success': True}
```

### Example 3: Parameter Optimization with Validation Period

```python
class FinalParametersOptimizer(BaseStep):
    async def execute(self, config):
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        # Get predictions from trained models
        predictions_view = self.versioned_store.get_view("predictions")

        # Filter to VALIDATION period
        validation_view = get_data_for_purpose(
            predictions_view,
            purpose='validation',
            config=temporal_config
        )

        validation_data = validation_view.materialize()

        # Optimize parameters on validation data
        best_params = self.optimize(validation_data)

        return {'success': True, 'params': best_params}
```

### Example 4: Backtesting on Training Period (Default)

```python
# Default behavior - runs on training period
result = await backtesting_step.execute({
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m'
    # No backtest_period specified → defaults to 'training'
    # No temporal_config → uses full dataset
})
```

### Example 5: Backtesting on Test Period

```python
temporal_config = create_temporal_split_config_for_pipeline(...)

# Run backtest on test period only
result = await backtesting_step.execute({
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'backtest_period': 'test',
    'temporal_config': temporal_config
})

# Result contains test period metrics
print(result['metrics']['total_return'])  # Return on test data
```

### Example 6: Compare Training vs Test (Overfitting Detection)

```python
temporal_config = create_temporal_split_config_for_pipeline(...)

# Run backtest on BOTH periods
result = await backtesting_step.execute({
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'backtest_period': 'both',
    'temporal_config': temporal_config
})

# Check for overfitting
train_test = result['metrics']['train_test_comparison']

if train_test['overfitting_detected']:
    print(f"⚠️ OVERFITTING: {train_test['avg_performance_degradation']:.1%} degradation")
else:
    print(f"✅ Good generalization: {train_test['generalization_quality']}")

# Detailed metrics
print(f"Training Sharpe: {train_test['sharpe_ratio_train']:.2f}")
print(f"Test Sharpe: {train_test['sharpe_ratio_test']:.2f}")
print(f"Degradation: {train_test['sharpe_ratio_degradation']:.1%}")
```

### Example 7: Monte Carlo on All Periods

```python
class MonteCarloSimulation(BaseStep):
    async def execute(self, config):
        temporal_config = create_temporal_split_config_for_pipeline(...)

        data_view = self.versioned_store.get_view("data")

        # Get ALL periods for monte carlo
        all_data_view = get_data_for_purpose(
            data_view,
            purpose='all',  # Special case for monte carlo
            config=temporal_config
        )

        all_data = all_data_view.materialize()

        # Run monte carlo on complete dataset
        mc_results = self.run_monte_carlo(all_data)

        return {'success': True}
```

---

## Next Steps

### 1. Update final_parameters_optimization

Add temporal filtering to use validation period:

```python
# In final_parameters_optimization.py
temporal_config = create_temporal_split_config_for_pipeline(...)

# Get validation period data
validation_view = get_data_for_purpose(view, 'validation', temporal_config)
validation_data = validation_view.materialize()

# Optimize parameters on validation period
```

### 2. Update Other Training Steps

Add temporal filtering to all model training steps:

```python
# In any training step
temporal_config = create_temporal_split_config_for_pipeline(...)
training_view = get_data_for_purpose(view, 'training', temporal_config)
training_data = training_view.materialize()
```

### 3. Run Backtesting with Overfitting Detection

For final validation, use backtest_period='both':

```python
result = await step.execute({
    ...,
    'backtest_period': 'both',
    'temporal_config': temporal_config
})
```

---

## Key Takeaways

✅ **Backward compatible** - existing code works without changes
✅ **1-day embargo** - practical default, still prevents leakage
✅ **Flexible periods** - training (default), validation, test, or all
✅ **Overfitting detection** - automatic when backtest_period='both'
✅ **Clean API** - `get_data_for_purpose(view, purpose, config)`

**Default behavior:** Without temporal config, everything works as before (uses full dataset)

**With temporal config:** Proper train/val/test separation with automatic enforcement
