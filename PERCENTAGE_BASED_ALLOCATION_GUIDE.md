# Percentage-Based Sample Allocation & Hyperparameter Optimization

## Overview

The training system has been updated to use **percentage-based sample allocation** instead of absolute numbers, and includes **automatic hyperparameter optimization** before training each model.

## Changes Made

### 1. Percentage-Based Sample Allocation

Previously, the config files (e.g., `tactician_base_config.yaml`, `analyst_base_config.yaml`) specified absolute sample counts:
```yaml
training:
  training_samples: 8000
  validation_samples: 2500
  test_samples: 1500
```

**Now**, these absolute numbers are **automatically overridden** by percentage-based calculations in `unified_models_training_step.py`.

#### Default Percentages
- **Training**: 70% of total samples
- **Validation**: 15% of total samples
- **Test**: 15% of total samples
- **CV Folds**: 5 (configurable)

#### How It Works

When training is initiated, the system:
1. Retrieves the total number of available samples
2. Calculates sample allocations based on percentages
3. Overrides the YAML config values with calculated counts
4. Logs the allocation details

Example output:
```
Sample allocation for 10000 total samples:
  Training: 7000 (70.0%)
  Validation: 1500 (15.0%)
  Test: 1500 (15.0%)
  CV Folds: 5
```

### 2. Hyperparameter Optimization (HPO)

The system now automatically performs hyperparameter optimization **before training each model**.

#### Supported Models
- **LightGBM**: n_estimators, learning_rate, max_depth, num_leaves, subsample, colsample_bytree
- **CatBoost**: iterations, learning_rate, depth, l2_leaf_reg
- **TCN/Temporal Models**: hidden_size, num_layers, kernel_size, dropout, learning_rate
- **GRU/LSTM**: hidden_units, num_layers, dropout, learning_rate

#### HPO Process
1. Splits data into HPO training (80%) and validation (20%)
2. Defines search space based on model type
3. Runs Bayesian TPE optimization (default: 20 trials)
4. Updates model config with best parameters
5. Trains final model with optimized hyperparameters

## Usage

### Default Behavior

No changes needed! The system automatically uses percentage-based allocation and HPO:

```python
# Training will automatically use percentage-based allocation and HPO
python ares_launcher.py train --training_type analyst_base --symbol ETHUSDT --timeframe 15m
```

### Custom Percentages

Override default percentages in your config:

```python
config = {
    'symbol': 'ETHUSDT',
    'timeframe': '15m',
    'training_type': 'tactician_base',
    'train_percentage': 0.75,      # 75% for training
    'validation_percentage': 0.15,  # 15% for validation
    'test_percentage': 0.10,        # 10% for testing
    'cv_folds': 5
}
```

### Disable HPO

To disable hyperparameter optimization:

```python
config = {
    'symbol': 'ETHUSDT',
    'timeframe': '15m',
    'training_type': 'analyst_base',
    'enable_hpo': False  # Disable HPO
}
```

### Custom HPO Trials

Adjust the number of HPO trials:

```python
config = {
    'symbol': 'ETHUSDT',
    'timeframe': '15m',
    'training_type': 'tactician_base',
    'enable_hpo': True,
    'hpo_max_trials': 50  # Run 50 HPO trials instead of default 20
}
```

## Benefits

### Percentage-Based Allocation
1. **Flexibility**: Works with any dataset size
2. **Consistency**: Same split ratios across different datasets
3. **Scalability**: Automatically adjusts to data availability
4. **No Manual Updates**: No need to update config files when data changes

### Hyperparameter Optimization
1. **Better Performance**: Automatically finds optimal hyperparameters
2. **Time Savings**: No manual tuning required
3. **Consistent Results**: Systematic optimization process
4. **Model-Specific**: Optimizes based on model type

## Modified Files

### Core Implementation
- `src/training/steps/model_training/unified_models_training_step.py`
  - Added `_calculate_sample_allocations()` method
  - Added `_override_training_config_with_allocations()` method
  - Added `_perform_hyperparameter_optimization()` method
  - Added `_get_hpo_search_space()` method
  - Updated `execute()` method to use percentage-based allocation and HPO

### Configuration Files (Documentation Updates)
- `src/training/steps/model_training/analyst_base_config.yaml`
- `src/training/steps/model_training/analyst_ensemble_config.yaml`
- `src/training/steps/model_training/tactician_base_config.yaml`
- `src/training/steps/model_training/tactician_ensemble_config.yaml`

All config files now include comments explaining that absolute sample numbers are deprecated and overridden by percentage-based calculations.

## Example Workflow

### Analyst Base Training
```python
# 1. Load training data (e.g., 50,000 samples)
# 2. System calculates allocation:
#    - Training: 35,000 (70%)
#    - Validation: 7,500 (15%)
#    - Test: 7,500 (15%)
# 3. Perform HPO for LGBMRegressor, TCN, CatBoost
# 4. Train models with optimized hyperparameters
# 5. Validate and save models
```

### Tactician Base Training
```python
# 1. Load training data (e.g., 10,000 samples)
# 2. System calculates allocation:
#    - Training: 7,000 (70%)
#    - Validation: 1,500 (15%)
#    - Test: 1,500 (15%)
# 3. Perform HPO for StandaloneGRU, TacticianLGBM
# 4. Train models with optimized hyperparameters
# 5. Validate and save models
```

## Logging

The system provides detailed logging:

```
🚀 Starting unified analyst_base training for ETHUSDT 15m long
Sample allocation for 50000 total samples:
  Training: 35000 (70.0%)
  Validation: 7500 (15.0%)
  Test: 7500 (15.0%)
  CV Folds: 5
Updated analyst_config with calculated allocations
✅ Configured training with percentage-based allocations
🔍 Performing hyperparameter optimization before training...
🔍 Starting hyperparameter optimization...
Running HPO with 28000 training samples and 7000 validation samples
Running 20 HPO trials...
✅ Updated lgbm with optimized hyperparameters
✅ Updated catboost with optimized hyperparameters
✅ Hyperparameter optimization completed
```

## Migration Notes

### For Existing Code
- No changes required! The system is backward compatible
- Absolute numbers in YAML files are now ignored (but kept for documentation)
- All training will use percentage-based allocation automatically

### For New Models
- Add HPO search space in `_get_hpo_search_space()` method
- Search spaces defined for: LGBM, CatBoost, TCN, GRU, LSTM
- System falls back to default parameters if search space not defined

## Testing

To verify the changes work correctly:

1. **Check Sample Allocation**:
   - Run training and verify log output shows percentage-based allocation
   - Verify total samples = training + validation + test

2. **Check HPO Execution**:
   - Verify HPO logs appear before training
   - Verify "Updated X with optimized hyperparameters" messages
   - Verify training uses optimized parameters

3. **Check Model Performance**:
   - Compare model performance before/after HPO
   - Verify models train successfully with new allocations
   - Check validation metrics are reasonable

## Troubleshooting

### HPO Not Running
- Check `enable_hpo=True` in config
- Verify training data is available
- Check HPO utilities are installed: `BayesianTPEOptimizer`, `AutoTuner`

### Allocation Issues
- Verify training data has sufficient samples (minimum ~100)
- Check percentages sum to 1.0 (system will normalize if not)
- Ensure data is not None before allocation

### Performance Issues
- Reduce `hpo_max_trials` for faster training
- Disable HPO for quick experiments: `enable_hpo=False`
- Use light mode for reduced sample count

## Future Enhancements

Potential improvements:
1. Advanced HPO algorithms (e.g., BOHB, Hyperband)
2. Multi-objective optimization (accuracy + training time)
3. Transfer learning from previous HPO runs
4. Adaptive percentage allocation based on data quality
5. Integration with MLflow for experiment tracking

## Support

For questions or issues:
1. Check logs for detailed error messages
2. Verify config files are properly formatted
3. Ensure all dependencies are installed
4. Review this guide for usage examples

