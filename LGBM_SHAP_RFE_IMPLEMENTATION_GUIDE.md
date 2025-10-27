# LGBM-SHAP RFE Feature Selection Implementation Guide

## Overview

This implementation provides a comprehensive LGBM-SHAP with Recursive Feature Elimination (RFE) feature selection system specifically designed for regime models training. The system removes 25% of features at each iteration until reaching the target of 60 features, with detailed logging and comprehensive reporting.

## Key Features

- **LGBM-SHAP Integration**: Combines LightGBM importance scores with SHAP values for robust feature ranking
- **Recursive Feature Elimination**: Iteratively removes 25% of least important features
- **Comprehensive Logging**: Detailed tprint logging for removed features and remaining counts
- **Detailed Reporting**: Global and per-feature metrics saved to outcomes/ directory
- **Regime-Specific Optimization**: Tailored for tree-based regime models training
- **Robust Error Handling**: Handles NaN values and data validation

## File Structure

```
/workspace/src/feature_generation/
├── feature_selection/
│   └── lgbm_shap_rfe_selector.py          # Core LGBM-SHAP RFE selector
└── integration/
    └── lgbm_shap_rfe_integration.py       # Integration with enhanced models training

/workspace/
├── test_lgbm_shap_rfe_integration.py      # Full test with realistic data
├── simple_test_lgbm_shap_rfe.py           # Basic structure test
└── LGBM_SHAP_RFE_IMPLEMENTATION_GUIDE.md  # This guide
```

## Dependencies

### Required Packages
```bash
pip install numpy pandas lightgbm shap scipy scikit-learn
```

### Optional Packages (for enhanced performance)
```bash
pip install vectorbt  # For VectorBT optimizations
```

## Usage Examples

### Basic Usage

```python
import numpy as np
import pandas as pd
from src.feature_generation.integration.lgbm_shap_rfe_integration import create_lgbm_shap_rfe_integration

# Create sample OHLCV data
data = pd.DataFrame({
    'open': np.random.randn(1000) * 0.01 + 100,
    'high': np.random.randn(1000) * 0.01 + 101,
    'low': np.random.randn(1000) * 0.01 + 99,
    'close': np.random.randn(1000) * 0.01 + 100,
    'volume': np.random.lognormal(10, 1, 1000)
})

# Create integration
integration = create_lgbm_shap_rfe_integration(
    target_features=60,
    removal_percentage=0.25,
    enable_detailed_logging=True
)

# Run feature selection
result = integration.select_features_for_regime_training(data)

print(f"Selected {result['selected_features']['count']} features")
print(f"Removed {result['removed_features']['count']} features")
```

### Advanced Configuration

```python
from src.feature_generation.integration.lgbm_shap_rfe_integration import create_lgbm_shap_rfe_integration

# Custom LGBM parameters
custom_lgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

# Create integration with custom parameters
integration = create_lgbm_shap_rfe_integration(
    target_features=60,
    removal_percentage=0.25,
    lgbm_params=custom_lgbm_params,
    enable_detailed_logging=True
)
```

## Configuration Options

### LGBMSHAPRFEConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_features` | int | 60 | Target number of features to select |
| `removal_percentage` | float | 0.25 | Percentage of features to remove per iteration |
| `lgb_params` | dict | {...} | LightGBM parameters |
| `shap_explainer` | str | 'tree' | SHAP explainer type ('tree', 'linear', 'kernel') |
| `shap_sample_size` | int | None | SHAP sample size (None for all samples) |
| `min_features_to_keep` | int | 10 | Minimum features to keep |
| `max_iterations` | int | 20 | Maximum RFE iterations |
| `early_stopping_patience` | int | 3 | Early stopping patience |
| `cv_folds` | int | 5 | Cross-validation folds |
| `validation_size` | float | 0.2 | Validation set size |
| `enable_detailed_logging` | bool | True | Enable detailed logging |
| `save_intermediate_results` | bool | True | Save intermediate results |
| `report_format` | str | 'both' | Report format ('json', 'markdown', 'both') |

## Output and Reporting

### Console Output

The system provides detailed console output with tprint logging:

```
🚀 Starting LGBM-SHAP RFE feature selection
📊 Step 1: Generating comprehensive features
✅ Generated 354 features
🎯 Step 2: Preparing target variable
📈 Created synthetic target (future returns)
🔗 Step 3: Aligning features and target
✅ Aligned data: 1000 samples, 354 features
🔍 Step 4: Running LGBM-SHAP RFE selection

🔄 RFE Iteration 1
📈 Current features: 354
🎯 Will remove 89 features (25% of 354)
🗑️ Removing 89 features:
   1. feature_123 (score: 0.000001)
   2. feature_456 (score: 0.000002)
   ...
✅ Iteration 1 complete. Features remaining: 265

🔄 RFE Iteration 2
📈 Current features: 265
🎯 Will remove 66 features (25% of 265)
...
```

### Detailed Reports

Reports are automatically saved to the `outcomes/` directory with timestamps:

#### JSON Report (`lgbm_shap_rfe_report_YYYYMMDD_HHMMSS.json`)
```json
{
  "timestamp": "2025-10-27T14:30:00.000000",
  "config": {
    "target_features": 60,
    "removal_percentage": 0.25,
    "lgb_params": {...}
  },
  "selection_summary": {
    "total_iterations": 8,
    "total_features_removed": 294,
    "final_feature_count": 60,
    "target_feature_count": 60
  },
  "global_metrics": {
    "n_samples": 1000,
    "n_features": 60,
    "avg_correlation": 0.1234,
    "avg_variance": 0.5678
  },
  "per_feature_metrics": [...],
  "selected_features": [...],
  "removed_features": [...]
}
```

#### Markdown Report (`lgbm_shap_rfe_report_YYYYMMDD_HHMMSS.md`)
```markdown
# LGBM-SHAP RFE Feature Selection Report

**Generated:** 2025-10-27T14:30:00.000000

## 📊 Selection Summary
- **Total Iterations:** 8
- **Features Removed:** 294
- **Final Features:** 60
- **Target Features:** 60
- **Removal Percentage:** 25.0%

## 🌍 Global Metrics
- **Samples:** 1,000
- **Features:** 60
- **Avg Correlation:** 0.1234
- **Avg Variance:** 0.5678

## ✅ Selected Features
1. regime_persistence_20
2. vol_regime_strength_20
3. trend_score_14
...
```

## Algorithm Details

### 1. Feature Generation
- Uses `EnhancedModelsTrainingIntegration` to generate comprehensive features
- Supports all feature categories: Volume, Trend, Volatility, Momentum, Regime, Clustering
- Handles NaN values and data validation

### 2. Target Variable Creation
- Creates synthetic target (future returns) if not provided
- Supports custom target columns
- Aligns features and target, removing invalid samples

### 3. LGBM-SHAP RFE Process
```python
while len(current_features) > target_features:
    # Train LGBM model
    model, performance = train_lgbm_model(X_current, y)
    
    # Calculate importance and SHAP values
    importance_scores = model.feature_importance()
    shap_values = calculate_shap_values(model, X_current)
    
    # Combine scores
    combined_scores = 0.5 * importance_normalized + 0.5 * shap_normalized
    
    # Remove 25% of least important features
    features_to_remove = select_lowest_scoring_features(combined_scores, 25%)
    
    # Update feature list
    current_features = remove_features(current_features, features_to_remove)
```

### 4. Score Combination
- **LGBM Importance**: Normalized gain-based importance scores
- **SHAP Values**: Mean absolute SHAP values across samples
- **Combined Score**: 50% importance + 50% SHAP (equal weights)

### 5. Feature Removal Strategy
- Removes 25% of features per iteration
- Selects features with lowest combined scores
- Maintains minimum feature count (default: 10)
- Supports early stopping based on performance

## Performance Optimization

### Memory Efficiency
- Processes features in batches when possible
- Uses efficient numpy operations
- Handles large datasets with chunked processing

### Computational Efficiency
- Vectorized operations where possible
- Parallel processing for SHAP calculations
- Early stopping to prevent unnecessary iterations

### Hardware Optimization
- M1/M2/M3 Mac optimizations when available
- GPU acceleration for SHAP calculations
- Memory-efficient data structures

## Error Handling

### Data Validation
- Checks for valid OHLCV data
- Handles missing values gracefully
- Validates feature-target alignment

### Dependency Management
- Graceful fallback when SHAP is unavailable
- Handles missing optional dependencies
- Provides informative error messages

### Robustness
- Handles edge cases (empty data, single features)
- Prevents infinite loops
- Provides recovery mechanisms

## Integration with Existing Systems

### Enhanced Models Training Integration
```python
from src.feature_generation.integration.enhanced_models_training_integration import EnhancedModelsTrainingIntegration

# The LGBM-SHAP RFE integration extends the enhanced models training
integration = create_lgbm_shap_rfe_integration()
result = integration.select_features_for_regime_training(data)
```

### Feature Bank Integration
- Automatically uses the feature bank system
- Supports all feature categories
- Maintains feature metadata and categories

### Regime Models Training
- Optimized for tree-based models (LGBM, Random Forest, XGBoost)
- Provides regime-specific feature selection
- Maintains feature interpretability

## Testing and Validation

### Unit Tests
```bash
python3 simple_test_lgbm_shap_rfe.py
```

### Integration Tests
```bash
python3 test_lgbm_shap_rfe_integration.py
```

### Performance Tests
- Memory usage monitoring
- Execution time tracking
- Feature quality validation

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify module structure

2. **Memory Issues**
   - Reduce dataset size
   - Use chunked processing
   - Increase system memory

3. **Performance Issues**
   - Adjust LGBM parameters
   - Reduce SHAP sample size
   - Use early stopping

4. **Feature Selection Issues**
   - Check data quality
   - Verify target variable
   - Adjust removal percentage

### Debug Mode
```python
# Enable debug logging
integration = create_lgbm_shap_rfe_integration(
    enable_detailed_logging=True
)

# Check intermediate results
result = integration.select_features_for_regime_training(data)
print(result['selection_process']['history'])
```

## Future Enhancements

### Planned Features
- Support for classification tasks
- Multi-objective optimization
- Advanced feature interaction detection
- Real-time feature selection updates

### Performance Improvements
- GPU acceleration for SHAP
- Distributed processing
- Caching mechanisms
- Incremental updates

### Integration Enhancements
- Web interface
- API endpoints
- Configuration management
- Monitoring dashboards

## Conclusion

The LGBM-SHAP RFE implementation provides a robust, efficient, and comprehensive solution for feature selection in regime models training. With its detailed logging, comprehensive reporting, and optimized performance, it serves as a powerful tool for selecting the most relevant features from large feature sets.

The system is designed to be both user-friendly for basic usage and highly configurable for advanced scenarios, making it suitable for a wide range of machine learning applications in financial markets.