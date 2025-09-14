# Final Feature Selection Pipeline Guide

This guide explains how to use the new multi-stage feature selection pipeline that runs at the end of the market analysis pipeline.

## Overview

The Final Feature Selection Pipeline implements a progressive feature reduction system:
- **Stage 1**: 120 → 100 features using RandomForest importance
- **Stage 2**: 100 → 80 features using SHAP (if available) or enhanced RandomForest
- **Stage 3**: 80 → 60 features using combined importance and cross-validation

## Pipeline Integration

The feature selection is automatically integrated into the market analysis pipeline as **Step 7** (final step):

```python
# The pipeline now includes:
steps_to_execute = [
    ('hmm_clustering', 'HMM Clustering'),
    ('regime_splitting', 'Regime Data Splitting'), 
    ('labeling', 'Triple Barrier Labeling'),
    ('feature_engineering', 'Feature Engineering'),
    ('matrix_operations', 'Matrix Operations'),
    ('feature_selection', 'Feature Selection'),
    ('final_feature_selection', 'Final Feature Selection (120→100→80→60)')  # NEW!
]
```

## Usage Examples

### 1. Automatic Integration (Recommended)

The feature selection runs automatically at the end of the market analysis pipeline:

```python
from src.training.steps.market_analysis.enhanced_market_analysis_orchestrator import run_enhanced_market_analysis_pipeline

# Run the complete pipeline - final feature selection happens automatically
config = {
    'force_rerun': True,
    'final_feature_selection': True,  # Enable final feature selection
    'final_feature_selection_config': {
        'initial_features': 120,
        'stage_1_target': 100,
        'stage_2_target': 80,
        'stage_3_target': 60,
        'rf_n_estimators': 100,
        'cv_folds': 5,
        'save_analysis': True
    }
}

success = await run_enhanced_market_analysis_pipeline(
    symbol='ETHUSDT', 
    exchange='BINANCE', 
    timeframe='1m', 
    data_dir='historical_data',
    **config
)
```

### 2. Standalone Usage

You can also run the feature selection independently:

```python
from src.training.steps.market_analysis.final_feature_selection_step import run_final_feature_selection_step

# Run only the final feature selection
config = {
    'initial_features': 120,
    'stage_1_target': 100,
    'stage_2_target': 80,
    'stage_3_target': 60,
    'rf_n_estimators': 150,
    'cv_folds': 5,
    'save_analysis': True,
    'output_directory': 'custom_feature_selection'
}

success = await run_final_feature_selection_step(
    symbol='ETHUSDT',
    exchange='BINANCE', 
    timeframe='1m',
    data_dir='historical_data',
    config=config
)
```

### 3. Direct Pipeline Usage

Use the feature selection pipeline directly:

```python
from src.training.steps.market_analysis.final_feature_selection_pipeline import (
    MultiStageFeatureSelector, FeatureSelectionConfig, run_final_feature_selection
)
import pandas as pd
import numpy as np

# Create sample data
X = pd.DataFrame(np.random.randn(1000, 150))  # 1000 samples, 150 features
y = pd.Series(np.random.randn(1000))  # Target variable

# Configure feature selection
config = FeatureSelectionConfig(
    initial_features=120,
    stage_1_target=100,
    stage_2_target=80,
    stage_3_target=60,
    rf_n_estimators=100,
    cv_folds=5,
    save_analysis=True,
    verbose=True
)

# Run feature selection
result = run_final_feature_selection(X, y, config)

# Get final selected features
final_features = result.final_features
print(f"Selected {len(final_features)} features: {final_features[:10]}...")
```

## Configuration Options

### FeatureSelectionConfig Parameters

```python
@dataclass
class FeatureSelectionConfig:
    # Stage targets
    initial_features: int = 120      # Starting number of features
    stage_1_target: int = 100        # Target after stage 1
    stage_2_target: int = 80         # Target after stage 2  
    stage_3_target: int = 60         # Final target
    
    # RandomForest parameters
    rf_n_estimators: int = 100       # Number of trees
    rf_max_depth: int = 10           # Max tree depth
    rf_min_samples_split: int = 5    # Min samples to split
    rf_random_state: int = 42        # Random seed
    
    # SHAP parameters
    shap_sample_size: int = 1000     # Sample size for SHAP
    shap_max_features: int = 200     # Max features for SHAP
    
    # Cross-validation
    cv_folds: int = 5                # CV folds
    cv_scoring: str = 'neg_mean_squared_error'  # Scoring metric
    
    # Quality thresholds
    min_feature_importance: float = 0.001      # Min importance threshold
    min_correlation_threshold: float = 0.95    # Max correlation allowed
    min_variance_threshold: float = 0.01       # Min variance threshold
    
    # Output settings
    save_models: bool = True         # Save trained models
    save_analysis: bool = True       # Save analysis results
    output_directory: str = "feature_selection_results"  # Output dir
    verbose: bool = True             # Verbose logging
```

## Algorithm Details

### Stage 1: RandomForest Importance (120→100)
- Trains RandomForest with 100 estimators
- Uses feature importance scores for ranking
- Selects top 100 features

### Stage 2: SHAP or Enhanced RF (100→80)
- **If SHAP available**: Uses SHAP values for more sophisticated selection
- **If SHAP unavailable**: Uses ensemble of RandomForest models with different parameters
- Selects top 80 features

### Stage 3: Combined Importance (80→60)
- Combines RandomForest importance with cross-validation scores
- Uses multiple CV folds for stability
- Selects top 60 features

## Output Files

The pipeline generates several output files:

```
final_feature_selection/
├── {symbol}_{timeframe}_final_features.json          # Final selected features
├── {symbol}_{timeframe}_selection_results.json       # Detailed results
├── feature_selection_results_{timestamp}.json        # Pipeline results
└── final_feature_selection_model_{timestamp}.joblib  # Trained model
```

### Final Features JSON Format

```json
{
  "symbol": "ETHUSDT",
  "exchange": "BINANCE", 
  "timeframe": "1m",
  "final_features": ["feature_1", "feature_2", ...],
  "feature_count": 60,
  "selection_method": "multi_stage_rf_shap",
  "stages": {
    "stage_1": 100,
    "stage_2": 80, 
    "stage_3": 60,
    "final": 60
  }
}
```

## Results Analysis

### FeatureSelectionResult Object

```python
result = run_final_feature_selection(X, y, config)

# Access results
print(f"Final features: {result.final_features}")
print(f"Feature counts: {result.feature_counts}")
print(f"Stage scores: {result.stage_1_scores}")
print(f"Selection time: {result.selection_time}s")

# Access model performance
print(f"CV score: {result.final_scores['cv_mean']}")
print(f"Model score: {result.final_scores['model_score']}")
```

### Key Metrics

- **Feature Counts**: Shows reduction at each stage
- **Stage Scores**: Quality metrics for each selection stage
- **CV Performance**: Cross-validation scores for final model
- **Selection Time**: Total time for feature selection
- **Feature Importance**: Importance scores for final features

## Advanced Usage

### Custom Target Counts

```python
# Custom progression: 200→150→100→75
config = FeatureSelectionConfig(
    initial_features=200,
    stage_1_target=150,
    stage_2_target=100, 
    stage_3_target=75
)
```

### Unsupervised Selection

```python
# No target variable - uses variance and correlation filtering
result = run_final_feature_selection(X, None, config)
```

### Integration with Existing Pipeline

```python
# Load existing features and run final selection
from pathlib import Path
import pandas as pd

# Load features from previous pipeline step
features_file = Path("historical_data/ethusdt_1m_features.parquet")
X = pd.read_parquet(features_file)

# Run final selection
result = run_final_feature_selection(X, None, config)
final_features = result.final_features

# Save final features for model training
final_features_df = X[final_features]
final_features_df.to_parquet("final_selected_features.parquet")
```

## Performance Considerations

### Memory Usage
- Large datasets (>10K features) may require chunking
- SHAP analysis is memory-intensive for large feature sets
- Consider reducing `shap_max_features` for large datasets

### Time Complexity
- Stage 1: O(n_features × n_samples × n_estimators)
- Stage 2: O(n_features × n_samples × shap_samples) 
- Stage 3: O(n_features × n_samples × cv_folds)

### Recommendations
- Use `rf_n_estimators=50-100` for faster execution
- Set `shap_sample_size=500` for large datasets
- Use `cv_folds=3` for faster cross-validation

## Troubleshooting

### Common Issues

1. **Insufficient Features**: If input has fewer features than target, all features are selected
2. **SHAP Import Error**: Falls back to enhanced RandomForest automatically
3. **Memory Issues**: Reduce `shap_sample_size` or `shap_max_features`
4. **Long Execution**: Reduce `rf_n_estimators` or `cv_folds`

### Debug Mode

```python
config = FeatureSelectionConfig(
    verbose=True,
    save_analysis=True,
    output_directory="debug_feature_selection"
)
```

## Integration with Model Training

After feature selection, use the selected features for model training:

```python
# Get final features
final_features = result.final_features

# Prepare training data
X_train = X_train[final_features]
X_test = X_test[final_features]

# Train your model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model score with selected features: {score}")
```

This completes the integration of the multi-stage feature selection pipeline into the market analysis workflow!