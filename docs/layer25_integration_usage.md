# Layer 2.5 → Layer 3 Integration Usage Guide

## Overview

The Layer 2.5 Chaser integration automatically feeds the top 2-3 performing Chaser models into Layer 3 as additional features. This enhances Layer 3's predictive power by incorporating non-linear residual learning from Layer 2.5.

## Configuration

### Enable Integration
```python
config = {
    'layer25_chaser_enabled': True,          # Enable integration (default: True)
    'layer25_top_models': 3,                  # Number of top models to use (default: 3)
    'layer25_chaser_results': chaser_data,   # Results from Layer 2.5 training
    'symbol': 'ETHUSDT',
    'exchange': 'binance', 
    'timeframe': '15m'
}
```

### Disable Integration
```python
config = {
    'layer25_chaser_enabled': False,         # Disable integration
    # ... other config
}
```

## Required Input Format

The `layer25_chaser_results` should be a dictionary with this structure:

```python
chaser_results = {
    'geometry_uuid_1': {
        'meta_features': {
            'chaser_xgb_regression': {
                'model': xgb_model_object,
                'predictions_oof': np.array([...]),  # OOF predictions
                'performance': {'auc': 0.65, 'ic': 0.12},
                'features': ['feature1', 'feature2', ...]
            },
            'chaser_lgb_classification': {
                'model': lgb_model_object,
                'predictions_oof': np.array([...]),
                'performance': {'auc': 0.68, 'pr_auc': 0.45},
                'features': ['feature1', 'feature2', ...]
            },
            # ... more chaser models
        }
    },
    'geometry_uuid_2': {
        'meta_features': {
            # ... more geometry results
        }
    }
    # ... more geometries
}
```

## Integration Process

### 1. Model Selection
- Automatically extracts all chaser models from results
- Ranks by performance (AUC for classification, IC for regression)
- Selects top N models (default: 3)

### 2. Artifact Storage
- **Models**: Saved using artifact manager with unique IDs
- **Predictions**: OOF predictions stored as parquet files
- **Metadata**: Performance metrics and model info saved as JSON

### 3. Feature Integration
- Chaser predictions added as features: `chaser_top_1_model_name`
- Features aligned with Layer 3 DataFrame index
- Missing values handled with forward fill

### 4. Reporting
- Integration report generated in `outcomes/`
- Artifact paths logged for reproducibility
- Performance metrics tracked

## Example Usage

### Step 1: Run Layer 2.5 Chaser
```python
from src.training.steps.labeling.layer2_5_chaser import Layer25Chaser

# Train Layer 2.5 chaser models
chaser = Layer25Chaser(...)
chaser_results = chaser.fit_predict(X, y, sample_weight=weights)
```

### Step 2: Run Layer 3 with Integration
```python
from src.training.steps.labeling.label_based_layer_3 import layer3_analyst_lgbm

# Configure integration
config = {
    'layer25_chaser_enabled': True,
    'layer25_top_models': 3,
    'layer25_chaser_results': chaser_results,
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    # ... other Layer 3 config
}

# Run Layer 3 (automatically integrates Layer 2.5)
df_result, models_dict = layer3_analyst_lgbm(
    oof_df=oof_data,
    base_model_cols=base_features,
    target_col='target',
    config=config
)
```

## Output Artifacts

### Saved Artifacts
```
outcomes/layer25_artifacts/
├── layer25_chaser_top_1_chaser_xgb_regression_20260118_192700.pkl
├── layer25_chaser_top_1_chaser_xgb_regression_predictions_20260118_192700.parquet
├── layer25_chaser_top_1_chaser_xgb_regression_metadata_20260118_192700.json
├── layer25_chaser_top_2_chaser_lgb_classification_20260118_192700.pkl
└── ...
```

### Integration Report
```
outcomes/layer25_integration_report_20260118_192700.md
```

### Enhanced Layer 3 DataFrame
```python
# New chaser features added to Layer 3
chaser_features = [col for col in df_result.columns if col.startswith('chaser_')]
print(f"Added {len(chaser_features)} chaser features:")
# Output: chaser_top_1_chaser_xgb_regression, chaser_top_2_chaser_lgb_classification, ...
```

## Performance Benefits

### Expected Improvements
- **Enhanced Alpha**: Non-linear residual learning from Layer 2.5
- **Model Diversity**: Additional model families beyond Layer 3's 6 models
- **Regime Awareness**: Layer 2.5's regime-specialized models
- **Uncertainty Weighting**: Focus on areas where linear models struggle

### Model Race Impact
- Chaser models appear in Layer 3 model race reports
- Additional candidates for best model selection
- Ensemble diversity metrics include chaser correlations

## Troubleshooting

### Common Issues

#### 1. No Layer 2.5 Results Available
```
⚠️ Skipping Layer 2.5 Chaser integration (disabled or no results)
```
**Solution**: Ensure `layer25_chaser_results` is properly configured

#### 2. Integration Failed
```
⚠️ Layer 2.5 integration failed: [error message]
```
**Solution**: Check chaser results format and model compatibility

#### 3. Artifact Manager Not Available
**Solution**: System falls back to local storage in `outcomes/layer25_artifacts/`

### Debug Mode
```python
# Enable verbose logging
config['layer25_chaser_enabled'] = True
config['layer25_chaser_results'] = chaser_results
# Run Layer 3 and check logs for integration details
```

## Best Practices

### 1. Model Selection
- Use `layer25_top_models: 2-3` for optimal diversity vs complexity
- Higher values may cause overfitting with too many correlated features

### 2. Performance Monitoring
- Monitor chaser feature importance in Layer 3 model race
- Check correlation between chaser and Layer 3 features
- Validate that integration improves overall performance

### 3. Storage Management
- Artifacts are automatically managed by artifact manager
- Local storage fallback ensures no data loss
- Integration reports provide full traceability

### 4. Reproducibility
- All artifacts have unique IDs with timestamps
- Metadata includes model parameters and performance
- Integration can be reproduced from saved artifacts

## Advanced Usage

### Custom Model Selection
```python
# Override top model selection
config['layer25_top_models'] = 2  # Use only top 2 models
config['layer25_chaser_results'] = chaser_results
```

### Manual Feature Loading
```python
from src.training.steps.labeling.layer3.layer25_integration import Layer25Integration

# Load specific artifacts
integrator = Layer25Integration('ETHUSDT', 'binance', '15m')
predictions = integrator.load_chaser_predictions(artifact_paths)
```

### Integration Metadata Access
```python
# Access integration results
integration_metadata = config.get('layer25_integration', {})
print(f"Status: {integration_metadata.get('status')}")
print(f"Features added: {integration_metadata.get('features_added')}")
print(f"Report: {integration_metadata.get('report_path')}")
```

This integration provides a seamless way to enhance Layer 3 with Layer 2.5's specialized residual learning capabilities while maintaining full traceability and reproducibility.
