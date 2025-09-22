# Global HMM Classifier Guide

## Overview

The Global HMM Classifier provides a single model approach to predict probability distributions over all 20 HMM states simultaneously. This is a shift from the per-regime training approach to a unified global classifier.

## Key Features

### 1. **Single Model for All States**
- One model predicts all 20 HMM states
- Returns probability distribution: `[p(state_0), p(state_1), ..., p(state_19)]`
- Eliminates need for regime-specific model selection

### 2. **Updated Objective Weights**
- **Accuracy (50%)**: Increased focus on overall classification correctness
- **F1-Score (35%)**: Balanced precision/recall across all states
- **Regime Stability (15%)**: Reduced weight for global approach

### 3. **Fast-Fail Feature Generation**
- No fallback to degraded features
- Fails immediately if comprehensive feature generation fails
- Ensures data quality for training

## Usage Examples

### Basic Global Classification

```python
from src.training.steps.market_analysis.hmm_models_training import (
    execute_global_hmm_training,
    GlobalHMMClassifier
)

# Train global classifier
results = execute_global_hmm_training(
    X=features,           # Input features
    y=hmm_state_labels,   # HMM state labels (0-19)
    regime_labels=regimes, # Regime labels for context
    feature_names=feature_names
)

# Get best model
best_model = results['models'][results['enhanced_reporting']['best_model_recommendation']['best_model']]

# Predict state probabilities for new data
state_probabilities = best_model.predict_state_probabilities(new_features)
# Returns array of shape (n_samples, 20) with probabilities

# Predict dominant state
dominant_states = best_model.predict_dominant_state(new_features)
# Returns array of shape (n_samples,) with most likely state indices
```

### Custom Configuration

```python
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
from src.training.steps.market_analysis.hmm_models_training import create_global_hmm_training

# Custom configuration
config = HMMTrainingConfig(
    model_name="custom_global_hmm",
    timeframe="15m",
    hpo_trials=100,
    enable_multi_objective=True,
    objectives=["accuracy", "f1_score", "regime_stability"],
    objective_weights=[0.5, 0.35, 0.15]  # Updated weights
)

# Create and execute training
training_step = create_global_hmm_training(config)
results = training_step.execute(X, y, regime_labels, feature_names)
```

### Model Selection and Evaluation

```python
# Get evaluation results
evaluation_results = results['evaluation_results']

# Compare models
for model_type, eval_results in evaluation_results.items():
    if 'error' not in eval_results:
        print(f"{model_type}:")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  F1-Macro: {eval_results['f1_macro']:.4f}")
        print(f"  Objective Score: {eval_results['objective_score']:.4f}")

# Get best model recommendation
best_recommendation = results['enhanced_reporting']['best_model_recommendation']
print(f"Best model: {best_recommendation['best_model']}")
print(f"Objective score: {best_recommendation['objective_score']:.4f}")
```

## Model Types

### 1. **Multi-Class Logistic Regression**
```python
# Fast, interpretable, good baseline
LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)
```

### 2. **Multi-Class LightGBM**
```python
# High performance, fast training
LGBMClassifier(
    objective='multiclass',
    num_class=20,
    metric='multi_logloss'
)
```

### 3. **Multi-Class XGBoost**
```python
# Gradient boosting with multi-class objective
XGBClassifier(
    objective='multi:softprob',
    num_class=20,
    eval_metric='mlogloss'
)
```

### 4. **Multi-Class Random Forest**
```python
# Robust ensemble method
RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1
)
```

### 5. **Multi-Class CatBoost**
```python
# Advanced gradient boosting
CatBoostClassifier(
    objective='MultiClass',
    classes_count=20,
    verbose=False
)
```

## Evaluation Metrics

### Standard Metrics
- **Accuracy**: Overall classification correctness
- **F1-Macro**: Balanced F1-score across all states
- **F1-Weighted**: F1-score weighted by class frequency

### HMM-Specific Metrics
- **State Distribution Accuracy**: Average probability assigned to true states
- **State Transition Consistency**: Accuracy of state transitions
- **Objective Score**: Weighted combination of all objectives

## Key Differences from Per-Regime Training

### Per-Regime Approach (Previous)
- Multiple models (one per regime)
- Regime-specific optimization
- Complex model selection logic
- Higher regime stability weight (30%)

### Global Approach (New)
- Single model for all states
- Unified optimization across states
- Simpler model selection
- Lower regime stability weight (15%)

## Benefits of Global Approach

1. **Simplified Deployment**: Single model to manage
2. **Unified Predictions**: Consistent probability distributions
3. **Better State Relationships**: Model learns state transitions
4. **Reduced Complexity**: No regime-specific logic needed
5. **Improved Accuracy**: Focus on overall classification performance

## Migration from Per-Regime Training

### Before (Per-Regime)
```python
# Multiple models per regime
results = execute_enhanced_hmm_models_training(X, y, regime_labels)
# Complex model selection per regime
```

### After (Global)
```python
# Single model for all states
results = execute_global_hmm_training(X, y, regime_labels)
# Unified model selection
```

## Error Handling

### Fast-Fail Feature Generation
```python
# Old: Fallback to degraded features
try:
    enhanced_features = create_enhanced_features(X, regime_labels)
except:
    enhanced_features = X  # Fallback

# New: Fast-fail with clear error
try:
    enhanced_features = create_enhanced_features(X, regime_labels)
except Exception as e:
    raise ValueError(f"Feature generation failed: {e}")
```

## Best Practices

1. **Data Validation**: Ensure HMM states are in range [0, 19]
2. **Feature Quality**: Use comprehensive feature bank when possible
3. **Model Selection**: Compare all 5 model types for best performance
4. **Probability Interpretation**: Use full probability distributions, not just dominant states
5. **Monitoring**: Track state transition consistency in production

## Production Deployment

```python
# Load trained global classifier
classifier = load_model('best_global_hmm_classifier.pkl')

# Real-time prediction
def predict_hmm_state(features):
    # Get probability distribution
    probabilities = classifier.predict_state_probabilities(features)
    
    # Get dominant state
    dominant_state = np.argmax(probabilities, axis=1)
    
    # Get confidence (max probability)
    confidence = np.max(probabilities, axis=1)
    
    return {
        'state': dominant_state[0],
        'confidence': confidence[0],
        'probabilities': probabilities[0]
    }
```

## Troubleshooting

### Common Issues

1. **HMM State Range Error**
   ```
   ValueError: HMM states must be in range [0, 19], found: [20, 21, 22]
   ```
   Solution: Ensure all HMM states are properly mapped to range [0, 19]

2. **Feature Generation Failure**
   ```
   ValueError: Feature generation failed: Missing OHLCV data
   ```
   Solution: Provide proper OHLCV DataFrame or use enhanced features

3. **Memory Issues with Large Datasets**
   Solution: Use batch processing or reduce feature dimensionality

### Performance Optimization

1. **Use LightGBM or XGBoost** for large datasets
2. **Reduce HPO trials** for faster training
3. **Use feature selection** to reduce dimensionality
4. **Enable parallel processing** where possible

## Conclusion

The Global HMM Classifier provides a streamlined, unified approach to HMM state prediction with improved accuracy and simplified deployment. The updated objective weights and fast-fail feature generation ensure high-quality training results.