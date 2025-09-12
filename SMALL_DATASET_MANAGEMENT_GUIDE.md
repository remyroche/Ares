# Small Dataset Management for SR ML Prediction

## Overview

This guide explains how to manage the challenge of training ML models with limited SR level data (91 samples) using advanced techniques specifically designed for small datasets.

## Problem Statement

Your SR ML prediction module faces the following challenges:
- **91 data points** for training
- **247+ features** (high feature-to-sample ratio ≈2.7:1)
- **Risk of overfitting** due to curse of dimensionality
- **Limited generalization** capability

## Solutions Implemented

### 1. Data Augmentation Strategies

#### Gaussian Noise Injection
```python
# Adds controlled noise to existing SR level features
noise_std = np.std(X, axis=0) * 0.05  # 5% noise level
noisy_X = X + np.random.normal(0, noise_std, X.shape)
```

#### Bootstrap Sampling
```python
# Creates variations through sampling with replacement
bootstrap_size = int(n_samples * 0.8)
indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
```

#### Feature Interpolation
```python
# Interpolates between similar SR levels
alpha = 0.5  # Midpoint interpolation
new_X = alpha * X[i] + (1 - alpha) * X[i + 1]
```

#### Synthetic SR Level Generation
```python
# Generates realistic synthetic SR levels based on statistical patterns
synthetic_features = np.random.normal(feature_means, feature_stds * 0.3)
synthetic_features = apply_sr_constraints(synthetic_features, feature_names)
```

### 2. Transfer Learning

#### Source Domain Identification
```python
# Finds most similar source domain
similarity = calculate_feature_similarity(target_X, source_X)
best_source = find_best_source_domain(target_X, source_data)
```

#### Model Adaptation
```python
# Fine-tunes pre-trained models on target data
base_model = train_base_model(source_X, source_y)
adapted_model = fine_tune_model(base_model, target_X, target_y)
```

#### Synthetic Source Data Creation
```python
# Creates market variations for transfer learning
variations = {
    'bull_market': {'trend': 1.1, 'volatility': 0.8},
    'bear_market': {'trend': 0.9, 'volatility': 1.2},
    'sideways_market': {'trend': 1.0, 'volatility': 1.0}
}
```

### 3. Regularized Ensemble Methods

#### Strong Regularization
```python
# Ridge Regression with strong L2 regularization
ridge = Ridge(alpha=1.0, random_state=42)

# Lasso Regression with L1 regularization
lasso = Lasso(alpha=0.1, max_iter=1000, random_state=42)

# Regularized Random Forest
rf = RandomForestRegressor(
    n_estimators=50,      # Reduced for small datasets
    max_depth=3,          # Strong regularization
    min_samples_split=5,
    min_samples_leaf=2
)
```

#### Ensemble Methods
```python
# Voting Regressor
estimators = [('ridge', ridge), ('lasso', lasso), ('rf', rf)]
voting_regressor = VotingRegressor(estimators)

# Stacking Regressor
stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=3  # Small CV for small datasets
)
```

### 4. Feature Selection for Small Datasets

#### Statistical Feature Selection
```python
# SelectKBest with statistical tests
selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
X_selected = selector.fit_transform(X, y)
```

#### Feature Importance Ranking
```python
# Correlation-based feature selection
correlations = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
top_indices = np.argsort(correlations)[-n_features_to_select:]
```

## Integration with Existing SR ML Enhancer

### Automatic Detection and Application

The system automatically detects small datasets and applies appropriate techniques:

```python
# In SRMLEnhancer.train_models()
if len(sr_levels) < 50:  # Small dataset threshold
    self.logger.info(f'🔧 Small dataset detected ({len(sr_levels)} samples) - applying enhanced techniques')
    
    # Apply small dataset techniques
    enhanced_results = small_dataset_manager.enhance_sr_ml_training(
        market_data, sr_levels, historical_performance
    )
```

### Enhanced Prediction

The prediction methods automatically use enhanced models when available:

```python
# In SRMLEnhancer.predict_sr_quality()
if hasattr(self, 'enhanced_training_results') and self.enhanced_training_results:
    enhanced_predictions = small_dataset_manager.predict_with_enhanced_models(
        market_data, sr_levels, model_type='ensemble'
    )
```

## Usage Examples

### Basic Usage

```python
from src.utils.ml_common.small_dataset_integration import integrate_small_dataset_management_with_sr_enhancer

# Initialize small dataset manager
small_dataset_manager = integrate_small_dataset_management_with_sr_enhancer()

# Apply enhanced training
results = small_dataset_manager.enhance_sr_ml_training(
    market_data, sr_levels, historical_performance
)

# Make predictions
predictions = small_dataset_manager.predict_with_enhanced_models(
    market_data, sr_levels, model_type='ensemble'
)
```

### Advanced Configuration

```python
from src.utils.ml_common.small_dataset_integration import SmallDatasetIntegrationConfig

config = SmallDatasetIntegrationConfig(
    enable_data_augmentation=True,
    enable_transfer_learning=True,
    enable_regularized_ensemble=True,
    min_samples_threshold=50,
    augmentation_factor=2.0,
    regularization_strength=1.0,
    feature_selection_ratio=0.3
)

integration = SmallDatasetSRIntegration(config)
```

## Performance Monitoring

### Key Metrics to Monitor

1. **Cross-Validation Scores**: Monitor variance in CV scores
2. **Feature Importance Stability**: Check if top features remain consistent
3. **Prediction Confidence**: Track confidence levels of predictions
4. **Overfitting Indicators**: Watch for high training vs validation gap

### Recommendations Generated

The system automatically generates recommendations:

```python
recommendations = [
    "⚠️ Very small dataset - monitor model performance closely",
    "📊 Consider collecting more SR level data over time",
    "✅ Dataset enhanced from 91 to 182 samples",
    "🔧 Applied techniques: data_augmentation, transfer_learning, regularized_ensemble",
    "🎯 Use ensemble predictions for better reliability"
]
```

## Best Practices

### 1. Data Quality
- Ensure SR level data is clean and consistent
- Validate feature engineering outputs
- Monitor for outliers and anomalies

### 2. Model Selection
- Use ensemble methods for better reliability
- Prefer regularized models over complex ones
- Monitor cross-validation scores closely

### 3. Feature Engineering
- Focus on domain-specific SR features
- Apply feature selection to reduce dimensionality
- Consider feature interactions carefully

### 4. Validation Strategy
- Use stratified cross-validation
- Implement leave-one-out for very small datasets
- Monitor for overfitting indicators

### 5. Continuous Learning
- Implement online learning to update models
- Collect new SR level data over time
- Retrain models periodically

## Troubleshooting

### Common Issues

1. **High Variance in CV Scores**
   - Solution: Increase regularization strength
   - Check: Feature selection effectiveness

2. **Poor Generalization**
   - Solution: Apply more data augmentation
   - Check: Feature quality and relevance

3. **Overfitting**
   - Solution: Reduce model complexity
   - Check: Cross-validation strategy

4. **Low Prediction Confidence**
   - Solution: Use ensemble methods
   - Check: Feature engineering quality

### Performance Optimization

```python
# Adjust configuration for better performance
config = SmallDatasetIntegrationConfig(
    augmentation_factor=3.0,        # More augmentation
    regularization_strength=2.0,    # Stronger regularization
    feature_selection_ratio=0.2,    # Fewer features
    cross_validation_folds=5        # More CV folds
)
```

## Future Improvements

1. **Active Learning**: Select most informative samples for labeling
2. **Meta-Learning**: Learn from similar small dataset problems
3. **Online Learning**: Update models with new data incrementally
4. **Uncertainty Quantification**: Provide prediction confidence intervals
5. **Multi-Task Learning**: Share knowledge across related SR tasks

## Conclusion

The small dataset management system provides comprehensive solutions for training ML models with limited SR level data. By combining data augmentation, transfer learning, and regularized ensemble methods, you can significantly improve model performance and reliability even with just 91 data points.

The system is designed to integrate seamlessly with your existing SR ML Enhancer, automatically detecting small datasets and applying appropriate techniques without requiring changes to your existing code.