# HMM Performance Metrics → ML Models Integration Guide

## Overview

This guide explains how HMM performance metrics are passed to feature generators and integrated with ML models for enhanced trading predictions. The integration provides regime-aware features, model quality assessment, and ensemble weighting capabilities.

## Integration Architecture

```
Market Data → HMM Analysis → Performance Metrics → Feature Generation → ML Training
     ↓              ↓              ↓                    ↓               ↓
   OHLCV       Regime Labels    25+ Metrics        40+ Features    Enhanced Models
   Volume      Probabilities    Stability          Dynamic         Ensemble
   Returns     Transitions      Balance            Rolling         Weighting
                               Confidence         Interaction
```

## Step-by-Step Integration Process

### 1. HMM Analysis → Performance Metrics

```python
from market_analysis.hmm_clustering.enhanced_hmm_clustering import (
    run_hmm_clustering_analysis, HMMClusteringConfig
)

# Configure HMM
config = HMMClusteringConfig(
    n_components=4,
    use_gpu=True,
    use_memory_optimization=True
)

# Run analysis
result = run_hmm_clustering_analysis(
    symbol="BTCUSDT",
    interval="1h", 
    config=config
)

# Access performance metrics
metrics = result.performance_metrics
print(f"Available metrics: {len(metrics)}")
```

**Performance Metrics Generated (25+ metrics):**
- **Stability Metrics**: `regime_stability`, `transition_rate`
- **Balance Metrics**: `regime_balance`, `regime_entropy`, `regime_gini_coefficient`
- **Confidence Metrics**: `avg_confidence`, `min_confidence`, `max_confidence`, `confidence_std`
- **Duration Metrics**: `avg_regime_duration`, `min_regime_duration`, `max_regime_duration`
- **Persistence Metrics**: `avg_regime_persistence`, `min_regime_persistence`, `max_regime_persistence`
- **Quality Metrics**: `regime_separation_ratio`, `avg_regime_distance`
- **Uncertainty Metrics**: `avg_uncertainty`, `uncertainty_std`

### 2. Performance Metrics → ML Features

```python
from src.feature_generation.categories.hmm_performance_metrics import (
    HMMPerformanceMetricsFeatureGenerator,
    create_hmm_performance_features_from_result
)

# Method 1: Direct feature creation
ml_features = create_hmm_performance_features_from_result(
    data, hmm_result, lookback_window=20
)

# Method 2: Using feature generator
generator = HMMPerformanceMetricsFeatureGenerator(lookback_window=20)
ml_features = generator.generate_features(
    data,
    hmm_performance_metrics=result.performance_metrics,
    regime_labels=result.regime_labels,
    regime_probabilities=result.regime_probabilities
)
```

**Feature Types Generated (40+ features):**

#### Static Features (Broadcast to all time points)
- `hmm_regime_stability`: Overall regime stability
- `hmm_regime_balance`: Regime distribution balance
- `hmm_avg_confidence`: Average model confidence
- `hmm_regime_separation_ratio`: How well regimes separate features

#### Dynamic Features (Time-varying)
- `hmm_current_regime`: Current regime label
- `hmm_regime_changed`: Binary indicator of regime change
- `hmm_regime_confidence`: Current regime probability
- `hmm_time_since_regime_change`: Duration in current regime

#### Rolling Features (Windowed calculations)
- `hmm_rolling_stability`: Rolling regime stability
- `hmm_rolling_avg_confidence`: Rolling average confidence
- `hmm_rolling_regime_diversity`: Number of unique regimes in window
- `hmm_rolling_transition_rate`: Rolling transition frequency

#### Interaction Features (Combined metrics)
- `hmm_confidence_stability_product`: Confidence × Stability
- `hmm_regime_quality_score`: Composite quality score
- `hmm_model_reliability`: Model reliability indicator

### 3. Feature Integration → ML Training

```python
from src.feature_generation.utils.hmm_ml_integration import (
    HMMMLIntegrator, quick_hmm_features_integration
)

# Method 1: Quick integration
integrated_features = quick_hmm_features_integration(
    data, "BTCUSDT", "1h"
)

# Method 2: Full pipeline
integrator = HMMMLIntegrator()
features, metadata = integrator.prepare_features_for_ml_training(
    data, "BTCUSDT", "1h",
    base_feature_generator=generate_base_features,
    feature_config={
        'lookback_window': 20,
        'include_regime_features': True,
        'include_rolling_features': True,
        'feature_selection_method': 'correlation',
        'max_correlation': 0.95
    }
)

# Train ML model
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(features, target)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
```

### 4. Ensemble Weighting → Model Combination

```python
from src.feature_generation.utils.hmm_ml_integration import (
    create_hmm_ensemble_pipeline
)

# Create ensemble with HMM-based weighting
symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
data_dict = {symbol: load_data(symbol) for symbol in symbols}

features_dict, ensemble_weights = create_hmm_ensemble_pipeline(
    data_dict, symbols, interval="1h"
)

# Use weights for ensemble prediction
predictions = []
for symbol in symbols:
    model = train_model(features_dict[symbol])
    pred = model.predict(test_data)
    predictions.append(pred)

# Weighted ensemble prediction
ensemble_pred = np.average(predictions, axis=0, weights=ensemble_weights)
```

## Integration Points in Existing Systems

### 1. Feature Generation Pipeline Integration

```python
# In your existing feature generation pipeline
from market_analysis.hmm_clustering.enhanced_hmm_clustering import EnhancedHMMClustering

class MyFeatureGenerator:
    def __init__(self):
        self.hmm_clustering = EnhancedHMMClustering()
    
    def generate_features(self, data):
        # Generate base features
        base_features = self.generate_base_features(data)
        
        # Run HMM analysis
        hmm_result = self.hmm_clustering.fit_hmm_model(data)
        
        # Integrate HMM features
        integrated_features = self.hmm_clustering.integrate_with_feature_pipeline(
            base_features, hmm_result, integration_method='selective'
        )
        
        return integrated_features
```

### 2. ML Training Pipeline Integration

```python
# In your ML training pipeline
from src.feature_generation.utils.hmm_ml_integration import HMMMLIntegrator

class MyMLPipeline:
    def __init__(self):
        self.hmm_integrator = HMMMLIntegrator(cache_dir="hmm_cache")
    
    def train_model(self, data, symbol, interval):
        # Prepare features with HMM integration
        features, metadata = self.hmm_integrator.prepare_features_for_ml_training(
            data, symbol, interval,
            base_feature_generator=self.generate_base_features
        )
        
        # Train model with enhanced features
        model = self.train_ml_model(features)
        
        return model, metadata
```

### 3. Ensemble System Integration

```python
# In your ensemble system
class MyEnsembleSystem:
    def __init__(self):
        self.hmm_integrator = HMMMLIntegrator()
    
    def create_ensemble(self, data_dict, symbols):
        # Create individual models with HMM features
        models = {}
        hmm_results = []
        
        for symbol in symbols:
            features, _ = self.hmm_integrator.prepare_features_for_ml_training(
                data_dict[symbol], symbol, "1h"
            )
            models[symbol] = self.train_model(features)
            
            # Get HMM result for weighting
            hmm_result = self.hmm_integrator.run_hmm_analysis_with_caching(
                data_dict[symbol], symbol, "1h"
            )
            hmm_results.append(hmm_result)
        
        # Create ensemble weights based on HMM performance
        weights = self.hmm_integrator.create_ensemble_weights_from_hmm(
            hmm_results, weighting_method='performance_based'
        )
        
        return models, weights
```

## Advanced Usage Patterns

### 1. Multi-Timeframe Integration

```python
# Integrate HMM features across multiple timeframes
timeframes = ["1h", "4h", "1d"]
all_features = []

for tf in timeframes:
    tf_data = resample_data(data, tf)
    tf_features = quick_hmm_features_integration(tf_data, symbol, tf)
    
    # Add timeframe suffix
    tf_features.columns = [f"{col}_{tf}" for col in tf_features.columns]
    all_features.append(tf_features)

# Combine multi-timeframe features
multi_tf_features = pd.concat(all_features, axis=1)
```

### 2. Dynamic Regime-Aware Models

```python
# Train different models for different regimes
regime_models = {}
for regime in range(n_regimes):
    regime_mask = hmm_result.regime_labels == regime
    regime_features = features[regime_mask]
    regime_target = target[regime_mask]
    
    regime_models[regime] = train_model(regime_features, regime_target)

# Predict using regime-specific models
current_regime = hmm_result.regime_labels[-1]
prediction = regime_models[current_regime].predict(current_features)
```

### 3. Meta-Learning with HMM Metrics

```python
# Use HMM performance metrics for meta-learning
meta_features = []
model_predictions = []

for model_config in model_configs:
    # Train model with current config
    model = train_model(features, config=model_config)
    pred = model.predict(test_features)
    
    # Get HMM metrics for this model
    hmm_result = run_hmm_on_predictions(pred)
    meta_features.append(list(hmm_result.performance_metrics.values()))
    model_predictions.append(pred)

# Train meta-model to select best prediction based on HMM metrics
meta_model = train_meta_model(meta_features, true_targets)
best_prediction_idx = meta_model.predict(current_meta_features)
final_prediction = model_predictions[best_prediction_idx]
```

## Performance Benefits

### 1. Regime-Aware Predictions
- Models automatically adapt to different market conditions
- Better performance during regime transitions
- Reduced overfitting to specific market periods

### 2. Model Quality Assessment
- Automatic evaluation of model reliability
- Early detection of model degradation
- Confidence-weighted predictions

### 3. Enhanced Feature Engineering
- 40+ additional features from HMM analysis
- Time-varying regime information
- Rolling performance metrics

### 4. Ensemble Optimization
- Automatic weighting based on model performance
- Dynamic rebalancing of ensemble components
- Improved ensemble diversity

## Best Practices

### 1. Feature Selection
```python
# Use correlation filtering to avoid redundant features
integrated_features = integrator.integrate_with_existing_features(
    base_features, hmm_features,
    feature_selection_method='correlation',
    max_correlation=0.95
)
```

### 2. Caching for Performance
```python
# Cache HMM results to avoid recomputation
integrator = HMMMLIntegrator(cache_dir="hmm_cache")
result = integrator.run_hmm_analysis_with_caching(
    data, symbol, interval, force_recompute=False
)
```

### 3. Validation
```python
# Use time series cross-validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(features):
    # Train and validate with temporal ordering preserved
    pass
```

### 4. Monitoring
```python
# Monitor HMM performance metrics over time
performance_history = []
for period in time_periods:
    hmm_result = run_hmm_analysis(period_data)
    performance_history.append(hmm_result.performance_metrics)

# Detect performance degradation
if current_stability < historical_average * 0.8:
    retrain_model()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all HMM modules are properly installed
2. **Memory Issues**: Use caching and reduce lookback windows
3. **Performance**: Enable hardware optimization for large datasets
4. **Feature Correlation**: Use selective integration to avoid redundancy

### Error Handling

```python
try:
    features = quick_hmm_features_integration(data, symbol, interval)
except ImportError:
    # Fallback to basic features
    features = generate_basic_features(data)
except Exception as e:
    logger.error(f"HMM integration failed: {e}")
    features = fallback_features
```

## Summary

The HMM-ML integration provides a complete pipeline for leveraging regime detection in machine learning models:

1. **Automatic Feature Generation**: 40+ features from HMM performance metrics
2. **Seamless Integration**: Works with existing feature pipelines
3. **Ensemble Weighting**: Performance-based model combination
4. **Regime Awareness**: Models adapt to market conditions
5. **Quality Assessment**: Continuous model performance monitoring

This integration significantly enhances ML model performance by providing regime-aware features and intelligent ensemble weighting based on model quality metrics.