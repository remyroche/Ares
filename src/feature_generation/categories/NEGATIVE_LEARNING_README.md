# Negative Learning Plugin for Analyst/Tactician Pipelines

A comprehensive plugin that adds negative learning ("this works... except if...") to your existing Analyst/Tactician tree pipelines with no new architectures. Time-series safe, fast, and keeps your latency budgets.

## 🎯 Overview

This plugin implements a sophisticated negative learning system that:

1. **Discovers failure contexts** - Data-driven detection of when features fail
2. **Encodes negative learning** - Gated twins and exception interactions
3. **Keeps models honest** - Monotone constraints and sample weights
4. **Validates performance** - Bucketed analysis and SHAP stability
5. **Manages budgets** - Feature selection and latency compliance

## 🚀 Quick Start

### Basic Integration

```python
from src.feature_generation.categories.negative_learning_pipeline_integration import create_negative_learning_integrator

# Create integrator
integrator = create_negative_learning_integrator()

# Initialize once per retrain
init_results = integrator.initialize_negative_learning(
    analyst_features=analyst_features,
    analyst_target=analyst_target,
    tactician_features=tactician_features,
    tactician_target=tactician_target,
    analyst_outputs=analyst_outputs
)

# Get enhanced features for inference
enhanced_analyst, enhanced_tactician = integrator.get_enhanced_features(
    analyst_features, tactician_features, analyst_outputs
)

# Get model configurations with constraints
model_configs = integrator.get_model_configs()
```

### Advanced Usage

```python
# Custom configuration
config = {
    'analyst': {
        'negative_learning': {
            'max_negative_features': 8,
            'enable_gated_twins': True,
            'enable_exception_interactions': True
        }
    },
    'tactician': {
        'negative_learning': {
            'max_negative_features': 6,
            'enable_gated_twins': True,
            'enable_exception_interactions': True
        }
    }
}

integrator = create_negative_learning_integrator(config)
```

## 📊 Key Features

### 1. Failure Context Discovery

Automatically detects when features fail using data-driven analysis:

```python
# High volatility detection
p_highvol = (volatility_ewma > volatility_threshold).astype(float)

# Chop detection (low R² of trend fit)
p_chop = (trend_r2 < 0.3).astype(float)

# Wide spread detection
p_widespread = (spread_zscore > 0.52).astype(float)
```

### 2. Negative Learning Features

Generates three types of features:

#### Gated Twins (Strong)
```python
# Positive: active where rule should hold
feature_pos = feature * (1 - p_fail)

# Negative: inverse where it tends to fail
feature_neg = -feature * p_fail
```

#### Exception Interactions (Light)
```python
# Let trees learn to down-weight when context is bad
feature_x_fail = feature * p_fail
```

#### Context Indicators
```python
# Include context flags for splitting
feature_p_context = p_fail
```

### 3. Model Constraints

#### Monotone Constraints
- `+1` for `*_pos` features (positive monotonicity)
- `-1` for `*_neg` features (negative monotonicity)
- `0` for interactions and context indicators

#### Sample Weights
- Down-weight observations in uncertain failure zones
- `weight = base_weight * (0.7 + 0.3 * (1 - p_fail_max))`

### 4. Feature Selection

- **Stability Selection**: Bootstrap-based feature selection
- **IC Improvement**: Only keep features that improve Information Coefficient
- **Budget Management**: Hard caps on feature count and latency

## 🎯 Concrete Examples

### Example 1: Momentum × High Volatility

**Problem**: Momentum signals flip in high volatility whipsaw

**Solution**:
```python
# Generated features
momentum_5m_pos = momentum_5m * (1 - p_highvol)  # Works in normal vol
momentum_5m_neg = -momentum_5m * p_highvol       # Inverse in high vol

# Monotone constraints
monotone_constraints = [1, -1]  # +1 for pos, -1 for neg
```

### Example 2: VWAP Distance × Wide Spread

**Problem**: VWAP pull signals fail when spread widens (exhaustion)

**Solution**:
```python
# Generated features
vwap_distance_x_fail = vwap_distance * p_widespread  # Trees learn to down-weight

# No monotone constraints - let trees learn
monotone_constraints = [0]
```

### Example 3: RSI × Chop

**Problem**: RSI extremes work in chop but fail in trending markets

**Solution**:
```python
# Generated features
rsi_low_pos = rsi_low * p_chop           # RSI oversold works in chop
rsi_high_neg = -rsi_high * (1 - p_chop)  # RSI overbought fails in trending

# Monotone constraints
monotone_constraints = [1, -1]
```

## 🔧 Integration Guide

### Step 1: Add to Training Pipeline

```python
# In your training pipeline
from src.feature_generation.categories.negative_learning_pipeline_integration import create_negative_learning_integrator

# Initialize once per retrain
integrator = create_negative_learning_integrator()
init_results = integrator.initialize_negative_learning(
    analyst_features, analyst_target,
    tactician_features, tactician_target,
    analyst_outputs
)

# Get enhanced features
enhanced_analyst, enhanced_tactician = integrator.get_enhanced_features(
    analyst_features, tactician_features, analyst_outputs
)
```

### Step 2: Update Model Training

```python
# Get model configurations with constraints
model_configs = integrator.get_model_configs()

# Use in LightGBM
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'monotone_constraints': model_configs['analyst']['monotone_constraints'],
    # ... other parameters
}

# Get sample weights
sample_weights = integrator.get_sample_weights(analyst_features, tactician_features)
```

### Step 3: Update Inference Pipeline

```python
# In your inference pipeline
enhanced_analyst, enhanced_tactician = integrator.get_enhanced_features(
    analyst_features, tactician_features, analyst_outputs, inference_timestamp
)

# Use enhanced features for prediction
analyst_predictions = analyst_model.predict(enhanced_analyst)
tactician_predictions = tactician_model.predict(enhanced_tactician)
```

## 📈 Performance Monitoring

### Validation Framework

```python
# Validate performance
validation_results = integrator.validate_performance(
    analyst_features, analyst_target,
    tactician_features, tactician_target,
    analyst_outputs
)

# Check bucketed performance
bucketed_perf = validation_results['analyst']['bucketed_performance']
print(f"IC improvement: {bucketed_perf['overall_performance']['improvement']:.4f}")
```

### Drift Monitoring

```python
# Monitor drift over time
drift_results = validation_results['analyst']['drift_monitoring']
if drift_results['drift_detected']:
    print("Warning: Performance drift detected")
```

## ⚙️ Configuration

### Default Configuration

```python
config = {
    'analyst': {
        'negative_learning': {
            'max_negative_features': 8,
            'enable_gated_twins': True,
            'enable_exception_interactions': True,
            'enable_context_indicators': True
        },
        'feature_selection': {
            'stability_threshold': 0.6,
            'min_ic_improvement': 0.10
        },
        'constraints': {
            'enable_monotone_constraints': True,
            'enable_sample_weights': True,
            'weight_uncertainty_factor': 0.3
        }
    },
    'tactician': {
        'negative_learning': {
            'max_negative_features': 6,
            'enable_gated_twins': True,
            'enable_exception_interactions': True,
            'enable_context_indicators': False  # Lighter for 15m
        }
    }
}
```

### Hyperparameters

#### LightGBM
```python
lgb_params = {
    'max_depth': 4,
    'num_leaves': 16,
    'min_child_samples': 1000,
    'lambda_l2': 40,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.85,
    'learning_rate': 0.05
}
```

#### XGBoost
```python
xgb_params = {
    'max_depth': 4,
    'min_child_weight': 1000,
    'lambda': 40,
    'alpha': 20,
    'colsample_bytree': 0.75,
    'subsample': 0.85,
    'learning_rate': 0.05
}
```

#### CatBoost
```python
catboost_params = {
    'depth': 5,
    'l2_leaf_reg': 30,
    'bootstrap_type': 'Bayesian',
    'learning_rate': 0.05
}
```

## 🔍 Validation Metrics

### Bucketed Performance
- IC improvement within each failure regime
- Performance across volatility buckets
- Context-specific performance analysis

### SHAP Stability
- Sign consistency of feature contributions
- Stability across time windows
- Alignment with expected behavior

### Drift Monitoring
- Performance degradation detection
- Feature importance drift
- Context probability changes

### Ablation Studies
- Baseline vs. interactions vs. full negative learning
- Feature contribution analysis
- Performance attribution

## 📊 ETHUSDT Specific Examples

### High Volatility Whipsaw
```python
# Momentum fails in high vol
momentum_5m_pos = momentum_5m * (1 - p_highvol)
momentum_5m_neg = -momentum_5m * p_highvol
```

### Spread Exhaustion
```python
# VWAP fails when spread is wide
vwap_distance_x_fail = vwap_distance * p_widespread
```

### Regime Adaptation
```python
# RSI works in chop, fails in trending
rsi_low_pos = rsi_low * p_chop
rsi_high_neg = -rsi_high * (1 - p_chop)
```

## 🚨 Important Notes

### Time-Series Safety
- All features built OOF on training data
- As-of joined at inference time
- No peeking past last HTF close

### Latency Budget
- ≤10 negative learning features per head
- Estimated +30ms latency impact
- Budget compliance monitoring

### Memory Usage
- Efficient feature generation
- Minimal memory overhead
- Garbage collection friendly

## 🐛 Troubleshooting

### Common Issues

1. **No negative features generated**
   - Check if failure contexts are detected
   - Verify feature names match expected patterns
   - Ensure sufficient training data

2. **Performance degradation**
   - Check monotone constraints
   - Verify sample weights
   - Monitor drift detection

3. **High latency**
   - Reduce max_negative_features
   - Disable context_indicators for Tactician
   - Check feature selection thresholds

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('NegativeLearning').setLevel(logging.DEBUG)

# Check integration status
status = integrator.get_integration_status()
print(f"Initialized: {status['is_initialized']}")
print(f"Analyst features: {status['analyst_negative_features']}")
print(f"Tactician features: {status['tactician_negative_features']}")
```

## 📚 API Reference

### Core Classes

- `NegativeLearningPlugin`: Main plugin class
- `FailureContextDetector`: Discovers failure contexts
- `NegativeLearningFeatureGenerator`: Generates negative features
- `NegativeLearningFeatureSelector`: Selects optimal features
- `ModelConstraintManager`: Manages model constraints
- `NegativeLearningValidator`: Validates performance

### Integration Classes

- `NegativeLearningPipelineIntegrator`: Main integrator
- `AnalystNegativeLearningIntegration`: Analyst-specific integration
- `TacticianNegativeLearningIntegration`: Tactician-specific integration

### Utility Functions

- `create_negative_learning_integrator()`: Create integrator
- `get_integration_config()`: Get default config
- `run_all_examples()`: Run ETHUSDT examples

## 🤝 Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure time-series safety
5. Maintain latency budgets

## 📄 License

This plugin is part of the Ares trading system and follows the same license terms.

---

**Note**: This plugin is designed to be a drop-in enhancement to existing Analyst/Tactician pipelines. It requires minimal code changes and maintains backward compatibility while providing significant performance improvements in challenging market conditions.