# Migrated ML Models Documentation

## Overview

This document provides comprehensive documentation for the migrated ML models across HMM, Analyst, and Tactician components. The migration includes all required model architectures with proper regularization, overfitting prevention, regime-aware training, and integration with the existing ML training pipeline.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Components](#model-components)
3. [HMM Models (15m - Regime Detection)](#hmm-models-15m---regime-detection)
4. [Analyst Models (5m - Trading Opportunities)](#analyst-models-5m---trading-opportunities)
5. [Tactician Models (1m - Entry Timing)](#tactician-models-1m---entry-timing)
6. [Regime-Aware Training](#regime-aware-training)
7. [Regularization and Overfitting Prevention](#regularization-and-overfitting-prevention)
8. [Training Pipeline Integration](#training-pipeline-integration)
9. [Configuration](#configuration)
10. [Usage Examples](#usage-examples)
11. [Testing](#testing)
12. [Performance Considerations](#performance-considerations)

## Architecture Overview

The migrated models follow a hierarchical architecture with three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    MIGRATED ML MODELS                       │
├─────────────────────────────────────────────────────────────┤
│  HMM Models (15m)     │  Analyst Models (5m)  │ Tactician  │
│  - Regime Detection   │  - Trading Opportunities │ Models   │
│  - 4D Regime Char.    │  - HMM Inputs          │ (1m)      │
│  - LightGBM, XGBoost  │  - DeepScaler, CatBoost │ - Entry  │
│  - FinancialResNet    │  - XGBoost, N-BEATS    │   Timing  │
│                       │  - AdvancedMambaHybrid │ - All     │
│                       │                       │   Inputs  │
└─────────────────────────────────────────────────────────────┘
```

## Model Components

### Core Files

1. **`src/utils/ml_common/models/migrated_model_configs.py`**
   - Model configurations for all components
   - Regime-aware parameter optimization
   - Architecture definitions

2. **`src/utils/ml_common/models/advanced_model_implementations.py`**
   - PyTorch implementations of advanced architectures
   - FinancialResNet, DeepScaler, N-BEATS, AdvancedMambaHybrid
   - ModelWrapper for scikit-learn compatibility

3. **`src/utils/ml_common/models/enhanced_migrated_factory.py`**
   - Factory for creating migrated models
   - Regime-aware model creation
   - Component-specific model management

4. **`src/utils/ml_common/training/migrated_training_integration.py`**
   - Comprehensive training integration
   - Multi-timeframe coordination
   - Regime-aware training support

5. **`config/migrated_models_config.yaml`**
   - Comprehensive configuration file
   - Model-specific parameters
   - Training and validation settings

## HMM Models (15m - Regime Detection)

### Purpose
Detect regime changes with high accuracy using 4D regime characteristics (volume, volatility, momentum, trend) on 15m timeframe.

### Models

#### Base Models

**LightGBM**
```python
parameters = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "reg_alpha": 0.1,        # L1 regularization
    "reg_lambda": 0.1,       # L2 regularization
    "subsample": 0.8,        # Bagging
    "colsample_bytree": 0.8, # Feature sampling
    "min_child_samples": 20, # Overfitting prevention
    "early_stopping_rounds": 50
}
```

**XGBoost**
```python
parameters = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,        # L1 regularization
    "reg_lambda": 0.1,       # L2 regularization
    "min_child_weight": 3,   # Overfitting prevention
    "early_stopping_rounds": 50
}
```

#### Meta-learner

**FinancialResNet**
- **Architecture**: ResNet blocks with temporal convolutions and attention
- **Blocks**: [32, 64, 128] (optimized for 15m data)
- **Temporal Conv Layers**: 3 (moderate temporal analysis)
- **Attention Heads**: 4 (efficient attention)
- **Dropout**: 0.15 (good regularization)
- **Regime Aware**: True (domain optimization)

### Key Features

1. **Regime Detection**: Classifies market regimes into discrete states
2. **4D Characteristics**: Uses volume, volatility, momentum, and trend
3. **High Accuracy**: Optimized for regime detection, not prediction
4. **Regularization**: Comprehensive overfitting prevention

## Analyst Models (5m - Trading Opportunities)

### Purpose
Detect trading opportunities on 5m timeframe, fed on all features + HMM inputs.

### Models

#### Base Models

**DeepScaler**
```python
parameters = {
    "hidden_layers": [512, 256, 128],
    "dropout": 0.2,
    "batch_norm": True,
    "activation": "relu",
    "l2_regularization": 0.01
}
```

**CatBoost**
```python
parameters = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,      # L2 regularization
    "bagging_temperature": 1.0,
    "subsample": 0.8,        # Bagging
    "colsample_bylevel": 0.8 # Feature sampling
}
```

**XGBoost**
- Same parameters as HMM XGBoost with regression focus

#### Specialized Models

**N-BEATS with Regime-Aware Optimization**
- **Architecture**: Neural basis expansion analysis
- **Num Blocks**: 10
- **Num Layers**: 4
- **Layer Widths**: [512, 512, 256, 256]
- **Regime Aware**: True with parameter optimization

**Regime-Specific Parameter Mapping**:
```python
regime_parameter_mapping = {
    "high_volume": {
        "learning_rate_multiplier": 1.2,
        "dropout_multiplier": 0.8,
        "layer_width_multiplier": 1.1
    },
    "high_volatility": {
        "learning_rate_multiplier": 0.8,
        "dropout_multiplier": 1.2,
        "layer_width_multiplier": 0.9
    },
    "high_momentum": {
        "learning_rate_multiplier": 1.1,
        "dropout_multiplier": 0.9,
        "layer_width_multiplier": 1.05
    },
    "strong_trend": {
        "learning_rate_multiplier": 0.9,
        "dropout_multiplier": 1.1,
        "layer_width_multiplier": 0.95
    }
}
```

#### Meta-learner

**AdvancedMambaHybrid**
- **Architecture**: Mamba + Convolution + Attention
- **Mamba Layers**: 2 (efficient temporal modeling)
- **Conv Layers**: 4 (pattern recognition)
- **Attention Heads**: 8 (multi-scale attention)
- **Hidden Dim**: 128 (balanced capacity)
- **State Expansion**: 4 (efficient state handling)
- **Multi-timeframe Fusion**: True (15m regime integration)

### Key Features

1. **Trading Opportunities**: Identifies profitable trading setups
2. **HMM Integration**: Uses HMM regime predictions as input
3. **Regime-Aware Training**: N-BEATS parameters adapt to regime characteristics
4. **Multi-timeframe**: Incorporates 15m regime information

## Tactician Models (1m - Entry Timing)

### Purpose
Find best entry timing on 1m timeframe, fed on all features + HMM inputs + Analyst inputs.

### Models

#### Base Models

**XGBoost**
- Same parameters as Analyst XGBoost, optimized for 1m timeframe

**LightGBM**
- Same parameters as HMM LightGBM, optimized for 1m timeframe

**DeepScaler1m**
```python
parameters = {
    "hidden_layers": [256, 128, 64],  # Reduced for 1m
    "dropout": 0.1,                   # Minimal regularization
    "batch_norm": True,
    "activation": "relu",
    "batch_size": 16,                 # Smaller batches
    "epochs": 50                      # Fewer epochs
}
```

**FinancialResNet**
```python
parameters = {
    "blocks": [16, 32, 64],          # Smaller for 1m
    "temporal_conv_layers": 2,        # Reduced layers
    "attention_heads": 2,             # Fewer heads
    "dropout": 0.1,                   # Minimal regularization
    "batch_size": 16,
    "epochs": 50
}
```

#### Meta-learner

**AdvancedMambaHybrid (Tactician Optimized)**
- **Mamba Layers**: 3 (maximum temporal depth)
- **Conv Layers**: 4 (detailed pattern analysis)
- **Attention Heads**: 6 (complex pattern recognition)
- **Hidden Dim**: 128 (high capacity for precision)
- **State Expansion**: 5 (rich state representation)
- **Execution Optimization**: True (speed optimization)
- **Micro Timing Attention**: True (micro-second precision)
- **Latency Aware**: True (real-time performance)

### Key Features

1. **Entry Timing**: Precise entry point identification
2. **All Inputs**: Uses HMM and Analyst predictions
3. **Real-time Performance**: Optimized for low latency
4. **Micro-timing**: Attention to micro-second precision

## Regime-Aware Training

### Regime Characteristics

The system uses 4D regime characteristics:

```python
@dataclass
class RegimeCharacteristics:
    volume: float      # Market volume level (0-1)
    volatility: float  # Market volatility level (0-1)
    momentum: float    # Price momentum level (0-1)
    trend: float       # Trend strength level (0-1)
```

### Parameter Optimization

#### N-BEATS Regime Optimization

The system automatically adjusts N-BEATS parameters based on regime characteristics:

```python
def optimize_nbeats_parameters(config, regime):
    # High volume: Increase learning rate, decrease dropout
    if regime.volume > 0.7:
        config.learning_rate *= 1.2
        config.dropout *= 0.8
        config.layer_widths = [w * 1.1 for w in config.layer_widths]
    
    # High volatility: Decrease learning rate, increase dropout
    if regime.volatility > 0.7:
        config.learning_rate *= 0.8
        config.dropout *= 1.2
        config.layer_widths = [w * 0.9 for w in config.layer_widths]
    
    # Similar optimizations for momentum and trend
    return config
```

#### FinancialResNet Regime Optimization

```python
def optimize_financial_resnet_parameters(config, regime):
    # Adjust attention heads based on volatility
    if regime.volatility > 0.7:
        config.attention_heads = min(config.attention_heads + 2, 8)
    
    # Adjust dropout based on momentum
    if regime.momentum > 0.7:
        config.dropout = max(config.dropout - 0.05, 0.05)
    
    # Adjust temporal conv layers based on trend
    if regime.trend > 0.7:
        config.temporal_conv_layers = min(config.temporal_conv_layers + 1, 5)
    
    return config
```

## Regularization and Overfitting Prevention

### Comprehensive Strategies

1. **L1/L2 Regularization**
   - LightGBM: `reg_alpha`, `reg_lambda`
   - XGBoost: `reg_alpha`, `reg_lambda`
   - Neural Networks: `weight_decay`

2. **Dropout**
   - FinancialResNet: 0.15
   - DeepScaler: 0.2
   - N-BEATS: 0.1
   - AdvancedMambaHybrid: 0.1

3. **Early Stopping**
   - All models: Configurable patience
   - Tree models: `early_stopping_rounds`
   - Neural networks: Validation loss monitoring

4. **Feature Sampling**
   - Tree models: `colsample_bytree`, `subsample`
   - Neural networks: Dropout layers

5. **Batch Normalization**
   - All neural networks: BatchNorm layers
   - Stabilizes training and prevents overfitting

6. **Data Augmentation**
   - Feature noise injection
   - Temporal augmentation for time series

### Validation Strategies

1. **Purged Cross-Validation**
   - Prevents lookahead bias
   - Purges data between train/test splits

2. **Walk-Forward Validation**
   - Time series specific validation
   - Maintains temporal order

3. **Monte Carlo Validation**
   - Multiple random splits
   - Robust performance estimation

## Training Pipeline Integration

### Multi-timeframe Coordination

The training pipeline coordinates models across timeframes:

```python
# 1. Train HMM models (15m)
hmm_results = train_hmm_models(hmm_data, regime_data)
hmm_predictions = extract_predictions(hmm_results)

# 2. Train Analyst models (5m) with HMM inputs
analyst_data_enhanced = combine_features(analyst_data, hmm_predictions)
analyst_results = train_analyst_models(analyst_data_enhanced, regime_data)
analyst_predictions = extract_predictions(analyst_results)

# 3. Train Tactician models (1m) with all inputs
tactician_data_enhanced = combine_features(
    tactician_data, hmm_predictions, analyst_predictions
)
tactician_results = train_tactician_models(tactician_data_enhanced, regime_data)
```

### Enhanced Training Utilities

The system integrates with existing enhanced training utilities:

- **Early Stopping**: Configurable patience and monitoring
- **Overfitting Detection**: Validation curve analysis
- **Performance Monitoring**: Real-time metrics tracking
- **Model Validation**: Comprehensive validation suite

## Configuration

### YAML Configuration

The system uses a comprehensive YAML configuration file (`config/migrated_models_config.yaml`):

```yaml
# Example configuration structure
hmm_models:
  lgbm:
    parameters:
      n_estimators: 1000
      learning_rate: 0.05
      # ... other parameters
    regularization:
      enable_l1_regularization: true
      enable_l2_regularization: true
    validation:
      enable_cross_validation: true
      cv_folds: 5

analyst_models:
  nbeats:
    regime_aware: true
    regime_optimization:
      enable_volume_adaptation: true
      regime_parameter_mapping:
        high_volume:
          learning_rate_multiplier: 1.2
          # ... other mappings
```

### Runtime Configuration

```python
# Create training configuration
config = MigratedTrainingConfig(
    enable_regime_aware_training=True,
    enable_overfitting_prevention=True,
    enable_regularization=True,
    validation_split=0.2,
    cross_validation_folds=5,
    enable_purged_cv=True
)

# Initialize training integration
training_integration = MigratedTrainingIntegration(config)
```

## Usage Examples

### Basic Usage

```python
from src.utils.ml_common.training.migrated_training_integration import (
    MigratedTrainingIntegration, MigratedTrainingConfig
)

# Initialize training integration
config = MigratedTrainingConfig()
training_integration = MigratedTrainingIntegration(config)

# Prepare data
data_config = {
    "hmm": {"X": X_15m, "y": y_15m},
    "analyst": {"X": X_5m, "y": y_5m},
    "tactician": {"X": X_1m, "y": y_1m}
}

regime_data = {
    "hmm": regime_15m,
    "analyst": regime_5m,
    "tactician": regime_1m
}

# Train all models
results = training_integration.train_all_models(data_config, regime_data)
```

### Individual Model Training

```python
from src.utils.ml_common.models.enhanced_migrated_factory import EnhancedMigratedModelFactory

# Initialize factory
factory = EnhancedMigratedModelFactory()

# Create HMM models
hmm_models = factory.create_all_hmm_models(
    input_dim=50, 
    output_dim=4,
    regime_characteristics=regime_chars
)

# Create Analyst models
analyst_models = factory.create_all_analyst_models(
    input_dim=55,  # 50 features + 5 HMM outputs
    output_dim=1,
    regime_characteristics=regime_chars
)
```

### Regime-Aware Parameter Optimization

```python
from src.utils.ml_common.models.migrated_model_configs import (
    RegimeCharacteristics, RegimeAwareParameterOptimizer, NBEATSConfig
)

# Define regime characteristics
regime = RegimeCharacteristics(
    volume=0.8, volatility=0.6, momentum=0.7, trend=0.5
)

# Optimize N-BEATS parameters
nbeats_config = NBEATSConfig()
optimized_config = RegimeAwareParameterOptimizer.optimize_nbeats_parameters(
    nbeats_config, regime
)

print(f"Original learning rate: {nbeats_config.learning_rate}")
print(f"Optimized learning rate: {optimized_config.learning_rate}")
```

## Testing

### Test Script

Run the comprehensive test script:

```bash
# Full test suite
python test_migrated_models.py --mode full

# Quick test (configurations and creation only)
python test_migrated_models.py --mode quick

# With custom configuration
python test_migrated_models.py --config config/migrated_models_config.yaml

# Save results
python test_migrated_models.py --output test_results.json
```

### Test Components

1. **Model Configuration Test**
   - Validates all model configurations
   - Checks regime-aware models
   - Verifies parameter structures

2. **Model Creation Test**
   - Tests model instantiation
   - Validates parameter application
   - Checks regime-aware optimization

3. **Training Integration Test**
   - Tests end-to-end training
   - Validates multi-timeframe coordination
   - Checks performance metrics

4. **Regime-Aware Optimization Test**
   - Tests parameter optimization
   - Validates regime-specific adjustments
   - Checks optimization effectiveness

### Expected Output

```
================================================================================
MIGRATED MODELS TEST RESULTS
================================================================================
Overall Success: True
Success Rate: 100.00%
Test Duration: 45.234s

Test Details:
  model_configurations: ✅
  model_creation: ✅
  regime_aware_optimization: ✅
  training_integration: ✅
================================================================================
```

## Performance Considerations

### Memory Optimization

1. **Model Wrappers**: Efficient PyTorch model wrappers
2. **Batch Processing**: Configurable batch sizes
3. **Memory Monitoring**: Real-time memory usage tracking
4. **Garbage Collection**: Automatic cleanup

### Computational Efficiency

1. **Early Stopping**: Prevents unnecessary training
2. **Parallel Processing**: Multi-core utilization
3. **GPU Acceleration**: PyTorch GPU support
4. **Model Caching**: Reuse trained models

### Scalability

1. **Modular Design**: Independent component training
2. **Configuration-Driven**: Easy parameter adjustment
3. **Pipeline Integration**: Seamless ML pipeline integration
4. **Model Registry**: Version control and management

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify file permissions

2. **Model Creation Failures**
   - Check parameter compatibility
   - Verify input dimensions
   - Ensure regime data format

3. **Training Failures**
   - Monitor memory usage
   - Check data quality
   - Verify validation splits

4. **Performance Issues**
   - Enable GPU acceleration
   - Adjust batch sizes
   - Use model caching

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use tprint for enhanced logging
from src.utils.tprint import tprint_debug
tprint_debug("Debug information")
```

## Conclusion

The migrated ML models provide a comprehensive, production-ready solution for regime detection, trading opportunity identification, and entry timing optimization. The system includes:

- **Complete Model Coverage**: All required architectures implemented
- **Regime-Aware Training**: Intelligent parameter optimization
- **Overfitting Prevention**: Comprehensive regularization strategies
- **Pipeline Integration**: Seamless ML pipeline integration
- **Production Ready**: Robust testing and validation

The migration successfully addresses all requirements while maintaining compatibility with the existing codebase and providing enhanced functionality for regime-aware trading strategies.