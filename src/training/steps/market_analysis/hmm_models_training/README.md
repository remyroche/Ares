# HMM Models Training

Enhanced HMM models training with comprehensive validation, error handling, and reporting.

## Complete Migration to Streamlined Approach ✅

The HMM training has been **completely migrated** to leverage the common_utils/ ML training pipeline for maximum efficiency and consistency.

### Files

- **`hmm_models_training_enhanced.py`** - Streamlined training class using common_utils/ pipeline (PRIMARY)
- **`validation_framework.py`** - Comprehensive validation framework
- **`enhanced_reporting.py`** - Enhanced reporting system
- **`__init__.py`** - Module exports and imports

## Key Features

### 1. Streamlined Architecture
- **Minimal custom code** - delegates to common_utils/ ML training pipeline
- **15m timeframe focus** - specifically designed for HMM state recognition
- **State recognition focus** - not prediction, optimized for HMM states
- **HPO integration** - leverages common hyperparameter optimization
- **Validation integration** - uses universal validation framework

### 2. Common Utils Integration
- **BaseTrainingStep inheritance** - leverages common training pipeline
- **Universal validation** - consistent validation across all training steps
- **Hardware optimization** - M1 GPU/CPU/memory optimization
- **Model management** - standardized model saving/loading
- **Reporting integration** - comprehensive reporting and metrics

### 3. HMM-Specific Optimizations
- **Optimized models** - LightGBM, XGBoost, Random Forest, Elastic Net
- **No ensemble models** - removed voting, stacking, bagging, ada boost, extra trees
- **No deep learning models** - removed TabNet and neural networks for HMM focus
- **Gradient booster comparison** - XGBoost vs LightGBM, training both to select best
- **Regime-aware training** - per-regime model training with regime prediction capability (models trained per regime for better specialization, but unified prediction capability)
- **Enhanced reporting** - comprehensive metrics and recommendations for all models
- **HMM search spaces** - optimized HPO spaces for state recognition
- **15m timeframe enforcement** - ensures consistent timeframe usage
- **Comprehensive feature bank** - 17 feature categories (momentum, volatility, trend, volume, support/resistance, returns, oscillator, candlestick_pattern, hmm_regime, entropy, order_flow, acceleration, cross_timeframe, autoencoder, interaction, microstructure, time)
- **Feature bank integration** - Automatic feature generation from comprehensive feature bank

## Usage

### Primary Approach (Streamlined - Complete Migration)

```python
from src.training.steps.market_analysis.hmm_models_training import (
    create_enhanced_hmm_models_training,
    execute_enhanced_hmm_models_training
)

# Simple execution with ensemble models included
results = execute_enhanced_hmm_models_training(
    X, y, regime_labels,
    feature_names=feature_names,
    hmm_states=hmm_states
)

# Or with custom config including gradient boosters
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
config = HMMTrainingConfig(
    model_types=[
        # Optimized models for HMM state classification
        "lightgbm", "xgboost", "random_forest", "elastic_net"
    ],
    hpo_trials=100,
    enable_multi_objective=True
)

training_step = create_enhanced_hmm_models_training(config)
results = training_step.execute(X, y, regime_labels, feature_names)
```

**Key Features:**
- ✅ **15m timeframe enforcement** - automatic HMM state recognition
- ✅ **Optimized models** - LightGBM, XGBoost, Random Forest, Elastic Net
- ✅ **No ensemble models** - removed voting, stacking, bagging, ada boost, extra trees
- ✅ **No deep learning models** - removed TabNet and neural networks for HMM focus
- ✅ **Gradient booster comparison** - XGBoost vs LightGBM, training both to select best
- ✅ **Enhanced reporting** - comprehensive metrics and recommendations for all models
- ✅ **HMM state focus** - optimized for state recognition, not prediction
- ✅ **Common_utils pipeline** - leverages robust ML training infrastructure
- ✅ **Comprehensive feature bank** - 17 feature categories for maximum signal extraction
- ✅ **Feature bank integration** - Automatic generation of all feature types (momentum, volatility, trend, volume, support/resistance, returns, oscillator, candlestick_pattern, hmm_regime, entropy, order_flow, acceleration, legacy)
- ✅ **Selective feature usage** - HMM training excludes complex categories (cross_timeframe, autoencoder, interaction, microstructure, time) for optimal performance
- ✅ **Short-term focus** - Optimized for 15m timeframe with reduced emphasis on long-term regime stability
- ✅ **Rapid regime switching** - Accepts frequent regime changes appropriate for short-term (10-30 minute) predictions

## Migration Complete ✅

### Complete Migration to Common_Utils Pipeline
The HMM training has been **completely migrated** to leverage the common_utils/ ML training pipeline:

- ✅ **`hmm_models_training_enhanced.py`** - NOW the streamlined implementation
- ✅ **90%+ reduction in custom code** - leverages BaseTrainingStep inheritance
- ✅ **Universal validation, HPO, and reporting** from common_utils
- ✅ **HMM state recognition focus** with 15m timeframe enforcement
- ✅ **Optimized models** - LightGBM, XGBoost, Random Forest, Elastic Net
- ✅ **No ensemble models** - removed voting, stacking, bagging, ada boost, extra trees
- ✅ **No deep learning models** - removed TabNet and neural networks for HMM focus
- ✅ **Enhanced reporting** - comprehensive metrics and recommendations for all models

### What Changed
- **Complete file replacement** - `hmm_models_training_enhanced.py` now contains the streamlined implementation
- **Model selection optimized** - Base models: top 2 (LightGBM, Random Forest) + gradient boosters (XGBoost, Elastic Net)
- **Ensemble models removed** - removed voting, stacking, bagging, ada boost, extra trees for HMM focus
- **Deep learning models removed** - removed TabNet and neural networks for HMM focus
- **Enhanced reporting added** - comprehensive metrics and recommendations for all models
- **Gradient booster comparison** - both XGBoost and LightGBM trained to select best performer
- **Single-step migration** - no gradual transition needed
- **Backward compatibility maintained** - existing function names preserved

## Regime Training Clarification

### Important: Per-Regime Training vs Regime Prediction

**Per-Regime Training**: ✅ **USED**
- Models are trained separately for each market regime
- Each regime gets its own optimized model
- Better specialization for regime-specific patterns
- Training data is split by regime labels

**Individual Regime Models**: ✅ **USED**
- Separate model instances for each regime
- Each model learns regime-specific HMM state patterns
- Better performance within each regime context

**Unified Regime Prediction**: ✅ **NEEDED**
- Single prediction interface that can predict current regime with probabilities
- Models should be able to determine "which regime are we in right now?"
- Probability distributions over possible regimes
- Confidence scores for regime classification

**Not Needed**: ❌ **Ensemble per regime**
- No need for separate ensemble models per regime
- Base models (LightGBM, RF, XGBoost, Elastic Net) are sufficient
- Focus on regime prediction capability rather than complex ensemble per regime

## Multi-Objective Optimization

### HMM Training Objectives

The HMM models training uses **multi-objective optimization** to balance multiple performance metrics:

**Primary Objectives:**
- **Accuracy (40% weight)**: Standard classification accuracy for HMM state recognition
- **F1-Score (30% weight)**: Harmonic mean of precision and recall, important for imbalanced regime data
- **Regime Stability (20% weight)**: Custom metric measuring consistency of regime predictions within similar market conditions - focuses on noise reduction rather than temporal persistence (reduced weight for short-term 15m predictions)

**Why Multi-Objective?**
- HMM state recognition requires balancing multiple competing goals
- Different regimes may have different optimal model configurations
- Ensures robust performance across various market conditions
- Provides better generalization than single-objective optimization
- Optimized for short-term (15m) predictions with appropriate regime transition handling

**Objective Weights Configuration:**
```python
objectives=["accuracy", "f1_score", "regime_stability"]
objective_weights=[0.4, 0.3, 0.2]  # Reduced regime stability weight for 15m short-term predictions
```

**Implementation:**
- Uses Pareto-front optimization for finding optimal trade-offs
- Random search with 100+ trials per model type
- Automatic selection of best configuration based on weighted objectives
- Regime-specific optimization when data allows

### How Regime Stability Helps

**Regime Stability** is crucial for HMM state recognition because:

1. **Noise Reduction**: Financial markets are inherently noisy. A model might achieve high accuracy by overfitting to short-term noise rather than learning true regime patterns. The stability metric helps identify models that learn meaningful patterns rather than noise.

2. **Short-term Reliability**: For 15m timeframe predictions, the focus is on reliable regime classification within the next 10-30 minutes, accepting that regime changes can happen rapidly in volatile market conditions.

3. **Consistency Within Market Conditions**: Rather than temporal persistence, the metric focuses on consistency of regime predictions when market conditions are similar, ensuring the model learns robust regime characteristics.

4. **Reduced Overfitting**: The stability metric helps prevent overfitting to random market fluctuations while still allowing the model to capture legitimate regime transitions.

**Implementation**: The regime stability metric measures:
- Consistency of regime predictions within similar market conditions (focus on market state rather than time)
- Minimal penalty for regime transitions (since rapid switching is acceptable for short-term predictions)
- Focus on noise reduction and reliability rather than temporal persistence
- Emphasis on meaningful regime patterns over random fluctuations

**Benefits**: This ensures the trained models provide:
- More reliable regime classification for live trading
- Better signal-to-noise ratio in regime predictions
- Reduced overfitting to short-term market noise
- Consistent performance across similar market conditions

## Benefits

### Streamlined Approach
- **Minimal custom code** - delegates to robust common_utils/ pipeline
- **15m timeframe enforcement** - consistent HMM state recognition
- **HMM state focus** - optimized for state recognition, not prediction
- **Hardware optimization** - leverages M1 GPU/CPU/memory optimization
- **Universal validation** - consistent validation across all training steps
- **HPO integration** - leverages common hyperparameter optimization
- **Standardized reporting** - consistent metrics and reporting

### Legacy Approach (Enhanced)
- **Comprehensive validation** - multi-level validation framework
- **Real metrics** - no placeholder values
- **Actionable insights** - detailed recommendations
- **Robust error handling** - comprehensive error management

## Configuration

### Streamlined Approach
The streamlined approach automatically configures:
- **Timeframe**: 15m (enforced for HMM state recognition)
- **Base models**: 4 models (LightGBM, Random Forest, XGBoost, Elastic Net)
- **No ensemble models** - removed voting, stacking, bagging, ada boost, extra trees
- **No deep learning models** - removed TabNet and neural networks for HMM focus
- **HPO**: Enabled with HMM-specific search spaces for all model types
- **Validation**: Universal validation integration
- **Enhanced reporting**: Comprehensive metrics and recommendations for all models
- **Gradient booster comparison**: XGBoost vs LightGBM to select best performer
- **Comprehensive features**: 13 feature categories from feature bank (momentum, volatility, trend, volume, support/resistance, returns, oscillator, candlestick_pattern, hmm_regime, entropy, order_flow, acceleration, legacy) - excludes complex categories (cross_timeframe, autoencoder, interaction, microstructure, time)
- **Feature bank integration**: Automatic feature generation for maximum signal extraction

### Custom Configuration
```python
config = HMMTrainingConfig(
    model_types=[
        # Base models (top 2)
        "logistic_regression", "lightgbm", "random_forest",
        # Ensemble models
        "voting_classifier", "stacking_classifier", "bagging_classifier",
        "ada_boost_classifier", "extra_trees_classifier", "xgboost",
        # Deep learning models
        "tabnet_classifier", "neural_network_classifier"
    ],
    hpo_trials=200,                                   # Custom HPO trials
    enable_multi_objective=False,                     # Disable multi-objective
    objectives=["accuracy", "f1_score"],              # Custom objectives
    objective_weights=[0.6, 0.4]                      # Custom weights
)
```