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
- **State recognition models** - logistic regression, LightGBM, Random Forest
- **Regime-aware training** - per-regime model training
- **HMM search spaces** - optimized HPO spaces for state recognition
- **15m timeframe enforcement** - ensures consistent timeframe usage

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

# Or with custom config including ensemble models
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
config = HMMTrainingConfig(
    model_types=[
        # Base models
        "logistic_regression", "lightgbm", "random_forest", "xgboost", "catboost",
        # Ensemble models
        "voting_classifier", "stacking_classifier", "bagging_classifier",
        "ada_boost_classifier", "extra_trees_classifier",
        # Deep learning models
        "tabnet_classifier", "neural_network_classifier"
    ],
    hpo_trials=100,
    enable_multi_objective=True
)

training_step = create_enhanced_hmm_models_training(config)
results = training_step.execute(X, y, regime_labels, feature_names)
```

**Key Features:**
- ✅ **15m timeframe enforcement** - automatic HMM state recognition
- ✅ **Ensemble models included** - voting, stacking, bagging, ada boost, extra trees
- ✅ **Deep learning models** - TabNet and neural networks when available
- ✅ **HMM state focus** - optimized for state recognition, not prediction
- ✅ **Common_utils pipeline** - leverages robust ML training infrastructure

## Migration Complete ✅

### Complete Migration to Common_Utils Pipeline
The HMM training has been **completely migrated** to leverage the common_utils/ ML training pipeline:

- ✅ **`hmm_models_training_enhanced.py`** - NOW the streamlined implementation
- ✅ **90%+ reduction in custom code** - leverages BaseTrainingStep inheritance
- ✅ **Universal validation, HPO, and reporting** from common_utils
- ✅ **HMM state recognition focus** with 15m timeframe enforcement
- ✅ **Ensemble models included** - voting, stacking, bagging, ada boost, extra trees
- ✅ **Deep learning support** - TabNet and neural networks when available

### What Changed
- **Complete file replacement** - `hmm_models_training_enhanced.py` now contains the streamlined implementation
- **Ensemble models added** - comprehensive model types for robust state recognition
- **Single-step migration** - no gradual transition needed
- **Backward compatibility maintained** - existing function names preserved

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
- **Model types**: 12 comprehensive models including ensembles and deep learning
- **HPO**: Enabled with HMM-specific search spaces for all model types
- **Validation**: Universal validation integration
- **Reporting**: Common reporting pipeline

### Custom Configuration
```python
config = HMMTrainingConfig(
    model_types=[
        # Base models
        "logistic_regression", "lightgbm", "random_forest", "xgboost", "catboost",
        # Ensemble models
        "voting_classifier", "stacking_classifier", "bagging_classifier",
        "ada_boost_classifier", "extra_trees_classifier",
        # Deep learning models
        "tabnet_classifier", "neural_network_classifier"
    ],
    hpo_trials=200,                                   # Custom HPO trials
    enable_multi_objective=False,                     # Disable multi-objective
    objectives=["accuracy", "f1_score"],              # Custom objectives
    objective_weights=[0.6, 0.4]                      # Custom weights
)
```