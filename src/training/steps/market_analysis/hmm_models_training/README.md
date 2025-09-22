# HMM Models Training

Enhanced HMM models training with comprehensive validation, error handling, and reporting.

## Streamlined Approach (Recommended)

The new streamlined HMM training leverages the common_utils/ ML training pipeline for maximum efficiency and consistency.

### Files

- **`streamlined_hmm_training.py`** - New streamlined training class using common_utils/ pipeline
- **`hmm_models_training_enhanced.py`** - Legacy enhanced training (backward compatibility)
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

### Streamlined Approach (Recommended)

```python
from src.training.steps.market_analysis.hmm_models_training import (
    create_streamlined_hmm_training,
    execute_streamlined_hmm_training
)

# Simple execution
results = execute_streamlined_hmm_training(
    X, y, regime_labels,
    feature_names=feature_names,
    hmm_states=hmm_states
)

# Or with custom config
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
config = HMMTrainingConfig(
    model_types=["logistic_regression", "lightgbm", "random_forest"],
    hpo_trials=100,
    enable_multi_objective=True
)

training_step = create_streamlined_hmm_training(config)
results = training_step.execute(X, y, regime_labels, feature_names)
```

### Legacy Approach (For Backward Compatibility)

```python
from src.training.steps.market_analysis.hmm_models_training import (
    create_enhanced_hmm_models_training,
    validate_hmm_training_inputs,
    ValidationLevel
)

# Legacy approach still available
validation_report = validate_hmm_training_inputs(
    X, y, regime_labels,
    validation_level=ValidationLevel.STANDARD
)

if validation_report.overall_result.value == "pass":
    training_step = create_enhanced_hmm_models_training(config)
    results = training_step.execute(X, y, regime_labels, feature_names)
```

## Migration from Old Code

### New Streamlined Approach
The new streamlined approach significantly reduces custom code by leveraging the common_utils/ ML training pipeline:

- ✅ **`streamlined_hmm_training.py`** (new) - 90%+ reduction in custom code
- ✅ Uses common BaseTrainingStep inheritance
- ✅ Leverages universal validation, HPO, and reporting
- ✅ Focuses specifically on HMM state recognition
- ✅ Enforces 15m timeframe for consistency

### Legacy Code
- ✅ **`hmm_models_training_enhanced.py`** - Available for backward compatibility
- ❌ Consider migrating to streamlined approach for better maintainability

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
- **Model types**: logistic_regression, lightgbm, random_forest
- **HPO**: Enabled with HMM-specific search spaces
- **Validation**: Universal validation integration
- **Reporting**: Common reporting pipeline

### Custom Configuration
```python
config = HMMTrainingConfig(
    model_types=["logistic_regression", "lightgbm"],  # Custom model selection
    hpo_trials=200,                                   # Custom HPO trials
    enable_multi_objective=False,                     # Disable multi-objective
    objectives=["accuracy", "f1_score"],              # Custom objectives
    objective_weights=[0.6, 0.4]                      # Custom weights
)
```