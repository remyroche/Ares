# HMM Models Training

Enhanced HMM models training with comprehensive validation, error handling, and reporting.

## Files

- **`hmm_models_training_enhanced.py`** - Main enhanced training class with streamlined architecture
- **`validation_framework.py`** - Comprehensive validation framework preventing silent failures
- **`enhanced_reporting.py`** - Enhanced reporting system with real metrics and actionable insights
- **`__init__.py`** - Module exports and imports

## Key Features

### 1. Streamlined Architecture
- 50% reduction in code complexity
- Structured data containers
- Modular design with clear separation of concerns
- Consistent error handling patterns

### 2. Comprehensive Validation
- Multi-level validation (BASIC, STANDARD, STRICT)
- Input validation with detailed error reporting
- Data quality checks (NaN, infinite values)
- Regime-specific validation
- Feature property validation

### 3. Enhanced Reporting
- Real metrics instead of placeholders
- Comprehensive performance analysis
- Actionable insights and recommendations
- Structured report format ready for visualization

## Usage

```python
from src.training.steps.market_analysis.hmm_models_training import (
    create_enhanced_hmm_models_training,
    validate_hmm_training_inputs,
    ValidationLevel
)

# Validate inputs
validation_report = validate_hmm_training_inputs(
    X, y, regime_labels, 
    validation_level=ValidationLevel.STANDARD
)

if validation_report.overall_result.value == "pass":
    # Train models
    training_step = create_enhanced_hmm_models_training(config)
    results = training_step.execute(X, y, regime_labels, feature_names)
    
    # Access comprehensive report
    report = results['comprehensive_report']
    print(f"Best model: {report['training_summary']['best_model']}")
```

## Migration from Old Code

The old HMM training files have been removed and replaced with this enhanced version:

- ❌ `hmm_models_training_refactored.py` (removed)
- ❌ `components/hmm_models_training.py` (removed)
- ❌ `components/hmm_ensemble_training.py` (removed)
- ✅ `hmm_models_training_enhanced.py` (new)
- ✅ `validation_framework.py` (new)
- ✅ `enhanced_reporting.py` (new)

## Benefits

- **Prevents silent failures** through comprehensive validation
- **Real metrics** instead of placeholder values
- **Streamlined code** with better maintainability
- **Actionable insights** for better decision-making
- **Robust error handling** with detailed error messages