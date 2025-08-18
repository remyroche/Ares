# Decorator Centralization and Improvements Summary

## Overview
This document summarizes the centralization of decorators and improvements made to the training pipeline steps.

## 1. Centralized Decorators Module

### Created: `src/utils/centralized_decorators.py`
This module centralizes all decorators used throughout the codebase for easy import and management.

#### Imported Decorators:

**Error Handling Decorators:**
- `handle_errors` - Enhanced error handling with recovery strategies
- `handle_specific_errors` - Specific error handling with custom handlers
- `handle_file_operations` - File operation error handling

**Training Pipeline Decorators:**
- `deterministic_seed` - Set global random seeds for reproducibility
- `idempotent_step` - Skip execution if step artifact already exists
- `artifact_write_lock` - Simple inter-process file lock during artifact writes
- `nan_inf_and_constant_guard` - Guard outputs for NaN/Inf and near-constant columns
- `artifact_versioning` - Attach schema version and timestamp to persisted artifacts
- `time_budget_watchdog` - Warn when step exceeds soft time budget
- `validate_step_prerequisites` - Validate step prerequisites
- `secure_data_processing` - Secure data processing with backup and integrity checks
- `prevent_data_leakage` - Prevent data leakage with temporal validation
- `resource_monitor` - Monitor resource usage
- `memory_efficient` - Memory efficient processing
- `debug_training_step` - Debug training steps with profiling
- `circuit_breaker_protection` - Circuit breaker protection
- `validate_step_output` - Validate step output
- `quality_gate` - Quality gate validation

**Data Quality Decorators:**
- `validate_data_quality` - Main data quality validation decorator
- `validate_feature_engineering_with_lookahead_bias_detection` - Feature engineering validation

**General Decorators:**
- `validate_call_or_runtime_types` - Type validation with fallbacks
- `pa_check_input` - Pandera input validation
- `pa_check_output` - Pandera output validation
- `pa_check_io` - Pandera input/output validation
- `enforce_ndarray` - Coerce arguments to numpy arrays
- `auto_vectorize` - Auto-vectorize scalar functions
- `guard_array_nan_inf` - Guard arrays for NaN/Inf values
- `guard_dataframe_nulls` - Guard DataFrames for null values
- `with_tracing_span` - Add correlation-aware entry/exit logs

**Enhanced Data Quality Decorators:**
- `validate_constant_features` - Detect and remove constant features
- `validate_low_variance_features` - Detect and remove low variance features
- `validate_data_completeness` - Validate data completeness and handle missing data
- `validate_datetime_index` - Validate and fix datetime index
- `validate_multi_timeframe_alignment` - Validate multi-timeframe data alignment
- `validate_hmm_data_requirements` - Validate HMM data requirements
- `validate_data_structure` - Validate data structure and completeness
- `optimize_memory_usage` - Optimize memory usage of DataFrames
- `comprehensive_data_validation` - Comprehensive data validation combining multiple checks
- `validate_memory_optimized_data_quality` - Memory-optimized validation
- `validate_feature_engineering_pipeline` - Specialized feature engineering pipeline validation
- `validate_hmm_regime_discovery` - Specialized HMM regime discovery validation
- `validate_multi_timeframe_processing` - Specialized multi-timeframe processing validation

**Other Decorators:**
- `auto_fix_data_quality_issues` - Auto-fix data quality issues

## 2. Enhanced Data Quality Decorators Module

### Created: `src/utils/enhanced_data_quality_decorators.py`
A proper module for the enhanced data quality decorators that were previously in test files.

## 3. Updated Training Steps

### Steps with Enhanced Decorator Coverage:

**Step 4: Processing & Labeling (`step4_processing_labeling.py`)**
- Added comprehensive decorator stack including:
  - Deterministic seeding
  - Idempotent execution
  - Data quality validation
  - Resource monitoring
  - Memory optimization
  - Circuit breaker protection
  - Quality gates

**Step 3: HMM Regime Discovery (`step3_hmm_regime_discovery.py`)**
- Enhanced with additional decorators:
  - HMM-specific validation decorators
  - Enhanced error handling
  - Tracing spans for key functions
  - Data quality guards

**Step 9.5: HMM-LM Generalist Training (`step9_5_hmm_lm_generalist_training.py`)**
- Added comprehensive decorator stack including:
  - Model training specific decorators
  - Performance monitoring
  - Quality gates for model accuracy
  - Data quality validation

### Steps Already Well-Decorated:
- Step 1: Data Collection
- Step 2: Feature Engineering
- Step 5.5: Unified Regime Intelligence
- Step 6: HMM-Based Training
- Step 8: Tactician Labeling
- Step 9: Tactician Specialist Training
- Step 11: Confidence Calibration
- Step 12: Final Parameters Optimization
- Step 13: Walk Forward Validation
- Step 14: Monte Carlo Validation
- Step 15: AB Testing
- Step 16: Saving

## 4. Import Updates

### Updated Import Statements:
All training steps now import decorators from the centralized module:
```python
from src.utils.centralized_decorators import (
    handle_errors,
    deterministic_seed,
    idempotent_step,
    # ... other decorators
)
```

## 5. Suggested Additional Decorators

### Performance Monitoring Decorators:
```python
@performance_monitor(
    enable_profiling=True,
    enable_memory_tracking=True,
    enable_cpu_tracking=True,
    save_profile_data=True
)
```

### Model Validation Decorators:
```python
@model_validation(
    check_overfitting=True,
    check_underfitting=True,
    validation_metrics=["accuracy", "precision", "recall", "f1"],
    cross_validation_folds=5
)
```

### Data Pipeline Decorators:
```python
@pipeline_checkpoint(
    save_intermediate_results=True,
    checkpoint_frequency=1000,
    enable_rollback=True
)
```

### Security Decorators:
```python
@secure_model_training(
    encrypt_model_artifacts=True,
    validate_model_integrity=True,
    audit_training_data=True
)
```

### Caching Decorators:
```python
@intelligent_caching(
    cache_intermediate_results=True,
    cache_validation_data=True,
    cache_model_artifacts=True,
    cache_ttl_hours=24
)
```

### Adaptive Decorators:
```python
@adaptive_resource_allocation(
    dynamic_memory_allocation=True,
    adaptive_batch_sizes=True,
    resource_scaling_threshold=0.8
)
```

### Validation Decorators:
```python
@comprehensive_validation(
    data_quality_checks=True,
    model_quality_checks=True,
    pipeline_quality_checks=True,
    output_validation=True
)
```

## 6. Benefits of Centralization

1. **Consistency**: All decorators are imported from a single location
2. **Maintainability**: Easy to update decorator implementations
3. **Discoverability**: All available decorators are documented in one place
4. **Reusability**: Decorators can be easily applied to new steps
5. **Testing**: Centralized decorators are easier to test
6. **Documentation**: Single source of truth for decorator usage

## 7. Usage Guidelines

### Basic Decorator Stack for Training Steps:
```python
@deterministic_seed(42)
@idempotent_step(step_key="step_name")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=3600.0)
@validate_step_prerequisites(...)
@secure_data_processing(...)
@prevent_data_leakage(...)
@resource_monitor(...)
@memory_efficient(...)
@debug_training_step(...)
@circuit_breaker_protection(...)
@validate_step_output(...)
@quality_gate(...)
@handle_errors(...)
async def run_step(...):
    # Step implementation
```

### Data Quality Decorators for Data Processing:
```python
@validate_data_quality(validation_level="WARNING")
@validate_feature_engineering_pipeline
@guard_dataframe_nulls(mode="warn", arg_index=0)
@with_tracing_span("function_name", log_args=False)
def process_data(data):
    # Data processing implementation
```

## 8. Future Improvements

1. **Dynamic Decorator Selection**: Automatically select appropriate decorators based on step type
2. **Decorator Composition**: Create higher-order decorators that combine multiple decorators
3. **Performance Optimization**: Implement decorator caching and lazy loading
4. **Configuration-Driven**: Allow decorator configuration through config files
5. **Metrics Collection**: Add decorators for collecting training metrics
6. **A/B Testing**: Add decorators for A/B testing different approaches
7. **Rollback Support**: Add decorators for automatic rollback on failures
8. **Distributed Training**: Add decorators for distributed training coordination

## 9. Testing Recommendations

1. **Unit Tests**: Test each decorator in isolation
2. **Integration Tests**: Test decorator combinations
3. **Performance Tests**: Measure decorator overhead
4. **Error Handling Tests**: Test decorator behavior under various error conditions
5. **Memory Tests**: Test memory usage with decorators
6. **Concurrency Tests**: Test decorator behavior in concurrent scenarios

## 10. Monitoring and Observability

The centralized decorators provide comprehensive monitoring and observability:
- **Performance Tracking**: Resource usage, execution time, memory consumption
- **Error Tracking**: Detailed error logging with context
- **Data Quality Monitoring**: Automatic detection and reporting of data issues
- **Pipeline Health**: Overall pipeline health and status monitoring
- **Debugging Support**: Enhanced debugging capabilities with tracing spans