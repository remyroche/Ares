# Training Pipeline Decorators Integration Guide

## Overview

This guide demonstrates how to integrate the new training pipeline decorators into the enhanced training manager to secure the training process and provide comprehensive troubleshooting capabilities.

## Decorators Implemented

We have implemented 9 comprehensive decorators for training pipeline security and troubleshooting:

1. **`@validate_step_prerequisites`** - Validates system resources and dependencies
2. **`@secure_data_processing`** - Secures data processing with backups and integrity checks
3. **`@prevent_data_leakage`** - Prevents data leakage and look-ahead bias
4. **`@resource_monitor`** - Monitors system resources in real-time
5. **`@memory_efficient`** - Optimizes memory usage for large datasets
6. **`@debug_training_step`** - Provides comprehensive debugging capabilities
7. **`@circuit_breaker_protection`** - Implements circuit breaker pattern for failure prevention
8. **`@validate_step_output`** - Validates step outputs and quality
9. **`@quality_gate`** - Enforces quality standards before proceeding

## Integration with Enhanced Training Manager

### 1. Import Decorators

First, import the decorators in your training step files:

```python
from src.utils.training_pipeline_decorators import (
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
    quality_gate,
)
```

### 2. Apply Decorators to Training Steps

Apply the decorators to your training step functions. Here's an example for the HMM regime discovery step:

```python
@validate_step_prerequisites(
    required_directories=["data/training", "artifacts", "reports"],
    min_memory_gb=4.0,
    min_disk_gb=2.0,
    required_packages=["hmmlearn", "sklearn", "numpy", "pandas"],
    data_quality_checks={
        "no_nan_values": False,
        "min_rows": 1000,
        "required_columns": ["open", "high", "low", "close", "volume"]
    }
)
@secure_data_processing(
    backup_before=True,
    integrity_checks=True,
    memory_cleanup=True,
    data_validation=True
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    cross_validation_isolation=True,
    lookahead_bias_prevention=True
)
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=5.0,
    monitor_interval=30.0,
    auto_cleanup=True
)
@memory_efficient(
    chunk_size=10000,
    streaming_processing=True,
    memory_pool=True,
    cleanup_frequency=50
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception=Exception,
    monitor_interval=30.0
)
@validate_step_output(
    required_files=[
        "data/training/{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet",
        "data/training/{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json"
    ],
    data_quality_checks={
        "no_nan_values": True,
        "min_rows": 100,
        "required_columns": ["regime_id", "posterior_probability"]
    },
    performance_thresholds={
        "silhouette_score": 0.3,
        "processing_time_minutes": 30.0
    }
)
@quality_gate(
    model_performance_thresholds={
        "silhouette_score": 0.3,
        "calinski_harabasz_score": 100.0
    },
    data_quality_metrics={
        "completeness": 0.95,
        "consistency": 0.9
    },
    convergence_checks=True,
    overfitting_detection=True,
    validation_score_requirements={
        "regime_separation": 0.7
    }
)
async def run_step_enhanced(symbol: str, exchange: str, **kwargs):
    # Your training step implementation
    pass
```

### 3. Integration with Enhanced Training Manager

Update the enhanced training manager to use the decorated steps:

```python
# In enhanced_training_manager.py

# Step 1_7: HMM Regime Discovery (enhanced)
if not should_skip_step("step1_7_hmm_regime_discovery"):
    self._heartbeat("Step 1_7: HMM Regime Discovery (Enhanced)")
    step_start_1_7 = time.time()
    try:
        # Use the enhanced version with decorators
        from src.training.steps import step1_7_hmm_regime_discovery_enhanced as _step1_7_enhanced
        step1_7_success = await _step1_7_enhanced.run_step_enhanced(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
            timeframe=timeframe,
            lookback_days=self.lookback_days,
            force_rerun=self.force_rerun,
        )
    except Exception as e:
        self.logger.error(f"❌ Error in Enhanced Step 1_7: {e}")
        step1_7_success = False

    if not step1_7_success:
        self._log_step_completion(
            "Step 1_7: HMM Regime Discovery (Enhanced)",
            step_start_1_7,
            step_times,
            success=False,
        )
        # Non-fatal: proceed but warn
        self.logger.warning("⚠️ Proceeding without Step 1_7 artifacts (non-fatal)")
    else:
        self._log_step_completion(
            "Step 1_7: HMM Regime Discovery (Enhanced)",
            step_start_1_7,
            step_times,
            success=True,
        )
```

## Configuration Examples

### 1. Data Collection Step

```python
@validate_step_prerequisites(
    required_directories=["data_cache", "data/training"],
    min_memory_gb=2.0,
    min_disk_gb=10.0,
    required_packages=["ccxt", "pandas", "numpy"]
)
@secure_data_processing(
    backup_before=True,
    integrity_checks=True,
    memory_cleanup=True
)
@prevent_data_leakage(
    temporal_validation=True,
    lookahead_bias_prevention=True
)
@resource_monitor(
    memory_threshold_gb=4.0,
    cpu_threshold_percent=70.0,
    disk_threshold_gb=20.0
)
@debug_training_step(
    save_debug_artifacts=True,
    performance_profiling=True
)
@circuit_breaker_protection(
    failure_threshold=5,
    recovery_timeout=300.0
)
@validate_step_output(
    required_files=["data_cache/{exchange}_{symbol}_klines.parquet"],
    data_quality_checks={
        "min_rows": 10000,
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume"]
    }
)
async def run_step_data_collection(symbol: str, exchange: str, **kwargs):
    # Data collection implementation
    pass
```

### 2. Feature Engineering Step

```python
@validate_step_prerequisites(
    required_files=["data_cache/{exchange}_{symbol}_klines.parquet"],
    min_memory_gb=8.0,
    required_packages=["pandas_ta", "scipy", "sklearn"]
)
@secure_data_processing(
    backup_before=True,
    integrity_checks=True,
    memory_cleanup=True
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True
)
@memory_efficient(
    chunk_size=50000,
    streaming_processing=True,
    cleanup_frequency=100
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=600.0
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features.parquet"],
    data_quality_checks={
        "no_nan_values": False,
        "min_rows": 5000
    }
)
@quality_gate(
    data_quality_metrics={
        "completeness": 0.9,
        "feature_correlation_threshold": 0.95
    }
)
async def run_step_feature_engineering(symbol: str, exchange: str, **kwargs):
    # Feature engineering implementation
    pass
```

### 3. Model Training Step

```python
@validate_step_prerequisites(
    required_files=["data/training/{exchange}_{symbol}_features.parquet"],
    min_memory_gb=16.0,
    required_packages=["lightgbm", "catboost", "sklearn"]
)
@secure_data_processing(
    backup_before=True,
    integrity_checks=True
)
@prevent_data_leakage(
    cross_validation_isolation=True,
    lookahead_bias_prevention=True
)
@resource_monitor(
    memory_threshold_gb=32.0,
    cpu_threshold_percent=90.0
)
@memory_efficient(
    chunk_size=10000,
    memory_pool=True
)
@debug_training_step(
    log_intermediate_results=True,
    performance_profiling=True,
    error_context_preservation=True
)
@circuit_breaker_protection(
    failure_threshold=2,
    recovery_timeout=1800.0
)
@validate_step_output(
    required_files=["models/{exchange}_{symbol}_model.pkl"],
    performance_thresholds={
        "accuracy": 0.6,
        "training_time_minutes": 60.0
    }
)
@quality_gate(
    model_performance_thresholds={
        "accuracy": 0.6,
        "f1_score": 0.5
    },
    convergence_checks=True,
    overfitting_detection=True
)
async def run_step_model_training(symbol: str, exchange: str, **kwargs):
    # Model training implementation
    pass
```

## Benefits of Integration

### 1. **Security**
- **Data Integrity**: Automatic backups and integrity checks
- **Leakage Prevention**: Temporal validation and feature leakage detection
- **Resource Protection**: Memory and disk space monitoring

### 2. **Reliability**
- **Circuit Breaker**: Prevents cascade failures
- **Automatic Recovery**: Retry mechanisms and fallback strategies
- **Error Context**: Detailed error information for troubleshooting

### 3. **Performance**
- **Memory Optimization**: Efficient memory usage for large datasets
- **Resource Monitoring**: Real-time system resource tracking
- **Performance Profiling**: Detailed performance analysis

### 4. **Quality Assurance**
- **Output Validation**: Ensures step outputs meet quality standards
- **Quality Gates**: Enforces quality thresholds before proceeding
- **Data Quality**: Comprehensive data quality checks

### 5. **Debugging**
- **Comprehensive Logging**: Detailed step-by-step logging
- **Debug Artifacts**: Saves intermediate results for analysis
- **Error Context**: Preserves error context for troubleshooting

## Monitoring and Troubleshooting

### 1. **Debug Artifacts**

The decorators automatically create debug artifacts in the `debug_artifacts/` directory:

```
debug_artifacts/
├── step1_7_hmm_regime_discovery/
│   ├── 20241201_143022/
│   │   ├── performance_report.txt
│   │   ├── performance_data.json
│   │   └── error_context.json (if errors occurred)
│   └── ...
```

### 2. **Resource Monitoring**

Real-time resource monitoring provides alerts when thresholds are exceeded:

```
📊 Resources - Memory: 6.2 GB, CPU: 75.1%, Disk: 3.8 GB
⚠️ High memory usage: 6.2 GB > 4.0 GB
🧹 Memory cleanup: collected 1250 objects
```

### 3. **Quality Gates**

Quality gates ensure only high-quality results proceed:

```
🚪 Quality gate check for step1_7_hmm_regime_discovery
✅ Model performance thresholds check passed
✅ Data quality metrics check passed
✅ Convergence check passed
✅ Overfitting check passed
✅ Validation score requirements check passed
✅ Quality gate passed for step1_7_hmm_regime_discovery
```

### 4. **Circuit Breaker Status**

Circuit breakers provide failure protection:

```
⚠️ Failure 2/3 for step1_7_hmm_regime_discovery: MemoryError
🚫 Circuit breaker opened for step1_7_hmm_regime_discovery after 3 failures
🔄 Circuit breaker transitioning to HALF_OPEN for step1_7_hmm_regime_discovery
✅ Circuit breaker recovered for step1_7_hmm_regime_discovery
```

## Best Practices

### 1. **Decorator Order**
Apply decorators in this order for optimal functionality:
1. `@validate_step_prerequisites`
2. `@secure_data_processing`
3. `@prevent_data_leakage`
4. `@resource_monitor`
5. `@memory_efficient`
6. `@debug_training_step`
7. `@circuit_breaker_protection`
8. `@validate_step_output`
9. `@quality_gate`

### 2. **Configuration**
- Set appropriate thresholds based on your system capabilities
- Configure quality gates based on your model requirements
- Adjust circuit breaker settings based on step reliability

### 3. **Monitoring**
- Regularly check debug artifacts for insights
- Monitor resource usage patterns
- Review quality gate results for model improvement

### 4. **Troubleshooting**
- Use debug artifacts to identify bottlenecks
- Check circuit breaker status for failure patterns
- Review error context for root cause analysis

## Conclusion

The training pipeline decorators provide a comprehensive security and troubleshooting framework for the enhanced training manager. By integrating these decorators, you can:

- **Secure** your training pipeline against data corruption and leakage
- **Monitor** system resources and performance in real-time
- **Debug** issues with comprehensive logging and artifact preservation
- **Ensure** quality standards are met at every step
- **Recover** automatically from failures with circuit breakers

This integration significantly enhances the reliability, security, and debuggability of your training pipeline while maintaining the existing architecture and adding powerful new capabilities for monitoring and troubleshooting.
