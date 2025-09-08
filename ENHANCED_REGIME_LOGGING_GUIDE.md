# Enhanced Regime-Aware Financial Logging Guide

This guide explains how to use the enhanced financial metrics logger with per-HMM regime logging and fail-fast validation for training steps that come after HMM-based data splitting.

## Overview

The enhanced regime-aware financial logging system provides:

1. **Per-HMM Regime Logging**: Automatic logging of regime-specific metrics for steps after HMM-based data splitting (step08)
2. **Fail-Fast Validation**: Prevents empty running or important degradation by validating data quality and regime integrity
3. **Automatic Regime Detection**: Automatically detects if a step comes after HMM-based data splitting
4. **Comprehensive Regime Tracking**: Tracks regime usage, validation history, and performance metrics

## Key Components

### 1. Enhanced Financial Metrics Logger (`enhanced_financial_metrics_logger.py`)

The core enhanced logger that extends the base financial metrics logger with:
- Regime data validation
- Fail-fast validation
- Per-regime metrics logging
- Comprehensive regime tracking

### 2. Regime-Aware Decorator (`regime_aware_financial_logging_decorator.py`)

Decorators that automatically add regime-aware logging to training steps:
- `@regime_aware_financial_logging`: Explicit regime-aware logging
- `@auto_regime_aware_logging`: Automatic detection and application

### 3. Smart Logger Integration

The base `financial_metrics_logger.py` has been enhanced to automatically choose between enhanced and base logging based on step type and availability.

## Usage Patterns

### Pattern 1: Using the Decorator (Recommended)

```python
from src.utils.regime_aware_financial_logging_decorator import regime_aware_financial_logging

class Step09HMMBasedTraining:
    @regime_aware_financial_logging(
        step_name="Step09_HMM_Based_Training",
        enable_regime_validation=True,
        enable_fail_fast=True,
        min_regime_samples=100,
        max_regime_imbalance=0.8,
        regime_column='composite_cluster_id',
        min_data_quality=0.7
    )
    async def execute(self, training_input, pipeline_state):
        # Your step implementation
        # The decorator automatically:
        # 1. Validates regime data
        # 2. Applies fail-fast validation
        # 3. Logs per-regime metrics
        # 4. Prevents empty running or degradation
        pass
```

### Pattern 2: Using Auto Decorator (Automatic Detection)

```python
from src.utils.regime_aware_financial_logging_decorator import auto_regime_aware_logging

class Step10UnifiedRegimeIntelligence:
    @auto_regime_aware_logging(
        enable_regime_validation=True,
        enable_fail_fast=True,
        min_regime_samples=100
    )
    async def execute(self, training_input, pipeline_state):
        # Your step implementation
        # The decorator automatically detects if this is a post-HMM step
        # and applies regime-aware logging only if needed
        pass
```

### Pattern 3: Manual Enhanced Logging

```python
from src.utils.enhanced_financial_metrics_logger import (
    get_enhanced_financial_metrics_logger,
    enhanced_financial_metrics_context,
    validate_and_log_regime_data
)

class Step11AnalystCreation:
    async def execute(self, training_input, pipeline_state):
        data = pipeline_state.get('dataframe', pd.DataFrame())
        
        # Use enhanced financial metrics context
        with enhanced_financial_metrics_context(
            step_name="Step11_Analyst_Creation",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            data=data
        ) as enhanced_logger:
            
            # Validate regime data
            if not data.empty and 'composite_cluster_id' in data.columns:
                validation_success = validate_and_log_regime_data(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step11_Analyst_Creation",
                    data=data,
                    regime_column='composite_cluster_id'
                )
                
                if not validation_success:
                    raise RuntimeError("Regime validation failed")
            
            # Your step implementation
            
            # Log per-regime metrics
            regime_metrics = {
                'regime_0': {'accuracy': 0.85, 'samples': 1000},
                'regime_1': {'accuracy': 0.82, 'samples': 800},
                'regime_2': {'accuracy': 0.88, 'samples': 1200}
            }
            
            enhanced_logger.log_per_regime_metrics(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                step_name="Step11_Analyst_Creation",
                regime_metrics=regime_metrics,
                data=data
            )
```

### Pattern 4: Using Smart Logger

```python
from src.utils.financial_metrics_logger import (
    get_smart_financial_metrics_logger,
    log_financial_metric_with_regime_awareness
)

class Step12AnalystEnhancement:
    async def execute(self, training_input, pipeline_state):
        data = pipeline_state.get('dataframe', pd.DataFrame())
        
        # Use smart logger that automatically chooses enhanced or base logging
        smart_logger = get_smart_financial_metrics_logger(use_enhanced=True)
        
        # Log metrics with regime awareness
        log_financial_metric_with_regime_awareness(
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            metric_name="enhancement_accuracy",
            metric_value=0.87,
            metric_type="performance",
            step_name="Step12_Analyst_Enhancement",
            data=data  # This enables regime validation
        )
```

## Configuration Options

### Regime-Aware Decorator Options

- `step_name`: Name of the training step
- `enable_regime_validation`: Enable regime data validation (default: True)
- `enable_fail_fast`: Enable fail-fast behavior (default: True)
- `min_regime_samples`: Minimum samples required per regime (default: 100)
- `max_regime_imbalance`: Maximum allowed regime imbalance ratio (default: 0.8)
- `regime_column`: Name of the regime column in data (default: 'composite_cluster_id')
- `expected_regimes`: List of expected regime IDs (optional)
- `min_data_quality`: Minimum required data quality score (default: 0.7)
- `log_regime_distribution`: Log regime distribution metrics (default: True)
- `log_regime_performance`: Log regime-specific performance metrics (default: True)
- `log_regime_transitions`: Log regime transition metrics (default: True)

### Enhanced Logger Options

- `fail_fast_enabled`: Enable fail-fast validation (default: True)
- `regime_validation_enabled`: Enable regime validation (default: True)
- `min_regime_samples`: Minimum samples required per regime (default: 100)
- `max_regime_imbalance`: Maximum allowed regime imbalance ratio (default: 0.8)

## Fail-Fast Validation

The fail-fast validation prevents empty running or important degradation by checking:

### Data Quality Checks
- Empty or None data
- Excessive NaN values (>50%)
- Too many constant columns (>30% of columns)
- Insufficient data variation
- Suspiciously small datasets (<10 samples)

### Regime-Specific Checks
- Missing regime column
- No valid regime data
- Insufficient regime diversity (<2 regimes)
- Empty regimes (0 samples)
- Small regimes (<min_regime_samples)
- Severe regime imbalance

### Performance Degradation Checks
- Multiple recent failures (≥3 in last 5 attempts)
- Data quality below threshold
- Missing expected regimes

## Regime Validation

The regime validation system checks:

1. **Regime Column Existence**: Ensures the regime column exists in the data
2. **Regime Data Quality**: Validates that regime data is not empty or all NaN
3. **Regime Diversity**: Ensures sufficient regime diversity (≥2 regimes)
4. **Regime Sample Sizes**: Checks that each regime has sufficient samples
5. **Regime Imbalance**: Detects severe imbalance between regimes
6. **Quality Score Calculation**: Computes overall data quality score

## Logging Output

### Console Output
The enhanced logger provides human-readable console output with emojis and formatting:

```
🚀 STARTING STEP09 HMM BASED TRAINING | Symbol: ETHUSDT | Exchange: BINANCE | Timeframe: 1m | Time: 2024-01-15 10:30:00
================================================================================
🌊 REGIME | ETHUSDT | STEP09 HMM BASED TRAINING | regime_0_sample_count: 1,000 | Regime: 0
🌊 REGIME | ETHUSDT | STEP09 HMM BASED TRAINING | regime_1_sample_count: 800 | Regime: 1
🌊 REGIME | ETHUSDT | STEP09 HMM BASED TRAINING | regime_2_sample_count: 1,200 | Regime: 2
📈 PERFORMANCE | ETHUSDT | STEP09 HMM BASED TRAINING | Return: 12.5% | Sharpe: 1.85 | MaxDD: 8.2% | Win Rate: 68.5%
✅ COMPLETED SUCCESSFULLY STEP09 HMM BASED TRAINING | Symbol: ETHUSDT | Exchange: BINANCE | Timeframe: 1m | Time: 2024-01-15 10:32:00
```

### File Output
- **CSV Files**: Structured financial metrics data
- **JSON Files**: Complex metrics and regime data
- **Log Files**: Detailed execution logs with timestamps

## Error Handling

### Fail-Fast Triggers
When fail-fast conditions are detected:

```
🚨 FAIL-FAST TRIGGERED for Step09_HMM_Based_Training
   Reason: Regime validation failed: Missing expected regimes: ['regime_3']
   Critical Issue: Missing expected regimes: ['regime_3']
   Critical Issue: Data quality score 0.45 below threshold 0.7
```

### Graceful Degradation
If enhanced logging is not available, the system automatically falls back to base logging without breaking the pipeline.

## Best Practices

### 1. Use Decorators for New Steps
For new training steps, use the `@regime_aware_financial_logging` decorator:

```python
@regime_aware_financial_logging(
    step_name="StepXX_Your_Step_Name",
    enable_fail_fast=True,
    min_regime_samples=100
)
async def execute(self, training_input, pipeline_state):
    # Your implementation
```

### 2. Update Existing Steps Gradually
For existing steps, you can:
1. Add the `@auto_regime_aware_logging` decorator (minimal changes)
2. Update to use `get_smart_financial_metrics_logger()`
3. Gradually migrate to full enhanced logging

### 3. Provide Data for Validation
Always pass the DataFrame in `pipeline_state['dataframe']` to enable regime validation:

```python
pipeline_state = {
    'dataframe': your_data_with_regime_column
}
```

### 4. Handle Fail-Fast Gracefully
Be prepared for fail-fast conditions and handle them appropriately:

```python
try:
    result = await step.execute(training_input, pipeline_state)
except RuntimeError as e:
    if "Fail-fast validation failed" in str(e):
        # Handle fail-fast condition
        logger.error(f"Step failed due to data quality issues: {e}")
        return {'success': False, 'error': str(e)}
    else:
        # Handle other errors
        raise
```

### 5. Monitor Regime Quality
Regularly check regime quality and adjust thresholds as needed:

```python
enhanced_logger = get_enhanced_financial_metrics_logger()
summary = enhanced_logger.get_regime_summary()
print(f"Regimes tracked: {summary['total_regimes_tracked']}")
print(f"Validation success rate: {summary['total_validations']}")
```

## Migration Guide

### From Basic to Enhanced Logging

1. **Step 1**: Add the `@auto_regime_aware_logging` decorator to your step
2. **Step 2**: Ensure your data includes the regime column (`composite_cluster_id`)
3. **Step 3**: Test with sample data to verify regime validation works
4. **Step 4**: Gradually add more regime-specific metrics
5. **Step 5**: Fine-tune fail-fast thresholds based on your data characteristics

### Example Migration

**Before (Basic Logging)**:
```python
class Step09HMMBasedTraining:
    async def execute(self, training_input, pipeline_state):
        # Your implementation
        logger.info("Step completed")
        return {'success': True}
```

**After (Enhanced Logging)**:
```python
from src.utils.regime_aware_financial_logging_decorator import auto_regime_aware_logging

class Step09HMMBasedTraining:
    @auto_regime_aware_logging(
        enable_fail_fast=True,
        min_regime_samples=100
    )
    async def execute(self, training_input, pipeline_state):
        # Your implementation (no changes needed)
        # The decorator automatically adds regime validation and logging
        return {'success': True}
```

## Troubleshooting

### Common Issues

1. **"Regime column not found"**
   - Ensure your data includes the `composite_cluster_id` column
   - Check that the column name matches the `regime_column` parameter

2. **"Fail-fast validation failed"**
   - Check data quality and regime distribution
   - Adjust `min_regime_samples` or `max_regime_imbalance` thresholds
   - Verify that regime data is not empty or all NaN

3. **"Enhanced logging not available"**
   - The system automatically falls back to base logging
   - Check that all required modules are imported correctly

4. **"Step not detected as post-HMM"**
   - Ensure step name follows the pattern `StepXX_*` where XX > 08
   - Use explicit `@regime_aware_financial_logging` decorator if needed

### Debug Mode

Enable debug logging to see detailed validation information:

```python
import logging
logging.getLogger('EnhancedFinancialMetrics').setLevel(logging.DEBUG)
```

## Performance Considerations

- **Regime Validation**: Adds minimal overhead (~1-2ms per validation)
- **Fail-Fast Checks**: Prevents expensive operations on poor data
- **Memory Usage**: Regime tracking uses minimal additional memory
- **File I/O**: Enhanced logging may create more log files (configurable)

## Conclusion

The enhanced regime-aware financial logging system provides comprehensive regime validation and fail-fast behavior to prevent empty running or important degradation in training steps after HMM-based data splitting. By using the provided decorators and following the best practices, you can ensure robust and reliable training pipeline execution with detailed regime-specific metrics tracking.