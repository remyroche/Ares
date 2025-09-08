# Enhanced Regime Logging Migration Guide

## Overview

This guide provides step-by-step instructions for migrating your existing training step files to use the enhanced regime logging system. The enhanced system provides:

- **Per-HMM Regime Logging**: Automatic logging of regime-specific metrics for steps after HMM-based data splitting
- **Fail-Fast Validation**: Prevents execution with poor data quality or regime imbalance
- **Backward Compatibility**: Existing code continues to work without changes
- **Easy Integration**: Multiple integration patterns to suit different needs

## Migration Patterns

### Pattern 1: Decorator-Based Integration (Recommended)

This is the simplest and most automated approach. Just add the `@auto_regime_aware_logging` decorator to your execute method.

#### Before (Original Code)
```python
class Step09HMMBasedTraining:
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        # Your existing implementation
        data = pipeline_state.get('dataframe', pd.DataFrame())
        # ... your logic ...
        return {'success': True}
```

#### After (Enhanced with Decorator)
```python
from src.utils.regime_aware_financial_logging_decorator import auto_regime_aware_logging

class Step09HMMBasedTraining:
    @auto_regime_aware_logging(
        enable_regime_validation=True,
        enable_fail_fast=True,
        min_regime_samples=100,
        max_regime_imbalance=0.8,
        regime_column='composite_cluster_id',
        min_data_quality=0.7
    )
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        # Your existing implementation - NO CHANGES NEEDED!
        data = pipeline_state.get('dataframe', pd.DataFrame())
        # ... your logic ...
        return {'success': True}
```

**Benefits:**
- ✅ Zero code changes to your existing logic
- ✅ Automatic regime validation and fail-fast checks
- ✅ Automatic regime-aware logging
- ✅ Works for all post-HMM steps (step number > 8)

### Pattern 2: Enhanced Financial Logger Integration

Use the enhanced financial loggers that automatically handle regime validation.

#### Before (Original Code)
```python
from src.training.steps.model_training.step09_financial_logging import Step09FinancialLogger

class Step09HMMBasedTraining:
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = Step09FinancialLogger(symbol, exchange, timeframe)
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        # Your implementation
        data = pipeline_state.get('dataframe', pd.DataFrame())
        
        # Log results
        self.financial_logger.log_step_execution(
            training_results=training_results,
            model_performance=model_performance,
            execution_data=execution_data,
            regime_models=regime_models
        )
        
        return {'success': True}
```

#### After (Enhanced Financial Logger)
```python
from src.training.steps.model_training.step09_financial_logging import EnhancedStep09FinancialLogger

class Step09HMMBasedTraining:
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = EnhancedStep09FinancialLogger(
            symbol, exchange, timeframe, enable_enhanced_logging=True
        )
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        # Your implementation
        data = pipeline_state.get('dataframe', pd.DataFrame())
        
        # Log results with enhanced regime validation
        logging_success = self.financial_logger.log_step_execution(
            training_results=training_results,
            model_performance=model_performance,
            execution_data=execution_data,
            regime_models=regime_models,
            data=data  # This enables regime validation
        )
        
        if not logging_success:
            print("⚠️ Enhanced regime logging failed, but step completed")
        
        return {'success': True, 'logging_success': logging_success}
```

**Benefits:**
- ✅ Enhanced regime validation and fail-fast checks
- ✅ Per-regime metrics logging
- ✅ Backward compatibility with existing code
- ✅ Detailed logging success feedback

### Pattern 3: Manual Integration

For maximum control, manually integrate regime validation and logging.

#### Before (Original Code)
```python
from src.utils.financial_metrics_logger import get_financial_metrics_logger

class Step09HMMBasedTraining:
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        # Your implementation
        data = pipeline_state.get('dataframe', pd.DataFrame())
        
        # Basic logging
        financial_logger = get_financial_metrics_logger()
        financial_logger.log_financial_metric(
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            metric_name="step_success",
            metric_value=1.0,
            metric_type="performance",
            step_name="Step09_HMM_Based_Training"
        )
        
        return {'success': True}
```

#### After (Manual Enhanced Integration)
```python
from src.utils.enhanced_financial_metrics_logger import (
    validate_and_log_regime_data,
    enhanced_financial_metrics_context
)
from src.utils.financial_metrics_logger import log_financial_metric_with_regime_awareness

class Step09HMMBasedTraining:
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        # Your implementation
        data = pipeline_state.get('dataframe', pd.DataFrame())
        
        # Manual regime validation
        if not data.empty and 'composite_cluster_id' in data.columns:
            validation_success = validate_and_log_regime_data(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                step_name="Step09_HMM_Based_Training",
                data=data,
                regime_column='composite_cluster_id'
            )
            
            if not validation_success:
                print("🚨 Regime validation failed - stopping execution")
                return {'success': False, 'error': 'Regime validation failed'}
        
        # Enhanced regime-aware logging
        if not data.empty:
            with enhanced_financial_metrics_context(
                step_name="Step09_HMM_Based_Training",
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                data=data
            ) as enhanced_logger:
                # Log individual metrics with regime awareness
                log_financial_metric_with_regime_awareness(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="step_success",
                    metric_value=1.0,
                    metric_type="performance",
                    step_name="Step09_HMM_Based_Training",
                    data=data
                )
                
                # Log regime-specific metrics
                regime_data = data['composite_cluster_id'].dropna()
                regime_counts = regime_data.value_counts()
                
                regime_metrics = {}
                for regime_id, count in regime_counts.items():
                    regime_metrics[str(regime_id)] = {
                        'sample_count': float(count),
                        'regime_processed': 1.0
                    }
                
                enhanced_logger.log_per_regime_metrics(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step09_HMM_Based_Training",
                    regime_metrics=regime_metrics,
                    data=data
                )
        
        return {'success': True}
```

**Benefits:**
- ✅ Full control over validation and logging
- ✅ Custom regime validation logic
- ✅ Detailed regime-specific metrics
- ✅ Flexible integration patterns

## Step-by-Step Migration Process

### Step 1: Choose Your Integration Pattern

1. **Decorator Pattern**: Best for quick migration with minimal changes
2. **Enhanced Logger Pattern**: Best for existing financial logging integration
3. **Manual Pattern**: Best for custom requirements and full control

### Step 2: Update Imports

Add the necessary imports to your step file:

```python
# For Decorator Pattern
from src.utils.regime_aware_financial_logging_decorator import auto_regime_aware_logging

# For Enhanced Logger Pattern
from src.training.steps.model_training.step09_financial_logging import EnhancedStep09FinancialLogger

# For Manual Pattern
from src.utils.enhanced_financial_metrics_logger import (
    validate_and_log_regime_data,
    enhanced_financial_metrics_context
)
from src.utils.financial_metrics_logger import log_financial_metric_with_regime_awareness
```

### Step 3: Apply the Integration

Choose one of the three patterns above and apply it to your step file.

### Step 4: Test the Integration

1. Run your step with sample data that includes regime information
2. Verify that regime validation works correctly
3. Check that regime-specific metrics are logged
4. Test fail-fast behavior with poor data quality

### Step 5: Update Configuration (Optional)

You can customize the regime validation parameters:

```python
@auto_regime_aware_logging(
    enable_regime_validation=True,      # Enable regime validation
    enable_fail_fast=True,              # Enable fail-fast behavior
    min_regime_samples=100,             # Minimum samples per regime
    max_regime_imbalance=0.8,           # Maximum regime imbalance ratio
    regime_column='composite_cluster_id', # Regime column name
    min_data_quality=0.7                # Minimum data quality threshold
)
```

## Configuration Parameters

### Regime Validation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_regime_validation` | `True` | Enable regime data validation |
| `enable_fail_fast` | `True` | Enable fail-fast behavior on validation failure |
| `min_regime_samples` | `100` | Minimum number of samples per regime |
| `max_regime_imbalance` | `0.8` | Maximum allowed regime imbalance ratio |
| `regime_column` | `'composite_cluster_id'` | Name of the regime column in data |
| `min_data_quality` | `0.7` | Minimum data quality threshold (0.0-1.0) |

### Data Quality Checks

The enhanced regime logging system performs the following data quality checks:

1. **Data Presence**: Ensures data is not empty
2. **Regime Column**: Verifies regime column exists
3. **Regime Distribution**: Checks for sufficient samples per regime
4. **Data Quality**: Validates data quality metrics
5. **Regime Imbalance**: Detects excessive regime imbalance

### Fail-Fast Conditions

The system will fail fast (stop execution) if:

1. **Empty Data**: Data is empty or has insufficient samples
2. **Missing Regime Column**: Regime column is missing from data
3. **Insufficient Regime Samples**: Any regime has fewer than `min_regime_samples`
4. **Excessive Imbalance**: Regime imbalance exceeds `max_regime_imbalance`
5. **Poor Data Quality**: Data quality falls below `min_data_quality`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required modules are available
2. **Data Format Issues**: Verify data contains the expected regime column
3. **Validation Failures**: Check data quality and regime distribution
4. **Performance Issues**: Adjust validation parameters for your data size

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('src.utils.enhanced_financial_metrics_logger').setLevel(logging.DEBUG)
```

### Fallback Behavior

The system is designed to gracefully fall back to standard logging if enhanced features are not available:

- If enhanced logger is not available, falls back to base logger
- If regime validation fails, continues with standard logging
- If decorator is not available, executes without regime awareness

## Best Practices

1. **Start with Decorator Pattern**: Use the decorator pattern for quick migration
2. **Test with Sample Data**: Always test with realistic data containing regime information
3. **Monitor Logging Success**: Check the return value of logging methods
4. **Customize Parameters**: Adjust validation parameters based on your data characteristics
5. **Handle Failures Gracefully**: Implement proper error handling for validation failures

## Migration Checklist

- [ ] Choose integration pattern (decorator, enhanced logger, or manual)
- [ ] Update imports in step file
- [ ] Apply integration pattern
- [ ] Test with sample data
- [ ] Verify regime validation works
- [ ] Check regime-specific metrics are logged
- [ ] Test fail-fast behavior
- [ ] Update documentation
- [ ] Deploy and monitor

## Support

For questions or issues with the enhanced regime logging system:

1. Check the troubleshooting section above
2. Review the example files in the repository
3. Enable debug logging for detailed information
4. Test with sample data to isolate issues

The enhanced regime logging system is designed to be robust and backward-compatible, ensuring a smooth migration process while providing powerful new capabilities for regime-aware financial metrics tracking.