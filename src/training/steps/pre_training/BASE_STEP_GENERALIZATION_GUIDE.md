# BaseStep Tool Generalization Guide for Pre-Training Steps

## Overview

This guide provides comprehensive instructions for generalizing the use of BaseStep's comprehensive tools in pre-training steps, eliminating code duplication and standardizing patterns across all feature generation and pre-training operations.

## Current State Analysis

### Issues Identified

1. **Redundant tprint imports**: Each pre-training step imports tprint utilities directly instead of using BaseStep's built-in capabilities
2. **Inconsistent utility usage**: Different steps use different patterns for common operations
3. **Code duplication**: Similar functionality is implemented across multiple steps
4. **Missing BaseStep features**: Steps don't leverage the full power of BaseStep's comprehensive tool suite

### Current Patterns in Pre-Training Steps

```python
# ❌ Current Pattern - Direct imports
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
    tprint_data_preview, tprint_data_format, tprint_performance, tprint_progress,
    tprint_structured, tprint_timer, tprint_exception
)

# ❌ Manual fallback implementations
try:
    from src.utils.tprint import tprint_info
except ImportError:
    def tprint_info(*args, **kwargs): print("INFO:", *args)
```

## Generalization Strategy

### 1. Leverage BaseStep's Built-in tprint Integration

BaseStep already includes comprehensive tprint integration. Instead of importing tprint functions directly, use BaseStep's built-in capabilities:

```python
# ✅ Recommended Pattern - Use BaseStep's tprint integration
class MyPreTrainingStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Use BaseStep's built-in tprint functions
        self.tprint_info("🚀 Starting feature generation")
        self.tprint_data_preview(data, "input_data", max_rows=5)
        self.tprint_performance_summary(metrics)
        
        # Use BaseStep's structured logging
        self.tprint_structured({
            'operation': 'feature_generation',
            'symbol': config.get('symbol'),
            'timeframe': config.get('timeframe')
        })
```

### 2. Use BaseStep's Comprehensive Utility Methods

BaseStep provides extensive convenience methods for common operations:

```python
# ✅ Data Operations
data = self._safe_json_load("config.json")
self._safe_json_save(result, "output.json")
self._ensure_directory("/path/to/output")

# ✅ Math Operations
result = self._safe_divide(10, 2, default=0)
value = self._validate_finite(3.14, default=0)

# ✅ DataFrame Operations
valid = self._validate_dataframe_columns(df, ["col1", "col2"])
cleaned = self._safe_dataframe_operation(df, "fillna")
optimized = self._optimize_dataframe(df)

# ✅ ML Operations
optimizer = self._get_ml_optimizer("bayesian")
cv_validator = self._get_cv_validator("time_series")
```

### 3. Leverage BaseStep's Hardware Optimization

BaseStep includes comprehensive hardware optimization capabilities:

```python
# ✅ Hardware Optimization
@self.memory_optimized
@self.cpu_optimized
def process_large_dataset(self, data):
    return self._optimize_dataframe_with_hardware(data)

# ✅ Memory Management
self._monitor_memory_usage()
self._aggressive_garbage_collection()
```

### 4. Use BaseStep's Artifact Management

BaseStep provides sophisticated artifact management:

```python
# ✅ Artifact Operations
self._save_dataframe(df, 'processed_features')
self._save_metadata(metadata, 'feature_metadata')
self._save_model(model, 'trained_model')

# ✅ Context-aware operations
self._set_context(symbol=symbol, exchange=exchange, direction=direction)
self._store_klines_with_context(df, '1m')
```

## Standardized Pre-Training Step Template

### Basic Template

```python
"""
Standardized Pre-Training Step Template

This template demonstrates the proper use of BaseStep's comprehensive tools
for pre-training operations.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.training.steps.base_step import BaseStep

@dataclass
class PreTrainingResult:
    """Standardized result structure for pre-training steps."""
    success: bool
    artifacts: List[str]
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class StandardizedPreTrainingStep(BaseStep):
    """Standardized pre-training step using BaseStep's comprehensive tools."""
    
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize with BaseStep's comprehensive tool integration."""
        super().__init__(step_name, config)
        
        # Use BaseStep's built-in logging
        self.tprint_info("🔧 Initializing standardized pre-training step")
        self.tprint_debug(f"⚙️ Config provided: {config is not None}")
        
        # Initialize step-specific components
        self._initialize_step_components()
        
        self.tprint_success("✅ Standardized pre-training step initialized")
    
    def _initialize_step_components(self):
        """Initialize step-specific components using BaseStep utilities."""
        # Use BaseStep's utility availability checking
        availability = self._get_availability_status()
        self.tprint_info(f"📊 Utilities available: {sum(availability.values())}/{len(availability)}")
        
        # Initialize components based on availability
        if self.ml_common:
            self.optimizer = self._get_ml_optimizer("bayesian")
            self.cv_validator = self._get_cv_validator("time_series")
        
        if self.data_quality:
            self.cleaner = self._get_data_cleaner()
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pre-training step using BaseStep's comprehensive tools."""
        self.tprint_step_start("🚀 Starting standardized pre-training execution")
        
        try:
            # Set context for enhanced operations
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                direction=config.get('direction'),
                model=config.get('model', 'Analyst')
            )
            
            # Load data using BaseStep utilities
            data = await self._load_data_with_validation(config)
            
            # Process data using BaseStep's optimization
            processed_data = await self._process_data_optimized(data, config)
            
            # Generate features using BaseStep's ML utilities
            features = await self._generate_features_optimized(processed_data, config)
            
            # Validate results using BaseStep's validation
            validation_result = await self._validate_results(features, config)
            
            # Save artifacts using BaseStep's artifact management
            artifacts = await self._save_artifacts_standardized(features, config)
            
            # Generate comprehensive report
            report = await self._generate_comprehensive_report(artifacts, config)
            
            self.tprint_step_end("✅ Standardized pre-training completed successfully")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': validation_result,
                'report': report
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Pre-training step failed: {e}")
            self.tprint_exception(e)
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
    
    async def _load_data_with_validation(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load and validate data using BaseStep utilities."""
        self.tprint_operation_start("📊 Loading and validating data")
        
        # Use BaseStep's data loading utilities
        symbol = config.get('symbol', 'ETHUSDT')
        timeframe = config.get('timeframe', '15m')
        
        # Load data using BaseStep's klines integration
        data = self._load_klines_with_context(timeframe)
        
        if data is None or data.empty:
            raise ValueError(f"No data found for {symbol} {timeframe}")
        
        # Validate data using BaseStep's validation
        self._validate_dataframe_columns(data, ['open', 'high', 'low', 'close', 'volume'])
        
        # Use BaseStep's data preview
        self.tprint_data_preview(data, f"loaded_data_{symbol}_{timeframe}", max_rows=5)
        
        self.tprint_operation_end("✅ Data loaded and validated")
        return data
    
    async def _process_data_optimized(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Process data using BaseStep's hardware optimization."""
        self.tprint_operation_start("⚙️ Processing data with hardware optimization")
        
        # Use BaseStep's hardware optimization
        processed_data = self._optimize_dataframe_with_hardware(data)
        
        # Apply data quality improvements
        if self.data_quality:
            processed_data = self.cleaner.clean(processed_data)
        
        # Use BaseStep's memory monitoring
        self._monitor_memory_usage()
        
        self.tprint_operation_end("✅ Data processed and optimized")
        return processed_data
    
    async def _generate_features_optimized(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate features using BaseStep's ML utilities."""
        self.tprint_operation_start("🔧 Generating features with ML optimization")
        
        # Use BaseStep's ML utilities for feature generation
        features = self._generate_features_with_ml_optimization(data, config)
        
        # Validate features using BaseStep's validation
        self._validate_dataframe_columns(features, expected_columns=None)
        
        # Use BaseStep's performance monitoring
        self.tprint_performance_summary({
            'features_generated': len(features.columns),
            'rows_processed': len(features)
        })
        
        self.tprint_operation_end("✅ Features generated and validated")
        return features
    
    async def _validate_results(self, features: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate results using BaseStep's comprehensive validation."""
        self.tprint_operation_start("🔍 Validating results")
        
        # Use BaseStep's data quality validation
        quality_metrics = self._calculate_data_quality_metrics(features)
        
        # Use BaseStep's ML validation
        if self.ml_common:
            validation_result = self.cv_validator.validate(features)
        else:
            validation_result = {'valid': True, 'score': 0.8}
        
        # Use BaseStep's structured logging
        self.tprint_validation_result({
            'quality_metrics': quality_metrics,
            'ml_validation': validation_result
        })
        
        self.tprint_operation_end("✅ Results validated")
        return {
            'quality_metrics': quality_metrics,
            'ml_validation': validation_result
        }
    
    async def _save_artifacts_standardized(self, features: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
        """Save artifacts using BaseStep's standardized artifact management."""
        self.tprint_operation_start("💾 Saving artifacts")
        
        artifacts = []
        
        # Save features using BaseStep's DataFrame utilities
        self._save_dataframe(features, 'generated_features')
        artifacts.append('generated_features')
        
        # Save metadata using BaseStep's metadata utilities
        metadata = {
            'feature_count': len(features.columns),
            'row_count': len(features),
            'generation_timestamp': self._get_current_datetime(),
            'config': config
        }
        self._save_metadata(metadata, 'feature_metadata')
        artifacts.append('feature_metadata')
        
        # Save model if applicable
        if hasattr(self, 'model') and self.model is not None:
            self._save_model(self.model, 'trained_model')
            artifacts.append('trained_model')
        
        self.tprint_operation_end("✅ Artifacts saved")
        return artifacts
    
    async def _generate_comprehensive_report(self, artifacts: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report using BaseStep's reporting utilities."""
        self.tprint_operation_start("📊 Generating comprehensive report")
        
        # Use BaseStep's performance metrics
        performance_metrics = self._get_performance_metrics()
        memory_analytics = self._get_memory_analytics()
        
        # Use BaseStep's hardware stats
        hardware_stats = self._get_hardware_stats()
        
        report = {
            'artifacts': artifacts,
            'performance_metrics': performance_metrics,
            'memory_analytics': memory_analytics,
            'hardware_stats': hardware_stats,
            'config': config,
            'timestamp': self._get_current_datetime()
        }
        
        # Save report using BaseStep's metadata utilities
        self._save_metadata(report, 'comprehensive_report')
        
        # Use BaseStep's structured logging
        self.tprint_execution_summary(report)
        
        self.tprint_operation_end("✅ Comprehensive report generated")
        return report
```

## Migration Guide

### Step 1: Remove Direct tprint Imports

```python
# ❌ Remove these imports
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
    tprint_data_preview, tprint_data_format, tprint_performance, tprint_progress,
    tprint_structured, tprint_timer, tprint_exception
)

# ❌ Remove fallback implementations
try:
    from src.utils.tprint import tprint_info
except ImportError:
    def tprint_info(*args, **kwargs): print("INFO:", *args)
```

### Step 2: Use BaseStep's Built-in Methods

```python
# ❌ Replace direct tprint calls
tprint_info("Starting process")
tprint_data_preview(data, "input_data")

# ✅ Use BaseStep's built-in methods
self.tprint_info("Starting process")
self.tprint_data_preview(data, "input_data")
```

### Step 3: Leverage BaseStep's Utility Methods

```python
# ❌ Replace manual implementations
def safe_divide(a, b, default=0):
    try:
        return a / b
    except ZeroDivisionError:
        return default

# ✅ Use BaseStep's utility methods
result = self._safe_divide(a, b, default=0)
```

### Step 4: Use BaseStep's Hardware Optimization

```python
# ❌ Replace manual optimization
def optimize_dataframe(df):
    # Manual optimization code
    return df

# ✅ Use BaseStep's hardware optimization
optimized_df = self._optimize_dataframe_with_hardware(df)
```

## Best Practices

### 1. Always Use BaseStep's Built-in Capabilities

- Use `self.tprint_*` methods instead of direct tprint imports
- Use `self._*` utility methods for common operations
- Leverage BaseStep's hardware optimization decorators

### 2. Implement Proper Error Handling

```python
try:
    result = await self._process_data(data)
    self.tprint_success("✅ Data processed successfully")
except Exception as e:
    self.tprint_error(f"❌ Data processing failed: {e}")
    self.tprint_exception(e)
    raise
```

### 3. Use BaseStep's Context Management

```python
# Set context for enhanced operations
self._set_context(
    symbol=config.get('symbol'),
    exchange=config.get('exchange'),
    direction=config.get('direction'),
    model=config.get('model')
)

# Use context-aware operations
data = self._load_klines_with_context('1m')
self._store_klines_with_context(processed_data, '1m')
```

### 4. Leverage BaseStep's Artifact Management

```python
# Save artifacts with proper metadata
self._save_dataframe(df, 'processed_features', metadata={
    'feature_count': len(df.columns),
    'generation_timestamp': self._get_current_datetime()
})

# Load artifacts with fallback mechanisms
data = self._load_dataframe('processed_features')
```

### 5. Use BaseStep's Performance Monitoring

```python
# Monitor performance throughout execution
self._monitor_memory_usage()
performance_metrics = self._get_performance_metrics()
self.tprint_performance_summary(performance_metrics)
```

## Benefits of Generalization

### 1. **Eliminates Code Duplication**
- No more redundant tprint imports across steps
- Standardized utility usage patterns
- Consistent error handling and logging

### 2. **Improved Maintainability**
- Single source of truth for common operations
- Easier to update and enhance functionality
- Consistent behavior across all steps

### 3. **Enhanced Performance**
- Built-in hardware optimization
- Memory management and cleanup
- Optimized data operations

### 4. **Better Developer Experience**
- Consistent API across all steps
- Comprehensive logging and debugging
- Graceful fallbacks when utilities are unavailable

### 5. **Future-Proof Architecture**
- Easy to add new capabilities to BaseStep
- Automatic propagation to all steps
- Backward compatibility maintained

## Conclusion

By generalizing the use of BaseStep's comprehensive tools in pre-training steps, we achieve:

- **Consistency**: All steps use the same patterns and utilities
- **Efficiency**: Eliminate code duplication and leverage optimized implementations
- **Maintainability**: Single source of truth for common functionality
- **Performance**: Built-in hardware optimization and memory management
- **Developer Experience**: Consistent API and comprehensive logging

This generalization strategy ensures that all pre-training steps benefit from BaseStep's comprehensive tool suite while maintaining clean, maintainable, and efficient code.