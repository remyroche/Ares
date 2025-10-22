# Comprehensive Tools Generalization Guide

## Overview

This guide explains how to use the generalized comprehensive tools from BaseStep in model training components. The generalization provides seamless access to all BaseStep utilities, hardware optimization, advanced logging, and performance monitoring capabilities.

## Architecture

### Core Components

1. **GeneralizedModelTrainingBase**: Base class that inherits from BaseStep and provides comprehensive tools access
2. **ComprehensiveToolsIntegration**: Utility class for simplified access to comprehensive tools
3. **Enhanced Components**: Updated model training components using the generalized approach

### Key Features

- **Full BaseStep Integration**: Access to all BaseStep comprehensive tools
- **Hardware Optimization**: M1 optimizations, memory management, matrix operations
- **Advanced Logging**: TPrint integration with rich data visualization
- **Performance Monitoring**: Comprehensive metrics and analytics
- **Error Handling**: Robust error handling and recovery mechanisms
- **Artifact Management**: Enhanced artifact storage and retrieval
- **Memory Optimization**: Advanced memory management and cleanup

## Usage Patterns

### 1. Basic Usage with GeneralizedModelTrainingBase

```python
from src.training.steps.models_training.core.generalized_model_training_base import (
    GeneralizedModelTrainingBase, ModelTrainingConfig, ModelTrainingRole, ModelType
)

class MyModelTraining(GeneralizedModelTrainingBase):
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(step_name, config)
        
        # All BaseStep comprehensive tools are now available
        # - self.common_ops
        # - self.common_utils
        # - self.math_validation
        # - self.core_decorators
        # - self.ml_common
        # - self.data_quality
        # - self.model_persistence
        # - self.hardware_utils
    
    async def train_models(self, data: pd.DataFrame, targets: Optional[pd.Series] = None):
        # Use comprehensive tools for data processing
        processed_data, processed_targets = self.preprocess_data_with_comprehensive_tools(data, targets)
        
        # Use BaseStep utilities directly
        if self.hardware_utils:
            optimized_data = self.hardware_utils['optimize_dataframe'](processed_data)
        
        # Use convenience methods
        result = self._safe_divide(10, 2, default=0)
        is_valid = self._validate_dataframe_columns(data, ['required_col'])
        
        # Use TPrint functions
        tprint_data_preview(data, "Training Data", max_rows=5)
        tprint_performance_summary(metrics)
        
        return ModelTrainingResult(success=True, models={}, metrics={})
```

### 2. Advanced Usage with ComprehensiveToolsIntegration

```python
from src.training.steps.models_training.utils.comprehensive_tools_integration import (
    ComprehensiveToolsIntegration, ComprehensiveToolsConfig,
    with_comprehensive_tools, with_memory_optimization, with_performance_tracking
)

class AdvancedModelTraining(GeneralizedModelTrainingBase):
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(step_name, config)
        
        # Initialize comprehensive tools integration
        self.comprehensive_tools = ComprehensiveToolsIntegration(
            self, 
            ComprehensiveToolsConfig(
                enable_logging=True,
                enable_performance_monitoring=True,
                enable_memory_optimization=True,
                enable_hardware_optimization=True,
                enable_error_handling=True
            )
        )
    
    @with_comprehensive_tools(
        enable_logging=True,
        enable_performance_monitoring=True,
        enable_memory_optimization=True,
        enable_hardware_optimization=True,
        enable_error_handling=True
    )
    async def train_models(self, data: pd.DataFrame, targets: Optional[pd.Series] = None):
        # Use comprehensive tools integration
        processed_data = self.comprehensive_tools.process_data_with_comprehensive_tools(
            data, "preprocess"
        )
        
        # Use performance monitoring
        @self.comprehensive_tools.monitor_performance("Model Training")
        def train_model():
            return self._train_specific_model(processed_data, targets)
        
        return train_model()
    
    @with_memory_optimization(level="AGGRESSIVE")
    def _process_large_dataset(self, data: pd.DataFrame):
        # Memory optimization is automatically applied
        return data
```

### 3. Decorator-Based Usage

```python
from src.training.steps.models_training.utils.comprehensive_tools_integration import (
    with_comprehensive_tools, with_memory_optimization, with_performance_tracking
)

class DecoratorBasedTraining(GeneralizedModelTrainingBase):
    @with_comprehensive_tools()
    async def train_models(self, data: pd.DataFrame, targets: Optional[pd.Series] = None):
        # Comprehensive tools are automatically available as self.comprehensive_tools
        processed_data = self.comprehensive_tools.process_data_with_comprehensive_tools(
            data, "preprocess"
        )
        
        return ModelTrainingResult(success=True, models={}, metrics={})
    
    @with_memory_optimization(level="AGGRESSIVE")
    def _process_data(self, data: pd.DataFrame):
        # Memory optimization is automatically applied
        return data
    
    @with_performance_tracking("Feature Engineering")
    def _engineer_features(self, data: pd.DataFrame):
        # Performance tracking is automatically applied
        return data
```

## Available Comprehensive Tools

### 1. Data Processing Tools

```python
# Data preprocessing with comprehensive tools
processed_data, processed_targets = self.preprocess_data_with_comprehensive_tools(data, targets)

# Data processing with specific operations
processed_data = self.comprehensive_tools.process_data_with_comprehensive_tools(
    data, "preprocess"  # or "clean", "validate", "optimize"
)

# Safe DataFrame operations
result = self._safe_dataframe_operation(df, "fillna", method="median")
is_valid = self._validate_dataframe_columns(df, ['required_col1', 'required_col2'])
```

### 2. Model Management Tools

```python
# Save models with comprehensive tools
saved_paths = self.save_models_with_comprehensive_tools(models, metadata)

# Load models with comprehensive tools
loaded_models = self.load_models_with_comprehensive_tools(model_paths)

# Individual model operations
model_path = self.comprehensive_tools.save_model_with_comprehensive_tools(
    model, "model_name", metadata
)
model = self.comprehensive_tools.load_model_with_comprehensive_tools("model_name")
```

### 3. Performance Monitoring Tools

```python
# Get comprehensive performance summary
performance_summary = self.get_comprehensive_performance_summary()

# Log comprehensive training summary
self.log_comprehensive_training_summary()

# Monitor specific operations
@self.comprehensive_tools.monitor_performance("Data Processing")
def process_data():
    return self._process_data(data)
```

### 4. Hardware Optimization Tools

```python
# Hardware optimization is automatically applied when enabled
if self.hardware_utils:
    optimized_data = self.hardware_utils['optimize_dataframe'](data)
    memory_stats = self.hardware_utils['get_memory_stats']()
    cleanup_result = self.hardware_utils['force_cleanup']()
```

### 5. Logging and Visualization Tools

```python
# Data preview and visualization
tprint_data_preview(data, "Training Data", max_rows=5)
tprint_data_format(data, "Data Format", level=LogLevel.DEBUG)
tprint_dataframe_info(data, "DataFrame Info")

# Performance logging
tprint_performance_summary(metrics)
tprint_memory_usage(memory_analytics)
tprint_hardware_stats(hardware_stats)

# Structured logging
tprint_dict(metrics, "Training Metrics")
tprint_list(features, "Selected Features")
tprint_model_info(model, "Model Information")
```

### 6. Utility Functions

```python
# Safe mathematical operations
result = self._safe_divide(numerator, denominator, default=0)
value = self._validate_finite(input_value, default=0)
positive_value = self._validate_positive(input_value, default=0)

# File operations
success = self._ensure_directory("/path/to/directory")
exists = self._safe_file_exists("file.txt")

# JSON operations
success = self._safe_json_save(data, "file.json")
data = self._safe_json_load("file.json")

# ML utilities
optimizer = self._get_ml_optimizer("bayesian")
cv_validator = self._get_cv_validator("time_series")
data_cleaner = self._get_data_cleaner()
model_cache = self._get_model_cache()
```

## Configuration

### ModelTrainingConfig

```python
config = {
    'role': 'analyst',  # or 'tactician', 'ensemble', 'regime', 'custom'
    'model_types': ['lightgbm', 'catboost'],
    'timeframe': '15m',
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'validation_split': 0.2,
    'cross_validation_folds': 5,
    'random_seed': 42,
    'enable_hyperparameter_optimization': True,
    'enable_ensemble': True,
    'enable_early_stopping': True,
    'early_stopping_patience': 10,
    'enable_hardware_optimization': True,
    'enable_memory_optimization': True,
    'enable_gpu_acceleration': False,
    'enable_detailed_logging': True,
    'enable_performance_monitoring': True,
    'enable_artifact_management': True,
    'custom_params': {
        'lightgbm_params': {...},
        'catboost_params': {...}
    }
}
```

### ComprehensiveToolsConfig

```python
tools_config = ComprehensiveToolsConfig(
    enable_logging=True,
    enable_performance_monitoring=True,
    enable_memory_optimization=True,
    enable_hardware_optimization=True,
    enable_error_handling=True,
    log_level="INFO",
    memory_optimization_level="AGGRESSIVE",
    cache_enabled=True
)
```

## Migration Guide

### From Existing BaseStep Components

1. **Replace BaseStep inheritance**:
   ```python
   # Old
   class MyComponent(BaseStep):
       pass
   
   # New
   class MyComponent(GeneralizedModelTrainingBase):
       pass
   ```

2. **Update configuration**:
   ```python
   # Old
   config = {'param1': 'value1'}
   
   # New
   config = {
       'role': 'analyst',
       'model_types': ['lightgbm'],
       'enable_hardware_optimization': True,
       'custom_params': {'param1': 'value1'}
   }
   ```

3. **Use comprehensive tools**:
   ```python
   # Old
   processed_data = self._preprocess_data(data)
   
   # New
   processed_data, processed_targets = self.preprocess_data_with_comprehensive_tools(data, targets)
   ```

### From Existing Model Training Components

1. **Update imports**:
   ```python
   from src.training.steps.models_training.core.generalized_model_training_base import (
       GeneralizedModelTrainingBase, ModelTrainingConfig, ModelTrainingResult
   )
   from src.training.steps.models_training.utils.comprehensive_tools_integration import (
       ComprehensiveToolsIntegration, with_comprehensive_tools
   )
   ```

2. **Update class inheritance**:
   ```python
   # Old
   class MyTraining(BaseStep):
       pass
   
   # New
   class MyTraining(GeneralizedModelTrainingBase):
       pass
   ```

3. **Add comprehensive tools integration**:
   ```python
   def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
       super().__init__(step_name, config)
       
       # Add comprehensive tools integration
       self.comprehensive_tools = ComprehensiveToolsIntegration(self)
   ```

## Best Practices

### 1. Use Comprehensive Tools Consistently

```python
# Always use comprehensive tools for data processing
processed_data, processed_targets = self.preprocess_data_with_comprehensive_tools(data, targets)

# Always use comprehensive tools for model management
saved_paths = self.save_models_with_comprehensive_tools(models, metadata)
```

### 2. Enable Appropriate Features

```python
# Enable features based on requirements
config = {
    'enable_hardware_optimization': True,  # For large datasets
    'enable_memory_optimization': True,    # For memory-constrained environments
    'enable_performance_monitoring': True, # For debugging and optimization
    'enable_detailed_logging': True        # For development
}
```

### 3. Use Decorators for Common Patterns

```python
@with_comprehensive_tools()
async def train_models(self, data, targets):
    # Comprehensive tools automatically available
    pass

@with_memory_optimization(level="AGGRESSIVE")
def process_large_data(self, data):
    # Memory optimization automatically applied
    pass
```

### 4. Monitor Performance

```python
# Always log comprehensive training summary
self.log_comprehensive_training_summary()

# Use performance monitoring decorators
@self.comprehensive_tools.monitor_performance("Critical Operation")
def critical_operation(self):
    pass
```

### 5. Handle Errors Gracefully

```python
# Use comprehensive error handling
try:
    result = self.comprehensive_tools.process_data_with_comprehensive_tools(data, "preprocess")
except Exception as e:
    tprint_error(f"❌ Data processing failed: {e}")
    # Handle error appropriately
```

## Examples

### Complete Example: Enhanced Analyst Training

```python
from src.training.steps.models_training.core.generalized_model_training_base import (
    GeneralizedModelTrainingBase, ModelTrainingConfig, ModelTrainingResult, 
    ModelTrainingRole, ModelType
)
from src.training.steps.models_training.utils.comprehensive_tools_integration import (
    ComprehensiveToolsIntegration, with_comprehensive_tools
)

class EnhancedAnalystTraining(GeneralizedModelTrainingBase):
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(step_name, config)
        
        # Initialize comprehensive tools
        self.comprehensive_tools = ComprehensiveToolsIntegration(self)
    
    @with_comprehensive_tools()
    async def train_models(self, data: pd.DataFrame, targets: Optional[pd.Series] = None):
        # 1. Data preprocessing with comprehensive tools
        processed_data, processed_targets = self.preprocess_data_with_comprehensive_tools(data, targets)
        
        # 2. Feature engineering with comprehensive tools
        engineered_data = self.comprehensive_tools.process_data_with_comprehensive_tools(
            processed_data, "preprocess"
        )
        
        # 3. Model training with comprehensive tools
        trained_models = await self._train_models_with_comprehensive_tools(engineered_data, processed_targets)
        
        # 4. Model validation with comprehensive tools
        validation_metrics = await self._validate_models_with_comprehensive_tools(
            engineered_data, processed_targets, trained_models
        )
        
        # 5. Model saving with comprehensive tools
        saved_paths = self.save_models_with_comprehensive_tools(trained_models, validation_metrics)
        
        # 6. Performance monitoring
        self.log_comprehensive_training_summary()
        
        return ModelTrainingResult(
            success=True,
            models=trained_models,
            metrics=validation_metrics,
            artifacts=list(saved_paths.keys())
        )
```

## Troubleshooting

### Common Issues

1. **Comprehensive tools not available**:
   ```python
   # Check availability
   availability = self._get_availability_status()
   print(availability)
   
   # Use fallbacks
   if not self.hardware_utils:
       tprint_warning("⚠️ Hardware utilities not available, using fallbacks")
   ```

2. **Memory issues**:
   ```python
   # Enable memory optimization
   config = {
       'enable_memory_optimization': True,
       'memory_limit_mb': 4096
   }
   
   # Use memory optimization decorators
   @with_memory_optimization(level="AGGRESSIVE")
   def process_data(self, data):
       pass
   ```

3. **Performance issues**:
   ```python
   # Enable performance monitoring
   config = {
       'enable_performance_monitoring': True,
       'enable_detailed_logging': True
   }
   
   # Use performance tracking
   @with_performance_tracking("Operation Name")
   def operation(self):
       pass
   ```

### Debugging

```python
# Print comprehensive tools help
self.print_comprehensive_tools_help()

# Get comprehensive tools status
status = self.get_comprehensive_tools_status()
print(status)

# Log utility availability
self._log_utility_availability()
```

## Conclusion

The generalized comprehensive tools provide a powerful and unified approach to model training with full access to BaseStep utilities. By following this guide, you can leverage all available tools for enhanced data processing, model management, performance monitoring, and error handling in your model training components.