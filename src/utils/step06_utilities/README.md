# Step06 Utilities Bank

This directory contains all the utilities that were previously part of step06 in the training pipeline. These utilities have been preserved as a bank of reusable components that can be imported and used by other parts of the system.

## Overview

Step06 was removed from the main training pipeline but its functionality has been preserved as utilities. This ensures zero loss of functionality while making the utilities available for use in other contexts.

## Available Utilities

### 1. Utility Container (`step06_utility_container.py`)
- **Purpose**: Dependency injection container for utility services
- **Key Classes**: `Step06UtilityContainer`, `UtilityConfig`
- **Features**: 
  - Service registration and lifecycle management
  - Dependency injection decorators
  - Health monitoring and reporting
  - M1 optimization support

### 2. Enhanced Feature Engineering (`step06_enhanced_feature_engineering.py`)
- **Purpose**: Advanced feature engineering with utility integration
- **Key Classes**: `EnhancedFeatureEngineering`
- **Features**:
  - Vectorized batch processing for indicator extraction
  - Sophisticated feature interactions (polynomial, cross-timeframe, pattern recognition)
  - Strict temporal validation to prevent lookahead bias
  - Memory-efficient chunking for large datasets
  - Mathematical safety with validation utilities

### 3. Comprehensive Implementation (`step06_comprehensive_implementation.py`)
- **Purpose**: Complete implementation of all step06 improvements
- **Key Classes**: `Step06ComprehensiveImplementation`
- **Features**:
  - Integration of all enhanced components
  - Extensive utility integration with dependency injection
  - M1 optimization for performance
  - Advanced data processing and validation

### 4. Feature Engineering Step (`step06_enhanced_feature_engineering_step.py`)
- **Purpose**: Feature engineering step utilities
- **Key Classes**: `EnhancedFeatureEngineeringStep`
- **Features**:
  - Step-specific feature engineering logic
  - Integration with the utility container
  - Comprehensive reporting and validation

### 5. Labeling Components (`step06_labeling_components/`)
- **Purpose**: Triple barrier labeling utilities
- **Key Classes**: 
  - `OptimizedTripleBarrierLabeling`
  - `FractionalTripleBarrierLabeling`
  - `RegimeSpecificTripleBarrierOptimizer`
- **Features**:
  - Vectorized triple barrier labeling
  - Fractional barrier calculations
  - Regime-specific optimization
  - Comprehensive labeling reports

## Usage Examples

### Basic Usage
```python
from src.utils.step06_utilities import (
    Step06UtilityContainer,
    EnhancedFeatureEngineering,
    OptimizedTripleBarrierLabeling
)

# Initialize utility container
utility_config = UtilityConfig(
    enable_common_operations=True,
    enable_data_processing=True,
    enable_math_validation=True
)

container = await get_utility_container(utility_config)

# Use feature engineering utilities
feature_engine = EnhancedFeatureEngineering(config, utility_config)
engineered_features = await feature_engine.process_data(data)

# Use labeling utilities
labeling = OptimizedTripleBarrierLabeling(config)
labeled_data = labeling.apply_triple_barrier_labeling_vectorized(data)
```

### Advanced Usage with Dependency Injection
```python
from src.utils.step06_utilities import inject_utilities

@inject_utilities('common_ops', 'data_proc', 'math_val')
async def process_data_with_utilities(data, common_ops, data_proc, math_val):
    # Use injected utility services
    validated_data = common_ops.get_operation('validation', 'validate_dataframe')(data)
    processed_data = data_proc.process_dataframe(validated_data)
    return processed_data
```

## Configuration

The utilities can be configured through the `UtilityConfig` class:

```python
utility_config = UtilityConfig(
    enable_common_operations=True,
    enable_data_processing=True,
    enable_math_validation=True,
    enable_parquet_utils=True,
    enable_serialization=True,
    enable_m1_gpu=True,
    enable_m1_memory=True,
    enable_m1_cpu=True,
    data_processing_chunk_size=10000,
    m1_memory_limit_gb=8.0,
    m1_max_workers=8
)
```

## Integration with Other Steps

These utilities can be used by other steps in the pipeline or by external components:

```python
# In step07 or any other step
from src.utils.step06_utilities import EnhancedFeatureEngineering

class Step07EnhancedMatrixOperations:
    def __init__(self, config):
        self.config = config
        # Use step06 utilities for feature engineering
        self.feature_engine = EnhancedFeatureEngineering(config)
    
    async def process_data(self, data):
        # Use the feature engineering utilities
        engineered_data = await self.feature_engine.process_data(data)
        # Continue with matrix operations
        return self.perform_matrix_operations(engineered_data)
```

## Migration Notes

- **Zero Loss of Functionality**: All step06 functionality has been preserved
- **Pipeline Integration**: Step06 has been removed from the main pipeline but utilities remain available
- **Dependency Updates**: Other steps that depended on step06 outputs now depend on step05 outputs
- **Configuration**: Step06-specific configuration has been moved to the utility bank

## Performance Considerations

- **M1 Optimization**: All utilities support M1 chip optimization
- **Memory Management**: Efficient memory usage with chunking and streaming
- **Parallel Processing**: Support for parallel and async processing
- **Caching**: Built-in caching mechanisms for improved performance

## Error Handling

All utilities include comprehensive error handling:
- **Graceful Degradation**: Utilities continue to work even if some services fail
- **Detailed Logging**: Comprehensive logging for debugging and monitoring
- **Validation**: Built-in validation for data quality and mathematical operations
- **Recovery**: Automatic recovery mechanisms for transient failures

## Testing

The utilities include comprehensive test suites:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality testing
- **Performance Tests**: Performance and memory usage testing
- **Validation Tests**: Data quality and mathematical validation testing