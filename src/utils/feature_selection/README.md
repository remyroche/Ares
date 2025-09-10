# Step08 Utilities Bank

This directory contains all the utilities that were previously part of step08 in the training pipeline. These utilities have been preserved as a bank of reusable components that can be imported and used by other parts of the system.

## Overview

Step08 was removed from the main training pipeline but its functionality has been preserved as utilities. This ensures zero loss of functionality while making the utilities available for use in other contexts.

## Available Utilities

### 1. Advanced Feature Selection Wrapper (`step08_advanced_feature_selection_wrapper.py`)
- **Purpose**: BaseStep wrapper for advanced feature selection
- **Key Classes**: `AdvancedFeatureSelectionStep`
- **Features**: 
  - BaseStep contract compliance
  - Pipeline integration
  - Error handling and validation
  - Comprehensive logging

### 2. Advanced Feature Selection (`step08_advanced_feature_selection.py`)
- **Purpose**: Main advanced feature selection implementation
- **Key Classes**: `Step08AdvancedFeatureSelection`
- **Features**:
  - M1 hardware optimizations
  - GPU acceleration support
  - Memory management
  - Parallel processing
  - Two-phase feature selection with redundancy reduction
  - Interpretability reporting

### 3. Per-Regime Feature Selection (`step08_advanced_feature_selection_per_regime.py`)
- **Purpose**: Regime-specific feature selection
- **Key Classes**: `Step08AdvancedFeatureSelectionPerRegime`
- **Features**:
  - Regime-aware feature selection
  - Regime-specific optimization
  - Cross-regime feature comparison
  - Regime transition handling

### 4. Optimized Implementations
- **step08_optimized_class.py**: Optimized class-based implementation
- **step08_optimized_execution.py**: Optimized execution strategies
- **step08_optimized_methods.py**: Optimized method implementations
- **step08_optimized.py**: General optimized implementation

### 5. Unified Implementations
- **step08_unified_class.py**: Unified class-based approach
- **step08_unified_complete.py**: Complete unified implementation
- **step08_unified_final.py**: Final unified version
- **step08_unified_methods.py**: Unified method implementations
- **step08_unified_risk.py**: Risk-aware unified implementation
- **step08_unified.py**: General unified implementation

### 6. Regime Data Splitting (`step08_regime_data_splitting.py`)
- **Purpose**: Regime-specific data splitting utilities
- **Key Classes**: `Step08RegimeDataSplitting`
- **Features**:
  - Regime-aware data splitting
  - Temporal consistency maintenance
  - Regime transition handling
  - Data quality validation

## Usage Examples

### Basic Usage
```python
from src.utils.step08_utilities import (
    AdvancedFeatureSelectionStep,
    Step08AdvancedFeatureSelection,
    Step08AdvancedFeatureSelectionPerRegime
)

# Initialize advanced feature selection
feature_selector = Step08AdvancedFeatureSelection(config)
selected_features = await feature_selector.execute(training_input, pipeline_state)

# Use per-regime feature selection
regime_selector = Step08AdvancedFeatureSelectionPerRegime(config)
regime_features = await regime_selector.execute_per_regime(data, regimes)

# Use wrapper for pipeline integration
wrapper = AdvancedFeatureSelectionStep(config)
result = await wrapper.execute_logic(training_input, pipeline_state)
```

### Advanced Usage with Optimizations
```python
from src.utils.step08_utilities import (
    Step08Optimized,
    Step08Unified,
    Step08RegimeDataSplitting
)

# Use optimized implementation
optimized_selector = Step08Optimized(config)
optimized_features = await optimized_selector.execute_optimized(data)

# Use unified implementation
unified_selector = Step08Unified(config)
unified_features = await unified_selector.execute_unified(data)

# Use regime data splitting
regime_splitter = Step08RegimeDataSplitting(config)
split_data = await regime_splitter.split_by_regime(data, regime_labels)
```

### Integration with Other Steps
```python
# In step09 or any other step
from src.utils.step08_utilities import Step08AdvancedFeatureSelection

class Step09HmmBasedTraining:
    def __init__(self, config):
        self.config = config
        # Use step08 utilities for feature selection
        self.feature_selector = Step08AdvancedFeatureSelection(config)
    
    async def process_data(self, data):
        # Use the feature selection utilities
        selected_features = await self.feature_selector.execute(training_input, pipeline_state)
        # Continue with HMM training
        return self.perform_hmm_training(selected_features)
```

## Configuration

The utilities can be configured through the standard configuration system:

```python
config = {
    'step08_advanced_feature_selection': {
        'use_m1_optimizations': True,
        'enable_gpu_acceleration': True,
        'memory_limit_gb': 8.0,
        'max_workers': 8,
        'feature_selection_method': 'mutual_info',
        'redundancy_threshold': 0.8,
        'interpretability_weight': 0.3
    }
}
```

## Key Features

### 1. M1 Hardware Optimizations
- **GPU Acceleration**: Leverages M1 GPU for parallel processing
- **Memory Management**: Efficient memory usage with M1 memory optimizer
- **CPU Optimization**: M1 CPU-specific optimizations
- **Vectorized Processing**: Optimized vectorized operations

### 2. Advanced Feature Selection
- **Two-Phase Selection**: Initial filtering followed by detailed selection
- **Redundancy Reduction**: Removes highly correlated features
- **Interpretability**: Maintains feature interpretability
- **Performance Metrics**: Comprehensive performance evaluation

### 3. Regime-Aware Processing
- **Regime-Specific Selection**: Different feature sets for different regimes
- **Regime Transitions**: Handles regime changes gracefully
- **Cross-Regime Analysis**: Compares features across regimes
- **Temporal Consistency**: Maintains temporal order

### 4. Error Handling and Validation
- **Comprehensive Error Handling**: Graceful handling of various error conditions
- **Data Validation**: Validates input data quality
- **Result Validation**: Ensures output quality
- **Logging**: Detailed logging for debugging and monitoring

## Performance Considerations

- **M1 Optimization**: All utilities support M1 chip optimization
- **Memory Management**: Efficient memory usage with chunking and streaming
- **Parallel Processing**: Support for parallel and async processing
- **Caching**: Built-in caching mechanisms for improved performance
- **GPU Acceleration**: Optional GPU acceleration for compute-intensive operations

## Error Handling

All utilities include comprehensive error handling:
- **Graceful Degradation**: Utilities continue to work even if some optimizations fail
- **Detailed Logging**: Comprehensive logging for debugging and monitoring
- **Validation**: Built-in validation for data quality and feature selection results
- **Recovery**: Automatic recovery mechanisms for transient failures

## Testing

The utilities include comprehensive test suites:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality testing
- **Performance Tests**: Performance and memory usage testing
- **Validation Tests**: Data quality and feature selection validation testing

## Migration Notes

- **Zero Loss of Functionality**: All step08 functionality has been preserved
- **Pipeline Integration**: Step08 has been removed from the main pipeline but utilities remain available
- **Dependency Updates**: Other steps that depended on step08 outputs now depend on step07 outputs
- **Configuration**: Step08-specific configuration has been moved to the utility bank

## Benefits

1. **Zero Loss of Functionality**: All step08 functionality is preserved as utilities
2. **Improved Modularity**: Utilities can be used independently of the pipeline
3. **Better Reusability**: Components can be imported and used by other parts of the system
4. **Cleaner Pipeline**: Simplified pipeline flow without losing capabilities
5. **Maintained Performance**: All optimizations and performance features are preserved
6. **M1 Optimization**: Full M1 hardware optimization support
7. **Regime Awareness**: Advanced regime-aware processing capabilities