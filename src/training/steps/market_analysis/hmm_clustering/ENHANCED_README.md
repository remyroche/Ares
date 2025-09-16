# Enhanced HMM Clustering with Common Utilities Integration

This package provides enhanced HMM clustering capabilities with full integration of all available common utilities for optimal performance and reliability in market analysis.

## Overview

The enhanced HMM clustering system integrates with the following common utilities:

- **Hardware Optimization**: M1 GPU, memory, and CPU optimizers
- **Data Operations**: Common operations, utilities, and validation
- **Matrix Operations**: Unified matrix operations for efficient computation
- **ML Common**: Cross-validation, hyperparameter optimization, and validation
- **Serialization**: JSON and pickle serialization utilities
- **Math Validation**: Safe mathematical operations

## Key Enhancements

### 1. Enhanced HMM Executor (`hmm_executor.py`)

**New Features:**
- Full integration with common utilities
- Enhanced data validation using `validate_dataframe_columns` and `calculate_data_quality_metrics`
- Safe math operations using `safe_divide`, `safe_log`, and `validate_finite`
- Matrix operations integration for efficient scaling
- Comprehensive validation metrics (AIC, BIC, convergence)
- Enhanced serialization support

**New Functions:**
- `create_hmm_dependencies()` - Initialize all common utilities
- `save_hmm_results()` - Save results using common serialization utilities

### 2. Enhanced HMM Utils (`hmm_utils.py`)

**New Features:**
- `HMMCommonUtilities` class with comprehensive utility integration
- Safe technical indicator calculations using `safe_divide` and `safe_log`
- Feature preparation with validation
- Cross-validation and hyperparameter optimization
- Results serialization and loading

**Enhanced Functions:**
- `calculate_rsi()` - Now uses safe math operations
- All technical indicators use safe mathematical operations

### 3. Enhanced Clustering Executor (`clustering_executor.py`)

**New Features:**
- Matrix operations integration for feature optimization
- CPU optimization support
- Enhanced quality metrics calculation
- Safe math operations for regime balance calculations
- Comprehensive serialization support

**New Functions:**
- `create_clustering_dependencies()` - Initialize common utilities
- `save_clustering_results()` - Save results using common serialization

## Usage Examples

### Basic Usage

```python
from src.training.steps.market_analysis.hmm_clustering import (
    create_hmm_dependencies,
    train_hmm_optimized,
    HMMCommonUtilities
)

# Initialize HMM common utilities
hmm_utils = HMMCommonUtilities()

# Prepare features with validation
features = hmm_utils.prepare_features_with_validation(your_data)

# Create HMM dependencies with all common utilities
hmm_deps = create_hmm_dependencies()

# Train HMM with optimization
results = train_hmm_optimized(
    features=features,
    n_components=3,
    covariance_type='full',
    n_iter=100,
    random_state=42,
    deps=hmm_deps
)

# Save results
from src.training.steps.market_analysis.hmm_clustering import save_hmm_results
save_hmm_results(results, 'hmm_results.json', hmm_deps)
```

### Advanced Usage with Cross-Validation

```python
from src.training.steps.market_analysis.hmm_clustering import HMMCommonUtilities

# Initialize utilities
hmm_utils = HMMCommonUtilities()

# Prepare features
features = hmm_utils.prepare_features_with_validation(data)

# Run cross-validation
cv_results = hmm_utils.run_cross_validation(
    model=your_hmm_model,
    data=features.values,
    cv_folds=5
)

# Optimize hyperparameters
param_grid = {
    'n_components': [2, 3, 4, 5],
    'covariance_type': ['full', 'tied', 'diag'],
    'n_iter': [50, 100, 200]
}

best_params = hmm_utils.optimize_hyperparameters(
    model_class=YourHMMModel,
    data=features.values,
    param_grid=param_grid
)
```

### Clustering with Common Utilities

```python
from src.training.steps.market_analysis.hmm_clustering import (
    create_clustering_dependencies,
    kmeans_standard,
    kmeans_minibatch
)

# Create clustering dependencies
clustering_deps = create_clustering_dependencies()

# Run KMeans clustering with optimization
kmeans_results = kmeans_standard(
    features_array=features.values,
    n_clusters=3,
    random_state=42,
    logger=clustering_deps.logger,
    deps=clustering_deps
)

# Run MiniBatch KMeans clustering
minibatch_results = kmeans_minibatch(
    features_array=features.values,
    n_clusters=3,
    random_state=42,
    logger=clustering_deps.logger,
    deps=clustering_deps
)
```

## Common Utilities Integration

### Hardware Utilities

The enhanced modules automatically use available hardware optimizations:

```python
# GPU acceleration (M1/M2/M3)
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
gpu_manager = get_m1_gpu_manager()

# Memory optimization
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
memory_optimizer = get_m1_memory_optimizer()

# CPU optimization
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
cpu_optimizer = get_m1_cpu_optimizer()
```

### Data Operations

```python
# Data validation and quality metrics
from src.utils.common_operations import (
    validate_dataframe_columns,
    calculate_data_quality_metrics
)

# Safe data type conversion
from src.utils.common_utilities import safe_convert_dtypes

# Safe mathematical operations
from src.utils.math_validation import safe_divide, safe_log, validate_finite
```

### Matrix Operations

```python
# Unified matrix operations
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

matrix_ops = UnifiedMatrixOperations()
optimized_data = matrix_ops.optimize_for_clustering(features)
```

### ML Common Utilities

```python
# Cross-validation
from src.utils.ml_common.validation.cross_validation import TimeSeriesCrossValidator

# Hyperparameter optimization
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer

# HMM regime detection
from src.utils.ml_common.hmm_regime_detection import HMMRegimeDetector
```

### Serialization

```python
# JSON and Pickle serialization
from src.utils.serialization_utils import JSONSerializer, PickleSerializer

json_serializer = JSONSerializer()
pickle_serializer = PickleSerializer()

# Save/load data
json_serializer.save(results, 'results.json')
pickle_serializer.save(model, 'model.pkl')
```

## Performance Benefits

### 1. Hardware Optimization
- **GPU Acceleration**: Automatic GPU usage when available (M1/M2/M3)
- **Memory Optimization**: Efficient memory usage for large datasets
- **CPU Optimization**: Optimal thread usage for multi-core systems

### 2. Data Processing
- **Validation**: Comprehensive data validation and quality checks
- **Safe Operations**: Robust mathematical operations with fallbacks
- **Matrix Optimization**: Efficient feature processing and scaling

### 3. ML Workflows
- **Cross-Validation**: Robust model evaluation
- **Hyperparameter Optimization**: Automated parameter tuning
- **Serialization**: Efficient model and results persistence

## Error Handling and Validation

The enhanced modules include comprehensive error handling:

- **Data Validation**: Automatic validation of input data
- **Safe Math Operations**: Protection against division by zero and invalid operations
- **Fallback Mechanisms**: Graceful degradation when utilities are not available
- **Detailed Logging**: Comprehensive logging for debugging and monitoring

## Dependencies

### Required
- numpy
- pandas
- scikit-learn
- hmmlearn

### Optional (for enhanced features)
- psutil (memory monitoring)
- tensorflow-metal (M1 GPU acceleration)
- optuna (advanced hyperparameter optimization)

## Example: Complete Workflow

```python
#!/usr/bin/env python3
"""Complete HMM clustering workflow with common utilities."""

import pandas as pd
import numpy as np
from src.training.steps.market_analysis.hmm_clustering import (
    create_hmm_dependencies,
    train_hmm_optimized,
    HMMCommonUtilities,
    create_clustering_dependencies,
    kmeans_standard
)

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'close': np.random.randn(1000).cumsum() + 100,
    'volume': np.random.lognormal(10, 0.5, 1000)
})

# Initialize HMM utilities
hmm_utils = HMMCommonUtilities()

# Prepare features with validation
features = hmm_utils.prepare_features_with_validation(data)

# Calculate technical indicators
features_with_indicators = hmm_utils.calculate_technical_indicators_safe(features)

# Create HMM dependencies
hmm_deps = create_hmm_dependencies()

# Train HMM
hmm_results = train_hmm_optimized(
    features=features_with_indicators,
    n_components=3,
    covariance_type='full',
    n_iter=100,
    random_state=42,
    deps=hmm_deps
)

# Run clustering
clustering_deps = create_clustering_dependencies()
numeric_features = features_with_indicators.select_dtypes(include=[np.number])
kmeans_results = kmeans_standard(
    features_array=numeric_features.values,
    n_clusters=3,
    random_state=42,
    logger=clustering_deps.logger,
    deps=clustering_deps
)

# Save results
hmm_utils.save_results(hmm_results, 'hmm_results.json')
hmm_utils.save_results(kmeans_results, 'kmeans_results.json')

print("✅ HMM clustering workflow completed successfully!")
```

## Migration Guide

### From Original HMM Clustering

1. **Replace direct function calls** with dependency-injected versions
2. **Use `create_hmm_dependencies()`** to initialize common utilities
3. **Add validation** using `HMMCommonUtilities.prepare_features_with_validation()`
4. **Use safe math operations** in custom calculations
5. **Save results** using the new serialization functions

### Benefits of Migration

- **Better Performance**: Hardware optimization and matrix operations
- **More Reliable**: Comprehensive validation and error handling
- **Easier to Use**: Simplified API with automatic utility integration
- **Better Monitoring**: Enhanced logging and metrics
- **Future-Proof**: Integration with all common utilities

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify all dependencies are installed
3. Ensure data is properly formatted
4. Check hardware compatibility
5. Review the enhanced usage examples

The enhanced HMM clustering system provides a robust, performant, and easy-to-use solution for market analysis with full integration of all available common utilities.