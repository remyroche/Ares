# Production-Ready Abstract Base Classes

This module provides comprehensive abstract base classes for building production-ready machine learning systems. All classes are fully implemented with production features including error handling, validation, logging, performance tracking, and hardware optimization.

## Overview

The abstract base classes provide a consistent interface for common ML operations while allowing for flexible implementations. Each class is designed to be:

- **Production-Ready**: Comprehensive error handling, logging, and monitoring
- **Extensible**: Easy to create custom implementations
- **Optimized**: Hardware optimization and memory management
- **Validated**: Built-in validation and quality checks
- **Documented**: Extensive documentation and type hints

## Base Classes

### 1. BaseValidator

Abstract base class for validation operations with comprehensive validation framework.

**Key Features:**
- Async and sync validation methods
- Multiple validation levels (Basic, Standard, Strict, Production)
- Detailed error reporting and metrics
- Performance tracking and history
- Integration with existing validation utilities

**Required Methods:**
- `validate(data, context)` - Async validation
- `get_validation_summary()` - Get validation statistics

**Example Usage:**
```python
from src.core.abstract_base_classes import BaseValidator, ValidationLevel
from src.core.concrete_implementations import DataValidator

# Create validator
validator = DataValidator(
    name="data_validator",
    validation_level=ValidationLevel.PRODUCTION,
    config={
        'required_columns': ['price', 'volume'],
        'min_samples': 100,
        'max_missing_ratio': 0.05
    }
)

# Validate data
result = await validator.validate(data)
if result.is_valid:
    print("Data validation passed")
else:
    print(f"Validation failed: {result.errors}")

# Get summary
summary = validator.get_validation_summary()
print(f"Success rate: {summary['success_rate']:.2%}")
```

### 2. BaseTrainingStep

Abstract base class for training steps in ML pipelines with full ML workflow support.

**Key Features:**
- Complete ML workflow (data prep, training, validation, evaluation)
- Hardware optimization and memory management
- Performance tracking and monitoring
- Artifact management and persistence
- Integration with existing training utilities

**Required Methods:**
- `_initialize_step_components()` - Initialize step-specific components
- `_process_data(data)` - Process input data for training
- `_generate_artifacts(model, results)` - Generate training artifacts
- `_calculate_metrics(model, test_data)` - Calculate performance metrics

**Example Usage:**
```python
from src.core.abstract_base_classes import BaseTrainingStep
from src.core.concrete_implementations import MLTrainingStep

# Create training step
training_step = MLTrainingStep(
    name="ml_training",
    model_type="random_forest",
    config={
        'n_estimators': 200,
        'max_depth': 10,
        'scale_features': True
    }
)

# Execute training
result = await training_step.execute_training(
    data=(X_train, y_train),
    test_data=(X_test, y_test)
)

if result.success:
    print(f"Training completed in {result.training_time:.2f}s")
    print(f"Model performance: {result.metrics}")
```

### 3. BaseClusteringAlgorithm

Abstract base class for clustering algorithms with comprehensive clustering framework.

**Key Features:**
- Multiple clustering algorithm support
- Performance optimization and validation
- Memory management and hardware optimization
- Detailed metrics and evaluation
- Integration with existing clustering utilities

**Required Methods:**
- `fit_predict(data)` - Fit clustering algorithm and predict labels

**Example Usage:**
```python
from src.core.abstract_base_classes import BaseClusteringAlgorithm, ClusteringAlgorithm
from src.core.concrete_implementations import KMeansClustering

# Create clustering algorithm
clustering = KMeansClustering(
    name="kmeans_clustering",
    n_clusters=5,
    config={
        'random_state': 42,
        'n_init': 10
    }
)

# Perform clustering
result = clustering.fit_predict(data)
print(f"Found {result.n_clusters} clusters")
print(f"Silhouette score: {result.silhouette_score:.3f}")

# Get cluster centers
centers = clustering.get_cluster_centers()
print(f"Cluster centers shape: {centers.shape}")
```

### 4. MultiOutputModel

Abstract base class for multi-output machine learning models with comprehensive multi-output framework.

**Key Features:**
- Support for multiple output targets
- Ensemble methods and stacking
- Performance optimization and validation
- Memory management and hardware optimization
- Detailed metrics and evaluation

**Required Methods:**
- `fit(X, y)` - Fit multi-output model
- `predict(X)` - Make predictions for all outputs

**Example Usage:**
```python
from src.core.abstract_base_classes import MultiOutputModel
from src.core.concrete_implementations import MultiOutputRandomForest

# Create multi-output model
model = MultiOutputRandomForest(
    name="multi_output_rf",
    n_outputs=3,
    output_names=['signal_strength', 'confidence', 'risk_score'],
    config={
        'n_estimators': 150,
        'max_depth': 8
    }
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(f"Predictions shape: {predictions.shape}")

# Evaluate performance
results = model.evaluate_performance(X_test, y_test)
print(f"Overall R²: {results['overall_metrics']['overall_r2']:.3f}")
```

### 5. BasePatternDiscoverer

Abstract base class for pattern discovery algorithms with comprehensive pattern discovery framework.

**Key Features:**
- Mathematical pattern definition and validation
- Pattern discovery and analysis
- Confidence scoring and evaluation
- Integration with existing pattern utilities

**Required Methods:**
- `discover_pattern(data, **kwargs)` - Discover patterns in data
- `get_pattern_definition()` - Get mathematical definition of the pattern

**Example Usage:**
```python
from src.core.abstract_base_classes import BasePatternDiscoverer, PatternType
from src.core.concrete_implementations import MomentumPatternDiscoverer

# Create pattern discoverer
discoverer = MomentumPatternDiscoverer(
    name="momentum_discoverer",
    config={
        'lookback_period': 20,
        'momentum_threshold': 0.03,
        'confidence_threshold': 0.7
    }
)

# Discover patterns
result = discoverer.discover_pattern(price_data)
print(f"Pattern frequency: {result.frequency:.3f}")
print(f"Average confidence: {np.mean(result.confidence_scores):.3f}")

# Get pattern definition
definition = discoverer.get_pattern_definition()
print(f"Pattern formula: {definition.mathematical_formula}")
```

### 6. BaseLabelingStrategy

Abstract base class for labeling strategies with comprehensive labeling framework.

**Key Features:**
- Multiple labeling strategy support
- Confidence calculation and validation
- Performance tracking and optimization
- Integration with existing labeling utilities

**Required Methods:**
- `generate_labels(data, **kwargs)` - Generate labels for data
- `calculate_confidence(labels, data, **kwargs)` - Calculate confidence scores

**Example Usage:**
```python
from src.core.abstract_base_classes import BaseLabelingStrategy, LabelingStrategy
from src.core.concrete_implementations import ProfitBasedLabeling

# Create labeling strategy
labeling = ProfitBasedLabeling(
    name="profit_labeling",
    config={
        'profit_threshold': 0.02,
        'lookforward_period': 5,
        'min_confidence': 0.6
    }
)

# Generate labels
result = labeling.generate_labels(price_data)
print(f"Positive labels: {np.mean(result.labels):.3f}")
print(f"Average confidence: {np.mean(result.confidence_scores):.3f}")

# Calculate confidence
confidence = labeling.calculate_confidence(result.labels, price_data)
print(f"Confidence range: {confidence.min():.3f} - {confidence.max():.3f}")
```

## Concrete Implementations

The module includes concrete implementations for each abstract base class:

- **DataValidator**: Comprehensive data validation with type checking, range validation, and missing value detection
- **MLTrainingStep**: ML model training with preprocessing, hyperparameter optimization, and evaluation
- **KMeansClustering**: K-means clustering with automatic cluster selection and performance metrics
- **MultiOutputRandomForest**: Multi-output random forest with individual models for each output
- **MomentumPatternDiscoverer**: Momentum pattern discovery with mathematical definition and confidence scoring
- **ProfitBasedLabeling**: Profit-based labeling strategy with confidence calculation

## Complete Pipeline Example

Here's how to use all the base classes together in a complete ML pipeline:

```python
import asyncio
from src.core.concrete_implementations import *

async def run_complete_pipeline():
    # 1. Data Validation
    validator = DataValidator("pipeline_validator")
    validation_result = await validator.validate(data)
    
    # 2. Pattern Discovery
    pattern_discoverer = MomentumPatternDiscoverer("momentum_discoverer")
    pattern_result = pattern_discoverer.discover_pattern(price_data)
    
    # 3. Labeling
    labeling = ProfitBasedLabeling("profit_labeling")
    labeling_result = labeling.generate_labels(price_data)
    
    # 4. Clustering
    clustering = KMeansClustering("kmeans_clustering", n_clusters=5)
    clustering_result = clustering.fit_predict(features)
    
    # 5. Multi-output Training
    model = MultiOutputRandomForest("multi_output_rf", n_outputs=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # 6. Performance Evaluation
    results = model.evaluate_performance(X_test, y_test)
    
    return {
        'validation': validation_result,
        'patterns': pattern_result,
        'labels': labeling_result,
        'clusters': clustering_result,
        'predictions': predictions,
        'performance': results
    }

# Run the pipeline
results = asyncio.run(run_complete_pipeline())
```

## Configuration

Each base class accepts a configuration dictionary for customization:

```python
config = {
    # Validation settings
    'required_columns': ['price', 'volume'],
    'min_samples': 100,
    'max_missing_ratio': 0.05,
    
    # Training settings
    'n_estimators': 200,
    'max_depth': 10,
    'scale_features': True,
    
    # Clustering settings
    'n_clusters': 5,
    'random_state': 42,
    
    # Pattern discovery settings
    'lookback_period': 20,
    'momentum_threshold': 0.03,
    
    # Labeling settings
    'profit_threshold': 0.02,
    'lookforward_period': 5
}
```

## Error Handling

All base classes include comprehensive error handling:

```python
try:
    result = await validator.validate(data)
    if not result.is_valid:
        print(f"Validation failed: {result.errors}")
        print(f"Warnings: {result.warnings}")
except Exception as e:
    print(f"Validation error: {e}")
```

## Performance Monitoring

All base classes track performance metrics:

```python
# Get performance summary
summary = validator.get_validation_summary()
print(f"Success rate: {summary['success_rate']:.2%}")
print(f"Average time: {summary['avg_validation_time']:.3f}s")

# Get training summary
training_summary = training_step.get_training_summary()
print(f"Training time: {training_summary['total_training_time']:.2f}s")
```

## Hardware Optimization

The base classes automatically enable hardware optimization when available:

```python
# Hardware optimization is automatically enabled
validator = DataValidator("validator")  # Uses M1 optimization if available
training_step = MLTrainingStep("training")  # Uses memory optimization
```

## Testing

Comprehensive tests are provided for all base classes:

```bash
# Run all tests
pytest src/tests/test_abstract_base_classes.py

# Run specific test class
pytest src/tests/test_abstract_base_classes.py::TestDataValidator

# Run with coverage
pytest src/tests/test_abstract_base_classes.py --cov=src.core
```

## Best Practices

1. **Always use async methods** for better performance in production
2. **Configure validation levels** appropriately for your use case
3. **Monitor performance metrics** to identify bottlenecks
4. **Use hardware optimization** when available
5. **Implement proper error handling** for production robustness
6. **Save and load models** for persistence
7. **Generate comprehensive reports** for monitoring

## Integration

The base classes integrate seamlessly with existing utilities:

- **Logger**: Uses `src.utils.logger.system_logger`
- **Common Operations**: Uses `src.utils.common_operations`
- **Math Validation**: Uses `src.utils.math_validation`
- **Hardware Optimization**: Uses `src.utils.hardware` modules
- **ML Utilities**: Uses existing ML training and validation utilities

This makes the base classes production-ready and consistent with the existing codebase architecture.