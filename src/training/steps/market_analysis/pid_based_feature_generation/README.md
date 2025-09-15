# PID-Based Feature Generation

This module provides data-driven feature generation using Partial Information Decomposition (PID) to create the most relevant interaction, polynomial, and cross-timeframe features.

## Overview

The PID-based feature generation system replaces the old cross_timeframe_analysis functionality with a comprehensive approach that:

- **Uses optimized lookback periods** from `feature_lookback_optimization` for all subsequent steps
- **Leverages matrix_operations/** for all calculations to ensure hardware-optimized performance
- **Generates up to 200 total features**:
  - Up to 100 interaction features
  - Up to 50 polynomial features  
  - Up to 50 cross-timeframe features
- **Provides comprehensive validation** and error handling
- **Uses hardware-optimized computations** for Apple Silicon M1/M2/M3 Macs

## Key Components

### 1. InteractionFeatureGenerator
Generates data-driven interaction features using PID analysis to identify the most relevant feature interactions.

**Features:**
- Multiplicative interactions (x1 * x2)
- Additive interactions (x1 + x2)
- Ratio interactions (x1 / x2)
- Difference interactions (x1 - x2)
- Correlation-based interactions
- Statistical interactions (normalized, ranked)

**Configuration:**
```python
from .interaction_feature_generator import InteractionFeatureGenerator, InteractionConfig

config = InteractionConfig(
    max_interaction_features=100,
    synergy_threshold=0.1,
    redundancy_threshold=0.15,
    enable_parallel_processing=True,
    enable_gpu_acceleration=True
)

generator = InteractionFeatureGenerator(config)
result = await generator.generate_interaction_features(data, feature_names, optimized_lookback_periods, target)
```

### 2. PolynomialFeatureGenerator
Generates data-driven polynomial features using PID analysis to identify the most relevant polynomial transformations.

**Features:**
- Power features (x^2, x^3, etc.)
- Cross-product features (x1 * x2)
- Interaction features (x1 * x2^2)
- Logarithmic features (log(x))
- Exponential features (exp(x))
- Square root features (sqrt(x))
- Cubic root features (cbrt(x))
- Reciprocal features (1/x)

**Configuration:**
```python
from .polynomial_feature_generator import PolynomialFeatureGenerator, PolynomialConfig

config = PolynomialConfig(
    max_polynomial_features=50,
    max_polynomial_degree=3,
    synergy_threshold=0.1,
    enable_parallel_processing=True,
    enable_gpu_acceleration=True
)

generator = PolynomialFeatureGenerator(config)
result = await generator.generate_polynomial_features(data, feature_names, optimized_lookback_periods, target)
```

### 3. CrossTimeframeFeatureGenerator
Generates data-driven cross-timeframe features using PID analysis to identify the most relevant cross-timeframe relationships.

**Features:**
- Ratio features between timeframes (tf1 / tf2)
- Difference features between timeframes (tf1 - tf2)
- Correlation features between timeframes
- Lag-based correlation features
- Momentum features between timeframes
- Volatility features between timeframes
- Trend alignment features
- Regime consistency features

**Configuration:**
```python
from .cross_timeframe_feature_generator import CrossTimeframeFeatureGenerator, CrossTimeframeConfig

config = CrossTimeframeConfig(
    max_cross_timeframe_features=50,
    timeframes=['1m', '5m', '15m', '1h', '4h', '1d'],
    max_lag_periods=5,
    synergy_threshold=0.1,
    enable_parallel_processing=True,
    enable_gpu_acceleration=True
)

generator = CrossTimeframeFeatureGenerator(config)
result = await generator.generate_cross_timeframe_features(data, feature_names, optimized_lookback_periods, target)
```

### 4. PIDBasedFeatureOrchestrator
Orchestrates all PID-based feature generation processes, integrating interaction, polynomial, and cross-timeframe feature generation.

**Features:**
- Parallel execution of all feature generators
- Comprehensive result combination
- Quality metrics calculation
- Performance monitoring
- Error handling and fallback mechanisms

**Configuration:**
```python
from .pid_based_feature_orchestrator import PIDBasedFeatureOrchestrator, OrchestratorConfig

config = OrchestratorConfig(
    max_interaction_features=100,
    max_polynomial_features=50,
    max_cross_timeframe_features=50,
    enable_interaction_features=True,
    enable_polynomial_features=True,
    enable_cross_timeframe_features=True,
    enable_parallel_processing=True,
    enable_gpu_acceleration=True
)

orchestrator = PIDBasedFeatureOrchestrator(config)
result = await orchestrator.orchestrate_feature_generation(data, feature_names, optimized_lookback_periods, target)
```

### 5. OptimizedLookbackIntegration
Integrates optimized lookback periods from `feature_lookback_optimization` with PID-based feature generation.

**Features:**
- Extracts optimized lookback periods from optimization results
- Applies optimized periods to all feature generation
- Provides fallback mechanisms for missing optimization results
- Validates lookback period effectiveness
- Supports emergency fallback scenarios

**Usage:**
```python
from .optimized_lookback_integration import OptimizedLookbackIntegration

integration = OptimizedLookbackIntegration()
result = integration.integrate_optimized_lookback_periods(feature_lookback_optimization_result, feature_names)
```

### 6. PIDBasedFeatureGenerationComponent
Main component that replaces the old `cross_timeframe_analysis` component while maintaining backward compatibility.

**Features:**
- Drop-in replacement for `CrossTimeframeAnalysisComponent`
- Comprehensive validation and error handling
- Integration with pipeline state
- Detailed reporting and metrics
- Hardware-optimized performance

## Integration with Existing Pipeline

The PID-based feature generation system integrates seamlessly with the existing market analysis pipeline:

1. **Receives optimized lookback periods** from `feature_lookback_optimization`
2. **Uses matrix_operations/** for all calculations
3. **Generates comprehensive feature sets** for downstream analysis
4. **Provides detailed artifacts** for pipeline continuation

## Usage Example

```python
from src.training.steps.market_analysis.pid_based_feature_generation import (
    PIDBasedFeatureOrchestrator, 
    OrchestratorConfig,
    OptimizedLookbackIntegration
)

# Configure orchestrator
config = OrchestratorConfig(
    max_interaction_features=100,
    max_polynomial_features=50,
    max_cross_timeframe_features=50
)

# Initialize components
orchestrator = PIDBasedFeatureOrchestrator(config)
lookback_integration = OptimizedLookbackIntegration()

# Integrate optimized lookback periods
lookback_result = lookback_integration.integrate_optimized_lookback_periods(
    feature_lookback_optimization_result, 
    feature_names
)

# Generate features
result = await orchestrator.orchestrate_feature_generation(
    market_data,
    feature_names,
    lookback_result.optimized_lookback_periods,
    target_variable
)

# Access results
print(f"Generated {result.total_features_generated} features")
print(f"Quality score: {result.overall_quality_score}")
print(f"Feature names: {result.combined_feature_names}")
```

## Performance Optimizations

The system includes several performance optimizations:

- **Hardware acceleration** for Apple Silicon M1/M2/M3 Macs
- **Parallel processing** for feature generation
- **Memory optimization** with chunked processing
- **Matrix operations** using optimized BLAS libraries
- **GPU acceleration** when available

## Quality Metrics

The system provides comprehensive quality metrics:

- **Overall quality score** based on individual generator scores
- **Feature diversity score** based on feature type distribution
- **Redundancy score** based on feature correlations
- **Stability score** based on feature consistency
- **Timeframe coverage** for cross-timeframe features
- **Lag effectiveness** for lag-based features

## Error Handling

Comprehensive error handling includes:

- **Graceful degradation** when components are unavailable
- **Fallback mechanisms** for missing optimization results
- **Validation checks** for data quality and feature validity
- **Detailed error reporting** with recommendations
- **Recovery strategies** for common failure scenarios

## Dependencies

The system requires:

- `numpy` for numerical computations
- `pandas` for data manipulation
- `src.utils.matrix_operations` for optimized matrix operations
- `src.training.utils.feature_selection.partial_information_decompositor` for PID analysis
- `src.utils.logger` for logging

## Migration from Cross Timeframe Analysis

The new system is a drop-in replacement for the old cross_timeframe_analysis:

1. **Same interface** - No code changes required
2. **Enhanced functionality** - More feature types and better quality
3. **Better performance** - Hardware-optimized computations
4. **Improved reliability** - Comprehensive error handling
5. **Better integration** - Uses optimized lookback periods

## Future Enhancements

Planned enhancements include:

- **Additional feature types** (fourier, wavelet, etc.)
- **Advanced PID analysis** with higher-order interactions
- **Real-time feature generation** for live trading
- **Feature importance ranking** using advanced ML techniques
- **Automated hyperparameter optimization** for PID thresholds