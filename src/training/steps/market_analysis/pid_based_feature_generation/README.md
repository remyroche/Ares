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

## Enhanced Usage Example with All Utilities

```python
from src.training.steps.market_analysis.pid_based_feature_generation import (
    PIDBasedFeatureOrchestrator,
    OrchestratorConfig,
    OptimizedLookbackIntegration
)
from src.utils.common_operations import get_m1_gpu_manager, get_m1_memory_optimizer
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.data.processing.data_processing import DataProcessor

# Enhanced configuration with all utilities enabled
config = OrchestratorConfig(
    # Feature generation limits
    max_interaction_features=100,
    max_polynomial_features=50,
    max_cross_timeframe_features=50,

    # Utility integration flags
    enable_common_operations=True,
    enable_serialization=True,
    enable_data_validation=True,
    enable_data_optimization=True,
    enable_m1_optimization=True,

    # Data quality settings
    min_data_quality_score=0.7,
    max_missing_data_ratio=0.1,
    enable_quality_reporting=True,

    # Performance settings
    enable_profiling=True,
    enable_memory_monitoring=True,
    enable_performance_logging=True,

    # Serialization settings
    save_intermediate_results=True,
    serialization_format='parquet',
    artifacts_directory='artifacts/pid_features'
)

# Initialize orchestrator with enhanced utilities
orchestrator = PIDBasedFeatureOrchestrator(config)
lookback_integration = OptimizedLookbackIntegration()

# Optional: Load optimized historical data using klines parquet
klines_manager = KlinesParquetManager()
historical_data = klines_manager.load_data_range(
    symbol="ETHUSDT",
    interval="1h",
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Use data processing utilities for enhanced data preparation
if historical_data is not None:
    data_processor = DataProcessor()
    market_data = data_processor.preprocess_for_feature_generation(historical_data)
    feature_names = list(market_data.columns)
else:
    # Use standard data
    market_data = your_standard_market_data
    feature_names = your_feature_names

# Integrate optimized lookback periods
lookback_result = lookback_integration.integrate_optimized_lookback_periods(
    feature_lookback_optimization_result,
    feature_names
)

# Generate features with comprehensive utility integration
result = await orchestrator.orchestrate_feature_generation(
    market_data,
    feature_names,
    lookback_result.optimized_lookback_periods,
    target_variable  # Multi-horizon profit probabilities
)

# Access comprehensive results
print(f"Generated {result.total_features_generated} features")
print(f"Quality score: {result.overall_quality_score}")
print(f"Feature names: {result.combined_feature_names}")
print(f"Utility integrations used: {sum(result.utility_integration_status.values())}/{len(result.utility_integration_status)}")
print(f"Memory usage: {result.memory_usage}")
print(f"Artifacts saved: {result.artifact_paths}")

# Access enhanced quality metrics if available
if 'enhanced_metrics' in result.data_quality_report:
    print(f"Enhanced metrics: {result.data_quality_report['enhanced_metrics']}")

# Clean up M1 optimizers
from src.utils.common_operations import cleanup_m1_optimizers
cleanup_m1_optimizers()
```

## Advanced Usage with Specific Utility Modules

### **Data Management with Klines Parquet**
```python
from src.utils.data.klines_parquet import KlinesParquetManager

# Initialize manager
klines_manager = KlinesParquetManager(data_dir="historical_data")

# Get data information
data_info = klines_manager.get_data_info("ETHUSDT", "1h")
print(f"Available data: {data_info['total_records']} records")

# Load optimized data
data = klines_manager.load_data_range("ETHUSDT", "1h", "2024-01-01", "2024-12-31")
```

### **Enhanced Data Processing**
```python
from src.utils.data.processing.data_processing import DataProcessor

# Initialize processor
processor = DataProcessor()

# Enhanced preprocessing
processed_data = processor.preprocess_for_feature_generation(raw_data)
quality_metrics = processor.calculate_enhanced_quality_metrics(processed_data)
```

### **Apple Silicon Optimization**
```python
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

# Initialize M1 optimizers
gpu_manager = M1GPUManager()
memory_optimizer = get_m1_memory_optimizer()
cpu_optimizer = get_m1_cpu_optimizer()

# Get hardware information
gpu_info = gpu_manager.get_gpu_info()
memory_info = memory_optimizer.get_memory_info()
```

### **Safe Mathematical Operations**
```python
from src.utils.math_validation import safe_divide, safe_log, validate_finite

# Safe operations with error handling
result = safe_divide(numerator, denominator, default=0.0)
log_value = safe_log(input_value, default=0.0)
validated_result = validate_finite(calculation_result, "feature_score")
```

### **Advanced Serialization**
```python
from src.utils.serialization_utils import UniversalSerializer

# Universal serialization with multiple format support
serializer = UniversalSerializer()
serializer.save(data, "features.parquet")  # Auto-detects format
loaded_data = serializer.load("features.parquet")
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
- `src.utils.matrix_operations` for optimized matrix operations with GPU support
- `src.training.utils.feature_selection.partial_information_decompositor` for PID analysis
- `src.utils.logger` for logging
- `src.utils.common_operations` for data validation and optimization
- `src.utils.math_validation` for safe mathematical operations
- `src.utils.serialization_utils` for artifact persistence
- `src.utils.data/klines_parquet` for data management
- `src.utils.hardware/m1_*` for Apple Silicon optimization

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