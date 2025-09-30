# Enhanced TAS Regime Detection System

## Overview

The Enhanced TAS (Tree-driven Advanced Statistics) Regime Detection System is a comprehensive solution for detecting market regimes with advanced performance optimizations and validation frameworks. This system extends the base TAS detector with memory-efficient processing, parallel computing, intelligent caching, and comprehensive validation.

## Key Features

### 1. Performance Optimizations

#### Memory Management
- **M1 Memory Optimizer Integration**: Leverages Apple Silicon's unified memory architecture
- **Chunked Processing**: Handles large datasets (>100MB) with automatic chunking
- **Memory Monitoring**: Real-time memory usage tracking and optimization
- **Garbage Collection**: Intelligent memory cleanup and optimization

#### Parallel Processing
- **M1 CPU Optimizer**: Utilizes Apple Silicon's performance and efficiency cores
- **Multi-threading**: Parallel regime detection across timeframes
- **Thread Pool Management**: Optimized thread pools for M1 architecture
- **Load Balancing**: Intelligent workload distribution

#### Intelligent Caching
- **Data Fingerprinting**: MD5-based caching keys for data and results
- **Cache Validation**: Automatic cache expiration and validation
- **Hit Rate Tracking**: Performance monitoring and optimization
- **Memory-Efficient Storage**: Optimized cache storage and retrieval

### 2. Advanced Validation

#### Cross-Validation
- **Time Series CV**: Prevents lookahead bias with time series cross-validation
- **Regime Stability Metrics**: Silhouette score, Calinski-Harabasz score
- **Regime Consistency**: KL divergence-based consistency measurement
- **Transition Stability**: Regime transition frequency analysis

#### Out-of-Sample Testing
- **Walk-Forward Validation**: Robust out-of-sample testing
- **Lookahead Prevention**: Strict temporal separation of training and testing
- **Economic Significance**: Real-world economic metric validation
- **Trading Viability**: Practical trading strategy validation

#### Regime Persistence Analysis
- **Duration Analysis**: Regime duration statistics and patterns
- **Temporal Stability**: Sliding window stability analysis
- **Transition Frequencies**: Regime transition probability analysis
- **Statistical Significance**: Bootstrap-based significance testing

### 3. Bayesian Optimization

#### TPE (Tree-structured Parzen Estimator)
- **Hyperparameter Tuning**: Automated optimization of regime detection parameters
- **Search Space Definition**: Flexible parameter space configuration
- **Multi-objective Optimization**: Combined accuracy, economic significance, and efficiency
- **Convergence Monitoring**: Optimization progress tracking and early stopping

#### Grid Search Integration
- **Coarse-to-Fine Search**: Multi-level parameter optimization
- **Parameter Importance**: Feature importance analysis
- **Optimization History**: Detailed optimization tracking and visualization

### 4. Matrix Operations

#### Unified Matrix Operations
- **GPU Acceleration**: M1 GPU acceleration for large matrix operations
- **Memory Optimization**: Efficient memory usage for large datasets
- **Parallel Processing**: Multi-threaded matrix operations
- **Safe Operations**: Robust error handling and validation

#### Advanced Operations
- **Correlation Analysis**: Safe correlation matrix computation
- **Matrix Inversion**: Conditioned matrix inversion with validation
- **Eigendecomposition**: Eigenvalue and eigenvector analysis
- **SVD Decomposition**: Singular value decomposition with dimensionality reduction

### 5. Hardware Integration

#### M1 GPU Optimization
- **Metal Performance Shaders**: Apple Silicon GPU acceleration
- **Tensor Operations**: Optimized tensor operations for regime detection
- **Memory Management**: Unified memory architecture optimization
- **Fallback Handling**: Graceful degradation to CPU operations

#### M1 Memory Optimization
- **Unified Memory**: Leverages Apple Silicon's unified memory system
- **Memory Pressure Monitoring**: Real-time memory usage tracking
- **Automatic Cleanup**: Intelligent garbage collection and memory optimization
- **Chunked Processing**: Large dataset processing with memory constraints

#### M1 CPU Optimization
- **Performance Cores**: Utilizes Apple Silicon's performance cores
- **Efficiency Cores**: Background tasks on efficiency cores
- **Thread Affinity**: Optimized thread scheduling for M1 architecture
- **Load Balancing**: Intelligent workload distribution

## Usage

### Basic Usage

```python
from src.training.steps.market_analysis.tas_regime.core.enhanced_tas_regime_detector import EnhancedTASRegimeDetector
from src.training.steps.market_analysis.tas_regime.core.tas_regime_config import TASRegimeConfig

# Create configuration
config = TASRegimeConfig.create_research_config()

# Initialize enhanced detector
detector = EnhancedTASRegimeDetector(config)

# Detect regimes with all optimizations
result = detector.detect_regimes_enhanced(
    market_data=market_data,
    timestamps=timestamps,
    enable_bayesian_optimization=True,
    enable_matrix_optimization=True,
    enable_hardware_optimization=True,
    enable_cross_validation=True,
    enable_out_of_sample_validation=True,
    enable_regime_persistence_analysis=True
)
```

### Advanced Usage

```python
# Custom configuration
config = TASRegimeConfig(
    n_regimes=8,
    tree_depth=8,
    n_estimators=1000,
    enable_hardware_optimization=True,
    enable_matrix_optimization=True,
    enable_memory_optimization=True
)

# Initialize with custom settings
detector = EnhancedTASRegimeDetector(config)

# Perform regime detection with specific optimizations
result = detector.detect_regimes_enhanced(
    market_data=market_data,
    enable_bayesian_optimization=True,  # Use Bayesian TPE optimization
    enable_matrix_optimization=True,    # Use enhanced matrix operations
    enable_hardware_optimization=True,  # Use M1 hardware optimizations
    enable_cross_validation=True,      # Perform cross-validation
    enable_out_of_sample_validation=True,  # Perform out-of-sample testing
    enable_regime_persistence_analysis=True  # Analyze regime persistence
)

# Access enhanced results
print(f"Regimes detected: {result.regime_count}")
print(f"Economic significance: {np.mean(result.economic_significance_scores):.3f}")
print(f"Trading viability: {np.mean(result.trading_viability_scores):.3f}")
print(f"Regime stability: {np.mean(result.regime_stability_scores):.3f}")

# Access validation results
if result.cv_scores:
    print(f"Cross-validation scores: {result.cv_scores}")
if result.oos_metrics:
    print(f"Out-of-sample metrics: {result.oos_metrics}")
if result.persistence_analysis:
    print(f"Persistence analysis: {result.persistence_analysis}")
```

## Configuration Options

### TASRegimeConfig Parameters

```python
config = TASRegimeConfig(
    # Core parameters
    n_regimes=8,                    # Number of regimes to detect
    primary_timeframe="15m",        # Primary timeframe
    min_regime_samples=50,         # Minimum samples per regime
    max_regime_samples=10000,      # Maximum samples per regime
    
    # Tree architecture
    tree_depth=8,                  # Maximum tree depth
    n_estimators=1000,            # Number of estimators
    min_samples_split=10,         # Minimum samples to split
    min_samples_leaf=5,            # Minimum samples per leaf
    max_features='sqrt',           # Maximum features per split
    
    # Optimization settings
    enable_hardware_optimization=True,    # Enable M1 hardware optimization
    enable_matrix_optimization=True,     # Enable matrix operations optimization
    enable_memory_optimization=True,     # Enable memory optimization
    optimization_level=TASOptimizationLevel.MAXIMUM,  # Optimization level
    
    # Validation settings
    enable_statistical_methods=True,     # Enable statistical methods
    enable_bootstrap_analysis=True,      # Enable bootstrap analysis
    bootstrap_iterations=1000,           # Number of bootstrap iterations
    
    # Economic evaluation
    enable_economic_evaluation=True,     # Enable economic evaluation
    economic_significance_threshold=0.7, # Economic significance threshold
    trading_viability_threshold=0.6,     # Trading viability threshold
    
    # Meta-learning
    enable_meta_learning=True,           # Enable meta-learning
    adaptation_rate=0.1,                # Adaptation rate
    memory_size=1000,                   # Memory size for meta-learning
)
```

### Enhanced Detection Parameters

```python
result = detector.detect_regimes_enhanced(
    market_data=market_data,
    timestamps=timestamps,
    
    # Optimization flags
    enable_bayesian_optimization=True,      # Bayesian TPE optimization
    enable_matrix_optimization=True,        # Enhanced matrix operations
    enable_hardware_optimization=True,      # M1 hardware optimization
    
    # Validation flags
    enable_cross_validation=True,           # Cross-validation
    enable_out_of_sample_validation=True,  # Out-of-sample validation
    enable_regime_persistence_analysis=True  # Regime persistence analysis
)
```

## Performance Metrics

### Execution Time Breakdown

- **Data Preparation**: Time spent preparing and enhancing data
- **Regime Detection**: Time spent on regime clustering and detection
- **Cross-Validation**: Time spent on cross-validation analysis
- **Out-of-Sample Testing**: Time spent on out-of-sample validation
- **Persistence Analysis**: Time spent on regime persistence analysis
- **Bayesian Optimization**: Time spent on hyperparameter optimization
- **Matrix Operations**: Time spent on matrix operations
- **Hardware Optimization**: Time spent on hardware-specific optimizations

### Memory Usage

- **Peak Memory Usage**: Maximum memory usage during execution
- **Memory Optimization**: Memory saved through optimization
- **Cache Hit Rate**: Percentage of cache hits vs. misses
- **Memory Pressure**: Memory pressure during execution

### Hardware Utilization

- **GPU Utilization**: M1 GPU usage statistics
- **CPU Utilization**: M1 CPU usage statistics
- **Memory Utilization**: Memory usage statistics
- **Optimization Effectiveness**: Hardware optimization effectiveness

## Validation Metrics

### Cross-Validation Metrics

- **Silhouette Score**: Clustering quality metric
- **Calinski-Harabasz Score**: Clustering quality metric
- **Regime Consistency**: Consistency between training and test sets
- **Transition Stability**: Stability of regime transitions

### Out-of-Sample Metrics

- **Regime Accuracy**: Accuracy of regime predictions
- **Regime Stability**: Stability of regime predictions
- **Economic Significance**: Economic significance of regimes
- **Trading Viability**: Trading viability of regimes

### Persistence Analysis

- **Regime Durations**: Duration statistics for each regime
- **Temporal Stability**: Stability over time
- **Transition Frequencies**: Frequency of regime transitions
- **Statistical Significance**: Statistical significance of persistence

## Examples

### Basic Example

```python
# Generate sample data
import numpy as np
import pandas as pd

# Create sample market data
n_samples = 5000
timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='15T')
market_data = pd.DataFrame({
    'timestamp': timestamps,
    'open': np.random.randn(n_samples).cumsum() + 100,
    'high': np.random.randn(n_samples).cumsum() + 102,
    'low': np.random.randn(n_samples).cumsum() + 98,
    'close': np.random.randn(n_samples).cumsum() + 100,
    'volume': np.random.exponential(1000, n_samples)
})

# Initialize enhanced detector
config = TASRegimeConfig.create_research_config()
detector = EnhancedTASRegimeDetector(config)

# Detect regimes
result = detector.detect_regimes_enhanced(market_data)

# Display results
print(f"Success: {result.success}")
print(f"Regimes detected: {result.regime_count}")
print(f"Execution time: {result.execution_time:.2f}s")
```

### Advanced Example

```python
# Custom configuration with all optimizations
config = TASRegimeConfig(
    n_regimes=8,
    tree_depth=10,
    n_estimators=1500,
    enable_hardware_optimization=True,
    enable_matrix_optimization=True,
    enable_memory_optimization=True,
    optimization_level=TASOptimizationLevel.MAXIMUM
)

# Initialize detector
detector = EnhancedTASRegimeDetector(config)

# Perform comprehensive regime detection
result = detector.detect_regimes_enhanced(
    market_data=market_data,
    timestamps=market_data['timestamp'].values,
    enable_bayesian_optimization=True,
    enable_matrix_optimization=True,
    enable_hardware_optimization=True,
    enable_cross_validation=True,
    enable_out_of_sample_validation=True,
    enable_regime_persistence_analysis=True
)

# Analyze results
if result.success:
    print(f"Regimes detected: {result.regime_count}")
    print(f"Economic significance: {np.mean(result.economic_significance_scores):.3f}")
    print(f"Trading viability: {np.mean(result.trading_viability_scores):.3f}")
    print(f"Regime stability: {np.mean(result.regime_stability_scores):.3f}")
    
    # Cross-validation results
    if result.cv_scores:
        print("Cross-validation scores:")
        for metric, value in result.cv_scores.items():
            print(f"  {metric}: {value:.3f}")
    
    # Out-of-sample results
    if result.oos_metrics:
        print("Out-of-sample metrics:")
        for metric, value in result.oos_metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    # Persistence analysis
    if result.persistence_analysis:
        print("Regime persistence analysis:")
        durations = result.persistence_analysis.get('regime_durations', [])
        if durations:
            print(f"  Mean regime duration: {np.mean(durations):.1f} periods")
            print(f"  Max regime duration: {np.max(durations)} periods")
            print(f"  Min regime duration: {np.min(durations)} periods")
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: If you encounter memory issues with large datasets, enable memory optimization and reduce chunk size.

2. **Performance Issues**: If performance is slow, ensure hardware optimization is enabled and check M1 hardware availability.

3. **Validation Issues**: If validation fails, check that you have sufficient data and that the data quality is good.

4. **Optimization Issues**: If Bayesian optimization fails, try reducing the number of trials or using grid search instead.

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for detailed information
detector = EnhancedTASRegimeDetector(config)
result = detector.detect_regimes_enhanced(market_data)
```

## Dependencies

### Required Dependencies

- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning algorithms
- `optuna`: Bayesian optimization (optional)
- `matplotlib`: Plotting (optional)

### Optional Dependencies

- `torch`: PyTorch for GPU acceleration
- `psutil`: System monitoring
- `tqdm`: Progress bars

### Hardware Requirements

- **Apple Silicon**: M1, M2, or M3 Mac for optimal performance
- **Memory**: 8GB+ RAM recommended for large datasets
- **Storage**: SSD recommended for caching

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please see the CONTRIBUTING.md file for guidelines.

## Support

For support and questions, please open an issue on the GitHub repository or contact the development team.