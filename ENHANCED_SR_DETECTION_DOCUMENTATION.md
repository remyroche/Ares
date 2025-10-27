# Enhanced SR Detection System Documentation

## Overview

The Enhanced SR Detection System is a comprehensive solution that integrates advanced machine learning techniques, optimization strategies, and hardware acceleration to provide superior Support/Resistance level detection. This system represents a significant upgrade from traditional SR detection methods, incorporating VectorBT optimization, SHAP/LIME explainability, advanced validation, and hyperparameter optimization.

## Key Features

### 🚀 **VectorBT Integration**
- **Efficient Time Series Operations**: Leverages VectorBT for optimized rolling window operations and feature generation
- **GPU Acceleration**: Automatic hardware detection and GPU utilization when available
- **Parallel Processing**: Multi-core CPU optimization for large datasets
- **Memory Optimization**: Intelligent memory management for large-scale operations

### 🧠 **ML Explainability**
- **SHAP Integration**: Provides feature importance and model interpretability
- **LIME Explanations**: Local interpretable model-agnostic explanations
- **Feature Importance Analysis**: Detailed breakdown of what drives SR level significance
- **Confidence Scoring**: ML-based confidence metrics for each detected level

### 🔍 **Advanced Validation**
- **Temporal Cross-Validation**: Time-series aware validation to prevent future data leakage
- **Data Leakage Detection**: Comprehensive analysis to identify and prevent data contamination
- **Lookahead Bias Detection**: Advanced techniques to ensure robust model performance
- **Purged Cross-Validation**: Specialized validation for financial time series

### 🎯 **Hyperparameter Optimization (HPO)**
- **Bayesian TPE Optimization**: Efficient hyperparameter tuning using Tree-structured Parzen Estimator
- **Multi-Objective Optimization**: Simultaneous optimization of multiple quality metrics
- **Adaptive Parameter Selection**: Dynamic parameter adjustment based on market conditions
- **Performance Monitoring**: Real-time tracking of optimization progress

### 🖥️ **Hardware Optimization**
- **M1 Mac Optimization**: Specialized optimizations for Apple Silicon
- **CPU/GPU Load Balancing**: Intelligent workload distribution
- **Memory Management**: Advanced memory optimization strategies
- **Performance Monitoring**: Real-time hardware utilization tracking

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced SR Detection System                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   VectorBT      │  │   Hardware      │  │   ML            │  │
│  │   Optimization  │  │   Optimization  │  │   Explainability│  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Advanced      │  │   HPO           │  │   Quality       │  │
│  │   Validation    │  │   Optimization  │  │   Metrics       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Core SR Detection Engine                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Fractal       │  │   Pivot         │  │   Volume        │  │
│  │   Detection     │  │   Detection     │  │   Detection     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Statistical   │  │   Psychological │  │   Fibonacci     │  │
│  │   Detection     │  │   Detection     │  │   Detection     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation and Setup

### Prerequisites

```bash
# Core dependencies
pip install numpy pandas scipy scikit-learn

# VectorBT and optimization
pip install vectorbt numba

# ML explainability
pip install shap lime

# Hardware optimization (M1 Mac)
pip install psutil

# Optional: GPU acceleration
pip install cupy  # For CUDA
pip install mps  # For M1 GPU
```

### Configuration

```python
from src.tactician.sr_levels.enhanced_sr_detection_optimized import (
    EnhancedSROptimizedDetector, SROptimizationConfig
)

# Create configuration
config = SROptimizationConfig(
    # Detection parameters
    min_touches=2,
    tolerance_pct=0.5,
    lookback_periods=100,
    
    # Quality thresholds
    min_r_squared=0.7,
    min_quality_score=0.6,
    min_consistency=0.5,
    
    # Optimization settings
    enable_vectorbt=True,
    enable_hardware_optimization=True,
    enable_explainability=True,
    enable_validation=True,
    enable_hpo=True,
    
    # Performance settings
    max_candidates=1000,
    batch_size=100,
    parallel_workers=4,
    
    # HPO settings
    hpo_trials=50,
    hpo_timeout=300,
    
    # Validation settings
    cv_folds=5,
    gap_periods=10
)
```

## Usage Examples

### Basic Usage

```python
import pandas as pd
import numpy as np
from src.tactician.sr_levels.enhanced_sr_detection_optimized import (
    EnhancedSROptimizedDetector, SROptimizationConfig
)

# Create sample market data
dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='15T')
np.random.seed(42)

base_price = 2000.0
returns = np.random.normal(0, 0.001, len(dates))
prices = base_price * np.exp(np.cumsum(returns))

market_data = pd.DataFrame({
    'open': prices,
    'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
    'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
    'close': prices,
    'volume': np.random.uniform(1000, 10000, len(dates))
}, index=dates)

# Initialize detector
detector = EnhancedSROptimizedDetector()

# Detect SR levels
sr_levels = detector.detect_sr_levels(market_data)

# Print results
print(f"Detected {len(sr_levels)} SR levels")
for level in sr_levels:
    print(f"{level.level_type}: {level.price:.2f} (strength: {level.strength:.2f}, quality: {level.quality_score:.2f})")
```

### Advanced Usage with Custom Configuration

```python
# Create custom configuration
config = SROptimizationConfig(
    min_touches=3,
    tolerance_pct=0.3,
    lookback_periods=200,
    min_quality_score=0.8,
    enable_explainability=True,
    enable_validation=True,
    hpo_trials=100
)

# Initialize detector with custom config
detector = EnhancedSROptimizedDetector(config)

# Detect SR levels
sr_levels = detector.detect_sr_levels(market_data)

# Access detailed information
for level in sr_levels:
    print(f"Level: {level.price:.2f}")
    print(f"  Type: {level.level_type}")
    print(f"  Strength: {level.strength:.3f}")
    print(f"  Quality Score: {level.quality_score:.3f}")
    print(f"  R-squared: {level.r_squared:.3f}")
    print(f"  Consistency: {level.consistency:.3f}")
    print(f"  Feature Importance: {level.feature_importance}")
    print(f"  SHAP Values: {level.shap_values}")
    print(f"  Validation Score: {level.validation_score:.3f}")
    print(f"  Data Leakage Risk: {level.data_leakage_risk:.3f}")
    print()
```

### Integration with Training Pipeline

```python
from src.training.steps.market_analysis.components.sr_detection_enhanced import (
    EnhancedSRDetectionStep
)

# Create step
step = EnhancedSRDetectionStep()

# Prepare input data
input_data = {
    'market_data': market_data,
    'config': {
        'sr_detection': {
            'min_touches': 2,
            'tolerance_pct': 0.5,
            'enable_vectorbt': True,
            'enable_explainability': True,
            'enable_validation': True
        }
    }
}

# Execute step
import asyncio
result = asyncio.run(step.execute(input_data))

# Access results
sr_levels = result['sr_levels']
metadata = result['metadata']
performance_metrics = result['performance_metrics']

print(f"Detected {len(sr_levels)} SR levels")
print(f"Performance: {performance_metrics}")
```

## Configuration Options

### Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_touches` | int | 2 | Minimum number of touches required for a valid SR level |
| `tolerance_pct` | float | 0.5 | Price tolerance percentage for level grouping |
| `lookback_periods` | int | 100 | Number of periods to look back for level detection |

### Quality Thresholds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_r_squared` | float | 0.7 | Minimum R-squared for level quality |
| `min_quality_score` | float | 0.6 | Minimum overall quality score |
| `min_consistency` | float | 0.5 | Minimum consistency score |

### Optimization Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_vectorbt` | bool | True | Enable VectorBT optimization |
| `enable_hardware_optimization` | bool | True | Enable hardware optimization |
| `enable_explainability` | bool | True | Enable ML explainability |
| `enable_validation` | bool | True | Enable advanced validation |
| `enable_hpo` | bool | True | Enable hyperparameter optimization |

### Performance Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_candidates` | int | 1000 | Maximum number of candidate levels |
| `batch_size` | int | 100 | Batch size for processing |
| `parallel_workers` | int | 4 | Number of parallel workers |

## SR Level Object Structure

```python
@dataclass
class SRLevel:
    price: float                    # Price level
    level_type: str                 # 'support' or 'resistance'
    strength: float                 # Level strength (0-1)
    touches: int                    # Number of touches
    first_touch: pd.Timestamp       # First touch timestamp
    last_touch: pd.Timestamp        # Last touch timestamp
    quality_score: float            # Overall quality score (0-1)
    r_squared: float                # R-squared value
    consistency: float              # Consistency score (0-1)
    volatility: float               # Volatility around level
    volume_profile: float           # Volume profile score
    confidence: float               # ML confidence score
    
    # ML explainability
    feature_importance: Dict[str, float]  # Feature importance scores
    shap_values: Optional[Dict[str, float]]  # SHAP values
    lime_explanation: Optional[Dict[str, Any]]  # LIME explanation
    
    # Validation results
    validation_score: float         # Validation score (0-1)
    data_leakage_risk: float        # Data leakage risk (0-1)
    temporal_stability: float       # Temporal stability (0-1)
```

## Performance Optimization

### VectorBT Optimization

The system automatically selects the optimal vectorization strategy based on:
- Data size and dimensions
- Available hardware (CPU/GPU)
- Operation type (technical indicators, backtesting, etc.)

```python
# Check optimization status
status = detector.get_optimization_status()
print(f"VectorBT available: {status['vectorization_available']}")
print(f"Hardware optimization: {status['hardware_optimization_available']}")
```

### Hardware Optimization

For M1 Mac users, the system provides specialized optimizations:
- CPU core utilization
- GPU acceleration (MPS)
- Memory management
- Thermal throttling prevention

### Memory Management

The system includes intelligent memory management:
- Automatic garbage collection
- Memory usage monitoring
- Large dataset handling
- Cache optimization

## Quality Metrics

### R-squared (R²)
Measures the consistency of price levels around the SR level. Higher values indicate more consistent levels.

### Consistency Score
Measures how stable the level is over time. Calculated using coefficient of variation.

### Volatility Score
Measures the volatility around the level. Lower volatility indicates stronger levels.

### Volume Profile
Measures the volume concentration at the level. Higher values indicate stronger levels.

### Quality Score
Overall quality metric combining all individual metrics with weighted importance.

## Validation and Testing

### Running Tests

```bash
# Run comprehensive test suite
python test_enhanced_sr_detection.py

# Run quick validation tests
python -c "from test_enhanced_sr_detection import run_quick_tests; run_quick_tests()"
```

### Test Coverage

The test suite covers:
- Basic functionality
- VectorBT integration
- Hardware optimization
- ML explainability
- Validation components
- HPO integration
- Error handling
- Performance benchmarking
- Configuration validation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce `max_candidates` or `batch_size`
3. **Performance Issues**: Enable hardware optimization
4. **Validation Errors**: Check data format and required columns

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging
detector = EnhancedSROptimizedDetector()
detector.logger.setLevel(logging.DEBUG)
```

### Performance Monitoring

```python
# Get performance metrics
metrics = detector.get_performance_metrics()
print(f"Detection time: {metrics['detection_time']:.3f}s")
print(f"Optimization gains: {metrics['optimization_gains']}")
```

## Best Practices

### Data Preparation
- Ensure OHLCV data is properly formatted
- Use appropriate time frequencies (15T, 1H, 1D)
- Clean data to remove outliers and gaps

### Configuration Tuning
- Start with default parameters
- Use HPO for automatic optimization
- Monitor performance metrics
- Adjust based on market conditions

### Quality Control
- Set appropriate quality thresholds
- Use validation scores for filtering
- Monitor data leakage risk
- Regular performance evaluation

## Future Enhancements

### Planned Features
- Real-time level updates
- Multi-timeframe analysis
- Advanced clustering algorithms
- Deep learning integration
- Cloud deployment support

### Contributing
- Follow code style guidelines
- Add comprehensive tests
- Update documentation
- Submit pull requests

## Support

For issues and questions:
- Check the troubleshooting section
- Review test cases
- Examine configuration options
- Contact the development team

---

**Version**: 2.0.0  
**Last Updated**: 2024  
**Author**: AI Assistant  
**License**: MIT