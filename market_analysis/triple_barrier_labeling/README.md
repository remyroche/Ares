# Triple Barrier Labeling for Market Analysis

A comprehensive triple barrier labeling system for market analysis, providing multiple implementations, regime-aware labeling, quality assessment, and cross-validation capabilities.

## Features

### Core Functionality
- **Multiple Triple Barrier Implementations**: Standard, regime-aware, fractional, profit-based, and volatility-based labeling
- **Regime-Aware Labeling**: Dynamic barrier parameters based on market regimes
- **Quality Assessment**: Comprehensive label quality validation and metrics
- **Cross-Validation**: Temporal and purged cross-validation for label validation
- **Hardware Optimization**: M1 GPU and CPU optimization support
- **Integration**: Seamless integration with existing utility infrastructure

### Labeling Methods
1. **Standard Triple Barrier**: Classic profit target and stop loss barriers
2. **Regime-Aware Triple Barrier**: Adaptive parameters based on HMM regime states
3. **Enhanced Optimized Triple Barrier**: Advanced optimization with matrix operations and hardware acceleration
4. **Fractional Triple Barrier**: Continuous target labeling
5. **Profit-Based Labels**: Transaction cost-aware labeling

### Advanced Optimizations
- **Three-Stage Optimization**: Coarse Grid → Fine Grid → Bayesian Optimization
- **Matrix Operations**: Vectorized computations using `src/utils/matrix_operations/`
- **Hardware Acceleration**: M1 GPU, CPU, and memory optimizations
- **Math Validation**: Safe mathematical operations using `src/utils/math_validation`
- **Parallel Processing**: Multi-threaded parameter evaluation
- **Memory Optimization**: Efficient memory usage for large datasets

### Three-Stage Optimization Process
1. **Coarse Grid Search**: Explores wide parameter space to find promising regions
2. **Fine Grid Search**: Refines around best coarse candidates with narrower ranges
3. **Bayesian Optimization**: Uses Optuna to fine-tune parameters in optimal regions

### Regime Detection
- **HMM-Based Only**: Uses existing HMM regime states from the pipeline (step03_hmm_regime_discovery)
- **No Custom Detection**: Does not implement custom regime detection methods
- **Pipeline Integration**: Expects HMM regime data to be provided by the existing pipeline

### Barrier Value Calculation
The triple barrier method calculates barrier values as follows:
- **Profit Target Price** = `entry_price * (1 + pt_mult)`
- **Stop Loss Price** = `entry_price * (1 - sl_mult)`

Where:
- `pt_mult`: Profit target multiplier (e.g., 0.002 = 0.2%)
- `sl_mult`: Stop loss multiplier (e.g., 0.001 = 0.1%)
- `entry_price`: The price at which the position is entered

Example: If entry price is $100, pt_mult=0.02, sl_mult=0.01:
- Profit target = $100 * (1 + 0.02) = $102
- Stop loss = $100 * (1 - 0.01) = $99

### Parameter Optimization
- **Optuna Integration**: Uses existing Optuna-based optimization system
- **Regime-Specific**: Optimizes parameters separately for each HMM regime
- **Transaction Costs**: Includes 0.08% fee per trade in optimization
- **Comprehensive Metrics**: Reports pt_mult, sl_mult, trading frequency, win rates, etc.
- **Long/Short Support**: Works with both long and short positions

### Quality Assessment
- Label distribution analysis
- Temporal consistency validation
- Profit consistency evaluation
- Regime balance assessment
- Cross-validation performance
- Statistical significance testing

### Cross-Validation Methods
- Temporal Cross-Validation
- Purged Cross-Validation
- Time Series Cross-Validation
- Regime-Aware Cross-Validation
- Walk-Forward Cross-Validation

## Installation

The module integrates with the existing project structure and requires the following dependencies:

```python
# Core dependencies (already available in the project)
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats

# Project-specific utilities
from src.utils.common_operations import *
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import MathValidation
from src.utils.serialization_utils import UniversalSerializer
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.ml_common.data_processing.data_labeling import EnhancedDataLabeler
```

## Quick Start

### Basic Usage

```python
from market_analysis.triple_barrier_labeling import (
    TripleBarrierLabeler, TripleBarrierConfig, LabelingMethod
)

# Create sample data
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [101, 102, 103, 104, 105],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Configure triple barrier
config = TripleBarrierConfig(
    pt_mult=0.02,  # 2% profit target
    sl_mult=0.01,  # 1% stop loss
    max_holding_period=50
)

# Create labeler and generate labels
labeler = TripleBarrierLabeler(config)
result = labeler.create_labels(data, method=LabelingMethod.TRIPLE_BARRIER)

print(f"Generated {len(result.labels)} labels")
print(f"Label distribution: {result.labels['label'].value_counts().to_dict()}")
```

### Regime-Aware Labeling

```python
from market_analysis.triple_barrier_labeling import (
    RegimeAwareLabeler, RegimeAwareConfig
)

# Create regime data
regime_data = pd.DataFrame({
    'regime': ['bull_market', 'bear_market', 'sideways']
}, index=data.index)

# Configure regime-aware labeling
regime_config = RegimeAwareConfig(
    regime_params={
        'bull_market': TripleBarrierConfig(pt_mult=0.03, sl_mult=0.015),
        'bear_market': TripleBarrierConfig(pt_mult=0.015, sl_mult=0.02),
        'sideways': TripleBarrierConfig(pt_mult=0.02, sl_mult=0.02)
    }
)

# Create regime-aware labels
regime_labeler = RegimeAwareLabeler(regime_config)
labels_df = regime_labeler.create_regime_aware_labels(data, regime_data)
```

### Quality Assessment

```python
from market_analysis.triple_barrier_labeling import LabelQualityAssessment

# Assess label quality
quality_assessor = LabelQualityAssessment()
quality_result = quality_assessor.assess_quality(result.labels, data)

print(f"Overall quality: {quality_result.overall_quality:.3f}")
print(f"Quality level: {quality_result.quality_level.value}")
```

### Cross-Validation

```python
from market_analysis.triple_barrier_labeling import (
    LabelCrossValidator, CVConfig, CVMethod
)

# Prepare features
X = data[['open', 'high', 'low', 'close', 'volume']]
y = result.labels['label']

# Configure cross-validation
cv_config = CVConfig(
    method=CVMethod.TEMPORAL_CV,
    n_splits=5,
    models=['random_forest', 'logistic_regression']
)

# Perform validation
cv_validator = LabelCrossValidator(cv_config)
cv_result = cv_validator.validate_labels(X, y, result.labels)

print(f"Best model: {cv_result.best_model}")
print(f"Validation passed: {cv_result.validation_passed}")
```

## Configuration

### TripleBarrierConfig

```python
@dataclass
class TripleBarrierConfig:
    pt_mult: float = 1.0  # Profit target multiplier
    sl_mult: float = 1.0  # Stop loss multiplier
    min_holding_period: int = 1
    max_holding_period: int = 100
    transaction_cost: float = 0.001
    spread_cost: float = 0.0005
    barrier_type: BarrierType = BarrierType.FIXED
    regime_aware: bool = False
    fractional_support: bool = False
    volatility_adjusted: bool = False
    quality_threshold: float = 0.7
    min_samples_per_label: int = 10
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 10000
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
```

### RegimeAwareConfig

```python
@dataclass
class RegimeAwareConfig:
    regime_detection_method: str = "hmm"  # "hmm", "volatility", "trend", "custom"
    regime_column: str = "regime"
    regime_transition_threshold: float = 0.1
    regime_params: Dict[str, TripleBarrierConfig] = field(default_factory=dict)
    default_config: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    adaptive_parameters: bool = True
    parameter_smoothing: bool = True
    smoothing_window: int = 20
    handle_transitions: bool = True
    transition_buffer: int = 5
    min_samples_per_regime: int = 100
    regime_quality_threshold: float = 0.7
```

## Advanced Usage

### Custom Labeling Method

```python
# Create custom labeling method
def custom_labeling_method(data, config):
    # Your custom implementation
    pass

# Use with the labeler
result = labeler.create_labels(
    data=data,
    method=LabelingMethod.CUSTOM,
    custom_method=custom_labeling_method
)
```

### Hardware Optimization

```python
# Enable M1 optimization
config = TripleBarrierConfig(
    enable_gpu_acceleration=True,
    enable_memory_optimization=True,
    enable_cpu_optimization=True
)

# The system will automatically detect and use M1 optimizations
labeler = TripleBarrierLabeler(config)
```

### Batch Processing

```python
# Process multiple symbols
symbols = ['ETHUSDT', 'BTCUSDT', 'ADAUSDT']
results = {}

for symbol in symbols:
    data = load_market_data(symbol, '1h')
    result = labeler.create_labels(data)
    results[symbol] = result
```

## Integration with Existing Utilities

The module seamlessly integrates with the existing utility infrastructure:

- **Common Operations**: Uses `src.utils.common_operations` for safe mathematical operations
- **Math Validation**: Leverages `src.utils.math_validation` for input validation
- **Data Management**: Integrates with `src.utils.data.klines_parquet` for data loading
- **ML Common**: Utilizes `src.utils.ml_common` for cross-validation and evaluation
- **Hardware Optimization**: Supports M1 GPU and CPU optimization utilities
- **Serialization**: Uses `src.utils.serialization_utils` for data persistence

## Performance Considerations

### Memory Optimization
- Automatic DataFrame dtype optimization
- Batch processing for large datasets
- Memory checkpointing with M1 optimizers

### GPU Acceleration
- M1 MPS support for matrix operations
- Automatic fallback to CPU if GPU unavailable
- Optimized data transfer between CPU and GPU

### Parallel Processing
- Multi-threaded label generation
- Configurable worker count
- Chunked processing for memory efficiency

## Quality Metrics

### Label Distribution
- Balance between positive, negative, and neutral labels
- Minimum and maximum class ratios
- Regime-specific distribution analysis

### Temporal Consistency
- Label transition analysis
- Entropy-based consistency measurement
- Time-based pattern validation

### Profit Consistency
- Coefficient of variation for profits
- Positive/negative profit ratio analysis
- Statistical significance testing

### Cross-Validation Performance
- Temporal cross-validation scores
- Model performance comparison
- Overfitting detection

## Error Handling

The system includes comprehensive error handling:

- Input validation with detailed error messages
- Graceful fallbacks for missing dependencies
- Hardware optimization fallbacks
- Memory management error recovery
- Cross-validation error handling

## Examples

See `example_usage.py` for comprehensive examples including:

1. Basic triple barrier labeling
2. Regime-aware labeling
3. Quality assessment
4. Cross-validation
5. Market analysis utilities
6. Comprehensive workflow

## Contributing

When extending the system:

1. Follow the existing code structure and patterns
2. Add comprehensive error handling
3. Include logging for debugging
4. Update documentation
5. Add unit tests for new functionality

## License

This module is part of the larger market analysis system and follows the same licensing terms.

## Support

For issues and questions:

1. Check the logging output for detailed error messages
2. Review the configuration parameters
3. Verify input data quality
4. Check hardware optimization availability
5. Consult the example usage patterns