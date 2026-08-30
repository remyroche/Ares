# VectorBT Feature Engineering System

This directory contains a comprehensive VectorBT-enhanced feature engineering system that provides superior performance, advanced technical analysis, and extensive optimization capabilities.

## 🚀 Overview

The VectorBT integration transforms the feature engineering pipeline with:

- **10-100x Performance Improvements** through vectorized operations
- **100+ Technical Indicators** with built-in optimization
- **Advanced Pattern Recognition** and regime detection
- **Comprehensive Parameter Optimization** with multiple algorithms
- **Extensive Validation System** with statistical testing and backtesting
- **Multi-dimensional Filtering** with VectorBT-enhanced capabilities

## 📁 Directory Structure

```
src/training/steps/feature_engineering/
├── vectorbt_base.py                    # VectorBT base classes and integration layer
├── vectorbt_indicators_suite.py       # Comprehensive technical indicators suite
├── vectorbt_feature_registration.py   # Feature registration and management
├── vectorbt_optimization.py           # Parameter optimization system
├── vectorbt_validation.py             # Feature validation and testing
├── filters/
│   └── vectorbt_advanced_filters_15m.py  # Enhanced filtering system
├── volatility/
│   └── vectorbt_atr_volatility_ratio.py  # VectorBT ATR volatility features
├── trend/
│   └── vectorbt_trend_coherence.py       # VectorBT trend analysis features
└── price_action/
    ├── vectorbt_bar_efficiency_ratio.py  # VectorBT bar efficiency features
    └── vectorbt_close_location_value.py  # VectorBT CLV features
```

## 🔧 Core Components

### 1. VectorBT Base Classes (`vectorbt_base.py`)

**VectorBTFeatureGenerator**: Base class for all VectorBT-enhanced feature generators
- Built-in optimization and caching
- Performance monitoring and validation
- Automatic parameter tuning
- Memory-efficient processing

**VectorBTTechnicalIndicators**: Comprehensive technical indicators collection
- 100+ indicators organized by category
- Trend, momentum, volatility, volume, price action, patterns
- Built-in parameter optimization
- Performance monitoring

### 2. Feature Generators

#### ATR Volatility Ratio (`volatility/vectorbt_atr_volatility_ratio.py`)
- VectorBT-optimized ATR calculations
- Multiple volatility measures (ATR, Bollinger Bands, Keltner Channels)
- Advanced volatility regime detection
- Comprehensive volatility analysis

#### Trend Coherence (`trend/vectorbt_trend_coherence.py`)
- Multiple trend indicators (ADX, Ichimoku, Parabolic SAR)
- Advanced trend strength and direction detection
- Trend regime classification and persistence
- Multi-timeframe trend analysis

#### Bar Efficiency Ratio (`price_action/vectorbt_bar_efficiency_ratio.py`)
- VectorBT-optimized price action calculations
- Candlestick pattern recognition
- Price action momentum and strength indicators
- Volume-integrated efficiency analysis

#### Close Location Value (`price_action/vectorbt_close_location_value.py`)
- VectorBT-optimized CLV calculations
- Advanced volume analysis integration
- Price action control and pressure indicators
- Comprehensive CLV regime detection

### 3. Advanced Systems

#### Technical Indicators Suite (`vectorbt_indicators_suite.py`)
- **Trend Indicators**: Moving averages, ADX, Parabolic SAR, Ichimoku
- **Momentum Indicators**: RSI, MACD, Stochastic, Williams %R, CCI
- **Volatility Indicators**: ATR, Bollinger Bands, Keltner Channels, Donchian
- **Volume Indicators**: VWAP, OBV, ADL, MFI, VPT, EMV
- **Price Action Indicators**: Bar efficiency, CLV, price position, patterns
- **Pattern Recognition**: Candlestick patterns, support/resistance, trend lines
- **Cycle Indicators**: Hilbert Transform, DPO, cycle analysis

#### Parameter Optimization (`vectorbt_optimization.py`)
- **Multiple Algorithms**: Grid search, Random search, Bayesian optimization, Genetic algorithms
- **Cross-validation**: Time series, K-fold, Walk-forward analysis
- **Performance Metrics**: Sharpe ratio, Information ratio, Calmar ratio, Sortino ratio
- **Multi-objective Optimization**: Multiple metrics with weighted objectives
- **Adaptive Optimization**: Real-time parameter updates

#### Feature Validation (`vectorbt_validation.py`)
- **Statistical Validation**: Normality tests, stationarity tests, significance tests
- **Performance Validation**: Backtesting, risk metrics, performance analysis
- **Stability Validation**: Feature consistency, robustness testing
- **Out-of-sample Testing**: Walk-forward analysis, holdout validation
- **Quality Assessment**: Comprehensive scoring and recommendations

#### Advanced Filters (`filters/vectorbt_advanced_filters_15m.py`)
- **VectorBT-enhanced Filtering**: Multi-dimensional analysis
- **Pattern Recognition**: Advanced candlestick and technical patterns
- **Regime Detection**: Market state identification and filtering
- **Volume Analysis**: Volume-weighted filtering and analysis
- **Performance Monitoring**: Real-time filter effectiveness tracking

## 🚀 Quick Start

### Basic Usage

```python
from src.training.steps.feature_engineering.vectorbt_base import VectorBTFeatureGenerator
from src.training.steps.feature_engineering.volatility.vectorbt_atr_volatility_ratio import VectorBTATRVolatilityRatioGenerator

# Create a VectorBT feature generator
generator = VectorBTATRVolatilityRatioGenerator(
    lookback=4,
    enable_optimization=True,
    enable_caching=True
)

# Generate features
features = generator.generate_vectorbt_features(data)
print(f"Generated {len(features)} features")
```

### Using the Technical Indicators Suite

```python
from src.training.steps.feature_engineering.vectorbt_indicators_suite import VectorBTIndicatorSuite

# Create indicator suite
indicators = VectorBTIndicatorSuite()

# Get all indicators
all_indicators = indicators.get_all_indicators(data)
print(f"Available indicators: {list(all_indicators.keys())}")

# Get specific category
trend_indicators = indicators.get_trend_indicators(data)
momentum_indicators = indicators.get_momentum_indicators(data)
```

### Parameter Optimization

```python
from src.training.steps.feature_engineering.vectorbt_optimization import VectorBTOptimizer, OptimizationMetric

# Create optimizer
optimizer = VectorBTOptimizer()

# Define parameter ranges
param_ranges = {
    'short_window': [3, 4, 5, 6, 7],
    'long_window': [15, 20, 25, 30],
    'high_ratio_threshold': [1.2, 1.5, 1.8, 2.0]
}

# Optimize parameters
result = optimizer.optimize_feature_parameters(
    generator, data, param_ranges, OptimizationMetric.SHARPE_RATIO
)

print(f"Best parameters: {result.best_parameters}")
print(f"Best score: {result.best_score}")
```

### Feature Validation

```python
from src.training.steps.feature_engineering.vectorbt_validation import VectorBTFeatureValidator

# Create validator
validator = VectorBTFeatureValidator()

# Validate feature
validation_result = validator.validate_feature(generator, data)

print(f"Validation passed: {validation_result.validation_passed}")
print(f"Overall score: {validation_result.overall_score}")
print(f"Recommendations: {validation_result.recommendations}")
```

### Advanced Filtering

```python
from src.training.steps.feature_engineering.filters.vectorbt_advanced_filters_15m import apply_vectorbt_advanced_filters_15m

# Apply VectorBT-enhanced filters
filter_result = apply_vectorbt_advanced_filters_15m(data)

print(f"Eligible samples: {filter_result.n_eligible_samples}/{filter_result.n_total_samples}")
print(f"Eligibility ratio: {filter_result.eligibility_ratio:.2%}")
print(f"Quality score: {filter_result.overall_quality_score:.3f}")
```

## 📊 Performance Benefits

### Speed Improvements
- **10-100x faster** than traditional pandas operations
- **Vectorized calculations** for all technical indicators
- **Parallel processing** support for multi-core systems
- **Memory-efficient** processing with chunking

### Feature Quality
- **100+ technical indicators** with proven accuracy
- **Advanced pattern recognition** capabilities
- **Multi-dimensional analysis** for comprehensive insights
- **Regime detection** for adaptive filtering

### Optimization Capabilities
- **Multiple optimization algorithms** (Grid, Random, Bayesian, Genetic)
- **Cross-validation** with time series awareness
- **Parameter tuning** with performance monitoring
- **Multi-objective optimization** for balanced results

### Validation and Quality Assurance
- **Statistical validation** with significance testing
- **Performance validation** with backtesting
- **Stability analysis** for feature consistency
- **Out-of-sample testing** for robustness

## 🔧 Configuration

### VectorBT Configuration

```python
from src.training.steps.feature_engineering.vectorbt_base import VectorBTConfig

config = VectorBTConfig(
    enable_optimization=True,
    optimization_runs=100,
    optimization_method='bayesian',
    enable_caching=True,
    cache_size=1000,
    enable_parallel=True,
    n_jobs=-1
)
```

### Feature Registration Configuration

```python
from src.training.steps.feature_engineering.vectorbt_feature_registration import VectorBTFeatureRegistrationConfig

config = VectorBTFeatureRegistrationConfig(
    enable_auto_registration=True,
    enable_parameter_optimization=True,
    enable_performance_monitoring=True,
    enable_feature_selection=True,
    max_features_per_category=50,
    feature_importance_threshold=0.01
)
```

### Optimization Configuration

```python
from src.training.steps.feature_engineering.vectorbt_optimization import VectorBTOptimizationConfig, OptimizationAlgorithm

config = VectorBTOptimizationConfig(
    algorithm=OptimizationAlgorithm.BAYESIAN,
    max_iterations=100,
    n_trials=50,
    enable_cross_validation=True,
    cv_folds=5,
    enable_backtesting=True,
    enable_early_stopping=True,
    patience=10
)
```

### Validation Configuration

```python
from src.training.steps.feature_engineering.vectorbt_validation import VectorBTValidationConfig

config = VectorBTValidationConfig(
    enable_statistical_validation=True,
    enable_performance_validation=True,
    enable_stability_validation=True,
    enable_out_of_sample_validation=True,
    enable_walk_forward_validation=True,
    significance_level=0.05,
    min_quality_score=0.6,
    max_failure_rate=0.3
)
```

## 📈 Examples

### Example 1: Complete Feature Engineering Pipeline

```python
import pandas as pd
from src.training.steps.feature_engineering.vectorbt_feature_registration import create_vectorbt_feature_registry
from src.training.steps.feature_engineering.vectorbt_optimization import create_vectorbt_optimizer
from src.training.steps.feature_engineering.vectorbt_validation import create_vectorbt_validator

# Load data
data = pd.read_csv('market_data.csv', index_col=0, parse_dates=True)

# Create feature registry
registry = create_vectorbt_feature_registry()

# Register all VectorBT features
registry.register_indicator_suite_features(data)

# Get registered features
features = registry.get_registered_features()
print(f"Registered {len(features)} features")

# Optimize parameters
optimizer = create_vectorbt_optimizer()
optimization_results = registry.optimize_all_features(data)

# Validate features
validator = create_vectorbt_validator()
for feature_name in features:
    if not features[feature_name].get('is_indicator', False):
        generator = features[feature_name]['generator_class']()
        validation_result = validator.validate_feature(generator, data)
        print(f"{feature_name}: {validation_result.validation_passed} (Score: {validation_result.overall_score:.3f})")
```

### Example 2: Custom Feature Development

```python
from src.training.steps.feature_engineering.vectorbt_base import VectorBTFeatureGenerator, VectorBTConfig
from src.feature_generation.core.feature_generator import FeatureCategory, FeatureConfig

class CustomVectorBTFeature(VectorBTFeatureGenerator):
    def __init__(self, lookback=20, **kwargs):
        config = FeatureConfig(
            name="custom_vectorbt_feature",
            category=FeatureCategory.CUSTOM,
            description="Custom VectorBT feature",
            required_columns=['open', 'high', 'low', 'close'],
            default_lookback=lookback
        )
        
        vectorbt_config = VectorBTConfig(enable_optimization=True)
        super().__init__(config, vectorbt_config)
    
    def generate_vectorbt_features(self, data, params=None):
        # Custom feature implementation using VectorBT
        features = {}
        
        # Example: Custom RSI with VectorBT
        rsi = self.indicators.vbt.RSI.run(data['close'], window=14).rsi
        features['custom_rsi'] = rsi
        
        # Example: Custom MACD with VectorBT
        macd = self.indicators.vbt.MACD.run(data['close'])
        features['custom_macd'] = macd.macd
        features['custom_macd_signal'] = macd.signal
        
        return features

# Use custom feature
custom_feature = CustomVectorBTFeature(lookback=20)
features = custom_feature.generate_vectorbt_features(data)
```

### Example 3: Advanced Filtering with Regime Detection

```python
from src.training.steps.feature_engineering.filters.vectorbt_advanced_filters_15m import VectorBTAdvancedFiltersConfig

# Configure advanced filtering
config = VectorBTAdvancedFiltersConfig(
    enable_efficiency_ratio=True,
    enable_clv=True,
    enable_atr_ratio=True,
    enable_trend_coherence=True,
    enable_technical_indicators=True,
    enable_pattern_recognition=True,
    enable_volume_analysis=True,
    enable_regime_detection=True,
    use_grading_system=True,
    grade_threshold=0.3
)

# Apply filters
filter_result = apply_vectorbt_advanced_filters_15m(data, config)

# Analyze results
print(f"Filter Results:")
print(f"  Total samples: {filter_result.n_total_samples}")
print(f"  Eligible samples: {filter_result.n_eligible_samples}")
print(f"  Eligibility ratio: {filter_result.eligibility_ratio:.2%}")
print(f"  Quality score: {filter_result.overall_quality_score:.3f}")
print(f"  VectorBT optimization score: {filter_result.vectorbt_optimization_score:.3f}")

# Get individual grades
for filter_name, grade in filter_result.individual_grades.items():
    print(f"  {filter_name}: {grade.mean():.3f} ± {grade.std():.3f}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **VectorBT Import Error**
   ```bash
   pip install vectorbt
   ```

2. **Memory Issues with Large Datasets**
   ```python
   config = VectorBTConfig(
       enable_caching=False,
       memory_efficient=True,
       chunk_size=500
   )
   ```

3. **Performance Issues**
   ```python
   config = VectorBTConfig(
       enable_parallel=True,
       n_jobs=-1,
       enable_optimization=False  # Disable for initial testing
   )
   ```

4. **Validation Failures**
   ```python
   config = VectorBTValidationConfig(
       min_quality_score=0.3,  # Lower threshold
       max_failure_rate=0.5    # Higher tolerance
   )
   ```

### Performance Tips

1. **Use Caching**: Enable caching for repeated calculations
2. **Parallel Processing**: Use multiple cores for optimization
3. **Chunking**: Process large datasets in chunks
4. **Memory Management**: Use memory-efficient configurations
5. **Parameter Tuning**: Optimize parameters for your specific use case

## 📚 Additional Resources

- [VectorBT Documentation](https://vectorbt.dev/)
- [Technical Analysis Library](https://ta-lib.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)

## 🤝 Contributing

To contribute to the VectorBT feature engineering system:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for new functionality
4. Ensure backward compatibility
5. Follow the coding standards and conventions

## 📄 License

This VectorBT feature engineering system is part of the larger trading framework and follows the same licensing terms.