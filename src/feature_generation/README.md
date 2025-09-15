# Unified Feature Generation System

A centralized, category-based feature generation system that consolidates all scattered feature generation code into a single source of truth while maintaining full backwards compatibility.

## 🚀 Key Features

- **Category-based Organization**: Features organized by categories (returns, momentum, volume, support/resistance, etc.)
- **Matrix Operations Integration**: Leverages the existing `matrix_operations/` framework for optimized computation
- **Lookback Optimization**: Data-driven optimization of feature lookback periods
- **Feature Bank**: Central registry where scripts can pick needed features by category
- **Backwards Compatibility**: Seamless integration with existing feature generation code
- **Hardware Acceleration**: Apple Silicon M1/M2/M3 optimization support

## 📁 Architecture

```
src/feature_generation/
├── __init__.py                 # Main module interface
├── core/                       # Core framework
│   ├── feature_bank.py        # Central feature registry and management
│   ├── feature_generator.py   # Base classes and interfaces
│   ├── feature_registry.py    # Feature organization and search
│   └── factory.py             # Factory functions
├── categories/                 # Category-specific generators
│   ├── returns.py             # Returns features (price returns, log returns, etc.)
│   ├── momentum.py            # Momentum indicators (RSI, MACD, etc.)
│   ├── volume.py              # Volume features (OBV, VWAP, etc.)
│   ├── volatility.py          # Volatility measures (Bollinger Bands, ATR, etc.)
│   ├── trend.py               # Trend indicators (moving averages, etc.)
│   ├── oscillator.py          # Oscillator indicators (Stochastic, Williams %R, etc.)
│   ├── support_resistance.py  # Support/resistance features
│   ├── candlestick_pattern.py # Candlestick pattern recognition
│   └── hmm_regime.py          # HMM regime features
├── optimization/               # Lookback optimization system
│   └── lookback_optimizer.py  # Leverages existing optimization code
├── matrix_integration/         # Matrix operations integration
│   └── matrix_processor.py    # Optimized computation using matrix_operations/
├── compatibility/              # Backwards compatibility layer
│   └── legacy_adapter.py      # Integration with existing code
└── convenience/                # Convenience functions
    └── convenience_functions.py # Easy-to-use functions
```

## 🎯 Usage Examples

### Basic Usage

```python
from src.feature_generation import (
    FeatureBank,
    generate_features_by_category,
    get_feature_summary
)

# Initialize feature bank
bank = FeatureBank()

# Generate features by category
features = bank.generate_features(
    data=df,
    categories=['returns', 'momentum', 'volume'],
    lookback_optimization=True,
    target_column='target'
)

# Or use convenience function
features = generate_features_by_category(
    data=df,
    categories=['returns', 'momentum', 'volume']
)
```

### Advanced Usage with Lookback Optimization

```python
from src.feature_generation import (
    FeatureBank,
    FeatureBankConfig,
    FeatureCategory
)

# Configure feature bank
config = FeatureBankConfig(
    enable_matrix_operations=True,
    enable_gpu_acceleration=True,
    enable_lookback_optimization=True,
    parallel_processing=True
)

bank = FeatureBank(config)

# Generate features with optimization
features = bank.generate_features(
    data=df,
    categories=[FeatureCategory.RETURNS, FeatureCategory.MOMENTUM],
    lookback_optimization=True,
    target_column='returns_1d'
)
```

### Category-Specific Feature Generation

```python
from src.feature_generation import (
    ReturnsFeatureGenerator,
    MomentumFeatureGenerator,
    VolumeFeatureGenerator
)

# Create specific generators
returns_gen = ReturnsFeatureGenerator()
momentum_gen = MomentumFeatureGenerator()
volume_gen = VolumeFeatureGenerator()

# Generate features
returns_features = returns_gen.generate(df)
momentum_features = momentum_gen.generate(df)
volume_features = volume_gen.generate(df)
```

### Using the Feature Bank

```python
from src.feature_generation import FeatureBank

bank = FeatureBank()

# List available categories
categories = bank.list_categories()
print(f"Available categories: {[cat.value for cat in categories]}")

# List features in a category
returns_features = bank.list_features(FeatureCategory.RETURNS)
print(f"Returns features: {returns_features}")

# Get specific generator
rsi_generator = bank.get_generator_by_name("rsi_14")

# Generate specific features
features = bank.generate_specific_features(
    data=df,
    feature_names=["rsi_14", "macd_12_26_9", "bb_upper_20_2"]
)
```

### Matrix Operations Integration

```python
from src.feature_generation import (
    enable_matrix_acceleration,
    get_matrix_processor
)

# Enable matrix acceleration
enable_matrix_acceleration(True)

# Get matrix processor
processor = get_matrix_processor(enable_gpu=True, enable_parallel=True)

# Process features with matrix optimization
results = processor.process_features(generators, data)
```

### Backwards Compatibility

```python
from src.feature_generation import (
    LegacyFeatureAdapter,
    migrate_legacy_features
)

# Create legacy adapter
adapter = LegacyFeatureAdapter()

# Migrate existing feature generation code
legacy_config = {
    'sma_20': {
        'category': 'trend',
        'description': 'Simple Moving Average',
        'required_columns': ['close'],
        'parameters': {'period': 20}
    }
}

generators = migrate_legacy_features(legacy_config)
```

## 📊 Available Feature Categories

### Returns Features
- Simple returns (1d, 5d, 10d, 20d)
- Log returns
- Cumulative returns
- Return volatility
- Return skewness and kurtosis

### Momentum Features
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R
- Rate of Change (ROC)
- Momentum indicator

### Volume Features
- Volume moving averages
- Volume ratios
- On-Balance Volume (OBV)
- Volume Weighted Average Price (VWAP)
- Volume Rate of Change
- Volume-Price Trend (VPT)
- Accumulation/Distribution Line

### Volatility Features
- Bollinger Bands
- Average True Range (ATR)
- Rolling volatility
- Volatility of volatility

### Trend Features
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- Trend strength indicators

### Support/Resistance Features
- Pivot points
- Support/resistance levels
- Breakout indicators

### Candlestick Pattern Features
- Doji patterns
- Hammer patterns
- Engulfing patterns
- Other candlestick formations

### HMM Regime Features
- Regime detection
- Regime-specific features
- Regime transition indicators

## 🔧 Configuration

### Feature Bank Configuration

```python
from src.feature_generation import FeatureBankConfig

config = FeatureBankConfig(
    enable_matrix_operations=True,      # Enable matrix operations integration
    enable_gpu_acceleration=True,       # Enable GPU acceleration
    enable_lookback_optimization=True,  # Enable lookback optimization
    enable_parallel_processing=True,    # Enable parallel processing
    max_workers=4,                      # Number of parallel workers
    chunk_size=1000,                    # Chunk size for processing
    memory_efficient=True,              # Enable memory optimization
    cache_results=True,                 # Cache generated features
    default_lookback=20                 # Default lookback period
)
```

### Lookback Optimization Configuration

```python
from src.feature_generation import FeatureOptimizationConfig, OptimizationMethod

config = FeatureOptimizationConfig(
    min_lookback=5,                     # Minimum lookback period
    max_lookback=252,                   # Maximum lookback period
    step_size=1,                        # Step size for optimization
    optimization_method=OptimizationMethod.STATISTICAL_ANALYSIS,
    cv_folds=5,                         # Cross-validation folds
    stability_threshold=0.8,            # Stability threshold
    performance_threshold=0.6,          # Performance threshold
    regime_aware=True,                  # Enable regime-aware optimization
    parallel_processing=True,           # Enable parallel optimization
    max_workers=4                       # Number of parallel workers
)
```

## 🚀 Performance Features

### Matrix Operations Integration
- Leverages existing `matrix_operations/` framework
- Vectorized operations for improved performance
- GPU acceleration support (Apple Silicon M1/M2/M3)
- Memory-efficient batch processing

### Parallel Processing
- Multi-threaded feature generation
- Configurable number of workers
- Automatic load balancing

### Caching
- Feature result caching
- Configurable cache management
- Memory-efficient storage

### Lookback Optimization
- Data-driven optimization of lookback periods
- Multiple optimization methods (cross-validation, statistical analysis, etc.)
- Regime-aware optimization
- Performance and stability metrics

## 🔄 Migration from Existing Code

### Step 1: Identify Current Feature Generation
```python
# Old way
from src.feature_engineering.feature_generators import FeatureGenerators
generator = FeatureGenerators()
features = generator.batch_technical_indicators(df, indicator_configs)
```

### Step 2: Use Unified System
```python
# New way
from src.feature_generation import FeatureBank
bank = FeatureBank()
features = bank.generate_features(df, categories=['momentum', 'volume'])
```

### Step 3: Leverage Backwards Compatibility
```python
# Migration path
from src.feature_generation import LegacyFeatureAdapter
adapter = LegacyFeatureAdapter()
# Existing code continues to work while you migrate
```

## 📈 Best Practices

1. **Use Categories**: Organize features by category for better management
2. **Enable Optimization**: Use lookback optimization for better performance
3. **Leverage Matrix Operations**: Enable matrix operations for large datasets
4. **Use Parallel Processing**: Enable parallel processing for multiple features
5. **Cache Results**: Enable caching for repeated feature generation
6. **Validate Data**: Always validate input data before feature generation

## 🛠️ Extending the System

### Adding New Feature Categories

```python
from src.feature_generation.core import FeatureGenerator, FeatureConfig, FeatureCategory

class CustomFeatureGenerator(FeatureGenerator):
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Implement your feature generation logic
        return pd.Series(feature_data, index=data.index)

# Register the generator
from src.feature_generation import register_feature_generator
generator = CustomFeatureGenerator(config)
register_feature_generator(generator)
```

### Adding New Optimization Methods

```python
from src.feature_generation.optimization import LookbackOptimizer

class CustomOptimizer(LookbackOptimizer):
    def _custom_optimization_method(self, generator, data, target_column):
        # Implement your optimization logic
        return optimal_lookback
```

## 📚 API Reference

### Core Classes
- `FeatureBank`: Central feature registry and management
- `FeatureGenerator`: Base class for feature generators
- `FeatureConfig`: Configuration for feature generators
- `FeatureCategory`: Enumeration of feature categories

### Convenience Functions
- `generate_features_by_category()`: Generate features by category
- `generate_all_features()`: Generate all available features
- `get_feature_summary()`: Get summary of available features
- `validate_feature_data()`: Validate input data
- `export_feature_config()`: Export feature configuration

### Optimization
- `LookbackOptimizer`: Optimize feature lookback periods
- `FeatureOptimizationConfig`: Configuration for optimization
- `optimize_feature_lookbacks()`: Optimize multiple features

### Matrix Integration
- `MatrixFeatureProcessor`: Process features with matrix operations
- `enable_matrix_acceleration()`: Enable/disable matrix acceleration
- `get_matrix_processor()`: Get matrix processor instance

### Backwards Compatibility
- `LegacyFeatureAdapter`: Adapter for legacy code
- `migrate_legacy_features()`: Migrate legacy configurations
- `enable_legacy_compatibility()`: Enable/disable legacy compatibility

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backwards compatibility
5. Follow the established naming conventions

## 📄 License

This module is part of the larger trading system and follows the same licensing terms.