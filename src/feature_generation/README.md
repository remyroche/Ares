# feature_generation/ - General Purpose Feature Generation

## Purpose

This is the **general-purpose feature generation system** for exploration, backtesting, and most models in the Ares trading platform.

## Key Features

- ✅ **100+ Feature Generators** - Comprehensive library of technical indicators and features
- ✅ **Flexible & Dynamic** - Create and register features at runtime with customizable parameters
- ✅ **Category-Based Organization** - Momentum, volatility, volume, oscillators, and more
- ✅ **Performance Optimized** - Matrix operations, GPU acceleration, caching support
- ✅ **Extensible** - Easy to add new feature generators by inheriting from base classes

## When to Use

Use `feature_generation/` for:
- 🎯 Exploratory feature engineering
- 🎯 Backtesting with custom indicators
- 🎯 Analyst model features
- 🎯 Tactician model features
- 🎯 General trading strategies
- 🎯 Feature discovery and research

**DO NOT use for:** End-to-end roadmap training (use `feature_engineering_roadmap/` instead)

## Quick Start

### Basic Usage

```python
from src.feature_generation.categories.momentum import RSIGenerator
from src.feature_generation.categories.volatility import ATRGenerator
from src.feature_generation.core.feature_registry import FeatureRegistry
import pandas as pd

# Load your data
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Create generators
rsi_gen = RSIGenerator(period=14)
atr_gen = ATRGenerator(period=20)

# Generate features
rsi_result = rsi_gen.generate(data)
atr_result = atr_gen.generate(data)

print(f"RSI: {rsi_result.data}")
print(f"ATR: {atr_result.data}")
print(f"Computation time: {rsi_result.computation_time:.3f}s")
```

### Using the Registry

```python
from src.feature_generation.core.feature_registry import FeatureRegistry
from src.feature_generation.categories.momentum import MomentumGenerator

# Create registry
registry = FeatureRegistry()

# Register generators
registry.register(MomentumGenerator(period=5))
registry.register(MomentumGenerator(period=10))
registry.register(MomentumGenerator(period=20))

# Generate all features
results = {}
for name in registry.list_names():
    generator = registry.get_by_name(name)
    result = generator.generate(data)
    results[name] = result.data

features_df = pd.DataFrame(results)
```

## Directory Structure

```
feature_generation/
├── core/                       # Core framework
│   ├── feature_generator.py   # Base classes (FeatureGenerator, VectorizedFeatureGenerator)
│   ├── feature_registry.py    # Dynamic registry for managing generators
│   ├── factory.py             # Feature factory pattern
│   ├── feature_bank.py        # Feature storage and caching
│   └── feature_cache.py       # Caching mechanisms
│
├── categories/                 # Feature categories (35+ files)
│   ├── momentum.py            # Momentum indicators (RSI, MACD, ROC, etc.)
│   ├── volatility.py          # Volatility features (ATR, Bollinger Bands, etc.)
│   ├── volume.py              # Volume indicators (OBV, VWAP, etc.)
│   ├── oscillator.py          # Oscillators (Stochastic, Williams %R, etc.)
│   ├── trend.py               # Trend indicators (SMA, EMA, ADX, etc.)
│   ├── interaction.py         # Feature interactions
│   ├── support_resistance.py  # Support/resistance levels
│   └── [30+ more categories]
│
├── base_calculations/          # Base calculation types
│   ├── base_calculator.py     # BaseCalculator interface
│   └── [calculation types]
│
├── utils/                      # Utilities and optimizations
│   ├── optimization/          # Performance optimization
│   ├── vectorization_optimizer.py
│   └── [40+ utility files]
│
├── examples/                   # Usage examples
│   ├── usage_example.py
│   └── enhanced_usage_examples.py
│
└── README.md                   # This file
```

## Available Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **momentum** | Momentum indicators | RSI, MACD, ROC, Momentum |
| **volatility** | Volatility measures | ATR, Bollinger Bands, Standard Deviation |
| **volume** | Volume-based features | OBV, VWAP, Volume MA |
| **oscillator** | Oscillator indicators | Stochastic, Williams %R, CCI |
| **trend** | Trend-following indicators | SMA, EMA, ADX, Supertrend |
| **returns** | Return calculations | Log returns, Simple returns |
| **normalization** | Normalization techniques | Z-score, Min-max, Robust scaling |
| **interaction** | Feature interactions | Momentum × Volume, Volatility × Price |
| **support_resistance** | S/R levels | Price levels, Break detection |
| **candlestick_pattern** | Candlestick patterns | Doji, Hammer, Engulfing |
| **microstructure** | Market microstructure | Bid-ask spread, Order flow |
| **entropy** | Entropy-based features | Sample entropy, Approximate entropy |
| **regime** | Regime detection | HMM regimes, Volatility regimes |

## Creating Custom Features

### Step 1: Inherit from FeatureGenerator

```python
from src.feature_generation.core.feature_generator import (
    FeatureGenerator, FeatureConfig, FeatureCategory
)
import pandas as pd

class MyCustomGenerator(FeatureGenerator):
    def __init__(self, period: int = 14):
        config = FeatureConfig(
            name=f"my_custom_feature_{period}",
            category=FeatureCategory.CUSTOM,
            description=f"My custom feature with period {period}",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period}
        )
        super().__init__(config)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Implement your feature calculation here."""
        return data['close'].rolling(window=self.period).mean()
```

### Step 2: Use Your Custom Generator

```python
# Create and use
custom_gen = MyCustomGenerator(period=20)
result = custom_gen.generate(data)

# Register for reuse
registry = FeatureRegistry()
registry.register(custom_gen)
```

## Base Classes

### FeatureGenerator
The main base class for all feature generators.

**Key methods:**
- `generate(data, **kwargs)` - Generate feature with error handling
- `_generate_feature(data, **kwargs)` - Override this to implement your feature
- `get_config()` - Get feature configuration
- `get_performance_stats()` - Get performance statistics

### VectorizedFeatureGenerator
Optimized base class for vectorized operations.

**Additional features:**
- Matrix operations support
- Hardware acceleration (GPU)
- Vectorized rolling operations

### FeatureConfig
Configuration dataclass for features.

**Fields:**
- `name` - Feature name
- `category` - FeatureCategory enum
- `description` - Human-readable description
- `required_columns` - Required DataFrame columns
- `optional_columns` - Optional columns
- `default_lookback` - Default lookback period
- `parameters` - Additional parameters

## Performance Optimization

### Caching
```python
from src.feature_generation.core.feature_cache import FeatureCache
from src.feature_generation.core.feature_bank import FeatureBank

# Create cache
cache = FeatureCache()

# Use with feature bank
bank = FeatureBank(cache=cache)
bank.register_generator(RSIGenerator(period=14))

# Generate with caching (faster on subsequent calls)
features = bank.generate_all(data, use_cache=True)
```

### Matrix Operations
```python
from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator

class OptimizedGenerator(VectorizedFeatureGenerator):
    def __init__(self, config):
        super().__init__(
            config,
            enable_matrix_ops=True,  # Enable matrix operations
            enable_vectorization_optimization=True  # Enable vectorization
        )
    
    def _generate_feature(self, data, **kwargs):
        # Optimize DataFrame processing
        data = self.optimize_dataframe_processing(data)
        
        # Use vectorized rolling operations
        result = self.vectorized_rolling_operations(
            data,
            operations=['mean', 'std'],
            windows=[10, 20],
            columns=['close', 'volume']
        )
        return result
```

## Integration with Other Systems

### Shared Base Classes
This system uses shared base classes from `features_common/`:

```python
from src.features_common.transforms.base_scaler import BaseScaler
from src.features_common.optimization.cv_base import BaseCVSplitter
from src.features_common.registry.base_registry import BaseFeatureRegistry
```

### Relationship to feature_engineering_roadmap
- **feature_generation/** - General purpose, flexible (this system)
- **feature_engineering_roadmap/** - Locked features for end-to-end roadmap only

**Rule:** Use feature_generation for everything except end-to-end roadmap training.

## Examples

See `examples/` directory for detailed examples:
- `usage_example.py` - Basic usage
- `enhanced_usage_examples.py` - Advanced patterns

## Testing

```python
import pytest
from src.feature_generation.categories.momentum import RSIGenerator

def test_rsi_generation():
    # Create test data
    data = pd.DataFrame({
        'close': [100, 102, 101, 103, 105, 104, 106]
    })
    
    # Generate RSI
    gen = RSIGenerator(period=3)
    result = gen.generate(data)
    
    # Verify
    assert result.success
    assert not result.data.empty
    assert result.computation_time > 0
```

## Common Patterns

### Pattern 1: Multi-Timeframe Features
```python
from src.feature_generation.categories.cross_timeframe import CrossTimeframeGenerator

gen = CrossTimeframeGenerator(
    base_feature='close',
    timeframes=['5m', '15m', '1h']
)
result = gen.generate(data)
```

### Pattern 2: Feature Interactions
```python
from src.feature_generation.categories.interaction import MomentumVolumeGenerator

gen = MomentumVolumeGenerator(period=14)
interaction = gen.generate(data)  # Momentum × Volume interaction
```

### Pattern 3: Batch Generation
```python
from src.feature_generation.core.factory import FeatureFactory

factory = FeatureFactory()
generators = [
    RSIGenerator(period=14),
    ATRGenerator(period=20),
    MomentumGenerator(period=10)
]

all_features = factory.generate_all(data, generators)
```

## Troubleshooting

### Issue: "Missing required columns"
```python
# Check required columns
gen = RSIGenerator(period=14)
print(gen.config.required_columns)  # ['close']

# Ensure your data has these columns
assert 'close' in data.columns
```

### Issue: "Insufficient data"
```python
# Check minimum lookback
gen = RSIGenerator(period=14)
print(gen.config.min_lookback)  # 14

# Ensure enough data points
assert len(data) >= gen.config.min_lookback
```

### Issue: Performance issues
```python
# Use vectorized generators
from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator

# Enable caching
cache = FeatureCache()
bank = FeatureBank(cache=cache)
features = bank.generate_all(data, use_cache=True)
```

## Contributing

When adding new features:
1. Inherit from `FeatureGenerator` or `VectorizedFeatureGenerator`
2. Place in appropriate category file
3. Add tests
4. Update documentation
5. Follow existing naming conventions

## Related Documentation

- [Feature Systems Guide](../FEATURE_SYSTEMS_GUIDE.md) - Overview of both systems
- [features_common/](../features_common/) - Shared base classes
- [feature_engineering_roadmap/](../feature_engineering_roadmap/) - Roadmap-specific features

---

Last updated: 2025-10-08  
Part of Strategy C Implementation
