# Mixins and GeneratorFactory - Complete Explanation

## 🛠️ Mixins - What They Are and How They're Wired

### ✅ **All Mixins Are Properly Wired**

The three utility mixins are fully integrated and available for import:

```python
from src.feature_generation import (
    OptimizationMixin,
    RollingOperationsMixin, 
    VectorBTOptimizationMixin
)
```

### 1. **OptimizationMixin** - Memory & Data Optimization
**Purpose:** Provides memory optimization, data compression, and chunked processing capabilities.

**Key Features:**
- **Memory Optimization:** Automatic data type optimization (int64→int32, float64→float32)
- **Data Compression:** Categorical data compression for memory efficiency
- **Chunked Processing:** Process large datasets in chunks to avoid memory issues
- **Performance Tracking:** Monitor memory savings and optimization statistics

**Usage:**
```python
class MyGenerator(VectorizedFeatureGenerator, OptimizationMixin):
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Automatically optimize memory usage
        optimized_data = self.optimize_dataframe_processing(data)
        
        # Process in chunks for large datasets
        result = self.chunked_processing(optimized_data, my_processing_function)
        
        return result
```

**Methods Provided:**
- `optimize_dataframe_processing(data)` - Optimize DataFrame for memory efficiency
- `chunked_processing(data, func, chunk_size)` - Process data in chunks
- `get_optimization_stats()` - Get memory optimization statistics
- `reset_optimization_stats()` - Reset performance tracking

### 2. **RollingOperationsMixin** - Enhanced Rolling Operations
**Purpose:** Provides optimized rolling operations with VectorBT integration and caching.

**Key Features:**
- **VectorBT Integration:** Automatic VectorBT usage for large datasets
- **Operation Caching:** Cache results to avoid recomputation
- **Batch Operations:** Perform multiple rolling operations efficiently
- **Performance Tracking:** Monitor VectorBT vs pandas usage

**Usage:**
```python
class MyGenerator(VectorizedFeatureGenerator, RollingOperationsMixin):
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Use optimized rolling operations
        sma = self.rolling_mean(data['close'], window=20)
        std = self.rolling_std(data['close'], window=20)
        
        # Batch operations for efficiency
        operations = [
            {'column': 'close', 'operation': 'mean', 'window': 20, 'name': 'sma_20'},
            {'column': 'close', 'operation': 'std', 'window': 20, 'name': 'std_20'}
        ]
        results = self.batch_rolling_operations(data, operations)
        
        return sma
```

**Methods Provided:**
- `rolling_mean(data, window)` - Optimized rolling mean
- `rolling_std(data, window)` - Optimized rolling standard deviation
- `rolling_var(data, window)` - Optimized rolling variance
- `rolling_min(data, window)` - Optimized rolling minimum
- `rolling_max(data, window)` - Optimized rolling maximum
- `rolling_sum(data, window)` - Optimized rolling sum
- `rolling_corr(data, other, window)` - Optimized rolling correlation
- `rolling_cov(data, other, window)` - Optimized rolling covariance
- `batch_rolling_operations(data, operations)` - Batch rolling operations
- `get_rolling_stats()` - Get performance statistics

### 3. **VectorBTOptimizationMixin** - VectorBT-Specific Optimizations
**Purpose:** Provides VectorBT-specific optimizations and GPU acceleration.

**Key Features:**
- **VectorBT Integration:** Native VectorBT operations for maximum performance
- **GPU Acceleration:** Automatic GPU usage when available
- **Advanced Caching:** VectorBT-specific caching with TTL
- **Performance Monitoring:** Track VectorBT vs pandas performance

**Usage:**
```python
class MyGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Use VectorBT-optimized operations
        result = self._vectorbt_rolling_operation(data['close'], 'mean', 20)
        
        # Use VectorBT technical indicators
        rsi = self._calculate_rsi_vectorbt(data, window=14)
        macd = self._calculate_macd_vectorbt(data, fast=12, slow=26)
        
        return result
```

**Methods Provided:**
- `_vectorbt_rolling_operation(data, operation, window)` - VectorBT rolling operations
- `_vectorbt_technical_indicator(data, indicator, **kwargs)` - VectorBT technical indicators
- `_calculate_rsi_vectorbt(data, window)` - VectorBT RSI calculation
- `_calculate_macd_vectorbt(data, fast, slow, signal)` - VectorBT MACD calculation
- `_calculate_bollinger_bands_vectorbt(data, window, std_dev)` - VectorBT Bollinger Bands
- `_calculate_atr_vectorbt(data, window)` - VectorBT ATR calculation
- `get_performance_stats()` - Get VectorBT performance statistics

## 🏭 GeneratorFactory - What It Is and How It Works

### **What is GeneratorFactory?**

The **GeneratorFactory** is a **Factory Pattern** implementation that provides a programmatic way to create feature generators without having to manually instantiate classes. It's like a "generator creation factory" that can build different types of generators based on specifications.

### **Why Use GeneratorFactory?**

1. **Reduces Boilerplate Code** - No need to manually create generator classes
2. **Dynamic Generator Creation** - Create generators at runtime based on configuration
3. **Consistent Interface** - All generators created through the same interface
4. **Easy Registration** - Register custom generators for reuse
5. **Batch Creation** - Create multiple generators efficiently

### **How GeneratorFactory Works**

```python
from src.feature_generation import GeneratorFactory, FeatureCategory

# Create factory instance
factory = GeneratorFactory()

# Method 1: Create a basic vectorized generator
generator = factory.create_vectorized_generator(
    name="my_sma",
    category=FeatureCategory.CUSTOM,
    required_columns=["close"],
    window=20
)

# Method 2: Create an optimized generator with all mixins
optimized_generator = factory.create_optimized_generator(
    name="optimized_sma",
    category=FeatureCategory.CUSTOM,
    required_columns=["close"]
)

# Method 3: Create from template
template_generator = factory.create_generator_from_template(
    template_name="sma_template",
    name="new_sma",
    window=30
)

# Method 4: Batch creation
generator_specs = [
    {'name': 'sma_20', 'category': 'CUSTOM', 'required_columns': ['close'], 'window': 20},
    {'name': 'sma_50', 'category': 'CUSTOM', 'required_columns': ['close'], 'window': 50},
    {'name': 'rsi_14', 'category': 'CUSTOM', 'required_columns': ['close'], 'window': 14}
]
generators = factory.create_batch_generators(generator_specs)
```

### **GeneratorFactory Methods**

#### 1. **`create_generator(name, **kwargs)`**
Create a generator by registered name.

```python
# Register a generator first
factory.register_generator('my_sma', MySMAGenerator)

# Then create it
generator = factory.create_generator('my_sma', window=20)
```

#### 2. **`create_vectorized_generator(name, category, required_columns, **kwargs)`**
Create a basic vectorized generator.

```python
generator = factory.create_vectorized_generator(
    name="sma_20",
    category=FeatureCategory.CUSTOM,
    required_columns=["close"],
    window=20
)
```

#### 3. **`create_optimized_generator(name, category, required_columns, **kwargs)`**
Create a generator with all optimization mixins.

```python
generator = factory.create_optimized_generator(
    name="optimized_sma",
    category=FeatureCategory.CUSTOM,
    required_columns=["close"]
)
# This generator automatically includes:
# - VectorizedFeatureGenerator (base class)
# - OptimizationMixin (memory optimization)
# - RollingOperationsMixin (enhanced rolling operations)
# - VectorBTOptimizationMixin (VectorBT optimization)
```

#### 4. **`create_custom_generator(name, generator_class, config, **kwargs)`**
Create a generator from a custom class.

```python
class MyCustomGenerator(VectorizedFeatureGenerator):
    def _generate_feature(self, data, **kwargs):
        return data['close'].rolling(20).mean()

config = FeatureConfig(
    name="custom_sma",
    category=FeatureCategory.CUSTOM,
    required_columns=["close"]
)

generator = factory.create_custom_generator(
    name="custom_sma",
    generator_class=MyCustomGenerator,
    config=config
)
```

#### 5. **`create_batch_generators(generator_specs)`**
Create multiple generators in batch.

```python
specs = [
    {
        'name': 'sma_20',
        'category': FeatureCategory.CUSTOM,
        'required_columns': ['close'],
        'window': 20
    },
    {
        'name': 'sma_50', 
        'category': FeatureCategory.CUSTOM,
        'required_columns': ['close'],
        'window': 50
    }
]

generators = factory.create_batch_generators(specs)
```

#### 6. **`create_generator_from_template(template_name, name, **kwargs)`**
Create a generator from a template.

```python
generator = factory.create_generator_from_template(
    template_name="sma_template",
    name="new_sma",
    window=30
)
```

### **Convenience Functions**

```python
from src.feature_generation import create_generator, get_generator_factory

# Get global factory
factory = get_generator_factory()

# Create generator using convenience function
generator = create_generator('my_sma', window=20)
```

## 🎯 **Complete Usage Example**

```python
from src.feature_generation import (
    VectorizedFeatureGenerator,
    OptimizationMixin,
    RollingOperationsMixin,
    VectorBTOptimizationMixin,
    GeneratorFactory,
    FeatureCategory
)
import pandas as pd

# Create sample data
data = pd.DataFrame({
    'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
})

# Method 1: Manual class creation with mixins
class MyEnhancedGenerator(VectorizedFeatureGenerator, OptimizationMixin, RollingOperationsMixin):
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Use optimization mixin
        optimized_data = self.optimize_dataframe_processing(data)
        
        # Use rolling operations mixin
        sma = self.rolling_mean(optimized_data['close'], window=5)
        
        return sma

# Method 2: Using GeneratorFactory
factory = GeneratorFactory()

# Create optimized generator with all mixins
generator = factory.create_optimized_generator(
    name="factory_sma",
    category=FeatureCategory.CUSTOM,
    required_columns=["close"]
)

# Both methods work the same way
manual_generator = MyEnhancedGenerator(FeatureConfig(...))
factory_generator = generator

# Generate features
result1 = manual_generator.generate(data)
result2 = factory_generator.generate(data)
```

## ✅ **Summary**

### **Mixins Are Fully Wired:**
- ✅ `OptimizationMixin` - Memory optimization and chunked processing
- ✅ `RollingOperationsMixin` - Enhanced rolling operations with VectorBT
- ✅ `VectorBTOptimizationMixin` - VectorBT-specific optimizations

### **GeneratorFactory Provides:**
- 🏭 **Factory Pattern** - Programmatic generator creation
- 🚀 **Multiple Creation Methods** - Various ways to create generators
- 📦 **Batch Operations** - Create multiple generators at once
- 🔧 **Template System** - Create generators from templates
- ⚡ **Optimized Generators** - Automatic inclusion of all mixins

All features are ready to use and fully integrated into the system!