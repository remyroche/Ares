# Feature Generation Migration Guide

## 🚀 Overview

This guide helps developers migrate from the old duplicate method pattern to the new consolidated base class approach. The migration eliminates 100+ duplicate methods and provides a cleaner, more maintainable codebase.

## 📋 What Changed

### Before (Old Pattern)
```python
class MyFeatureGenerator(VectorizedFeatureGenerator):
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data

    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data
```

### After (New Pattern)
```python
class MyFeatureGenerator(VectorizedFeatureGenerator):
    # No need to implement these methods - they're inherited from base class!
    # The methods work exactly the same way
    pass
```

## 🔄 Migration Steps

### Step 1: Remove Duplicate Methods
Simply delete the duplicate `optimize_dataframe_processing` and `vectorized_rolling_operations` methods from your feature generator classes. These methods are now provided by the `VectorizedFeatureGenerator` base class.

### Step 2: Update Imports (if needed)
No import changes are required. The base class methods are automatically available.

### Step 3: Test Your Generators
Run your existing tests to ensure everything works correctly. The functionality is identical - only the implementation location has changed.

## ✅ Benefits of Migration

### Immediate Benefits
- **Cleaner Code**: 100+ duplicate methods removed
- **Easier Maintenance**: Single source of truth for common methods
- **Reduced Memory Usage**: Less duplicate code in memory
- **Faster Development**: No need to copy-paste common methods

### Long-term Benefits
- **Consistency**: All classes use the same optimized implementations
- **Bug Prevention**: No risk of methods diverging over time
- **Easier Testing**: Centralized testing for common functionality
- **Better Documentation**: Single place to document common patterns

## 🛠️ New Features Available

### Enhanced Base Class Methods
The base class methods now include:
- **VectorBT Optimization**: Automatic VectorBT usage for large datasets
- **Memory Optimization**: Automatic memory usage optimization
- **GPU Acceleration**: Support for GPU-accelerated operations
- **Comprehensive Logging**: Detailed performance and error logging
- **Fallback Support**: Graceful fallback when optimizations aren't available

### New Utility Mixins
- **OptimizationMixin**: Memory and data optimization utilities
- **RollingOperationsMixin**: Enhanced rolling operations with caching
- **VectorBTOptimizationMixin**: VectorBT-specific optimizations

### Factory Pattern
- **GeneratorFactory**: Programmatic generator creation
- **Batch Operations**: Create multiple generators efficiently
- **Template System**: Create generators from templates

## 📚 Usage Examples

### Basic Usage (No Changes Required)
```python
# Your existing code works exactly the same
generator = MyFeatureGenerator(config)
result = generator.generate(data)

# These methods are now inherited and work identically
optimized_data = generator.optimize_dataframe_processing(data)
rolling_features = generator.vectorized_rolling_operations(
    data, ['mean', 'std'], [20, 50]
)
```

### Using New Mixins
```python
class MyOptimizedGenerator(VectorizedFeatureGenerator, OptimizationMixin, RollingOperationsMixin):
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Use mixin methods
        optimized_data = self.optimize_dataframe_processing(data)
        rolling_mean = self.rolling_mean(data['close'], window=20)
        return rolling_mean
```

### Using Factory Pattern
```python
from src.feature_generation.core.generator_factory import get_generator_factory

factory = get_generator_factory()
generator = factory.create_generator('sma', window=20)
result = generator.generate(data)
```

## 🐛 Troubleshooting

### Common Issues

#### Issue: "Method not found" errors
**Solution**: Ensure your class inherits from `VectorizedFeatureGenerator` or includes the appropriate mixins.

#### Issue: Performance regression
**Solution**: The new implementation is more optimized. If you see performance issues, check that VectorBT is properly installed and configured.

#### Issue: Import errors
**Solution**: Ensure all required dependencies are installed:
```bash
pip install vectorbt pandas numpy scipy
```

### Getting Help

1. **Check the logs**: The new implementation provides detailed logging
2. **Review the documentation**: Comprehensive docstrings are now available
3. **Test with small datasets**: Start with small datasets to verify functionality
4. **Check performance stats**: Use `get_performance_stats()` to monitor performance

## 📊 Performance Monitoring

### Check Performance Statistics
```python
# Get performance statistics
stats = generator.get_performance_stats()
print(f"VectorBT operations: {stats['vectorbt_operations']}")
print(f"Memory optimizations: {stats['memory_optimizations']}")
print(f"Average computation time: {stats['average_computation_time']:.3f}s")
```

### Monitor Memory Usage
```python
# Check memory optimization stats
if hasattr(generator, 'get_optimization_stats'):
    opt_stats = generator.get_optimization_stats()
    print(f"Memory saved: {opt_stats['memory_saved_mb']:.2f}MB")
```

## 🔧 Advanced Configuration

### VectorBT Configuration
```python
# Configure VectorBT settings
config = FeatureConfig(
    name="my_feature",
    category=FeatureCategory.CUSTOM,
    required_columns=["close"],
    use_vectorbt=True,
    vectorbt_threshold=1000,
    enable_gpu=True,
    enable_parallel=True
)
```

### Memory Optimization
```python
# Configure memory optimization
generator = MyFeatureGenerator(config)
generator.enable_memory_optimization = True
generator.memory_threshold_mb = 200
generator.enable_data_compression = True
```

## 📈 Migration Checklist

- [ ] Remove duplicate `optimize_dataframe_processing` methods
- [ ] Remove duplicate `vectorized_rolling_operations` methods
- [ ] Ensure classes inherit from `VectorizedFeatureGenerator`
- [ ] Run existing tests to verify functionality
- [ ] Update any custom implementations that override these methods
- [ ] Test with your specific datasets
- [ ] Monitor performance and memory usage
- [ ] Update documentation if needed

## 🎯 Next Steps

After migration:
1. **Explore new features**: Try the new mixins and factory pattern
2. **Optimize performance**: Configure VectorBT and memory settings
3. **Monitor usage**: Use performance statistics to optimize further
4. **Contribute**: Help improve the base class implementations

## 📞 Support

If you encounter issues during migration:
1. Check this guide first
2. Review the comprehensive docstrings in the base classes
3. Test with small datasets to isolate issues
4. Check the logs for detailed error information

---

*This migration guide is part of the Feature Generation Duplicate Cleanup Plan. For more information, see `DUPLICATE_CLEANUP_PLAN.md`.*