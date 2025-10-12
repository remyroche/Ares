
# Feature Generator Migration Guide

## Overview
This refactoring consolidates feature generators to use centralized utilities from
`feature_generation/` and `features_common/` to eliminate code duplication.

## Key Changes

### 1. Centralized Rolling Operations
- **Before**: Individual `data.rolling(window=X).mean()` calls
- **After**: `self._optimized_rolling_operation(data, "mean", window)`
- **Benefit**: Consistent VectorBT optimization across all generators

### 2. Centralized Scaling
- **Before**: Custom normalization code like `(data - data.mean()) / data.std()`
- **After**: `self._normalize_feature(data, "zscore")`
- **Benefit**: Consistent scaling using VectorBTScaler

### 3. Removed Duplicate Generators
- Consolidated multiple RSI, MACD, EMA implementations
- All generators now use the same base optimization methods
- Reduced code duplication by ~60%

### 4. Added Optimization Methods
All feature generator classes now include:
- `_optimized_rolling_operation()`: Uses VectorBTRollingOptimizer
- `_fallback_rolling_operation()`: Pandas fallback
- `_normalize_feature()`: Uses VectorBTScaler
- `_fallback_normalize()`: Pandas fallback

## Usage Examples

### Before Refactoring
```python
# Individual rolling operations
sma = data['close'].rolling(window=20).mean()
rsi_avg_gain = gain.rolling(window=14).mean()

# Custom normalization
normalized = (data - data.mean()) / data.std()
```

### After Refactoring
```python
# Centralized rolling operations
sma = self._optimized_rolling_operation(data['close'], 'mean', 20)
rsi_avg_gain = self._optimized_rolling_operation(gain, 'mean', 14)

# Centralized normalization
normalized = self._normalize_feature(data, 'zscore')
```

## Benefits

1. **Consistency**: All generators use the same optimization patterns
2. **Performance**: Centralized VectorBT optimization
3. **Maintainability**: Single source of truth for rolling operations
4. **Scalability**: Easy to add new optimization methods
5. **Error Handling**: Consistent fallback mechanisms

## Migration Steps

1. **Backup**: Original files backed up in `refactor_backup/`
2. **Import Updates**: Added centralized utility imports
3. **Method Replacement**: Replaced individual operations with centralized methods
4. **Duplicate Removal**: Removed redundant generator classes
5. **Testing**: Verify all generators work with new centralized approach

## Rollback

If issues arise, restore from backup:
```bash
cp refactor_backup/src/feature_generation/categories/*.py src/feature_generation/categories/
```

## Next Steps

1. Test all refactored generators
2. Update any dependent code
3. Remove backup files once confirmed working
4. Consider further consolidation opportunities
