# Gate Feature Step Refactoring Plan

## Current Issues & Opportunities

### 1. **Redundant Code Patterns**
- Manual correlation/variance calculations that exist in utilities
- Custom time-series CV that exists in ml_common
- Manual numeric column filtering that exists in common_utilities
- Custom error handling that exists in common_operations

### 2. **Missing VectorBT Integration**
- Not using VectorBT for efficient rolling calculations
- Missing hardware optimization decorators
- Not leveraging unified framework

### 3. **Inconsistent tprint Usage**
- Not using tprint_data_preview/data_format consistently
- Missing tprint calls for data operations

## Refactoring Strategy

### Phase 1: Import & Setup
- Add proper imports from utility modules
- Use VectorBT imports from src.vectorbt
- Add hardware optimization decorators

### Phase 2: Replace Manual Operations
- Replace correlation calculations with VectorBT
- Use common_utilities for DataFrame operations
- Use math_validation for safe operations
- Use ml_common for CV and validation

### Phase 3: Add Comprehensive tprint
- Add tprint_data_preview for all data loading/saving
- Add tprint_data_format for data compatibility
- Add tprint calls for all operations

### Phase 4: Hardware Optimization
- Add performance tracking decorators
- Use memory optimization
- Leverage unified framework

## Implementation Plan

### 1. **Imports & Dependencies**
```python
# VectorBT imports
from src.vectorbt import (
    vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
    rolling_sum, rolling_apply, VECTORBT_AVAILABLE
)

# Common utilities
from src.utils.common_utilities import (
    safe_dataframe_operation, get_numeric_columns, validate_dataframe
)
from src.utils.common_operations import (
    safe_operation, memory_managed, MemoryStrategy
)
from src.utils.math_validation import (
    safe_divide, safe_mean, safe_std, safe_correlation
)

# ML common utilities
from src.utils.ml_common.optimization import (
    create_consolidated_hpo, HPOConfig
)
from src.utils.ml_common.validation import (
    PurgedGroupTimeSeriesSplit, validate_no_leakage
)

# Feature selection
from src.feature_selection.core.framework import (
    get_feature_selection_framework, select_features
)

# Hardware optimization
from src.utils.hardware import (
    performance_tracked, memory_efficient, smart_cache
)
```

### 2. **Replace Manual Operations**

#### **Correlation Calculations**
```python
# Before: Manual correlation with edge cases
corr_matrix = numeric_features.corr().abs()
triu_indices = np.triu_indices_from(corr_matrix.values, k=1)

# After: Use VectorBT + common utilities
numeric_features = get_numeric_columns(features_df)
corr_matrix = safe_correlation_matrix(numeric_features)
max_corr, mean_corr = extract_correlation_stats(corr_matrix)
```

#### **Variance Calculations**
```python
# Before: Manual variance with edge cases
feature_variances = numeric_features.var()
valid_variances = feature_variances.dropna()

# After: Use VectorBT rolling operations
variances = rolling_var(numeric_features, window=1).iloc[-1]
min_var, mean_var, low_var_count = extract_variance_stats(variances)
```

#### **Time-Series CV**
```python
# Before: Manual TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3, test_size=0.2)

# After: Use ml_common purged CV
cv = PurgedGroupTimeSeriesSplit(
    n_splits=self.gate_learning_config.n_splits,
    test_size=self.gate_learning_config.test_size,
    gap=self.gate_learning_config.gap
)
```

### 3. **Add Comprehensive tprint**

#### **Data Loading/Saving**
```python
# Load data
features_df = artifact_manager.get_dataframe(...)
tprint_data_preview(features_df, "gate_input_features")
tprint_data_format(features_df, "gate_input_features")

# Save data
artifact_manager.save_dataframe(...)
tprint_data_preview(gate_features_df, "gate_output_features")
tprint_data_format(gate_features_df, "gate_output_features")
```

#### **Operations**
```python
# All operations should have tprint
tprint_step("🔍 Learning selective classification gate")
tprint_performance("Gate learning", execution_time)
tprint_success(f"✅ Generated {len(gate_features)} gate features")
```

### 4. **Hardware Optimization**

#### **Decorators**
```python
@performance_tracked
@memory_efficient
@smart_cache
async def _generate_data_driven_gate_features(self, ...):
    # Implementation
```

#### **Memory Management**
```python
@memory_managed(MemoryStrategy.MODERATE)
def _extract_interpretable_gate_features(self, ...):
    # Implementation
```

## Benefits

### 1. **Code Reduction**
- ~200 lines of manual code replaced with utility calls
- Consistent error handling across all operations
- Standardized data validation

### 2. **Performance**
- VectorBT for efficient rolling calculations
- Hardware optimization decorators
- Memory management

### 3. **Maintainability**
- Consistent patterns with rest of codebase
- Centralized utility functions
- Better error handling

### 4. **Debugging**
- Comprehensive tprint logging
- Data format validation
- Performance tracking

## Implementation Steps

1. **Update imports** - Add all utility imports
2. **Replace correlation/variance** - Use VectorBT + utilities
3. **Replace CV** - Use ml_common purged CV
4. **Add tprint** - Comprehensive logging
5. **Add decorators** - Hardware optimization
6. **Test** - Ensure functionality preserved
7. **Optimize** - Fine-tune performance

## Expected Results

- **50% reduction** in manual code
- **Better performance** through VectorBT
- **Consistent logging** with tprint
- **Hardware optimization** for production
- **Maintainable code** using established patterns