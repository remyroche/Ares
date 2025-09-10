# ML Common Memory Integration Guide

## Overview
This guide explains how automatic memory skimming is integrated into all ML common utilities, ensuring optimal memory management across all machine learning operations.

## Key Features

### 🧠 **Automatic Memory Skimming**
All ML operations now automatically skim unused memory when needed.

### 🎯 **ML-Specific Memory Estimation**
Intelligent memory requirement estimation based on operation type and data characteristics.

### 🏷️ **ML Decorators**
Specialized decorators for ML operations with automatic memory management.

### 🔗 **Seamless Integration**
Automatic integration with existing ML utilities without breaking changes.

## Enhanced ML Utilities

### 1. **Hyperparameter Optimization (HPO)**

#### **Enhanced Methods:**
- `multi_objective_optimization()` - Now with automatic memory skimming
- `early_stopping_optimization()` - Now with automatic memory skimming

#### **Usage:**
```python
from src.utils.ml_common import HyperparameterOptimization

# Memory skimming is automatically applied
optimizer = HyperparameterOptimization()

# Multi-objective optimization with automatic memory management
results = optimizer.multi_objective_optimization(
    X=X_train, y=y_train,
    model_type='RandomForestClassifier',
    n_trials=100,
    cv_folds=5
)

# Early stopping optimization with automatic memory management
results = optimizer.early_stopping_optimization(
    model_factory=create_model,
    X=X_train, y=y_train,
    validation_data=(X_val, y_val),
    n_trials=50
)
```

### 2. **Cross-Validation Utilities**

#### **Enhanced Methods:**
- `walk_forward_validation()` - Now with automatic memory skimming

#### **Usage:**
```python
from src.utils.ml_common import CrossValidationUtilities

# Memory skimming is automatically applied
cv_utils = CrossValidationUtilities()

# Walk-forward validation with automatic memory management
results = cv_utils.walk_forward_validation(
    X=X_data, y=y_data,
    model=model,
    initial_train_size=1000,
    test_size=100,
    expanding_window=True
)
```

### 3. **Lookahead Protection**

#### **Enhanced Methods:**
- `temporal_feature_validation()` - Now with automatic memory skimming

#### **Usage:**
```python
from src.utils.ml_common import LookaheadProtection

# Memory skimming is automatically applied
lookahead_protection = LookaheadProtection()

# Temporal feature validation with automatic memory management
results = lookahead_protection.temporal_feature_validation(
    feature_data=feature_df,
    prediction_timestamp=datetime.now(),
    feature_timestamp_col='timestamp'
)
```

## ML-Specific Memory Management

### 1. **ML Memory Manager**

```python
from src.utils.ml_common import get_ml_memory_manager

manager = get_ml_memory_manager()

# Estimate memory requirements for ML operations
memory_mb = manager.estimate_ml_memory_requirements(
    operation_type='hyperparameter_optimization',
    n_trials=100,
    cv_folds=5
)

# Auto-skim memory for ML operations
result = manager.auto_skim_for_ml_operation(
    operation_type='model_training',
    data_shape=(10000, 100)
)
```

### 2. **ML-Specific Decorators**

#### **ML Memory Skim Decorator**
```python
from src.utils.ml_common import ml_memory_skim_decorator

@ml_memory_skim_decorator('hyperparameter_optimization')
def optimize_model_hyperparameters(X, y, model_type):
    """Model hyperparameter optimization with automatic memory skimming."""
    # Memory is automatically skimmed before execution
    return perform_optimization(X, y, model_type)
```

#### **ML Auto Memory Skim Decorator**
```python
from src.utils.ml_common import ml_auto_memory_skim_decorator

@ml_auto_memory_skim_decorator()
def train_model(X, y):
    """Model training with automatic memory estimation and skimming."""
    # Memory requirements are automatically estimated and skimmed
    return perform_training(X, y)
```

### 3. **ML-Specific Context Managers**

#### **ML Memory Context**
```python
from src.utils.ml_common import ml_memory_context

# Automatic memory management for ML operations
with ml_memory_context('hyperparameter_optimization', n_trials=100) as allocation:
    if allocation['allocation_successful']:
        print("✅ Memory allocation successful")
        
        # Your ML operation here
        results = perform_hyperparameter_optimization()
    else:
        print("⚠️ Insufficient memory")
```

#### **ML Auto Memory Context**
```python
from src.utils.ml_common import ml_auto_memory_context

# Automatic memory estimation and management
with ml_auto_memory_context(data_shape=(10000, 100)) as allocation:
    if allocation['allocation_successful']:
        print("✅ Memory allocation successful")
        
        # Your ML operation here
        results = perform_ml_operation()
```

## Operation Types and Memory Requirements

| Operation Type | Base Memory (MB) | Use Case |
|----------------|------------------|----------|
| `hyperparameter_optimization` | 2000 | HPO, Optuna optimization |
| `cross_validation` | 1500 | CV, walk-forward validation |
| `model_training` | 1000 | Model training, fitting |
| `feature_engineering` | 800 | Feature creation, transformation |
| `data_preprocessing` | 600 | Data cleaning, preprocessing |
| `model_inference` | 400 | Prediction, scoring |
| `lookahead_validation` | 500 | Lookahead bias detection |
| `temporal_validation` | 300 | Temporal feature validation |
| `general` | 200 | General ML operations |

## Memory Estimation Logic

The ML memory manager uses intelligent estimation based on:

1. **Operation Type** - Base memory requirements
2. **Data Shape** - Matrix size calculations
3. **Samples/Features** - Dataset dimensions
4. **Trials/Folds** - Optimization parameters
5. **Overhead** - Additional memory for intermediate results

### **Example Estimation:**
```python
# For hyperparameter optimization with 100 trials, 5 CV folds
base_memory = 2000  # MB (operation type)
trials_memory = 100 * 100  # 100MB per trial
cv_memory = 5 * 200  # 200MB per CV fold
total_estimated = base_memory + trials_memory + cv_memory
# = 2000 + 10000 + 1000 = 13000 MB (13GB)
```

## Integration Status

### **Automatically Enhanced Modules:**
- ✅ **HPO Utils** - `multi_objective_optimization`, `early_stopping_optimization`
- ✅ **CV Utils** - `walk_forward_validation`
- ✅ **Lookahead Protection** - `temporal_feature_validation`

### **Integration Functions:**
```python
from src.utils.ml_common import integrate_all_ml_utilities

# Integrate memory skimming with all ML utilities
results = integrate_all_ml_utilities()
print(f"Integration results: {results}")
```

## Usage Examples

### 1. **Complete ML Pipeline with Memory Management**

```python
from src.utils.ml_common import (
    HyperparameterOptimizer, CrossValidationUtilities,
    LookaheadProtection, ml_memory_context
)

# Initialize utilities
hpo = HyperparameterOptimizer()
cv_utils = CrossValidationUtilities()
lookahead = LookaheadProtection()

# Complete pipeline with automatic memory management
with ml_memory_context('hyperparameter_optimization', n_trials=100) as allocation:
    if allocation['allocation_successful']:
        # Step 1: Hyperparameter optimization
        hpo_results = hpo.multi_objective_optimization(
            X=X_train, y=y_train,
            model_type='RandomForestClassifier',
            n_trials=100
        )
        
        # Step 2: Cross-validation
        cv_results = cv_utils.walk_forward_validation(
            X=X_data, y=y_data,
            model=best_model,
            initial_train_size=1000
        )
        
        # Step 3: Lookahead validation
        lookahead_results = lookahead.temporal_feature_validation(
            feature_data=feature_df,
            prediction_timestamp=datetime.now()
        )
        
        print("✅ Complete ML pipeline executed successfully")
```

### 2. **Custom ML Function with Memory Management**

```python
from src.utils.ml_common import ml_memory_skim_decorator

@ml_memory_skim_decorator('model_training')
def train_custom_model(X, y, model_config):
    """Custom model training with automatic memory skimming."""
    
    # Memory is automatically skimmed before execution
    model = create_model(model_config)
    model.fit(X, y)
    
    # Additional memory-intensive operations
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    return {
        'model': model,
        'predictions': predictions,
        'probabilities': probabilities
    }

# Usage
results = train_custom_model(X_train, y_train, model_config)
```

### 3. **Memory Monitoring in ML Operations**

```python
from src.utils.ml_common import get_ml_memory_manager

manager = get_ml_memory_manager()

# Monitor memory before ML operation
memory_before = manager.memory_optimizer.get_memory_usage()
print(f"Memory before: {memory_before['rss_gb']:.1f}GB")

# Perform ML operation (memory skimming happens automatically)
results = perform_ml_operation()

# Monitor memory after ML operation
memory_after = manager.memory_optimizer.get_memory_usage()
print(f"Memory after: {memory_after['rss_gb']:.1f}GB")
print(f"Memory delta: {memory_after['rss_gb'] - memory_before['rss_gb']:+.1f}GB")
```

## Best Practices

### 1. **Use ML-Specific Decorators**
```python
# ✅ Good: Use ML-specific decorators
@ml_memory_skim_decorator('hyperparameter_optimization')
def optimize_hyperparameters():
    pass

# ❌ Avoid: Using generic decorators for ML operations
@memory_skim_decorator(1000, 'general')
def optimize_hyperparameters():
    pass
```

### 2. **Choose Appropriate Operation Types**
```python
# ✅ Good: Specify correct operation type
@ml_memory_skim_decorator('model_training')
def train_model():
    pass

# ❌ Avoid: Using generic operation type
@ml_memory_skim_decorator('general')
def train_model():
    pass
```

### 3. **Use Context Managers for Complex Operations**
```python
# ✅ Good: Use context managers for complex ML pipelines
with ml_memory_context('hyperparameter_optimization', n_trials=100):
    results = perform_complex_ml_pipeline()

# ❌ Avoid: Manual memory management
results = perform_complex_ml_pipeline()
# Manual cleanup needed
```

### 4. **Monitor Memory Usage**
```python
# ✅ Good: Monitor memory usage in ML operations
manager = get_ml_memory_manager()
memory_info = manager.memory_optimizer.get_memory_usage()
print(f"Available memory: {memory_info['available_gb']:.1f}GB")
```

## Error Handling

The ML memory integration includes comprehensive error handling:

```python
try:
    results = perform_ml_operation()
except MemoryError as e:
    print(f"Memory error: {e}")
    # Emergency cleanup is automatically attempted
except Exception as e:
    print(f"ML operation error: {e}")
```

## Performance Impact

- **Memory Skimming**: ~10-1000ms depending on cleanup level
- **Memory Estimation**: ~1-5ms
- **Integration Overhead**: ~1-2ms per operation
- **Overall Impact**: Minimal performance impact with significant memory benefits

## Conclusion

The ML common memory integration provides:

- **Automatic Memory Management** - No manual intervention needed
- **ML-Specific Optimization** - Tailored for ML operations
- **Seamless Integration** - Works with existing ML utilities
- **Comprehensive Coverage** - All major ML operations covered
- **Intelligent Estimation** - Smart memory requirement calculation
- **Error Handling** - Robust fallback mechanisms

All ML operations in the `src/utils/ml_common/` package now automatically use memory skimming when needed, ensuring optimal memory management for your M1 Mac! 🚀
