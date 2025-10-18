# Tactician Mode Conditional Optimization

## Overview

The feature bank now supports conditional optimization based on the mode of operation:
- **Regular Mode**: Uses standard lookback optimization
- **Tactician Mode**: Uses complementary scoring with analyst signals and regime analysis

## ✅ **Implementation Details**

### 1. **Dual Optimizer Support**

The feature bank now initializes both optimizers:

```python
# Initialize lookback optimizers if enabled
self.lookback_optimizer = None      # Regular optimization
self.tactician_optimizer = None     # Tactician optimization

if self.config.enable_lookback_optimization:
    # Initialize regular lookback optimizer
    try:
        from ..utils.optimization import LookbackOptimizer
        self.lookback_optimizer = LookbackOptimizer()
        self.logger.info("✅ Regular lookback optimization enabled")
    except ImportError:
        self.logger.warning("⚠️ Regular lookback optimization not available")
    
    # Initialize complementary lookback optimizer for Tactician mode
    try:
        from ..utils.optimization.complementary_lookback_optimizer import ComplementaryLookbackOptimizer
        self.tactician_optimizer = ComplementaryLookbackOptimizer()
        self.logger.info("✅ Tactician lookback optimization enabled")
    except ImportError:
        self.logger.warning("⚠️ Tactician lookback optimization not available")
```

### 2. **Conditional Optimization Logic**

The `_optimize_lookbacks` method now supports mode selection:

```python
def _optimize_lookbacks(self,
                       generators: List[FeatureGenerator],
                       data: pd.DataFrame,
                       target_column: str,
                       analyst_signals: Optional[pd.Series] = None,
                       regime_series: Optional[pd.Series] = None,
                       tactician_mode: bool = False) -> List[FeatureGenerator]:
    """
    Optimize lookback periods for generators.
    
    Args:
        tactician_mode: Whether to use Tactician optimization (complementary scoring)
    """
    # Select appropriate optimizer based on mode
    if tactician_mode and self.tactician_optimizer:
        optimizer = self.tactician_optimizer
        self.logger.info("🔧 Optimizing lookback periods using Tactician complementary scoring...")
    elif self.lookback_optimizer:
        optimizer = self.lookback_optimizer
        self.logger.info("🔧 Optimizing lookback periods using regular scoring...")
    else:
        return generators
```

### 3. **Mode Detection in generate_features**

The main feature generation method automatically detects the mode:

```python
# Optimize lookbacks if requested
if lookback_optimization and target_column and (self.lookback_optimizer or self.tactician_optimizer):
    # Check if this is Tactician mode based on kwargs
    tactician_mode = kwargs.get('tactician_mode', False)
    analyst_signals = kwargs.get('analyst_signals', None)
    regime_series = kwargs.get('regime_series', None)
    
    generators_to_use = self._optimize_lookbacks(
        generators_to_use, data, target_column, analyst_signals, regime_series, tactician_mode
    )
```

## ✅ **Usage Patterns**

### 1. **Regular Mode (Default)**

```python
# Standard feature generation - uses regular optimization
features = feature_bank.generate_features(
    data=market_data,
    categories=['returns', 'momentum', 'volume'],
    lookback_optimization=True,
    target_column='close'  # Simple target
)
# Result: Uses self.lookback_optimizer with standard correlation-based optimization
```

### 2. **Tactician Mode (Explicit)**

```python
# Tactician feature generation - uses complementary optimization
features = feature_bank.generate_features(
    data=market_data,
    categories=['returns', 'momentum', 'volume'],
    lookback_optimization=True,
    target_column='y_success',
    tactician_mode=True,  # Enable Tactician mode
    analyst_signals=analyst_oof_score,
    regime_series=regime_assignments
)
# Result: Uses self.tactician_optimizer with complementary scoring
```

### 3. **TacticianFeatureOptimizer Integration**

```python
# Using TacticianFeatureOptimizer for advanced integration
tactician_optimizer = TacticianFeatureOptimizer(config)

# Generate features with Tactician mode
features = tactician_optimizer.generate_tactician_features(
    feature_bank=feature_bank,
    data=market_data,
    tactician_targets=tactician_targets,
    analyst_outputs=analyst_outputs,
    regime_assignments=regime_assignments,
    categories=['returns', 'momentum', 'volume']
)
# Result: Automatically calls feature bank with tactician_mode=True
```

### 4. **Convenience Function**

```python
# Using convenience function
features = generate_tactician_features_with_optimization(
    feature_bank=feature_bank,
    data=market_data,
    tactician_targets=tactician_targets,
    analyst_outputs=analyst_outputs,
    regime_assignments=regime_assignments,
    categories=['returns', 'momentum', 'volume']
)
# Result: Simplified interface for Tactician mode
```

## ✅ **Mode Selection Logic**

### 1. **Automatic Detection**

The feature bank automatically detects the mode based on kwargs:

```python
tactician_mode = kwargs.get('tactician_mode', False)
```

- **Default**: `tactician_mode=False` → Regular mode
- **Explicit**: `tactician_mode=True` → Tactician mode

### 2. **Optimizer Selection**

```python
if tactician_mode and self.tactician_optimizer:
    # Use Tactician optimization (complementary scoring)
    optimizer = self.tactician_optimizer
elif self.lookback_optimizer:
    # Use regular optimization (standard correlation)
    optimizer = self.lookback_optimizer
else:
    # No optimization available
    return generators
```

### 3. **Parameter Requirements**

**Regular Mode**:
- `target_column`: Required for optimization
- No additional parameters needed

**Tactician Mode**:
- `target_column`: Required for optimization
- `analyst_signals`: Optional but recommended for complementary scoring
- `regime_series`: Optional but recommended for regime-invariant optimization

## ✅ **Integration Points**

### 1. **TacticianFeatureOptimizer**

The `TacticianFeatureOptimizer` automatically calls the feature bank with Tactician mode:

```python
def generate_tactician_features(self, feature_bank, data, tactician_targets, ...):
    # Call feature bank with Tactician mode enabled
    tactician_kwargs = {
        'tactician_mode': True,
        'analyst_signals': analyst_signals,
        'regime_series': regime_assignments,
        'lookback_optimization': True,
        'target_column': 'y_success',
        **kwargs
    }
    
    features_df = feature_bank.generate_features(
        data=data_with_target,
        categories=categories,
        features=features,
        **tactician_kwargs
    )
```

### 2. **Convenience Functions**

The convenience functions provide simplified interfaces:

```python
def generate_tactician_features_with_optimization(feature_bank, data, tactician_targets, ...):
    optimizer = TacticianFeatureOptimizer(config)
    return optimizer.generate_tactician_features(
        feature_bank, data, tactician_targets, analyst_outputs, regime_assignments,
        categories, features, **kwargs
    )
```

## ✅ **Benefits**

### 1. **Backward Compatibility**
- Existing code continues to work unchanged
- Regular mode is the default behavior
- No breaking changes to existing APIs

### 2. **Conditional Optimization**
- Automatic mode detection based on parameters
- Appropriate optimizer selection
- No manual configuration required

### 3. **Advanced Capabilities**
- Tactician mode provides complementary scoring
- Regime-invariant optimization
- Analyst signal integration
- Advanced optimization methods (Bayesian TPE, vectorized operations)

### 4. **Seamless Integration**
- TacticianFeatureOptimizer handles mode switching
- Feature bank automatically uses correct optimizer
- Transparent to end users

## ✅ **Error Handling**

### 1. **Missing Optimizers**
```python
if tactician_mode and self.tactician_optimizer:
    # Use Tactician optimization
elif self.lookback_optimizer:
    # Use regular optimization
else:
    # No optimization available - return generators unchanged
    return generators
```

### 2. **Fallback Behavior**
- If Tactician optimizer not available → falls back to regular optimizer
- If no optimizers available → returns generators unchanged
- Graceful degradation with appropriate logging

### 3. **Parameter Validation**
- Tactician mode works without analyst_signals (uses basic correlation)
- Regime analysis works without regime_series (assumes single regime)
- Robust error handling with informative messages

## ✅ **Files Updated**

### 1. **feature_bank.py**
- Added dual optimizer initialization
- Updated `_optimize_lookbacks` with mode selection
- Enhanced `generate_features` with mode detection

### 2. **tactician_feature_optimization.py**
- Added `generate_tactician_features` method
- Added convenience function for feature generation
- Enhanced integration with feature bank

### 3. **Examples**
- Created comprehensive usage examples
- Demonstrated mode selection logic
- Showed integration patterns

The feature bank now provides conditional optimization that automatically selects the appropriate optimizer based on the mode, ensuring backward compatibility while providing advanced capabilities for Tactician training.
