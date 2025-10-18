# Complementary Feature Optimization Implementation

## Overview

This implementation corrects the feature optimization approach for Tactician training by implementing:

1. **Complementary Scoring** instead of alignment scoring
2. **Regime-Invariant Optimization** instead of regime-specific optimization
3. **Multi-Objective Optimization** for Tactician targets

## Key Changes

### 1. Complementary Lookback Optimizer (`complementary_lookback_optimizer.py`)

**Core Principle**: Features should provide **complementary information** beyond what the Analyst already knows, not align with Analyst signals.

**Key Features**:
- `_calculate_complementary_score()`: Measures information gain beyond analyst
- `_calculate_regime_consistency()`: Ensures single lookback works across all regimes
- `_calculate_temporal_stability()`: Measures consistency over time
- `_calculate_overall_score()`: Weighted combination with complementary focus

**Scoring Formula**:
```python
complementary_score = target_correlation * complementary_bonus * (1 + info_gain)
overall_score = complementary_score * 0.5 + regime_consistency * 0.3 + temporal_stability * 0.2
```

### 2. Tactician Feature Optimizer (`tactician_feature_optimization.py`)

**Purpose**: Integration layer for Tactician training with complementary optimization.

**Key Features**:
- `optimize_for_tactician_training()`: Primary optimization for tactician targets
- `optimize_with_multi_target_objectives()`: Multi-target optimization with weights
- `get_optimization_report()`: Comprehensive analysis and recommendations

**Multi-Target Support**:
- `y_success`: Primary profit success (weight: 0.4)
- `r_H`: Realized returns (weight: 0.3)
- `time_to_hit`: Timing optimization (weight: 0.2)
- `direction`: Trade direction (weight: 0.1)

### 3. Feature Bank Integration (`feature_bank.py`)

**Updated Methods**:
- `_optimize_lookbacks()`: Now accepts analyst_signals and regime_column
- `generate_features()`: Passes analyst signals and regime info to optimizer

**Integration Points**:
```python
# Extract analyst signals and regime information from kwargs
analyst_signals = kwargs.get('analyst_signals', None)
regime_column = kwargs.get('regime_column', None)
generators_to_use = self._optimize_lookbacks(
    generators_to_use, data, target_column, analyst_signals, regime_column
)
```

## Usage Examples

### Basic Complementary Optimization

```python
from src.feature_generation.utils.optimization.tactician_feature_optimization import (
    optimize_tactician_features,
    get_tactician_optimization_config
)

# Configure optimization
config = get_tactician_optimization_config(
    analyst_alignment_penalty=0.7,  # High penalty for analyst alignment
    complementary_bonus=2.0,        # High bonus for complementary info
    regime_consistency_weight=0.4,   # High weight for regime consistency
)

# Optimize features
optimal_lookbacks = optimize_tactician_features(
    generators=feature_generators,
    data=market_data,
    tactician_targets={
        'y_success': tactician_profit_labels,
        'r_H': tactician_returns,
        'time_to_hit': tactician_timing
    },
    analyst_outputs={
        'analyst_oof_score': analyst_predictions
    },
    regime_assignments=regime_data
)
```

### Multi-Target Optimization

```python
from src.feature_generation.utils.optimization.tactician_feature_optimization import (
    TacticianFeatureOptimizer
)

optimizer = TacticianFeatureOptimizer(config)

# Multi-target optimization with custom weights
optimal_lookbacks = optimizer.optimize_with_multi_target_objectives(
    generators=generators,
    data=market_data,
    tactician_targets=tactician_targets,
    analyst_outputs=analyst_outputs,
    regime_assignments=regime_assignments,
    target_weights={
        'y_success': 0.4,      # Primary: profit success
        'r_H': 0.3,            # Secondary: realized returns
        'time_to_hit': 0.2,    # Tertiary: timing
        'direction': 0.1       # Quaternary: direction
    }
)
```

### Integration with Feature Bank

```python
from src.feature_generation.core.feature_bank import FeatureBank

# Initialize feature bank with complementary optimization
feature_bank = FeatureBank()

# Generate features with complementary optimization
features = feature_bank.generate_features(
    data=market_data,
    categories=['returns', 'momentum', 'volume', 'volatility'],
    lookback_optimization=True,
    target_column='y_success',
    analyst_signals=analyst_oof_score,  # For complementary scoring
    regime_column=regime_assignments    # For regime-invariant optimization
)
```

## Key Benefits

### 1. Complementary Information Focus
- **Before**: Features aligned with analyst signals (redundant)
- **After**: Features provide unique information beyond analyst
- **Result**: Tactician learns what analyst misses

### 2. Regime-Invariant Optimization
- **Before**: Different lookbacks per regime (overfitting)
- **After**: Single lookback that works across all regimes
- **Result**: Consistent performance regardless of market conditions

### 3. Multi-Objective Optimization
- **Before**: Single target optimization
- **After**: Weighted optimization across multiple tactician targets
- **Result**: Balanced optimization for profit, timing, and direction

### 4. Comprehensive Analysis
- **Complementary Analysis**: Identifies high/low complementary features
- **Regime Analysis**: Measures consistency across market regimes
- **Temporal Analysis**: Ensures stability over time
- **Recommendations**: Actionable insights for feature selection

## Configuration Options

### Complementary Optimization Config
```python
config = ComplementaryOptimizationConfig(
    # Basic parameters
    min_lookback=5,
    max_lookback=252,
    step_size=1,
    
    # Complementary scoring
    analyst_alignment_penalty=0.5,  # Penalty for analyst alignment
    complementary_bonus=1.5,         # Bonus for complementary info
    
    # Multi-objective weights
    regime_consistency_weight=0.3,   # Weight for regime consistency
    temporal_stability_weight=0.2,   # Weight for temporal stability
    
    # Performance
    parallel_processing=True,
    max_workers=4
)
```

### Tactician-Specific Config
```python
tactician_config = get_tactician_optimization_config(
    analyst_alignment_penalty=0.7,  # Higher penalty for tactician
    complementary_bonus=2.0,        # Higher bonus for tactician
    regime_consistency_weight=0.4,  # Higher weight for tactician
    temporal_stability_weight=0.3   # Higher weight for tactician
)
```

## Implementation Status

✅ **Completed**:
- Complementary lookback optimizer
- Tactician feature optimizer
- Feature bank integration
- Multi-target optimization
- Comprehensive reporting
- Usage examples

✅ **Key Features**:
- Complementary scoring algorithm
- Regime-invariant optimization
- Multi-objective optimization
- Comprehensive analysis and reporting
- Integration with existing feature bank

✅ **Testing**:
- No linting errors
- Proper imports and dependencies
- Backward compatibility maintained

## Next Steps

1. **Integration Testing**: Test with real tactician training pipeline
2. **Performance Benchmarking**: Compare with previous alignment-based approach
3. **Documentation**: Add to main feature generation documentation
4. **Training Pipeline Integration**: Update tactician training to use new optimizer

## Files Created/Modified

### New Files:
- `complementary_lookback_optimizer.py`: Core complementary optimization
- `tactician_feature_optimization.py`: Tactician-specific integration
- `tactician_complementary_optimization_example.py`: Usage examples
- `COMPLEMENTARY_OPTIMIZATION_IMPLEMENTATION.md`: This documentation

### Modified Files:
- `feature_bank.py`: Updated to use complementary optimizer
- `__init__.py`: Added new optimizer imports

This implementation ensures that Tactician features are optimized for complementary information beyond what the Analyst already knows, using regime-invariant optimization for consistent performance across all market conditions.
