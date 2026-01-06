# Layer 4 Unified Feature Generation

This directory contains the consolidated Layer 4 feature generation system for position sizing and risk management.

## Overview

The Layer 4 system has been refactored to consolidate all feature generation into a unified, configurable system. This addresses the previous issue where features were scattered across multiple locations.

## Architecture

### Core Components

1. **feature_registry.py** - Centralized feature patterns and validation
2. **feature_generator.py** - Unified Layer4FeatureGenerator class
3. **__init__.py** - Module exports and imports

### Feature Categories

- **Performance**: PSR, precision, entropy, probability products
- **Regime**: Volatility, trend, SADF, market state
- **Market**: Relative strength, VWAP, drawdown
- **Technical**: ADX, choppiness, variance ratio, efficiency
- **Structural**: Break scores, change points
- **Model**: Disagreement, ensemble features
- **Time**: Temporal patterns, session effects
- **Contextual**: Residuals, harmonization features

### Integration Points

The unified system integrates with:
- `label_based_layer_4.py` - MetaLearnerFeatures and regime features
- `layer4_extratrees_pnl.py` - ExtraTrees model training
- External feature generators (ensemble disagreement, contextual residuals, De Prado)

## Usage

### Basic Usage

```python
from src.training.steps.labeling.layer4 import Layer4FeatureGenerator

# Initialize generator
generator = Layer4FeatureGenerator(
    window=50,
    config={
        'enable_performance': True,
        'enable_regime': True,
        'enable_market': True,
        'enable_contextual': True
    }
)

# Generate all features
features_df = generator.generate_all_features(
    df=market_data,
    layer3_predictions=layer3_preds,
    target_col='realized_return',
    prob_col='meta_prob'
)
```

### Feature Selection

```python
from src.training.steps.labeling.layer4 import get_layer4_features_from_dataframe

# Get available features
available_features = get_layer4_features_from_dataframe(df)

# Validate feature patterns
from src.training.steps.labeling.layer4 import validate_layer4_features
validation = validate_layer4_features(df)
```

## Configuration

The generator supports fine-grained control over feature categories:

```python
config = {
    'enable_performance': True,    # PSR, precision, entropy
    'enable_regime': True,        # Volatility, SADF, trends
    'enable_market': True,        # Relative strength, VWAP
    'enable_technical': True,     # ADX, choppiness, etc.
    'enable_structural': True,    # Break scores, drawdown
    'enable_model': True,         # Disagreement, ensemble
    'enable_time': True,          # Temporal patterns
    'enable_contextual': True,    # Residuals, harmonization
    'harmonization_type': 'direction',
    'max_residual_features': 50
}
```

## Backward Compatibility

All existing interfaces maintain backward compatibility:
- `MetaLearnerFeatures.generate()` - Uses unified generator with fallback
- `compute_layer4_regime_features()` - Uses unified generator with fallback
- `generate_layer4_features()` - Uses unified generator with fallback

## Benefits

1. **Consolidation** - All feature generation in one place
2. **Configuration** - Enable/disable feature categories
3. **Validation** - Centralized feature pattern management
4. **Performance** - Optimized computation with caching
5. **Maintainability** - Single source of truth for features
6. **Extensibility** - Easy to add new feature categories

## Migration

The refactoring is transparent to existing code. All imports and function calls remain the same, with the unified system operating behind the scenes.

## Testing

To test the unified system:

```python
# Test basic functionality
from src.training.steps.labeling.layer4 import Layer4FeatureGenerator, validate_layer4_features

generator = Layer4FeatureGenerator()
features = generator.generate_all_features(test_df, test_layer3_df)
validation = validate_layer4_features(features)

print(f"Generated {len(features.columns)} features")
print(f"Feature coverage: {validation['coverage_rate']:.2%}")
```
