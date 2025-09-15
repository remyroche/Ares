# Migration Guide: Updating Imports to Use the New Unified Feature Generation System

This guide shows how to update existing code to use the new unified feature generation system with enhanced base calculations and interaction features.

## Overview

The new unified system provides:
- **Base Calculations**: Support for price returns, returns-based VWAP, price levels, and volume-weighted calculations
- **Enhanced Indicators**: RSI, MACD, Bollinger Bands, and other indicators can now be based on different calculation methods
- **Interaction Features**: Cross-timeframe interactions, feature ratios, polynomial features, and correlation interactions
- **Backwards Compatibility**: Existing code continues to work without changes

## Migration Steps

### 1. Update Basic Feature Generation Imports

#### Before (Old Scattered Imports)
```python
# Old scattered imports
from src.feature_engineering.feature_generators import FeatureGenerators
from src.feature_engineering.step06_enhanced_feature_engineering_step import EnhancedFeatureEngineeringStep
from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
from src.feature_engineering.cross_timeframe_interaction_features import CrossTimeframeFeatureGenerator
```

#### After (New Unified Imports)
```python
# New unified imports
from src.feature_generation import (
    # Core system
    FeatureBank,
    get_feature_generator,
    
    # Base calculations
    BaseCalculationType,
    calculate_price_returns,
    calculate_returns_vwap,
    
    # Feature generators
    RSIGenerator,
    MACDGenerator,
    BollingerBandsGenerator,
    SMAGenerator,
    
    # Interaction features
    InteractionFeatureGenerator,
    CrossTimeframeInteractionGenerator,
    create_interaction_generators
)
```

### 2. Update Feature Generation Code

#### Before (Old Approach)
```python
# Old approach - using scattered feature generators
feature_generators = FeatureGenerators()
indicators_config = {
    'sma': [5, 10, 20, 50],
    'ema': [5, 10, 20, 50],
    'rsi': [14, 21],
    'macd': [(12, 26, 9)],
    'bb': [(20, 2), (20, 2.5)]
}

features = feature_generators.batch_technical_indicators(
    data=df,
    indicator_configs=indicators_config,
    use_gpu=True
)
```

#### After (New Approach with Base Calculations)
```python
# New approach - using unified system with base calculations
from src.feature_generation import (
    RSIGenerator, MACDGenerator, BollingerBandsGenerator, SMAGenerator,
    BaseCalculationType
)

# Create generators with different base calculations
rsi_returns = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS)
rsi_vwap = RSIGenerator(period=14, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)

macd_levels = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.PRICE_LEVELS)
macd_vwap = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)

bb_levels = BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_LEVELS)
bb_vwap = BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)

# Generate features
features = pd.DataFrame(index=df.index)
features['rsi_returns'] = rsi_returns.generate(df)
features['rsi_vwap'] = rsi_vwap.generate(df)
features['macd_levels'] = macd_levels.generate(df)
features['macd_vwap'] = macd_vwap.generate(df)
features['bb_upper_levels'] = bb_levels.generate(df)
features['bb_upper_vwap'] = bb_vwap.generate(df)
```

### 3. Update Cross-Timeframe Feature Generation

#### Before (Old Cross-Timeframe Approach)
```python
# Old approach
from src.feature_engineering.cross_timeframe_interaction_features import CrossTimeframeFeatureGenerator

config = CrossTimeframeConfig(
    momentum_timeframes=[1, 3, 5, 10, 15, 20],
    volatility_timeframes=[3, 5, 10, 15, 20, 30],
    volume_timeframes=[5, 10, 15, 30]
)

generator = CrossTimeframeFeatureGenerator(config)
features = generator.generate_features(df)
```

#### After (New Interaction Features Approach)
```python
# New approach - using interaction features
from src.feature_generation import (
    CrossTimeframeInteractionGenerator,
    FeatureRatioGenerator,
    create_interaction_generators
)

# Create cross-timeframe interactions
cross_timeframe_ratio = CrossTimeframeInteractionGenerator(
    short_period=5,
    long_period=20,
    interaction_type="ratio"
)

cross_timeframe_diff = CrossTimeframeInteractionGenerator(
    short_period=5,
    long_period=20,
    interaction_type="difference"
)

# Create feature ratios
sma_ratio = FeatureRatioGenerator(
    numerator_period=5,
    denominator_period=20,
    feature_type="sma"
)

# Or create multiple interaction generators at once
interaction_generators = create_interaction_generators({
    'cross_timeframe': {
        'short_periods': [5, 10],
        'long_periods': [20, 50],
        'interaction_types': ['ratio', 'difference', 'product']
    },
    'feature_ratios': {
        'periods': [(5, 20), (10, 30)],
        'feature_types': ['sma', 'ema', 'volatility']
    }
})

# Generate features
features = pd.DataFrame(index=df.index)
features['cross_timeframe_ratio'] = cross_timeframe_ratio.generate(df)
features['cross_timeframe_diff'] = cross_timeframe_diff.generate(df)
features['sma_ratio'] = sma_ratio.generate(df)
```

### 4. Update Feature Bank Usage

#### Before (Old Feature Bank Approach)
```python
# Old approach - if using any feature bank
from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator

config = {
    'enable_advanced_features': True,
    'enable_autoencoder_features': True,
    'enable_microstructure_features': True
}

orchestrator = FeatureEngineeringOrchestrator(config)
features = await orchestrator.generate_all_features(df)
```

#### After (New Feature Bank Approach)
```python
# New approach - using unified feature bank
from src.feature_generation import FeatureBank, BaseCalculationType

# Initialize feature bank
bank = FeatureBank()

# Generate features with different base calculations
rsi_returns = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS)
rsi_vwap = RSIGenerator(period=14, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)

# Generate and store features
features = pd.DataFrame(index=df.index)
features['rsi_returns'] = rsi_returns.generate(df)
features['rsi_vwap'] = rsi_vwap.generate(df)

# Store in bank
bank.add_features("momentum_features", features)

# Retrieve features
retrieved_features = bank.get_features("momentum_features")
```

### 5. Update Matrix Operations Integration

#### Before (Old Matrix Operations)
```python
# Old approach - direct matrix operations
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

matrix_ops = UnifiedMatrixOperations()
result = matrix_ops.matrix_multiply(A, B)
```

#### After (New Matrix Operations Integration)
```python
# New approach - matrix operations integrated into feature generation
from src.feature_generation import RSIGenerator, BaseCalculationType

# Matrix operations are automatically used by feature generators
rsi_generator = RSIGenerator(
    period=14,
    base_calculation=BaseCalculationType.PRICE_RETURNS
)

# The generator automatically uses optimized matrix operations
features = rsi_generator.generate(df)
```

### 6. Update Lookback Optimization

#### Before (Old Optimization)
```python
# Old approach - using existing optimization
from src.feature_engineering.feature_generation_optimization import FeatureGenerationOptimizer

optimizer = FeatureGenerationOptimizer()
result = await optimizer.optimize_feature_lookback(
    data=df,
    feature_name="rsi",
    target_column="target",
    feature_generator=lambda data, lookback: calculate_rsi(data, lookback)
)
```

#### After (New Optimization)
```python
# New approach - using unified optimization
from src.feature_generation import LookbackOptimizer, RSIGenerator

# Create optimizer
optimizer = LookbackOptimizer()

# Optimize RSI with different base calculations
rsi_returns_generator = RSIGenerator(
    period=14,
    base_calculation=BaseCalculationType.PRICE_RETURNS
)

result = await optimizer.optimize_feature(
    data=df,
    feature_name="rsi_returns",
    target_column="target",
    feature_generator_func=lambda data, lookback: rsi_returns_generator.generate(data)
)
```

## Common Migration Patterns

### Pattern 1: Simple Feature Generation
```python
# Before
from src.feature_engineering.feature_generators import FeatureGenerators
generator = FeatureGenerators()
features = generator.batch_technical_indicators(df, config)

# After
from src.feature_generation import RSIGenerator, MACDGenerator, BaseCalculationType
rsi = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS)
macd = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.RETURNS_VWAP)
features = pd.DataFrame(index=df.index)
features['rsi'] = rsi.generate(df)
features['macd'] = macd.generate(df)
```

### Pattern 2: Cross-Timeframe Features
```python
# Before
from src.feature_engineering.cross_timeframe_interaction_features import CrossTimeframeFeatureGenerator
generator = CrossTimeframeFeatureGenerator(config)
features = generator.generate_features(df)

# After
from src.feature_generation import CrossTimeframeInteractionGenerator, create_interaction_generators
generators = create_interaction_generators(config)
features = pd.DataFrame(index=df.index)
for generator in generators:
    features[generator.name] = generator.generate(df)
```

### Pattern 3: Feature Bank Usage
```python
# Before
from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
orchestrator = FeatureEngineeringOrchestrator(config)
features = await orchestrator.generate_all_features(df)

# After
from src.feature_generation import FeatureBank, get_feature_generator
bank = FeatureBank()
generator = get_feature_generator('momentum')
features = generator.generate(df)
bank.add_features('momentum', features)
```

## Benefits of Migration

1. **Enhanced Flexibility**: Indicators can now be based on different calculation methods (price returns, returns-based VWAP, etc.)
2. **Better Organization**: All feature generation is centralized in one system
3. **Improved Performance**: Matrix operations are automatically optimized
4. **Rich Interaction Features**: Cross-timeframe interactions, feature ratios, polynomial features
5. **Backwards Compatibility**: Existing code continues to work
6. **Easy Extension**: Simple to add new feature types and base calculations

## Testing Migration

After updating imports, test your code with:

```python
# Test basic functionality
from src.feature_generation import RSIGenerator, BaseCalculationType

# Test with different base calculations
rsi_returns = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS)
rsi_vwap = RSIGenerator(period=14, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)

# Generate features
features_returns = rsi_returns.generate(df)
features_vwap = rsi_vwap.generate(df)

print(f"RSI (returns): {features_returns.name}")
print(f"RSI (VWAP): {features_vwap.name}")
```

## Support

If you encounter issues during migration:
1. Check that all required columns are present in your data
2. Verify that the new imports are available
3. Test with small datasets first
4. Use the backwards compatibility layer if needed

The new system is designed to be a drop-in replacement for most existing feature generation code while providing enhanced capabilities.