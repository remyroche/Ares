# Feature Interaction Constructor

## Overview

This module provides the infrastructure to map selected interaction features back to their base feature_bank features and automatically calculate them during live trading.

## Problem Statement

During training, the `feature_generation_final_feature_selection_step` selects 60-50-40 features from a pool that includes:

1. **Base features** from feature_bank (e.g., `rsi_14`, `ema_20`)
2. **Variant features** with transformations (e.g., `rsi_14_volnorm`, `ema_20_vwap`)
3. **Cross-timeframe features** (e.g., `rsi_14_base_3x_ratio`, `ema_20_vwap_27x_ratio`)
4. **Complex interaction features** (e.g., `fibonacci_0.236_5_price_returns_vwap_27x_ratio_x_wavelet_energy_base_6x_ratio`)

These selected features have different names than the features in our feature_bank, making it impossible to automatically calculate them during live trading without a mapping mechanism.

## Feature Naming Convention

### Base Features
- Format: `{indicator_name}_{period}_{data_type}`
- Example: `rsi_14_price_returns`

### Variant Features
Features can have 4 variant types:
- `_base`: Original feature (no transformation)
- `_volnorm`: Volatility-normalized variant
- `_vwap`: Volume-weighted average price variant
- `_trend_adj`: Trend-adjusted variant

Example: `rsi_14_volnorm`, `ema_20_vwap`

### Cross-Timeframe Features
Features with extended lookback periods creating ratio interactions:
- Format: `{base_feature}_{variant}_{multiplier}x_ratio`
- Multipliers: 3x, 6x, 9x, 27x
- Example: `rsi_14_base_3x_ratio`, `ema_20_volnorm_27x_ratio`

### Complex Interaction Features
Mathematical combinations of features using operators:
- Operators: `x` (multiply), `div` (divide), `log`, `log_ratio`, `minus`, `plus`
- Format: `{feature1}_{operator}_{feature2}`
- Example: `fibonacci_0.236_5_price_returns_vwap_27x_ratio_x_wavelet_energy_base_6x_ratio`

## Solution Architecture

### 1. Feature Decomposition Parser
**File:** `feature_decomposer.py`

Parses complex feature names into their constituent parts:
```python
{
    'feature_name': 'rsi_14_volnorm_3x_ratio',
    'base_feature': 'rsi_14',
    'variant_type': 'volnorm',
    'timeframe_multiplier': 3,
    'operators': [],
    'dependencies': ['rsi_14']
}
```

### 2. Feature Calculation Graph
**File:** `calculation_graph.py`

Creates a directed acyclic graph (DAG) of feature dependencies:
- Identifies base features needed from feature_bank
- Determines calculation order
- Handles complex interaction chains

### 3. Feature Transformer
**File:** `feature_transformer.py`

Applies transformations to convert base features into variants:
- Volatility normalization
- VWAP weighting
- Trend adjustment
- Cross-timeframe ratio calculation

### 4. Feature Calculator
**File:** `feature_calculator.py`

Main orchestrator that:
1. Takes selected feature names
2. Decomposes them into dependencies
3. Retrieves base features from feature_bank
4. Applies transformations and calculations
5. Returns calculated feature values for live trading

### 5. Feature Metadata Store
**File:** `feature_metadata_store.py`

Stores metadata about selected features:
- Feature dependencies
- Calculation steps
- Parameters (lookback periods, etc.)
- Category information

## Usage Example

```python
from src.interaction_features_constructor import FeatureCalculator

# Initialize with selected features
selected_features = [
    'fibonacci_0.236_5_price_returns_vwap_27x_ratio_x_wavelet_energy_base_6x_ratio',
    'candlestick_doji_pattern_base_27x_ratio',
    'vectorbt_enhanced_obv_10_base_3x_ratio'
]

calculator = FeatureCalculator(selected_features)

# Get base features needed
base_features_needed = calculator.get_required_base_features()
# Returns: ['fibonacci_0.236_5_price_returns', 'wavelet_energy',
#           'candlestick_doji_pattern', 'vectorbt_enhanced_obv_10']

# Calculate features from OHLCV data
ohlcv_data = get_live_ohlcv()
calculated_features = calculator.calculate(ohlcv_data, feature_bank)
# Returns: DataFrame with the selected features calculated
```

## Integration Points

### 1. Training Pipeline
After feature selection in `feature_generation_final_feature_selection_step`:
```python
# Save feature metadata along with selected features
metadata = FeatureMetadataStore.create_from_selection(selected_features)
metadata.save(f'artifacts/feature_metadata_{timestamp}.json')
```

### 2. Live Trading
Load metadata and calculate features:
```python
# Load saved metadata
metadata = FeatureMetadataStore.load('artifacts/feature_metadata_latest.json')

# Initialize calculator
calculator = FeatureCalculator.from_metadata(metadata)

# Calculate features for current candle
features = calculator.calculate(current_ohlcv, feature_bank)
```

## Implementation Status

- [ ] Feature Decomposition Parser
- [ ] Feature Calculation Graph
- [ ] Feature Transformer
- [ ] Feature Calculator
- [ ] Feature Metadata Store
- [ ] Integration with training pipeline
- [ ] Integration with live trading system

## Next Steps

1. Implement feature decomposition parser to parse complex feature names
2. Create calculation graph to determine dependency order
3. Build feature transformer for variant generation
4. Develop main feature calculator orchestrator
5. Add metadata storage and retrieval
6. Integrate with existing training and trading systems
