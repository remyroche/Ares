# Fractal Location Classifier

## Overview

The UnifiedRegimeClassifier has been refactored to focus exclusively on **location-based classification** with fractal (multi-timeframe) analysis. Market regime classification (trending, ranging, volatile, etc.) is now handled by HMM models in the training pipeline.

## Key Changes

### 1. Removed Regime Classification
- Removed directional regimes (TRENDING_UP, TRENDING_DOWN)
- Removed non-directional regimes (RANGING, VOLATILE, ACCUMULATION, DISTRIBUTION)
- Regime classification is now handled by HMM models during training

### 2. Enhanced Location Classification
The classifier now provides fractal location analysis with:

#### Location Types
- **STRONG_SUPPORT / STRONG_RESISTANCE**: Major levels confirmed across multiple timeframes
- **SUPPORT_[TIMEFRAME] / RESISTANCE_[TIMEFRAME]**: Timeframe-specific levels (e.g., SUPPORT_1H, RESISTANCE_4H)
- **BREAKOUT_SUPPORT / BREAKOUT_RESISTANCE**: Price breaking through levels with volume confirmation
- **FALSE_BREAKOUT_SUPPORT / FALSE_BREAKOUT_RESISTANCE**: Breakouts without volume confirmation
- **RETEST_SUPPORT / RETEST_RESISTANCE**: Price retesting previously broken levels
- **CONSOLIDATION_RANGE**: Price consolidating between nearby support and resistance
- **OPEN_RANGE**: No significant levels nearby

#### Fractal Analysis
Multi-timeframe analysis across:
- 1m (micro structure)
- 5m (short-term)
- 15m (medium-term)
- 1h (base timeframe)
- 4h (macro structure)
- 1d (daily structure)

Each timeframe has a weight that contributes to the overall strength of identified levels.

## Implementation

### New Class: UnifiedRegimeClassifierFractal
Located in `src/analyst/unified_regime_classifier_fractal.py`

Key methods:
- `classify_location()`: Main classification method
- `_analyze_fractal_levels()`: Analyzes S/R across timeframes
- `_classify_price_location()`: Determines location type
- `get_location_features()`: Converts classification to ML features

### Integration with Analyst
The `analyst.py` has been updated to:
1. Import `UnifiedRegimeClassifierFractal` instead of `UnifiedRegimeClassifier`
2. Initialize the fractal classifier in `_initialize_regime_classifier()`
3. Add `analyze_regime()` method for supervisor compatibility
4. Update `_perform_regime_classification()` to use fractal classification

### Configuration
See `fractal_classifier_config.yaml` for configuration options:
- Fractal timeframes and weights
- Proximity and breakout thresholds
- Level strength requirements

## Usage Example

```python
# Initialize
classifier = UnifiedRegimeClassifierFractal(config, exchange, symbol)
await classifier.initialize()

# Classify location
location_result = await classifier.classify_location(market_data_df)

# Result structure
{
    'primary_location': 'STRONG_SUPPORT',
    'location_strength': 0.85,
    'action_bias': 'BULLISH',
    'location_details': {
        'nearest_support': {...},
        'nearest_resistance': {...},
        'support_distance_pct': 0.15,
        'resistance_distance_pct': 2.3,
        'volume_confirmation': True,
        'price_range_pct': 1.2
    },
    'nearby_levels': [...],
    'fractal_analysis': {
        '1m': {...},
        '5m': {...},
        ...
    }
}
```

## Benefits

1. **Clearer Separation of Concerns**: Location analysis is separate from regime classification
2. **Enhanced Granularity**: Fractal analysis provides more detailed location information
3. **Better ML Features**: Location-based features are more directly actionable for trading
4. **Improved S/R Detection**: Multi-timeframe confirmation increases reliability

## Migration Notes

- The `regime` field in results is now set to "LOCATION_BASED" as a placeholder
- Actual regime classification comes from HMM models in the training pipeline
- Location features can be extracted using `get_location_features()` for ML models
- The classifier maintains compatibility with existing interfaces through the `analyze_regime()` method