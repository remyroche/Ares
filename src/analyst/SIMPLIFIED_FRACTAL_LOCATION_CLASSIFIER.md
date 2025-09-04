# Simplified Fractal Location Classifier

## Overview

The UnifiedRegimeClassifier has been simplified to focus exclusively on two key metrics:

1. **Distance from S/R levels** - How far the current price is from support/resistance
2. **Strength of S/R levels** - How strong/reliable those levels are

Fractal analysis is used to quantify these metrics across multiple timeframes.

## Key Metrics

### Distance Metrics
- **support_distance**: Normalized distance to nearest support (positive value)
- **resistance_distance**: Normalized distance to nearest resistance (positive value)
- Can be normalized by:
  - Percentage of price (default)
  - ATR (Average True Range)

### Strength Metrics  
- **support_strength**: Strength of nearest support (0-1)
- **resistance_strength**: Strength of nearest resistance (0-1)
- Strength is calculated based on:
  - Number of touches
  - Presence across multiple timeframes
  - Volume at the level
  - Age and isolation of the level

### Combined Metrics
- **combined_location_score**: Overall position (-1 to 1)
  - Negative = closer to support
  - Positive = closer to resistance
  - Weighted by both distance and strength
- **location_quality**: Quality of the analysis (0-1)
  - Based on clarity and strength of S/R structure

## Implementation

### Fractal Analysis Process

1. **Multi-timeframe S/R Detection**
   - Analyzes 6 timeframes: 1m, 5m, 15m, 1h, 4h, 1d
   - Each timeframe has a weight contribution

2. **Level Aggregation**
   - Clusters nearby levels across timeframes
   - Combines strength based on:
     - Timeframe weights
     - Number of touches
     - Multi-timeframe confirmation

3. **Distance Calculation**
   - Measures distance from current price to nearest S/R
   - Normalizes by percentage or ATR

## Output Format

```python
{
    # Core metrics
    'support_distance': 0.023,        # 2.3% below current price
    'resistance_distance': 0.015,     # 1.5% above current price
    'support_strength': 0.85,         # Strong support (0-1)
    'resistance_strength': 0.62,      # Moderate resistance (0-1)
    
    # Combined analysis
    'combined_location_score': 0.35,  # Slightly closer to resistance
    'location_quality': 0.78,         # Good quality S/R structure
    
    # Price levels
    'nearest_support_price': 50000,
    'nearest_resistance_price': 51000,
    
    # Additional details
    'support_details': {
        'touches': 8,
        'timeframe_count': 4,         # Confirmed on 4 timeframes
        'cluster_size': 3             # 3 levels clustered together
    },
    'resistance_details': {
        'touches': 5,
        'timeframe_count': 3,
        'cluster_size': 2
    }
}
```

## ML Features

The classifier provides continuous features for ML models:
- `support_distance`, `resistance_distance`
- `support_strength`, `resistance_strength`
- `combined_location_score`
- `location_quality`
- `support_touches`, `resistance_touches`
- `support_timeframes`, `resistance_timeframes`
- `distance_ratio` (support_dist / resistance_dist)
- `strength_ratio` (support_strength / resistance_strength)

## Configuration

```yaml
analyst:
  unified_regime_classifier:
    # Distance calculation method
    distance_normalization: "percentage"  # or "atr"
    
    # Minimum strength to consider a level valid
    min_strength_threshold: 0.3
    
    # Maximum relevant distance (ignore levels beyond this)
    max_relevant_distance: 0.05  # 5%
    
    # Fractal timeframes and weights
    fractal_timeframes:
      - name: "1m"
        periods: 60
        weight: 0.1
      # ... etc
```

## Benefits

1. **Simplicity**: Only two core concepts - distance and strength
2. **Continuity**: All outputs are continuous values (no discrete labels)
3. **Interpretability**: Clear meaning - how far and how strong
4. **ML-Friendly**: Continuous features work well with all ML algorithms
5. **Robust**: Multi-timeframe analysis provides stable S/R identification