# Enhanced S/R Detection Method Summary

## Overview

The enhanced S/R detection method now includes two critical improvements:
1. **S/R Flip Logic** - Explicitly tracks and rewards levels that have acted as both support and resistance
2. **Time-Decay Factor** - Applies decay to level strength based on age to prioritize recent levels

## S/R Level Detection Process

### 1. **Fractal Analysis with Flip Detection**
- **Swing Highs/Lows**: Identifies price points that are higher/lower than surrounding prices
- **Flip Logic**: Tracks price interactions with each level to detect role reversals
- **Strength Calculation**: Base strength (60% price movement + 40% volume) + flip bonus
- **Flip Bonus**: Up to 0.3 additional strength for confirmed role reversals

### 2. **Volume-Weighted Levels with Time-Decay**
- **Volume Profile**: Creates price bins and calculates volume distribution
- **Time-Decay Weights**: Applies exponential decay to favor recent volume activity
- **Strength Calculation**: Volume ratio × (0.7 + 0.3 × time_decay_factor)
- **Recent Bias**: More recent volume activity gets higher weight

### 3. **Pivot Points with Flip Detection**
- **Traditional Pivots**: Uses existing regime classifier's rolling pivots
- **Flip Check**: Analyzes recent price action around pivot levels
- **Flip Bonus**: Up to 0.2 additional strength for pivot levels with role reversals
- **Support/Resistance**: S1, S2 (support) and R1, R2 (resistance)

### 4. **Consolidation with Enhanced Logic**
- **Proximity Grouping**: Groups nearby levels (within 0.5% proximity)
- **Flip Consideration**: Additional flip detection on merged levels
- **Strength Enhancement**: Combines base strength + flip bonuses
- **Deduplication**: Removes redundant levels while preserving flip information

### 5. **Time-Decay Application**
- **Age Calculation**: Determines how old each level is in hours
- **Half-Life**: 168 hours (1 week) for exponential decay
- **Decay Formula**: `decay_factor = exp(-ln(2) * age_hours / 168)`
- **Strength Adjustment**: `final_strength = base_strength × (0.5 + 0.5 × decay_factor)`

### 6. **ATR-Based Activation Ranges**
- **Dynamic Ranges**: Activation ranges based on ATR and level strength
- **Fallback**: Percentage-based ranges if ATR is too small
- **Strength Integration**: Stronger levels get wider activation ranges

## S/R Flip Logic Details

### **Flip Detection Process**
1. **Proximity Tracking**: Monitors price touches within 0.5% of level price
2. **Role Classification**: 
   - Support touch: Price touches from below
   - Resistance touch: Price touches from above
3. **Sequence Analysis**: Tracks the order of support/resistance touches
4. **Flip Confirmation**: Identifies when a level switches roles

### **Flip Bonus Structure**
- **Multiple Flips (≥2)**: Up to 0.3 bonus strength
- **Single Flip**: 0.15 bonus strength  
- **Dual Touches**: 0.05 bonus for levels touched from both sides
- **No Flips**: 0.0 bonus

### **Flip Sequence Examples**
- **R→S**: Level acted as resistance, then as support
- **S→R**: Level acted as support, then as resistance
- **Multiple**: R→S→R or S→R→S patterns

## Time-Decay Factor Details

### **Decay Calculation**
```python
# Half-life of 1 week (168 hours)
half_life_hours = 168
decay_factor = exp(-ln(2) * age_hours / half_life_hours)

# Apply to strength
final_strength = base_strength × (0.5 + 0.5 × decay_factor)
```

### **Decay Examples**
- **1 hour old**: decay_factor ≈ 1.0 (no decay)
- **1 day old**: decay_factor ≈ 0.9 (10% decay)
- **1 week old**: decay_factor ≈ 0.5 (50% decay)
- **1 month old**: decay_factor ≈ 0.06 (94% decay)

### **Benefits**
- **Recent Relevance**: Newer levels get higher priority
- **Natural Cleanup**: Old levels fade away automatically
- **Market Adaptation**: System adapts to changing market conditions

## Enhanced Features for HMM

The HMM model now receives only two essential features:

### **SR_Score**
- **Formula**: `Directional_Pressure × Strength_Score`
- **Directional_Pressure**: `(NDRS - NDSS) / (NDRS + NDSS)`
- **Strength_Score**: Combines level strength, density, and volume (replaces Clarity Factor)
- **Range**: -1.0 to +1.0

### **ΔSR_Score**
- **Formula**: `SR_Score_t - SR_Score_t-1`
- **Purpose**: Detects changes in S/R context (breakouts, bounces)
- **Range**: Unbounded (typically -2.0 to +2.0)

## Internal Calculations (Not for HMM)

The following calculations are performed internally but not used by HMM:

### **Distance Calculations**
- `distance_to_support` & `distance_to_resistance`
- `normalized_distance_to_support` & `normalized_distance_to_resistance`
- ATR-normalized distances

### **Strength Components**
- **Level Strength**: Based on touches, volume, age
- **Level Density**: Number of nearby levels
- **Volume Factor**: Current volume relative to average
- **Flip Bonus**: Additional strength for role reversals
- **Time Decay**: Age-based strength reduction

### **Proximity and Clarity**
- **Proximity Score**: Market congestion indicator
- **Directional Pressure**: Bias toward support vs resistance
- **Enhanced Clarity**: Strength-adjusted distances

## Configuration Parameters

### **Flip Logic**
- **Proximity Threshold**: 0.5% of level price
- **Lookback Period**: 50 periods for flip detection
- **Flip Bonus Range**: 0.0 to 0.3

### **Time Decay**
- **Half-Life**: 168 hours (1 week)
- **Minimum Strength**: 50% of original after decay
- **Default Age**: 24 hours for levels without timestamps

### **Detection Methods**
- **Fractal Lookback**: 5 periods (configurable)
- **Volume Window**: 100 periods (configurable)
- **Pivot Periods**: 24 periods (configurable)

## Benefits of Enhanced Detection

### **Improved Accuracy**
- **Role Reversal Recognition**: Identifies key levels that switch roles
- **Time Relevance**: Prioritizes recent and active levels
- **Strength Validation**: Multiple confirmation methods

### **Better Regime Detection**
- **Granular Patterns**: More detailed S/R context for HMM
- **Breakout Detection**: ΔSR_Score captures regime changes
- **Market Adaptation**: System evolves with market conditions

### **Trading Applications**
- **Entry/Exit Points**: Strong S/R levels with flips
- **Risk Management**: ATR-based activation ranges
- **Position Sizing**: Strength-based confidence scoring

## Implementation Notes

### **Performance Considerations**
- **Efficient Tracking**: Level interactions tracked in memory
- **Batch Processing**: Flip detection optimized for large datasets
- **Caching**: Time-decay calculations cached for efficiency

### **Error Handling**
- **Graceful Degradation**: Fallback to basic detection if enhanced fails
- **Data Validation**: Robust handling of missing or invalid data
- **Logging**: Comprehensive logging for debugging and monitoring

### **Extensibility**
- **Modular Design**: Easy to add new detection methods
- **Configurable Parameters**: All thresholds and weights adjustable
- **Plugin Architecture**: New flip detection algorithms can be added

This enhanced S/R detection method provides much more accurate and relevant support/resistance levels, leading to better HMM regime detection and improved trading performance.