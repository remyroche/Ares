# 🎯 Micro-Movement Detection System

## Overview

The Ares trading bot includes a sophisticated **micro-movement detection system** designed to capture small profit opportunities (even less than 0.5%) with high leverage, especially around Support/Resistance zones and during huge candle events.

## 🔍 **Detection Components**

### 1. **Micro-Movement Detection** (`src/tactician/tactician.py`)
- **Threshold**: 0.2% price movement (configurable via `micro_movement_threshold`)
- **Detection**: Compares current price to previous candle close
- **Purpose**: Identifies small, precise price movements for high-leverage opportunities

### 2. **Huge Candle Detection**
- **Threshold**: 5% price movement (configurable via `huge_candle_threshold`)
- **Detection**: Identifies significant volatility events
- **Purpose**: Captures momentum opportunities with enhanced position sizing

### 3. **S/R Zone Proximity Detection**
- **Threshold**: 1% proximity to S/R levels (configurable via `sr_zone_proximity`)
- **Detection**: Uses S/R levels from `src/analyst/sr_analyzer.py`
- **Purpose**: Identifies opportunities near key support/resistance levels

## 🎯 **Opportunity Types**

The system classifies opportunities into distinct types:

| Opportunity Type | Conditions | Strategy |
|------------------|------------|----------|
| **SR_FADE** | Micro-movement + Near S/R | Fade against S/R level |
| **SR_BREAKOUT** | Huge candle + Near S/R | Breakout from S/R level |
| **MOMENTUM** | Huge candle only | Follow momentum |
| **MICRO_MOVEMENT** | Micro-movement only | Small, precise moves |
| **STANDARD** | None of the above | Regular trading |

## ⚡ **Enhanced Position Sizing**

### Position Size Multipliers
```python
"position_size_multiplier": {
    "high_confidence": 1.5,      # High confidence signals
    "sr_zone": 1.3,             # Near S/R zones
    "huge_candle": 1.4,         # Huge candle events
    "micro_movement": 1.5,      # Micro-movement opportunities
    "combined": 2.0             # Multiple conditions met
}
```

### Leverage Enhancements
- **Micro-movements**: 2.0x leverage boost
- **S/R zones**: 1.5x leverage boost
- **Huge candles**: 2.0x leverage boost
- **High confidence**: 1.8x leverage boost

## 🔧 **Configuration Parameters**

### Tactician Configuration (`src/config.py`)
```python
"tactician": {
    "micro_movement_threshold": 0.002,  # 0.2% for micro movements
    "huge_candle_threshold": 0.05,      # 5% for huge candles
    "sr_zone_proximity": 0.01,          # 1% proximity to S/R zones
    "high_confidence_threshold": 0.85,   # High confidence threshold
    "max_leverage_cap": 100,            # Maximum leverage
    "position_size_multiplier": {
        "micro_movement": 1.5,
        "high_confidence": 1.5,
        "sr_zone": 1.3,
        "huge_candle": 1.4,
        "combined": 2.0
    }
}
```

## 🔄 **Integration Flow**

### 1. **Data Flow**
```
Analyst → Technical Analysis → S/R Levels → Tactician → Micro-Movement Detection
```

### 2. **Detection Process**
1. **Analyst** provides technical analysis with recent klines and S/R levels
2. **Tactician** analyzes price movements and S/R proximity
3. **Detection** identifies opportunity type and market conditions
4. **Position Sizing** applies appropriate multipliers
5. **Leverage** adjusts based on opportunity type and confidence

### 3. **Execution Process**
1. **Order Preparation**: Specialized order types for each opportunity
2. **Risk Management**: Tighter stops for micro-movements
3. **Position Sizing**: Enhanced sizing for high-probability setups
4. **Monitoring**: Enhanced logging for micro-movement opportunities

## 📊 **Specialized Order Types**

### SR Fade Orders
- **Entry**: Limit orders at S/R levels
- **Stops**: Tighter stops (0.5x ATR multiplier)
- **Targets**: 2.0x ATR multiplier
- **Purpose**: Fade against S/R levels during micro-movements

### SR Breakout Orders
- **Entry**: Market orders on breakout
- **Stops**: Below/above S/R level
- **Targets**: Based on ATR and S/R distance
- **Purpose**: Capture breakouts from S/R zones

## 🎯 **Key Features**

### 1. **Enhanced Logging**
- Detailed opportunity detection logs
- Position sizing breakdown
- Leverage calculation details
- Market condition analysis

### 2. **Risk Management**
- Tighter stops for micro-movements
- Enhanced position sizing for high-probability setups
- Leverage caps to prevent over-exposure
- Multiple condition validation

### 3. **Adaptive Parameters**
- Configurable thresholds for all detection parameters
- Dynamic position sizing based on market conditions
- Leverage adjustments based on opportunity type
- Confidence-based enhancements

## 🔍 **Detection Logic**

### Micro-Movement Detection
```python
price_change = abs(current_price - prev_price) / prev_price
is_micro_movement = price_change <= micro_movement_threshold  # 0.2%
```

### S/R Zone Detection
```python
for sr_level in sr_levels:
    distance = abs(current_price - level_price) / current_price
    if distance <= sr_zone_proximity:  # 1%
        is_near_sr = True
```

### Opportunity Classification
```python
if huge_candle and near_sr:
    return "SR_BREAKOUT"
elif micro_movement and near_sr:
    return "SR_FADE"
elif huge_candle:
    return "MOMENTUM"
elif micro_movement:
    return "MICRO_MOVEMENT"
else:
    return "STANDARD"
```

## 📈 **Performance Monitoring**

### Key Metrics
- **Micro-movement detection rate**
- **S/R zone interaction frequency**
- **Position sizing effectiveness**
- **Leverage utilization**
- **Profit capture from small moves**

### Logging Examples
```
🎯 MICRO-MOVEMENT OPPORTUNITY DETECTED:
   Price Change: 0.0015 (0.15%)
   Micro Movement: True (threshold: 0.002)
   Near S/R Zone: True (distance: 0.008)
   Nearest S/R: Support at 45000.00
   Opportunity Type: SR_FADE
```

## ⚙️ **Configuration Tuning**

### For More Aggressive Micro-Movement Trading
```python
"micro_movement_threshold": 0.001,  # 0.1% threshold
"sr_zone_proximity": 0.005,         # 0.5% proximity
"position_size_multiplier": {
    "micro_movement": 2.0,          # Higher multiplier
    "combined": 3.0                 # Higher combined multiplier
}
```

### For Conservative Micro-Movement Trading
```python
"micro_movement_threshold": 0.003,  # 0.3% threshold
"sr_zone_proximity": 0.015,         # 1.5% proximity
"position_size_multiplier": {
    "micro_movement": 1.2,          # Lower multiplier
    "combined": 1.5                 # Lower combined multiplier
}
```

## 🚀 **Integration Status**

✅ **Fully Integrated** with:
- Analyst's technical analysis
- S/R level detection
- Position sizing system
- Leverage calculation
- Risk management
- Order execution
- Performance monitoring

The micro-movement detection system is **fully operational** and ready for live trading. It provides enhanced capabilities for capturing small profit opportunities while maintaining robust risk management. 