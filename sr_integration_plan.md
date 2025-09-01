# SR Features Integration Plan for ML Models

## Current Situation Analysis

### **What We Have**
1. **SRBreakoutPredictor**: Sophisticated S/R detection and analysis
2. **Analyst Integration**: Basic SR features already integrated in `unified_regime_classifier.py`
3. **Tactician Integration**: Basic feature extraction in `ml_tactics_manager.py`
4. **Trained ML Models**: Models already trained to predict breakouts

### **What We Need**
1. **Enhanced SR Features**: More comprehensive SR features for both Analyst and Tactician
2. **Consistent Integration**: Ensure both components use the same SR feature set
3. **Real-time Updates**: SR features updated with each prediction cycle
4. **Feature Synchronization**: Same SR context used across all ML models

## Integration Strategy

### **1. Enhanced SR Feature Set**

#### **Core SR Features (for both Analyst and Tactician)**
```python
sr_features = {
    # Proximity Features
    "sr_proximity": 0.02,                    # Distance to nearest S/R level
    "support_proximity": 0.015,              # Distance to nearest support
    "resistance_proximity": 0.025,           # Distance to nearest resistance
    "sr_zone_width": 0.04,                   # Width of current S/R zone

    # Strength Features
    "sr_strength": 0.75,                     # Strength of nearest S/R level
    "support_strength": 0.8,                 # Strength of nearest support
    "resistance_strength": 0.7,              # Strength of nearest resistance
    "sr_enhanced_strength": 0.82,            # Enhanced strength score

    # Level Count Features
    "support_level_count": 3,                # Number of support levels
    "resistance_level_count": 2,             # Number of resistance levels
    "total_sr_levels": 5,                    # Total S/R levels
    "sr_cluster_count": 2,                   # Number of S/R clusters

    # Advanced Analysis Features
    "sr_fibonacci_proximity": 0.03,          # Distance to Fibonacci levels
    "sr_elliott_proximity": 0.04,            # Distance to Elliott Wave levels
    "sr_order_flow_imbalance": 0.15,         # Order flow imbalance

    # Historical Features
    "sr_touch_count": 5,                     # Average touch count
    "sr_bounce_rate": 0.8,                   # Average bounce rate
    "sr_isolation_score": 0.3,               # Level isolation score

    # Breakout Features
    "support_breakout_probability": 0.15,    # Probability of support breakout
    "resistance_breakout_probability": 0.0,  # Probability of resistance breakout
    "sr_breakout_confidence": 0.6,           # Overall breakout confidence
}
```

### **2. Analyst Integration Enhancement**

#### **Update `unified_regime_classifier.py`**
```python
async def _add_enhanced_sr_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced S/R feature integration for Analyst ML models.
    """
    try:
        # Initialize SR features
        sr_feature_columns = [
            "sr_proximity", "support_proximity", "resistance_proximity", "sr_zone_width",
            "sr_strength", "support_strength", "resistance_strength", "sr_enhanced_strength",
            "support_level_count", "resistance_level_count", "total_sr_levels", "sr_cluster_count",
            "sr_fibonacci_proximity", "sr_elliott_proximity", "sr_order_flow_imbalance",
            "sr_touch_count", "sr_bounce_rate", "sr_isolation_score",
            "support_breakout_probability", "resistance_breakout_probability", "sr_breakout_confidence"
        ]

        for col in sr_feature_columns:
            features_df[col] = 0.0

        # Calculate SR features for each data point
        for i in range(50, len(features_df)):
            window_data = features_df.iloc[max(0, i-100):i+1]
            current_price = features_df["close"].iloc[i]

            # Get comprehensive SR context
            sr_context = await self.sr_predictor.get_sr_context(window_data, current_price)

            if sr_context:
                # Update all SR features
                self._update_sr_features_row(features_df, i, sr_context, current_price)

        return features_df

    except Exception as e:
        self.logger.error(f"Error adding enhanced SR features: {e}")
        return self._add_basic_sr_features(features_df)

def _update_sr_features_row(self, features_df: pd.DataFrame, index: int, sr_context: dict, current_price: float):
    """
    Update a single row with SR features.
    """
    # Proximity features
    features_df.loc[features_df.index[index], "sr_proximity"] = min(
        sr_context.get("support_proximity", 1.0),
        sr_context.get("resistance_proximity", 1.0)
    )
    features_df.loc[features_df.index[index], "support_proximity"] = sr_context.get("support_proximity", 1.0)
    features_df.loc[features_df.index[index], "resistance_proximity"] = sr_context.get("resistance_proximity", 1.0)
    features_df.loc[features_df.index[index], "sr_zone_width"] = sr_context.get("sr_zone_width", 0.0)

    # Strength features
    features_df.loc[features_df.index[index], "sr_strength"] = max(
        sr_context.get("support_strength", 0.5),
        sr_context.get("resistance_strength", 0.5)
    )
    features_df.loc[features_df.index[index], "support_strength"] = sr_context.get("support_strength", 0.5)
    features_df.loc[features_df.index[index], "resistance_strength"] = sr_context.get("resistance_strength", 0.5)

    # Level count features
    support_levels = sr_context.get("support_levels", [])
    resistance_levels = sr_context.get("resistance_levels", [])
    features_df.loc[features_df.index[index], "support_level_count"] = len(support_levels)
    features_df.loc[features_df.index[index], "resistance_level_count"] = len(resistance_levels)
    features_df.loc[features_df.index[index], "total_sr_levels"] = len(support_levels) + len(resistance_levels)

    # Advanced features
    features_df.loc[features_df.index[index], "sr_fibonacci_proximity"] = sr_context.get("fibonacci_proximity", 1.0)
    features_df.loc[features_df.index[index], "sr_elliott_proximity"] = sr_context.get("elliott_proximity", 1.0)
    features_df.loc[features_df.index[index], "sr_order_flow_imbalance"] = sr_context.get("order_flow_imbalance", 0.0)

    # Historical features
    features_df.loc[features_df.index[index], "sr_touch_count"] = sr_context.get("avg_touch_count", 0)
    features_df.loc[features_df.index[index], "sr_bounce_rate"] = sr_context.get("avg_bounce_rate", 0.5)
    features_df.loc[features_df.index[index], "sr_isolation_score"] = sr_context.get("avg_isolation_score", 0.5)

    # Breakout features
    features_df.loc[features_df.index[index], "support_breakout_probability"] = sr_context.get("support_breakout_prob", 0.0)
    features_df.loc[features_df.index[index], "resistance_breakout_probability"] = sr_context.get("resistance_breakout_prob", 0.0)
    features_df.loc[features_df.index[index], "sr_breakout_confidence"] = sr_context.get("breakout_confidence", 0.5)
```

### **3. Tactician Integration Enhancement**

#### **Update `ml_tactics_manager.py`**
```python
def _extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
    """
    Enhanced feature extraction with comprehensive SR features.
    """
    try:
        features = []

        if len(market_data) < 20:
            return np.array([0.5] * 30)  # Increased feature count

        # Basic technical features (existing)
        close_prices = market_data['close'].values
        high_prices = market_data['high'].values
        low_prices = market_data['low'].values
        volumes = market_data['volume'].values

        # Technical indicators (existing)
        price_momentum = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns[-20:])
        volume_trend = (volumes[-1] - volumes[-5]) / volumes[-5] if volumes[-5] > 0 else 0
        price_range = (high_prices[-1] - low_prices[-1]) / close_prices[-1]

        # Moving averages
        ma_short = np.mean(close_prices[-5:])
        ma_long = np.mean(close_prices[-20:])
        ma_ratio = ma_short / ma_long if ma_long > 0 else 1.0

        # RSI
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 1.0
        rsi = 100 - (100 / (1 + rs))

        # Additional technical features
        latest_return = close_prices[-1] / close_prices[-2] - 1
        volume_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1.0
        upper_shadow = (high_prices[-1] - close_prices[-1]) / close_prices[-1]
        lower_shadow = (close_prices[-1] - low_prices[-1]) / close_prices[-1]

        # Add basic features
        features.extend([
            price_momentum, volatility, volume_trend, price_range, ma_ratio,
            rsi / 100, latest_return, volume_ratio, upper_shadow, lower_shadow
        ])

        # NEW: SR Features (same as Analyst)
        sr_features = self._extract_sr_features(market_data)
        features.extend(sr_features)

        return np.array(features)

    except Exception as e:
        self.logger.error(f"Feature extraction failed: {e}")
        return np.array([0.5] * 30)

async def _extract_sr_features(self, market_data: pd.DataFrame) -> list[float]:
    """
    Extract SR features for Tactician ML models.
    """
    try:
        current_price = market_data['close'].iloc[-1]

        # Get SR context
        sr_context = await self.sr_predictor.get_sr_context(market_data, current_price)

        if not sr_context:
            return [0.5] * 20  # Default values

        # Extract SR features (same set as Analyst)
        sr_features = [
            # Proximity features
            min(sr_context.get("support_proximity", 1.0), sr_context.get("resistance_proximity", 1.0)),
            sr_context.get("support_proximity", 1.0),
            sr_context.get("resistance_proximity", 1.0),
            sr_context.get("sr_zone_width", 0.0),

            # Strength features
            max(sr_context.get("support_strength", 0.5), sr_context.get("resistance_strength", 0.5)),
            sr_context.get("support_strength", 0.5),
            sr_context.get("resistance_strength", 0.5),
            sr_context.get("enhanced_strength", 0.5),

            # Level count features
            len(sr_context.get("support_levels", [])),
            len(sr_context.get("resistance_levels", [])),
            len(sr_context.get("support_levels", [])) + len(sr_context.get("resistance_levels", [])),
            sr_context.get("cluster_count", 0),

            # Advanced features
            sr_context.get("fibonacci_proximity", 1.0),
            sr_context.get("elliott_proximity", 1.0),
            sr_context.get("order_flow_imbalance", 0.0),

            # Historical features
            sr_context.get("avg_touch_count", 0),
            sr_context.get("avg_bounce_rate", 0.5),
            sr_context.get("avg_isolation_score", 0.5),

            # Breakout features
            sr_context.get("support_breakout_prob", 0.0),
            sr_context.get("resistance_breakout_prob", 0.0),
            sr_context.get("breakout_confidence", 0.5)
        ]

        return sr_features

    except Exception as e:
        self.logger.error(f"SR feature extraction failed: {e}")
        return [0.5] * 20
```

### **4. SRBreakoutPredictor Enhancement**

#### **Add Feature Extraction Methods**
```python
async def extract_ml_features(self, market_data: pd.DataFrame, current_price: float) -> dict[str, float]:
    """
    Extract standardized SR features for ML models.
    """
    try:
        # Get comprehensive SR context
        sr_context = await self.get_sr_context(market_data, current_price)

        if not sr_context:
            return self._get_default_sr_features()

        # Extract standardized features
        features = {
            # Proximity features
            "sr_proximity": min(sr_context.get("support_proximity", 1.0), sr_context.get("resistance_proximity", 1.0)),
            "support_proximity": sr_context.get("support_proximity", 1.0),
            "resistance_proximity": sr_context.get("resistance_proximity", 1.0),
            "sr_zone_width": sr_context.get("sr_zone_width", 0.0),

            # Strength features
            "sr_strength": max(sr_context.get("support_strength", 0.5), sr_context.get("resistance_strength", 0.5)),
            "support_strength": sr_context.get("support_strength", 0.5),
            "resistance_strength": sr_context.get("resistance_strength", 0.5),
            "sr_enhanced_strength": sr_context.get("enhanced_strength", 0.5),

            # Level count features
            "support_level_count": len(sr_context.get("support_levels", [])),
            "resistance_level_count": len(sr_context.get("resistance_levels", [])),
            "total_sr_levels": len(sr_context.get("support_levels", [])) + len(sr_context.get("resistance_levels", [])),
            "sr_cluster_count": sr_context.get("cluster_count", 0),

            # Advanced features
            "sr_fibonacci_proximity": sr_context.get("fibonacci_proximity", 1.0),
            "sr_elliott_proximity": sr_context.get("elliott_proximity", 1.0),
            "sr_order_flow_imbalance": sr_context.get("order_flow_imbalance", 0.0),

            # Historical features
            "sr_touch_count": sr_context.get("avg_touch_count", 0),
            "sr_bounce_rate": sr_context.get("avg_bounce_rate", 0.5),
            "sr_isolation_score": sr_context.get("avg_isolation_score", 0.5),

            # Breakout features
            "support_breakout_probability": sr_context.get("support_breakout_prob", 0.0),
            "resistance_breakout_probability": sr_context.get("resistance_breakout_prob", 0.0),
            "sr_breakout_confidence": sr_context.get("breakout_confidence", 0.5)
        }

        return features

    except Exception as e:
        self.logger.error(f"Error extracting ML features: {e}")
        return self._get_default_sr_features()

def _get_default_sr_features(self) -> dict[str, float]:
    """
    Return default SR features when analysis fails.
    """
    return {
        "sr_proximity": 0.5, "support_proximity": 0.5, "resistance_proximity": 0.5, "sr_zone_width": 0.0,
        "sr_strength": 0.5, "support_strength": 0.5, "resistance_strength": 0.5, "sr_enhanced_strength": 0.5,
        "support_level_count": 0, "resistance_level_count": 0, "total_sr_levels": 0, "sr_cluster_count": 0,
        "sr_fibonacci_proximity": 1.0, "sr_elliott_proximity": 1.0, "sr_order_flow_imbalance": 0.0,
        "sr_touch_count": 0, "sr_bounce_rate": 0.5, "sr_isolation_score": 0.5,
        "support_breakout_probability": 0.0, "resistance_breakout_probability": 0.0, "sr_breakout_confidence": 0.5
    }
```

## Implementation Steps

### **Step 1: Update SRBreakoutPredictor**
1. Add `extract_ml_features()` method
2. Add `_get_default_sr_features()` method
3. Ensure consistent feature naming

### **Step 2: Update Analyst Integration**
1. Enhance `_add_enhanced_sr_features()` in `unified_regime_classifier.py`
2. Add `_update_sr_features_row()` method
3. Ensure all 20 SR features are included

### **Step 3: Update Tactician Integration**
1. Enhance `_extract_features()` in `ml_tactics_manager.py`
2. Add `_extract_sr_features()` method
3. Increase feature array size to accommodate SR features

### **Step 4: Feature Synchronization**
1. Ensure both Analyst and Tactician use the same SR feature set
2. Use the same SR context calculation method
3. Maintain feature consistency across all ML models

### **Step 5: Testing and Validation**
1. Test SR feature extraction
2. Validate feature consistency between Analyst and Tactician
3. Ensure ML models receive updated SR features

## Benefits

### **1. Enhanced ML Model Performance**
- **Comprehensive SR Context**: 20+ SR features for better predictions
- **Consistent Feature Set**: Same features across all ML models
- **Real-time Updates**: SR features updated with each prediction cycle

### **2. Improved Breakout Prediction**
- **Advanced SR Analysis**: Fibonacci, Elliott Wave, Order Flow
- **Breakout Probabilities**: Direct breakout probability features
- **Strength Metrics**: Enhanced strength scoring

### **3. Better Risk Management**
- **Proximity Analysis**: Distance to key S/R levels
- **Zone Analysis**: S/R zone width and characteristics
- **Historical Context**: Touch count, bounce rate, isolation score

### **4. Unified Architecture**
- **Single SR Source**: SRBreakoutPredictor as the authoritative source
- **Consistent Integration**: Same feature extraction method everywhere
- **Maintainable Code**: Centralized SR feature logic

## Summary

This integration plan ensures that:

✅ **Both Analyst and Tactician ML models** receive comprehensive SR features
✅ **SR features are updated in real-time** with each prediction cycle
✅ **Feature consistency is maintained** across all components
✅ **ML models can leverage advanced SR analysis** for better predictions
✅ **The existing trained models** can immediately benefit from enhanced SR context

The key is to use the SRBreakoutPredictor as the single source of truth for SR features and ensure both the Analyst and Tactician extract the same comprehensive feature set for their respective ML models.