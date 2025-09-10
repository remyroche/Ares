# Backtesting Metrics Detailed Explanation

## 🔬 **How Backtesting Metrics Are Calculated**

### 1. **Touch Detection**
```python
# For each price bar, check if it touches the SR level
tolerance = level.price * touch_tolerance  # Default: 0.2%

# Support level touch detection:
if low <= level.price + tolerance and high >= level.price - tolerance:
    # Price touched the support level

# Resistance level touch detection:
if high >= level.price - tolerance and low <= level.price + tolerance:
    # Price touched the resistance level
```

**What it measures**: Whether price actually interacted with the SR level within a reasonable tolerance.

### 2. **Success Rate Calculation**
```python
# For each touch, analyze if the level held
for i in range(max_hold_time):  # Default: 24 hours
    if level.level_type == 'support':
        # Check if price bounced up from support
        if current_data['low'] <= level.price * (1 + tolerance):
            hold_time = i  # Still at support
        else:
            # Price moved away - calculate bounce strength
            bounce_strength = (current_data['close'] - level.price) / level.price
            if bounce_strength >= min_bounce_strength:  # Default: 0.1%
                successful = True
            break

success_rate = successful_touches / total_touches
```

**What it measures**: Percentage of touches where the level successfully held and price bounced in the expected direction.

### 3. **Bounce Strength Measurement**
```python
# For support levels:
bounce_strength = (close_price - level_price) / level_price

# For resistance levels:
bounce_strength = (level_price - close_price) / level_price

# Average bounce strength across all successful touches
avg_bounce_strength = mean([bounce for bounce in bounce_strengths if bounce >= min_bounce_strength])
```

**What it measures**: How strongly price reacted when touching the level (as percentage of price).

### 4. **Volume Confirmation**
```python
# Calculate average volume at level touches
volumes_at_touches = [touch['volume'] for touch in touches]

# Compare to average volume in the dataset
avg_volume_at_level = mean(volumes_at_touches)
avg_volume_overall = mean(data['volume'])

# Volume confirmation ratio
volume_confirmation = avg_volume_at_level / avg_volume_overall
```

**What it measures**: Whether there was above-average volume when price touched the level (institutional interest).

### 5. **Time Persistence**
```python
# How long the level remains relevant after detection
time_persistence = min(1.0, total_touches / 10.0)  # Normalized to 0-1
```

**What it measures**: How long the level continued to be touched after initial detection (relevance duration).

### 6. **Quality Score Calculation**
```python
quality_score = (
    success_rate_weight * success_rate +                    # 30% weight
    bounce_strength_weight * min(avg_bounce_strength * 10, 1.0) +  # 25% weight
    volume_confirmation_weight * min(avg_volume / 1000000, 1.0) +  # 20% weight
    time_persistence_weight * time_persistence +            # 15% weight
    touch_frequency_weight * min(total_touches / 5.0, 1.0) # 10% weight
)
```

**What it measures**: Combined quality score (0-1) based on all performance metrics.

## 🧠 **What Features Are Tested**

### Primary Features:
1. **Price Behavior**: How price reacts when touching the level
2. **Volume Patterns**: Volume confirmation at level touches
3. **Time Dynamics**: How long the level remains relevant
4. **Touch Frequency**: How often price interacts with the level
5. **Bounce Consistency**: Consistency of price reactions

### Secondary Features:
1. **Market Context**: Overall market conditions during level activity
2. **Volatility Impact**: How market volatility affects level performance
3. **Trend Alignment**: Whether level performance varies with market trend
4. **Time of Day**: Whether level performance varies by time of day

## 🎯 **How Quality Scoring and Prediction Works**

### Quality Scoring:
```python
def predict_level_quality(self, level: SRLevel, data: pd.DataFrame) -> float:
    # Extract features for prediction
    features = self._extract_prediction_features(level, data)
    
    # Apply learned rules
    quality_score = self._apply_learned_rules(features)
    
    return quality_score
```

### Prediction Features:
1. **Historical Performance**: Past success rate and bounce strength
2. **Volume Profile**: Volume patterns around the level
3. **Market Structure**: Support/resistance context
4. **Time Factors**: Detection time and market conditions
5. **Price Position**: Level position relative to current price

### Rule Application:
```python
def _apply_learned_rules(self, features: Dict[str, float]) -> float:
    quality_score = 0.0
    
    # Apply discriminative features
    for feature, info in self.learned_rules.get('discriminative_features', {}).items():
        if features[feature] >= info['threshold']:
            quality_score += info['discriminative_power'] * 0.2
    
    return min(max(quality_score, 0.0), 1.0)
```

## 🔍 **How Discriminative Features Are Identified**

### Feature Analysis:
```python
def _find_discriminative_features(self, high_quality: List[BacktestResult], 
                                low_quality: List[BacktestResult]) -> Dict[str, Any]:
    features = ['success_rate', 'avg_bounce_strength', 'total_volume_at_level', 'total_touches']
    discriminative_features = {}
    
    for feature in features:
        high_values = [getattr(r, feature) for r in high_quality]
        low_values = [getattr(r, feature) for r in low_quality]
        
        high_mean = np.mean(high_values)
        low_mean = np.mean(low_values)
        
        # Calculate discriminative power (difference normalized by variance)
        high_var = np.var(high_values)
        low_var = np.var(low_values)
        combined_var = (high_var + low_var) / 2
        
        if combined_var > 0:
            discriminative_power = abs(high_mean - low_mean) / np.sqrt(combined_var)
            discriminative_features[feature] = {
                'high_mean': high_mean,
                'low_mean': low_mean,
                'discriminative_power': discriminative_power,
                'threshold': (high_mean + low_mean) / 2
            }
    
    return discriminative_features
```

**What it does**: Identifies which features best separate high-quality from low-quality levels using statistical analysis.

## ⚖️ **How Optimal Weights Are Learned**

### Weight Learning Process:
```python
def _learn_feature_weights(self, results: List[BacktestResult]) -> Dict[str, float]:
    # This would use machine learning to optimize weights
    # For now, return the configured weights
    return {
        'success_rate': self.config.success_rate_weight,
        'bounce_strength': self.config.bounce_strength_weight,
        'volume_confirmation': self.config.volume_confirmation_weight,
        'time_persistence': self.config.time_persistence_weight,
        'touch_frequency': self.config.touch_frequency_weight
    }
```

**Future Enhancement**: Will use machine learning (e.g., gradient descent) to optimize weights based on actual performance outcomes.

## 🔗 **How Clustering Parameters Are Adjusted**

### Quality-Based Parameter Adjustment:
```python
def _adjust_proximity_by_quality(self, levels: List[Dict]) -> float:
    # Calculate average quality
    qualities = [level.get('backtest_quality', 0.5) for level in levels]
    avg_quality = np.mean(qualities)
    
    # Higher quality levels can be clustered more tightly
    # Lower quality levels need more separation
    quality_factor = 0.5 + (avg_quality * 0.5)  # Range: 0.5 to 1.0
    
    adjusted_proximity = self.config.proximity_threshold * quality_factor
    
    return adjusted_proximity
```

**What it does**: Adjusts clustering tightness based on the quality distribution of levels.

## 🎯 **How Quality-Weighted Merging Works**

### Quality-Weighted Price Calculation:
```python
def _merge_cluster_backtesting_enhanced(self, cluster: List[SRLevel], data: pd.DataFrame, cluster_id: int, clustering_result) -> SRLevel:
    # Calculate weighted average price (weighted by strength and quality)
    total_weight = 0
    weighted_price = 0
    
    for level in cluster:
        # Weight by both strength and any backtesting quality score
        quality_score = getattr(level, 'backtest_quality', level.strength)
        weight = level.strength * quality_score
        weighted_price += level.price * weight
        total_weight += weight
    
    if total_weight > 0:
        final_price = weighted_price / total_weight
    else:
        final_price = sum(level.price for level in cluster) / len(cluster)
```

**What it does**: When merging clustered levels, gives more weight to levels with higher quality scores, resulting in better final level positions.