# SR Pipeline Improvements Analysis

## Executive Summary
Analysis of the current SR detection pipeline focusing on:
1. **Strength×Prominence Filtering** - How levels are ranked and filtered
2. **ML Feature Enhancement** - How ML-ready features are added to SR levels

---

## 1. Strength×Prominence Filtering Analysis

### Current Implementation
Located in: `src/tactician/sr_levels/enhanced_sr_detection.py:4282-4381`

**Current Approach:**
```python
composite_score = strength_component × prominence_component
```

**Current Prominence Calculation:**
- **For Resistance**: Uses `scipy.signal.peak_prominences` with wlen=20
- **For Support**: Simplified calculation: `strength × (price_range × 0.1)`
- Normalized by price range

**Current Width Calculation:**
- Uses `scipy.signal.peak_widths` with rel_height=0.5
- **Problem**: Width is calculated but NOT used in composite score

### Issues Identified

#### 1.1 Asymmetric Support/Resistance Treatment
- ❌ **Resistance levels** get proper prominence calculation using scipy
- ❌ **Support levels** use simplified heuristic: `strength × (price_range × 0.1)`
- **Impact**: Support levels may be under/over-valued relative to resistance

#### 1.2 Width Score Ignored
- ✅ Width is calculated
- ❌ Width is NOT incorporated into composite_score
- **Impact**: Missing important zone width information that indicates level strength

#### 1.3 Simple Multiplicative Model
- Current: `score = strength × prominence`
- **Limitations**:
  - No consideration for interaction effects
  - No weighting flexibility
  - Missing other important dimensions (volume, age, consistency)

#### 1.4 Prominence Calculation Limitations
- Fixed window length (`wlen=20`) regardless of market volatility
- No consideration of local vs global prominence
- Support prominence is essentially a duplicate of strength

---

## 2. ML Feature Enhancement Analysis

### Current Implementation
Located in: `src/tactician/sr_levels/enhanced_sr_detection.py:1129-1197`

**Current Features Added:**
```python
1. dist_to_level_atr          # ATR-normalized distance from current price
2. break_success_rate         # Fraction of touches → breakouts
3. persistence_score          # Time since formation without breach
4. multi_tf_support          # Simulated multi-timeframe confirmation
5. avg_reaction_atr          # Mean reaction normalized by ATR
6. time_since_last_touch     # Bars since last touch
7. volume_at_level           # Alias for volume_confirmation_score
8. prominence_score          # From scipy.signal
9. width_score               # From scipy.signal
```

### Issues Identified

#### 2.1 Multi-Timeframe Support is Simulated
```python
# Currently in _calculate_multi_tf_support (line 965-1004)
# NOT actual multi-timeframe analysis, just heuristic scoring
support_score = 0
if level.strength > 0.7: support_score += 1
if level.age_bars > 100: support_score += 1
# etc...
```
- **Impact**: Feature name is misleading; not true multi-TF analysis

#### 2.2 Missing Important Features
Notable gaps:
- ❌ **Price velocity at level** - How fast price approaches/leaves the level
- ❌ **Rejection velocity** - Speed of bounces
- ❌ **Level clustering density** - Confluence with nearby levels  
- ❌ **Historical win rate** - Past success of trades at this level
- ❌ **Recency-weighted strength** - Recent touches weighted more
- ❌ **Volatility-adjusted metrics** - Context-aware features
- ❌ **Order flow imbalance** - If order book data available
- ❌ **Time-of-day effects** - Session-based statistics
- ❌ **Regime-conditional features** - Different in trending vs ranging

#### 2.3 Feature Quality Issues
- `volume_at_level` is just an alias for `volume_confirmation_score` (line 1187)
- No feature normalization/standardization
- No handling of missing/invalid values beyond try/except
- No feature interaction terms (e.g., strength × volume)

#### 2.4 Limited Context Features
- No market regime awareness
- No volatility state
- No trend strength context
- No order book depth (if available)

---

## 3. Recommended Improvements

### 3.1 Enhanced Strength×Prominence Filtering

#### Option A: Weighted Composite Score (Simple & Effective)
```python
composite_score = (
    α₁ × strength +
    α₂ × prominence_normalized +
    α₃ × width_normalized +
    α₄ × volume_confirmation +
    α₅ × consistency_score +
    α₆ × recency_factor
)
```

**Benefits:**
- Incorporates width (currently ignored)
- Adds volume and consistency dimensions
- Weights can be learned via optimization
- Recency factor prevents stale levels

**Suggested Weights (initial):**
- α₁ = 0.30 (strength)
- α₂ = 0.25 (prominence)
- α₃ = 0.15 (width)
- α₄ = 0.15 (volume)
- α₅ = 0.10 (consistency)
- α₆ = 0.05 (recency)

#### Option B: ML-Based Ranking (Advanced)
Train a LightGBM/XGBoost model to predict level "quality":

**Features for quality model:**
- All existing SRLevel attributes
- Prominence, width, strength
- Volume metrics
- Age and recency
- Touch patterns
- Confluence with other levels

**Target label:**
- Historical success rate of trades at level
- Or: Level persistence in future data
- Or: Price reaction magnitude

**Benefits:**
- Automatically learns optimal weighting
- Can capture non-linear interactions
- Adapts to market conditions
- Can be regime-specific

#### Option C: Hybrid Approach (Recommended)
1. Use weighted composite (Option A) as baseline
2. Train ML model (Option B) to refine rankings
3. Ensemble the scores:
   ```python
   final_score = 0.6 × weighted_composite + 0.4 × ml_score
   ```

### 3.2 Fix Support/Resistance Asymmetry

**Current Problem:**
```python
# Support uses heuristic
if level_type == 'support':
    prominence = level.strength * (price_range * 0.1)
```

**Proposed Fix:**
```python
def _calculate_level_prominence_unified(self, level, data, level_type, price_range):
    """Unified prominence for both support and resistance."""
    
    if level_type == 'support':
        price_data = -data['low'].values  # Invert for valleys
    else:
        price_data = data['high'].values
    
    closest_idx = np.argmin(np.abs(np.abs(price_data) - level.price))
    
    try:
        # Use scipy for both types
        prominences = peak_prominences(price_data, [closest_idx], wlen=adaptive_wlen)
        prominence = prominences[0][0]
    except:
        prominence = level.strength * (price_range * 0.1)
    
    return prominence / price_range
```

**Key change:** Invert support data to treat valleys as peaks

### 3.3 Adaptive Window Lengths

**Current:** Fixed `wlen=20` for prominence/width

**Proposed:** Adaptive based on volatility
```python
def _get_adaptive_window(self, data):
    """Calculate adaptive window based on volatility regime."""
    atr = self._calculate_atr(data)
    volatility_ratio = atr.iloc[-1] / atr.mean()
    
    if volatility_ratio > 1.5:  # High volatility
        return 30  # Wider window
    elif volatility_ratio < 0.7:  # Low volatility
        return 15  # Narrower window
    else:
        return 20  # Normal
```

### 3.4 Enhanced ML Features

#### Add These Features:

**A. Price Dynamics Features**
```python
# 1. Approach velocity
approach_velocity = self._calculate_approach_velocity(level, data)
# Measures: How fast price moved toward level in recent bars

# 2. Rejection velocity  
rejection_velocity = self._calculate_rejection_velocity(level, data)
# Measures: Average bounce speed from this level

# 3. Dwell time
dwell_time = self._calculate_dwell_time(level, data)
# Measures: Average time price spends near level (consolidation)
```

**B. Clustering/Confluence Features**
```python
# 4. Nearby level count
nearby_count = self._count_levels_within_atr(level, all_levels, atr, distance=0.5)
# Measures: Confluence - more nearby levels = stronger zone

# 5. Cluster density
cluster_density = self._calculate_cluster_density(level, all_levels, atr)
# Measures: Density of SR levels around this level

# 6. Fibonacci confluence
fib_confluence = self._check_fibonacci_confluence(level, data)
# Measures: Proximity to Fibonacci retracement levels
```

**C. Temporal Features**
```python
# 7. Recency-weighted strength
recency_strength = self._calculate_recency_weighted_strength(level, data)
# Recent touches weighted exponentially higher

# 8. Touch frequency
touch_frequency = level.touch_count / level.age_bars
# Touches per bar (normalized)

# 9. Time since formation (normalized)
formation_recency = (current_time - level.formation_time) / pd.Timedelta(days=1)

# 10. Breach recovery rate
breach_recovery = self._calculate_breach_recovery(level, data)
# How often price returns after breaking level
```

**D. Context Features**
```python
# 11. Volatility regime
volatility_regime = self._get_volatility_regime(data)  # low/med/high
# Levels behave differently in different volatility regimes

# 12. Trend regime
trend_regime = self._get_trend_regime(data)  # strong_up/weak_up/ranging/weak_down/strong_down
# Support stronger in uptrends, resistance in downtrends

# 13. Volume profile at level
volume_profile = self._calculate_volume_profile_at_level(level, data)
# Volume distribution when price was at this level

# 14. Session effectiveness
session_stats = self._calculate_session_statistics(level, data)
# Does level work better in certain sessions? (Asia/London/NY)
```

**E. Interaction Features**
```python
# 15. Strength × Volume interaction
strength_volume = level.strength * level.volume_confirmation_score

# 16. Prominence × Age interaction
prominence_age = level.prominence_score * np.log1p(level.age_bars)

# 17. Touch consistency
touch_consistency = level.touch_count / (1 + level.failure_count)
```

**F. Statistical Features**
```python
# 18. Z-score of level price
price_zscore = (level.price - data['close'].mean()) / data['close'].std()
# Where level sits in price distribution

# 19. Percentile rank
percentile_rank = percentileofscore(data['close'], level.price) / 100
# 0-1 range

# 20. Distance to key MAs
distance_to_ma20 = abs(level.price - data['close'].rolling(20).mean().iloc[-1]) / atr
distance_to_ma50 = abs(level.price - data['close'].rolling(50).mean().iloc[-1]) / atr
# Confluence with moving averages
```

### 3.5 Feature Engineering Pipeline

**Proposed Structure:**
```python
class EnhancedSRFeatureEngineer:
    """Enhanced feature engineering for SR levels."""
    
    def __init__(self):
        self.feature_groups = {
            'basic': self._extract_basic_features,
            'dynamics': self._extract_dynamics_features,
            'clustering': self._extract_clustering_features,
            'temporal': self._extract_temporal_features,
            'context': self._extract_context_features,
            'interaction': self._extract_interaction_features,
            'statistical': self._extract_statistical_features
        }
    
    def enhance_levels(self, levels, data, config):
        """Add all feature groups to levels."""
        for level in levels:
            for group_name, extractor in self.feature_groups.items():
                if config.get(f'enable_{group_name}_features', True):
                    features = extractor(level, data, levels)
                    self._add_features_to_level(level, features, prefix=group_name)
        
        return levels
    
    def _extract_dynamics_features(self, level, data, all_levels):
        """Extract price dynamics features."""
        return {
            'approach_velocity': self._calc_approach_velocity(level, data),
            'rejection_velocity': self._calc_rejection_velocity(level, data),
            'dwell_time': self._calc_dwell_time(level, data),
            'reaction_strength': self._calc_reaction_strength(level, data)
        }
    
    # ... implement other extractors
```

### 3.6 Feature Validation & Selection

**Add Feature Quality Checks:**
```python
def validate_features(self, levels):
    """Validate feature quality."""
    features_df = self._levels_to_dataframe(levels)
    
    issues = []
    
    # Check for NaN/Inf
    nan_cols = features_df.columns[features_df.isna().any()].tolist()
    if nan_cols:
        issues.append(f"NaN values in: {nan_cols}")
    
    # Check for zero variance
    zero_var = features_df.columns[features_df.std() < 1e-10].tolist()
    if zero_var:
        issues.append(f"Zero variance in: {zero_var}")
    
    # Check for high correlation (multicollinearity)
    corr_matrix = features_df.corr().abs()
    high_corr = (corr_matrix > 0.95) & (corr_matrix < 1.0)
    if high_corr.any().any():
        issues.append(f"High correlation detected")
    
    return issues
```

---

## 4. Implementation Priority

### Phase 1 (Quick Wins - 1-2 days)
1. ✅ Fix support/resistance prominence asymmetry
2. ✅ Incorporate width_score into composite score
3. ✅ Implement weighted composite score (Option A)
4. ✅ Add top 10 missing features (approach_velocity, cluster_density, etc.)

### Phase 2 (Medium - 3-5 days)
5. ⏳ Implement adaptive window lengths
6. ⏳ Add all context features (regime, volatility, trend)
7. ⏳ Add interaction features
8. ⏳ Implement feature validation pipeline

### Phase 3 (Advanced - 1-2 weeks)
9. 🔮 Train ML-based quality model (Option B)
10. 🔮 Implement hybrid scoring (Option C)
11. 🔮 Add session-based statistics
12. 🔮 Multi-timeframe analysis (real, not simulated)

---

## 5. Expected Impact

### Strength×Prominence Improvements
- **Better level selection**: Width + volume + consistency incorporated
- **Fair treatment**: Support and resistance treated symmetrically
- **Adaptability**: Adaptive windows handle different market conditions
- **Fewer false levels**: Better filtering of weak/irrelevant levels

### ML Feature Improvements
- **Better predictions**: 20+ new informative features
- **Context awareness**: Regime and volatility-adjusted features
- **Reduced overfitting**: Interaction terms capture non-linear effects
- **More robust**: Feature validation catches data issues early

### Quantitative Targets
- **Precision increase**: +15-25% in identifying tradeable levels
- **Recall maintenance**: Keep 95%+ of truly important levels
- **False positive reduction**: -30-40% fewer weak levels making it through
- **Feature importance**: Expect 5-10 features to dominate model performance

---

## 6. Testing Strategy

### Unit Tests
```python
def test_prominence_symmetry():
    """Ensure support and resistance use same calculation method."""
    # Create synthetic data with obvious support/resistance
    # Verify prominence scores are comparable
    
def test_width_incorporation():
    """Ensure width_score affects composite_score."""
    # Create levels with different widths but same strength
    # Verify wider levels score higher
    
def test_feature_validity():
    """Ensure all features are valid numbers."""
    # Extract features from sample data
    # Check for NaN, Inf, extreme values
```

### Integration Tests
```python
def test_end_to_end_filtering():
    """Test full pipeline on historical data."""
    # Load historical data
    # Run SR detection with new filtering
    # Compare level quality metrics vs old approach
    
def test_performance_regression():
    """Ensure improvements don't slow down system."""
    # Benchmark old vs new implementation
    # Ensure <20% slowdown (acceptable for quality gain)
```

### Backtesting Validation
```python
def validate_level_quality():
    """Validate that higher-scored levels perform better."""
    # For historical data:
    #   1. Detect SR levels with scores
    #   2. Simulate trades at each level
    #   3. Measure win rate, profit factor, etc.
    #   4. Verify correlation: higher score → better performance
```

---

## 7. Configuration Recommendations

### Add to SR Config
```yaml
sr_detection:
  filtering:
    method: "weighted_composite"  # "simple_multiplicative", "weighted_composite", "ml_based", "hybrid"
    
    # Weighted composite weights
    weights:
      strength: 0.30
      prominence: 0.25
      width: 0.15
      volume: 0.15
      consistency: 0.10
      recency: 0.05
    
    # Adaptive windows
    adaptive_windows: true
    window_volatility_adjustment: true
    base_window: 20
    min_window: 10
    max_window: 50
    
  ml_features:
    # Feature groups to enable
    enable_basic_features: true
    enable_dynamics_features: true
    enable_clustering_features: true
    enable_temporal_features: true
    enable_context_features: true
    enable_interaction_features: true
    enable_statistical_features: true
    
    # Feature engineering config
    normalize_features: true
    handle_missing: "median"  # "median", "mean", "zero", "drop"
    
    # Feature selection
    feature_selection: true
    max_features: 50  # Prevent feature explosion
    min_feature_importance: 0.01  # Drop low-importance features
```

---

## 8. Code Locations

### Files to Modify
1. **`src/tactician/sr_levels/enhanced_sr_detection.py`**
   - `_apply_unified_strength_prominence_filtering` (line 4282)
   - `_calculate_level_prominence_simple` (line 866)
   - `_calculate_level_width` (line 936)
   - `_enhance_levels_with_ml_features` (line 1129)

2. **`src/tactician/sr_levels/sr_modules/sr_feature_extractor.py`**
   - Add new feature extraction methods

3. **Create new files:**
   - `src/tactician/sr_levels/sr_modules/sr_feature_engineer.py`
   - `src/tactician/sr_levels/sr_modules/sr_quality_model.py`

---

## 9. Monitoring & Metrics

### Metrics to Track
```python
metrics = {
    # Level quality
    'avg_composite_score': ...,
    'score_distribution': ...,
    'levels_filtered_pct': ...,
    
    # Feature statistics
    'feature_coverage': ...,  # % non-null
    'feature_correlation_max': ...,
    'feature_importance_top10': ...,
    
    # Performance
    'detection_time_ms': ...,
    'feature_extraction_time_ms': ...,
    'memory_usage_mb': ...,
    
    # Validation (if backtesting)
    'level_precision': ...,  # % of kept levels that are tradeable
    'level_recall': ...,  # % of good levels that were kept
    'avg_level_win_rate': ...
}
```

### Logging Recommendations
```python
self.logger.info(f"🎯 Filtering: {len(levels)} → {len(filtered)} levels")
self.logger.info(f"📊 Composite score range: {min_score:.3f} - {max_score:.3f}")
self.logger.info(f"✨ Top weights: strength={w1:.2f}, prominence={w2:.2f}, width={w3:.2f}")
self.logger.info(f"🔧 Features: {num_features} extracted, {num_valid} valid, {num_selected} selected")
```

---

## 10. Next Steps

### Immediate Actions
1. **Review this document** with team
2. **Prioritize improvements** - which ones give most value?
3. **Prototype Phase 1** changes on sample data
4. **Validate improvements** using backtesting
5. **Roll out incrementally** - don't change everything at once

### Questions to Answer
- Do we have order book data for order flow features?
- What's our target for level count after filtering?
- Do we want regime-specific models or unified?
- Should we optimize weights via grid search or genetic algorithm?
- How do we handle the cold start problem for ML-based scoring?

---

## Appendix A: Pseudocode for Key Improvements

### A.1 Weighted Composite Score
```python
def calculate_weighted_composite_score(level, data, weights):
    """Calculate weighted composite score."""
    
    # Normalize all components to [0, 1]
    strength_norm = level.strength  # Already 0-1
    
    prominence_norm = normalize(
        level.prominence_score,
        min_val=0,
        max_val=data['price_range'] * 0.5
    )
    
    width_norm = normalize(
        level.width_score,
        min_val=1,
        max_val=50
    )
    
    volume_norm = level.volume_confirmation_score  # Already 0-1
    
    consistency_norm = level.consistency_score  # Already 0-1
    
    # Recency factor: exponential decay
    age_days = (current_time - level.last_touch_time).days
    recency_factor = np.exp(-age_days / 30)  # Half-life of 30 days
    
    # Weighted sum
    composite = (
        weights['strength'] * strength_norm +
        weights['prominence'] * prominence_norm +
        weights['width'] * width_norm +
        weights['volume'] * volume_norm +
        weights['consistency'] * consistency_norm +
        weights['recency'] * recency_factor
    )
    
    return composite
```

### A.2 Symmetric Prominence Calculation
```python
def calculate_prominence_unified(level, data, level_type, price_range):
    """Unified prominence calculation for support and resistance."""
    
    # Get appropriate price data
    if level_type == 'support':
        # For support (valleys), invert the data
        price_data = -data['low'].values
        search_price = -level.price
    else:
        # For resistance (peaks), use as-is  
        price_data = data['high'].values
        search_price = level.price
    
    # Find closest point
    closest_idx = np.argmin(np.abs(price_data - search_price))
    
    # Adaptive window
    atr = calculate_atr(data)
    volatility_regime = get_volatility_regime(atr)
    wlen = get_adaptive_window(volatility_regime)
    
    # Calculate prominence
    try:
        from scipy.signal import peak_prominences
        prominences, _, _ = peak_prominences(
            price_data,
            [closest_idx],
            wlen=wlen
        )
        prominence = abs(prominences[0])
    except:
        # Fallback
        prominence = level.strength * (price_range * 0.1)
    
    # Normalize by price range
    prominence_normalized = prominence / price_range
    
    return prominence_normalized
```

### A.3 Feature Extraction Example
```python
def extract_dynamics_features(level, data, atr):
    """Extract price dynamics features."""
    
    features = {}
    
    # 1. Approach velocity
    # Find recent approaches to level
    threshold = level.price * 0.005  # 0.5% threshold
    touches_mask = abs(data['close'] - level.price) < threshold
    touch_indices = np.where(touches_mask)[0]
    
    if len(touch_indices) > 0:
        velocities = []
        for touch_idx in touch_indices:
            if touch_idx >= 5:  # Need lookback
                # Calculate velocity as price change over 5 bars before touch
                price_before = data['close'].iloc[touch_idx - 5]
                price_at = data['close'].iloc[touch_idx]
                velocity = abs(price_at - price_before) / (5 * atr.iloc[touch_idx])
                velocities.append(velocity)
        
        features['approach_velocity'] = np.mean(velocities) if velocities else 0.0
    else:
        features['approach_velocity'] = 0.0
    
    # 2. Rejection velocity (bounce speed)
    if len(touch_indices) > 0:
        rejection_velocities = []
        for touch_idx in touch_indices:
            if touch_idx < len(data) - 5:  # Need lookforward
                # Calculate bounce speed over 5 bars after touch
                price_at = data['close'].iloc[touch_idx]
                price_after = data['close'].iloc[touch_idx + 5]
                
                if level.type == 'support':
                    # Bounce up from support
                    bounce = max(0, price_after - price_at)
                else:
                    # Bounce down from resistance
                    bounce = max(0, price_at - price_after)
                
                velocity = bounce / (5 * atr.iloc[touch_idx])
                rejection_velocities.append(velocity)
        
        features['rejection_velocity'] = np.mean(rejection_velocities) if rejection_velocities else 0.0
    else:
        features['rejection_velocity'] = 0.0
    
    # 3. Dwell time
    # Average bars spent near level
    if len(touch_indices) > 0:
        dwell_times = []
        i = 0
        while i < len(touch_indices):
            # Count consecutive touches as one dwell period
            start_idx = touch_indices[i]
            end_idx = start_idx
            while i < len(touch_indices) - 1 and touch_indices[i+1] == touch_indices[i] + 1:
                end_idx = touch_indices[i+1]
                i += 1
            dwell_times.append(end_idx - start_idx + 1)
            i += 1
        
        features['dwell_time'] = np.mean(dwell_times)
    else:
        features['dwell_time'] = 0.0
    
    return features
```

---

## Summary

The current SR pipeline has a solid foundation but can be significantly improved:

1. **Strength×Prominence Filtering** needs:
   - Symmetric treatment of support/resistance
   - Incorporation of width score
   - Multi-dimensional composite scoring
   - Adaptive parameters

2. **ML Feature Enhancement** needs:
   - 20+ new informative features
   - True multi-timeframe analysis (not simulated)
   - Context-aware features (regime, volatility)
   - Feature validation and selection

Implementing these improvements should increase precision by 15-25% while maintaining high recall, resulting in better level selection and more profitable trading signals.

