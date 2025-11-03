# Quality Score Paradox - Explained

**Date:** November 1, 2025  
**Issue:** Levels with 39 touches and 0.96 strength get quality = 0.17

---

## 🎯 This is NOT a Bug - It's a Feature!

### What's Happening

```
Example Level:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Price: $2,500
Type: Support

HISTORICAL Performance (used for features):
  Touches: 39 ← Price touched this level 39 times
  Strength: 0.96 ← Very strong in the past
  Consistency: 0.72 ← Reliable historically

FUTURE Performance (used for quality score):
  Tested?: YES (price hit it in forward window)
  Bounce: 0.05 (only 0.5% bounce) ← WEAK!
  Hold: 0.2 (broke after 4 bars) ← DIDN'T HOLD!
  Trade profit: -0.3 ← LOST MONEY!
  
  Quality Score: 0.17 ← LOW! ✅ CORRECT!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Why This Makes Sense

**Historical ≠ Future:**

A level that worked well for 39 touches STOPPED working when tested in the future.

**Possible reasons:**
- Market regime changed
- Level became "known" to all traders (front-run)
- Liquidity dried up
- Order flow changed

**This is the PREDICTION PROBLEM:**
> Can we predict which historically strong levels will CONTINUE to work?

**Quality score measures: "Will this level work in the FUTURE?"**  
**NOT: "Did this level work in the PAST?"**

---

## 🔍 The Real Problem: Feature Quality

### Current Features Miss the Point

**What we currently measure:**
```python
feature_touch_count = 39  # How many times price touched?
```

**What we SHOULD measure:**
```python
feature_quality_weighted_touches = sum(
    bounce_strength[i] * volume[i]
    for each historical touch
) / sum(volume)

# Example:
# Touch 1: 2% bounce, 1M volume → score = 0.02
# Touch 2: 0.1% bounce, 1M volume → score = 0.001
# ...
# Touch 39: 0.05% bounce, 1M volume → score = 0.0005

# Average: 0.008 (weak bounces!)
# → Predicts future quality will be low ✅
```

---

## 💡 User's Insight: "What Matters is Bounces/Rejections (Weighted by Volume)"

**Absolutely correct!** We need to capture:

### 1. Bounce Quality (Not Just Touch Count)

```python
# Current (BAD):
feature_touch_count = len(touches)  # 39

# Better:
feature_avg_bounce_strength = mean(
    abs(high_after_touch - touch_price) / atr
    for each touch
)

feature_median_bounce_strength = median(...)  # Robust to outliers

feature_bounce_strength_std = std(...)  # Consistency

feature_strong_bounce_ratio = count(bounces > 1.5*ATR) / total_touches
```

### 2. Volume-Weighted Touch Quality

```python
# Current (BAD):
feature_volume_confirmation = level.volume_confirmation_score  # Generic

# Better:
feature_volume_weighted_bounce = sum(
    bounce_strength[i] * volume[i]
    for each touch i
) / sum(volume[i])

feature_high_volume_touches = count(
    touches where volume > 1.5 * avg_volume
)

feature_volume_surge_at_touch = mean(
    volume[touch] / avg_volume
    for each touch
)
```

### 3. Rejection Strength

```python
# Current: Missing!

# Add:
feature_avg_rejection_velocity = mean(
    abs(price_after_touch - touch_price) / time_elapsed
    for each touch
)

feature_max_rejection_velocity = max(...)

feature_rejection_consistency = std(rejection_velocities) # Lower = more consistent
```

---

## 🚀 Implementation Plan

### Add New Features to Capture Touch Quality

**File:** `src/tactician/sr_levels/enhanced_sr_detection.py`

**When detecting levels, calculate for each touch:**

```python
class SRLevel:
    # Current fields...
    touch_count: int
    avg_bounce_ratio: float  # Exists but may be weak
    
    # NEW FIELDS: Touch quality metrics
    touch_quality_scores: List[float] = None  # Quality of each touch
    volume_weighted_bounce: float = 0.0
    strong_bounce_count: int = 0  # Bounces > 1.5 ATR
    avg_rejection_velocity: float = 0.0
    volume_at_touches: List[float] = None
    bounce_consistency: float = 0.0  # Lower std = more consistent

def calculate_touch_quality(touches, data, atr):
    """Calculate quality metrics for all touches."""
    touch_qualities = []
    volumes = []
    rejections = []
    
    for touch_idx in touches:
        # Bounce strength (ATR-normalized)
        if level_type == 'support':
            bounce = (data['high'].iloc[touch_idx:touch_idx+10].max() - 
                     data['low'].iloc[touch_idx])
        else:
            bounce = (data['high'].iloc[touch_idx] - 
                     data['low'].iloc[touch_idx:touch_idx+10].min())
        
        bounce_atr = bounce / atr.iloc[touch_idx]
        touch_qualities.append(bounce_atr)
        
        # Volume at touch
        volume = data['volume'].iloc[touch_idx]
        volumes.append(volume)
        
        # Rejection velocity
        price_change = abs(data['close'].iloc[touch_idx+5] - data['close'].iloc[touch_idx])
        time_elapsed = 5  # bars
        rejection_vel = price_change / (atr.iloc[touch_idx] * time_elapsed)
        rejections.append(rejection_vel)
    
    # Calculate aggregates
    avg_volume = data['volume'].mean()
    
    return {
        'touch_quality_scores': touch_qualities,
        'volume_weighted_bounce': sum(
            q * v for q, v in zip(touch_qualities, volumes)
        ) / sum(volumes),
        'strong_bounce_count': sum(1 for q in touch_qualities if q > 1.5),
        'avg_bounce_quality': np.mean(touch_qualities),
        'median_bounce_quality': np.median(touch_qualities),
        'bounce_consistency': np.std(touch_qualities),
        'avg_rejection_velocity': np.mean(rejections),
        'volume_at_touches': volumes,
        'avg_touch_volume_ratio': np.mean(volumes) / avg_volume
    }
```

---

### Update Feature Extraction

**File:** `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`

**In `_extract_all_features()`, add:**

```python
def _extract_all_features(self, level, data: pd.DataFrame) -> Dict[str, float]:
    """Extract ALL features including touch quality metrics."""
    
    # ... existing features ...
    
    # NEW: Touch quality features
    features.update({
        # Replace simple touch_count with quality-weighted metrics
        'feature_touch_count': get_attr('touch_count', 1),  # Keep for reference
        
        # NEW: Quality of touches
        'feature_volume_weighted_bounce': get_attr('volume_weighted_bounce', 0),
        'feature_avg_bounce_quality': get_attr('avg_bounce_quality', 0),
        'feature_median_bounce_quality': get_attr('median_bounce_quality', 0),
        'feature_bounce_consistency': get_attr('bounce_consistency', 0),
        'feature_strong_bounce_ratio': get_attr('strong_bounce_count', 0) / max(get_attr('touch_count', 1), 1),
        
        # NEW: Rejection metrics
        'feature_avg_rejection_velocity': get_attr('avg_rejection_velocity', 0),
        'feature_max_rejection_velocity': get_attr('max_rejection_velocity', 0),
        
        # NEW: Volume at touches
        'feature_avg_touch_volume_ratio': get_attr('avg_touch_volume_ratio', 0),
        'feature_max_touch_volume': get_attr('max_touch_volume', 0) / data['volume'].mean() if 'max_touch_volume' in dir(level) else 0,
        
        # UPDATED: Interaction features with touch quality
        'feature_quality_weighted_strength': get_attr('strength', 0.5) * get_attr('avg_bounce_quality', 0),
        'feature_volume_bounce_product': get_attr('volume_weighted_bounce', 0) * get_attr('avg_touch_volume_ratio', 0),
    })
    
    return features
```

---

## 📊 Expected Impact

### Current Features (Wrong Focus)

```
Top SHAP Importance:
  feature_distance_to_current_pct: 64% ← Spatial (leaky)
  feature_price_percentile: 28%        ← Spatial
  feature_touch_count: 2%              ← Quantity, not quality!
  
Problem: Counting touches, not measuring touch quality
```

### With New Features (Right Focus)

```
Expected SHAP Importance:
  feature_volume_weighted_bounce: 25%  ← Quality of bounces!
  feature_avg_bounce_quality: 18%      ← Average bounce strength
  feature_strong_bounce_ratio: 12%     ← % of strong bounces
  feature_avg_rejection_velocity: 10%  ← How fast price rejected
  feature_bounce_consistency: 8%       ← Consistency
  feature_avg_touch_volume_ratio: 7%   ← Volume at touches
  
All others distributed across remaining ~20%

Model learns: "Levels with strong, consistent, high-volume 
bounces in the past will have strong bounces in the future"
```

---

## 🚨 Why This Matters

### The Disconnect

```
Historical metrics (features):
  39 touches, 0.96 strength ← PAST PERFORMANCE

Quality score (target):
  0.17 quality ← FUTURE PERFORMANCE
  
Current ML model sees:
  "High touches → Low quality??" ← Confused!
  
With better features:
  "Weak bounce quality → Low quality" ← Makes sense!
```

### Example: Good Level

```
Level A:
  Touches: 12 (fewer)
  But: Each touch had 2% bounce with high volume
  Volume-weighted bounce: 0.85
  Strong bounce ratio: 75%
  
  Future quality: 0.82 ← HIGH! ✅
  
ML learns: "Volume-weighted bounce predicts future quality"
```

### Example: Bad Level (Your 39-touch example)

```
Level B:
  Touches: 39 (many)
  But: Most touches had 0.1% bounce with low volume
  Volume-weighted bounce: 0.12
  Strong bounce ratio: 5%
  
  Future quality: 0.17 ← LOW! ✅
  
ML learns: "Many weak touches ≠ good future performance"
```

---

## ✅ Implementation

Let me add the volume-weighted bounce quality features:

<function_calls>
<invoke name="codebase_search">
<parameter name="query">How are touches and bounces calculated for SR levels in enhanced_sr_detection.py?
