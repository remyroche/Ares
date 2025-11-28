# S/R Level Quality Improvement Recommendations

## Executive Summary

Analysis of the breakout/bounce regime classification reveals two critical issues:

1. **S/R levels are too weak** → High breakout rate (41.3%) vs low bounce rate (20.3%)
2. **Trap classification is economically misaligned** → Resistance traps show strong mean reversion (Sharpe 0.19) but aren't leveraged

## Issue 1: Weak S/R Level Selection

### Root Cause Analysis

**File:** `src/training/steps/market_analysis/ml_breakout_bounce_regime_step.py`
**Method:** `RollingKDELevelGenerator.compute_levels()` (lines 4695-4890)

**Current Implementation:**
```python
# Lines 4804-4819: Level selection is PURELY proximity-based
best_above = above_sorted[0][0] if above_sorted else None
best_below = below_sorted[0][0] if below_sorted else None

if best_above is not None and best_below is not None:
    # Selects closest level by distance, NO quality filtering
    if abs(float(best_above["price"]) - close_price) <= abs(float(best_below["price"]) - close_price):
        primary_level = best_above
    else:
        primary_level = best_below
```

**Problem:** Levels are selected by proximity alone, with **zero quality filtering**:
- ❌ No minimum `touch_count` required
- ❌ No minimum `volume_depth_ratio` required
- ❌ No `prominence` threshold
- ❌ No strength score filtering

**Result:** Weak, untested levels are included → They break easily (41.3% breakout rate)

### Recommended Solution

#### Phase 1: Add Mandatory Level Filtering

Insert filtering **before** selecting best_above/best_below:

```python
# After line 4768: Filter candidate_levels by quality
filtered_levels = []
for level in candidate_levels:
    # Calculate strength metrics
    touch_count = level.get("touch_count", 0)
    volume_depth = level.get("volume_depth_ratio", 0.0)
    prominence = level.get("prominence", 0.0)

    # Quality filters (configurable via config)
    min_touches = config.get("sr_min_touch_count", 2)
    min_volume_depth = config.get("sr_min_volume_depth_ratio", 0.8)
    min_prominence = config.get("sr_min_prominence", 0.5)

    # Apply filters
    if (touch_count >= min_touches and
        volume_depth >= min_volume_depth and
        prominence >= min_prominence):
        filtered_levels.append(level)

# Use filtered_levels instead of candidate_levels
if not filtered_levels:
    continue  # No strong levels found, skip this timestamp

# Then proceed with best_above/best_below selection
for level in filtered_levels:  # Changed from candidate_levels
    lp = float(level["price"])
    ...
```

#### Phase 2: Add Weighted Selection (Optional Enhancement)

Instead of pure proximity, use **strength-weighted distance**:

```python
def calculate_weighted_distance(level, close_price):
    """Combine proximity with strength for better level selection."""
    distance = abs(float(level["price"]) - close_price)
    strength = calculate_level_strength(level)  # 0-1 score

    # Lower effective distance for stronger levels
    weighted_distance = distance / (0.5 + strength)  # strength=1 → 67% discount
    return weighted_distance

# Apply weighted sorting
above_sorted = sorted(above, key=lambda x: calculate_weighted_distance(x[0], close_price))
below_sorted = sorted(below, key=lambda x: calculate_weighted_distance(x[0], close_price))
```

#### Phase 3: Add Strength Score Calculation

```python
def calculate_level_strength(level: Dict[str, Any]) -> float:
    """Calculate 0-1 strength score combining multiple factors."""
    touch_count = level.get("touch_count", 0)
    volume_depth = level.get("volume_depth_ratio", 0.0)
    prominence = level.get("prominence", 0.0)

    # Normalized components
    touch_score = min(touch_count / 5.0, 1.0)  # 5+ touches = max
    volume_score = min(volume_depth / 2.0, 1.0)  # 2x median vol = max
    prominence_score = min(prominence / 2.0, 1.0)  # normalized

    # Weighted combination
    strength = (
        0.40 * touch_score +      # 40% weight on testing
        0.35 * volume_score +     # 35% weight on volume
        0.25 * prominence_score   # 25% weight on prominence
    )

    return strength
```

### Recommended Default Thresholds

Based on typical market behavior:

```python
# Conservative (fewer but stronger levels)
sr_min_touch_count: 3
sr_min_volume_depth_ratio: 1.2
sr_min_prominence: 0.7

# Balanced (recommended)
sr_min_touch_count: 2
sr_min_volume_depth_ratio: 0.8
sr_min_prominence: 0.5

# Aggressive (more levels, accept weaker ones)
sr_min_touch_count: 1
sr_min_volume_depth_ratio: 0.5
sr_min_prominence: 0.3
```

---

## Issue 2: Missing Strength Indicators in ML Features

### Current State

**File:** `ml_breakout_bounce_regime_step.py`
**Method:** `_add_directional_edge_features()` (lines 5778-5890)

The code calculates a **combined strength score** (lines 5841-5880) but:
- ❌ Individual components aren't exposed to the model
- ❌ Model can't learn which strength aspects matter most for each regime
- ❌ Limits model's ability to differentiate strong vs weak levels

### Recommended Solution

Add individual strength metrics as ML features:

```python
def _add_sr_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add S/R level strength indicators for ML model."""

    out = df.copy()

    # 1. Raw strength components (already available in data)
    out["sr_touch_count"] = pd.to_numeric(
        out.get("primary_level_touch_count"), errors="coerce"
    ).fillna(0)

    out["sr_volume_depth_ratio"] = pd.to_numeric(
        out.get("primary_level_volume_depth_ratio"), errors="coerce"
    ).fillna(0)

    out["sr_prominence"] = pd.to_numeric(
        out.get("primary_level_prominence"), errors="coerce"
    ).fillna(0)

    # 2. Derived strength metrics
    # Level age (hours since first touch)
    if "primary_level_first_touch_ts" in out.columns:
        first_touch = pd.to_datetime(out["primary_level_first_touch_ts"])
        out["sr_age_hours"] = (out.index - first_touch).total_seconds() / 3600
        out["sr_age_log_hours"] = np.log1p(out["sr_age_hours"])

    # Touch recency (hours since last touch)
    if "primary_level_last_touch_ts" in out.columns:
        last_touch = pd.to_datetime(out["primary_level_last_touch_ts"])
        out["sr_recency_hours"] = (out.index - last_touch).total_seconds() / 3600
        out["sr_recency_log_hours"] = np.log1p(out["sr_recency_hours"])

    # 3. Normalized strength components
    out["sr_touch_score"] = np.clip(out["sr_touch_count"] / 5.0, 0, 1)
    out["sr_volume_score"] = np.clip(out["sr_volume_depth_ratio"] / 2.0, 0, 1)
    out["sr_prominence_score"] = np.clip(out["sr_prominence"] / 2.0, 0, 1)

    # 4. Combined strength score (existing logic, now explicit)
    out["sr_combined_strength"] = (
        0.40 * out["sr_touch_score"] +
        0.35 * out["sr_volume_score"] +
        0.25 * out["sr_prominence_score"]
    )

    # 5. Confidence flags
    out["sr_high_confidence"] = (
        (out["sr_touch_count"] >= 3) &
        (out["sr_volume_depth_ratio"] >= 1.0) &
        (out["sr_prominence"] >= 0.7)
    ).astype(int)

    out["sr_low_confidence"] = (
        (out["sr_touch_count"] < 2) |
        (out["sr_volume_depth_ratio"] < 0.6) |
        (out["sr_prominence"] < 0.3)
    ).astype(int)

    return out
```

**Integration Point:** Call this method in the main data preparation flow:

```python
# After line 5264 in _prepare_breakout_training_data
aligned_df = self._add_sr_strength_features(aligned_df)
```

### Expected Benefits

1. **Model can learn strength-regime relationships:**
   - "High touch_count → more bounces"
   - "Low prominence → more traps"
   - "High volume_depth → cleaner breakouts"

2. **Better regime predictions:**
   - Strong levels → Predict bounce with confidence
   - Weak levels → Predict breakout/trap more likely

3. **Interpretability:**
   - Feature importance shows which strength metrics matter
   - Can validate against domain knowledge

---

## Issue 3: Trap Classification Economic Misalignment

### Current Problem

**User Observation:**
```
Resistance, Regime 2 (Trap/Fakeout):
Mean return: 0.0173
Std: 0.091
Sharpe ≈ 0.190 (strongest region)
Intuition: traps at resistance (failed breaks) lead to strong mean reversion moves beneficial to longs.
-> if traps lead to reversion, why aren't they considered as bounces in our classification?
```

**Current Classification (lines 5560-5607):**

```python
# Regime 0 (Bounce): Price bounces WITHOUT breaking through
res_bounce = is_resistance & (down_move_cross >= bounce_move) & ~res_break

# Regime 2 (Trap): Price BREAKS through, THEN reverses
res_trap = is_resistance & (up_move_cross >= cross_buf) & (down_move_hold >= trap_revert)
```

**Key Difference:**
- **Bounce:** Never crosses 0.25% above resistance → Bounces down immediately
- **Trap:** Crosses 0.25%+ above → Then reverses back below

### Why They're Currently Different (Correct)

From a **trading execution** perspective, they're fundamentally different:

1. **Bounce:** Breakout traders never enter (no signal) → Safe for longs
2. **Trap:** Breakout traders enter (stop loss triggered) → Then price reverses → Mean reversion opportunity

**However**, the user is correct that **economically for longs, both are bullish**:
- Bounce: Price respects resistance, no breakout occurs
- Trap: Price briefly breaks, then **stronger reversion** occurs

### Recommended Solution: Add Directional Trap Features

Instead of merging traps with bounces (which would lose information), add features that capture **trap quality and directionality**:

```python
def _add_trap_reversion_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Add trap quality metrics to help model understand mean reversion potential."""

    out = df.copy()
    horizon = int(config.get("breakout_horizon_bars", 96))

    # Get forward price action for trap analysis
    fwd_high = out["high"].rolling(horizon, min_periods=1).max().shift(-horizon)
    fwd_low = out["low"].rolling(horizon, min_periods=1).min().shift(-horizon)
    fwd_close = out["close"].shift(-horizon)

    primary_level = out["primary_level_price"]
    is_resistance = out["is_resistance"]
    is_support = out["is_support"]

    # Calculate trap metrics
    # 1. Excursion depth (how far beyond level did price go?)
    resistance_excursion = (fwd_high - primary_level) / primary_level
    support_excursion = (primary_level - fwd_low) / primary_level

    out["trap_excursion_pct"] = np.where(
        is_resistance,
        resistance_excursion * 100,  # Positive = break above
        support_excursion * 100       # Positive = break below
    )

    # 2. Reversion strength (how far did it reverse back?)
    resistance_reversion = (fwd_high - fwd_close) / primary_level
    support_reversion = (fwd_close - fwd_low) / primary_level

    out["trap_reversion_pct"] = np.where(
        is_resistance,
        resistance_reversion * 100,   # Positive = reversed down
        support_reversion * 100        # Positive = reversed up
    )

    # 3. Trap efficiency (reversion / excursion ratio)
    # Values > 1.0 = full reversion + more
    # Values 0.5-1.0 = partial reversion
    # Values < 0.5 = weak reversion (real breakout)
    out["trap_efficiency"] = np.where(
        out["trap_excursion_pct"].abs() > 0.1,  # Avoid division by ~zero
        out["trap_reversion_pct"] / out["trap_excursion_pct"],
        0.0
    )

    # 4. Trap quality score (for ML model)
    # High score = strong mean reversion (trap worked well)
    # Low score = weak reversion (trap failed, breakout succeeded)
    out["trap_reversion_quality"] = np.clip(
        out["trap_efficiency"] * out["trap_reversion_pct"].abs() / 0.5,  # Normalize
        0, 1
    )

    # 5. Directional trap flags for regime analysis
    # Resistance trap with strong reversion = bullish for longs
    out["trap_resistance_bullish"] = (
        is_resistance &
        (out["trap_excursion_pct"] > 0.25) &      # Did break above
        (out["trap_reversion_pct"] > 0.25) &       # Did reverse down
        (out["trap_efficiency"] > 0.7)             # Strong reversion
    ).astype(int)

    # Support trap with strong reversion = bearish for longs, but BULLISH for shorts
    out["trap_support_bearish"] = (
        is_support &
        (out["trap_excursion_pct"] > 0.25) &
        (out["trap_reversion_pct"] > 0.25) &
        (out["trap_efficiency"] > 0.7)
    ).astype(int)

    return out
```

### Alternative: Reclassify Strong Traps as "Reversion Bounces"

If you want to explicitly merge economically similar regimes:

```python
# After initial regime labeling (line ~5607)
# Identify high-quality reversion traps
strong_reversion_mask = (
    (labels == 2) &  # Currently labeled as trap
    (out["trap_efficiency"] > 0.8) &  # Strong reversion
    (out["trap_reversion_pct"] > 0.3)  # Meaningful reversion distance
)

# Reclassify as bounces
labels[strong_reversion_mask] = 0  # Bounce regime

# Result: Model learns three regimes
# 0: Bounce (includes strong reversion traps)
# 1: Breakout (clean breaks that hold)
# 2: Weak Trap (ambiguous/choppy behavior)
```

### Recommended Approach

**Option 3 (Hybrid):** Keep 3 regimes, add trap quality features

1. ✅ Preserve trading distinction (bounce vs trap vs breakout)
2. ✅ Add `trap_reversion_quality` and `trap_efficiency` features
3. ✅ Let model learn: "Regime 2 + high trap_reversion_quality = long opportunity"
4. ✅ Downstream consumers can use probabilities + quality scores

**Benefits:**
- Model captures nuance (not all traps are equal)
- Trading logic can differentiate:
  - High-quality resistance traps → Long entry
  - Low-quality resistance traps → Avoid
- Better Sharpe ratios by using trap quality in position sizing

---

## Implementation Plan

### Phase 1: Quick Wins (Immediate Impact)

1. **Add S/R level filtering to `compute_levels()` method**
   - Lines: 4768-4770
   - Add: `filtered_levels` with min touch_count, volume_depth, prominence
   - Config: Add `sr_min_*` parameters with conservative defaults

2. **Add strength features to ML model**
   - Method: Create `_add_sr_strength_features()`
   - Integration: Call in `_prepare_breakout_training_data()`
   - Features: touch_count, volume_depth_ratio, prominence, age, recency

### Phase 2: Trap Enhancement (1-2 days)

3. **Add trap reversion quality features**
   - Method: Create `_add_trap_reversion_features()`
   - Features: excursion, reversion, efficiency, quality score
   - Integration: Call in feature preparation pipeline

4. **Update regime analysis reports**
   - Separate trap analysis by direction (resistance vs support)
   - Add trap quality distribution analysis
   - Show Sharpe by trap_efficiency quintiles

### Phase 3: Validation & Tuning (2-3 days)

5. **Retrain model with new features**
   - Compare regime distributions (expect more bounces, fewer breakouts)
   - Validate Sharpe improvements
   - Feature importance analysis

6. **Optimize thresholds via grid search**
   - Test combinations of sr_min_* thresholds
   - Target: 30-35% bounce, 25-30% breakout, 25-30% trap, 10-15% chop
   - Optimize for downstream Sharpe, not just balance

7. **Update documentation**
   - Document new features and their interpretation
   - Update regime classification guide
   - Add trap quality interpretation guide

---

## Expected Results

### Regime Distribution (After Implementation)

**Current (Weak Levels):**
- Regime 0 (Bounce): 20.3% ❌ Too low
- Regime 1 (Breakout): 41.3% ❌ Too high
- Regime 2 (Trap): 27.8% ✅ Reasonable
- Regime 3 (Chop): ~10.6% ✅ Reasonable

**Expected (Strong Levels):**
- Regime 0 (Bounce): 30-35% ✅ More realistic for strong levels
- Regime 1 (Breakout): 25-30% ✅ Cleaner breakouts
- Regime 2 (Trap): 25-30% ✅ Maintained
- Regime 3 (Chop): 10-15% ✅ Maintained

### Performance Metrics

**Current:**
- Resistance Trap Sharpe: 0.19 (untapped potential)
- Bounce identification accuracy: Low (due to weak levels)

**Expected:**
- Resistance Trap Sharpe: 0.25+ (with quality filtering)
- Bounce identification accuracy: +15-20% (stronger levels hold better)
- Overall downstream Sharpe: +0.05-0.10 improvement

---

## Configuration Parameters

Add to `ml_breakout_bounce_regime_step` config:

```python
# S/R Level Quality Filtering
sr_min_touch_count: 2                 # Minimum times level was tested
sr_min_volume_depth_ratio: 0.8        # Minimum volume confirmation (0.8 = 80% of median)
sr_min_prominence: 0.5                # Minimum KDE prominence score
sr_enable_weighted_selection: false   # Use strength-weighted distance (Phase 2)
sr_strength_weight: 0.5               # Weight for distance vs strength trade-off

# Trap Reversion Analysis
trap_min_excursion_pct: 0.25          # Minimum break distance to qualify as trap
trap_min_reversion_pct: 0.25          # Minimum reversion to qualify as strong trap
trap_strong_efficiency_threshold: 0.7  # Efficiency ratio for "strong" trap classification
trap_reclassify_strong_as_bounce: false  # Merge strong traps into bounce regime (Phase 3)

# Feature Engineering
enable_sr_strength_features: true      # Add individual strength components
enable_trap_quality_features: true     # Add trap reversion analysis features
```

---

## Files to Modify

1. **`src/training/steps/market_analysis/ml_breakout_bounce_regime_step.py`**
   - `RollingKDELevelGenerator.compute_levels()` (lines 4695-4890)
   - Add: `_add_sr_strength_features()` method
   - Add: `_add_trap_reversion_features()` method
   - Update: `_prepare_breakout_training_data()` to call new methods

2. **Config files** (if separate config exists)
   - Add new parameters for S/R filtering and trap analysis

3. **Documentation** (after validation)
   - Update regime interpretation guide
   - Add feature engineering documentation
   - Update training procedure docs

---

## Risk Mitigation

### Potential Issues

1. **Too aggressive filtering → insufficient data**
   - Mitigation: Start with balanced thresholds, monitor sample counts
   - Fallback: If <100 samples/day, relax thresholds dynamically

2. **Feature correlation → model confusion**
   - Mitigation: Monitor VIF (variance inflation factor) for strength features
   - Solution: Use hierarchical feature selection if needed

3. **Regime imbalance → biased predictions**
   - Mitigation: Use class weights in XGBoost training
   - Current: Already implemented (compute_sample_weight)

### Validation Checks

Before deployment:
- ✅ Regime distribution is balanced (15-40% each)
- ✅ Total samples > 10,000 (sufficient for training)
- ✅ Feature importance shows strength metrics are used
- ✅ OOF validation accuracy ≥ current baseline
- ✅ Downstream Sharpe maintains or improves

---

## Questions for Discussion

1. **Threshold Selection:** Should we use conservative, balanced, or aggressive defaults?
   - Recommendation: Start with balanced, allow config override

2. **Trap Reclassification:** Keep separate or merge strong traps with bounces?
   - Recommendation: Keep separate, add quality features (more information retained)

3. **Weighted Selection:** Implement Phase 2 strength-weighted distance now or later?
   - Recommendation: Later, after validating Phase 1 improvements

4. **Feature Set:** Include all proposed strength features or subset?
   - Recommendation: Include all, let model's feature importance guide pruning

---

## Conclusion

The high breakout rate (41.3%) and low bounce rate (20.3%) clearly indicate that S/R levels lack sufficient quality filtering. The three-phase implementation plan addresses this systematically:

1. **Phase 1:** Add quality filtering → More selective level detection → Stronger levels
2. **Phase 2:** Add strength features → Model learns quality-regime relationships
3. **Phase 3:** Add trap quality → Leverage mean reversion opportunities

Expected outcome: **More realistic regime distribution, better downstream performance, and utilization of the strong trap-reversion signal identified by the user.**

---

**Next Steps:**
1. Review and approve recommendations
2. Implement Phase 1 (level filtering + strength features)
3. Retrain and validate
4. Proceed to Phase 2/3 based on results
