# SR Hierarchical HPO - Parameter Groups Explained

**Total Groups**: 5  
**Optimization Strategy**: Coarse Grid → Fine Grid → TPE Bayesian

---

## Parameter Groups Overview

### Group 1: Core Detection ⚡ (Priority 1 - Most Impactful)
**Purpose**: What gets detected as SR levels

| Parameter | Type | Range | Default | Meaning |
|-----------|------|-------|---------|---------|
| `min_touches` | int | 2-5 | 2 | Minimum # of price touches required |
| `strength_threshold` | float | 0.3-0.8 | 0.5 | Minimum strength to be a valid level |

**Why First?** These parameters control WHAT gets detected. If these are wrong, everything downstream fails.

**Example**:
- `min_touches=2`: Detects weak levels (many false positives)
- `min_touches=5`: Detects only very strong levels (may miss valid ones)
- HPO finds the optimal balance

---

### Group 2: Quality Filtering 🔍 (Priority 2)
**Purpose**: Filter out low-quality detected levels

| Parameter | Type | Range | Default | Meaning |
|-----------|------|-------|---------|---------|
| `distance_threshold` | float | 0.005-0.03 | 0.01 | How close prices must be to cluster |
| `volume_threshold` | float | 0.5-2.0 | 1.0 | Minimum volume confirmation required |

**Depends On**: Group 1 (needs detected levels to filter)

**Example**:
- `distance_threshold=0.005`: Very strict clustering (only exact hits)
- `distance_threshold=0.03`: Lenient clustering (wider tolerance)
- HPO finds optimal filtering tightness

---

### Group 3: Temporal Lookback ⏰ (Priority 3)
**Purpose**: How far back to search for patterns

| Parameter | Type | Range | Default | Meaning |
|-----------|------|-------|---------|---------|
| `lookback_periods` | int | 20-100 | 50 | Number of bars to look back |

**Depends On**: Group 1

**Example**:
- `lookback_periods=20`: Only recent patterns (may miss longer-term levels)
- `lookback_periods=100`: Deep historical search (computationally expensive)
- HPO finds balance between completeness and speed

---

### Group 4: Market Context 📊 (Priority 4)
**Purpose**: Trend and breakout refinement

| Parameter | Type | Range | Default | Meaning |
|-----------|------|-------|---------|---------|
| `trend_strength_threshold` | float | 0.3-0.7 | 0.5 | Minimum trend strength to consider |
| `breakout_threshold` | float | 0.01-0.05 | 0.02 | Price change % to confirm breakout |

**Depends On**: Groups 1 + 2

**Example**:
- `breakout_threshold=0.01`: Very sensitive (many false breakouts)
- `breakout_threshold=0.05`: Less sensitive (may miss real breakouts)
- HPO optimizes sensitivity

---

### Group 5: Strength Weights 🎯 (Priority 5) **NEW!**
**Purpose**: Optimize strength calculation formula

#### Positive Boosts (7 parameters)
| Parameter | Type | Range | Default | Meaning |
|-----------|------|-------|---------|---------|
| `touch_weight` | float | 0.05-0.3 | 0.1 | How much each touch adds |
| `volume_weight` | float | 0.1-0.4 | 0.2 | Volume confirmation importance |
| `consistency_weight` | float | 0.1-0.4 | 0.2 | Regular pattern importance |
| `confluence_weight` | float | 0.05-0.2 | 0.1 | Multiple method agreement |
| `pivot_boost` | float | 0.05-0.2 | 0.1 | Pivot point bonus |
| `psychological_boost` | float | 0.02-0.1 | 0.05 | Round number bonus |
| `hvn_boost` | float | 0.05-0.2 | 0.1 | High Volume Node bonus |

#### Negative Penalties (3 parameters)
| Parameter | Type | Range | Default | Meaning |
|-----------|------|-------|---------|---------|
| `failure_penalty_base` | float | 0.1-0.5 | 0.2 | Base penalty per breakout |
| `failure_volume_multiplier` | float | 1.0-2.5 | 1.5 | Volume scaling strength |
| `failure_max_penalty` | float | 0.4-1.0 | 0.6 | Maximum total penalty |

**Depends On**: Groups 1 + 2 (needs levels with quality scores)

**Example**:
- `touch_weight=0.15`: Touches matter more
- `failure_penalty_base=0.3`: More punishment for breakouts
- `failure_volume_multiplier=2.5`: Heavily penalize low-volume breakouts
- HPO finds optimal feature importance

---

## Hierarchical Optimization Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: COARSE GRID (Broad Exploration)                   │
├─────────────────────────────────────────────────────────────┤
│  Group 1: Core Detection     (4 points/param = 16 combos)  │
│  ↓                                                           │
│  Group 2: Quality Filtering  (4 points/param = 16 combos)  │
│  ↓                                                           │
│  Group 3: Temporal Lookback  (4 points/param = 4 combos)   │
│  ↓                                                           │
│  Group 4: Market Context     (4 points/param = 16 combos)  │
│  ↓                                                           │
│  Group 5: Strength Weights   (4 points/param = millions*)  │
│                                                              │
│  *Sampled intelligently using stratified sampling           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: FINE GRID (Dense Sampling Around Best Region)    │
├─────────────────────────────────────────────────────────────┤
│  All Groups: 6 points/param, sampling around best          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: TPE BAYESIAN (Final Refinement)                   │
├─────────────────────────────────────────────────────────────┤
│  50 Bayesian trials using Tree-structured Parzen Estimator │
│  → Learns which param combinations work best                │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Best Parameters Found!
```

---

## Why This Order?

**Priority 1-4** optimize **detection**:
- What gets detected
- How it's filtered
- How far we look
- Market context

**Priority 5** optimizes **strength**:
- How we score already-detected levels
- Can't optimize scoring without levels to score

**Dependencies**:
```
Group 1 (Core) ──┐
                 ├─→ Group 2 (Filter) ──┐
                 │                       ├─→ Group 4 (Context)
Group 1 (Core) ──┘                       │
                                         ├─→ Group 5 (Strengths)
                                         │
                         Group 3 (Lookback) ──┘
```

---

## Search Space Size

### Without Strength Weights (Groups 1-4 only)
- **Total params**: 7 parameters
- **Coarse grid**: 4^7 = 16,384 combinations
- **Fine grid**: 6^7 = 279,936 combinations
- **TPE trials**: 50 trials
- **Estimated time**: ~20-30s

### With Strength Weights (All 5 groups)
- **Total params**: 17 parameters (7+10)
- **Coarse grid**: 4^17 ≈ 17 billion (sampled intelligently)
- **Fine grid**: 6^17 ≈ 169 trillion (sampled around best)
- **TPE trials**: 50 trials
- **Estimated time**: ~35-45s

**Key**: Hierarchical optimization uses **stratified sampling** to handle large spaces efficiently.

---

## Example: Optimized Parameters

After running HPO, you might get:

```python
{
    # Group 1: Core Detection
    'min_touches': 3,              # 3 touches required (not 2, not 5)
    'strength_threshold': 0.45,    # Medium threshold
    
    # Group 2: Quality Filtering
    'distance_threshold': 0.008,   # Tight clustering (0.5% tolerance)
    'volume_threshold': 1.2,       # Require 20% above avg volume
    
    # Group 3: Temporal Lookback
    'lookback_periods': 75,        # 75 bars back (balanced)
    
    # Group 4: Market Context
    'trend_strength_threshold': 0.55,  # Medium trend required
    'breakout_threshold': 0.025,       # 2.5% breakout threshold
    
    # Group 5: Strength Weights (10 params)
    'touch_weight': 0.15,                      # Touches matter
    'volume_weight': 0.22,                     # Volume important
    'consistency_weight': 0.18,                # Consistency helps
    'confluence_weight': 0.12,                 # Confluence bonus
    'pivot_boost': 0.12,                       # Pivots valuable
    'psychological_boost': 0.06,               # Round numbers
    'hvn_boost': 0.14,                         # High volume zones
    'failure_penalty_base': 0.25,              # Moderate failures
    'failure_volume_multiplier': 2.0,          # Heavy volume scaling
    'failure_max_penalty': 0.7                 # Allow compounding
}
```

---

## How HPO Works

### Objective Function

For each parameter combination:
```python
1. Filter all pre-detected levels using groups 1-4 params
2. Recalculate strengths using group 5 weights
3. Score = level_count_score × 0.4 + avg_strength × 0.6
4. Return score (higher = better)
```

### Why Pre-Detection?

**Optimization**: Instead of re-detecting 1000s of times (slow), we:
1. Detect once with very relaxed params
2. Filter this set for each trial (fast)
3. This speeds up HPO by **10-100x**

---

## Performance by Group

| Group | Params | Impact | Time/% | Why Important |
|-------|--------|--------|--------|---------------|
| **1: Core** | 2 | 🔥🔥🔥 | 25% | Controls what exists |
| **2: Filter** | 2 | 🔥🔥 | 20% | Removes noise |
| **3: Lookback** | 1 | 🔥 | 10% | Historical depth |
| **4: Context** | 2 | 🔥 | 15% | Market awareness |
| **5: Strengths** | 10 | 🔥🔥🔥 | 30% | Scoring accuracy |
| **Total** | **17** | - | **100%** | **Complete SR system** |

---

## Complete Workflow

```
START
  ↓
Detect SR levels with relaxed params (once, fast)
  ↓
┌───────────────────────────────────────────────┐
│ Hierarchical HPO Optimization                 │
│                                                │
│  Group 1: Core Detection          (2 params)  │
│    ↳ min_touches, strength_threshold          │
│                                                │
│  Group 2: Quality Filtering       (2 params)  │
│    ↳ distance_threshold, volume_threshold     │
│                                                │
│  Group 3: Temporal Lookback       (1 param)   │
│    ↳ lookback_periods                         │
│                                                │
│  Group 4: Market Context          (2 params)  │
│    ↳ trend_strength, breakout_threshold       │
│                                                │
│  Group 5: Strength Weights        (10 params) │
│    ↳ 7 boosts + 3 penalties                   │
│                                                │
│  Strategy: Coarse → Fine → TPE                │
└───────────────────────────────────────────────┘
  ↓
Return best params for all groups
  ↓
Use optimized params for final SR detection
  ↓
END
```

---

## Next: What About Detection Methods?

**Future consideration**: Optimize detection method weights too!

Currently, detection methods are:
- Pivot points
- Fractal patterns
- Volume profile
- Statistical levels
- Psychological levels

**Potential Group 6**: 
- Optimize relative importance of each method
- Example: Maybe volume_profile >> pivot_points for ETHUSDT

This could be added later if current optimization plateaus.

---

**Summary**: 5 hierarchical groups optimizing 17 parameters total, with strength weights as the newest addition (Group 5)!

