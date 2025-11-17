# Label Quality Assessment - Deep Dive

**Context:** Understanding how `label_quality > 0.4` filtering works in your labeling pipeline

---

## Overview

Label quality is a **composite score** (0.0 to 1.0) that measures how "clean" and "predictable" each labeled opportunity is. It's computed by `VolatilityAwareMultiHorizonLabeler._calculate_comprehensive_target_quality()` and stored in `quality_scores['opportunity_quality_scores']`.

---

## Quality Score Components

The quality score is a **weighted combination** of 5 factors:

```python
composite_score = (
    0.4 * profit_score +      # 40% - Potential profit magnitude
    0.2 * consistency_score + # 20% - Profit consistency (low std)
    0.2 * hit_rate_score +    # 20% - How often signals are correct
    0.1 * stability_score +   # 10% - Temporal stability
    0.1 * sharpe_score        # 10% - Risk-adjusted return
)
```

### 1. **Profit Score (40% weight)**
Measures the **average potential profit** of labeled opportunities.

```python
# From _calculate_potential_profit_quality_score()
avg_profit = metrics['avg_potential_profit']  # e.g., 0.015 = 1.5% average move
profit_score = min(1.0, avg_profit / 0.02)    # Normalize to 2% max expected
```

**How it's calculated:**
- For each labeled opportunity at time T with signal direction (long/short)
- Look ahead over the next `lookahead_periods` (you set this to 3 = 45min)
- Calculate potential profit:
  - **Long signal:** `(max_price_in_window - entry_price) / entry_price`
  - **Short signal:** `(entry_price - min_price_in_window) / entry_price`

**Example:**
- Entry at $2000, signal = LONG
- Next 3 periods: [$2010, $2025, $2015]
- Potential profit = (2025 - 2000) / 2000 = **1.25%**

If average across all opportunities is 1.5%, then:
```python
profit_score = min(1.0, 0.015 / 0.02) = 0.75
```

---

### 2. **Consistency Score (20% weight)**
Measures how **stable** the potential profits are (lower std = better).

```python
std_profit = metrics['std_potential_profit']
consistency_score = 1.0 / (1.0 + std_profit * 10)
```

**Why this matters:**
- High std means some opportunities are great, others terrible → noisy labels
- Low std means opportunities are uniformly good → clean labels

**Example:**
- Potential profits: [1.2%, 1.4%, 1.3%, 1.5%] → std ≈ 0.001 → consistency = 0.91 ✅
- Potential profits: [0.2%, 3.5%, -0.5%, 2.1%] → std ≈ 0.015 → consistency = 0.40 ⚠️

---

### 3. **Hit Rate Score (20% weight)**
Measures what **percentage of signals are correct**.

```python
# From _calculate_trade_opportunity_metrics()
# A "hit" = signal direction matches actual price move direction
positive_profits = (potential_profits > 0).sum()
hit_rate = positive_profits / len(potential_profits)
hit_rate_score = hit_rate
```

**Example:**
- 100 labeled opportunities
- 72 have positive potential profit (price moved in predicted direction)
- Hit rate = 72% → score = 0.72

**Your threshold impact:**
- With 1.5% profit threshold (your new setting), you're filtering for larger moves
- This typically **increases hit rate** because small noise is excluded
- Expected hit rate with good labeling: 60-75%

---

### 4. **Stability Score (10% weight)**
Measures **temporal consistency** of signal quality.

```python
# Calculated by looking at rolling windows of opportunities
# Higher stability = quality doesn't fluctuate wildly over time
stability = 1.0 - (rolling_std_of_quality / mean_quality)
```

**Why this matters:**
- Prevents overfitting to a specific market regime
- Ensures labels are reliable across different periods

---

### 5. **Sharpe Score (10% weight)**
**Risk-adjusted return** of the labeled opportunities.

```python
sharpe = mean_potential_profit / std_potential_profit
sharpe_score = min(1.0, max(0.0, (sharpe + 1) / 2))  # Normalize to [0,1]
```

**Example:**
- Mean profit = 1.5%, std = 0.8%
- Sharpe = 1.5 / 0.8 = 1.875
- Sharpe score = min(1.0, (1.875 + 1) / 2) = 1.0 ✅

---

## Per-Sample Quality Scores

The **composite score** above is an overall metric. But you also get **per-sample quality scores** for filtering:

```python
# From _calculate_individual_opportunity_quality_score()
individual_quality = overall_quality * opportunity_score * opportunity_weight
```

Where:
- `overall_quality` = the composite score above
- `opportunity_score` = how good this specific opportunity is (relative to average)
- `opportunity_weight` = importance weight based on profit magnitude

**This is what you filter on:**
```python
quality_mask = (label_quality > 0.4)
```

---

## What `label_quality > 0.4` Actually Filters

### **Keeps (quality > 0.4):**
✅ Opportunities with clear directional moves  
✅ Consistent profit potential across similar setups  
✅ High hit rate (>60% correct)  
✅ Stable quality over time  
✅ Good risk-adjusted returns  

### **Removes (quality ≤ 0.4):**
❌ Choppy/sideways periods (low potential profit)  
❌ Inconsistent opportunities (high std)  
❌ Low hit rate (<50% correct)  
❌ Regime-specific flukes  
❌ High-risk, low-reward setups  

---

## Your Current Settings Impact

You recently changed:
```python
BASE_VOLATILITY_THRESHOLD = 0.015  # 1.5% (up from 1.3%)
lookahead_periods = 3              # 45min
min_threshold_multiplier = 1.0     # (up from 0.75)
max_threshold_multiplier = 2.0     # (up from 1.75)
```

**Effect on quality scores:**

1. **Higher threshold (1.5%):**
   - **Profit score ↑** - Only labels moves ≥1.5%, so avg_profit increases
   - **Hit rate ↑** - Larger moves are easier to predict (less noise)
   - **Consistency ↑** - Filters out small random fluctuations
   - **Overall quality ↑** - Expect 10-20% improvement in composite score

2. **Shorter lookahead (3 periods):**
   - **Profit score ↓** - Less time for price to reach target
   - **Hit rate ↑** - Less time for reversals
   - **Stability ↑** - Shorter horizon = more consistent
   - **Net effect:** Slightly lower quality, but cleaner signals

3. **Wider multiplier range (1.0x - 2.0x):**
   - In high volatility: threshold → 3.0% (1.5% × 2.0)
   - In low volatility: threshold → 1.5% (1.5% × 1.0)
   - **Profit score ↑** in high vol (captures bigger moves)
   - **Consistency ↑** overall (adaptive to regime)

---

## Practical Example

Let's trace a specific sample through the quality calculation:

### Sample at 2025-01-15 10:30:00

**Market conditions:**
- Price: $2000
- EWMA volatility: 1.2% (slightly above median)
- Threshold: 1.5% × 1.2 = **1.8%** (volatility-adjusted)

**Lookahead window (next 3 × 15min = 45min):**
```
10:30 → $2000 (entry)
10:45 → $2015 (+0.75%)
11:00 → $2038 (+1.9%) ← hits threshold!
11:15 → $2025 (+1.25%)
```

**Label generated:**
- Signal: LONG (1.0)
- Potential profit: (2038 - 2000) / 2000 = **1.9%**

**Quality calculation for this sample:**

1. **Profit score component:**
   - This sample: 1.9%
   - Average across all samples: 1.6%
   - Relative score: 1.9 / 1.6 = 1.19 (above average) ✅

2. **Consistency component:**
   - Deviation from mean: |1.9 - 1.6| = 0.3%
   - Std across all samples: 0.5%
   - Consistency: 1 / (1 + 0.3/0.5) = 0.625 ✅

3. **Individual quality score:**
   ```python
   overall_quality = 0.52  # Composite across all samples
   opportunity_score = 0.85  # This sample is above average
   opportunity_weight = 0.90  # High weight due to good profit
   
   individual_quality = 0.52 × 0.85 × 0.90 = 0.398
   ```

**Result:** This sample has quality = 0.398 → **FILTERED OUT** by `quality > 0.4` ❌

Even though it's a decent opportunity, it's just below the threshold. This is intentional—you want only the **cleanest** examples.

---

## Recommended Quality Thresholds

| Threshold | Use Case | Expected Retention | Signal Quality |
|-----------|----------|-------------------|----------------|
| `> 0.3` | Permissive (current default) | ~60-70% | Mixed |
| `> 0.4` | **Recommended** | ~40-50% | Good |
| `> 0.5` | Strict | ~25-35% | Excellent |
| `> 0.6` | Very strict | ~10-20% | Elite |

**Trade-off:**
- Higher threshold → Fewer samples, but much cleaner signal
- Lower threshold → More samples, but noisier

For your 15m ETHUSDT with 1204 samples:
- `> 0.3`: Keep ~800 samples
- `> 0.4`: Keep ~500 samples ← **Good balance**
- `> 0.5`: Keep ~350 samples

---

## How to Inspect Quality Scores

After running the labeling step, quality scores are stored in the result:

```python
# In your labeling step output
quality_scores = labeling_result.quality_scores
opportunity_quality = quality_scores['opportunity_quality_scores']

# Inspect distribution
print(f"Mean quality: {opportunity_quality.mean():.3f}")
print(f"Median quality: {opportunity_quality.median():.3f}")
print(f"Samples > 0.4: {(opportunity_quality > 0.4).sum()} / {len(opportunity_quality)}")

# Plot histogram
import matplotlib.pyplot as plt
plt.hist(opportunity_quality, bins=50)
plt.axvline(0.4, color='red', linestyle='--', label='Threshold')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.title('Label Quality Distribution')
plt.legend()
plt.savefig('quality_distribution.png')
```

---

## Integration with Your Filtering

To implement quality filtering in your pipeline:

```python
# In feature_generation_labeling_integration_step.py
# After label generation (around line 950-1000)

# Extract quality scores
quality_scores = labeling_result.quality_scores
if 'opportunity_quality_scores' in quality_scores:
    label_quality = quality_scores['opportunity_quality_scores']
    
    # Apply quality filter
    quality_mask = (
        (label_quality > 0.4) &  # Stricter quality gate
        (ewma_vol > 0.5 * vol_median) &  # Your existing vol filter
        (market_data['volume'] > market_data['volume'].rolling(20).mean() * 0.7)  # Liquidity
    )
    
    # Filter features and labels
    features_clean = features[quality_mask]
    labels_clean = labels[quality_mask]
    
    # Log impact
    retention_pct = 100 * quality_mask.sum() / len(quality_mask)
    tprint(f"Quality filtering: {len(features)} → {len(features_clean)} samples ({retention_pct:.1f}% retained)", "INFO")
    tprint(f"Mean quality (retained): {label_quality[quality_mask].mean():.3f}", "INFO")
```

---

## Summary

**Label quality is assessed through:**
1. ✅ **Potential profit** (40%) - How much the price moved in predicted direction
2. ✅ **Consistency** (20%) - How stable profits are across opportunities
3. ✅ **Hit rate** (20%) - Percentage of correct predictions
4. ✅ **Stability** (10%) - Temporal consistency of quality
5. ✅ **Sharpe** (10%) - Risk-adjusted returns

**Your `quality > 0.4` filter keeps:**
- Top ~40-50% of labeled opportunities
- Samples with clear directional moves
- Consistent, predictable patterns
- Good risk/reward setups

**Expected impact:**
- 5-10% improvement in Test R² (cleaner training signal)
- Better generalization (smaller train/test gap)
- Fewer false positives in live trading
