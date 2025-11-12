# Understanding the 40% Stability Metric

## TL;DR

**40% stable features is actually fine!** The binary "stable/unstable" classification is just for reporting. The actual ranking uses **continuous stability scores** for ALL features via log multiplication.

## Two Different Metrics

### 1. Binary Classification (Reporting Only)

```
Stability analysis: 24/60 features stable (threshold=0.61)
```

This means:
- **24 features** have stability score ≥ 0.61 (labeled "stable")
- **36 features** have stability score < 0.61 (labeled "unstable")
- Threshold = 60th percentile (adaptive)

**Purpose**: Just for reporting and monitoring trends

### 2. Continuous Weighting (Actual Ranking)

```
Formula: importance^0.7 × stability^0.3
```

This uses **ALL stability scores** (not just binary):
- Feature with stability 0.9 → Strong boost
- Feature with stability 0.6 → Moderate boost
- Feature with stability 0.4 → Small boost
- Feature with stability 0.1 → Penalty

**Purpose**: Actually affects which features are selected

## Example: How It Works

Let's say we have 5 features:

| Feature | Importance | Stability | Binary Label | Combined Score |
|---------|-----------|-----------|--------------|----------------|
| A       | 0.95      | 0.85      | ✅ Stable    | 0.92           |
| B       | 0.90      | 0.55      | ❌ Unstable  | 0.78           |
| C       | 0.85      | 0.65      | ✅ Stable    | 0.79           |
| D       | 0.80      | 0.40      | ❌ Unstable  | 0.67           |
| E       | 0.75      | 0.20      | ❌ Unstable  | 0.56           |

**Binary metric**: 2/5 stable (40%)

**But look at the ranking**:
1. Feature A (0.92) - Stable, high importance ✅
2. Feature C (0.79) - Stable, good importance ✅
3. Feature B (0.78) - **Unstable but still ranked high** because stability is 0.55
4. Feature D (0.67) - Unstable, penalized
5. Feature E (0.56) - Very unstable, heavily penalized

**Key insight**: Even "unstable" features with decent stability scores (like B with 0.55) still contribute!

## Why 40% is Reasonable for Crypto

### 1. **Market Regime Changes**

Crypto markets have distinct regimes:
- Bull markets (trending up)
- Bear markets (trending down)
- Sideways/consolidation
- High volatility periods
- Low volatility periods

**Different features work in different regimes**, so perfect stability across all regimes is unrealistic.

### 2. **High Threshold**

The threshold (0.61) is the **60th percentile**, meaning:
- It's above average
- It's a relatively strict bar
- Features below it aren't "bad", just less consistent

### 3. **Interaction Features**

Your top features are interaction features, which:
- Capture complex relationships
- May be regime-specific
- Still provide value even with moderate stability

## What the Numbers Really Mean

### Stability Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.8 - 1.0   | Very stable | Excellent - works across all regimes |
| 0.6 - 0.8   | Stable | Good - consistent performance |
| 0.4 - 0.6   | Moderate | Acceptable - useful but regime-dependent |
| 0.2 - 0.4   | Low | Caution - may be overfitting |
| 0.0 - 0.2   | Very low | Warning - likely noise or regime-specific |

### Your Distribution (Estimated)

Based on 40% above 0.61:
- ~24 features: 0.61 - 1.0 (stable)
- ~20 features: 0.4 - 0.61 (moderate)
- ~16 features: 0.0 - 0.4 (low)

**This is a healthy distribution!** You have:
- A solid core of stable features (40%)
- A good set of moderately stable features (33%)
- Some regime-specific features (27%)

## When to Worry

You should worry if:

### ❌ Red Flags
- **< 20% stable**: Very few consistent features
- **Average stability < 0.3**: Most features are unstable
- **Top features have stability < 0.4**: Best features are unreliable

### ✅ Your Situation
- **40% stable**: Good core of consistent features
- **Threshold 0.61**: Reasonable bar
- **Top 5 are interaction features**: Complex but valuable

## How to Improve Stability

If you want to increase the % of stable features:

### 1. **Increase Stability Weight**
```yaml
stability_weight: 0.5  # Give more weight to stability
```
This will more aggressively filter out unstable features.

### 2. **Longer Time Windows**
Currently using 5 windows. Try:
```python
n_windows: 7  # More granular stability analysis
```

### 3. **Feature Engineering**
- Add more moving averages (naturally stable)
- Use longer lookback periods
- Create regime-normalized features

### 4. **Filter by Minimum Stability**
Add a hard threshold:
```python
if stability_score < 0.3:
    continue  # Skip very unstable features
```

## Recommended Actions

### ✅ Keep Current Settings

Your 40% is **fine** because:
1. You're using log multiplication (already penalizing unstable features)
2. You have a balanced weight (0.3)
3. Crypto markets are inherently regime-dependent
4. Top features are interaction features (expected to be more complex)

### 📊 Monitor Over Time

Track stability metrics across multiple runs:
- Is 40% consistent or declining?
- Are the same features stable each time?
- Does out-of-sample performance correlate with stability?

### 🔬 Experiment (Optional)

Try different weights and compare results:
```yaml
# Test 1: Current (balanced)
stability_weight: 0.3

# Test 2: More aggressive
stability_weight: 0.5

# Test 3: Very aggressive
stability_weight: 0.7
```

Then backtest to see which performs best.

## Conclusion

**40% stable features is not a problem!**

The key points:
1. ✅ Binary metric is just for reporting
2. ✅ Continuous weighting affects all features
3. ✅ 40% is reasonable for crypto markets
4. ✅ Log multiplication already penalizes instability
5. ✅ Top features are interaction features (expected)

**What matters most**: Out-of-sample performance in backtesting, not the % of "stable" features.

If you want to be more conservative, increase `stability_weight` to 0.4 or 0.5 and re-run. But I'd recommend backtesting first to see if the current 0.3 setting is already optimal.
