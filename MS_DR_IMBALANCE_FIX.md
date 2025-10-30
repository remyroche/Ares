# MS-DR Imbalance Fix: From 2 Regimes (83.1% imbalanced) to Balanced Multi-Regime

**Problem:** MS-DR only finding 2 regimes with highly imbalanced distribution (16.9% / 83.1%)

---

## 🚨 Root Cause Analysis

### Why Only 2 Regimes?

**BIC Penalty**: BIC (Bayesian Information Criterion) heavily penalizes model complexity
- BIC = -2 * log_likelihood + k * log(n)
- Where k = number of parameters (more regimes = more parameters)
- BIC favors simpler models → prefers fewer regimes

**Signal Still Too Smooth**: Despite 42 components:
- Components still moderately correlated (especially vol/trend derived from same OHLCV)
- Linear combinations smooth out regime differences
- Need more aggressive non-linear transformations

**Model Selection**: MS-DR tried 2, 3, 4, 5 regimes and selected 2 as "best" by BIC

---

## ✅ Aggressive Solutions

### Solution 1: Use AIC Instead of BIC

**AIC** (Akaike Information Criterion) is less conservative:
- AIC = -2 * log_likelihood + 2 * k
- Smaller penalty for complexity than BIC
- **Favors more regimes** when data supports them

```python
config = MSDRConfig(
    ic_criterion='aic',  # Changed from 'bic'
    min_regimes=3,       # Force minimum
    max_regimes=8        # Try more regimes
)
```

### Solution 2: Add Aggressive Regime Indicators

**New indicators** in `ms_dr_aggressive_fix.py`:

1. **Percentile-based transformations** (non-linear):
   ```python
   vol_20_pct = vol_20.rolling(252).apply(
       lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
   )
   ```

2. **Regime transition detection**:
   ```python
   vol_regime_binary = (vol_20 > vol_20.median()).astype(int)
   transitions = vol_regime_binary.diff().abs()
   ```

3. **Market stress indicators**:
   ```python
   stress_score = (vol_spike + negative_returns + range_expansion) / 3
   ```

4. **Volatility of volatility** (regime instability):
   ```python
   vol_of_vol = vol_20.rolling(20).std()
   ```

### Solution 3: Force Minimum Regimes

```python
config = MSDRConfig(
    min_regimes=3,  # FORCE at least 3
    max_regimes=8,  # Try up to 8
    ic_criterion='aic'  # Less penalty
)
```

### Solution 4: Increase AR Order

```python
config = MSDRConfig(
    order=3,  # AR(3) instead of AR(2)
    # Richer dynamics, can capture more complex regime patterns
)
```

### Solution 5: PCA Decorrelation

If components highly correlated (mean corr > 0.3):
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep 95% variance
indicators_pca = pca.fit_transform(indicators_scaled)

# Weight by explained variance
weights = pca.explained_variance_ratio_
regime_signal = (indicators_pca * weights).sum(axis=1)
```

---

## 🧪 Testing Strategy

### Test Multiple Configurations

Run `ms_dr_aggressive_fix.py` which tests:

1. **AIC, min=3 regimes** (least conservative)
2. **AIC, min=4 regimes** (force more regimes)
3. **BIC, min=3 regimes** (conservative baseline)

Selects configuration with:
- ✅ Most balanced distribution
- ✅ Meets minimum regime requirement
- ✅ Highest quality score

### Usage:

```bash
# Test with AIC, minimum 3 regimes
python3 ms_dr_aggressive_fix.py --min-regimes 3 --use-aic

# Test with AIC, minimum 4 regimes (more aggressive)
python3 ms_dr_aggressive_fix.py --min-regimes 4 --use-aic
```

---

## 📊 Expected Results

### Before (Current Problem):
- ❌ 2 regimes: 16.9% / 83.1%
- ❌ Highly imbalanced
- ❌ Using BIC (conservative)
- ❌ Simple signal (moderate correlation)

### After (Aggressive Fix):
- ✅ 3-5 regimes: ~20-40% each
- ✅ Balanced distribution
- ✅ Using AIC (less conservative)
- ✅ Aggressive signal (regime transitions, stress indicators)

---

## 🔧 Alternative Approaches

If aggressive MS-DR still produces poor results:

### 1. Try HDP-HMM (Already Available)
```bash
python ares_launcher.py hdp_hmm_clustering --symbol ETHUSDT --timeframe 1h
```

**Advantages:**
- Non-parametric (automatically infers regime count)
- No IC selection bias
- Designed for temporal regime discovery

### 2. Try GMM (Gaussian Mixture Models)
```bash
python ares_launcher.py gmm_regime_discovery --symbol ETHUSDT --timeframe 1h
```

**Advantages:**
- Soft clustering (probabilistic)
- Can discover complex regime structures
- No temporal assumptions

### 3. Manual Regime Count Selection

Override auto-selection:
```python
config = MSDRConfig(
    auto_select_regimes=False,  # Disable auto-selection
    n_regimes=4,  # Force 4 regimes
    # ... other params
)
```

### 4. Visual Inspection

Plot the composite signal and manually identify regimes:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.plot(regime_signal)
plt.title('Regime Signal - Visual Inspection')
plt.ylabel('Signal Value')
plt.xlabel('Time')
plt.grid(True)
plt.savefig('regime_signal_inspection.png')
```

Look for clear "regime zones" visually.

---

## ✅ Action Items

### Immediate:
1. **Run aggressive fix:**
   ```bash
   python3 ms_dr_aggressive_fix.py --min-regimes 3 --use-aic
   ```

2. **Compare with original:**
   - Original: 2 regimes (16.9% / 83.1%)
   - Aggressive: ? regimes (hopefully more balanced)

### If Still Imbalanced:
1. **Try min=4 regimes:**
   ```bash
   python3 ms_dr_aggressive_fix.py --min-regimes 4 --use-aic
   ```

2. **Switch to HDP-HMM:**
   ```bash
   python ares_launcher.py hdp_hmm_clustering --symbol ETHUSDT --timeframe 1h
   ```

3. **Visual inspection:**
   - Plot the signal
   - Manually identify regime zones
   - Set n_regimes manually

---

## 📋 Summary

**Problem:** 2 regimes, 83.1% imbalance

**Solutions:**
1. ✅ **AIC instead of BIC** - favors more regimes
2. ✅ **Force min=3 regimes** - prevents too-simple models
3. ✅ **Aggressive indicators** - regime transitions, stress, percentiles
4. ✅ **PCA decorrelation** - reduces component correlation
5. ✅ **AR(3) model** - richer dynamics
6. ✅ **Test multiple configs** - find best balance

**Expected Outcome:** 3-5 balanced regimes (20-40% each)

**Files:**
- `ms_dr_aggressive_fix.py` - Aggressive signal + configuration testing
- `MS_DR_IMBALANCE_FIX.md` - This guide

**Next:** Run `python3 ms_dr_aggressive_fix.py --min-regimes 3 --use-aic`

---

*Status: Ready to fix imbalance issue*


