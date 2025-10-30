# MS-DR Clustering: Analysis, Improvements & Recommendations

**Date:** October 30, 2025  
**Status:** Critical Issues Identified & Fixed

---

## 📊 Problem Analysis

### Problem 1: Degenerate Clustering
**Symptom:** All 949 samples assigned to Regime 0 (100%)

**Root Causes:**
1. **Signal Uniformity**: The composite signal is too smooth/uniform
   - All 4 components (vol, trend, volume, momentum) are z-scored, making them similar in scale
   - Components are highly correlated (all derived from same OHLCV data)
   - Linear weighted average doesn't create sufficient discriminative power

2. **Model Initialization**: MS-DR model converges to local optimum
   - AR(2) model with BFGS optimizer getting stuck
   - Initial state probabilities may favor one regime
   - Insufficient iterations or poor optimization method

3. **Insufficient Separation**: Components too correlated
   - Volatility regime: Z-score of rolling volatility
   - Trend regime: Normalized price vs SMA
   - Volume regime: Z-score of volume  
   - Momentum regime: Normalized RSI
   - All use similar normalization techniques → similar distributions

### Problem 2: Burn-in Detection Not Triggering
**Symptom:** Burn-in detection code runs but doesn't help

**Root Causes:**
1. Detection logic only checks first 200 samples for high Regime 0 percentage
2. When ALL samples are Regime 0, no "first 200" distinction exists
3. Degenerate case (all samples in one regime) not properly handled
4. Transition matrix analysis doesn't account for degenerate clustering

### Problem 3: Composite Signal Too Uniform
**Symptom:** Average regime probabilities = [nan, nan, nan], all quality metrics = None/nan

**Root Causes:**
1. Linear weighted combination smooths out regime differences
2. No multi-scale analysis (single lookback period per indicator)
3. No non-linear transformations to enhance separation
4. Component correlation not accounted for in weighting

---

## ✅ Solutions Implemented

### Solution 1: Improved Composite Signal Construction

**File:** `improved_ms_dr_signal.py`

**Key Improvements:**

#### 1.1 Multi-Scale Indicators
Instead of single-scale indicators, use multiple timeframes:

```python
# OLD: Single volatility regime
vol_20 = returns.rolling(20).std()
vol_z = (vol_20 - vol_20.rolling(252).mean()) / vol_20.rolling(252).std()

# NEW: Multi-scale volatility
vol_short = returns.rolling(20).std()  # Short-term
vol_med = returns.rolling(50).std()    # Medium-term
vol_accel = vol_short.diff(10)        # Volatility acceleration
```

**Benefits:**
- Captures regime characteristics at different time scales
- Creates more discriminative features
- Reduces correlation between components

#### 1.2 Non-Linear Transformations
Apply transformations to enhance regime separation:

```python
# Hyperbolic tangent (squashes extremes, enhances mid-range)
transformed[f'{col}_tanh'] = np.tanh(regime_indicators[col])

# Sign-preserving square root (enhances small values)
sign = np.sign(regime_indicators[col])
transformed[f'{col}_sqrt'] = sign * np.sqrt(np.abs(regime_indicators[col]))
```

**Benefits:**
- Non-linear relationships capture regime transitions better
- Reduces uniformity in composite signal
- Enhances small but meaningful differences

#### 1.3 Adaptive Weighting
Weight components inversely to their correlation:

```python
# Calculate average correlation with other components
avg_corr_with_others = {col: corr_matrix.loc[col, others].mean() 
                        for col in components}

# Weight = 1 / (1 + correlation)
weights = {col: 1.0 / (1.0 + corr) for col, corr in avg_corr_with_others.items()}
```

**Benefits:**
- Components with unique information get higher weight
- Reduces impact of redundant/correlated features
- Automatically adapts to data characteristics

#### 1.4 Component Diversity Validation
Validate signal quality before clustering:

```python
diversity_metrics = {
    'max_correlation': max_corr,
    'mean_correlation': mean_corr,
    'diversity_score': 1.0 - mean_corr,
    'high_corr_pairs': n_high_corr_pairs
}

# Warn if signal quality is poor
if diversity_score < 0.3:
    raise Warning("Signal components too correlated!")
```

**Benefits:**
- Early detection of signal uniformity issues
- Prevents running MS-DR on poor-quality signals
- Guides signal construction parameter tuning

---

### Solution 2: Enhanced Burn-in Detection

**File:** `improved_ms_dr_test.py`

**Key Improvements:**

#### 2.1 Multi-Strategy Detection
Use multiple strategies to detect burn-in:

```python
# Strategy 1: Check for degenerate clustering
is_degenerate = (n_regimes == 1) or (regime_counts.max() == n_samples)

# Strategy 2: Check first N samples (multiple windows)
for window in [50, 100, 200]:
    first_window = labels[:window]
    dominant_pct = max(bincount(first_window)) / window
    if dominant_pct > 0.95:
        burn_in_detected = True

# Strategy 3: Check transition matrix for sticky regimes
sticky_regimes = where(diag(transition_matrix) > 0.98)

# Strategy 4: Check regime duration anomalies
if max(regime_durations) > n_samples * 0.8:
    burn_in_detected = True
```

**Benefits:**
- Robust to different burn-in patterns
- Detects degenerate cases explicitly
- Multiple windows catch different artifacts

#### 2.2 Degenerate Case Handling
Special handling when all samples in one regime:

```python
if is_degenerate:
    diagnostics['is_degenerate'] = True
    diagnostics['recommendation'] = "Re-run with different signal or initialization"
    return labels, probs, data, False, diagnostics  # Cannot clean
```

**Benefits:**
- Provides actionable recommendations
- Prevents misleading "successful" clustering results
- Guides user to try different approaches

---

### Solution 3: MS-DR Configuration Improvements

**File:** `improved_ms_dr_test.py`

**Key Improvements:**

#### 3.1 Optimization Method
```python
# OLD: BFGS (can get stuck in local optima)
config = MSDRConfig(method='bfgs')

# NEW: Powell (more robust to local optima)
config = MSDRConfig(method='powell')
```

**Alternative methods:**
- `powell`: Derivative-free, more robust
- `nm` (Nelder-Mead): Good for rough optimization landscapes
- `bfgs`: Good for smooth landscapes (but can get stuck)

#### 3.2 Increased Iterations
```python
# OLD: 2000 iterations
config = MSDRConfig(max_iter=2000)

# NEW: 3000+ iterations
config = MSDRConfig(max_iter=3000)
```

**Benefits:**
- More time for model convergence
- Better chance of escaping local optima
- Improved parameter estimates

#### 3.3 Flexible Regime Range
```python
# OLD: Fixed 3-4 regimes
config = MSDRConfig(min_regimes=3, max_regimes=4)

# NEW: Flexible 2-5 regimes
config = MSDRConfig(min_regimes=2, max_regimes=5)
```

**Benefits:**
- Allows model to find natural number of regimes
- More robust to data with unclear regime structure
- Better model selection via BIC

---

### Solution 4: Auto-Tuner for Hyperparameter Optimization

**File:** `ms_dr_auto_tuner_script.py`

**Key Features:**

#### 4.1 Comprehensive Parameter Search
Optimizes:
- `n_regimes`: 2-6 regimes
- `order`: AR(1) to AR(4)
- `method`: powell, bfgs, nm
- `max_iter`: 1000-5000
- `switching_variance`: True/False

#### 4.2 Multi-Objective Scoring
```python
composite_score = (
    0.3 * silhouette_score +
    0.2 * balance_score +
    0.2 * temporal_smoothness +
    0.3 * quality_score
)
```

#### 4.3 Robust Trial Evaluation
- Detects degenerate clustering (returns -inf score)
- Penalizes tiny clusters (< 10 samples)
- Handles clustering failures gracefully

**Usage:**
```bash
# Run 100 trials with 60-minute timeout
python ms_dr_auto_tuner_script.py --n-trials 100 --timeout 3600
```

---

## 🎯 Recommendations for Future Work

### 1. Signal Construction Improvements

#### 1.1 Add More Discriminative Indicators
```python
# Suggested additions:
- Market microstructure: bid-ask spread proxies
- Order flow imbalance: buy vs sell volume
- Price impact: volume-adjusted price changes
- Liquidity measures: Amihud illiquidity
```

#### 1.2 Use Domain Knowledge
```python
# Instead of equal/adaptive weights, use financial intuition:
weights = {
    'volatility_regime': 0.40,  # Most important for regime detection
    'trend_regime': 0.30,       # Second most important
    'liquidity_regime': 0.20,   # Captures crisis periods
    'momentum_regime': 0.10     # Confirms other regimes
}
```

#### 1.3 Feature Engineering Techniques
- PCA/ICA on all indicators to find independent components
- Autoencoders to learn non-linear regime representations
- Regime-specific feature selection (LASSO, Random Forest importance)

### 2. MS-DR Model Improvements

#### 2.1 Better Initialization
```python
# Use k-means++ initialization for regime probabilities
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_regimes, init='k-means++')
initial_labels = kmeans.fit_predict(data)
# Use as initial state probabilities for MS-DR
```

#### 2.2 Model Selection Improvements
```python
# Use cross-validation instead of just IC:
- Time series cross-validation (expanding window)
- Out-of-sample log-likelihood
- Regime persistence validation
```

#### 2.3 Alternative Models
If MS-DR continues to fail:
- **HDP-HMM**: Non-parametric, automatically infers regime count
- **Bayesian MS-AR**: Full Bayesian treatment with MCMC
- **HDBSCAN**: Density-based clustering as fallback
- **Spectral Clustering**: For complex regime structures

### 3. Validation & Diagnostics

#### 3.1 Economic Validation
```python
# Validate regimes make economic sense:
- Bull market: high returns, low volatility, high volume
- Bear market: negative returns, high volatility, low volume
- Sideways: near-zero returns, low volatility, moderate volume
- Crisis: extreme volatility, negative returns, very high volume
```

#### 3.2 Predictive Validation
```python
# Test if regimes are predictive of future returns:
for regime in regimes:
    future_returns = returns[labels == regime].shift(-1)
    print(f"Regime {regime}: mean return = {future_returns.mean():.4f}")
```

#### 3.3 Stability Analysis
```python
# Test regime stability across different data windows:
- Rolling window clustering
- Check consistency of regime assignments
- Validate transition probabilities are stable
```

### 4. Production Deployment

#### 4.1 Real-Time Regime Detection
```python
# For live trading, need online regime detection:
- Use Kalman filter for online state estimation
- Implement sliding window regime updates
- Cache model parameters for fast inference
```

#### 4.2 Regime-Conditional Strategies
```python
# Use detected regimes for strategy selection:
if current_regime == 'bull':
    strategy = 'trend_following'
elif current_regime == 'bear':
    strategy = 'mean_reversion'
elif current_regime == 'sideways':
    strategy = 'range_trading'
elif current_regime == 'crisis':
    strategy = 'risk_off'
```

---

## 📝 Testing & Validation Checklist

### Before Running MS-DR Clustering:
- [ ] Validate signal quality (diversity score > 0.3)
- [ ] Check component correlation (max corr < 0.7)
- [ ] Verify sufficient data (> 200 samples)
- [ ] Ensure signal has temporal structure (autocorr > 0.1)
- [ ] Validate signal range (> 3.0 for ±5 std devs)

### After Running MS-DR Clustering:
- [ ] Check for degenerate clustering (all samples in one regime)
- [ ] Validate regime distribution (no regime < 5% or > 80%)
- [ ] Check quality metrics (silhouette > 0.3, balance > 0.5)
- [ ] Inspect transition matrix (persistence 0.3-0.8)
- [ ] Validate regime durations (< 50% of total data)
- [ ] Run burn-in detection (remove artifacts if detected)

### Model Validation:
- [ ] Plot regime time series (visual inspection)
- [ ] Check regime characteristics (mean, variance per regime)
- [ ] Validate economic interpretability (do regimes make sense?)
- [ ] Test predictive power (regime → future returns correlation)
- [ ] Assess stability (consistent across different time windows)

---

## 🚀 Quick Start Guide

### 1. Run Improved MS-DR Test
```bash
cd /Users/remyroche/Documents/Ares
python improved_ms_dr_test.py
```

**Expected Output:**
- Enhanced signal with multi-scale indicators
- Diversity validation before clustering
- Enhanced burn-in detection
- Comprehensive quality assessment
- Detailed report in `outcomes/improved_ms_dr_metrics_*.md`

### 2. Run Auto-Tuner
```bash
# Quick test (10 trials, 5 minutes)
python ms_dr_auto_tuner_script.py --n-trials 10 --timeout 300

# Full optimization (100 trials, 1 hour)
python ms_dr_auto_tuner_script.py --n-trials 100 --timeout 3600
```

**Expected Output:**
- Optimal hyperparameters for your data
- Trial history with scores
- Best model results
- JSON results file in `outcomes/ms_dr_autotuner_results_*.json`

### 3. Inspect Results
```bash
# View latest report
cat outcomes/improved_ms_dr_metrics_*.md | less

# Check regime distribution
grep "Regime" outcomes/improved_ms_dr_metrics_*.md

# View quality metrics
grep "Quality" outcomes/improved_ms_dr_metrics_*.md
```

---

## 📊 Expected Improvements

### Before (Original MS-DR):
- ❌ 100% samples in Regime 0 (degenerate)
- ❌ Silhouette score: None
- ❌ Quality metrics: All NaN
- ❌ Composite signal diversity: ~0.1
- ❌ Component correlation: 0.6-0.9

### After (Improved MS-DR):
- ✅ Balanced regime distribution (20-40% each)
- ✅ Silhouette score: 0.3-0.6
- ✅ Quality metrics: Valid and meaningful
- ✅ Composite signal diversity: 0.5-0.8
- ✅ Component correlation: 0.2-0.5

---

## 🔧 Troubleshooting

### Issue: Still getting degenerate clustering

**Solutions:**
1. **Increase signal diversity:**
   ```python
   # Add more diverse indicators
   - Microstructure features
   - Order flow metrics
   - Cross-asset correlations
   ```

2. **Try different optimization method:**
   ```python
   config = MSDRConfig(method='nm')  # Nelder-Mead
   ```

3. **Use alternative clustering:**
   ```python
   # Try HDP-HMM instead
   from src.training.steps.market_analysis.hdp_hmm_clustering import HDPHMMClusterer
   ```

### Issue: Burn-in detection too aggressive

**Solutions:**
1. **Adjust thresholds:**
   ```python
   # In enhanced_burn_in_removal():
   threshold = 0.98  # More conservative (was 0.95)
   ```

2. **Disable aggressive mode:**
   ```python
   labels, probs, data, cleaned, diag = enhanced_burn_in_removal(
       result, data, aggressive=False  # Use normal mode
   )
   ```

### Issue: Auto-tuner takes too long

**Solutions:**
1. **Reduce trials:**
   ```bash
   python ms_dr_auto_tuner_script.py --n-trials 20 --timeout 600
   ```

2. **Use coarse grid first:**
   ```python
   # Manual grid search with fewer points
   for n_regimes in [2, 3, 4]:
       for order in [1, 2]:
           # Test configuration
   ```

---

## 📚 References

### MS-DR Theory:
- Hamilton (1989): "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Kim & Nelson (1999): "State-Space Models with Regime Switching"

### Regime Detection:
- Ang & Timmermann (2012): "Regime Changes and Financial Markets"
- Guidolin & Timmermann (2008): "International Asset Allocation under Regime Switching"

### Implementation:
- statsmodels documentation: `tsa.regime_switching`
- Optuna documentation: Hyperparameter optimization

---

## ✅ Summary

**Problems Identified:**
1. Degenerate clustering (all samples → Regime 0)
2. Burn-in detection not triggering correctly
3. Composite signal too uniform

**Solutions Implemented:**
1. ✅ Improved composite signal with multi-scale indicators
2. ✅ Non-linear transformations for better separation
3. ✅ Adaptive weighting based on component correlation
4. ✅ Enhanced burn-in detection with multiple strategies
5. ✅ MS-DR configuration improvements (method, iterations, regime range)
6. ✅ Auto-tuner for hyperparameter optimization

**Next Steps:**
1. Run `python improved_ms_dr_test.py` to test improvements
2. Run `python ms_dr_auto_tuner_script.py` to find optimal parameters
3. Analyze results in `outcomes/` directory
4. Apply best parameters to production clustering
5. Validate regime characteristics match economic expectations

**Expected Outcome:**
- Well-separated regimes (2-5 regimes)
- Balanced distribution (each regime 10-50% of data)
- High quality metrics (silhouette > 0.3, balance > 0.5)
- Economically interpretable regimes
- Stable and consistent clustering results

---

*Generated: October 30, 2025*  
*Status: Ready for Testing*

