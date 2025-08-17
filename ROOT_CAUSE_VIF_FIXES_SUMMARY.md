# Root Cause VIF Fixes Summary

## Problem Identified 🔍

The logs showed extensive high VIF (Variance Inflation Factor) values, indicating severe multicollinearity issues:
- **20 features with VIF > 20.0** were identified for PCA combination
- **sma_20: VIF = 183.11** (very high multicollinearity)
- **nearest_support_distance: VIF = 96.54** (high multicollinearity)
- Multiple wavelet features with VIF > 30-60
- Multiple momentum and volatility features with VIF > 30-50

## Root Cause Analysis 🔧

### **Primary Issues:**
1. **Redundant Feature Calculations**: Multiple features using identical calculation methods
2. **Overlapping Time Windows**: Features calculated with similar window sizes (5, 10, 20)
3. **Similar Mathematical Operations**: Multiple features using the same underlying calculations
4. **Wavelet Parameter Overlap**: Multiple wavelet features using similar scales and types
5. **Price-Based Features**: Raw price-based features highly correlated with each other

### **Specific Problems:**
- `sma_20` and `ema_20` both highly correlated with price
- `momentum_5`, `momentum_10`, `roc_5`, `roc_10` all use similar calculations
- `realized_volatility`, `1m_price_volatility` both use similar volatility estimators
- `morl_*` and `cmor1.5-1.0_*` wavelet features use similar parameters

## Root Cause Fixes Implemented 🛠️

### **1. Moving Averages - Reduced Multicollinearity**
**Before:**
```python
feats["sma_20"] = close.rolling(20, min_periods=1).mean().values
feats["sma_50"] = close.rolling(50, min_periods=1).mean().values
feats["ema_20"] = close.ewm(span=20, adjust=False).mean().values
feats["ema_50"] = close.ewm(span=50, adjust=False).mean().values
```

**After:**
```python
# Price deviation from moving averages (more informative than raw MAs)
feats["price_deviation_sma20"] = ((close - sma_20) / sma_20).values
feats["price_deviation_sma50"] = ((close - sma_50) / sma_50).values

# Moving average crossover (trend indicator)
feats["ma_crossover"] = (sma_20 - sma_50).values

# Price acceleration (second derivative) instead of slope
feats["price_acceleration"] = close.diff().diff().values
```

**Impact:** Eliminates correlation between price and moving averages by using deviations and crossovers.

### **2. Momentum Indicators - Diversified Calculations**
**Before:**
```python
momentum_5_series = price_diff.rolling(5).sum()
momentum_10_series = price_diff.rolling(10).sum()
momentum_20_series = price_diff.rolling(20).sum()
roc_5_series = price_data["close"].pct_change(5)
roc_10_series = price_data["close"].pct_change(10)
```

**After:**
```python
# Different time windows to reduce correlation
momentum_3_series = price_diff.rolling(3).sum()
momentum_7_series = price_diff.rolling(7).sum()
momentum_15_series = price_diff.rolling(15).sum()

# Momentum acceleration (change in momentum)
momentum_accel_series = momentum_7_series.diff()

# Exponential momentum (weighted average)
exp_momentum_series = price_diff.ewm(span=5).mean()

# Different rate of change windows
roc_3_series = price_data["close"].pct_change(3)
roc_7_series = price_data["close"].pct_change(7)
```

**Impact:** Uses non-overlapping windows and different calculation methods to reduce correlation.

### **3. Volatility Indicators - Multiple Estimators**
**Before:**
```python
features["price_volatility"] = returns_series.rolling(20).std()
```

**After:**
```python
# Garman-Klass volatility (more efficient than realized volatility)
garman_klass = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
features["garman_klass_volatility"] = garman_klass.rolling(10).mean()

# Parkinson volatility (high-frequency efficient)
parkinson = np.sqrt(np.log(high/low)**2 / (4 * np.log(2)))
features["parkinson_volatility"] = parkinson.rolling(15).mean()

# Rogers-Satchell volatility (includes overnight gaps)
rs = np.sqrt(np.log(high/close) * np.log(high/open) + np.log(low/close) * np.log(low/open))
features["rogers_satchell_volatility"] = rs.rolling(12).mean()

# Add volatility derivatives
features["volatility_change"] = realized_vol.diff().fillna(0)
features["volatility_acceleration"] = realized_vol.diff().diff().fillna(0)
```

**Impact:** Uses different volatility estimators with different windows and adds derivatives.

### **4. Wavelet Features - Diversified Parameters**
**Before:**
```python
for wavelet_type in ["morl", "cmor1.5-1.0"]:
```

**After:**
```python
# Use different wavelet types to reduce multicollinearity
for wavelet_type in ["db4", "coif4", "sym4"]:
```

**Before:**
```python
self.cwt_scale_method = "logarithmic"
scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
```

**After:**
```python
self.cwt_scale_method = "reduced_multicollinearity"
# Use non-overlapping scales to reduce multicollinearity
scales = np.array([2, 4, 8, 16, 32, 64])  # Powers of 2
```

**Impact:** Uses different wavelet types and non-overlapping scales to reduce correlation.

### **5. Multi-Timeframe Features - Different Methods**
**Before:**
```python
if timeframe == "1m":
    features["momentum_1m"] = returns_series.rolling(5).apply(...)
elif timeframe == "5m":
    features["momentum_5m"] = returns_series.rolling(10).apply(...)
```

**After:**
```python
if timeframe == "1m":
    # High-frequency features
    features["hf_momentum"] = returns_series.rolling(3).sum()
    features["hf_volatility"] = returns_series.rolling(5).std()
elif timeframe == "5m":
    # Medium-frequency features
    features["mf_momentum"] = returns_series.rolling(7).sum()
    features["mf_volatility"] = returns_series.rolling(10).std()
```

**Impact:** Uses different calculation methods and non-overlapping windows for each timeframe.

## Expected Results 📊

### **Before Fixes:**
- 20+ features with VIF > 20.0
- High correlation between similar features
- Redundant information in feature set
- Poor feature diversity

### **After Fixes:**
- **Reduced VIF values** across all feature categories
- **Better feature diversity** with different calculation methods
- **Non-overlapping windows** to reduce correlation
- **Multiple volatility estimators** instead of redundant ones
- **Different wavelet types** and scales
- **Price derivatives** instead of raw price-based features

## Implementation Details 🔧

### **Files Modified:**
- `src/training/steps/vectorized_advanced_feature_engineering.py`
  - Moving average calculations
  - Momentum indicator calculations
  - Volatility estimator calculations
  - Wavelet parameter configuration
  - Multi-timeframe feature calculations

### **Key Changes:**
1. **Price Deviations**: Use `(price - MA) / MA` instead of raw MAs
2. **Non-overlapping Windows**: 3, 7, 15 instead of 5, 10, 20
3. **Multiple Estimators**: Garman-Klass, Parkinson, Rogers-Satchell
4. **Different Wavelet Types**: db4, coif4, sym4 instead of morl, cmor
5. **Non-overlapping Scales**: Powers of 2 (2, 4, 8, 16, 32, 64)
6. **Feature Derivatives**: Add acceleration and change features

## Monitoring & Validation 🔍

### **Metrics to Track:**
- VIF values for all features
- Feature correlation matrix
- Number of features removed due to high VIF
- Feature diversity scores

### **Expected Improvements:**
- **VIF Reduction**: 50-80% reduction in high VIF features
- **Correlation Reduction**: 30-60% reduction in feature correlations
- **Feature Retention**: More features retained after VIF filtering
- **Model Performance**: Better generalization due to reduced multicollinearity

## Conclusion 🎯

The root cause fixes address the fundamental issue of redundant feature calculations by:

1. **Diversifying calculation methods** across feature types
2. **Using non-overlapping parameters** (windows, scales, types)
3. **Implementing multiple estimators** for the same concept
4. **Adding feature derivatives** instead of raw values
5. **Using price deviations** instead of raw price-based features

These changes should significantly reduce multicollinearity while maintaining or improving the informational content of the feature set.
