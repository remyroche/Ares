# MS-DR Clustering Quick Reference Guide

## 🚀 Quick Start

```python
from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
    perform_enhanced_ms_dr_clustering
)

# Simple usage with auto regime selection
result = perform_enhanced_ms_dr_clustering(
    data=market_data,
    auto_select_regimes=True,
    min_regimes=2,
    max_regimes=10
)

# Access results
regime_labels = result['cluster_labels']
n_regimes = result['n_clusters']
transition_matrix = result['transition_matrix']
```

---

## ⚙️ Configuration Options

### MSDRConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_regimes` | int | 5 | Number of regimes (if not auto-selecting) |
| `switching_variance` | bool | True | Allow variance to switch across regimes |
| `model_type` | str | 'autoregression' | Model type: 'autoregression' or 'regression' |
| `order` | int | 1 | Autoregression order |
| `enable_pca` | bool | True | Enable PCA dimensionality reduction |
| `pca_components` | int | 10 | Number of PCA components |
| `pca_aggregation` | str | 'first' | How to aggregate: 'first', 'weighted_average', 'none' |
| `auto_select_regimes` | bool | True | Auto-select number of regimes using IC |
| `min_regimes` | int | 2 | Minimum number of regimes |
| `max_regimes` | int | 10 | Maximum number of regimes |
| `ic_criterion` | str | 'aic' | Information criterion: 'aic', 'bic', 'hqic' |
| `min_samples_required` | int | 200 | Minimum samples for reliable estimation |

---

## 📊 Understanding Results

### MSDRResult Fields

```python
result = clusterer.fit_predict(data)

# Clustering results
result.cluster_labels          # np.ndarray: Most likely regime at each time point
result.cluster_probabilities   # np.ndarray: Probability of each regime (n_samples × n_regimes)
result.n_clusters             # int: Number of discovered regimes

# Model artifacts
result.transition_matrix      # np.ndarray: Regime transition probabilities (n_regimes × n_regimes)
result.regime_params         # dict: Statistical parameters for each regime
result.regime_variances      # np.ndarray: Volatility in each regime

# Quality metrics
result.silhouette_score      # float: Clustering quality (-1 to 1, higher better)
result.aic                   # float: Akaike Information Criterion (lower better)
result.bic                   # float: Bayesian Information Criterion (lower better)
result.log_likelihood        # float: Model log-likelihood

# Regime statistics
result.regime_durations      # np.ndarray: Average duration of each regime
result.transition_persistence # float: Average self-transition probability

# Metadata
result.processing_time       # float: Time taken (seconds)
result.memory_usage_mb      # float: Peak memory usage (MB)
result.success              # bool: Whether clustering succeeded
```

---

## 🎯 Common Use Cases

### 1. Market Regime Identification

```python
# Identify bull/bear/sideways markets
result = perform_enhanced_ms_dr_clustering(
    data=market_data,
    n_regimes=3,  # Bull, Bear, Sideways
    auto_select_regimes=False,
    switching_variance=True  # Capture volatility differences
)

# Analyze regime characteristics
for i in range(result['n_clusters']):
    mask = result['cluster_labels'] == i
    regime_returns = market_data['close'][mask].pct_change()
    print(f"Regime {i}:")
    print(f"  Mean return: {regime_returns.mean():.4f}")
    print(f"  Volatility: {regime_returns.std():.4f}")
```

### 2. Volatility Regime Detection

```python
# Focus on volatility changes
result = perform_enhanced_ms_dr_clustering(
    data=market_data,
    min_features=50,
    max_features=100,
    auto_select_regimes=True,
    min_regimes=2,
    max_regimes=5,
    switching_variance=True
)

# Identify high/low volatility periods
volatilities = result['ms_result'].regime_variances
high_vol_regimes = np.where(volatilities > volatilities.mean())[0]
```

### 3. Custom Configuration

```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    MSDRClusterer, MSDRConfig
)

# Advanced configuration
config = MSDRConfig(
    n_regimes=5,
    model_type='autoregression',
    order=2,  # AR(2) model
    switching_variance=True,
    auto_select_regimes=True,
    min_regimes=3,
    max_regimes=8,
    ic_criterion='bic',  # Use BIC instead of AIC
    enable_pca=True,
    pca_components=15,
    pca_aggregation='weighted_average',  # Use weighted average instead of first component
    min_samples_required=300,
    show_progress=True
)

clusterer = MSDRClusterer(config)
result = clusterer.fit_predict(features)
```

---

## 🔧 PCA Aggregation Strategies

### Option 1: First Component (Default)
```python
config.pca_aggregation = 'first'
```
- Uses first principal component only
- Captures most variance
- Fastest option
- **Recommended for most use cases**

### Option 2: Weighted Average
```python
config.pca_aggregation = 'weighted_average'
```
- Variance-weighted average of all components
- Retains more information
- Slightly slower
- **Better when first component alone is insufficient**

### Option 3: No Aggregation
```python
config.pca_aggregation = 'none'
```
- Keeps all components
- May not work with standard MS models
- **Experimental - use with caution**

---

## 📈 Interpreting Transition Matrices

```python
transition_matrix = result['transition_matrix']

# Example 3-regime transition matrix:
# [[0.95, 0.03, 0.02],   # From regime 0: 95% stay, 3% → regime 1, 2% → regime 2
#  [0.10, 0.85, 0.05],   # From regime 1: 10% → regime 0, 85% stay, 5% → regime 2
#  [0.05, 0.10, 0.85]]   # From regime 2: 5% → regime 0, 10% → regime 1, 85% stay

# Persistence: Diagonal elements (probability of staying in same regime)
persistence = np.diag(transition_matrix)
print(f"Regime persistence: {persistence}")

# Most stable regime (highest self-transition)
most_stable = np.argmax(persistence)
print(f"Most stable regime: {most_stable} (p={persistence[most_stable]:.3f})")

# Transition from regime i to regime j
i, j = 0, 1
prob = transition_matrix[i, j]
print(f"Probability of {i}→{j}: {prob:.3f}")
```

---

## ⚠️ Common Pitfalls

### 1. Not Enough Data
```python
# ❌ Bad: Only 50 samples
result = clusterer.fit_predict(data[:50])  # Will warn or fail

# ✅ Good: At least 200 samples recommended
result = clusterer.fit_predict(data)  # Has 300+ samples
```

### 2. Ignoring Temporal Order
```python
# ❌ Bad: Shuffling time series
shuffled_data = data.sample(frac=1)  # Breaks temporal structure!
result = clusterer.fit_predict(shuffled_data)

# ✅ Good: Preserve temporal order
result = clusterer.fit_predict(data)  # Keep chronological order
```

### 3. Too Many Regimes
```python
# ❌ Bad: More regimes than data supports
result = perform_enhanced_ms_dr_clustering(
    data=data[:100],  # Only 100 samples
    min_regimes=10,
    max_regimes=20    # Way too many!
)

# ✅ Good: Reasonable regime range
result = perform_enhanced_ms_dr_clustering(
    data=data,       # 300+ samples
    min_regimes=2,
    max_regimes=6    # Reasonable range
)
```

### 4. Misinterpreting Clusters
```python
# ❌ Bad: Treating as static clusters
result = clusterer.fit_predict(data)
# Don't use regime_labels for new data classification!

# ✅ Good: Use for temporal regime analysis
result = clusterer.fit_predict(data)
# Analyze how regimes evolve over time
regime_changes = np.diff(result.cluster_labels) != 0
print(f"Number of regime switches: {regime_changes.sum()}")
```

---

## 🐛 Troubleshooting

### Issue: "All regime selection attempts failed"
**Cause:** All models failed to fit  
**Solution:**
- Check data quality (NaN values, outliers)
- Reduce `max_regimes` parameter
- Increase data size
- Check for convergence issues

### Issue: "Input has X samples, but 200+ recommended"
**Cause:** Insufficient data for reliable estimation  
**Solution:**
- Collect more data
- Reduce number of regimes
- Use simpler model (AR(1) instead of AR(2))

### Issue: Poor silhouette scores
**Cause:** Regimes may not be well-separated  
**Solution:**
- Try different feature sets
- Adjust PCA components
- Use different IC criterion (BIC vs AIC)
- Consider if MS-DR is appropriate for your data

### Issue: Convergence warnings
**Cause:** Optimization didn't converge  
**Solution:**
```python
config.max_iter = 2000  # Increase from 1000
config.method = 'bfgs'  # Try different optimizer
```

---

## 📚 Additional Resources

- **Module Documentation:** See comprehensive docstrings in source files
- **Test File:** `minimal_test_ms_dr.py` - Working examples
- **Statsmodels Docs:** [Markov-Switching Models](https://www.statsmodels.org/stable/regime_switching.html)

---

## 💡 Tips for Best Results

1. **Use Sufficient Data:** 200+ samples minimum, 500+ recommended
2. **Preserve Temporal Order:** Don't shuffle time series data
3. **Start Simple:** Begin with 2-5 regimes, increase if needed
4. **Check Metrics:** Compare AIC/BIC for different regime counts
5. **Validate Regimes:** Ensure discovered regimes have economic meaning
6. **Monitor Transitions:** Check transition persistence (should be high)
7. **Use BIC for Selection:** More conservative than AIC, prevents overfitting

---

## 🎓 Advanced Topics

### Parallel Model Selection (Future Enhancement)
Currently model selection is sequential. For large regime ranges, consider:
```python
# Future enhancement idea:
from joblib import Parallel, delayed

def fit_model_parallel(data, k):
    return self._fit_ms_model(data, k, store_model=False)

results = Parallel(n_jobs=-1)(
    delayed(fit_model_parallel)(data, k) 
    for k in range(min_regimes, max_regimes + 1)
)
```

### Custom Feature Engineering
```python
# Create custom features focused on regime properties
features = pd.DataFrame({
    'returns': data['close'].pct_change(),
    'volatility': data['close'].pct_change().rolling(20).std(),
    'volume_surge': data['volume'] / data['volume'].rolling(50).mean(),
    'price_momentum': data['close'] / data['close'].rolling(20).mean()
})

result = perform_enhanced_ms_dr_clustering(features)
```

---

**Last Updated:** 2025-10-28  
**Version:** 2.0 (After comprehensive fixes)
