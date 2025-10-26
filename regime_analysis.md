# Why Only 2 Regimes Instead of 4-8?

## 🔍 Root Cause Analysis

### Current Situation:
- **Regimes Found**: 2 (Regime 0: 64 samples, Regime 1: 232 samples)
- **Noise**: 184 samples (38.3%)
- **Total Samples**: 480

### Effective Parameters:
```
min_cluster_size_pct: 0.015 (1.5%)
min_cluster_size_floor: 25
Data size: 480 samples
Effective min_cluster_size: max(480 * 0.015, 25) = max(7.2, 25) = 25 samples
```

## 🚨 Problem Identified

### 1. **Data Size Issue**
- Only **480 samples** in 1h timeframe
- With min_cluster_size=25, maximum possible clusters: 480/25 = **~19 clusters**
- But with noise at 38.3%, only 296 valid samples → max ~12 clusters
- **NOT the limiting factor**

### 2. **Feature Count Issue**
- Only **17 features** after preprocessing
- Correlation threshold of 0.95 is removing features
- HDBSCAN needs sufficient dimensionality to separate regimes

### 3. **Clustering Algorithm Behavior**
- HDBSCAN is finding natural density clusters
- With current data, it's naturally separating into 2 major regimes
- The 38.3% noise suggests data isn't well-separated

## 💡 Why It's Only Finding 2 Regimes

### **Hypothesis 1: Insufficient Feature Discrimination**
- Only 17 features may not capture enough market regime variation
- Correlation threshold too aggressive (0.95)
- Need more diverse features

### **Hypothesis 2: Natural Data Structure**
- Market only has 2 distinct regimes in this time period
- Regime 0: Low volatility (64 samples, 13.3%)
- Regime 1: High volatility (232 samples, 48.3%)
- Noise: Transitional/uncertain periods (184 samples, 38.3%)

### **Hypothesis 3: Distance Metric Issue**
- Using 'euclidean' distance by default
- May not capture regime differences well
- Could try 'manhattan' or 'cosine'

### **Hypothesis 4: Cluster Selection Method**
- Using EOM (Excess of Mass) method
- Leaf method might create more balanced clusters
- EOM might be merging clusters

## 🎯 Solutions to Get 4-8 Regimes

### Solution 1: Increase Feature Count
```python
correlation_threshold=0.85  # Lower threshold = more features
# Target: 30-50 features instead of 17
```

### Solution 2: Use Different Distance Metric
```python
metric='manhattan'  # More robust to outliers
# Or 'cosine' for normalized data
```

### Solution 3: Use Leaf Method
```python
cluster_selection_method='leaf'  # Creates more balanced clusters
# Instead of 'eom' which merges clusters
```

### Solution 4: Reduce Min Cluster Size More Aggressively
```python
min_cluster_size_pct=0.008  # 0.8% instead of 1.5%
min_cluster_size_floor=15   # Lower floor
# Target smaller clusters
```

### Solution 5: Adjust Epsilon More Aggressively
```python
cluster_selection_epsilon=0.005  # Much tighter clusters
# Instead of 0.02
```

## 📊 Recommended Parameter Combination

```python
# For 4-8 regimes with lower noise:
{
    'min_cluster_size_pct': 0.008,      # 0.8% for more clusters
    'min_cluster_size_floor': 15,       # Lower floor
    'min_samples': 10,                  # Lower samples for more flexibility
    'cluster_selection_method': 'leaf', # Balanced clusters
    'cluster_selection_epsilon': 0.005, # Tight clusters
    'metric': 'manhattan',              # Robust distance
    'correlation_threshold': 0.85       # More features
}
```

## 🔬 Investigation Needed

1. **Check feature diversity** - Are all 17 features distinct?
2. **Visualize cluster structure** - Is data naturally bimodal?
3. **Test different timeframes** - Does 15m have more samples?
4. **Reduce feature filtering** - Keep more features

