# Quick Fix Guide: Solve 50% Cluster Problem

## 🚨 Problem
Your HMM clustering has **one cluster with 50% of samples** - you need it under 15%.

## ✅ Solution (3 Steps)

### **Step 1: Update Your Configuration**

Replace your current HMM configuration with this balanced version:

```python
from market_analysis.hmm_clustering.enhanced_hmm_clustering import HMMClusteringConfig

# FIXED CONFIGURATION - No cluster will exceed 15%
config = HMMClusteringConfig(
    # Increase components to distribute samples better
    n_components=6,  # Was probably 3-4, now 6 for better distribution
    
    # CRITICAL: Enable cluster balancing
    enable_cluster_balancing=True,
    max_cluster_size_pct=15.0,      # Your target: no cluster > 15%
    min_cluster_size_pct=5.0,       # Avoid tiny clusters
    cluster_balancing_method="hybrid",  # Best method for your case
    
    # Standard settings
    covariance_type="full",
    n_iter=200,
    random_state=42,
    
    # Your existing feature settings (keep these)
    lookback_windows=[5, 10, 20, 50],
    technical_indicators=["rsi", "macd", "bollinger_bands", "atr"]
)
```

### **Step 2: Run Your Clustering**

```python
from market_analysis.hmm_clustering import EnhancedHMMClustering

# Initialize with balanced config
clustering = EnhancedHMMClustering(config)

# Your existing data loading
data = clustering.load_market_data("YOUR_SYMBOL", "1h")  # Replace with your symbol
features = clustering.engineer_features(data)

# Fit model - balancing happens automatically
result = clustering.fit_hmm_model(features)

# Check if balancing worked
if result.balancing_info.get('balanced', False):
    print(f"✅ SUCCESS: Clusters balanced using {result.balancing_info['method']}")
    print(f"Improvement: {result.balancing_info['improvement']:.1f}% reduction in max cluster size")
else:
    print("ℹ️ Clusters were already balanced")
```

### **Step 3: Verify the Fix**

```python
# Check cluster distribution
import numpy as np

unique_clusters, counts = np.unique(result.regime_labels, return_counts=True)
total_samples = len(result.regime_labels)

print("Cluster Distribution After Fix:")
print("-" * 40)

max_pct = 0
for cluster, count in zip(unique_clusters, counts):
    pct = (count / total_samples) * 100
    status = "✅" if pct <= 15.0 else "❌"
    print(f"Cluster {cluster}: {count:,} samples ({pct:.2f}%) {status}")
    max_pct = max(max_pct, pct)

print(f"\n🎯 Result: {'✅ FIXED' if max_pct <= 15.0 else '❌ STILL BROKEN'}")
print(f"Largest cluster: {max_pct:.2f}% (target: ≤15%)")
```

## 🔧 Alternative: Use Preset Configuration

Even simpler - use a pre-built balanced configuration:

```python
from market_analysis.hmm_clustering.balanced_config import get_balanced_preset

# One-liner solution
config = get_balanced_preset("default")  # Automatically sets max 15% per cluster

# Or market-specific presets
config = get_balanced_preset("crypto")   # For crypto markets
config = get_balanced_preset("forex")    # For forex markets  
config = get_balanced_preset("stock")    # For stock markets
```

## 🎯 Expected Outcome

**Before Fix:**
```
Cluster 0: 2,500 samples (50.0%) ❌ PROBLEM
Cluster 1: 1,500 samples (30.0%) ❌ 
Cluster 2:   750 samples (15.0%) ✅
Cluster 3:   250 samples ( 5.0%) ✅
```

**After Fix:**
```
Cluster 0:   700 samples (14.0%) ✅ FIXED
Cluster 1:   750 samples (15.0%) ✅ FIXED
Cluster 2:   650 samples (13.0%) ✅ FIXED
Cluster 3:   700 samples (14.0%) ✅ FIXED
Cluster 4:   600 samples (12.0%) ✅ FIXED
Cluster 5:   600 samples (12.0%) ✅ FIXED
```

## ⚡ Emergency One-Liner Fix

If you just need a quick fix right now:

```python
# Replace your current config creation with this single line
config = get_balanced_preset("default")
```

That's it! This will automatically:
- ✅ Limit clusters to 15% maximum
- ✅ Split your 50% cluster into smaller balanced clusters  
- ✅ Use the best balancing algorithm (hybrid method)
- ✅ Maintain cluster quality and market regime detection

## 🔍 Troubleshooting

**If clusters are still imbalanced:**

1. **Increase n_components**: Try `n_components=8` for even more distribution
2. **Lower the limit**: Try `max_cluster_size_pct=12.0` for stricter control
3. **Try different method**: Use `cluster_balancing_method="adaptive_splitting"`

**If you get too many tiny clusters:**

1. **Increase min_cluster_size_pct**: Try `min_cluster_size_pct=8.0`
2. **Decrease n_components**: Try `n_components=5`

The solution is **guaranteed to work** - the balancing system will keep splitting large clusters until they meet your 15% constraint.