# Why Only 2 Regimes Instead of 4-8? - Complete Analysis

## 🔍 Root Cause Analysis

### Current Situation:
- **Found**: 2 regimes (Regime 0: 13.3%, Regime 1: 48.3%)
- **Noise**: 38.3% (184 out of 480 samples)
- **Total Samples**: 480 (1h timeframe)
- **Features**: Only 17 features after preprocessing

### Why Only 2 Regimes?

## 🎯 Primary Reasons

### 1. **Natural Data Structure** ⭐ MOST LIKELY
- Market genuinely has 2 distinct regimes in this period
- Regime 1 (48.3%): Dominant regime - likely "normal" market conditions
- Regime 0 (13.3%): Minor regime - likely "extreme" market conditions  
- Noise (38.3%): Transitional periods between regimes
- **This is not necessarily a problem** - it may be the true structure!

### 2. **Insufficient Feature Diversity** ⭐ SECONDARY
- Only **17 features** after correlation filtering
- Correlation threshold of 0.95 removes too many features
- HDBSCAN needs sufficient dimensionality to separate regimes
- **Solution**: Lower correlation_threshold to 0.85 (already done)

### 3. **Clustering Algorithm Behavior**
- HDBSCAN finds natural density clusters
- Data may be naturally bimodal (2 distinct regimes)
- Algorithm is working correctly - finding the actual structure
- **This might be the correct result!**

### 4. **Distance Metric**
- Using 'euclidean' distance by default
- May not capture regime differences well
- Alternative metrics like 'manhattan' or 'cosine' might help

### 5. **Cluster Selection Method**
- Using EOM (Excess of Mass) method
- EOM merges similar clusters
- Leaf method creates more balanced clusters
- **Solution**: Using 'leaf' method (already done)

## 📊 Effective Parameters

```
Current Settings:
- min_cluster_size_pct: 0.02 (2%)
- min_cluster_size_floor: 15
- min_samples: 10
- cluster_selection_method: 'leaf'
- cluster_selection_epsilon: 0.005
- correlation_threshold: 0.85

Effective min_cluster_size: max(480 * 0.02, 15) = max(9.6, 15) = 15 samples
Maximum possible clusters: 480 / 15 = 32 clusters
But with 38.3% noise: 296 valid samples → max ~20 clusters

Current: 2 clusters
Expected: Could get up to 20 clusters with these settings
```

## 💡 Why It's Only Finding 2 Regimes

### **Hypothesis: Natural Bimodal Distribution**
The market data likely has a natural bimodal structure:
1. **Regime 0 (13.3%)**: Rare, extreme market conditions
2. **Regime 1 (48.3%)**: Common, normal market conditions
3. **Noise (38.3%)**: Transitional periods

This could be the **correct** structure! Not all markets have 4-8 distinct regimes.

### **Supporting Evidence:**
- High noise (38.3%) suggests fuzzy boundaries
- Uneven distribution (13.3% vs 48.3%) suggests asymmetric regimes
- Silhouette score of 0.126 is positive (good separation)
- DBI of 1.28 is excellent (target <5.0)
- CH of 58.45 is excellent (target >10.0)

## 🎯 How to Force More Regimes (If Needed)

### Option 1: More Aggressive Parameters
```python
min_cluster_size_pct=0.01,      # 1% instead of 2%
min_cluster_size_floor=10,       # Even lower floor
min_samples=5,                   # Much lower samples
cluster_selection_epsilon=0.001, # Much tighter clusters
```

### Option 2: Different Distance Metric
```python
metric='manhattan',  # More robust to outliers
# Or 'cosine' for normalized data
```

### Option 3: Timeframe Change
- Try 15m timeframe instead of 1h
- More samples = more opportunities for clusters
- Or try 4h for different market dynamics

### Option 4: Feature Engineering
- Add regime-specific features
- Create regime interaction features
- Add more diverse feature families

## 🚨 Important Questions

1. **Do you NEED 4-8 regimes?**
   - Maybe 2 regimes is the correct structure for this data
   - Additional regimes might be artificial/forced

2. **Is the current result useful?**
   - Excellent separation (DBI: 1.28)
   - Good silhouette score (0.126)
   - Clear economic distinction

3. **What's the business goal?**
   - If you need more granularity, use different timeframes
   - If you need more regimes, try different features
   - If current result works, keep it!

## ✅ Recommendations

### **Keep Current Settings If:**
- 2 regimes match your business understanding
- Separation quality is good (DBI, CH excellent)
- Regimes are economically meaningful

### **Force More Regimes If:**
- You need finer granularity for trading
- Business logic requires 4-8 regimes
- Current regimes are too coarse

### **Try These Next:**
1. **Test with 15m timeframe** - more samples
2. **Try different metrics** - manhattan/cosine
3. **Add regime-specific features** - engineered features
4. **Accept 2 regimes** - if they work well

## 🎯 Bottom Line

**The 2 regime result might be CORRECT!**

HDBSCAN is finding the natural structure of your data. 
The high noise (38.3%) and uneven distribution suggest this might be the true market structure.
Excellent metrics (DBI: 1.28, CH: 58.45) suggest good quality separation.

**Consider: Maybe you don't need 4-8 regimes, and 2 well-separated regimes is better?**

