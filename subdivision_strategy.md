# Strategy: Subdivide Regime 1 & Recluster Noise

## 🎯 Goal
- Subdivide the 48.3% regime into 2-3 sub-regimes
- Reclassify some of the 38.3% noise as actual regimes

## 📊 Current State
- Regime 0: 64 samples (13.3%) - Small, stable
- Regime 1: 232 samples (48.3%) - Large, needs subdivision
- Noise: 184 samples (38.3%) - Some may be legitimate regimes

## 🔬 Why Current Parameters Don't Work

### Parameters Applied:
```
min_cluster_size_pct: 0.01 (1%)
min_cluster_size_floor: 10
min_samples: 5
cluster_selection_method: 'leaf'
cluster_selection_epsilon: 0.001
```

### Why Still Only 2 Regimes:
1. **Leaf method** might still be merging clusters
2. **Epsilon too tight** (0.001) might be creating too many clusters that get merged
3. **Data density structure** might be genuinely bimodal
4. **Not enough feature diversity** to separate sub-regimes

## 🎯 New Strategy: Multi-Stage Approach

### Stage 1: Pre-cluster the Large Regime
Use a 2-step clustering approach:
1. First pass: Find the major regimes (current 2)
2. Second pass: Re-cluster Regime 1 with different parameters

### Stage 2: Re-cluster the Noise
1. Filter out true outliers
2. Apply a second clustering pass on noise points
3. Use KMeans as fallback for noise points

## 🔧 Proposed Parameter Changes

### Approach 1: Two-Pass Clustering (Recommended)
```python
# Pass 1: Find major regimes (current approach)
# Pass 2: Re-cluster large regime with aggressive params

# For Regime 1 re-clustering:
min_cluster_size_pct=0.005,  # 0.5% - very granular
min_cluster_size_floor=5,    # Very low floor
min_samples=3,               # Very flexible
metric='manhattan',          # Different distance
cluster_selection_method='eom',  # Try EOM instead
```

### Approach 2: Change Distance Metric
```python
metric='manhattan'  # More robust to outliers
# Or
metric='cosine'     # For normalized data
```

### Approach 3: Preprocessing Enhancement
```python
# Add regime-specific features
# Create interaction features
# Use different normalization
```

## 💡 Alternative: Feature Engineering

Instead of parameter tuning, create **regime-specific features**:
1. Features that distinguish sub-regimes within Regime 1
2. Features that separate noise from true outliers
3. Temporal features that capture regime transitions

## 🚀 Implementation Plan

### Immediate Steps:
1. Try different distance metric (manhattan/cosine)
2. Test different cluster selection methods
3. Add manual feature engineering for regime subdivision

### If That Doesn't Work:
1. Implement two-pass clustering
2. Use KMeans fallback for noise points
3. Consider hierarchical clustering approach

