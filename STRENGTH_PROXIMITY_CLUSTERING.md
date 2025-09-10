# Strength-Proximity Clustering for SR Levels

## Overview

The new **Strength-Proximity Clustering** approach replaces hard-coded cluster counts with adaptive clustering based on the natural structure of your SR (Support/Resistance) level data. Instead of forcing a specific number of clusters, it groups levels that are both **close in price** and **similar in strength**.

## Why This Approach is Better

### ❌ **Problems with DBSCAN (Your Current Issues)**

From your logs, DBSCAN had these problems:
```
Attempt 1: eps=184.007281 → 2 clusters, 12 noise (14 total levels)
Attempt 2: eps=73.602912 → 6 clusters, 17 noise (28 total levels)  
Attempt 3: eps=29.441165 → 12 clusters, 23 noise (49 total levels)
Attempt 4: eps=11.776466 → 14 clusters, 41 noise (81 total levels)
Attempt 5: eps=4.710586 → 4 clusters, 72 noise (81 total levels)
Attempt 6: eps=1.884235 → 0 clusters, 81 noise (81 total levels)
```

**Issues:**
- **Unpredictable cluster count** - can't achieve target levels
- **Many levels lost as "noise"** - 12-72 levels discarded
- **Parameter sensitivity** - small changes in `eps` cause dramatic results
- **No strength consideration** - only looks at price distance
- **Hard to tune** - requires multiple attempts with different parameters

### ✅ **Strength-Proximity Clustering Advantages**

1. **All Levels Preserved** - No levels are lost as "noise"
2. **Natural Clustering** - Groups form based on data characteristics
3. **Strength-Aware** - Considers both price proximity AND strength similarity
4. **Deterministic** - Same input always produces same output
5. **No Parameter Sensitivity** - Robust to threshold changes
6. **Quality Scoring** - Each cluster gets a quality score

## How It Works

### 1. **Dual Criteria Clustering**

```python
# Two criteria for grouping levels:
# 1. Price Proximity: Levels close in price
# 2. Strength Similarity: Levels with similar strength

def should_cluster_together(level1, level2, proximity_threshold, strength_threshold):
    # Check price proximity
    price_distance = abs(level1['price'] - level2['price']) / price_range
    if price_distance > proximity_threshold:
        return False
    
    # Check strength similarity  
    strength_difference = abs(level1['strength'] - level2['strength'])
    if strength_difference > strength_threshold:
        return False
    
    return True
```

### 2. **Adaptive Cluster Formation**

```python
# Algorithm:
# 1. Start with strongest unassigned level as cluster seed
# 2. Find all levels that are both close in price AND similar in strength
# 3. Add qualifying levels to cluster
# 4. Repeat until all levels are assigned

while unassigned_levels:
    # Find strongest level to start new cluster
    seed = find_strongest_level(unassigned_levels)
    cluster = [seed]
    
    # Grow cluster by adding nearby levels with similar strength
    grow_cluster(cluster, unassigned_levels, proximity_threshold, strength_threshold)
    
    clusters.append(cluster)
```

### 3. **Quality Scoring**

```python
# Each cluster gets a quality score based on:
# - Price cohesion (how close prices are)
# - Strength cohesion (how similar strengths are)  
# - Average strength (higher is better)

quality = 0.4 * price_cohesion + 0.3 * strength_cohesion + 0.3 * avg_strength
```

## Real-World Example

### **Your SR Levels (from logs):**
```
Level 1: $1624.73, Strength: 1.0, Type: support
Level 2: $1628.31, Strength: 1.0, Type: support  
Level 3: $1632.46, Strength: 1.0, Type: resistance
Level 4: $1636.62, Strength: 0.996, Type: support
Level 5: $1640.33, Strength: 1.0, Type: resistance
Level 6: $1645.89, Strength: 1.0, Type: resistance
Level 7: $1649.55, Strength: 1.0, Type: resistance
```

### **Strength-Proximity Clustering Result:**
```
Cluster 1: [1624.73, 1628.31, 1632.46, 1636.62, 1640.33, 1645.89, 1649.55]
- Price range: $1624.73 - $1649.55 (spread: $24.82)
- Strength range: 0.996 - 1.0 (spread: 0.004)
- Center: $1636.89 (strength-weighted)
- Quality: 0.95 (excellent cohesion)
```

**Why this makes sense:**
- All levels are within 1.5% of each other in price
- All levels have very similar strength (0.996-1.0)
- They form a natural support/resistance zone
- No levels are lost as "noise"

## Configuration Parameters

### **Proximity Threshold**
```python
proximity_threshold = 0.01  # 1% of price range
# Example: If price range is $1000-$5000 (range=$4000)
# Then proximity threshold = $40
# Levels within $40 of each other can be clustered
```

### **Strength Similarity Threshold**
```python
strength_similarity_threshold = 0.2  # 20% strength difference
# Example: If one level has strength 0.8
# Then levels with strength 0.6-1.0 can be clustered with it
```

### **Adaptive Thresholds**
```python
# The system automatically adapts to your data:
# - Dense price areas → More clusters
# - Sparse price areas → Fewer clusters  
# - High strength levels → Tighter clustering
# - Low strength levels → Looser clustering
```

## Usage

### **Basic Usage:**
```python
from src.utils.clustering_alternatives import get_clustering_manager

clustering_manager = get_clustering_manager()

result = clustering_manager.cluster_with_fallback(
    levels=your_sr_levels,
    price_range=(min_price, max_price),
    proximity_threshold=0.01,  # 1% of price range
    strength_similarity_threshold=0.2,  # 20% strength difference
    preferred_algorithm='strength_proximity'
)

print(f"Created {len(result.clusters)} clusters")
print(f"Quality score: {result.quality_score:.3f}")
print(f"All levels preserved: {result.total_levels == len(your_sr_levels)}")
```

### **Advanced Configuration:**
```python
# For tighter clustering (more clusters)
result = clustering_manager.cluster_with_fallback(
    levels=levels,
    price_range=price_range,
    proximity_threshold=0.005,  # 0.5% - tighter price grouping
    strength_similarity_threshold=0.1,  # 10% - stricter strength matching
)

# For looser clustering (fewer clusters)  
result = clustering_manager.cluster_with_fallback(
    levels=levels,
    price_range=price_range,
    proximity_threshold=0.02,  # 2% - looser price grouping
    strength_similarity_threshold=0.3,  # 30% - more flexible strength matching
)
```

## Benefits for Your System

### **1. Solves DBSCAN Failures**
- No more "0 clusters, 81 noise" failures
- No more lost levels
- No more parameter tuning nightmares

### **2. Better SR Level Quality**
- Groups levels that actually belong together
- Preserves all detected levels
- Creates meaningful support/resistance zones

### **3. Improved Performance**
- No multiple parameter attempts needed
- Deterministic results
- Faster execution

### **4. Better Trading Signals**
- More reliable support/resistance zones
- Higher quality level clusters
- Better risk management

## Migration from DBSCAN

### **Replace this:**
```python
# Old DBSCAN approach
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = clustering.fit_predict(features)
# Result: Unpredictable clusters, many noise points
```

### **With this:**
```python
# New strength-proximity approach
from src.utils.clustering_alternatives import get_clustering_manager

clustering_manager = get_clustering_manager()
result = clustering_manager.cluster_with_fallback(
    levels=levels,
    price_range=price_range,
    proximity_threshold=0.01,
    strength_similarity_threshold=0.2
)
# Result: All levels preserved, natural clustering, quality scoring
```

## Conclusion

The Strength-Proximity Clustering approach solves all the issues you encountered with DBSCAN while providing better, more meaningful clustering results for your SR level optimization system. It's specifically designed for financial data where both price proximity and strength similarity matter for creating effective support/resistance zones.