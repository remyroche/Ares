# Market State Clustering

## 🎯 **Objective**

Define coherent market states based on implicit market dimensions, balancing within-cluster homogeneity with sufficient sample counts for statistical validity.

## 🔬 **Research Focus**

**Market Dimensions → Market States**

- Optimal cluster number selection
- Multiple clustering methodologies
- Temporal stability validation
- Economic coherence testing

## 📁 **Components**

### **`regime_discovery.py`**
Core clustering algorithms:
- K-Means clustering
- Gaussian Mixture Models (GMM)
- Hierarchical clustering
- DBSCAN for density-based clustering
- Ensemble clustering methods

### **`similarity_clustering.py`**
Similarity-based approaches:
- Correlation-based clustering
- Dynamic Time Warping (DTW) clustering
- Graph-based community detection
- Spectral clustering

### **`optimal_cluster_selection.py`**
Data-driven cluster optimization:
- Elbow method, Silhouette analysis
- Gap statistic, BIC/AIC criteria
- Economic significance thresholds
- Sample size constraints

### **`validation_metrics.py`**
Comprehensive cluster validation:
- **Statistical**: Silhouette, Calinski-Harabasz, Davies-Bouldin
- **Temporal**: Stability, persistence, transition frequency
- **Economic**: Return/volatility separability, Sharpe ratios
- **Trading**: Signal quality, transaction costs

## 🚀 **Usage**

```python
from research.cluster_analysis.clustering import (
    MarketStateClusterer,
    OptimalClusterSelector,
    ClusterValidator
)

# 1. Find optimal number of clusters
selector = OptimalClusterSelector()
optimal_k = selector.find_optimal_clusters(
    market_dimensions,
    k_range=(2, 15),
    min_samples_per_cluster=100
)

# 2. Discover market states
clusterer = MarketStateClusterer()
market_states = clusterer.discover_market_states(
    market_dimensions,
    n_clusters=optimal_k,
    methods=['kmeans', 'gmm', 'hierarchical']
)

# 3. Validate cluster quality
validator = ClusterValidator()
validation_results = validator.validate_clusters(
    market_dimensions, 
    market_states.labels,
    market_data=price_data
)

# 4. Analyze temporal stability
stability_results = validator.analyze_temporal_stability(
    market_states.labels,
    window_size=250  # ~1 year of daily data
)
```

## 📊 **Clustering Constraints**

### **Sample Size Requirements**
- **Minimum**: 100 observations per cluster
- **Target**: 200+ observations per cluster
- **Rationale**: Statistical validity for downstream analysis

### **Homogeneity vs Count Trade-off**
```python
# Balance equation
cluster_quality = (within_cluster_homogeneity * sample_count_adequacy) / n_clusters

# Constraints
min_samples_per_cluster = 100
max_clusters = len(data) // min_samples_per_cluster
optimal_clusters = argmax(cluster_quality) subject to constraints
```

### **Temporal Stability Requirements**
- Cluster assignments should be stable over time
- Avoid excessive regime switching
- Minimum regime duration: 20-50 periods

## 📊 **Expected Market States**

Based on market dimension analysis, expect to discover:

### **Low Volatility, Trending** 
- Low volatility dimension scores
- High momentum dimension scores
- Stable correlation patterns

### **High Volatility, Mean-Reverting**
- High volatility dimension scores  
- High mean-reversion dimension scores
- Unstable correlation patterns

### **High Volume, Breakout**
- High volume dimension scores
- Low correlation dimension scores
- Mixed momentum/reversion signals

### **Low Volume, Consolidation**
- Low volume dimension scores
- Low momentum dimension scores
- High correlation dimension scores

### **Microstructure Stress**
- High microstructure dimension scores
- High volatility dimension scores
- Extreme correlation patterns

## 📊 **Outputs**

### **Market State Labels**
```python
market_states = pd.Series([0, 0, 1, 2, 1, 0, ...])  # State assignments
state_probabilities = pd.DataFrame({
    'state_0': [0.8, 0.7, 0.1, 0.0, 0.2, 0.9, ...],
    'state_1': [0.1, 0.2, 0.8, 0.1, 0.7, 0.1, ...],
    'state_2': [0.1, 0.1, 0.1, 0.9, 0.1, 0.0, ...]
})
```

### **Cluster Characteristics**
```python
cluster_profiles = {
    'state_0': {
        'volume_score': 0.3,      # Low volume
        'volatility_score': 0.2,  # Low volatility  
        'momentum_score': 0.8,    # High momentum
        'description': 'Low Vol Trending'
    },
    'state_1': {
        'volume_score': 0.7,      # High volume
        'volatility_score': 0.9,  # High volatility
        'momentum_score': 0.2,    # Low momentum
        'description': 'High Vol Mean Reverting'
    }
}
```

### **Validation Metrics**
- Silhouette scores, cluster separation
- Temporal stability measures
- Economic significance tests
- Trading utility assessments

## 🔗 **Integration**

**Input Sources:**
- `market_factor_analysis/`: Market dimension features
- Price data for validation
- Volume data for context

**Downstream Usage:**
1. **Economic Relevance**: Analyze how patterns behave in different states
2. **ML Training**: Use states for regime-aware model training  
3. **Trading Systems**: Generate state-dependent signals

**Key Outputs for Next Steps:**
- `market_states.csv`: State labels and probabilities
- `cluster_profiles.json`: State characteristics and interpretations
- `validation_results.json`: Cluster quality and stability metrics