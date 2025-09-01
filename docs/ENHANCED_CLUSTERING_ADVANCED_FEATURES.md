# Enhanced Clustering Advanced Features

## Overview

The Enhanced Regime Clustering System now includes three advanced features that make it more intelligent and automated:

1. **LIME/SHAP Explainable AI**: Understand which features most define each cluster
2. **Smart Splitting**: Select clusters for splitting based on internal quality metrics
3. **Automated K-means**: Automatically determine optimal number of clusters

## 🤖 LIME/SHAP Explainable AI

### Purpose
LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) provide qualitative feedback to complement quantitative clustering scores by showing which features are most important for defining each cluster.

### Implementation

```python
def analyze_cluster_with_lime_shap(self, features, labels, feature_names, cluster_id):
    """Analyze cluster characteristics using LIME and SHAP."""
    
    # Create a classifier to predict cluster membership
    cluster_labels = (labels == cluster_id).astype(int)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(features, cluster_labels)
    
    # LIME Analysis
    explainer = lime.lime_tabular.LimeTabularExplainer(
        features, feature_names=feature_names, 
        class_names=['not_cluster', 'cluster'], mode='classification'
    )
    
    # SHAP Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_features)
    
    # Combine LIME and SHAP importance
    combined_importance = (lime_weights + shap_weights) / 2
```

### Benefits
- **Qualitative Insights**: Understand why points belong to specific clusters
- **Feature Importance**: Identify the most defining features per cluster
- **Interpretability**: Make clustering results more explainable
- **Validation**: Verify that clusters make intuitive sense

### Configuration
```python
config = {
    "use_lime_shap": True,        # Enable/disable explainable AI
    "lime_samples": 500,          # Number of LIME samples
    "shap_samples": 50,           # Number of SHAP samples
}
```

### Fallback
If LIME/SHAP is not available, the system falls back to variance-based feature importance:
```python
importance_scores = cluster_var / (overall_var + 1e-8)
```

## 🎯 Smart Splitting

### Purpose
Instead of arbitrarily splitting the largest cluster, smart splitting selects the cluster with the lowest internal silhouette score - the "unhappiest" cluster that would benefit most from being split.

### Implementation

```python
def select_cluster_for_splitting(self, features, labels):
    """Smart cluster selection based on internal silhouette scores."""
    
    # Calculate silhouette scores for each cluster
    cluster_silhouettes = self.calculate_cluster_silhouette_scores(features, labels)
    
    # Filter by minimum size
    valid_clusters = {
        cluster_id: sil_score 
        for cluster_id, sil_score in cluster_silhouettes.items()
        if sum(labels == cluster_id) >= self.min_cluster_size_for_split
    }
    
    # Select cluster with lowest silhouette score
    worst_cluster = min(valid_clusters, key=valid_clusters.get)
    return worst_cluster
```

### Benefits
- **Quality-Driven**: Split clusters that would benefit most from splitting
- **Intelligent Selection**: Avoid splitting well-formed clusters
- **Better Results**: Improve overall clustering quality
- **Efficient**: Focus computational effort where it matters most

### Configuration
```python
config = {
    "smart_splitting": True,              # Enable/disable smart splitting
    "min_cluster_size_for_split": 20,     # Minimum size for splitting
}
```

### Fallback
If smart splitting is disabled, the system falls back to selecting the largest cluster:
```python
largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
```

## 🔧 Automated K-means

### Purpose
Automatically determine the optimal number of clusters (k) for K-means operations using either the Elbow Method or Silhouette Method, removing the need for manual parameter tuning.

### Implementation

```python
def find_optimal_k_automated(self, features, max_k=10, method="silhouette"):
    """Automatically determine optimal k using Elbow or Silhouette method."""
    
    if method == "silhouette":
        # Silhouette Method
        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            sil_score = silhouette_score(features, cluster_labels)
            silhouette_scores.append(sil_score)
        
        optimal_k = range(2, max_k + 1)[np.argmax(silhouette_scores)]
        
    elif method == "elbow":
        # Elbow Method
        inertias = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point using second derivative
        second_derivatives = [
            inertias[i+1] - 2*inertias[i] + inertias[i-1]
            for i in range(1, len(inertias) - 1)
        ]
        elbow_idx = np.argmax(second_derivatives)
        optimal_k = range(2, max_k + 1)[elbow_idx + 1]
    
    return optimal_k
```

### Methods

#### 1. Silhouette Method
- **Principle**: Choose k that maximizes silhouette score
- **Advantage**: Direct quality metric
- **Disadvantage**: Can be computationally expensive

#### 2. Elbow Method
- **Principle**: Find the "elbow" point where inertia reduction slows
- **Advantage**: Faster computation
- **Disadvantage**: Subjective interpretation

### Benefits
- **Automated**: No manual parameter tuning required
- **Adaptive**: Optimal k varies with data characteristics
- **Robust**: Multiple methods available
- **Efficient**: Focuses on relevant k range

### Configuration
```python
config = {
    "auto_k_means": True,                 # Enable/disable automated k selection
    "max_k_for_auto": 8,                  # Maximum k to consider
    "k_selection_method": "silhouette",   # "silhouette" or "elbow"
}
```

### Usage Examples

#### 1. Noise Point Clustering
```python
# Automatically determine optimal k for noise points
if self.auto_k_means:
    n_noise_clusters = self.find_optimal_k_automated(
        noise_features, 
        max_k=min(5, noise_count // 10), 
        method=self.k_selection_method
    )
```

#### 2. Cluster Splitting
```python
# Automatically determine optimal k for splitting
optimal_k = self.find_optimal_k_automated(
    cluster_features, 
    max_k=self.max_k_for_auto, 
    method=self.k_selection_method
)
```

## 📊 Integration with Main Pipeline

### Enhanced Configuration
```python
enhanced_config = {
    # Basic settings
    "target_clusters": target_clusters,
    "min_quality_threshold": 0.3,
    "quality_drop_threshold": 0.8,
    "max_iterations": 50,
    "no_improvement_limit": 10,
    "min_coverage_threshold": 0.98,
    "bayesian_calls": 50,
    
    # Explainable AI settings
    "use_lime_shap": True,
    "lime_samples": 500,
    "shap_samples": 50,
    
    # Smart splitting settings
    "smart_splitting": True,
    "min_cluster_size_for_split": 20,
    
    # Automated K-means settings
    "auto_k_means": True,
    "max_k_for_auto": 8,
    "k_selection_method": "silhouette"
}
```

### Enhanced Reporting
The system now generates reports with:

1. **Explainable AI Insights**: Feature importance per cluster
2. **Smart Splitting Decisions**: Which clusters were split and why
3. **Automated K Selection**: Optimal k values chosen
4. **Quality Metrics**: Before and after improvements

### Example Report Section
```
🤖 EXPLAINABLE AI SUMMARY
----------------------------------------
Top Features by Explainable AI Importance:
   1. momentum_15: 0.234 (LIME/SHAP importance)
   2. volatility_20: 0.189 (LIME/SHAP importance)
   3. volume_15: 0.156 (LIME/SHAP importance)
   4. sr_ratio: 0.123 (LIME/SHAP importance)
   5. hmm_state: 0.098 (LIME/SHAP importance)

Analysis based on 15 clusters with explainable AI data
```

## 🚀 Performance Considerations

### Computational Cost
- **LIME/SHAP**: O(samples × features) per cluster
- **Smart Splitting**: O(n log n) for silhouette calculation
- **Automated K**: O(k × n × features) for k selection

### Optimization Strategies
1. **Sampling**: Use subset of data for LIME/SHAP analysis
2. **Caching**: Cache silhouette scores during refinement
3. **Early Stopping**: Stop k selection when clear optimum found
4. **Parallelization**: Process multiple clusters simultaneously

### Memory Usage
- **LIME**: Requires model training per cluster
- **SHAP**: Stores SHAP values for samples
- **Silhouette**: Stores pairwise distances

## 🔍 Troubleshooting

### Common Issues

1. **LIME/SHAP Not Available**
   - Install: `pip install lime shap`
   - System falls back to variance-based importance

2. **Slow Performance**
   - Reduce `lime_samples` and `shap_samples`
   - Disable explainable AI for large datasets
   - Use "elbow" method instead of "silhouette"

3. **Poor K Selection**
   - Adjust `max_k_for_auto`
   - Try different `k_selection_method`
   - Check data quality and scaling

### Debug Mode
```python
import logging
logging.getLogger('src.training.steps.enhanced_regime_clustering').setLevel(logging.DEBUG)
```

## 📋 Dependencies

### Required for Advanced Features
```bash
pip install lime>=0.2.0 shap>=0.41.0
```

### Optional Dependencies
- `matplotlib>=3.7.0` (for visualizations)
- `seaborn>=0.12.0` (for enhanced plots)

## 🎯 Best Practices

1. **Start Simple**: Begin with basic clustering, then enable advanced features
2. **Monitor Performance**: Watch execution time and memory usage
3. **Validate Results**: Check that explainable AI insights make sense
4. **Iterate**: Adjust parameters based on results and performance
5. **Document**: Keep track of which features work best for your data

## 🔮 Future Enhancements

1. **Multi-Modal Explanations**: Combine LIME/SHAP with other explainability methods
2. **Dynamic Sampling**: Adaptive sample sizes based on cluster characteristics
3. **Ensemble K Selection**: Combine multiple methods for robust k selection
4. **Real-Time Analysis**: Online explainable AI for streaming data
5. **Interactive Visualization**: Web-based cluster exploration tools