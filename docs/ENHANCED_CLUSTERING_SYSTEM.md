# Enhanced Regime Clustering System

## Overview

The Enhanced Regime Clustering System implements a sophisticated, quality-driven approach to regime discovery that combines DBSCAN with Bayesian optimization and hybrid refinement strategies. This system ensures high-quality clusters while maintaining computational efficiency.

## 🎯 Key Features

### 1. **Quality-Driven DBSCAN + Bayesian Optimization**
- Uses DBSCAN to find natural clusters based on data density
- Employs Bayesian optimization to intelligently search parameter space
- Avoids brute-force grid search for better efficiency
- Falls back to grid search if Bayesian optimization is unavailable

### 2. **Intelligent Noise Point Handling**
- **Strategy 1**: Cluster noise points if sufficient quantity (>50 points)
- **Strategy 2**: Assign noise points to nearest existing clusters
- Ensures no data points are lost during clustering

### 3. **Hybrid Refinement with Quality Preservation**
- **Split Strategy**: Split large, low-quality clusters
- **Merge Strategy**: Merge small, similar clusters
- **Quality-Driven**: Every change must improve composite score
- **Early Stopping**: Multiple conditions to prevent over-optimization

### 4. **Comprehensive Quality Metrics**
```python
Composite Score = (0.4 × Silhouette) + (0.2 × Calinski-Harabasz) - 
                  (0.2 × Davies-Bouldin) - (0.1 × Skew Penalty) - 
                  (0.1 × Volatility Penalty)
```

### 5. **Adaptive Target Clusters**
- **Light Mode**: 2 clusters
- **Blank Mode**: 4 clusters  
- **Full Mode**: 20 clusters
- Stops early if quality drops below threshold

## 🚀 Implementation Details

### Configuration Parameters

```python
enhanced_config = {
    "target_clusters": 20,              # Target number of clusters
    "min_quality_threshold": 0.3,       # Minimum acceptable quality
    "quality_drop_threshold": 0.8,      # Stop if quality drops below 80% of initial
    "max_iterations": 50,               # Maximum refinement iterations
    "no_improvement_limit": 10,         # Stop after 10 iterations without improvement
    "min_coverage_threshold": 0.98,     # Minimum data coverage (98%)
    "bayesian_calls": 50                # Number of Bayesian optimization calls
}
```

### Quality Metrics Explained

1. **Silhouette Score (40%)**: Measures cluster separation and cohesion
2. **Calinski-Harabasz Score (20%)**: Ratio of between-cluster to within-cluster variance
3. **Davies-Bouldin Score (20%)**: Average similarity measure of clusters
4. **Skew Penalty (10%)**: Penalizes uneven cluster size distribution
5. **Volatility Penalty (10%)**: Penalizes small clusters (<5 points)

### Early Stopping Conditions

1. **Target Reached**: Achieved desired number of clusters
2. **Quality Drop**: Score drops below 80% of initial quality
3. **Coverage Threshold**: 98%+ of data points covered
4. **No Improvement**: 10 consecutive iterations without improvement
5. **Max Iterations**: Reached maximum iteration limit

## 📊 Comprehensive Reporting

The system generates detailed reports including:

### Executive Summary
- Target vs. final clusters
- Data coverage percentage
- Composite quality score
- Quality improvement metrics

### Clustering Process
- Initial vs. final clusters
- Number of iterations
- Quality improvements
- Execution time

### Quality Metrics
- Individual metric scores
- Composite score breakdown
- Penalty analysis

### Cluster Analysis
- Top clusters by size
- Feature importance ranking
- Cluster characteristics
- Iteration history

### Cluster Characteristics
- Feature means and standard deviations
- Z-scores vs. overall distribution
- Most distinctive features per cluster

## 🔧 Usage Examples

### Basic Usage
```python
from src.training.steps.enhanced_regime_clustering import EnhancedRegimeClustering

# Initialize with configuration
config = {
    "target_clusters": 20,
    "bayesian_calls": 100
}

enhanced_clustering = EnhancedRegimeClustering(config)

# Run clustering
results = enhanced_clustering.run_enhanced_clustering(features, feature_names)

# Access results
final_labels = results["final_labels"]
report = results["report"]
```

### Integration with Training Pipeline
The enhanced clustering is automatically integrated into `step03_5_final_regime_clustering.py` and respects the training mode:

- **Light Mode**: 2 clusters
- **Blank Mode**: 4 clusters
- **Full Mode**: 20 clusters

## 📈 Performance Characteristics

### Computational Complexity
- **DBSCAN**: O(n log n) average case
- **Bayesian Optimization**: O(k) where k is number of calls
- **Hybrid Refinement**: O(n × iterations × clusters)

### Memory Usage
- Scales linearly with data size
- Efficient numpy operations
- Minimal memory overhead

### Quality Guarantees
- **Monotonic Improvement**: Score never decreases during refinement
- **Coverage Preservation**: Maintains high data coverage
- **Cluster Stability**: Robust to parameter variations

## 🎛️ Advanced Configuration

### Bayesian Optimization Tuning
```python
config = {
    "eps_range": (0.01, 2.0),           # DBSCAN eps parameter range
    "min_samples_range": (2, 50),       # DBSCAN min_samples range
    "bayesian_calls": 100               # Optimization iterations
}
```

### Quality Thresholds
```python
config = {
    "min_quality_threshold": 0.3,       # Minimum acceptable quality
    "quality_drop_threshold": 0.8,      # Stop if quality drops
    "min_coverage_threshold": 0.98      # Minimum data coverage
}
```

### Refinement Parameters
```python
config = {
    "max_iterations": 50,               # Maximum refinement steps
    "no_improvement_limit": 10,         # Stop after no improvement
    "min_clusters": 5,                  # Minimum clusters allowed
    "max_clusters": 30                  # Maximum clusters allowed
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Low Quality Scores**
   - Check feature scaling
   - Verify data preprocessing
   - Adjust quality thresholds

2. **Slow Performance**
   - Reduce Bayesian optimization calls
   - Use grid search fallback
   - Limit maximum iterations

3. **Poor Cluster Distribution**
   - Adjust skew penalty weight
   - Modify volatility penalty
   - Check feature importance

### Debug Mode
Enable detailed logging:
```python
import logging
logging.getLogger('src.training.steps.enhanced_regime_clustering').setLevel(logging.DEBUG)
```

## 📋 Dependencies

### Required
- `scikit-learn>=1.3.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scipy>=1.10.0`

### Optional
- `scikit-optimize>=0.9.0` (for Bayesian optimization)
- `matplotlib>=3.7.0` (for visualizations)
- `seaborn>=0.12.0` (for enhanced plots)

## 🚀 Future Enhancements

1. **Multi-Objective Optimization**: Balance quality vs. interpretability
2. **Dynamic Feature Selection**: Adaptive feature importance
3. **Ensemble Methods**: Combine multiple clustering approaches
4. **Real-time Adaptation**: Online clustering updates
5. **Visualization Tools**: Interactive cluster exploration

## 📚 References

1. Ester, M., et al. "A density-based algorithm for discovering clusters in large spatial databases with noise." KDD 1996.
2. Snoek, J., et al. "Practical Bayesian optimization of machine learning algorithms." NeurIPS 2012.
3. Rousseeuw, P. J. "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." J. Comput. Appl. Math. 1987.