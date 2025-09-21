# Regime Clustering Pipeline

This package clusters small HMM regimes into larger, coherent clusters suitable for ML model training. The HMM regime discovery creates 537 small regimes based on 3D features (Momentum, Volatility, Volume), and this pipeline consolidates them into ~20 clusters of 3-8% each with <5% noise.

## Problem Statement

The HMM regime discovery creates many very small regimes:
- **537 total regimes** with average of 32.3 samples each
- **Largest regime**: only 1.69% of total data
- **Most regimes**: much smaller than 1%
- **3D structure**: Momentum (0-8), Volatility (0-7), Volume (0-8)

## Solution Approach

### 1. Hierarchical Clustering
- Uses **Agglomerative Clustering** on regime centroids in 3D space
- **Ward linkage** for compact, spherical clusters
- **Standardized coordinates** for proper distance calculation

### 2. Size Constraints
- **Target clusters**: ~20 clusters
- **Size range**: 3-8% of total data each
- **Noise threshold**: <5% of total data
- **Constraint enforcement**: Merges small clusters, splits large ones

### 3. Quality Validation
- **Internal coherence**: Within-cluster similarity
- **Validity metrics**: Silhouette score, Calinski-Harabasz, Davies-Bouldin
- **Distinction**: Between-cluster differences
- **Size compliance**: Constraint satisfaction

### 4. Cluster Interpretation
- **Market type classification**: Quiet, Active, Volatile, Trending, etc.
- **Risk assessment**: Low/Medium/High based on volatility
- **Trading implications**: Strategy recommendations per cluster

## Usage

### Basic Usage

```python
from src.training.steps.market_analysis.regime_clustering.main_clustering_pipeline import RegimeClusteringPipeline
from src.training.steps.market_analysis.regime_clustering.config import get_config_template

# Use predefined configuration
config = get_config_template('balanced')
pipeline = RegimeClusteringPipeline(config.to_dict())

# Run clustering
results = pipeline.run_clustering_pipeline(
    hmm_outcome_path="path/to/hmm_outcome.json",
    output_dir="path/to/output"
)
```

### Command Line Usage

```bash
python main_clustering_pipeline.py \
    --hmm-outcome /path/to/hmm_outcome.json \
    --output-dir /path/to/output \
    --target-clusters 20 \
    --min-cluster-size 0.03 \
    --max-cluster-size 0.08 \
    --max-noise 0.05
```

### Custom Configuration

```python
from src.training.steps.market_analysis.regime_clustering.config import create_custom_config

config = create_custom_config(
    target_clusters=18,
    min_cluster_size_pct=0.035,
    max_cluster_size_pct=0.075,
    max_noise_pct=0.04,
    linkage_method='complete'
)
```

## Configuration Templates

### Conservative
- **Target clusters**: 15
- **Size range**: 5-10%
- **Max noise**: 3%
- **Quality focus**: High silhouette score (≥0.4)

### Balanced (Default)
- **Target clusters**: 20
- **Size range**: 3-8%
- **Max noise**: 5%
- **Quality focus**: Good balance

### Aggressive
- **Target clusters**: 25
- **Size range**: 2-6%
- **Max noise**: 8%
- **Quality focus**: More clusters, relaxed constraints

### Research
- **Target clusters**: 30
- **Size range**: 1.5-5%
- **Max noise**: 10%
- **Quality focus**: Maximum granularity

## Output Files

### Core Results
- `regime_clustering_results.json`: Complete pipeline results
- `cluster_mapping.json`: Regime ID → Cluster ID mapping
- `cluster_characteristics.json`: Cluster interpretations
- `cluster_summary.json`: Statistical summary

### Analysis Files
- `cluster_analysis.csv`: Tabular cluster data
- `clustering_results.json`: Raw clustering results
- `validation_results.json`: Quality validation metrics

## Cluster Interpretation

### Market Types
- **Quiet Market**: Low momentum, volatility, volume
- **Active Market**: High momentum, volatility, volume
- **Volatile Market**: High volatility, variable momentum/volume
- **Trending Market**: High momentum, variable volatility/volume
- **High Activity Market**: High volume, variable momentum/volatility
- **Balanced Market**: Medium values across all dimensions

### Trading Implications
- **Conservative strategies**: Quiet markets
- **Aggressive strategies**: Active markets
- **Risk management focus**: Volatile markets
- **Trend-following**: Trending markets
- **Balanced strategies**: Balanced markets

## Quality Metrics

### Internal Coherence
- **Intra-cluster distance**: Average distance within clusters
- **Coherence score**: Inverse of intra-cluster distance
- **Diversity score**: Standard deviation within clusters

### Validity
- **Silhouette score**: Overall clustering quality (-1 to 1)
- **Calinski-Harabasz**: Between/within cluster ratio
- **Davies-Bouldin**: Average similarity ratio

### Distinction
- **Inter-cluster distance**: Average distance between cluster centroids
- **Separation ratio**: Inter-cluster / intra-cluster distance
- **Distinction score**: Normalized separation measure

## Recommendations

The pipeline provides automated recommendations based on validation results:

### Quality Issues
- Low silhouette score → Adjust clustering parameters
- Poor size compliance → Merge/split clusters
- Limited market diversity → Increase cluster count

### Optimization Suggestions
- High coherence, low distinction → Increase cluster count
- Low coherence, high distinction → Decrease cluster count
- Imbalanced sizes → Apply size constraints

## Integration with ML Training

### Cluster Mapping
Each regime is mapped to a cluster ID for ML model training:

```python
# Load cluster mapping
with open('cluster_mapping.json', 'r') as f:
    cluster_mapping = json.load(f)

# Map regimes to clusters
cluster_assignments = [cluster_mapping[regime_id] for regime_id in regime_assignments]
```

### Model Training Per Cluster
Train separate ML models for each cluster:

```python
# Group data by cluster
for cluster_id, cluster_data in data.groupby(cluster_assignments):
    # Train model for this cluster
    model = train_model(cluster_data)
    model.save(f'model_cluster_{cluster_id}.pkl')
```

## Performance Considerations

### Computational Complexity
- **Time complexity**: O(n² log n) for hierarchical clustering
- **Space complexity**: O(n²) for distance matrix
- **Scalability**: Suitable for regimes up to ~1000

### Memory Usage
- **Peak memory**: ~2-4x regime count
- **Optimization**: Uses sparse representations where possible
- **Large datasets**: Consider sampling for very large regime sets

## Troubleshooting

### Common Issues

#### Too Many Small Clusters
- **Cause**: Insufficient size constraints
- **Solution**: Increase `min_cluster_size_pct` or merge small clusters

#### Poor Quality Scores
- **Cause**: Suboptimal clustering parameters
- **Solution**: Try different linkage methods or adjust target cluster count

#### Size Imbalance
- **Cause**: Natural data distribution
- **Solution**: Adjust size constraints or use custom merging logic

### Debug Mode
Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger('RegimeClusterer').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features
- **Visualization**: 3D cluster plots and dendrograms
- **Alternative algorithms**: K-means, DBSCAN, Gaussian Mixture
- **Dynamic clustering**: Online clustering for streaming data
- **Multi-timeframe**: Cross-timeframe regime clustering

### Research Directions
- **Regime persistence**: Temporal clustering with regime transitions
- **Economic validation**: Financial significance of clusters
- **Ensemble clustering**: Multiple algorithm consensus
- **Adaptive parameters**: Self-tuning clustering parameters