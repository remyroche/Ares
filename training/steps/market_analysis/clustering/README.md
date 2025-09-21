# Regime Clustering Pipeline

This module provides a complete pipeline for consolidating HMM discovery regimes into balanced, coherent clusters suitable for ML model training.

## Overview

The regime clustering pipeline takes the output from HMM discovery (which creates many small, fragmented regimes) and consolidates them into 20-ish balanced clusters that capture 90-95% of the market state distribution. Each cluster is designed to have 3-8% of the total samples, ensuring balanced training data for ML models.

## Key Features

- **Complete Coverage**: Ensures 100% of market regimes are accounted for
- **Similarity-Based Merging**: Preserves market information by merging similar regimes
- **Balanced Clusters**: Creates clusters with 3-8% size distribution
- **ML-Ready Outputs**: Generates training datasets, cluster profiles, and feature mappings
- **Quality Validation**: Comprehensive validation and quality metrics
- **Integration**: Seamless integration with existing HMM discovery pipeline

## Architecture

```
HMM Discovery Results → Regime Consolidation → ML Training Outputs
```

### Components

1. **HMMDiscoveryIntegration**: Loads and validates HMM discovery outputs
2. **RegimeConsolidator**: Consolidates regimes into balanced clusters
3. **MLOutputGenerator**: Creates ML-ready training datasets and profiles
4. **ClusteringPipeline**: Orchestrates the complete pipeline

## Quick Start

### Using the Command Line

```bash
# Run with specific HMM results file
python run_clustering_pipeline.py \
  --hmm-results /path/to/hmm_results.json \
  --symbol BTCUSDT \
  --timeframe 1h

# Run from outcomes directory (finds latest results)
python run_clustering_pipeline.py \
  --outcomes-dir /path/to/outcomes \
  --symbol ETHUSDT \
  --timeframe 15m

# Custom clustering parameters
python run_clustering_pipeline.py \
  --hmm-results /path/to/hmm_results.json \
  --symbol BTCUSDT \
  --timeframe 1h \
  --target-clusters 25 \
  --min-cluster-size 0.02 \
  --max-cluster-size 0.10 \
  --coverage-target 0.98
```

### Using as a Python Module

```python
from training.steps.market_analysis.clustering import create_clustering_pipeline

# Create pipeline with custom configuration
pipeline = create_clustering_pipeline(
    target_clusters=20,
    min_cluster_size_pct=0.03,
    max_cluster_size_pct=0.08,
    coverage_target=0.95
)

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    hmm_results_file="path/to/hmm_results.json",
    symbol="BTCUSDT",
    timeframe="1h"
)

# Access results
consolidation_result = results['consolidation_result']
training_dataset = results['training_dataset']
cluster_profiles = results['cluster_profiles']
```

## Configuration Options

### ConsolidationConfig Parameters

- **target_clusters** (int): Number of target clusters (default: 20)
- **min_cluster_size_pct** (float): Minimum cluster size as percentage (default: 0.03 = 3%)
- **max_cluster_size_pct** (float): Maximum cluster size as percentage (default: 0.08 = 8%)
- **coverage_target** (float): Target coverage by top clusters (default: 0.95 = 95%)
- **merge_similarity_threshold** (float): Threshold for merging similar regimes (default: 0.90)
- **assignment_similarity_threshold** (float): Threshold for assigning remaining regimes (default: 0.70)

### Example Configurations

```python
# Conservative clustering (more clusters, smaller size range)
config = ConsolidationConfig(
    target_clusters=25,
    min_cluster_size_pct=0.02,  # 2%
    max_cluster_size_pct=0.06,  # 6%
    coverage_target=0.98        # 98%
)

# Aggressive clustering (fewer clusters, larger size range)
config = ConsolidationConfig(
    target_clusters=15,
    min_cluster_size_pct=0.04,  # 4%
    max_cluster_size_pct=0.10,  # 10%
    coverage_target=0.90        # 90%
)
```

## Output Files

The pipeline generates several output files:

### ML Training Files
- **training_dataset_{symbol}_{timeframe}_{timestamp}.csv**: Feature matrix for ML training
- **cluster_labels_{symbol}_{timeframe}_{timestamp}.npy**: Cluster labels for each sample
- **cluster_metadata_{symbol}_{timeframe}_{timestamp}.json**: Detailed cluster metadata

### Analysis Files
- **cluster_profiles_{symbol}_{timeframe}_{timestamp}.json**: Human-readable cluster profiles
- **ml_outputs_summary_{symbol}_{timeframe}_{timestamp}.json**: Comprehensive summary
- **pipeline_results_{symbol}_{timeframe}_{timestamp}.json**: Complete pipeline results

### Clustering Results
- **regime_consolidation_results_{timestamp}.json**: Detailed consolidation analysis
- **cluster_summary_{timestamp}.csv**: Cluster statistics summary

## Understanding the Results

### Cluster Profiles

Each cluster profile contains:
- **Market Regime**: Interpreted market state (e.g., "High_Volatility_Bull")
- **Sample Count**: Number of samples in the cluster
- **Sample Percentage**: Percentage of total market samples
- **Centroid**: Average feature values for the cluster
- **Feature Ranges**: Min/max values for each feature
- **Trainability**: Whether the cluster has enough samples for training

### Market Regime Interpretation

Clusters are automatically interpreted as market regimes based on their feature centroids:

- **High_Volatility_Bull**: High volatility + positive momentum
- **Low_Volatility_Sideways**: Low volatility + neutral momentum
- **Moderate_Volatility_Bear**: Moderate volatility + negative momentum
- etc.

### Quality Metrics

The pipeline provides several quality metrics:

- **Coverage Percentage**: Percentage of total samples covered by all clusters
- **Top Clusters Coverage**: Percentage covered by top N clusters
- **Balance Score**: How well cluster sizes are balanced
- **Trainability**: Number of clusters suitable for ML training

## Integration with ML Training

### Loading Training Data

```python
import pandas as pd
import numpy as np

# Load training dataset
features = pd.read_csv('training_dataset_BTCUSDT_1h_20241201_120000.csv')
labels = np.load('cluster_labels_BTCUSDT_1h_20241201_120000.npy')

# Load cluster metadata
with open('cluster_metadata_BTCUSDT_1h_20241201_120000.json', 'r') as f:
    cluster_metadata = json.load(f)
```

### Training ML Models

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare data
X = features.values
y = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy:.3f}")
```

## Troubleshooting

### Common Issues

1. **Low Coverage**: If top clusters don't reach coverage target
   - Increase `coverage_target` or adjust cluster size constraints
   - Check if HMM discovery created too many fragmented regimes

2. **Unbalanced Clusters**: If balance score is low
   - Adjust `min_cluster_size_pct` and `max_cluster_size_pct`
   - Consider different similarity thresholds

3. **Too Many Small Clusters**: If many clusters have few samples
   - Increase `merge_similarity_threshold` to merge more regimes
   - Decrease `target_clusters` to focus on larger clusters

4. **Validation Warnings**: If data validation fails
   - Check HMM discovery results quality
   - Ensure sufficient data points and regime diversity

### Performance Considerations

- **Memory Usage**: Large datasets may require chunked processing
- **Processing Time**: Similarity calculations scale quadratically with regime count
- **Output Size**: Detailed results can be large for many regimes

## Advanced Usage

### Custom Similarity Metrics

```python
from sklearn.metrics.pairwise import euclidean_distances

# Use Euclidean distance instead of cosine similarity
def custom_similarity_metric(features1, features2):
    distances = euclidean_distances(features1, features2)
    return 1.0 / (1.0 + distances)  # Convert distance to similarity
```

### Custom Market Regime Interpretation

```python
def custom_regime_interpreter(cluster):
    # Custom logic for interpreting cluster characteristics
    centroid = cluster['centroid']
    
    if centroid[1] > 0.8:  # High volatility
        return "Crisis_Mode"
    elif centroid[0] > 0.5:  # Strong momentum
        return "Trending_Up"
    else:
        return "Consolidation"
```

## Contributing

When extending the clustering pipeline:

1. Follow the existing architecture patterns
2. Add comprehensive logging and validation
3. Include unit tests for new functionality
4. Update documentation and examples
5. Ensure backward compatibility

## License

This module is part of the market analysis system and follows the same licensing terms.