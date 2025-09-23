# Optimal Regime Clustering System

This system creates 20 optimal clusters from HMM regime discovery output to capture 90-95% of the total sample distribution. Each cluster represents 3-8% of the distribution with noise kept below 5%. The clusters are designed for maximum internal coherence, validity, and distinction for ML model training.

## Features

- **Optimal Clustering**: Creates 20 high-quality clusters from HMM regime data
- **Matrix Optimization**: GPU-accelerated operations using unified matrix operations system
- **Noise Reduction**: Hybrid approach using HDBSCAN/DBSCAN + K-means with vectorized operations
- **Quality Validation**: Comprehensive metrics and validation with performance tracking
- **ML Integration**: Ready-to-use datasets for each cluster with memory optimization
- **Flexible Configuration**: Multiple modes for different use cases (standard, high-quality, fast, matrix-optimized)

## Installation

The system is part of the training pipeline and requires:

```bash
pip install scikit-learn numpy pandas hdbscan
```

For matrix optimization features (recommended):

```bash
# Matrix operations are already included in the codebase
# The system automatically detects and uses matrix operations when available
```

## Auto-Detection & Matrix Optimization

The system automatically detects the latest HMM discovery results and uses matrix optimization:

### Auto-Detection Features
- **Latest HMM Results**: Automatically finds the most recent HMM regime discovery output
- **Smart Path Resolution**: Searches multiple common locations for HMM data
- **Same Output Location**: Creates optimal clusters in the same directory as HMM discovery
- **Flexible Input**: Accepts DataFrames or file paths with automatic detection

### Matrix Optimization Features
- **GPU Acceleration**: Apple Silicon M1/M2/M3 optimization
- **Vectorized Operations**: Batch processing with optimized memory usage
- **Enhanced Clustering**: Matrix-optimized HDBSCAN, DBSCAN, and K-means
- **Performance Monitoring**: Detailed timing and memory efficiency tracking
- **Adaptive Processing**: Automatic parameter optimization based on data characteristics

### Automatic Detection
The system automatically detects and uses matrix operations:

```python
from optimal_regime_clustering import OptimalRegimeClusteringOrchestrator

orchestrator = OptimalRegimeClusteringOrchestrator()
# Automatically uses matrix optimization if available
```

### Manual Matrix Optimization
For maximum performance, use the dedicated matrix-optimized function:

```python
from optimal_regime_clustering import run_matrix_optimized_clustering

results = run_matrix_optimized_clustering(
    data_path="hmm_data.parquet",
    output_dir="matrix_optimized_clusters/"
)
```

## Quick Start

### Matrix-Optimized Clustering (Recommended - Default)

```python
from optimal_regime_clustering import run_optimal_clustering

# This automatically uses matrix optimization with GPU acceleration
# AND auto-detects the latest HMM discovery results!
results = run_optimal_clustering(
    # data_path="hmm_cluster_data.parquet",  # Optional - auto-detects latest
    # output_dir="optimal_clusters/",        # Optional - uses HMM location
    symbol="ETHUSDT",      # Default: ETHUSDT
    exchange="binance",    # Default: binance
    timeframe="15m"        # Default: 15m
)

# Features:
# ✅ Matrix optimization with GPU acceleration (Apple Silicon M1/M2/M3)
# ✅ 4D feature space processing (volume, volatility, momentum, trend)
# ✅ 20 optimal clusters with 90-95% coverage
# ✅ <5% noise with advanced filtering
# ✅ Maximum performance with vectorized operations
```

### High-Quality Clustering

```python
from optimal_regime_clustering import run_high_quality_clustering

results = run_high_quality_clustering(
    # data_path="hmm_data.parquet",  # Optional - auto-detects latest
    # output_dir="high_quality_clusters/",  # Optional - uses HMM location
    symbol="BTCUSDT",      # Custom symbol
    exchange="binance",    # Default: binance
    timeframe="15m"        # Default: 15m
)
```

### Fast Processing

```python
from optimal_regime_clustering import run_fast_clustering

results = run_fast_clustering(
    # data_path="hmm_data.parquet",  # Optional - auto-detects latest
    # output_dir="fast_clusters/",  # Optional - uses HMM location
    symbol="ADAUSDT",      # Custom symbol
    exchange="binance",    # Default: binance
    timeframe="15m"        # Default: 15m
)
```

### Matrix-Optimized Clustering

```python
from optimal_regime_clustering import run_matrix_optimized_clustering

results = run_matrix_optimized_clustering(
    # data_path="hmm_data.parquet",  # Optional - auto-detects latest
    # output_dir="matrix_optimized_clusters/",  # Optional - uses HMM location
    symbol="BTCUSDT",      # Custom symbol
    exchange="binance",    # Default: binance
    timeframe="15m"        # Default: 15m
)

# Matrix optimization provides:
# - GPU acceleration (if available)
# - Vectorized batch processing
# - Enhanced memory efficiency
# - Detailed performance metrics
```

## Configuration

### Default Parameters (Updated)

The system now uses these defaults:

```python
# Function defaults (all functions)
symbol: str = "ETHUSDT"        # Default symbol
exchange: str = "binance"      # Default exchange
timeframe: str = "15m"         # Default timeframe (updated from 1h)

# Clustering defaults
target_n_clusters: int = 20    # 20 optimal clusters
target_coverage_pct: float = 0.95  # 90-95% coverage
max_noise_pct: float = 0.05    # <5% noise
min_cluster_size_pct: float = 0.03  # 3% minimum per cluster
max_cluster_size_pct: float = 0.08  # 8% maximum per cluster
```

### Default Configuration

```python
from optimal_regime_clustering import OptimalClusteringConfig

config = OptimalClusteringConfig()
config.target_n_clusters = 20
config.target_coverage_pct = 0.95
config.max_noise_pct = 0.05
config.min_cluster_size_pct = 0.03
config.max_cluster_size_pct = 0.08
```

### High-Quality Configuration

```python
from optimal_regime_clustering import get_clustering_config

config = get_clustering_config("high_quality")
```

### Fast Processing Configuration

```python
config = get_clustering_config("fast_processing")
```

## Output Files

The system generates several output files:

1. **Cluster Labels**: `optimal_cluster_labels.parquet`
2. **Summary Report**: `clustering_summary_report.json`
3. **Detailed Analysis**: `detailed_cluster_analysis.json`
4. **Cluster Characteristics**: `cluster_characteristics.json`
5. **ML Datasets**: `cluster_X_dataset.parquet` (one per cluster)
6. **Combined Dataset**: `all_clusters_dataset.parquet`

## Algorithm Overview

### Multi-Stage Clustering

1. **Noise Reduction**: HDBSCAN/DBSCAN identifies and removes noise points
2. **Main Clustering**: K-means creates initial clusters
3. **Optimization**: Combines and optimizes results for quality
4. **Validation**: Comprehensive quality assessment

### Quality Metrics

- **Silhouette Score**: Cluster separation and cohesion
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Score**: Average similarity of each cluster with its most similar cluster
- **Coverage**: Percentage of data captured in clusters
- **Noise Level**: Percentage of data identified as noise
- **Cluster Balance**: Distribution of cluster sizes

### Validation Criteria

- Coverage ≥ 90-95%
- Noise ≤ 5%
- Cluster sizes between 3-8% of total
- Silhouette score ≥ 0.3
- Calinski-Harabasz score ≥ 100
- Davies-Bouldin score ≤ 1.5

## Integration with HMM Discovery

The system integrates seamlessly with HMM regime discovery output:

```python
# Load HMM regime data
from optimal_regime_clustering import load_hmm_regime_data

data = load_hmm_regime_data("hmm_composite_clusters.parquet", config)

# Run clustering
results = run_optimal_clustering(
    data_path=data,
    output_dir="hmm_optimized_clusters/"
)
```

## ML Training Workflow

### 1. Prepare Data

```python
results = run_optimal_clustering(...)
cluster_datasets = results['ml_datasets']
```

### 2. Train Models per Cluster

```python
for cluster_name, dataset_path in cluster_datasets.items():
    if cluster_name.startswith('cluster_'):
        # Load cluster data
        cluster_data = pd.read_parquet(dataset_path)

        # Train model on cluster data
        model = train_model(cluster_data)
        save_model(model, f"model_{cluster_name}.pkl")
```

### 3. Ensemble Models

```python
# Use cluster characteristics for ensemble weighting
cluster_info = results['reports']['cluster_characteristics']
ensemble_models(cluster_info)
```

## Examples

See `example_usage.py` for comprehensive examples:

```bash
cd /workspace/src/training/steps/market_analysis/optimal_regime_clustering
python example_usage.py
```

## Troubleshooting

### Common Issues

1. **No Data Found**
   - Ensure HMM regime data exists
   - Check file paths and formats
   - Use sample data for testing

2. **Poor Cluster Quality**
   - Adjust configuration parameters
   - Try high-quality mode
   - Review feature preprocessing

3. **Memory Issues**
   - Use fast processing mode
   - Reduce chunk size
   - Process data in batches

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance

### Benchmarks

- **Matrix-Optimized Mode (Default)**: ~20-45 seconds for 10k samples (with GPU acceleration)
- **High-Quality Mode**: ~45-60 seconds for 10k samples (stricter validation)
- **Fast Mode**: ~15-30 seconds for 10k samples (relaxed validation)
- **Standard Mode** (fallback): ~1-2 minutes for 10k samples (no matrix optimization)

### Memory Usage

- **10k samples**: ~100-200 MB
- **50k samples**: ~500-800 MB
- **100k samples**: ~1-2 GB

### Scalability

The system is designed to handle:
- Up to 1M samples with standard memory
- Multi-core processing for large datasets
- Chunked processing for memory efficiency

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| target_n_clusters | 20 | Number of target clusters |
| target_coverage_pct | 0.95 | Target data coverage |
| max_noise_pct | 0.05 | Maximum noise percentage |
| min_cluster_size_pct | 0.03 | Minimum cluster size |
| max_cluster_size_pct | 0.08 | Maximum cluster size |
| min_silhouette_score | 0.3 | Quality threshold |
| clustering_method | "hybrid" | Clustering algorithm |

## API Reference

### Main Functions

- `run_optimal_clustering()`: **Matrix-optimized clustering (recommended - default)**
- `run_high_quality_clustering()`: Enhanced quality clustering with matrix optimization
- `run_fast_clustering()`: Quick processing clustering with matrix optimization
- `run_matrix_optimized_clustering()`: **Alias for run_optimal_clustering() (explicit emphasis)**
- `OptimalClusteringConfig()`: Configuration class
- `OptimalRegimeClusteringOrchestrator()`: Full pipeline orchestration

### Matrix-Optimized Functions

- `MatrixOptimizedClusterer()`: Matrix-optimized clustering algorithm
- `cluster_regimes_optimized()`: Optimized clustering with performance tracking
- `create_matrix_optimized_clusterer()`: Factory for matrix-optimized clusterer

### Utility Functions

- `load_regime_data()`: Load regime data
- `calculate_cluster_statistics()`: Compute cluster statistics
- `validate_cluster_quality()`: Validate clustering results
- `create_cluster_summary_report()`: Generate reports

## Contributing

To extend the system:

1. Add new clustering algorithms in `clustering.py`
2. Extend configuration options in `config.py`
3. Add validation metrics in `utils.py`
4. Update documentation and examples

## License

This system is part of the trading pipeline and follows the same license terms.

## Support

For questions and issues:
1. Check the examples in `example_usage.py`
2. Review the configuration options
3. Enable debug logging for troubleshooting
4. Check the generated reports for validation details