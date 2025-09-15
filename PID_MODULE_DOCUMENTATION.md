# Partial Information Decompositor Module

## Overview

The Partial Information Decompositor (PID) module has been successfully added to the feature selection framework at `/workspace/src/training/utils/feature_selection/partial_information_decompositor.py`. This module provides advanced feature engineering capabilities based on partial information decomposition principles.

## Key Features

### 1. **Partial Information Decomposition**
- **Redundancy Analysis**: Identifies redundant information between feature pairs
- **Synergy Detection**: Finds synergistic relationships between features
- **Unique Information**: Calculates unique information contribution of each feature

### 2. **Feature Engineering Capabilities**
- **Polynomial Features**: Creates polynomial features based on significant interactions
- **Interaction Features**: Generates multiplicative, additive, and ratio-based interactions
- **Cross-Timeframe Features**: Analyzes dependencies across different timeframes

### 3. **Advanced Analysis**
- **Mutual Information**: Uses sklearn's mutual information for accurate calculations
- **Statistical Interactions**: Creates rank-based and normalized interactions
- **Lag-Based Features**: Generates temporal lag interactions

## Usage Examples

### Basic Usage

```python
from src.training.utils.feature_selection import (
    PartialInformationDecompositor, PIDConfig, PIDResult
)

# Initialize with custom configuration
config = PIDConfig(
    synergy_threshold=0.1,
    redundancy_threshold=0.15,
    max_polynomial_degree=3,
    max_interaction_features=50,
    cross_timeframe_threshold=0.15
)

decompositor = PartialInformationDecompositor(config)

# Run PID analysis
pid_result = decompositor.decompose_information(X, y, feature_names)

# Generate expanded feature matrix
expanded_X, expanded_names = decompositor.generate_feature_matrix(X, feature_names, pid_result)
```

### Integration with Main Framework

```python
from src.training.utils.feature_selection import FeatureSelectionFramework

# Configure framework with PID analysis
config = {
    'mode': 'full',
    'partial_information_decompositor': {
        'synergy_threshold': 0.1,
        'redundancy_threshold': 0.15,
        'max_polynomial_degree': 3,
        'max_interaction_features': 50,
        'cross_timeframe_threshold': 0.15
    }
}

framework = FeatureSelectionFramework(config)

# Run comprehensive feature selection with PID analysis
results = framework.run_comprehensive_feature_selection(
    X, y, feature_names,
    target_features=60,
    enable_pid_analysis=True
)

# Access PID results
pid_results = results['pid_results']
print(f"Significant interactions: {pid_results['significant_interactions']}")
print(f"Polynomial features: {len(pid_results['polynomial_features'])}")
print(f"Interaction features: {len(pid_results['interaction_features'])}")
print(f"Cross-timeframe features: {len(pid_results['cross_timeframe_features'])}")
```

## Configuration Options

### PIDConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `synergy_threshold` | 0.1 | Minimum synergy score for significant interactions |
| `redundancy_threshold` | 0.15 | Maximum redundancy threshold |
| `unique_info_threshold` | 0.05 | Minimum unique information threshold |
| `cross_timeframe_threshold` | 0.15 | Threshold for cross-timeframe dependencies |
| `max_timeframe_lag` | 5 | Maximum lag for temporal features |
| `max_polynomial_degree` | 3 | Maximum degree for polynomial features |
| `max_interaction_features` | 50 | Maximum number of interaction features |
| `max_features_for_full_pid` | 20 | Maximum features for full PID analysis |
| `max_interaction_order` | 3 | Maximum interaction order |
| `convergence_threshold` | 1e-6 | Convergence threshold for iterative methods |
| `max_iterations` | 100 | Maximum iterations for convergence |
| `sample_size` | None | Sample size for large datasets |
| `random_state` | 42 | Random state for reproducibility |

## Generated Feature Types

### 1. Polynomial Features
- `feature_name_pow_degree`: Single feature powers (e.g., `price_pow_2`, `volume_pow_3`)
- `feat1_x_feat2_pow_degree`: Cross features with powers (e.g., `price_x_volume_pow_2`)

### 2. Interaction Features
- `feat1_x_feat2`: Multiplicative interactions
- `feat1_plus_feat2`: Additive interactions
- `feat1_minus_feat2`: Subtractive interactions
- `feat1_ratio_feat2`: Ratio-based interactions
- `sqrt_feat1_x_feat2`: Square root interactions
- `log_feat1_x_feat2`: Logarithmic interactions

### 3. Cross-Timeframe Features
- `base_feat_1m_to_5m_ratio`: Ratio between timeframes
- `base_feat_1m_to_5m_diff`: Difference between timeframes
- `base_feat_1m_to_5m_corr`: Correlation between timeframes
- `feat1_lag_1_x_feat2`: Lag-based interactions

## Output Structure

### PIDResult
```python
@dataclass
class PIDResult:
    # Information measures
    redundancy: Dict[Tuple[str, str], float]
    synergy: Dict[Tuple[str, str], float]
    unique_info: Dict[str, float]
    
    # Generated features
    polynomial_features: List[str]
    interaction_features: List[str]
    cross_timeframe_features: List[str]
    
    # Analysis metadata
    feature_pairs_analyzed: int
    significant_interactions: int
    execution_time: float
    convergence_info: Dict[str, Any]
```

## Performance Considerations

### Computational Complexity
- **Pairwise Analysis**: O(n²) where n is the number of features
- **Mutual Information**: Computationally intensive for large feature sets
- **Sampling**: Automatically samples large datasets to maintain performance

### Memory Optimization
- **Chunked Processing**: Processes features in chunks for memory efficiency
- **Sparse Storage**: Uses efficient storage for large correlation matrices
- **Garbage Collection**: Automatic cleanup of intermediate results

### Scalability Features
- **Sample Size Limiting**: Configurable sample size for large datasets
- **Feature Limiting**: Maximum feature count for full PID analysis
- **Early Stopping**: Convergence-based early stopping for iterative methods

## Integration Points

### 1. Main Feature Selection Framework
- Integrated as Step 7 in the comprehensive pipeline
- Works alongside stability, temporal, and causal analysis
- Contributes to final feature selection decisions

### 2. Quality Assessment
- PID results contribute to overall quality scores
- Synergy and redundancy measures inform feature importance
- Interaction analysis validates feature relationships

### 3. Feature Matrix Generation
- Expands original feature matrix with generated features
- Maintains compatibility with existing selection methods
- Preserves feature name mappings for interpretability

## Best Practices

### 1. Configuration Tuning
- Start with default thresholds and adjust based on domain knowledge
- Use higher thresholds for noisy data
- Increase polynomial degree for non-linear relationships

### 2. Performance Optimization
- Set appropriate sample sizes for large datasets
- Limit maximum interaction features to prevent overfitting
- Use mode-specific configurations ('blank', 'light', 'full')

### 3. Feature Selection
- Review generated features for domain relevance
- Filter out redundant or highly correlated generated features
- Validate interaction features with domain experts

## Error Handling

The module includes comprehensive error handling:
- **Graceful Degradation**: Falls back to correlation-based methods if sklearn unavailable
- **Input Validation**: Validates data quality and feature names
- **Exception Handling**: Catches and logs errors without stopping execution
- **Recovery Mechanisms**: Continues processing even if some features fail

## Future Enhancements

Potential future improvements:
1. **Advanced PID Measures**: Implementation of more sophisticated PID measures
2. **GPU Acceleration**: CUDA-based mutual information calculations
3. **Incremental Updates**: Support for streaming data updates
4. **Custom Interaction Types**: User-defined interaction patterns
5. **Visualization Tools**: Interactive plots for PID analysis results

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scikit-learn**: Mutual information calculations
- **scipy**: Statistical functions
- **src.utils**: Internal utility functions

The PID module is now fully integrated and ready for use in creating cross-timeframe, polynomial, and interaction features for your feature selection pipeline!