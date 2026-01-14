# Composite Clustering Validation Implementation Plan

Implement a comprehensive composite validation approach for ONC clustering using 5 metrics with equal 20% weights each: Davies-Bouldin Index, Calinski-Harabasz Index, Economic Significance (twice), and Gap Statistic.

## Implementation Overview

Replace the current silhouette-only validation in `de_prado_feature_engine.py` with a composite scoring system that combines multiple validation metrics for more robust cluster selection in financial regime detection.

## Key Components

### 1. **Metric Implementations**
- **Davies-Bouldin Index**: Lower values = better clustering (sklearn.metrics.davies_bouldin_score)
- **Calinski-Harabasz Index**: Higher values = better clustering (sklearn.metrics.calinski_harabasz_score)  
- **Gap Statistic**: Statistical significance vs random reference (custom implementation)
- **Economic Significance**: ANOVA F-test on regime returns with effect size (eta-squared)

### 2. **Composite Scoring System**
```python
def composite_cluster_score(X, labels, returns=None):
    scores = {}
    
    # Traditional metrics (normalized)
    scores['davies'] = 1 / (1 + davies_bouldin_score(X, labels))  # Convert to higher=better
    scores['calinski'] = normalize_calinski(calinski_harabasz_score(X, labels))
    scores['gap'] = normalize_gap(calculate_gap_statistic(X, labels))
    
    # Economic metrics (requires returns data)
    if returns is not None:
        econ_score = calculate_economic_significance(returns, labels)
        scores['economic_1'] = econ_score
        scores['economic_2'] = econ_score  # Double weight as requested
    else:
        scores['economic_1'] = 0.5  # Default fallback
        scores['economic_2'] = 0.5
    
    # Equal 20% weights
    weights = {'davies': 0.2, 'calinski': 0.2, 'gap': 0.2, 
               'economic_1': 0.2, 'economic_2': 0.2}
    
    return sum(scores[m] * weights[m] for m in scores)
```

### 3. **Integration Points**

#### **Primary: de_prado_feature_engine.py**
- Replace `_get_onc_clusters()` method's silhouette scoring
- Add composite scoring function
- Update logging to show all metric contributions
- Maintain backward compatibility

#### **Secondary: validation_metrics.py** 
- Add economic significance validator
- Add gap statistic calculator
- Enhance composite scoring capabilities

### 4. **Economic Significance Implementation**
```python
def calculate_economic_significance(returns, regime_labels):
    """Calculate economic significance of regime differences."""
    unique_regimes = np.unique(regime_labels)
    regime_returns = {}
    
    for regime in unique_regimes:
        mask = regime_labels == regime
        regime_returns[regime] = returns[mask].dropna()
    
    # ANOVA F-test for return differences
    if len(regime_returns) >= 2:
        f_stat, p_value = stats.f_oneway(*regime_returns.values())
        
        # Effect size (eta-squared)
        ss_between = sum(len(group) * (np.mean(group) - np.mean(returns)) ** 2 
                        for group in regime_returns.values())
        ss_total = np.sum((returns - np.mean(returns)) ** 2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Normalize to [0,1] where higher = more economically significant
        economic_score = min(1.0, eta_squared * 10)  # Scale effect size
        
        return economic_score
    
    return 0.0  # No economic significance if <2 regimes
```

### 5. **Gap Statistic Implementation**
```python
def calculate_gap_statistic(X, labels, n_refs=10):
    """Calculate gap statistic comparing to random reference."""
    # Actual within-cluster dispersion
    wcss_actual = calculate_wcss(X, labels)
    
    # Reference dispersion from random data
    reference_wcss = []
    for _ in range(n_refs):
        random_data = np.random.uniform(
            low=np.min(X, axis=0), high=np.max(X, axis=0), size=X.shape
        )
        # Apply same clustering structure
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
        random_labels = kmeans.fit_predict(random_data)
        reference_wcss.append(calculate_wcss(random_data, random_labels))
    
    # Gap = log(E[W_ref]) - log(W_actual)
    expected_wcss = np.mean(reference_wcss)
    gap = np.log(expected_wcss) - np.log(wcss_actual) if wcss_actual > 0 else 0
    
    return gap
```

## Implementation Steps

### **Step 1: Create Composite Validation Module**
- Create `composite_clustering_validation.py`
- Implement all 5 metric calculators
- Add normalization functions
- Create composite scoring function

### **Step 2: Update De Prado Engine**
- Modify `_get_onc_clusters()` method
- Replace silhouette loop with composite scoring
- Add detailed metric logging
- Handle missing returns data gracefully

### **Step 3: Add Economic Data Integration**
- Extract returns from feature data when available
- Add fallback when no returns data exists
- Ensure economic metrics work with different data formats

### **Step 4: Testing and Validation**
- Test with synthetic data
- Validate against known cluster structures
- Compare results to current silhouette-only approach
- Ensure performance remains acceptable

## Expected Benefits

1. **More Robust Selection**: Multiple metrics reduce reliance on single flawed metric
2. **Financial Relevance**: Economic significance ensures clusters matter for trading
3. **Statistical Rigor**: Gap statistic provides significance testing
4. **Balanced Evaluation**: Equal weights prevent any single metric from dominating

## Backward Compatibility

- Maintain existing API for `DePradoFeatureEngine`
- Add optional `returns_data` parameter
- Fallback to silhouette-only if composite scoring fails
- Preserve all existing functionality

## Configuration Options

```python
# Optional configuration for customizing weights
config = {
    'metric_weights': {
        'davies': 0.2,
        'calinski': 0.2, 
        'gap': 0.2,
        'economic_1': 0.2,
        'economic_2': 0.2
    },
    'gap_refs': 10,  # Reference datasets for gap statistic
    'economic_scale': 10.0  # Scaling factor for economic significance
}
```

This implementation will provide significantly more reliable and financially meaningful cluster selection for the ONC clustering in the De Prado feature selection pipeline.
