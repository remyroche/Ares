# HMM Cluster Relevance Testing Guide

## Overview

This guide provides comprehensive methods to test the relevance of HMM clusters generated in Step 3 before proceeding to ML model training. Proper cluster validation ensures that your regime discovery is meaningful and will contribute positively to model performance.

## 1. Cluster Quality Metrics (Already Implemented)

Your codebase already includes several cluster quality metrics in `step3_hmm_regime_discovery.py`:

### 1.1 Silhouette Score
- **Range**: -1 to 1 (higher is better)
- **Interpretation**: 
  - > 0.7: Strong cluster structure
  - 0.5-0.7: Reasonable cluster structure
  - 0.25-0.5: Weak cluster structure
  - < 0.25: Poor cluster structure

### 1.2 Calinski-Harabasz Score
- **Range**: Higher is better
- **Interpretation**: Measures ratio of between-cluster dispersion to within-cluster dispersion

### 1.3 Davies-Bouldin Score
- **Range**: Lower is better
- **Interpretation**: Average similarity measure of each cluster with its most similar cluster

### 1.4 Cluster Balance
- **Range**: 0 to 1 (lower is better)
- **Interpretation**: Coefficient of variation of cluster sizes

## 2. Enhanced Cluster Relevance Testing

### 2.1 Predictive Power Assessment

```python
def test_cluster_predictive_power(cluster_data: pd.DataFrame) -> dict:
    """
    Test the predictive power of clusters for future price movements.
    """
    results = {}
    
    # 1. Regime Transition Predictability
    regimes = cluster_data["composite_cluster_id"].values
    transition_counts = {}
    
    for i in range(len(regimes) - 1):
        current = regimes[i]
        next_regime = regimes[i + 1]
        
        if current not in transition_counts:
            transition_counts[current] = {}
        if next_regime not in transition_counts[current]:
            transition_counts[current][next_regime] = 0
        transition_counts[current][next_regime] += 1
    
    # Calculate predictability scores
    predictability_scores = {}
    for regime, transitions in transition_counts.items():
        total_transitions = sum(transitions.values())
        if total_transitions > 0:
            probabilities = [count / total_transitions for count in transitions.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(len(transitions))
            predictability = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            predictability_scores[regime] = predictability
    
    results["transition_predictability"] = predictability_scores
    results["avg_predictability"] = np.mean(list(predictability_scores.values()))
    
    return results
```

### 2.2 Cluster Stability Over Time

```python
def test_cluster_stability(cluster_data: pd.DataFrame, window_size: int = 1000) -> dict:
    """
    Test cluster stability over rolling windows.
    """
    results = {}
    
    # Calculate cluster consistency over rolling windows
    stability_scores = []
    for i in range(0, len(cluster_data) - window_size, window_size // 2):
        window_data = cluster_data.iloc[i:i+window_size]
        window_clusters = window_data["composite_cluster_id"].values
        
        # Calculate cluster distribution consistency
        cluster_counts = np.bincount(window_clusters)
        cluster_proportions = cluster_counts / len(window_clusters)
        
        # Calculate entropy (lower = more stable)
        entropy = -sum(p * np.log2(p) for p in cluster_proportions if p > 0)
        stability_scores.append(1 - (entropy / np.log2(len(cluster_counts))))
    
    results["stability_scores"] = stability_scores
    results["avg_stability"] = np.mean(stability_scores)
    results["stability_std"] = np.std(stability_scores)
    
    return results
```

### 2.3 Market Condition Differentiation

```python
def test_market_condition_differentiation(cluster_data: pd.DataFrame) -> dict:
    """
    Test if clusters effectively differentiate market conditions.
    """
    results = {}
    
    # Calculate average characteristics for each cluster
    cluster_characteristics = {}
    
    for cluster_id in cluster_data["composite_cluster_id"].unique():
        cluster_mask = cluster_data["composite_cluster_id"] == cluster_id
        cluster_subset = cluster_data[cluster_mask]
        
        characteristics = {
            "avg_volatility": cluster_subset["volatility_20"].mean() if "volatility_20" in cluster_subset.columns else 0,
            "avg_momentum": cluster_subset["price_momentum_10"].mean() if "price_momentum_10" in cluster_subset.columns else 0,
            "avg_volume": cluster_subset["volume_ratio_10"].mean() if "volume_ratio_10" in cluster_subset.columns else 1,
            "avg_returns": cluster_subset["returns"].mean() if "returns" in cluster_subset.columns else 0,
            "size": len(cluster_subset)
        }
        cluster_characteristics[cluster_id] = characteristics
    
    # Calculate differentiation scores
    differentiation_scores = {}
    for cluster_id, char in cluster_characteristics.items():
        # Calculate how different this cluster is from others
        differences = []
        for other_id, other_char in cluster_characteristics.items():
            if other_id != cluster_id:
                diff = abs(char["avg_volatility"] - other_char["avg_volatility"]) + \
                       abs(char["avg_momentum"] - other_char["avg_momentum"]) + \
                       abs(char["avg_volume"] - other_char["avg_volume"])
                differences.append(diff)
        
        differentiation_scores[cluster_id] = np.mean(differences) if differences else 0
    
    results["cluster_characteristics"] = cluster_characteristics
    results["differentiation_scores"] = differentiation_scores
    results["avg_differentiation"] = np.mean(list(differentiation_scores.values()))
    
    return results
```

### 2.4 Return Predictability Test

```python
def test_return_predictability(cluster_data: pd.DataFrame, forward_periods: list = [1, 5, 10]) -> dict:
    """
    Test if clusters can predict future returns.
    """
    results = {}
    
    for period in forward_periods:
        # Calculate forward returns
        cluster_data[f"forward_return_{period}"] = cluster_data["close"].pct_change(period).shift(-period)
        
        # Calculate average returns by cluster
        cluster_returns = {}
        for cluster_id in cluster_data["composite_cluster_id"].unique():
            cluster_mask = cluster_data["composite_cluster_id"] == cluster_id
            cluster_subset = cluster_data[cluster_mask]
            
            # Remove NaN values
            valid_returns = cluster_subset[f"forward_return_{period}"].dropna()
            if len(valid_returns) > 0:
                cluster_returns[cluster_id] = {
                    "mean_return": valid_returns.mean(),
                    "std_return": valid_returns.std(),
                    "sharpe_ratio": valid_returns.mean() / valid_returns.std() if valid_returns.std() > 0 else 0,
                    "sample_size": len(valid_returns)
                }
        
        # Calculate return predictability score
        if cluster_returns:
            return_spreads = []
            for cluster_id, returns in cluster_returns.items():
                for other_id, other_returns in cluster_returns.items():
                    if cluster_id != other_id:
                        spread = abs(returns["mean_return"] - other_returns["mean_return"])
                        return_spreads.append(spread)
            
            predictability_score = np.mean(return_spreads) if return_spreads else 0
        else:
            predictability_score = 0
        
        results[f"period_{period}"] = {
            "cluster_returns": cluster_returns,
            "predictability_score": predictability_score
        }
    
    return results
```

## 3. Comprehensive Cluster Validation Function

```python
def comprehensive_cluster_validation(cluster_data: pd.DataFrame, 
                                   quality_thresholds: dict = None) -> dict:
    """
    Comprehensive validation of HMM clusters.
    """
    if quality_thresholds is None:
        quality_thresholds = {
            "min_silhouette": 0.3,
            "min_predictability": 0.4,
            "min_stability": 0.5,
            "min_differentiation": 0.1,
            "min_return_predictability": 0.001
        }
    
    results = {
        "quality_metrics": {},
        "predictive_power": {},
        "stability": {},
        "market_differentiation": {},
        "return_predictability": {},
        "overall_score": 0,
        "recommendations": []
    }
    
    # 1. Quality Metrics (from existing implementation)
    # This would use your existing _calculate_cluster_quality_metrics method
    
    # 2. Predictive Power
    results["predictive_power"] = test_cluster_predictive_power(cluster_data)
    
    # 3. Stability
    results["stability"] = test_cluster_stability(cluster_data)
    
    # 4. Market Differentiation
    results["market_differentiation"] = test_market_condition_differentiation(cluster_data)
    
    # 5. Return Predictability
    results["return_predictability"] = test_return_predictability(cluster_data)
    
    # 6. Calculate Overall Score
    scores = []
    
    # Predictive power score
    if results["predictive_power"]["avg_predictability"] > quality_thresholds["min_predictability"]:
        scores.append(1.0)
    else:
        scores.append(results["predictive_power"]["avg_predictability"] / quality_thresholds["min_predictability"])
    
    # Stability score
    if results["stability"]["avg_stability"] > quality_thresholds["min_stability"]:
        scores.append(1.0)
    else:
        scores.append(results["stability"]["avg_stability"] / quality_thresholds["min_stability"])
    
    # Differentiation score
    if results["market_differentiation"]["avg_differentiation"] > quality_thresholds["min_differentiation"]:
        scores.append(1.0)
    else:
        scores.append(results["market_differentiation"]["avg_differentiation"] / quality_thresholds["min_differentiation"])
    
    results["overall_score"] = np.mean(scores)
    
    # 7. Generate Recommendations
    if results["overall_score"] < 0.6:
        results["recommendations"].append("Consider reducing number of clusters or adjusting HMM parameters")
    if results["predictive_power"]["avg_predictability"] < quality_thresholds["min_predictability"]:
        results["recommendations"].append("Clusters show low predictive power - consider feature engineering improvements")
    if results["stability"]["avg_stability"] < quality_thresholds["min_stability"]:
        results["recommendations"].append("Clusters are unstable over time - consider longer lookback periods")
    
    return results
```

## 4. Integration with Existing Pipeline

### 4.1 Add to Step 3

You can integrate these tests into your existing `step3_hmm_regime_discovery.py`:

```python
# Add to the _analyze_composite_clusters method
def _analyze_composite_clusters(self, features: Any, hmm_states: Any, cluster_labels: Any, cluster_metrics: dict[str, Any]) -> dict[str, Any]:
    """Analyze composite clusters and their characteristics."""
    try:
        # ... existing code ...
        
        # Add comprehensive validation
        cluster_data = pd.DataFrame({
            "composite_cluster_id": cluster_labels,
            "volatility_20": features.get("volatility_20", np.zeros(len(cluster_labels))),
            "price_momentum_10": features.get("price_momentum_10", np.zeros(len(cluster_labels))),
            "volume_ratio_10": features.get("volume_ratio_10", np.ones(len(cluster_labels))),
            "returns": features.get("returns", np.zeros(len(cluster_labels))),
            "close": features.get("close", np.zeros(len(cluster_labels)))
        })
        
        validation_results = comprehensive_cluster_validation(cluster_data)
        analysis["validation_results"] = validation_results
        
        # Log validation results
        self.logger.info(f"📊 Cluster Validation Results:")
        self.logger.info(f"   - Overall Score: {validation_results['overall_score']:.3f}")
        self.logger.info(f"   - Predictive Power: {validation_results['predictive_power']['avg_predictability']:.3f}")
        self.logger.info(f"   - Stability: {validation_results['stability']['avg_stability']:.3f}")
        self.logger.info(f"   - Differentiation: {validation_results['market_differentiation']['avg_differentiation']:.3f}")
        
        if validation_results["recommendations"]:
            self.logger.warning("⚠️ Cluster validation recommendations:")
            for rec in validation_results["recommendations"]:
                self.logger.warning(f"   - {rec}")
        
        return analysis
        
    except Exception as e:
        self.logger.exception(f"❌ Error in composite cluster analysis: {e}")
        return {}
```

### 4.2 Quality Gates

Add quality gates to prevent poor clusters from proceeding:

```python
def _validate_cluster_quality(self, validation_results: dict) -> bool:
    """Validate cluster quality before proceeding."""
    quality_thresholds = {
        "min_overall_score": 0.6,
        "min_predictability": 0.4,
        "min_stability": 0.5
    }
    
    # Check overall score
    if validation_results["overall_score"] < quality_thresholds["min_overall_score"]:
        self.logger.error(f"❌ Cluster quality too low: {validation_results['overall_score']:.3f} < {quality_thresholds['min_overall_score']}")
        return False
    
    # Check individual metrics
    if validation_results["predictive_power"]["avg_predictability"] < quality_thresholds["min_predictability"]:
        self.logger.error(f"❌ Predictive power too low: {validation_results['predictive_power']['avg_predictability']:.3f} < {quality_thresholds['min_predictability']}")
        return False
    
    if validation_results["stability"]["avg_stability"] < quality_thresholds["min_stability"]:
        self.logger.error(f"❌ Stability too low: {validation_results['stability']['avg_stability']:.3f} < {quality_thresholds['min_stability']}")
        return False
    
    self.logger.info("✅ Cluster quality validation passed")
    return True
```

## 5. Usage Examples

### 5.1 Quick Validation Check

```python
# After running step3, validate clusters
from src.training.steps.step3_hmm_regime_discovery import HMMRegimeDiscoveryStep

# Load your cluster data
cluster_data = pd.read_parquet("path/to/cluster_data.parquet")

# Run validation
validation_results = comprehensive_cluster_validation(cluster_data)

print(f"Overall Score: {validation_results['overall_score']:.3f}")
print(f"Recommendations: {validation_results['recommendations']}")
```

### 5.2 Automated Quality Check

```python
# In your pipeline, add quality gates
if not self._validate_cluster_quality(validation_results):
    raise ValueError("Cluster quality below acceptable thresholds")
```

## 6. Threshold Guidelines

### 6.1 Conservative Thresholds (High Quality)
- Overall Score: > 0.7
- Predictive Power: > 0.5
- Stability: > 0.6
- Differentiation: > 0.15

### 6.2 Moderate Thresholds (Balanced)
- Overall Score: > 0.6
- Predictive Power: > 0.4
- Stability: > 0.5
- Differentiation: > 0.1

### 6.3 Relaxed Thresholds (Minimum Viable)
- Overall Score: > 0.5
- Predictive Power: > 0.3
- Stability: > 0.4
- Differentiation: > 0.05

## 7. Troubleshooting Poor Clusters

### 7.1 Low Predictive Power
- Increase feature engineering complexity
- Try different HMM parameters (n_components)
- Use longer lookback periods
- Add regime-specific features

### 7.2 Low Stability
- Increase minimum cluster size
- Use longer rolling windows
- Reduce number of clusters
- Add temporal smoothing

### 7.3 Low Differentiation
- Increase number of clusters
- Use more diverse features
- Try different clustering algorithms
- Add market regime indicators

## 8. Monitoring and Alerting

```python
def monitor_cluster_quality(cluster_data: pd.DataFrame, 
                          alert_threshold: float = 0.5) -> None:
    """Monitor cluster quality and send alerts if below threshold."""
    validation_results = comprehensive_cluster_validation(cluster_data)
    
    if validation_results["overall_score"] < alert_threshold:
        # Send alert
        logger.warning(f"🚨 Cluster quality alert: {validation_results['overall_score']:.3f}")
        logger.warning(f"Recommendations: {validation_results['recommendations']}")
```

This comprehensive approach ensures that your HMM clusters are meaningful and will contribute positively to your ML model training process.