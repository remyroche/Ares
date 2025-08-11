# Autoencoder Feature Importance Analysis

## Overview

The Autoencoder Feature Importance Analysis system provides comprehensive analysis of autoencoder-generated features to understand their predictive power, stability, and interpretability. This system helps traders and researchers make informed decisions about which autoencoder features to use in their trading strategies.

## Features

### 🔍 **Multi-Method Feature Importance**
- **Random Forest Importance**: Traditional tree-based feature importance
- **Gradient Boosting Importance**: More robust ensemble-based importance
- **Permutation Importance**: Model-agnostic importance that measures prediction accuracy drop
- **Ensemble Importance**: Combined importance from multiple methods

### 📊 **Statistical Correlation Analysis**
- **Pearson Correlations**: Linear correlation with target variables
- **Mutual Information**: Non-linear dependency measures
- **High/Low Correlation Detection**: Automatic identification of relevant features

### 📈 **Feature Stability Analysis**
- **Rolling Statistics**: Time-based stability assessment
- **Trend Stability**: Measures how features change over time
- **Coefficient of Variation**: Statistical stability metrics
- **Stable/Unstable Feature Classification**

### 🔄 **Regime-Specific Analysis**
- **Cross-Regime Consistency**: Identifies features that work across market regimes
- **Regime-Specific Importance**: Importance scores per market regime
- **Consistency Scoring**: Measures feature reliability across regimes

### 🔄 **Comparison with Original Features**
- **Performance Comparison**: How autoencoder features compare to original features
- **Feature Overlap Analysis**: Identifies which original features are captured
- **Value Addition Assessment**: Quantifies the benefit of autoencoder features

## Usage

### Basic Usage

```python
from src.analyst.autoencoder_feature_generator import AutoencoderFeatureGenerator

# Initialize generator
generator = AutoencoderFeatureGenerator()

# Generate features with analysis
enhanced_features = generator.generate_features(
    features_df=your_features,
    regime_name="bull_market",
    labels=your_labels,
    regime_labels=your_regime_labels,  # Optional
    enable_analysis=True
)

# Access analysis results
analysis_results = generator.get_last_analysis_results()
```

### Accessing Analysis Results

```python
# Get feature rankings
ensemble_ranking = generator.get_feature_ranking(method='ensemble')
print("Top features:", ensemble_ranking.head(10))

# Get stable features
stable_features = generator.get_stable_features(threshold=0.7)
print("Stable features:", stable_features)

# Get high correlation features
high_corr_features = generator.get_high_correlation_features(threshold=0.5)
print("High correlation features:", high_corr_features)

# Get recommendations
recommendations = generator.get_recommendations()
for rec in recommendations:
    print(f"💡 {rec}")
```

### Configuration

The analysis can be configured through the YAML configuration file:

```yaml
feature_analysis:
  enable_analysis: true
  high_correlation_threshold: 0.7
  low_correlation_threshold: 0.1
  stability_window: 100
  stability_threshold: 0.7
  regime_analysis_enabled: true
  comparison_with_original: true
```

## Analysis Components

### 1. Feature Importance Analysis

**Methods Used:**
- **Random Forest**: Uses Gini importance from Random Forest classifier
- **Gradient Boosting**: Uses feature importance from Gradient Boosting
- **Permutation Importance**: Measures accuracy drop when feature is permuted
- **Ensemble**: Average of normalized importance scores

**Output:**
- Ranked list of features by importance
- Importance scores and confidence intervals
- Top/bottom feature identification

### 2. Correlation Analysis

**Metrics:**
- **Pearson Correlation**: Linear correlation with target
- **Mutual Information**: Non-linear dependency
- **Absolute Correlation**: Magnitude of correlation

**Output:**
- Correlation coefficients for each feature
- High/low correlation feature lists
- Correlation distribution statistics

### 3. Stability Analysis

**Metrics:**
- **Mean Stability**: Based on rolling standard deviation
- **Trend Stability**: Based on rolling mean changes
- **Coefficient of Variation**: Overall variability measure
- **Overall Stability**: Combined stability score

**Output:**
- Stability scores for each feature
- Stable/unstable feature classification
- Stability distribution statistics

### 4. Regime Analysis

**Analysis:**
- **Per-Regime Importance**: Feature importance in each market regime
- **Cross-Regime Consistency**: How importance varies across regimes
- **Consistency Scoring**: Reliability measure across regimes

**Output:**
- Regime-specific importance rankings
- Consistency scores for features
- Regime-consistent feature lists

## Interpretation Guide

### Feature Importance Scores

| Score Range | Interpretation | Recommendation |
|-------------|----------------|----------------|
| 0.8 - 1.0 | Very High | Definitely include |
| 0.6 - 0.8 | High | Strongly consider |
| 0.4 - 0.6 | Medium | Consider if stable |
| 0.2 - 0.4 | Low | Consider only if stable |
| 0.0 - 0.2 | Very Low | Exclude |

### Stability Scores

| Score Range | Interpretation | Recommendation |
|-------------|----------------|----------------|
| 0.8 - 1.0 | Very Stable | Excellent for production |
| 0.6 - 0.8 | Stable | Good for production |
| 0.4 - 0.6 | Moderate | Use with caution |
| 0.2 - 0.4 | Unstable | Monitor closely |
| 0.0 - 0.2 | Very Unstable | Avoid in production |

### Correlation Thresholds

| Threshold | Interpretation | Use Case |
|-----------|----------------|----------|
| > 0.7 | Very High | Strong predictive signal |
| 0.5 - 0.7 | High | Good predictive signal |
| 0.3 - 0.5 | Moderate | Moderate predictive signal |
| 0.1 - 0.3 | Low | Weak predictive signal |
| < 0.1 | Very Low | Minimal predictive signal |

## Best Practices

### 1. **Feature Selection Strategy**

```python
# Get top features by importance
top_features = generator.get_feature_ranking().head(10)['feature'].tolist()

# Get stable features
stable_features = generator.get_stable_features(threshold=0.7)

# Select features that are both important and stable
recommended_features = list(set(top_features) & set(stable_features))
```

### 2. **Regime-Aware Selection**

```python
# Get regime-consistent features
analysis_results = generator.get_last_analysis_results()
if 'regime_analysis' in analysis_results:
    consistent_features = analysis_results['regime_analysis']['consistent_features']
    print("Regime-consistent features:", consistent_features)
```

### 3. **Monitoring and Maintenance**

```python
# Regular analysis for feature drift
def monitor_feature_drift(generator, new_data, labels):
    """Monitor feature performance over time."""
    analysis_results = generator.generate_features(
        new_data, "monitoring", labels, enable_analysis=True
    )
    
    # Check for stability changes
    stable_features = generator.get_stable_features()
    recommendations = generator.get_recommendations()
    
    return stable_features, recommendations
```

## Troubleshooting

### Common Issues

1. **Low Feature Importance**
   - **Cause**: Autoencoder not capturing relevant patterns
   - **Solution**: Retrain with different parameters or more data

2. **Unstable Features**
   - **Cause**: Market regime changes or insufficient data
   - **Solution**: Use longer training windows or regime-specific models

3. **High Correlation with Original Features**
   - **Cause**: Autoencoder not learning new representations
   - **Solution**: Increase encoding dimension or change architecture

4. **Analysis Timeout**
   - **Cause**: Large dataset or complex analysis
   - **Solution**: Reduce sample size or disable regime analysis

### Performance Optimization

```python
# For large datasets, use sampling
custom_config = {
    "feature_analysis": {
        "enable_analysis": True,
        "stability_window": 50,  # Smaller window
        "regime_analysis_enabled": False,  # Disable if slow
    }
}
```

## Example Output

```
🔍 Starting comprehensive autoencoder feature importance analysis...
📊 Encoded features shape: (1000, 33)
🎯 Labels shape: (1000,)
📈 Unique labels: 2

📊 Correlation analysis complete:
   📈 Mean correlation: 0.2345
   📈 Max correlation: 0.6789
   📈 High correlation features: 5
   📈 Low correlation features: 12

🤖 ML importance analysis complete:
   🏆 Top 5 features: ['autoencoder_1', 'autoencoder_3', 'autoencoder_7', 'autoencoder_12', 'autoencoder_15']
   📊 Mean importance: 0.4567

📈 Stability analysis complete:
   📊 Mean stability: 0.7234
   📊 Stable features: 18
   📊 Unstable features: 3

💡 Recommendations:
   🎉 High feature importance detected. Autoencoder is generating valuable features.
   ✅ Good feature stability detected. These features should perform well in production.
   💡 High correlation features detected. Consider feature selection to reduce redundancy.
```

## Integration with Trading Strategy

### Feature Selection for Live Trading

```python
def select_production_features(generator):
    """Select features for live trading based on analysis."""
    
    # Get analysis results
    analysis_results = generator.get_last_analysis_results()
    
    if not analysis_results:
        return []
    
    # Select features based on multiple criteria
    top_features = generator.get_feature_ranking().head(15)['feature'].tolist()
    stable_features = generator.get_stable_features(threshold=0.7)
    high_corr_features = generator.get_high_correlation_features(threshold=0.5)
    
    # Intersection of criteria
    production_features = list(set(top_features) & set(stable_features))
    
    # Add high correlation features if not too many
    if len(production_features) < 10:
        additional_features = [f for f in high_corr_features if f not in production_features]
        production_features.extend(additional_features[:5])
    
    return production_features[:20]  # Limit to top 20 features
```

### Regime-Specific Feature Selection

```python
def get_regime_features(generator, current_regime):
    """Get features optimized for current market regime."""
    
    analysis_results = generator.get_last_analysis_results()
    
    if 'regime_analysis' not in analysis_results:
        return generator.get_feature_ranking().head(10)['feature'].tolist()
    
    regime_importance = analysis_results['regime_analysis']['regime_importance']
    
    if current_regime in regime_importance:
        regime_features = regime_importance[current_regime]
        if 'ensemble' in regime_features:
            return [item['feature'] for item in regime_features['ensemble'][:10]]
    
    # Fallback to general features
    return generator.get_feature_ranking().head(10)['feature'].tolist()
```

## Conclusion

The Autoencoder Feature Importance Analysis system provides comprehensive insights into the quality and usefulness of autoencoder-generated features. By combining multiple analysis methods, it helps traders make informed decisions about feature selection and provides actionable recommendations for improving model performance.

The system is designed to be:
- **Comprehensive**: Multiple analysis methods for thorough evaluation
- **Configurable**: Flexible settings for different use cases
- **Actionable**: Clear recommendations and thresholds
- **Production-Ready**: Stability and regime analysis for live trading

Use this system to validate your autoencoder features and ensure they provide genuine value to your trading strategy. 