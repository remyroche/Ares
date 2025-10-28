# HDP-HMM Feature Selection: min_features and max_features Explained

## Overview

The `min_features` and `max_features` parameters control how many features from the **Feature Bank** are selected for HDP-HMM clustering. Understanding these parameters is crucial for optimal regime discovery.

## What is the Feature Bank?

The Feature Bank (`feature_bank_integration.py`) is a comprehensive repository of technical indicators and features organized into categories:

1. **Volume Features** (~30 features)
   - Volume moving averages (SMA, EMA)
   - Volume oscillators and momentum
   - On-Balance Volume (OBV)
   - Accumulation/Distribution
   - Money Flow Index (MFI)
   - Volume clustering patterns

2. **Trend Features** (~25 features)
   - Moving averages (SMA, EMA, WMA)
   - ADX and directional indicators
   - Trend strength scores
   - Support/resistance levels

3. **Volatility Features** (~20 features)
   - Bollinger Bands
   - Average True Range (ATR)
   - Garman-Klass volatility
   - Parkinson volatility
   - Rogers-Satchell volatility
   - Yang-Zhang volatility

4. **Momentum Features** (~20 features)
   - RSI (multiple timeframes)
   - MACD components
   - Stochastic oscillator
   - Williams %R
   - Momentum oscillators

5. **Regime Features** (~30 features)
   - Statistical regime indicators
   - Structural trend features
   - Volume regime patterns
   - Entropy measures
   - Complexity metrics
   - Fractal dimensions

6. **Clustering Features** (~15 features)
   - Distance metrics
   - Separation measures
   - Stability indicators

**Total Available**: ~140+ features

## How Feature Selection Works

### Step 1: Category Weighting

The system assigns weights to feature categories based on their importance for regime discovery:

```python
# From enhanced_hdp_hmm_clustering_integration.py
weights = {
    FeatureBankCategory.VOLATILITY: 0.3,   # High priority - regime changes
    FeatureBankCategory.TREND: 0.25,       # Important for regime dynamics
    FeatureBankCategory.MOMENTUM: 0.2,     # Regime shift indicators
    FeatureBankCategory.VOLUME: 0.15,      # Volume regime patterns
    FeatureBankCategory.CLUSTERING: 0.1    # Auxiliary features
}
```

### Step 2: Feature Scoring

For each feature, the system calculates:
1. **Variance Score**: How much the feature varies (higher = more informative)
2. **Correlation Score**: How unique the feature is (lower correlation with others = better)
3. **Category Weight**: Priority of the feature's category
4. **Temporal Stability**: How consistent the feature is over time

**Final Score** = (0.4 × variance) + (0.3 × uniqueness) + (0.2 × category_weight) + (0.1 × stability)

### Step 3: Feature Selection

Features are ranked by score and selected based on `min_features` and `max_features`:

```python
# Pseudocode
all_features = compute_all_features(data)  # ~140 features
scored_features = score_and_rank(all_features)
selected_features = scored_features[min_features:max_features]
```

## Parameter Guidelines

### min_features

**Purpose**: Ensures sufficient signal for regime discovery

**Effect**:
- **Too Low (< 30)**: May miss important regime characteristics
- **Optimal (40-60)**: Captures key regime signals without noise
- **Too High (> 80)**: May include redundant features

**Default**: 50

**Recommendation**:
- Data-rich markets (Bitcoin, major pairs): 40-50
- Less liquid markets: 30-40
- Complex regimes (multiple cycles): 50-60

### max_features

**Purpose**: Prevents overfitting and reduces computational cost

**Effect**:
- **Too Low (< 60)**: May miss subtle regime patterns
- **Optimal (80-120)**: Comprehensive without redundancy
- **Too High (> 150)**: Risk of overfitting, slow computation

**Default**: 100

**Recommendation**:
- Fast exploration: 60-80
- Standard analysis: 80-100
- Detailed regime discovery: 100-120
- Never exceed 150 (diminishing returns + overfitting risk)

## Trade-offs

### More Features (Higher max_features)

**Advantages**:
- More comprehensive regime characterization
- Captures subtle regime transitions
- Better handles complex market dynamics

**Disadvantages**:
- Longer computation time
- Higher memory usage
- Risk of overfitting (noise as signal)
- Harder to interpret results

### Fewer Features (Lower max_features)

**Advantages**:
- Faster computation
- Lower memory usage
- More robust (less overfitting)
- Easier interpretation

**Disadvantages**:
- May miss subtle regimes
- Less detailed characterization
- Potential information loss

## Practical Examples

### Example 1: Quick Exploration
```python
results = run_hdp_hmm_clustering(
    market_data=df,
    min_features=30,  # Fast, core features only
    max_features=60,  # Limit complexity
    n_iterations=50   # Quick convergence
)
# Use case: Initial exploration, testing
# Time: ~2-5 minutes
```

### Example 2: Standard Analysis
```python
results = run_hdp_hmm_clustering(
    market_data=df,
    min_features=50,   # Good signal
    max_features=100,  # Comprehensive
    n_iterations=100   # Standard convergence
)
# Use case: Production regime discovery
# Time: ~5-10 minutes
```

### Example 3: Detailed Research
```python
results = run_hdp_hmm_clustering(
    market_data=df,
    min_features=60,   # Strong signal
    max_features=120,  # Very comprehensive
    n_iterations=200   # Full convergence
)
# Use case: Academic research, detailed analysis
# Time: ~15-30 minutes
```

### Example 4: Auto-Tuning
```python
# Let the auto-tuner find optimal feature counts
best_params, best_score, tuning_results = run_hdp_hmm_auto_tuning(
    market_data=df,
    symbol="ETHUSDT",
    timeframe="1h",
    # Search space:
    # min_features: 40-60
    # max_features: 80-120
    tpe_trials=50
)
# Use case: Finding optimal configuration
# Time: ~30-60 minutes
```

## Feature Selection Under the Hood

### 1. Feature Generation
```python
# All categories generate features
volume_features = VolumeFeatureGenerator().generate(data)      # 30 features
trend_features = TrendFeatureGenerator().generate(data)        # 25 features
volatility_features = VolatilityFeatureGenerator().generate(data)  # 20 features
momentum_features = MomentumFeatureGenerator().generate(data)  # 20 features
regime_features = RegimeFeatureGenerator().generate(data)      # 30 features
clustering_features = ClusteringFeatureGenerator().generate(data)  # 15 features

total_features = 140  # Approximately
```

### 2. Feature Scoring
```python
scores = {}
for feature in all_features:
    # Calculate variance (normalized)
    variance = feature.std() / (feature.mean() + 1e-8)
    
    # Calculate uniqueness (1 - max_correlation)
    correlations = feature.corr(other_features)
    uniqueness = 1 - correlations.abs().max()
    
    # Get category weight
    category_weight = weights[feature.category]
    
    # Calculate temporal stability
    stability = calculate_temporal_consistency(feature)
    
    # Combined score
    scores[feature] = (
        0.4 * variance + 
        0.3 * uniqueness + 
        0.2 * category_weight + 
        0.1 * stability
    )
```

### 3. Feature Selection
```python
# Sort by score
ranked_features = sorted(features, key=lambda f: scores[f], reverse=True)

# Select based on min/max
if len(ranked_features) < min_features:
    selected = ranked_features  # Use all available
elif len(ranked_features) > max_features:
    selected = ranked_features[:max_features]  # Take top N
else:
    selected = ranked_features[min_features:max_features]  # Use range
```

## When to Adjust Parameters

### Increase min_features when:
- Complex market with many regime types
- Long time series (> 10,000 bars)
- Multiple market cycles present
- High-frequency data (1m, 5m)

### Decrease min_features when:
- Simple market dynamics
- Short time series (< 1,000 bars)
- Low-frequency data (1d, 1w)
- Quick exploration needed

### Increase max_features when:
- Research/analysis (not production)
- Ample computational resources
- Complex regime patterns observed
- Low noise in data

### Decrease max_features when:
- Production trading system
- Limited computational resources
- High data noise
- Need fast execution

## Relationship with Other Parameters

### With alpha (regime diversity)
- High alpha + many features = many detailed regimes
- Low alpha + few features = few broad regimes

### With kappa (regime persistence)
- High kappa + many features = stable, well-defined regimes
- Low kappa + few features = quick, coarse regime switches

### With n_iterations (convergence)
- More features → need more iterations
- Guideline: `n_iterations ≥ max_features`

### With PCA components
- Many features → PCA becomes important
- `pca_components = min(10, max_features // 10)` is reasonable

## Auto-Tuning Behavior

The auto-tuner explores feature counts as part of the search space:

```python
# Default search space
min_features: 40-60   # Ensures good signal
max_features: 80-120  # Balances comprehensiveness and efficiency
```

The tuner will find the optimal balance by maximizing the composite score, which considers:
1. **Cluster quality** (silhouette, DBI, CH)
2. **Temporal stability** (smooth regime transitions)
3. **Balance** (evenly-sized regimes)
4. **Computational efficiency** (implicit through trial speed)

## Best Practices

1. **Start Conservative**: Begin with defaults (50, 100)
2. **Monitor Quality**: Check composite_score and individual metrics
3. **Iterate**: Adjust based on results
4. **Use Auto-Tuning**: Let the system find optimal values
5. **Consider Context**: Adjust for your specific use case
6. **Don't Overfit**: More features ≠ better results

## Common Mistakes

❌ **Setting min_features = max_features**
- Removes flexibility in feature selection
- Better to use a range

❌ **max_features > 150**
- Diminishing returns
- High overfitting risk
- Very slow

❌ **min_features < 20**
- Insufficient signal
- Unstable regime discovery

❌ **Ignoring data size**
- Large max_features on small datasets = overfitting
- Small max_features on large datasets = underfitting

## Summary

- **min_features**: Minimum feature count for adequate signal (default: 50)
- **max_features**: Maximum feature count to prevent overfitting (default: 100)
- Features are selected from ~140 total features in the Feature Bank
- Selection based on variance, uniqueness, category importance, and stability
- Use auto-tuning to find optimal values for your specific data
- Balance comprehensiveness with computational efficiency
- More features ≠ better regimes (quality over quantity)

## Further Reading

- `feature_bank_integration.py`: Full feature bank implementation
- `enhanced_hdp_hmm_clustering_integration.py`: Feature selection logic
- `cluster_quality_assessor.py`: Quality metrics for tuning
- `HDP_HMM_USAGE_GUIDE.md`: Complete usage documentation
