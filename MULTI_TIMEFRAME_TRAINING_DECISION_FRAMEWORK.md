# Multi-Timeframe Training Decision Framework

## Quick Decision Tree

```
Do you have comprehensive cross-timeframe features?
├── YES (≥30% cross-timeframe features, high diversity)
│   ├── Is your dataset large (>100k samples)?
│   │   ├── YES → Consider Regime-Aware Single-Timeframe
│   │   └── NO → Use Single-Timeframe with Cross-Timeframe Features
│   └── Do you need predictions at multiple time horizons?
│       ├── YES → Use Hierarchical Multi-Timeframe
│       └── NO → Use Single-Timeframe with Cross-Timeframe Features
└── NO (<30% cross-timeframe features, low diversity)
    └── Use Multi-Timeframe Parallel Training
```

## Detailed Analysis

### 🎯 **Recommendation: Single-Timeframe with Cross-Timeframe Features**

**For most cases, this is the optimal approach because:**

1. **Cross-timeframe features already capture multi-timeframe information**
2. **Simpler architecture = better maintainability**
3. **Reduced overfitting risk**
4. **Better performance in most scenarios**

### 📊 **When to Use Each Approach**

#### ✅ **Single-Timeframe with Cross-Timeframe Features (Recommended)**

**Use when:**
- Cross-timeframe features represent ≥30% of your feature set
- Feature diversity is high (>0.5)
- You have good cross-timeframe feature engineering
- You want simplicity and maintainability

**Benefits:**
- Simpler architecture
- Better interpretability
- Reduced overfitting
- Faster training and inference
- Easier deployment

**Example:**
```python
# Train one model with all cross-timeframe features
model = train_model(
    features=cross_timeframe_features,  # Includes all timeframe relationships
    target=base_timeframe_returns,      # e.g., 1m returns
    timeframe='1m'
)
```

#### ⚖️ **Multi-Timeframe Parallel Training**

**Use when:**
- Cross-timeframe features are limited (<30%)
- Different timeframes have fundamentally different patterns
- You need specialized models for each timeframe
- You have sufficient compute resources

**Benefits:**
- Specialized models for each timeframe
- Better uncertainty estimation
- Robustness through diversity

**Drawbacks:**
- Increased complexity
- Higher computational cost
- Model coordination challenges

#### 🎭 **Regime-Aware Single-Timeframe**

**Use when:**
- You have large datasets (>100k samples)
- Market regimes are distinct and identifiable
- You want adaptive behavior
- Cross-timeframe features + regime awareness

**Benefits:**
- Adaptive to market conditions
- Specialized models for different regimes
- Best of both worlds

**Drawbacks:**
- Requires regime detection
- Increased complexity
- Needs sufficient data for each regime

### 🔬 **Experimental Validation**

To determine the best approach for your specific case:

```python
from src.feature_engineering.multi_timeframe_training_analysis import analyze_training_approach

# Analyze your specific case
analysis = analyze_training_approach(
    cross_timeframe_features=your_features,
    target_returns=your_targets,
    data_size=len(your_data),
    available_compute="medium"  # or "low", "high"
)

print(f"Recommended approach: {analysis['recommendation']['recommended_approach']}")
print(f"Confidence: {analysis['recommendation']['confidence']}")
print(f"Reasoning: {analysis['recommendation']['reasoning']}")
```

### 📈 **Performance Comparison**

| Approach | Complexity | Performance | Maintainability | Compute Cost |
|----------|------------|-------------|-----------------|--------------|
| Single-Timeframe | Low | High | High | Low |
| Multi-Timeframe | High | Medium-High | Medium | High |
| Regime-Aware | Medium | High | Medium | Medium |

### 🛠️ **Implementation Recommendations**

#### **For High-Frequency Trading (1m-5m timeframes):**
```python
# Recommended: Single-timeframe with cross-timeframe features
config = {
    'base_timeframe': '1m',
    'cross_timeframe_features': True,
    'feature_selection': 'mutual_info',
    'model_type': 'RandomForestRegressor',
    'validation': 'time_series_split'
}
```

#### **For Medium-Frequency Trading (5m-15m timeframes):**
```python
# Consider regime-aware if you have enough data
config = {
    'base_timeframe': '5m',
    'regime_awareness': True,
    'cross_timeframe_features': True,
    'regime_detection': 'hmm_clustering'
}
```

#### **For Multi-Horizon Predictions:**
```python
# Use hierarchical approach
config = {
    'timeframes': ['1m', '5m', '15m'],
    'hierarchical': True,
    'ensemble_method': 'weighted_average'
}
```

### 🎯 **Key Takeaways**

1. **Cross-timeframe features often make multi-timeframe training redundant**
2. **Single-timeframe training is usually better** when you have good cross-timeframe features
3. **Consider regime-aware training** for large datasets with distinct market regimes
4. **Multi-timeframe training is only beneficial** when cross-timeframe features are limited
5. **Always validate experimentally** with your specific data and use case

### 🔍 **Validation Checklist**

Before deciding, answer these questions:

- [ ] Do cross-timeframe features represent ≥30% of your feature set?
- [ ] Is feature diversity high (>0.5)?
- [ ] Do you need predictions at multiple time horizons?
- [ ] Is your dataset large enough for regime detection?
- [ ] Do different timeframes have fundamentally different patterns?
- [ ] Do you have sufficient compute resources for multiple models?
- [ ] Is model interpretability important?
- [ ] Do you need fast inference times?

**If you answered "YES" to questions 1-2 and "NO" to most others → Use Single-Timeframe with Cross-Timeframe Features**

**If you answered "NO" to questions 1-2 → Use Multi-Timeframe Parallel Training**

**If you answered "YES" to questions 4-5 → Consider Regime-Aware Single-Timeframe**