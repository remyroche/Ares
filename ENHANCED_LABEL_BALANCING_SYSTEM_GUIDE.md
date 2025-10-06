# Enhanced Label Balancing & Sample Weighting System

## 🎯 Overview

This comprehensive system addresses the fundamental problem in financial machine learning: **extreme class imbalance**. In financial datasets, 80-95% of samples are "do nothing" (Analyst = 0), making it trivial for models to achieve high accuracy by always predicting no-trade.

The system implements multiple techniques to "teach the model what matters" by ensuring balanced exposure to different classes and weighting samples by information content rather than just class balance.

## 🚀 Key Features

### 1. Advanced Balancing Techniques
- **Under-sampling**: Reduces majority class samples to balance dataset
- **Over-sampling**: Uses SMOTE/ADASYN to synthesize minority class samples
- **Mixup Augmentation**: Creates synthetic samples by mixing existing ones
- **Stratified Batching**: Ensures each mini-batch sees balanced classes
- **Adaptive Balancing**: Automatically selects the best technique based on dataset characteristics

### 2. Comprehensive Sample Weighting Schemes
- **Volatility Weighting**: `w_t ∝ 1/σ_t` (de-emphasize noisy periods)
- **Confidence Weighting**: `w_t ∝ Δp` (weight by label confidence)
- **Event Overlap Weighting**: López de Prado method for overlapping events
- **Time Decay Weighting**: Exponential decay for recency adaptation
- **Regime-Aware Weighting**: Inverse frequency weighting for rare regimes
- **Information Content Weighting**: Combined weighting based on multiple factors

### 3. Regime-Aware Rebalancing
- **Inverse Frequency**: Weight samples inversely to regime frequency
- **Stratified Balancing**: Ensure equal representation per regime
- **Regime Validation Fairness**: Check regime mix in validation sets

### 4. Validation Fairness
- **Class Ratio Fairness**: Ensure validation has similar class distribution
- **Regime Mix Fairness**: Ensure validation represents all regimes
- **Temporal Drift Detection**: Monitor distribution shifts over time

## 📊 Quick Start

### Basic Usage

```python
from src.training.steps.pre_training.profit_labeling.enhanced_balancing_integration import (
    create_trading_balancing_manager
)

# Create balancing manager
manager = create_trading_balancing_manager()

# Prepare your data
X = your_features  # pd.DataFrame
y = your_labels    # pd.Series
regime_data = your_regime_assignments  # Optional pd.Series

# Apply balancing and weighting
result = manager.balance_and_weight_data(
    X, y,
    additional_features={'regime': regime_data, 'volatility': volatility_data},
    dataset_characteristics={
        'n_samples': len(X),
        'n_classes': y.nunique(),
        'imbalance_ratio': y.value_counts().min() / y.value_counts().max(),
        'has_regime_data': regime_data is not None,
        'has_volatility_data': 'volatility' in X.columns,
        'dataset_type': 'trading'
    }
)

# Use balanced data for training
X_balanced = result['X_balanced']
y_balanced = result['y_balanced']
sample_weights = result['sample_weights']

# Train your model
model.fit(X_balanced, y_balanced, sample_weight=sample_weights)
```

### Integration with Training Pipelines

```python
from src.training.steps.pre_training.profit_labeling.enhanced_balancing_integration import (
    integrate_with_analyst_training,
    integrate_with_tactician_training
)

# For Analyst training
analyst_result = integrate_with_analyst_training(X, y, regime_data)
X_balanced = analyst_result['X_balanced']
y_balanced = analyst_result['y_balanced']
sample_weights = analyst_result['sample_weights']

# For Tactician training
tactician_result = integrate_with_tactician_training(X, y, regime_data)
X_balanced = tactician_result['X_balanced']
y_balanced = tactician_result['y_balanced']
sample_weights = tactician_result['sample_weights']
```

## 🔧 Configuration Options

### Balancing Configuration

```python
from src.training.steps.pre_training.profit_labeling.label_balancing import (
    BalancingConfig, BalancingTechnique
)

config = BalancingConfig(
    balancing_technique=BalancingTechnique.ADAPTIVE,
    under_sampling_ratio=0.7,  # Keep 70% of majority class
    over_sampling_ratio=0.3,   # Generate 30% synthetic samples
    stratified_batching=True,  # Use stratified mini-batches
    adaptive_imbalance_threshold=0.1,  # Threshold for imbalance detection
    adaptive_min_samples=50,   # Minimum samples per class
    enable_quality_control=True,
    random_state=42
)
```

### Weighting Configuration

```python
from src.training.steps.pre_training.profit_labeling.label_balancing import (
    WeightingConfig, WeightingScheme
)

config = WeightingConfig(
    weighting_scheme=WeightingScheme.INFORMATION_CONTENT,
    volatility_window=20,         # Volatility lookback window
    confidence_scale=2.0,         # Confidence weight multiplier
    time_decay_half_life=30,      # Days for time decay
    regime_frequency_threshold=0.2,  # Threshold for rare regimes
    regime_weight_multiplier=5.0,    # Multiplier for rare regimes
    information_entropy_weight=0.3,  # Weight for entropy component
    information_uncertainty_weight=0.3,  # Weight for uncertainty component
    information_volatility_weight=0.2,   # Weight for volatility component
    information_regime_weight=0.2,       # Weight for regime component
    min_weight=0.1,               # Minimum sample weight
    max_weight=10.0,              # Maximum sample weight
    weight_normalization="l2"     # Normalization method
)
```

## 🎛️ Balancing Techniques

### Under-Sampling
- **When to use**: Abundant no-trade samples
- **What it does**: Reduces majority class to balance dataset, faster training
- **Configuration**: `under_sampling_ratio` controls how much to keep

### Over-Sampling (SMOTE/ADASYN)
- **When to use**: Few positive samples
- **What it does**: Synthesizes similar positives to improve recall
- **Configuration**: `over_sampling_strategy` = "smote" or "adasyn"

### Mixup Augmentation
- **When to use**: Very few positive samples
- **What it does**: Creates synthetic samples by mixing existing ones
- **Configuration**: Uses beta distribution for mixing ratios

### Stratified Batching
- **When to use**: Streaming training, large datasets
- **What it does**: Ensures each mini-batch sees balanced classes
- **Configuration**: `batch_size` and `min_samples_per_class`

### Adaptive Balancing
- **When to use**: Unknown dataset characteristics
- **What it does**: Automatically selects the best technique based on data
- **Configuration**: `adaptive_imbalance_threshold` and `adaptive_min_samples`

## ⚖️ Weighting Schemes

### Volatility Weighting
```python
# Weight inversely proportional to volatility
weights = 1.0 / (volatility + volatility_floor)
```
- **Benefit**: De-emphasizes noisy high-volatility periods
- **Use when**: Market conditions vary significantly

### Confidence Weighting
```python
# Weight by label confidence/probability
weights = confidence_scale * label_confidence
```
- **Benefit**: Emphasizes high-confidence, informative labels
- **Use when**: You have label confidence scores

### Event Overlap Weighting (López de Prado)
```python
# Weight inversely to event overlap count
weights = 1.0 / (1 + overlap_count * overlap_decay)
```
- **Benefit**: Prevents duplicated exposure from overlapping labels
- **Use when**: Labels have temporal overlap

### Time Decay Weighting
```python
# Exponential decay by recency
weights = exp(-days_since / half_life)
```
- **Benefit**: Keeps model adaptive to latest market dynamics
- **Use when**: Markets are non-stationary

### Regime-Aware Weighting
```python
# Weight inversely to regime frequency
regime_freq = regime_labels.value_counts(normalize=True)
weights = 1.0 / regime_freq[regime_labels]
```
- **Benefit**: Ensures balanced exposure to all market regimes
- **Use when**: You have regime classifications

### Information Content Weighting
```python
# Combined weighting using geometric mean
combined_weights = (volatility_weights * confidence_weights * 
                   overlap_weights * time_weights * regime_weights) ** (1/5)
```
- **Benefit**: Combines multiple information sources for optimal weighting
- **Use when**: You have multiple information sources available

## 🔧 Advanced Usage

### Custom Balancing Strategy

```python
from src.training.steps.pre_training.profit_labeling.label_balancing import (
    LabelBalancer, BalancingConfig, BalancingTechnique
)

# Create custom balancing configuration
config = BalancingConfig(
    balancing_technique=BalancingTechnique.HYBRID,
    under_sampling_ratio=0.8,
    over_sampling_ratio=0.5,
    target_distribution={0: 0.4, 1: 0.3, -1: 0.3}  # Custom distribution
)

balancer = LabelBalancer(config)
X_balanced, y_balanced, _ = balancer.balance_dataset(X, y)
```

### Custom Weighting Scheme

```python
from src.training.steps.pre_training.profit_labeling.label_balancing import (
    SampleWeighter, WeightingConfig, WeightingScheme
)

# Create custom weighting configuration
config = WeightingConfig(
    weighting_scheme=WeightingScheme.INFORMATION_CONTENT,
    volatility_window=30,
    time_decay_half_life=60,
    regime_weight_multiplier=3.0,
    information_volatility_weight=0.5,  # Higher weight on volatility
    information_regime_weight=0.3,      # Higher weight on regime
    information_entropy_weight=0.1,     # Lower weight on entropy
    information_uncertainty_weight=0.1  # Lower weight on uncertainty
)

weighter = SampleWeighter(config)
sample_weights = weighter.compute_weights(X, y, additional_features)
```

### Regime-Aware Rebalancing

```python
from src.training.steps.pre_training.profit_labeling.label_balancing import (
    RegimeAwareBalancer, RegimeConfig
)

config = RegimeConfig(
    enable_regime_detection=True,
    regime_balance_method="inverse_frequency",
    regime_balance_strength=2.0,
    regime_frequency_threshold=0.2,
    regime_weight_multiplier=5.0
)

regime_balancer = RegimeAwareBalancer(config)
regime_weights = regime_balancer.compute_regime_weights(X, y, regime_labels)
```

## 📈 Performance Impact

### Expected Improvements
- **Better Recall**: Models learn to identify positive cases more effectively
- **Reduced Overfitting**: Less bias toward majority class
- **Improved Generalization**: Better performance on unseen data
- **Regime Robustness**: Better performance across different market conditions

### Computational Cost
- **Under-sampling**: Reduces training time
- **Over-sampling**: Increases training time (more samples)
- **SMOTE/ADASYN**: Moderate computational overhead for synthetic sample generation
- **Weighting**: Minimal computational overhead

## 🛠️ Integration Examples

### With Multi-Horizon Profit Labeler (Recommended)

```python
from src.training.steps.pre_training.profit_labeling.enhanced_multi_horizon_labeler import (
    EnhancedMultiHorizonProfitLabeler, EnhancedMultiHorizonConfig
)

# Create enhanced labeler with balancing enabled
config = EnhancedMultiHorizonConfig(
    enable_label_balancing=True,
    enable_sample_weighting=True,
    enable_regime_balancing=True
)

labeler = EnhancedMultiHorizonProfitLabeler(config)

# Generate balanced labels with automatic balancing and weighting
result = await labeler.execute_labeling(
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="15m",
    regime_data=regime_data
)

# Extract balanced labels and weights for training
balanced_labels = result['multi_horizon_labeling_result']['labeled_data']
sample_weights = result['multi_horizon_labeling_result']['sample_weights']
```

### With Existing Training Pipeline (Standalone)

```python
# 1. Generate features and labels (existing code)
features, labels = generate_features_and_labels(market_data)

# 2. Apply balancing and weighting
from src.training.steps.pre_training.profit_labeling.enhanced_balancing_integration import (
    create_trading_balancing_manager
)

manager = create_trading_balancing_manager()
result = manager.balance_and_weight_data(
    features, labels, 
    additional_features={'regime': regime_data, 'volatility': volatility_data}
)

# 3. Train model with balanced data
X_balanced = result['X_balanced']
y_balanced = result['y_balanced']
sample_weights = result['sample_weights']

model = YourModel()
model.fit(X_balanced, y_balanced, sample_weight=sample_weights)
```

### Validation Fairness Check

```python
# Check if validation set is representative
fairness_report = manager.check_validation_fairness(
    train_data={'y': y_train, 'regime': regime_train},
    val_data={'y': y_val, 'regime': regime_val}
)

if not fairness_report['class_ratio_fair']:
    print("⚠️ Validation class ratios are not representative!")

if not fairness_report['regime_mix_fair']:
    print("⚠️ Validation regime mix is not representative!")
```

## 🔍 Monitoring and Debugging

### Balancing Report
```python
# Get detailed balancing report
result = manager.balance_and_weight_data(X, y, additional_features)

print(f"Original samples: {result['original_samples']}")
print(f"Balanced samples: {result['balanced_samples']}")
print(f"Class distribution before: {result['class_distribution_before']}")
print(f"Class distribution after: {result['class_distribution_after']}")
print(f"Weight statistics: {result['weight_statistics']}")
print(f"Balancing technique: {result['balancing_technique']}")
print(f"Weighting scheme: {result['weighting_scheme']}")
```

### Weight Distribution Analysis
```python
# Analyze weight distribution
weights = result['sample_weights']
print(f"Weight statistics: mean={weights.mean():.3f}, std={weights.std():.3f}")
print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

# Check for extreme weights
extreme_weights = (weights > 5.0) | (weights < 0.2)
print(f"Extreme weight percentage: {extreme_weights.mean()*100:.1f}%")
```

### Performance Monitoring
```python
# Get comprehensive monitoring report
report = manager.get_balancing_report()

print(f"Monitoring data: {report['monitoring_data']}")
print(f"Performance metrics: {report['performance_metrics']}")
print(f"System config: {report['balancing_system_config']}")
print(f"Timestamp: {report['timestamp']}")
```

## 🚨 Common Issues and Solutions

### Issue: Overly Aggressive Balancing
**Problem**: Too much balancing can remove important information
**Solution**: Start with conservative ratios (0.7-0.8) and gradually increase

### Issue: Synthetic Samples Too Similar
**Problem**: SMOTE samples may not capture complex relationships
**Solution**: Use mixup augmentation or reduce over-sampling ratio

### Issue: Weight Distribution Too Extreme
**Problem**: Some samples get very high/low weights
**Solution**: Adjust `min_weight`/`max_weight` or use weight normalization

### Issue: Poor Regime Detection
**Problem**: Regime-aware weighting doesn't help if regimes are poorly defined
**Solution**: Validate regime classification quality first

### Issue: Memory Issues with Large Datasets
**Problem**: System runs out of memory with very large datasets
**Solution**: Use `max_samples_for_balancing` parameter or enable memory sampling

## 📚 References

1. **López de Prado, M. (2018)**: "Advances in Financial Machine Learning" - Event overlap weighting
2. **SMOTE**: Chawla et al. (2002) - Synthetic Minority Over-sampling Technique
3. **ADASYN**: He et al. (2008) - Adaptive Synthetic Sampling
4. **Mixup**: Zhang et al. (2018) - Beyond Empirical Risk Minimization

## 🤝 Contributing

The system is designed to be modular and extensible:

1. **Custom Balancing Techniques**: Add new balancing methods to `LabelBalancer`
2. **Custom Weighting Schemes**: Add new weighting methods to `SampleWeighter`
3. **Custom Regime Methods**: Extend `RegimeAwareBalancer`
4. **Custom Fairness Metrics**: Add new fairness checks to `ValidationFairnessChecker`

## 🎯 Expected Impact

This system should significantly improve your model's performance by:

- **Better Recall**: Models learn to identify positive cases more effectively
- **Reduced Overfitting**: Less bias toward majority "no-trade" class
- **Improved Generalization**: Better performance on unseen data
- **Regime Robustness**: Better performance across different market conditions

The system is designed to be **production-ready** and is now **fully integrated** into the Multi-Horizon Profit Labeler. All models in your pipeline will automatically benefit from:

- **Automatic Label Balancing**: Applied during label generation
- **Intelligent Sample Weighting**: Based on volatility, confidence, and regime information
- **Regime-Aware Processing**: Different balancing strategies per market regime
- **Validation Fairness**: Ensured representative validation sets

The integration is **seamless** - no changes needed to your existing training code. Simply use the Multi-Horizon Profit Labeler and all downstream models will receive properly balanced and weighted labels.

## 📄 License

This module is part of the Ares Trading System and follows the same licensing terms.
