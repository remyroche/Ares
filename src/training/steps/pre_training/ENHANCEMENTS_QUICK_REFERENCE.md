# Pre-Training Enhancements - Quick Reference Guide

## 🎯 Quick Start

### Installation
No new dependencies required - all enhancements use existing stack.

### Import Everything
```python
from src.training.steps.pre_training.pipeline_enhancements_integration import (
    EnhancedPipelineOrchestrator,
    EnhancedPipelineConfig
)
```

### Basic Usage
```python
# Create with defaults
orchestrator = EnhancedPipelineOrchestrator()

# Or with custom config
config = EnhancedPipelineConfig(
    enable_time_splitting=True,
    enable_enhanced_labeling=True,
    enable_redundancy_control=True,
    enable_drift_monitoring=True,
    enable_enhanced_lookback=True,
    enable_enhanced_selection=True,
    enable_quantitative_validation=True,
    enable_reproducibility_tracking=True
)
orchestrator = EnhancedPipelineOrchestrator(config=config)
```

---

## 📦 New Modules Summary

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `time_split_manager.py` | Proper temporal splitting | `TimeSplitManager`, `SplitConfig` |
| `quantitative_validation.py` | 6 validation tests | `QuantitativeValidator`, `ValidationReport` |
| `feature_redundancy_control.py` | Redundancy & drift | `RedundancyController`, `DriftMonitor` |
| `enhanced_lookback_optimizer.py` | Robust lookback | `EnhancedLookbackOptimizer` |
| `enhanced_feature_selector.py` | Stable selection | `EnhancedFeatureSelector` |
| `enhanced_label_design.py` | Better labels | `EnhancedLabeler` |
| `reproducibility_tracker.py` | Full tracking | `ReproducibilityTracker` |
| `pipeline_enhancements_integration.py` | Unified API | `EnhancedPipelineOrchestrator` |

---

## 🔧 Enhanced Existing Files

### 1. `profit_labeling/volatility_aware_labeler.py`
**New Config Parameters**:
```python
enable_non_overlapping_sampling: bool = True
volatility_lookback_frozen: int = 48
volatility_estimation_method: str = "std"
enable_triple_barrier: bool = False
profit_target_sigma: float = 2.0
stop_loss_sigma: float = 2.0
transaction_cost_bps: float = 6.0
adjust_labels_for_costs: bool = True
```

### 2. `feature_lookback_optimization/core/optimizer.py`
**New Fields in `LookbackConstraints`**:
```python
optimization_objective: str = "max_ic"
preferred_min: float = 40.0
preferred_max: float = 80.0
enable_bootstrap_stability: bool = True
n_bootstrap_samples: int = 10
track_sensitivity: bool = True
```

**New Fields in `OptimizationResult`**:
```python
resampled_lookbacks: List[int]
objective_name: str
regularization_penalty: float
raw_objective_value: float
is_stable: bool
```

### 3. `final_feature_selection_step.py`
**New Config Parameters**:
```python
preserve_economic_themes: bool = True
min_features_per_theme: int = 1
track_ic_over_time: bool = True
ic_window_size: int = 100
min_ic_threshold: float = 0.02
min_ic_t_stat: float = 2.0
validate_with_factor_portfolio: bool = True
min_factor_sharpe: float = 0.3
```

---

## 🚀 Common Operations

### 1. Data Splitting
```python
train, val, test, split_result = orchestrator.process_data_splitting(
    data=ohlcv_df,
    regime_labels=regime_series  # Optional
)
print(f"Train: {split_result.train_size}, Val: {split_result.validation_size}")
```

### 2. Enhanced Labeling
```python
labels = orchestrator.process_labeling(
    prices=close_prices,
    horizon_bars=48,
    ohlc=ohlcv_df,
    regime_labels=regime_series
)
```

### 3. Redundancy Control
```python
reduced_features, report = orchestrator.process_feature_redundancy(
    features=feature_df,
    feature_importance=importance_dict
)
print(f"Reduced from {len(feature_df.columns)} to {len(report.retained_features)}")
```

### 4. Drift Monitoring
```python
drift_report = orchestrator.process_drift_monitoring(
    train_features=train_features,
    val_features=val_features
)
if drift_report.drifted_features:
    print(f"⚠️ Drift detected in: {drift_report.drifted_features}")
```

### 5. Lookback Optimization
```python
lookback_result = orchestrator.process_lookback_optimization(
    prices=close_prices,
    labels=labels['label']
)
print(f"Optimal lookback: {lookback_result.optimal_lookback}")
print(f"Stability: {lookback_result.stability_score:.3f}")
```

### 6. Feature Selection
```python
selected_features, selection_result = orchestrator.process_feature_selection(
    features=feature_df,
    labels=labels['label'],
    target_n_features=80
)
print(f"Selected {selection_result.n_features} features")
print(f"Theme distribution: {selection_result.theme_distribution}")
```

### 7. Validation
```python
validation_report = orchestrator.validate_outputs(
    labels=labels,
    features=selected_features,
    lookback_results=lookback_result.__dict__
)
print(f"Validation: {'✅ PASSED' if validation_report.passed else '❌ FAILED'}")
for result in validation_report.results:
    print(f"  {result.test_name}: {result.status.value}")
```

### 8. Save Reproducibility Manifest
```python
manifest = orchestrator.save_reproducibility_manifest(
    output_path=Path('artifacts/manifest.json'),
    metadata={'experiment': 'test_run_001'}
)
```

---

## ⚙️ Configuration Presets

### Preset 1: Maximum Quality (Slow)
```python
config = EnhancedPipelineConfig(
    # Time splitting
    enable_purging=True,
    purge_window_hours=48,
    embargo_window_hours=24,
    
    # Labeling
    enable_enhanced_labeling=True,
    enable_triple_barrier=True,
    adjust_labels_for_costs=True,
    
    # Redundancy
    correlation_threshold=0.80,
    enable_vif=True,
    
    # Drift
    kl_threshold=0.10,
    
    # Lookback
    enable_lookback_regularization=True,
    n_bootstrap_samples=15,
    
    # Selection
    n_bootstrap_folds=10,
    min_selection_frequency=0.70,
    
    # Validation
    strict_validation=True
)
```

### Preset 2: Balanced (Recommended)
```python
config = EnhancedPipelineConfig()  # Defaults
```

### Preset 3: Fast Development (Quick)
```python
config = EnhancedPipelineConfig(
    enable_purging=False,
    enable_vif=False,
    enable_drift_monitoring=False,
    n_bootstrap_samples=3,
    n_bootstrap_folds=3,
    strict_validation=False
)
```

---

## 📊 6 Validation Tests

| # | Test | Metric | Threshold | Purpose |
|---|------|--------|-----------|---------|
| 1 | Label Autocorrelation | ρ(h) | < 0.1 for h>3 | Not trivially predictable |
| 2 | Feature-Target MI | Mean MI | ≥ 0.01 | Filter noise features |
| 3 | Feature Stability | KS p-value | > 0.05 | Robust across regimes |
| 4 | Synthetic Sharpe | Sharpe | ≥ 0.5 | Economic plausibility |
| 5 | Lookback Sensitivity | CV | < 0.15 | Stable under resampling |
| 6 | Information Coefficient | Mean IC, t-stat | ≥ 0.02, > 2.0 | Predictive quality |

---

## 🎨 Economic Themes for Features

Automatically inferred from feature names:
- **TREND**: ma, sma, ema, trend
- **MOMENTUM**: rsi, macd, momentum, roc
- **VOLATILITY**: vol, atr, std, volatility
- **MICROSTRUCTURE**: spread, bid, ask, orderbook
- **VOLUME**: volume, vwap
- **TECHNICAL**: default for others

---

## 🔍 Troubleshooting Quick Fixes

### Problem: Memory Error
```python
config.n_bootstrap_samples = 3  # Reduce from 10
config.n_bootstrap_folds = 3  # Reduce from 5
config.enable_vif = False  # Disable VIF
```

### Problem: Too Slow
```python
config.enable_drift_monitoring = False
config.lookback_max = 200  # Reduce from 300
config.enable_bootstrap_stability = False
```

### Problem: No Features Selected
```python
config.min_selection_frequency = 0.40  # Lower from 0.60
config.correlation_threshold = 0.90  # Higher from 0.85
```

### Problem: Validation Failing
```python
config.strict_validation = False
config.min_ic_threshold = 0.01  # Lower from 0.02
config.min_factor_sharpe = 0.2  # Lower from 0.3
```

---

## 📈 Key Metrics to Monitor

### Data Splitting
- Train/Val/Test sizes
- Timestamp ranges
- Purge/embargo gap sizes

### Labeling
- Label balance (should be 35-65%)
- Volatility distribution
- Transaction cost impact

### Redundancy
- Feature reduction rate (30-50% typical)
- Max VIF score (should be < 10)
- Cluster sizes

### Drift
- Number of drifted features
- Max KL divergence
- Correlation drift magnitude

### Lookback
- Optimal lookback value
- Stability score (should be > 0.7)
- Resampled lookback std

### Selection
- Number of selected features
- Selection frequencies
- Theme distribution
- Factor portfolio Sharpe

### Validation
- Number of passed tests (6/6 ideal)
- Number of warnings
- Specific test values

---

## 🔗 Integration with Existing Pipeline

### Option 1: Replace Existing Components
```python
# Old
from src.training.steps.pre_training import multi_horizon_profit_labeler
labels = multi_horizon_profit_labeler.apply(data)

# New
orchestrator = EnhancedPipelineOrchestrator()
labels = orchestrator.process_labeling(prices, horizon_bars=48)
```

### Option 2: Add as Enhancement Layer
```python
# Keep existing pipeline
existing_labels = run_existing_pipeline(data)

# Add enhancements
orchestrator = EnhancedPipelineOrchestrator()
validation_report = orchestrator.validate_outputs(
    labels=existing_labels,
    features=existing_features
)
```

### Option 3: Gradual Migration
```python
config = EnhancedPipelineConfig(
    enable_time_splitting=True,  # Enable first
    enable_enhanced_labeling=False,  # Add next week
    enable_redundancy_control=False,  # Add later
    # ... etc
)
```

---

## 💡 Tips & Best Practices

1. **Start with defaults** - they're well-tuned
2. **Enable one enhancement at a time** during migration
3. **Always check validation report** before deploying
4. **Save reproducibility manifests** for all production runs
5. **Monitor drift continuously** in live systems
6. **Use bootstrap validation** for critical models
7. **Preserve economic themes** for interpretability
8. **Track IC over time** to detect degradation
9. **Use purging and embargo** for realistic evaluation
10. **Run quantitative validation** before each deployment

---

## 📞 Quick Help

### Get Configuration
```python
print(orchestrator.config.__dict__)
```

### Check Module Availability
```python
print(orchestrator.__dict__.keys())
```

### Get Validation Details
```python
for result in validation_report.results:
    print(f"{result.test_name}:")
    print(f"  Status: {result.status.value}")
    print(f"  Value: {result.value:.4f}")
    print(f"  Threshold: {result.threshold:.4f}")
    print(f"  Message: {result.message}")
```

### Export Results
```python
import json

# Export validation report
with open('validation_report.json', 'w') as f:
    json.dump(validation_report.to_dict(), f, indent=2)

# Export reproducibility manifest
manifest.save(Path('manifest.json'))
```

---

**Quick Reference Version**: 1.0.0  
**Last Updated**: 2025-10-08