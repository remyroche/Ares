# Pre-Training Validation Integration Guide

## Overview

This document describes the comprehensive validation framework implemented for the pre-training pipeline. The framework addresses all 7 critical aspects identified in the pre-training audit.

## 📋 Components Overview

### 1. **TimeSplitManager** (`time_split_manager.py`)
**Purpose:** Ensure proper temporal data segmentation without lookahead bias

**Key Features:**
- Chronological train/validation/test splitting (70/20/10 default)
- Purged K-fold cross-validation with configurable windows
- Lookahead bias detection and prevention
- Distribution analysis per segment
- Regime-aware splitting support

**Usage:**
```python
from src.training.steps.pre_training.time_split_manager import (
    TimeSplitManager, TimeSplitConfig
)

# Configure
config = TimeSplitConfig(
    train_ratio=0.70,
    validation_ratio=0.20,
    test_ratio=0.10,
    enable_purging=True,
    purge_window=pd.Timedelta(hours=24),
    embargo_window=pd.Timedelta(hours=12)
)

# Create manager
manager = TimeSplitManager(config)

# Create splits
splits = manager.create_temporal_split(
    data=market_data,
    timestamp_column='timestamp',
    target_columns=['target_small', 'target_medium'],
    regime_column='regime_state'
)

# Access splits
train_data = splits['train']
val_data = splits['val']
test_data = splits['test']

# Get metadata
train_metadata = manager.get_split_metadata('train')
print(f"Train period: {train_metadata.timestamp_range}")
print(f"Target volatility: {train_metadata.target_volatility}")

# Validate no lookahead
validation_results = manager.validate_no_lookahead(
    train_data, val_data, test_data
)
```

### 2. **EnhancedLabelDesigner** (`enhanced_label_design.py`)
**Purpose:** Create high-quality labels with transaction cost adjustment

**Key Features:**
- Transaction cost modeling (maker/taker fees, slippage)
- Triple-barrier method for label generation
- Regime-dependent labeling with adaptive thresholds
- Non-overlapping sample generation
- Meta-labeling for sizing models
- Proper volatility estimation with freezing

**Usage:**
```python
from src.training.steps.pre_training.enhanced_label_design import (
    EnhancedLabelDesigner,
    TransactionCostConfig,
    VolatilityConfig,
    TripleBarrierConfig
)

# Configure transaction costs
cost_config = TransactionCostConfig(
    maker_fee=0.0002,
    taker_fee=0.0004,
    slippage_bps=2.0
)

# Configure volatility
vol_config = VolatilityConfig(
    lookback_window=48,
    method="ewm",
    ewm_halflife=24,
    freeze_during_training=True
)

# Configure barriers
barrier_config = TripleBarrierConfig(
    profit_barrier_sigma=2.0,
    stop_loss_barrier_sigma=2.0,
    max_holding_period=24
)

# Create designer
designer = EnhancedLabelDesigner(cost_config, vol_config, barrier_config)

# Calculate volatility
volatility = designer.calculate_volatility(prices)

# Adjust returns for costs
adjusted_returns = designer.adjust_returns_for_costs(forward_returns)

# Create triple-barrier labels
labels, touch_times, returns = designer.create_triple_barrier_labels(
    prices=prices,
    volatility=volatility,
    horizons=[1, 3, 6, 12, 24]
)

# Create non-overlapping samples
non_overlapping_labels = designer.create_non_overlapping_samples(
    labels=labels,
    horizon=12,
    touch_times=touch_times
)

# Create regime-dependent labels
regime_labels = designer.create_regime_dependent_labels(
    forward_returns=forward_returns,
    volatility=volatility,
    regimes=regime_assignments
)

# Validate label quality
quality_metrics = designer.validate_label_quality(labels, returns)
```

### 3. **FeatureDriftMonitor** (`feature_drift_monitor.py`)
**Purpose:** Monitor feature drift and ensure stability

**Key Features:**
- KL divergence and Jensen-Shannon distance calculation
- Feature stability across regimes (KS test)
- Nested cross-validation for feature selection
- Feature correlation clustering
- VIF (Variance Inflation Factor) analysis
- Temporal stability tracking

**Usage:**
```python
from src.training.steps.pre_training.feature_drift_monitor import (
    FeatureDriftMonitor,
    DriftThresholds
)

# Configure thresholds
thresholds = DriftThresholds(
    max_kl_divergence=0.5,
    max_mean_shift=2.0,
    max_std_ratio=2.0,
    max_correlation=0.9,
    max_vif=10.0
)

# Create monitor
monitor = FeatureDriftMonitor(thresholds)

# Detect drift between train and validation
drift_reports = monitor.detect_feature_drift(
    train_features=train_features,
    val_features=val_features
)

# Check for drifted features
for feature, report in drift_reports.items():
    if report.drift_detected:
        print(f"Drift detected in {feature}: KL={report.kl_divergence:.3f}")

# Perform nested CV
nested_results = monitor.perform_nested_cv(
    X=features,
    y=targets,
    estimator=Ridge(),
    inner_cv=3,
    outer_cv=5
)

# Get stable features (appear in >60% of folds)
stable_features = [
    feature for feature, result in nested_results.items()
    if result.stable
]

# Cluster correlated features
clusters = monitor.cluster_correlated_features(features)

# Calculate VIF
vif_values = monitor.calculate_vif(features)
high_vif_features = [f for f, v in vif_values.items() if v > 10.0]

# Track stability over time
stability_data = monitor.track_feature_stability_over_time(
    features=features,
    window_size=100,
    step_size=20
)

# Export drift report
monitor.export_drift_report("outputs/drift_report.json")
```

### 4. **EnhancedLookbackOptimizer** (`enhanced_lookback_optimizer.py`)
**Purpose:** Optimize feature lookback with proper constraints

**Key Features:**
- Constrained search space (e.g., 5-300 bars)
- Multiple objective functions (Sharpe, IC, R²)
- Regularization to avoid extreme lookbacks
- Stability analysis across time segments
- Sensitivity analysis with bootstrap resampling

**Usage:**
```python
from src.training.steps.pre_training.enhanced_lookback_optimizer import (
    EnhancedLookbackOptimizer,
    LookbackConstraints,
    OptimizationObjective
)

# Configure constraints
constraints = LookbackConstraints(
    min_lookback=5,
    max_lookback=300,
    search_step=5,
    enable_regularization=True,
    regularization_strength=0.1,
    preferred_lookback=50,
    min_stability_score=0.7
)

# Configure objective
objective = OptimizationObjective(
    objective_type='ic',  # or 'sharpe', 'r2'
    maximize=True,
    in_sample_weight=0.3,
    out_of_sample_weight=0.7
)

# Create optimizer
optimizer = EnhancedLookbackOptimizer(constraints, objective)

# Optimize lookback
result = optimizer.optimize_lookback(
    features=features,
    targets=targets,
    n_cv_splits=5
)

print(f"Optimal lookback: {result.optimal_lookback}")
print(f"Objective score: {result.objective_score:.4f}")
print(f"Stability: {result.stability_score:.3f}")

# Perform sensitivity analysis
sensitivity = optimizer.sensitivity_analysis(
    features=features,
    targets=targets,
    optimal_lookback=result.optimal_lookback,
    perturbation_range=10,
    n_resamples=10
)

if sensitivity['stable']:
    print("✅ Lookback is stable under resampling")
else:
    print(f"⚠️ High sensitivity: {sensitivity['sensitivity']:.3f}")
```

### 5. **EnhancedFeatureSelector** (`enhanced_feature_selection.py`)
**Purpose:** Robust feature selection with economic interpretability

**Key Features:**
- Bootstrap-based stability testing
- Economic theme grouping (trend, momentum, volatility, etc.)
- IC tracking and validation
- Factor portfolio backtesting
- Feature orthogonalization

**Usage:**
```python
from src.training.steps.pre_training.enhanced_feature_selection import (
    EnhancedFeatureSelector,
    STANDARD_THEMES
)

# Create selector
selector = EnhancedFeatureSelector(
    themes=STANDARD_THEMES,
    stability_threshold=0.6,
    min_ic=0.01,
    min_ic_tstat=2.0
)

# Select features with bootstrap
result = selector.select_features_with_bootstrap(
    X=features,
    y=targets,
    n_bootstrap=20,
    subsample_ratio=0.8,
    max_features=100
)

print(f"Selected {len(result.selected_features)} features")
print(f"Stable features: {len(result.stable_features)}")
print(f"Theme coverage: {result.theme_coverage}")

# Check theme distribution
for theme, features in result.features_by_theme.items():
    print(f"  {theme}: {len(features)} features")

# Orthogonalize features
X_orthogonal = selector.orthogonalize_features(
    X=features,
    selected_features=result.selected_features
)

# Backtest factor portfolio
backtest_results = selector.backtest_factor_portfolio(
    X=features,
    y=returns,
    selected_features=result.selected_features
)

print(f"Sharpe ratio: {backtest_results['sharpe_ratio']:.2f}")
print(f"Total return: {backtest_results['total_return']:.2%}")
```

### 6. **PreTrainingValidator** (`pre_training_validation_framework.py`)
**Purpose:** Comprehensive validation of entire pre-training pipeline

**Key Features:**
- Label autocorrelation testing
- Feature-target mutual information
- Feature stability across regimes
- Sharpe ratio of synthetic signals
- Information coefficient validation
- Lookback sensitivity analysis
- Reproducibility checks (git commit, checksums)

**Usage:**
```python
from src.training.steps.pre_training.pre_training_validation_framework import (
    PreTrainingValidator,
    ValidationThresholds
)

# Configure thresholds
thresholds = ValidationThresholds(
    label_autocorr_max=0.1,
    min_mutual_info_percentile=10.0,
    feature_stability_pvalue=0.05,
    min_sharpe_ratio=0.5,
    max_lookback_sensitivity=0.15,
    min_ic_mean=0.02,
    max_ic_mean=0.05,
    min_ic_tstat=2.0
)

# Create validator
validator = PreTrainingValidator(thresholds)

# Run comprehensive validation
report = validator.run_comprehensive_validation(
    labels=labels,
    features=features,
    targets=targets,
    config=pipeline_config,
    lookback_results=lookback_optimization_results,
    regime_column='regime_state'
)

# Check results
if report.all_tests_passed:
    print("✅ All validation tests passed!")
else:
    print(f"❌ {report.failed_tests} tests failed")
    
    # Review failed tests
    for result in report.data_integrity_results:
        if not result.passed:
            print(f"Failed: {result.test_name}")
            print(f"  Score: {result.score:.4f} vs threshold: {result.threshold}")
            print(f"  Warnings: {result.warnings}")
            print(f"  Recommendations: {result.recommendations}")

# Export report
validator.export_report(report, "outputs/validation_report.json")
```

## 🔬 Integration Example

Here's a complete example integrating all components:

```python
import pandas as pd
from datetime import datetime

# Import all components
from src.training.steps.pre_training.time_split_manager import TimeSplitManager
from src.training.steps.pre_training.enhanced_label_design import EnhancedLabelDesigner
from src.training.steps.pre_training.feature_drift_monitor import FeatureDriftMonitor
from src.training.steps.pre_training.enhanced_lookback_optimizer import EnhancedLookbackOptimizer
from src.training.steps.pre_training.enhanced_feature_selection import EnhancedFeatureSelector
from src.training.steps.pre_training.pre_training_validation_framework import PreTrainingValidator

# 1. Create temporal splits
print("=" * 60)
print("STEP 1: Temporal Data Splitting")
print("=" * 60)

split_manager = TimeSplitManager()
splits = split_manager.create_temporal_split(
    data=market_data,
    timestamp_column='timestamp',
    target_columns=['target_small', 'target_medium'],
    regime_column='regime_state'
)

train_data = splits['train']
val_data = splits['val']
test_data = splits['test']

# Validate no lookahead
split_manager.validate_no_lookahead(train_data, val_data, test_data)

# 2. Create enhanced labels
print("\n" + "=" * 60)
print("STEP 2: Enhanced Label Generation")
print("=" * 60)

label_designer = EnhancedLabelDesigner()

# Calculate volatility (frozen at training cutoff)
volatility_train = label_designer.calculate_volatility(
    train_data['close'],
    freeze_at=train_data.index[-1]
)

# Create triple-barrier labels
labels_train, touch_times, returns = label_designer.create_triple_barrier_labels(
    prices=train_data['close'],
    volatility=volatility_train,
    horizons=[1, 3, 6, 12, 24]
)

# Adjust for transaction costs
adjusted_returns = label_designer.adjust_returns_for_costs(returns)

# 3. Feature drift monitoring
print("\n" + "=" * 60)
print("STEP 3: Feature Drift Monitoring")
print("=" * 60)

drift_monitor = FeatureDriftMonitor()

# Detect drift
drift_reports = drift_monitor.detect_feature_drift(
    train_features=train_data[feature_columns],
    val_features=val_data[feature_columns]
)

# Nested CV for feature selection
nested_results = drift_monitor.perform_nested_cv(
    X=train_data[feature_columns],
    y=labels_train.iloc[:, 0],
    estimator=Ridge(),
    inner_cv=3,
    outer_cv=5
)

stable_features = [f for f, r in nested_results.items() if r.stable]

# 4. Lookback optimization
print("\n" + "=" * 60)
print("STEP 4: Lookback Optimization")
print("=" * 60)

lookback_optimizer = EnhancedLookbackOptimizer()

lookback_result = lookback_optimizer.optimize_lookback(
    features=train_data[stable_features],
    targets=labels_train.iloc[:, 0],
    n_cv_splits=5
)

# 5. Feature selection with bootstrap
print("\n" + "=" * 60)
print("STEP 5: Feature Selection")
print("=" * 60)

feature_selector = EnhancedFeatureSelector()

selection_result = feature_selector.select_features_with_bootstrap(
    X=train_data[stable_features],
    y=labels_train.iloc[:, 0],
    n_bootstrap=20,
    max_features=80
)

# Backtest selected features
backtest_results = feature_selector.backtest_factor_portfolio(
    X=train_data[selection_result.selected_features],
    y=train_data['returns'],
    selected_features=selection_result.selected_features
)

# 6. Comprehensive validation
print("\n" + "=" * 60)
print("STEP 6: Comprehensive Validation")
print("=" * 60)

validator = PreTrainingValidator()

validation_report = validator.run_comprehensive_validation(
    labels=labels_train,
    features=train_data[selection_result.selected_features],
    targets=labels_train,
    config={'symbol': 'ETHUSDT', 'exchange': 'binance', 'random_seed': 42},
    lookback_results=lookback_result.to_dict(),
    regime_column='regime_state'
)

# Print summary
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Total tests: {validation_report.total_tests}")
print(f"Passed: {validation_report.passed_tests}")
print(f"Failed: {validation_report.failed_tests}")
print(f"Pass rate: {validation_report.passed_tests / validation_report.total_tests:.1%}")

if validation_report.all_tests_passed:
    print("\n✅ Pre-training validation PASSED - Ready for model training!")
else:
    print("\n⚠️ Some validation tests failed - Review recommendations")

# Export all results
split_manager.export_split_metadata("outputs/split_metadata.json")
drift_monitor.export_drift_report("outputs/drift_report.json")
validator.export_report(validation_report, "outputs/validation_report.json")

print("\n✅ All reports exported to outputs/ directory")
```

## 📊 Validation Tests Summary

| Test | Purpose | Threshold | Status |
|------|---------|-----------|--------|
| Label Autocorrelation | Ensure labels not trivially predictable | ρ(h) < 0.1 for h>3 | ✅ |
| Feature-Target MI | Filter out noise features | Top 10% retained | ✅ |
| Feature Stability | Robustness across regimes | KS test p>0.05 | ✅ |
| Sharpe of Signal | Economic plausibility | >0.5 on validation | ✅ |
| Lookback Sensitivity | Robustness of optimal window | <15% change under resampling | ✅ |
| Information Coefficient | Predictive quality | Mean(IC)≈0.02-0.05; t-stat>2 | ✅ |
| Reproducibility | Scientific rigor | Git commit, checksums captured | ✅ |

## 🎯 Best Practices

1. **Always use temporal splits** - Never use random splits for time series data
2. **Enable purging and embargo** - Prevent label leakage from overlapping horizons
3. **Adjust for transaction costs** - Critical for realistic profit labels
4. **Monitor feature drift** - Retrain if significant drift detected
5. **Use nested CV** - Prevent overfitting in feature selection
6. **Ensure theme coverage** - Maintain diversity across economic factors
7. **Track reproducibility** - Always log git commit, random seed, data checksum
8. **Validate comprehensively** - Run all validation tests before training

## 📚 References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Bailey, D. H., et al. (2014). "Pseudomathematics and Financial Charlatanism"
- Cochrane, J. H. (2011). "Presidential Address: Discount Rates"

## 🔧 Troubleshooting

### Common Issues

1. **High feature drift detected**
   - Solution: Increase training data, use regime-specific models, or retrain more frequently

2. **Low IC values**
   - Solution: Engineer better features, check for data quality issues, validate lookback windows

3. **Unstable lookback optimization**
   - Solution: Increase regularization, constrain search space more tightly, use longer training periods

4. **Poor factor portfolio Sharpe**
   - Solution: Adjust transaction costs, check label quality, validate feature orthogonality

## 📞 Support

For questions or issues, please refer to the code documentation or contact the ML team.