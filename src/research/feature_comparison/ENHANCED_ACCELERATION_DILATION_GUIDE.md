# Enhanced Feature Acceleration and Window Dilation Guide

This guide covers the statistically robust feature acceleration and window dilation system with proper time-series validation, multiple testing control, and production hygiene.

## Overview

The enhanced system addresses all the refinements requested:

1. **Statistical Correctness**: Time-series CV, Diebold-Mariano tests, multiple testing control
2. **Robust MI/HSIC Estimation**: kNN MI with bootstrap confidence intervals
3. **Dilation Semantics**: Proper EMA span mapping and scale equivalence checks
4. **Cost/Turnover Awareness**: Pareto optimization for FQS vs turnover
5. **Drift & Production Hygiene**: PSI monitoring, shadow features, zero-vol guards
6. **Redundancy & Family Diversity**: mRMR within families, VIF recomputation
7. **Reporting & Traceability**: Comprehensive variant cards and decision rationale

## Key Enhancements

### 1. Statistical Correctness

#### Time-Series CV with Purged and Embargoed Splits
```python
# Purged and embargoed CV to prevent lookahead bias
tscv = self._create_purged_embargoed_cv(X, y)
# 5% purging between train and test sets
# Embargo period to prevent data leakage
```

#### Diebold-Mariano Test for MSE Improvement
```python
# Statistical test for forecast accuracy improvement
dm_stat, dm_pvalue = self._diebold_mariano_test(base_errors, joint_errors)
# Only accept if p ≤ 0.05 (after FDR correction)
```

#### Multiple Testing Control
```python
# Benjamini-Hochberg FDR correction across all variants
from statsmodels.stats.multitest import multipletests
rejected, pvals_corrected, _, _ = multipletests(pvalues, alpha=0.1, method='fdr_bh')
```

### 2. Robust MI/HSIC Estimation

#### kNN MI with Multiple k Values
```python
# Use k ∈ {5, 10} and report median across folds
mi_values = []
for k in [5, 10]:
    mi = mutual_info_regression(X, y, n_neighbors=k)[0]
    mi_values.append(mi)
median_mi = np.median(mi_values)
```

#### Bootstrap Confidence Intervals for Conditional MI
```python
# CMI(X; Y | Z) = MI(X, Z; Y) - MI(Z; Y)
cmi_ci = self._bootstrap_cmi_ci(base, variant, y)
# Keep only if CI low > 0
```

### 3. Dilation Semantics & EMA Quirks

#### Proper EMA Span Mapping
```python
# For EMA: effective window ≈ 2/(α) - 1
# Map 3× lookback to span, not window length
if 'ema_' in feature_name:
    original_span = extract_ema_span(feature_name)
    new_span = int(original_span * factor)
    dilated = feature.ewm(span=new_span).mean()
```

#### Scale Equivalence Check
```python
# Auto-drop if dilated EMA is >0.97 correlated with existing larger-span EWM
if self._is_scale_equivalent(dilated, original, factor):
    return None  # Drop due to scale equivalence
```

### 4. Cost/Turnover Awareness

#### Turnover Calculation
```python
# Track turnover as avg|signal_t - signal_{t-1}|
turnover = feature.diff().abs().mean()
```

#### Pareto Optimization
```python
# Find Pareto frontier for (FQS ↑, turnover ↓)
pareto_frontier = self._find_pareto_frontier(feature_metrics)
# Only keep variants on the frontier
```

### 5. Drift & Production Hygiene

#### PSI Monthly Monitoring
```python
# Compute PSI monthly (train vs recent)
psi_base = self._calculate_psi_monthly(base_feature)
psi_variant = self._calculate_psi_monthly(variant_feature)
psi_delta = abs(psi_variant - psi_base)
# Keep only if ΔPSI ≤ +0.05 vs base
```

#### Shadow Feature Check
```python
# Add randomized shadow per real feature
shadow_perm_imp = self._calculate_shadow_perm_imp(variant_feature)
# Drop any variant with perm-imp below shadow by ≥ 1σ
```

#### Zero/Near-Zero Vol Guards
```python
# Clamp denominator: σ ← max(σ, ε)
def _clamp_volatility(self, series, epsilon=1e-8):
    return series.replace(0, epsilon).clip(lower=-1/epsilon, upper=1/epsilon)
# Log any clamping rate >0.1%
```

### 6. Redundancy & Family Diversity

#### mRMR Within Families
```python
# Enforce mRMR within a family before cross-family selection
# Prevents single base spawning 3 near-duplicates
```

#### VIF Recomputation
```python
# After acceptance, re-compute VIF on kept set
# Pruning can change VIF values
```

### 7. Reporting & Traceability

#### Variant Cards
```python
variant_card = {
    'feature': feature_name,
    'fqs': rank_stability,
    'delta_fqs': fqs - base_fqs,
    'dm_pvalue': dm_pvalue,
    'dm_pvalue_corrected': dm_pvalue_corrected,
    'perm_imp_mean': perm_imp_mean,
    'perm_imp_std': perm_imp_std,
    'mi_median': mi_median,
    'mi_iqr': mi_q75 - mi_q25,
    'rank_stability': rank_stability,
    'max_correlation': max_correlation,
    'vif': vif,
    'turnover': turnover,
    'psi': psi,
    'decision': 'Keep/Drop/Watchlist',
    'rationale': '1-2 sentence explanation'
}
```

#### Global Pareto Plot
```python
# FQS vs turnover scatter plot
# Sankey diagram showing gate removal process
```

## Sensible Default Thresholds

### Statistical Thresholds
- **DM test**: p ≤ 0.05 (after BH-FDR q=0.1)
- **CMI**: CI low > 0 (bootstrap 500 resamples)
- **Rank stability**: ≥ 0.6
- **Regime Δrank**: ≤ 10

### Correlation Thresholds
- **General correlation**: ≤ 0.90
- **Same-family correlation**: ≤ 0.85

### Production Thresholds
- **PSI (monthly)**: ≤ 0.2
- **ΔPSI vs base**: ≤ 0.05
- **Shadow check**: variant's perm-imp > shadow by ≥ 1σ
- **Turnover**: ≤ 0.1

## Edge-Case Guards

### Bounded Oscillators
```python
# Center (e.g., RSI-50) before accel/normalization
if self._is_bounded_feature(feature_series):
    centered = feature_series - 50
```

### Asymmetric Tails
```python
# Winsorize before accel; otherwise accel inflates spikes
winsorized = self._winsorize_robust(centered)
```

### Availability Lag
```python
# Ensure dilated windows don't push features past live inference SLA
# Check feature availability against deployment requirements
```

## Usage Example

```python
from feature_comparison.feature_acceleration_dilation_enhanced import EnhancedFeatureAccelerationDilation

# Initialize enhanced system
system = EnhancedFeatureAccelerationDilation(
    acceleration_lags=[1, 3],
    dilation_factors=[2.0, 3.0],
    mi_k_values=[5, 10],
    dm_alpha=0.05,
    fdr_q=0.1,
    cmi_ci_low_threshold=0.0,
    rank_stability_threshold=0.6,
    correlation_threshold=0.90,
    same_family_correlation_threshold=0.85,
    psi_threshold=0.2,
    psi_delta_threshold=0.05,
    shadow_sigma_threshold=1.0,
    turnover_threshold=0.1,
    enable_matrix_ops=True,
    n_bootstrap=500,
    n_cv_folds=5,
    enable_parallel=True
)

# Generate features
acceleration_features = system.generate_acceleration_features(X)
dilation_features = system.generate_dilation_features(X)

# Evaluate with full statistical rigor
results = system.evaluate_features_with_ts_cv(
    X, y, acceleration_features, dilation_features, base_features
)

# Access results
variant_cards = results['variant_cards']
pareto_frontier = results['pareto_frontier']
global_metrics = results['global_metrics']
```

## What You'll Gain

### Fewer False "Wins"
- **DM + FDR + SPA**: Statistical significance with multiple testing control
- **Bootstrap CI**: Robust confidence intervals for conditional MI
- **Shadow features**: Guard against data snooping

### Cleaner, Cheaper Feature Sets
- **Pareto optimization**: Only keep variants on FQS vs turnover frontier
- **Turnover awareness**: Avoid high-cost, high-turnover features
- **Scale equivalence**: Auto-drop redundant dilated features

### Production-Ready Robustness
- **PSI monitoring**: Detect drift in production
- **Zero-vol guards**: Handle edge cases gracefully
- **Shadow features**: Catch overfitting early

### Traceable Decisions
- **Variant cards**: Complete audit trail for each decision
- **Rationale**: Clear explanation for Keep/Drop/Watchlist decisions
- **Global metrics**: Summary statistics across all evaluations

## Integration with Pre-screening Pipeline

The enhanced system integrates seamlessly with the existing pre-screening pipeline:

```python
from feature_comparison.pre_screening_pipeline import PreScreeningPipeline
from feature_comparison.feature_acceleration_dilation_enhanced import EnhancedFeatureAccelerationDilation

# Step 1: Pre-screen base features
pipeline = PreScreeningPipeline()
prescreening_results = pipeline.run_pre_screening(X, y)

# Step 2: Generate and evaluate variants
base_features = prescreening_results['final_features']
enhanced_system = EnhancedFeatureAccelerationDilation()
variant_results = enhanced_system.evaluate_features_with_ts_cv(
    X, y, acceleration_features, dilation_features, base_features
)

# Step 3: Combine results
final_features = base_features + [
    f for f, c in variant_results['variant_cards'].items() 
    if c['decision'] == 'Keep'
]
```

This approach ensures that only high-quality base features are considered for variant generation, and only statistically significant, practically useful variants are added to the final feature set.