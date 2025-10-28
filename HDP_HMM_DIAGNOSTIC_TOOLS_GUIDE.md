# HDP-HMM Diagnostic Tools Guide

## 📊 Comprehensive Diagnostic Toolset

This guide documents the complete diagnostic toolset for HDP-HMM regime models, including median & IQR calculations, CRPS/PIT calibration, and tail quantile analysis.

---

## 🎯 7 Key Diagnostic Categories

### 1. Median & IQR of Predictive LL Across Folds ✅

**What it tells you**: Stability of out-of-sample predictive skill

**When to worry**: Large IQR → model overfits certain windows

**Metrics Available**:
```python
{
    'predictive_ll_median': float,      # Median log-likelihood
    'predictive_ll_iqr': float,         # Interquartile range
    'predictive_ll_q25': float,         # 25th percentile
    'predictive_ll_q75': float,         # 75th percentile
    'delta_ll_across_folds': List[float],  # ΔLL per fold
    'predictive_ll_effect_size': float  # Effect size vs noise
}
```

**Interpretation**:
```
✅ GOOD:    IQR < 10% of median, effect_size > 1.0
⚠️ WARNING: IQR 10-20% of median
❌ POOR:    IQR > 20% of median → unstable across windows
```

**Usage**:
```python
# Access diagnostic metrics
predictive_ll_median = metrics.predictive_ll_median
predictive_ll_iqr = metrics.predictive_ll_iqr

# Check stability
if predictive_ll_iqr / abs(predictive_ll_median) < 0.10:
    print("✅ Stable predictive performance across folds")
else:
    print("⚠️ High variation across folds - model may overfit certain windows")
```

---

### 2. Median & IQR of Sharpe and Turnover ✅

**What it tells you**: Economic consistency & tradability

**When to worry**: High Sharpe + high turnover → overfitting or unrealistic

**Metrics Available**:
```python
{
    # Sharpe distribution
    'sharpe_across_folds': List[float],  # Sharpe per fold
    'sharpe_median': float,              # Median Sharpe
    'sharpe_iqr': float,                 # Sharpe IQR
    'sharpe_q25': float,                 # 25th percentile
    'sharpe_q75': float,                 # 75th percentile
    
    # Turnover distribution
    'turnover_across_folds': List[float],  # Turnover per fold
    'turnover_median': float,              # Median turnover
    'turnover_iqr': float,                 # Turnover IQR
    'turnover_q25': float,                 # 25th percentile
    'turnover_q75': float                  # 75th percentile
}
```

**Interpretation**:
```
✅ GOOD:    sharpe_median > 1.0, sharpe_iqr < 0.5, turnover_median < 0.3
⚠️ WARNING: sharpe_iqr > 1.0 OR (sharpe_median > 2.0 AND turnover_median > 0.5)
❌ POOR:    sharpe_median < 0.5 OR sharpe_iqr > 2.0 (unreliable)
```

**Usage**:
```python
# Economic consistency check
sharpe_median = metrics.sharpe_median
sharpe_iqr = metrics.sharpe_iqr
turnover_median = metrics.turnover_median

# Check for overfitting patterns
if sharpe_median > 2.0 and turnover_median > 0.5:
    print("⚠️ High Sharpe + High turnover → possible overfitting")
    print(f"   Sharpe: {sharpe_median:.2f} (IQR: {sharpe_iqr:.2f})")
    print(f"   Turnover: {turnover_median:.1%} (IQR: {metrics.turnover_iqr:.1%})")
```

---

### 3. ARI Across Restarts ✅

**What it tells you**: Regime label stability

**When to worry**: ARI < 0.4–0.5 ⇒ unstable clustering of regimes

**Metrics Available**:
```python
{
    'ari_across_restarts': List[float],  # All ARI values
    'ari_median': float,                 # Median ARI
    'ari_iqr': float,                    # ARI IQR
    'ari_q25': float,                    # 25th percentile
    'ari_q75': float,                    # 75th percentile
    'refit_stability_ari': float,        # Mean ARI (backward compat)
    'refit_stability_nmi': float         # Normalized Mutual Information
}
```

**Interpretation**:
```
✅ GOOD:    ari_median > 0.6, ari_iqr < 0.2
⚠️ WARNING: ari_median 0.4-0.6 OR ari_iqr 0.2-0.3
❌ POOR:    ari_median < 0.4 (unstable regime identification)
```

**Usage**:
```python
# Stability assessment
ari_median = metrics.ari_median
ari_iqr = metrics.ari_iqr

if ari_median < 0.4:
    print("❌ REJECT: Unstable regime identification")
    print(f"   ARI median: {ari_median:.3f} (IQR: {ari_iqr:.3f})")
    print("   Try: Increase kappa, use kmeans initialization, or simplify model")
elif ari_median > 0.6 and ari_iqr < 0.2:
    print("✅ ACCEPT: Stable and consistent regime identification")
```

---

### 4. State Occupancy Distribution ✅

**What it tells you**: Are all regimes meaningful?

**When to worry**: States with <1–3% occupancy → spurious

**Metrics Available**:
```python
{
    'state_occupancy': Dict[int, float],  # Occupancy per state
    'occupancy_distribution': List[float],  # Sorted occupancies
    'occupancy_entropy': float,           # Shannon entropy
    'min_occupancy_pct': float,           # Minimum occupancy %
    'max_occupancy_pct': float,           # Maximum occupancy %
    'tiny_state_count': int               # States with <1% occupancy
}
```

**Interpretation**:
```
✅ GOOD:    min_occupancy_pct > 3%, tiny_state_count = 0
⚠️ WARNING: min_occupancy_pct 1-3% OR tiny_state_count = 1
❌ POOR:    min_occupancy_pct < 1% OR tiny_state_count > 1 (spurious states)
```

**Usage**:
```python
# Check for spurious states
tiny_states = metrics.tiny_state_count
min_occupancy = metrics.min_occupancy_pct

if tiny_states > 0:
    print(f"⚠️ Found {tiny_states} tiny states (<1% occupancy)")
    print(f"   Minimum occupancy: {min_occupancy:.2f}%")
    print("   Consider: Reduce n_states or increase kappa")
else:
    print(f"✅ All states meaningful (min occupancy: {min_occupancy:.2f}%)")

# Analyze occupancy distribution
occupancy_dist = metrics.occupancy_distribution
print(f"\n📊 Occupancy distribution: {occupancy_dist}")
print(f"   Entropy: {metrics.occupancy_entropy:.3f}")
```

---

### 5. Expected Duration Per State ✅

**What it tells you**: Persistence realism

**When to worry**: Too short (e.g., <7 days for daily) ⇒ noise; too long ⇒ under-responsive

**Metrics Available**:
```python
{
    'expected_state_durations': Dict[int, Dict[str, float]],  # E[D] per state
    # Each entry: {'samples': X, 'hours': Y, 'days': Z}
    'min_expected_duration': float,      # Minimum duration (days)
    'max_expected_duration': float,      # Maximum duration (days)
    'duration_quality_flag': str         # 'good'/'acceptable'/'warning'/'poor'
}
```

**Interpretation (for 1h data)**:
```
✅ GOOD:       min_duration ≥ 7 days
✅ ACCEPTABLE: min_duration ≥ 2 days
⚠️ WARNING:    min_duration ≥ 1 day
❌ POOR:       min_duration < 1 day (likely noise)
```

**Usage**:
```python
# Duration quality check
duration_flag = metrics.duration_quality_flag
min_duration = metrics.min_expected_duration
max_duration = metrics.max_expected_duration

print(f"📊 Regime Persistence:")
print(f"   Duration range: {min_duration:.1f} - {max_duration:.1f} days")
print(f"   Quality: {duration_flag}")

if duration_flag == 'poor':
    print("   ❌ REJECT: Regimes too short-lived (likely noise)")
    print("   Action: Increase kappa to 150-300")
elif duration_flag in ['good', 'acceptable']:
    print("   ✅ ACCEPT: Persistent, meaningful regimes")

# Detailed per-state durations
for state, dur in metrics.expected_state_durations.items():
    print(f"   State {state}: {dur['days']:.1f} days ({dur['hours']:.0f} hours)")
```

---

### 6. CRPS or PIT Calibration ✅

**What it tells you**: Predictive density calibration

**When to worry**: Deviations from uniform PIT ⇒ mis-calibrated likelihoods

**Metrics Available**:
```python
{
    'crps_score': float,                 # Continuous Ranked Probability Score
    'pit_uniformity_pvalue': float,      # KS test p-value
    'probability_calibration_score': float,  # Overall calibration (0-1)
    'predictive_density_calibration': str    # 'well_calibrated'/'too_narrow'/'too_wide'
}
```

**Interpretation**:
```
✅ GOOD:    pit_uniformity_pvalue > 0.05, calibration_score > 0.7
⚠️ WARNING: pit_uniformity_pvalue > 0.01 AND calibration_score > 0.5
❌ POOR:    pit_uniformity_pvalue < 0.01 (mis-calibrated likelihoods)
```

**Usage**:
```python
# Calibration assessment
pit_pvalue = metrics.pit_uniformity_pvalue
crps = metrics.crps_score
calib_score = metrics.probability_calibration_score
calib_flag = metrics.predictive_density_calibration

print(f"📊 Predictive Calibration:")
print(f"   PIT uniformity p-value: {pit_pvalue:.4f}")
print(f"   CRPS score: {crps:.4f}")
print(f"   Calibration: {calib_flag} (score={calib_score:.3f})")

if pit_pvalue < 0.01:
    print("   ⚠️ WARNING: Mis-calibrated predictive densities")
    if calib_flag == 'too_narrow':
        print("   → Underestimates uncertainty (too confident)")
    elif calib_flag == 'too_wide':
        print("   → Overestimates uncertainty (too cautious)")
else:
    print("   ✅ Well-calibrated predictive densities")
```

---

### 7. Simulated vs Empirical Tail Quantiles ✅

**What it tells you**: Distributional fidelity (risk tails)

**When to worry**: Under- or over-represents extreme moves

**Metrics Available**:
```python
{
    'tail_quantile_comparison': Dict[str, Dict[str, float]],
    # Keys: 'q01', 'q05', 'q25', 'q75', 'q95', 'q99'
    # Each: {'empirical': X, 'simulated': Y, 'diff': Z, 'rel_diff': W}
    'tail_coverage_score': float         # How well tails match (0-1)
}
```

**Interpretation**:
```
✅ GOOD:    tail_coverage_score > 0.8, max_rel_diff < 0.2
⚠️ WARNING: tail_coverage_score 0.6-0.8 OR max_rel_diff 0.2-0.4
❌ POOR:    tail_coverage_score < 0.6 (mis-represents extremes)
```

**Usage**:
```python
# Tail quantile analysis
tail_comp = metrics.tail_quantile_comparison
tail_score = metrics.tail_coverage_score

print(f"📊 Tail Quantile Coverage: {tail_score:.3f}")
print("\n   Quantile comparison:")

for q_name, q_data in tail_comp.items():
    emp = q_data['empirical']
    sim = q_data['simulated']
    rel_diff = q_data['rel_diff']
    
    status = "✅" if rel_diff < 0.2 else "⚠️" if rel_diff < 0.4 else "❌"
    print(f"   {status} {q_name}: empirical={emp:.4f}, simulated={sim:.4f}, diff={rel_diff:.1%}")

# Check for tail mis-specification
extreme_tails = ['q01', 'q99']
extreme_errors = [tail_comp[q]['rel_diff'] for q in extreme_tails if q in tail_comp]

if max(extreme_errors) > 0.4:
    print("\n   ⚠️ WARNING: Poor extreme tail coverage")
    print("   → Model under/over-represents crash/rally scenarios")
else:
    print("\n   ✅ Good tail coverage across all quantiles")
```

---

## 🔧 Complete Usage Workflow

### Step 1: Run HDP-HMM with Comprehensive Diagnostics

```python
from src.training.steps.market_analysis.hdp_hmm_clustering import (
    HDPHMMRegimeDiscoveryStep
)

# Configure and run
step = HDPHMMRegimeDiscoveryStep()
config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'run_optimization': True
}

results = await step.execute(config)
metrics = results['metrics']
```

### Step 2: Extract All Diagnostic Metrics

```python
# Organize diagnostics by category
diagnostics = {
    '1. Predictive LL (Median & IQR)': {
        'median': metrics.predictive_ll_median,
        'iqr': metrics.predictive_ll_iqr,
        'q25': metrics.predictive_ll_q25,
        'q75': metrics.predictive_ll_q75,
        'effect_size': metrics.predictive_ll_effect_size
    },
    
    '2. Sharpe & Turnover (Median & IQR)': {
        'sharpe_median': metrics.sharpe_median,
        'sharpe_iqr': metrics.sharpe_iqr,
        'turnover_median': metrics.turnover_median,
        'turnover_iqr': metrics.turnover_iqr
    },
    
    '3. ARI Across Restarts': {
        'ari_median': metrics.ari_median,
        'ari_iqr': metrics.ari_iqr,
        'ari_q25': metrics.ari_q25,
        'ari_q75': metrics.ari_q75
    },
    
    '4. State Occupancy Distribution': {
        'occupancy_dist': metrics.occupancy_distribution,
        'occupancy_entropy': metrics.occupancy_entropy,
        'min_occupancy_pct': metrics.min_occupancy_pct,
        'tiny_state_count': metrics.tiny_state_count
    },
    
    '5. Expected Duration Per State': {
        'durations': metrics.expected_state_durations,
        'min_duration': metrics.min_expected_duration,
        'max_duration': metrics.max_expected_duration,
        'quality_flag': metrics.duration_quality_flag
    },
    
    '6. CRPS & PIT Calibration': {
        'crps_score': metrics.crps_score,
        'pit_pvalue': metrics.pit_uniformity_pvalue,
        'calibration_score': metrics.probability_calibration_score,
        'calibration_flag': metrics.predictive_density_calibration
    },
    
    '7. Tail Quantiles': {
        'tail_comparison': metrics.tail_quantile_comparison,
        'tail_coverage_score': metrics.tail_coverage_score
    }
}
```

### Step 3: Generate Diagnostic Summary Table

```python
import pandas as pd

def create_diagnostic_summary(metrics):
    """Create diagnostic summary table."""
    
    summary = {
        'Diagnostic': [],
        'Value': [],
        'Status': [],
        'Threshold': [],
        'Action': []
    }
    
    # 1. Predictive LL stability
    ll_iqr_ratio = metrics.predictive_ll_iqr / abs(metrics.predictive_ll_median) if metrics.predictive_ll_median else None
    summary['Diagnostic'].append('Predictive LL IQR/Median')
    summary['Value'].append(f"{ll_iqr_ratio:.2%}" if ll_iqr_ratio else "N/A")
    summary['Status'].append("✅" if ll_iqr_ratio and ll_iqr_ratio < 0.10 else "⚠️")
    summary['Threshold'].append("< 10%")
    summary['Action'].append("OK" if ll_iqr_ratio and ll_iqr_ratio < 0.10 else "Check fold consistency")
    
    # 2. Sharpe consistency
    summary['Diagnostic'].append('Sharpe Median')
    summary['Value'].append(f"{metrics.sharpe_median:.3f}" if metrics.sharpe_median else "N/A")
    summary['Status'].append("✅" if metrics.sharpe_median and metrics.sharpe_median > 1.0 else "⚠️")
    summary['Threshold'].append("> 1.0")
    summary['Action'].append("OK" if metrics.sharpe_median and metrics.sharpe_median > 1.0 else "Improve strategy")
    
    # 3. ARI stability
    summary['Diagnostic'].append('ARI Median')
    summary['Value'].append(f"{metrics.ari_median:.3f}" if metrics.ari_median else "N/A")
    summary['Status'].append("✅" if metrics.ari_median and metrics.ari_median > 0.6 else "❌" if metrics.ari_median and metrics.ari_median < 0.4 else "⚠️")
    summary['Threshold'].append("> 0.6")
    summary['Action'].append("OK" if metrics.ari_median and metrics.ari_median > 0.6 else "Increase kappa/restarts")
    
    # 4. Occupancy
    summary['Diagnostic'].append('Tiny States')
    summary['Value'].append(str(metrics.tiny_state_count))
    summary['Status'].append("✅" if metrics.tiny_state_count == 0 else "⚠️")
    summary['Threshold'].append("= 0")
    summary['Action'].append("OK" if metrics.tiny_state_count == 0 else "Reduce n_states")
    
    # 5. Duration quality
    summary['Diagnostic'].append('Duration Quality')
    summary['Value'].append(metrics.duration_quality_flag or "unknown")
    summary['Status'].append("✅" if metrics.duration_quality_flag == 'good' else "❌" if metrics.duration_quality_flag == 'poor' else "⚠️")
    summary['Threshold'].append("good")
    summary['Action'].append("OK" if metrics.duration_quality_flag == 'good' else "Increase kappa")
    
    # 6. Calibration
    summary['Diagnostic'].append('PIT Uniformity p-value')
    summary['Value'].append(f"{metrics.pit_uniformity_pvalue:.4f}" if metrics.pit_uniformity_pvalue else "N/A")
    summary['Status'].append("✅" if metrics.pit_uniformity_pvalue and metrics.pit_uniformity_pvalue > 0.05 else "⚠️")
    summary['Threshold'].append("> 0.05")
    summary['Action'].append("OK" if metrics.pit_uniformity_pvalue and metrics.pit_uniformity_pvalue > 0.05 else "Check calibration")
    
    # 7. Tail coverage
    summary['Diagnostic'].append('Tail Coverage Score')
    summary['Value'].append(f"{metrics.tail_coverage_score:.3f}" if metrics.tail_coverage_score else "N/A")
    summary['Status'].append("✅" if metrics.tail_coverage_score and metrics.tail_coverage_score > 0.8 else "⚠️")
    summary['Threshold'].append("> 0.8")
    summary['Action'].append("OK" if metrics.tail_coverage_score and metrics.tail_coverage_score > 0.8 else "Check tail fit")
    
    return pd.DataFrame(summary)

# Generate and display
summary_table = create_diagnostic_summary(metrics)
print("\n📊 DIAGNOSTIC SUMMARY TABLE")
print("="*80)
print(summary_table.to_string(index=False))
print("="*80)
```

### Step 4: Filter and Select Best Models

```python
def filter_models_by_diagnostics(model_results: List[Dict]) -> List[Dict]:
    """
    Filter models based on comprehensive diagnostic criteria.
    
    Args:
        model_results: List of model results with metrics
        
    Returns:
        Filtered list of acceptable models
    """
    filtered = []
    
    for result in model_results:
        metrics = result['metrics']
        
        # Critical filters
        checks = {
            'duration_quality': metrics.duration_quality_flag not in ['poor'],
            'ari_stability': metrics.ari_median is None or metrics.ari_median >= 0.4,
            'tiny_states': metrics.tiny_state_count <= 1,
            'economic_utility': metrics.sharpe_uplift_vs_baseline is None or metrics.sharpe_uplift_vs_baseline >= 0.0
        }
        
        # Score model
        score = 0
        if metrics.ari_median and metrics.ari_median > 0.6:
            score += 1
        if metrics.duration_quality_flag == 'good':
            score += 1
        if metrics.sharpe_median and metrics.sharpe_median > 1.0:
            score += 1
        if metrics.pit_uniformity_pvalue and metrics.pit_uniformity_pvalue > 0.05:
            score += 1
        if metrics.tail_coverage_score and metrics.tail_coverage_score > 0.8:
            score += 1
        
        result['diagnostic_score'] = score
        result['passes_filters'] = all(checks.values())
        
        if result['passes_filters']:
            filtered.append(result)
    
    # Sort by diagnostic score
    filtered.sort(key=lambda x: x['diagnostic_score'], reverse=True)
    
    return filtered

# Usage
top_models = filter_models_by_diagnostics(all_model_results)
print(f"✅ {len(top_models)}/{len(all_model_results)} models passed diagnostics")

if top_models:
    best_model = top_models[0]
    print(f"\n🏆 Best model: Diagnostic score = {best_model['diagnostic_score']}/5")
```

---

## 📋 Diagnostic Decision Matrix

| Diagnostic | Threshold | Action if Failed |
|------------|-----------|------------------|
| **Predictive LL IQR/Median** | < 10% | Reduce model complexity OR increase training data |
| **Sharpe Median** | > 1.0 | Improve features OR adjust allocation strategy |
| **Sharpe IQR** | < 0.5 | Check for overfitting across folds |
| **ARI Median** | > 0.6 | Increase `kappa`, use `kmeans` init, add restarts |
| **Tiny States** | = 0 | Reduce `n_states` OR increase `kappa` |
| **Duration Quality** | 'good' | Increase `kappa` to 150-300 |
| **PIT p-value** | > 0.05 | Adjust `covariance_floor` OR emission model |
| **Tail Coverage** | > 0.8 | Check GMM mixtures OR increase sampling iterations |

---

## 🎯 Production Checklist

Before deploying, ensure ALL diagnostics pass:

- [ ] **Predictive LL**: IQR/Median < 10%, effect_size > 1.0
- [ ] **Sharpe**: median > 1.0, IQR < 0.5
- [ ] **Turnover**: median < 0.3, IQR < 0.1
- [ ] **ARI**: median > 0.6, IQR < 0.2
- [ ] **Occupancy**: No tiny states, entropy reasonable
- [ ] **Duration**: Quality = 'good', min_duration > 7 days (1h data)
- [ ] **Calibration**: PIT p-value > 0.05, CRPS reasonable
- [ ] **Tail Coverage**: Score > 0.8

---

## 📊 Example Output

```
🔬 COMPREHENSIVE DIAGNOSTIC REPORT
================================================================================

1. PREDICTIVE LL ACROSS FOLDS
   Median:  -145.32
   IQR:      12.45 (8.6% of median) ✅
   Q25/Q75: -151.54 / -139.09
   Effect Size: 1.24 ✅
   → PASS: Stable predictive performance

2. SHARPE & TURNOVER DISTRIBUTION
   Sharpe:   Median=1.35, IQR=0.42 ✅
   Turnover: Median=0.18, IQR=0.05 ✅
   → PASS: Consistent economic utility

3. ARI ACROSS RESTARTS
   Median: 0.68 ✅
   IQR:    0.15
   Q25/Q75: 0.61 / 0.76
   → PASS: Stable regime identification

4. STATE OCCUPANCY
   Distribution: [0.32, 0.25, 0.21, 0.15, 0.07]
   Entropy: 1.48
   Min Occupancy: 7.2% ✅
   Tiny States: 0 ✅
   → PASS: All states meaningful

5. EXPECTED DURATIONS
   State 0: 5.2 days
   State 1: 8.3 days
   State 2: 18.5 days
   State 3: 12.1 days
   State 4: 3.8 days
   Min: 3.8 days, Max: 18.5 days
   Quality: acceptable ✅
   → PASS: Persistent regimes

6. CRPS & PIT CALIBRATION
   CRPS: 0.0142 ✅
   PIT p-value: 0.23 ✅
   Calibration Score: 0.78
   Flag: well_calibrated ✅
   → PASS: Well-calibrated densities

7. TAIL QUANTILES
   q01: empirical=-0.0421, simulated=-0.0398, diff=5.5% ✅
   q05: empirical=-0.0198, simulated=-0.0205, diff=3.5% ✅
   q95: empirical= 0.0215, simulated= 0.0223, diff=3.7% ✅
   q99: empirical= 0.0489, simulated= 0.0512, diff=4.7% ✅
   Tail Coverage Score: 0.86 ✅
   → PASS: Excellent tail coverage

================================================================================
✅ ALL DIAGNOSTICS PASSED - MODEL READY FOR PRODUCTION
================================================================================
```

---

**Version**: 1.0  
**Date**: 2025-10-28  
**Status**: Production Ready ✅
