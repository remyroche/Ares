# HDP-HMM Validation Quick Reference Guide

## 🎯 7-Category Validation Framework

### I. Predictive/Generalization ✅
**What it measures**: Does the model predict better than baseline?

**Key Metrics**:
- `predictive_ll_effect_size`: Effect size vs noise
- `delta_ll_across_folds`: ΔLL consistency across folds
- `positive_ratio`: % of folds with positive ΔLL

**Interpretation**:
```
✅ GOOD:    positive_ratio > 70%, effect_size > 1.0
⚠️ WARNING: positive_ratio 50-70%, effect_size 0.5-1.0
❌ POOR:    positive_ratio < 50%
```

**Action if poor**:
- Increase model complexity (more states, mixtures)
- Improve feature engineering
- Check for data quality issues

---

### II. Stability & Reproducibility ✅
**What it measures**: Do refits give similar regimes?

**Key Metrics**:
- `refit_stability_ari`: Adjusted Rand Index (median)
- `refit_stability_nmi`: Normalized Mutual Information

**Interpretation**:
```
✅ GOOD:    ARI > 0.6 (stable regime identification)
⚠️ WARNING: ARI 0.4-0.6 (moderate stability)
❌ POOR:    ARI < 0.4 (unstable - crypto is noisy)
```

**Action if poor**:
- Increase `kappa` (stickiness) parameter
- Use more data for fitting
- Try different initialization methods
- Consider simpler model (fewer states)

---

### III. Regime Occupancy & Persistence ✅
**What it measures**: Are regime durations meaningful?

**Key Metrics**:
- `expected_state_durations`: E[D] = 1/(1-p_ii) for each state (in days)
- `duration_quality_flag`: 'good', 'acceptable', 'warning', 'poor'
- `tiny_state_count`: States with <1% occupancy

**Interpretation (for 1h data)**:
```
✅ GOOD:       min_duration ≥ 7 days
✅ ACCEPTABLE: min_duration ≥ 2 days  
⚠️ WARNING:    min_duration ≥ 1 day
❌ POOR:       min_duration < 1 day (likely noise)

⚠️ CAUTION: tiny_state_count > 0 (rare crash states may be acceptable)
```

**Action if poor**:
- Increase `kappa` (stickiness)
- Reduce `alpha` (fewer states)
- Check if timeframe is appropriate (consider 4h or 1d)
- May be capturing noise rather than persistent regimes

---

### IV. Transition Matrix Sensibility ✅
**What it measures**: Are transitions interpretable?

**Key Metrics**:
- `transition_interpretability_score`: 0-1, higher = more interpretable
- `unrealistic_oscillation_detected`: True if states flip too frequently
- `change_rate`: Fraction of samples with state changes

**Interpretation**:
```
✅ GOOD:    interpretability > 0.7, change_rate < 30%
⚠️ WARNING: interpretability 0.5-0.7, change_rate 30-50%
❌ POOR:    unrealistic_oscillation = True (states flipping every bar)
```

**Action if poor**:
- Increase `kappa` dramatically (50-100)
- Increase `dirichlet_concentration` for smoother transitions
- Consider if regimes are too granular (reduce states)

---

### V. Emission/Geometric Diagnostics ✅
**What it measures**: Are states economically distinct?

**Key Metrics**:
- `emission_distinctiveness`: 0-1, average pairwise distance
- `state_conditioned_stats`: Mean, std, skew, kurtosis per state

**Interpretation**:
```
✅ GOOD:    distinctiveness > 0.6 (clearly different states)
⚠️ WARNING: distinctiveness 0.4-0.6 (moderate separation)
❌ POOR:    distinctiveness < 0.4 (overlapping states)
```

**Action if poor**:
- Improve feature engineering (add regime-discriminative features)
- Reduce number of states (merge similar regimes)
- Check if PCA is losing important information

---

### VI. Posterior Predictive Checks ✅
**What it measures**: Does model capture data-generating process?

**Key Metrics**:
- `probability_calibration_score`: 0-1, calibration quality
- `predictive_density_calibration`: 'well_calibrated', 'too_narrow', 'too_wide'

**Interpretation**:
```
✅ GOOD:    calibration_score > 0.7, 'well_calibrated'
⚠️ WARNING: calibration_score 0.5-0.7, 'acceptable'
❌ POOR:    calibration_score < 0.5
           'too_narrow' = underestimating uncertainty
           'too_wide' = overestimating uncertainty
```

**Action if poor**:
- Adjust `covariance_floor` (regularization)
- Check emission model (diag vs full covariance)
- May need more Gibbs sampling iterations

---

### VII. Economic Utility & Robustness ✅
**What it measures**: Do regimes have trading value?

**Key Metrics**:
- `out_of_sample_sharpe`: Sharpe ratio of regime-aware strategy
- `sharpe_uplift_vs_baseline`: Improvement vs buy-and-hold
- `out_of_sample_max_drawdown`: Maximum drawdown %
- `strategy_turnover`: Trading frequency
- `bootstrap_significant`: Statistical significance

**Interpretation**:
```
✅ EXCELLENT: sharpe_uplift > 0.5, significant, survives costs
✅ GOOD:      sharpe_uplift > 0.2, significant
⚠️ MODERATE:  sharpe > 0.5 but low uplift
❌ POOR:      no uplift or negative after transaction costs
```

**Action if poor**:
- Regimes may not capture economically meaningful states
- Try different feature sets (volatility, momentum, sentiment)
- Consider combining with other signals
- May need more sophisticated regime-to-allocation mapping

---

## 🚦 Overall Quality Assessment

### Minimum Requirements for Production

| Category | Minimum Threshold | Critical? |
|----------|------------------|-----------|
| **I. Predictive LL** | effect_size > 0.5 | ⚠️ Important |
| **II. Stability** | ARI > 0.4 | ⚠️ Important |
| **III. Duration** | min_duration > 2 days (1h) | ✅ **Critical** |
| **IV. Transitions** | change_rate < 50% | ✅ **Critical** |
| **V. Distinctiveness** | > 0.4 | ⚠️ Important |
| **VI. Calibration** | > 0.5 | Optional |
| **VII. Economic Utility** | sharpe_uplift > 0.2 | ✅ **Critical** |

### Decision Tree

```
START
  │
  ├─ duration_quality = 'poor'? 
  │  YES → ❌ REJECT (likely noise, not persistent regimes)
  │  NO  → Continue
  │
  ├─ unrealistic_oscillation = True?
  │  YES → ❌ REJECT (states flipping too fast)
  │  NO  → Continue
  │
  ├─ sharpe_uplift < 0?
  │  YES → ❌ REJECT (economically useless)
  │  NO  → Continue
  │
  ├─ refit_stability_ari < 0.3?
  │  YES → ⚠️ CAUTION (unstable, use with care)
  │  NO  → Continue
  │
  ├─ All metrics ≥ "WARNING" thresholds?
  │  YES → ✅ ACCEPT for production
  │  NO  → ⚠️ CAUTION (needs improvement)
  │
END
```

---

## 🔧 Common Issues & Solutions

### Issue 1: Poor Duration Quality (< 2 days)
**Symptoms**:
- `duration_quality_flag = 'poor' or 'warning'`
- `min_expected_duration` < 2 days

**Solutions**:
1. **Increase stickiness**: Set `kappa = 100-200` (vs default 50)
2. **Reduce states**: Set `n_states_max = 6` (vs default 10)
3. **Use longer timeframe**: Switch from 1h → 4h or 1d
4. **Increase dirichlet_concentration**: Set to 5-10 (vs default 1-3)

**Code**:
```python
config = HDPHMMConfig(
    kappa=150.0,         # ✅ High stickiness
    alpha=2.0,           # ✅ Fewer states
    dirichlet_concentration=8.0,  # ✅ Smooth transitions
    max_states=6         # ✅ Limit complexity
)
```

### Issue 2: Unrealistic Oscillation
**Symptoms**:
- `unrealistic_oscillation_detected = True`
- `change_rate > 40%`

**Solutions**:
1. **Dramatically increase kappa**: Try 200-500
2. **Smooth transitions**: Increase `dirichlet_concentration`
3. **Fewer states**: Reduce `n_states`
4. **Check data quality**: May have look-ahead bias or noise

**Code**:
```python
config = HDPHMMConfig(
    kappa=300.0,         # ✅ Very sticky
    dirichlet_concentration=10.0,
    max_states=4,        # ✅ Fewer states
    n_iterations=200     # ✅ Better convergence
)
```

### Issue 3: Low Economic Utility
**Symptoms**:
- `sharpe_uplift_vs_baseline < 0.1`
- `economic_utility_score < 0.3`

**Solutions**:
1. **Better features**: Add volatility, momentum, sentiment
2. **Check regime-to-allocation mapping**: May need smarter allocation
3. **Combine signals**: Use regimes as one input among many
4. **Transaction costs**: May be killing returns (reduce turnover)

**Code**:
```python
# Add more discriminative features
from src.feature_generation import VolatilityFeatures, MomentumFeatures

vol_features = VolatilityFeatures().generate(data)
mom_features = MomentumFeatures().generate(data)
enhanced_data = pd.concat([data, vol_features, mom_features], axis=1)

# Refit model
results = run_hdp_hmm_clustering(
    market_data=enhanced_data,
    # ... other params
)
```

### Issue 4: Unstable Regime Identification (Low ARI)
**Symptoms**:
- `refit_stability_ari < 0.4`
- Different runs give very different regimes

**Solutions**:
1. **More data**: Need > 1000 samples minimum
2. **Better initialization**: Try `initialization='kmeans'` or `'hdbscan'`
3. **Multiple restarts**: Set `n_restarts=5-10`
4. **Simpler model**: Reduce states, use diagonal covariance

**Code**:
```python
# Run with multiple restarts
search_space = HDPHMMSearchSpace(
    n_restarts_min=5,
    n_restarts_max=10,
    initialization_methods=['kmeans', 'hdbscan'],  # Better than random
    emission_cov_types=['diag']  # Simpler model
)
```

---

## 📊 Typical Benchmark Values

### Good Crypto Regime Model (BTCUSDT, 1h)

```python
{
    # I. Predictive
    'predictive_ll_effect_size': 1.2,
    'positive_ratio': 0.75,
    
    # II. Stability
    'refit_stability_ari': 0.65,
    
    # III. Duration
    'duration_quality_flag': 'good',
    'min_expected_duration': 5.2,  # days
    'max_expected_duration': 18.5,  # days
    
    # IV. Transitions
    'transition_interpretability_score': 0.78,
    'change_rate': 0.18,  # 18% of bars change state
    'unrealistic_oscillation': False,
    
    # V. Emissions
    'emission_distinctiveness': 0.72,
    
    # VI. Calibration
    'probability_calibration_score': 0.68,
    'predictive_density_calibration': 'well_calibrated',
    
    # VII. Economic
    'out_of_sample_sharpe': 1.35,
    'sharpe_uplift_vs_baseline': 0.48,
    'out_of_sample_max_drawdown': -0.185,  # -18.5%
    'strategy_turnover': 0.12,  # 12% turnover
    'bootstrap_significant': True
}
```

---

## 🎓 Best Practices

### 1. Start with Comprehensive HPO
```python
# Always use hierarchical optimization first
best_params, _, _ = run_hdp_hmm_auto_tuning(
    market_data=data,
    symbol='BTCUSDT',
    exchange='binance',
    timeframe='1h',
    use_hierarchical=True,  # ✅ 3-5x faster
    tpe_trials=50
)
```

### 2. Review All 7 Categories
```python
# Don't rely on single metrics
metrics = results['quality_metrics']

# Check ALL categories
checks = [
    ('Duration', metrics.get('duration_quality_flag') != 'poor'),
    ('Oscillation', not metrics.get('unrealistic_oscillation_detected', True)),
    ('Economic', metrics.get('sharpe_uplift_vs_baseline', 0) > 0.2),
    ('Stability', metrics.get('refit_stability_ari', 0) > 0.4),
]

print("Validation Results:")
for name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"{status} {name}")
```

### 3. Iterate Based on Results
```python
def tune_based_on_validation(metrics: dict) -> dict:
    """Suggest parameter adjustments based on validation results."""
    adjustments = {}
    
    if metrics.get('duration_quality_flag') in ['poor', 'warning']:
        adjustments['kappa'] = 150.0  # Increase stickiness
        adjustments['alpha'] = 2.0    # Fewer states
    
    if metrics.get('unrealistic_oscillation_detected'):
        adjustments['kappa'] = 300.0  # Very sticky
        adjustments['dirichlet_concentration'] = 10.0
    
    if metrics.get('refit_stability_ari', 1.0) < 0.4:
        adjustments['n_restarts'] = 10
        adjustments['initialization'] = 'kmeans'
    
    return adjustments

# Use it
adjustments = tune_based_on_validation(metrics)
if adjustments:
    print(f"Suggested adjustments: {adjustments}")
    # Refit with adjusted parameters
```

### 4. Document Your Regime Model
```python
# Save comprehensive validation report
validation_report = {
    'timestamp': datetime.now().isoformat(),
    'symbol': 'BTCUSDT',
    'timeframe': '1h',
    'parameters': best_params,
    'validation_results': {
        'predictive': {
            'effect_size': metrics['predictive_ll_effect_size'],
            'positive_ratio': metrics['positive_ratio']
        },
        'stability': {
            'ari': metrics['refit_stability_ari']
        },
        'duration': {
            'quality': metrics['duration_quality_flag'],
            'min_days': metrics['min_expected_duration']
        },
        'economic': {
            'sharpe': metrics['out_of_sample_sharpe'],
            'uplift': metrics['sharpe_uplift_vs_baseline']
        }
    },
    'production_ready': all([
        metrics.get('duration_quality_flag') != 'poor',
        not metrics.get('unrealistic_oscillation_detected'),
        metrics.get('sharpe_uplift_vs_baseline', 0) > 0.2
    ])
}

# Save to artifact manager
artifact_manager.save_artifact(
    data=validation_report,
    artifact_name='regime_model_validation_report',
    artifact_type='metadata'
)
```

---

## 📞 Quick Troubleshooting

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| `duration_quality = 'poor'` | States too short-lived | ↑ `kappa` to 150-300 |
| `unrealistic_oscillation = True` | States flipping | ↑ `kappa` to 300+ |
| `ARI < 0.3` | Unstable fitting | Use `initialization='kmeans'`, ↑ `n_restarts` |
| `sharpe_uplift < 0` | Not economically useful | Better features, check allocation logic |
| `emission_distinctiveness < 0.4` | States not different | ↓ `n_states`, better features |
| `change_rate > 50%` | Too many transitions | ↑ `kappa`, ↑ `dirichlet_concentration` |

---

## 🎯 Production Checklist

Before deploying to production:

- [ ] `duration_quality_flag` ≠ 'poor'
- [ ] `unrealistic_oscillation_detected` = False
- [ ] `sharpe_uplift_vs_baseline` > 0.2
- [ ] `bootstrap_significant` = True
- [ ] `refit_stability_ari` > 0.4
- [ ] `min_expected_duration` > 2 days (for 1h data)
- [ ] Transaction costs properly accounted
- [ ] Out-of-sample testing completed
- [ ] Multiple refits show consistent results
- [ ] Validation report documented

---

**Version**: 1.0  
**Date**: 2025-10-28  
**Status**: Production Ready ✅
