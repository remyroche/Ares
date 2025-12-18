# Detailed Changes: fb7d25e09 → 95665d038

**Date range:** Dec 14, 2024 → Dec 16, 2024 (08:14)  
**File:** `src/training/steps/labeling/meta_labeling_hpo_sample_weighted.py`  
**Total diff:** ~9,054 lines

---

## 1. NEW FUNCTIONS ADDED (19 functions)

| Function | Purpose |
|----------|---------|
| `_soft_sharpe_scale` | Soft scaling for Sharpe ratios to prevent extreme values |
| `_layer2_sanity_checks` | Validation checks for Layer2 inputs |
| `_summarize_arr` | Array statistics helper |
| `_align_weights_to_index` | Align sample weights to DataFrame index |
| `_normal_cdf` | Standard normal CDF for PSR calculation |
| `_moment_skew_kurt` | Compute skewness and kurtosis |
| `_psr_from_returns` | **Probabilistic Sharpe Ratio** (De Prado) |
| `_compute_regime_dispersion` | Variance of metrics across regimes |
| `_compute_early_late_gap` | Detect temporal degradation (early vs late fold gap) |
| `_compute_probability_mapping` | Map probability bins to returns |
| `_compute_taken_trade_deciles` | Decile analysis for taken trades |
| `_compute_oof_all_event_deciles` | OOF decile breakdown for all events |
| `_sweep_prob_thresholds_for_profitability` | Sweep thresholds to find optimal cutoff |
| `_weighted_avg_abs_corr` | Expert correlation/diversity metric |
| `_apply_hpo_quality_penalty` | Quality-based utility penalty |
| `_compute_regime_conditional_barrier_geometry` | Regime-aware barrier sizing |
| `_map_param` | Parameter mapping helper |
| `layer2_objective` | Refactored Layer2 HPO objective function |
| `_compute_layer2_metrics` | Compute Layer2 metrics with diagnostics |

---

## 2. UTILITY PARAMETER CHANGES (Critical)

### 2.1 `calculate_hpo_utility` Default Parameters

| Parameter | OLD (fb7d25e09) | NEW (95665d038) | Change |
|-----------|-----------------|-----------------|--------|
| `lambda_vol` | **1.2** | 0.6 | -50% (less fold variance penalty) |
| `w_auc` | **1.0** | 0.5 | -50% (softer AUC gate) |
| `w_den` | **0.5** | 0.15 | -70% (much lower density power) |
| `density_lower` | **0.5** | 0.3 | More lenient |
| `density_sweet_spot` | **(1.5, 5.0)** | (1.0, 6.0) | Widened band |
| `density_upper` | **8.0** | 10.0 | More lenient |
| `clip_max` | **10.0** | 20.0 | Allow larger utility values |

### 2.2 Log Compression REMOVED

**OLD (fb7d25e09):**
```python
base_norm = float(np.sign(base_score) * np.log1p(abs(float(base_score))))
utility = float(np.clip(float(base_norm) * float(modifier), -1.0, 10.0))
```

**NEW (95665d038):**
```python
# Log compression REMOVED - raw linear score used
combined_base = base_score + return_contribution - dd_penalty
utility = float(combined_base) * float(modifier)
```

### 2.3 Density Gate Floor Added

**NEW (95665d038):**
```python
phi_density = 0.2 + 0.8 * phi_density  # Range: [0.2, 1.0] instead of [0.0, 1.0]
```

### 2.4 New Utility Terms Added

| Term | Weight | Formula |
|------|--------|---------|
| `mean_return` | `w_return=3.0` | `return_contribution = mean_return * 100.0 * w_return` |
| `max_drawdown` | `w_dd=1.0` | `dd_penalty = (dd_val - 0.05) * w_dd * 10.0` |

---

## 3. CV/SPLITS CHANGES

### 3.1 Purged Cross-Validation Added

**OLD (fb7d25e09):**
```python
cv = TimeSeriesSplit(n_splits=n_splits)
for train_idx, test_idx in cv.split(X, y_arr):
```

**NEW (95665d038):**
```python
# Try t1-aware purged splits first
splits = _build_t1_aware_purged_splits_for_events(...)

# Fallback to purged_kfold_splits
if splits is None:
    from src.utils.ml_common.labeling.meta_labeling import purged_kfold_splits
    splits = purged_kfold_splits(n_samples, n_splits, embargo)

# Final fallback to TimeSeriesSplit
if splits is None:
    cv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(cv.split(X, y_arr))
```

---

## 4. BACKTEST METRICS CHANGES

### 4.1 Max Drawdown Calculation Fixed

**OLD (fb7d25e09):**
```python
results['max_drawdown'] = float(net_returns.cumsum().min())  # WRONG: cumsum min != drawdown
```

**NEW (95665d038):**
```python
results['max_drawdown'] = float(np.max(dd)) if dd.size else 0.0  # Correct drawdown
```

### 4.2 Sharpe Calculation Changed

**OLD (fb7d25e09):**
```python
# Simple mean/std Sharpe
if sized_arr.size > 1:
    m_val = float(np.mean(sized_arr))
    s_val = float(np.std(sized_arr))
    adj_std = max(s_val, 1e-5)
    raw_sharpe = m_val / adj_std
    fold_sharpes.append(float(np.clip(raw_sharpe, -20.0, 20.0)))
```

**NEW (95665d038):**
```python
# Uses compute_backtest_metrics for annualized Sharpe
bt = compute_backtest_metrics(
    y_prob=size_arr,
    returns=sized_arr,
    threshold=1e-12,
    transaction_cost=0.0,
    direction=direction,
    event_times=fold_event_times,
    returns_are_net=True,
    annualize=True,
    verbose=False,
)
sharpe_val = float(bt.get("sharpe_ratio", np.nan))
```

---

## 5. LAYER 2 CHANGES

### 5.1 Utility Calculation Uses Config

**NEW (95665d038):**
```python
lambda_vol = float(config.get("layer2_lambda_vol", 0.6))
w_auc = float(config.get("layer2_w_auc", 0.5))
w_den = float(config.get("layer2_w_den", 0.15))
```

### 5.2 Profitability Penalty Added

**NEW (95665d038):**
```python
# Penalty for unprofitable configurations
profitability_penalty = ...
utility = utility - profitability_penalty
```

---

## 6. LAYER 3 CHANGES

### 6.1 Configurable Density Gate

**NEW (95665d038):**
```python
l3_den_lower = float(config.get("layer3_density_gate_lower", 0.5))
l3_den_s0 = float(config.get("layer3_density_gate_sweet_spot_min", 1.5))
l3_den_s1 = float(config.get("layer3_density_gate_sweet_spot_max", 5.0))
l3_den_upper = float(config.get("layer3_density_gate_upper", 8.0))
l3_w_den = float(config.get("layer3_w_den", 0.0))
```

---

## 7. CALIBRATION CHANGES

### 7.1 MCE (Maximum Calibration Error) Added

**NEW (95665d038):**
```python
# compute_brier_and_ece now returns 3 values
brier, ece, mce = compute_brier_and_ece(y_true, y_pred)
```

---

## 8. SUMMARY: LIKELY REGRESSION CAUSES

### High Impact Changes (Most Likely Causes)

1. **Utility parameters loosened** (`lambda_vol`, `w_auc`, `w_den` all reduced)
   - Allows HPO to converge to suboptimal configurations
   
2. **Log compression removed**
   - Larger utility swings, less stable optimization
   
3. **Density gate floor added** (`phi_density = 0.2 + 0.8 * phi_density`)
   - Reduces penalty for low-trade configurations
   
4. **Purged CV splits**
   - May reduce effective training data, but this is actually a good change for preventing leakage

### Medium Impact Changes

5. **New utility terms** (`mean_return`, `max_drawdown`)
   - Changes optimization landscape
   
6. **Backtest metrics calculation changed**
   - Annualized Sharpe vs simple Sharpe

### Low Impact Changes

7. **New diagnostic functions** (mostly logging, shouldn't affect optimization)
8. **Config-driven parameters** (good for flexibility)

---

## 9. RECOMMENDED REVERTS

To restore Dec 15 performance, revert these in `calculate_hpo_utility`:

```python
# Revert to stricter values
lambda_vol: float = 1.2      # Was 0.6
w_auc: float = 1.0           # Was 0.5
w_den: float = 0.5           # Was 0.15
density_lower: float = 0.5   # Was 0.3
density_sweet_spot: Tuple[float, float] = (1.5, 5.0)  # Was (1.0, 6.0)
density_upper: float = 8.0   # Was 10.0

# Restore log compression
base_norm = float(np.sign(base_score) * np.log1p(abs(float(base_score))))

# Remove density floor
# phi_density = 0.2 + 0.8 * phi_density  # REMOVE THIS LINE
```
