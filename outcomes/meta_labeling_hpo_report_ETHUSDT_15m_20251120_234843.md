# Meta-Labeling HPO Report

**Generated:** 2025-11-20 23:48:43 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 145
- **Total Trials:** 170
- **Best Edge:** 0.006580
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0000 | 100 |
| Stage 2 (Refinement) | medium | 0.0018 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0066 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "profit_thr_base": 0.01921355953313817,
  "stop_to_profit_ratio": 0.34919103055927564,
  "min_event_spacing": 3,
  "iso_min_prob": 0.09707245604739435,
  "target_clip_high_q": 0.9486259797634907,
  "kalman_Q": 1.1055318319745503e-05,
  "kalman_R": 0.027726265918888132,
  "vol_baseline_window": 139,
  "profit_mult_min": 0.950604791341038,
  "profit_mult_max": 1.377520602900033,
  "stop_mult_min": 0.7014254552307111,
  "stop_mult_max": 1.1565129103451839
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 0 | 167 | 4.073 | 0.02891 | -0.02670 | 0.000000 | 0.500 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.1981
- **Profitability:** 12.9935
- **Mean AUC:** 0.6200
- **Sharpe (Winners):** 1.2799

```json
{
  "profit_thr_base": 0.01921355953313817,
  "stop_to_profit_ratio": 0.34919103055927564,
  "min_event_spacing": 3,
  "iso_min_prob": 0.09707245604739435,
  "target_clip_high_q": 0.9486259797634907,
  "kalman_Q": 1.1055318319745503e-05,
  "kalman_R": 0.027726265918888132,
  "vol_baseline_window": 139,
  "profit_mult_min": 0.950604791341038,
  "profit_mult_max": 1.377520602900033,
  "stop_mult_min": 0.7014254552307111,
  "stop_mult_max": 1.1565129103451839,
  "horizon_bars": 32
}
```

## Underfit Diagnostics (Final Stage)

**✅ No Significant Underfit:** The model appears well-fitted.

### Learning Curves (Data Fractions)

| Data Fraction | AUC |
|---------------|-----|
| 40% | 0.6755 |
| 60% | 0.6959 |
| 80% | 0.6361 |
| 100% | 0.4716 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.6220 |
| 5 | 0.6274 |
| 7 | 0.6274 |

**Feature Importance Concentration (Top 5):** 29.1%

**Feature Group Importance:**

- volatility: 33.3%
- signal: 0.0%
- other: 37.8%

**Top Features by Importance:**

- event_mfe_mae_ratio_mean_last_50: 8.63%
- volatility_4h_agg: 5.76%
- event_r_multiple_mean_last_50: 5.40%
- vol_of_vol: 5.40%
- event_mean_return_last_50: 3.96%
- vol_price_corr: 3.60%
- volume_spike_ema: 3.42%
- returns_std_10: 3.42%
- sma_slope: 3.42%
- rsi_raw: 3.24%

**Probe vs Deep Model AUC Improvement:** 0.5%

## Regularization & Scoring

### Realistic P&L Edge Metric

The primary scoring metric is the **Realistic P&L Edge**:

```
Edge = (Mean_Return_Label1 - Transaction_Cost) × max(0, 2×AUC - 1)
```

This metric penalizes 'profitable but unlearnable' strategies more realistically:
- If AUC = 0.5 (random), Edge = 0 regardless of profitability
- If AUC = 1.0 (perfect), you capture full mean return minus cost

### Regularization Checks

All configurations were evaluated with:

1. **Isotonic Calibration:** Probabilities calibrated to align with real expected returns
2. **Temporal Stability:** Rolling window AUC variance penalty
3. **Learnability Threshold:** Mean AUC < 0.7 heavily penalized
4. **Profit/Stop Constraint:** Profit threshold must be ≥ 1.5× stop threshold
5. **Label Balance:** Entropy-based balance scoring
6. **Early Stopping:** Per-trial and global early stopping to prevent overfitting

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251120_234843.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251120_234843.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251120_234843.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251120_234843.md`

