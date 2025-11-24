# Meta-Labeling HPO Report

**Generated:** 2025-11-22 23:56:32 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 33
- **Total Trials:** 56
- **Best Edge:** 0.012382
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0048 | 24 |
| Stage 2 (Refinement) | medium | 0.0032 | 12 |
| Stage 3 (Production Proxy) | strong | 0.0124 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "profit_thr_base": 0.02034341848400196,
  "stop_to_profit_ratio": 0.3351396338196895,
  "min_event_spacing": 3,
  "iso_min_prob": 0.13980260909722764,
  "target_clip_high_q": 0.9290448398603106,
  "econ_min_return_multiple": 2.0406763467093145,
  "label_low_q": 0.33410641647821876,
  "label_high_q": 0.836089329042614,
  "signal_strength_scale_max": 1.4063787724611572,
  "kalman_Q": 2.1304399188638292e-05,
  "kalman_R": 0.0018184821581778178,
  "vol_baseline_window": 101,
  "profit_mult_min": 0.9492470170182604,
  "profit_mult_max": 1.3972425940619129,
  "stop_mult_min": 0.7056723508031656,
  "stop_mult_max": 1.1575634157787376
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| -1.0 | 214 | 0.596 | 0.01972 | -0.00834 | 0.003410 | 0.671 |
| 2.0 | 337 | 0.939 | 0.02238 | -0.00890 | 0.004905 | 0.671 |
| 4.0 | 1419 | 3.953 | 0.02068 | -0.00864 | 0.009247 | 0.671 |
| 0.0 | 614 | 1.710 | 0.01938 | -0.00832 | 0.005668 | 0.671 |
| 3.0 | 691 | 1.925 | 0.01927 | -0.00824 | 0.005977 | 0.671 |
| 1.0 | 1304 | 3.632 | 0.01782 | -0.00778 | 0.007539 | 0.671 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.5246
- **Profitability:** 143.9150
- **Mean AUC:** 0.6714
- **Sharpe (Winners):** 5.8175

```json
{
  "profit_thr_base": 0.02034341848400196,
  "stop_to_profit_ratio": 0.3351396338196895,
  "min_event_spacing": 3,
  "iso_min_prob": 0.13980260909722764,
  "target_clip_high_q": 0.9290448398603106,
  "econ_min_return_multiple": 2.0406763467093145,
  "label_low_q": 0.33410641647821876,
  "label_high_q": 0.836089329042614,
  "signal_strength_scale_max": 1.4063787724611572,
  "kalman_Q": 2.1304399188638292e-05,
  "kalman_R": 0.0018184821581778178,
  "vol_baseline_window": 101,
  "profit_mult_min": 0.9492470170182604,
  "profit_mult_max": 1.3972425940619129,
  "stop_mult_min": 0.7056723508031656,
  "stop_mult_max": 1.1575634157787376,
  "horizon_bars": 32
}
```

## Underfit Diagnostics (Final Stage)

**✅ No Significant Underfit:** The model appears well-fitted.

### Learning Curves (Data Fractions)

| Data Fraction | AUC |
|---------------|-----|
| 20% | 0.4896 |
| 40% | 0.5917 |
| 60% | 0.6554 |
| 80% | 0.6777 |
| 100% | 0.6198 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.6860 |
| 5 | 0.6852 |
| 7 | 0.6838 |

**Feature Importance Concentration (Top 5):** 27.4%

**Feature Group Importance:**

- volatility: 24.9%
- signal: 14.5%
- other: 21.6%

**Top Features by Importance:**

- volatility_1d: 7.83%
- event_mfe_mae_ratio_mean_last_50: 5.44%
- regime_2_prob: 4.94%
- vol_ratio: 4.89%
- event_mean_return_last_50: 4.34%
- kalman_trend_x_vol_ratio: 3.94%
- event_r_multiple_mean_last_50: 3.74%
- volume_spike_ema: 3.64%
- event_tto_mean_last_50: 2.49%
- day_of_week: 2.14%

**Probe vs Deep Model AUC Improvement:** -0.1%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251122_235632.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251122_235632.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251122_235632.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251122_235632.md`

