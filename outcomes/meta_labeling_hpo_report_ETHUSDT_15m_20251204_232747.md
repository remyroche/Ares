# Meta-Labeling HPO Report

**Generated:** 2025-12-04 23:27:47 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 127
- **Total Trials:** 170
- **Best Edge:** 0.000763
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0008 | 100 |
| Stage 2 (Refinement) | medium | 0.0007 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0006 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "min_event_spacing": 4,
  "profit_mult_min": 0.9492503668726507,
  "profit_mult_max": 1.3972416118085056,
  "stop_mult_min": 0.6296351573759321,
  "stop_mult_max": 1.0019719288545847,
  "kalman_Q": 0.0004673952530772555,
  "kalman_R": 0.00162028797444862,
  "iso_min_prob": 0.13925898901151346,
  "target_clip_high_q": 0.9291265495447268,
  "econ_min_return_multiple": 1.9802930450216765,
  "label_low_q": 0.2870450772895661,
  "label_high_q": 0.8274731112864718,
  "signal_strength_scale_max": 1.5382874728079698
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 3 | 127 | 0.116 | 0.00985 | -0.01234 | 0.000641 | 0.694 |
| 1 | 50 | 0.046 | 0.01188 | -0.00816 | 0.000521 | 0.694 |
| 4 | 34 | 0.031 | 0.01195 | -0.00855 | 0.000434 | 0.694 |
| 2 | 39 | 0.036 | 0.00982 | -0.00878 | 0.000354 | 0.694 |
| 0 | 67 | 0.061 | 0.00973 | -0.00845 | 0.000457 | 0.694 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.6659
- **Profitability:** 38.2194
- **Mean AUC:** 0.6943
- **Sharpe (Winners):** 4.1236

```json
{
  "min_event_spacing": 4,
  "profit_mult_min": 0.9492503668726507,
  "profit_mult_max": 1.3972416118085056,
  "stop_mult_min": 0.6296351573759321,
  "stop_mult_max": 1.0019719288545847,
  "kalman_Q": 0.0004673952530772555,
  "kalman_R": 0.00162028797444862,
  "iso_min_prob": 0.13925898901151346,
  "target_clip_high_q": 0.9291265495447268,
  "econ_min_return_multiple": 1.9802930450216765,
  "label_low_q": 0.2870450772895661,
  "label_high_q": 0.8274731112864718,
  "signal_strength_scale_max": 1.5382874728079698,
  "horizon_bars": 10
}
```

## Underfit Diagnostics (Final Stage)

**⚠️ Underfit Detected:** The model shows signs of underfitting.

Indicators:
- AUC still rising with more data

### Learning Curves (Data Fractions)

| Data Fraction | AUC |
|---------------|-----|
| 20% | 0.5000 |
| 40% | 0.8048 |
| 60% | 0.7436 |
| 80% | 0.5527 |
| 100% | 0.6208 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.5927 |
| 5 | 0.6119 |
| 7 | 0.6115 |

**Feature Importance Concentration (Top 5):** 22.9%

**Feature Group Importance:**

- volatility: 24.2%
- signal: 8.0%
- other: 22.3%

**Top Features by Importance:**

- volatility_1d: 7.60%
- absorption_ratio: 4.59%
- signal_rsi_distance_50: 3.93%
- range_position_x_vol_ratio: 3.80%
- zigzag_bars_since_pivot: 3.01%
- volatility_4h_agg: 2.88%
- hour_cos: 2.62%
- volatility_1h_agg: 2.62%
- vol_ratio: 2.62%
- kalman_trend_x_vol_ratio: 2.49%

**Probe vs Deep Model AUC Improvement:** 1.9%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251204_232747.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251204_232747.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251204_232747.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251204_232747.md`

