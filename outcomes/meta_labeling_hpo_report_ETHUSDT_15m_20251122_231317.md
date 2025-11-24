# Meta-Labeling HPO Report

**Generated:** 2025-11-22 23:13:17 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 33
- **Total Trials:** 56
- **Best Edge:** 0.012254
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0049 | 24 |
| Stage 2 (Refinement) | medium | 0.0039 | 12 |
| Stage 3 (Production Proxy) | strong | 0.0123 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "profit_thr_base": 0.02034341848400196,
  "stop_to_profit_ratio": 0.3351396338196895,
  "min_event_spacing": 3,
  "iso_min_prob": 0.14831244251709105,
  "target_clip_high_q": 0.9297529141164123,
  "econ_min_return_multiple": 2.040670741520151,
  "label_low_q": 0.3341055914316779,
  "label_high_q": 0.8360860869871302,
  "signal_strength_scale_max": 1.469987346102821,
  "kalman_Q": 2.1304399188638292e-05,
  "kalman_R": 0.0018184821581778178,
  "vol_baseline_window": 118,
  "profit_mult_min": 0.9492439229314148,
  "profit_mult_max": 1.3972472158509834,
  "stop_mult_min": 0.7056681185144098,
  "stop_mult_max": 1.1575638315574754
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| -1.0 | 215 | 0.599 | 0.01988 | -0.00832 | 0.003393 | 0.669 |
| 2.0 | 337 | 0.939 | 0.02251 | -0.00889 | 0.004856 | 0.669 |
| 4.0 | 1419 | 3.953 | 0.02081 | -0.00863 | 0.009159 | 0.669 |
| 0.0 | 613 | 1.708 | 0.01953 | -0.00830 | 0.005621 | 0.669 |
| 3.0 | 692 | 1.928 | 0.01939 | -0.00819 | 0.005926 | 0.669 |
| 1.0 | 1306 | 3.638 | 0.01785 | -0.00768 | 0.007440 | 0.669 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.5081
- **Profitability:** 131.1764
- **Mean AUC:** 0.6687
- **Sharpe (Winners):** 5.5922

```json
{
  "profit_thr_base": 0.02034341848400196,
  "stop_to_profit_ratio": 0.3351396338196895,
  "min_event_spacing": 3,
  "iso_min_prob": 0.14831244251709105,
  "target_clip_high_q": 0.9297529141164123,
  "econ_min_return_multiple": 2.040670741520151,
  "label_low_q": 0.3341055914316779,
  "label_high_q": 0.8360860869871302,
  "signal_strength_scale_max": 1.469987346102821,
  "kalman_Q": 2.1304399188638292e-05,
  "kalman_R": 0.0018184821581778178,
  "vol_baseline_window": 118,
  "profit_mult_min": 0.9492439229314148,
  "profit_mult_max": 1.3972472158509834,
  "stop_mult_min": 0.7056681185144098,
  "stop_mult_max": 1.1575638315574754,
  "horizon_bars": 32
}
```

## Underfit Diagnostics (Final Stage)

**✅ No Significant Underfit:** The model appears well-fitted.

### Learning Curves (Data Fractions)

| Data Fraction | AUC |
|---------------|-----|
| 20% | 0.4376 |
| 40% | 0.5768 |
| 60% | 0.6622 |
| 80% | 0.6915 |
| 100% | 0.6256 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.6825 |
| 5 | 0.6840 |
| 7 | 0.6798 |

**Feature Importance Concentration (Top 5):** 26.5%

**Feature Group Importance:**

- volatility: 24.3%
- signal: 12.4%
- other: 23.0%

**Top Features by Importance:**

- volatility_1d: 8.17%
- event_mfe_mae_ratio_mean_last_50: 5.32%
- vol_ratio: 4.55%
- volume_spike_ema: 4.26%
- event_mean_return_last_50: 4.16%
- regime_2_prob: 4.11%
- kalman_trend_x_vol_ratio: 4.06%
- regime_1_prob: 2.76%
- event_tto_mean_last_50: 2.66%
- event_r_multiple_mean_last_50: 2.51%

**Probe vs Deep Model AUC Improvement:** 0.2%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251122_231317.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251122_231317.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251122_231317.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251122_231317.md`

