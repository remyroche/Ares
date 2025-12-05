# Meta-Labeling HPO Report

**Generated:** 2025-12-04 22:24:18 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 143
- **Total Trials:** 170
- **Best Edge:** 0.001402
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0008 | 100 |
| Stage 2 (Refinement) | medium | 0.0007 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0014 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "profit_thr_base": 0.019219142056878084,
  "stop_to_profit_ratio": 0.33697724522256517,
  "min_event_spacing": 3,
  "iso_min_prob": 0.13925893985071913,
  "target_clip_high_q": 0.9291265465297547,
  "econ_min_return_multiple": 1.980198431957296,
  "label_low_q": 0.2870446139453419,
  "label_high_q": 0.8274628226960046,
  "signal_strength_scale_max": 1.5382869160326884,
  "kalman_Q": 0.00043413753441341334,
  "kalman_R": 0.0010061964568163714,
  "vol_baseline_window": 95,
  "profit_mult_min": 0.9492502643182411,
  "profit_mult_max": 1.3972413426308736,
  "stop_mult_min": 0.58123772411216,
  "stop_mult_max": 1.1575633343466678
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 3 | 193 | 0.176 | 0.02026 | -0.01177 | 0.001525 | 0.649 |
| 1 | 63 | 0.058 | 0.01884 | -0.00920 | 0.000800 | 0.649 |
| 0 | 90 | 0.082 | 0.01404 | -0.00942 | 0.000666 | 0.649 |
| 2 | 50 | 0.046 | 0.01726 | -0.00986 | 0.000641 | 0.649 |
| 4 | 45 | 0.041 | 0.01582 | -0.00994 | 0.000547 | 0.649 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.3858
- **Profitability:** 63.8255
- **Mean AUC:** 0.6488
- **Sharpe (Winners):** 0.7685

```json
{
  "profit_thr_base": 0.019219142056878084,
  "stop_to_profit_ratio": 0.33697724522256517,
  "min_event_spacing": 3,
  "iso_min_prob": 0.13925893985071913,
  "target_clip_high_q": 0.9291265465297547,
  "econ_min_return_multiple": 1.980198431957296,
  "label_low_q": 0.2870446139453419,
  "label_high_q": 0.8274628226960046,
  "signal_strength_scale_max": 1.5382869160326884,
  "kalman_Q": 0.00043413753441341334,
  "kalman_R": 0.0010061964568163714,
  "vol_baseline_window": 95,
  "profit_mult_min": 0.9492502643182411,
  "profit_mult_max": 1.3972413426308736,
  "stop_mult_min": 0.58123772411216,
  "stop_mult_max": 1.1575633343466678,
  "horizon_bars": 16
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
| 40% | 0.5882 |
| 60% | 0.6359 |
| 80% | 0.5632 |
| 100% | 0.5953 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.6310 |
| 5 | 0.6252 |
| 7 | 0.6276 |

**Feature Importance Concentration (Top 5):** 22.7%

**Feature Group Importance:**

- volatility: 21.1%
- signal: 4.9%
- other: 24.9%

**Top Features by Importance:**

- volatility_1d: 7.46%
- zigzag_swing_magnitude: 5.64%
- event_mean_return_last_50: 3.30%
- signal_rsi_long_distance_50: 3.21%
- volume_spike_ema: 3.12%
- volume_imbalance: 2.52%
- return_autocorr_lag1_w50: 2.25%
- volume_ratio: 2.25%
- drawdown_100: 2.08%
- momentum_per_vol: 2.08%

**Probe vs Deep Model AUC Improvement:** -0.3%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251204_222418.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251204_222418.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251204_222418.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251204_222418.md`

