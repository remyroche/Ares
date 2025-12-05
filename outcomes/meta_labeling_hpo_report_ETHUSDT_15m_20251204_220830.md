# Meta-Labeling HPO Report

**Generated:** 2025-12-04 22:08:30 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 139
- **Total Trials:** 170
- **Best Edge:** 0.000755
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0006 | 100 |
| Stage 2 (Refinement) | medium | 0.0006 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0008 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "profit_thr_base": 0.017664145186232827,
  "stop_to_profit_ratio": 0.34996394889350135,
  "min_event_spacing": 4,
  "iso_min_prob": 0.13925888390630609,
  "target_clip_high_q": 0.9291265439260346,
  "econ_min_return_multiple": 1.9808320188075486,
  "label_low_q": 0.28704427044895475,
  "label_high_q": 0.8274434032734639,
  "signal_strength_scale_max": 1.5382900872478982,
  "kalman_Q": 5.5122357455865016e-05,
  "kalman_R": 0.08056329596899621,
  "vol_baseline_window": 108,
  "profit_mult_min": 0.9492517502426182,
  "profit_mult_max": 1.397242061564778,
  "stop_mult_min": 0.5524193234224243,
  "stop_mult_max": 1.1575631885490032
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 3 | 115 | 0.105 | 0.01480 | -0.01305 | 0.000703 | 0.630 |
| 1 | 43 | 0.039 | 0.01830 | -0.00944 | 0.000557 | 0.630 |
| 2 | 34 | 0.031 | 0.01426 | -0.00919 | 0.000365 | 0.630 |
| 4 | 29 | 0.026 | 0.01595 | -0.00937 | 0.000387 | 0.630 |
| 0 | 56 | 0.051 | 0.01126 | -0.00913 | 0.000343 | 0.630 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.2629
- **Profitability:** 26.5239
- **Mean AUC:** 0.6299
- **Sharpe (Winners):** 3.0625

```json
{
  "profit_thr_base": 0.017664145186232827,
  "stop_to_profit_ratio": 0.34996394889350135,
  "min_event_spacing": 4,
  "iso_min_prob": 0.13925888390630609,
  "target_clip_high_q": 0.9291265439260346,
  "econ_min_return_multiple": 1.9808320188075486,
  "label_low_q": 0.28704427044895475,
  "label_high_q": 0.8274434032734639,
  "signal_strength_scale_max": 1.5382900872478982,
  "kalman_Q": 5.5122357455865016e-05,
  "kalman_R": 0.08056329596899621,
  "vol_baseline_window": 108,
  "profit_mult_min": 0.9492517502426182,
  "profit_mult_max": 1.397242061564778,
  "stop_mult_min": 0.5524193234224243,
  "stop_mult_max": 1.1575631885490032,
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
| 40% | 0.7000 |
| 60% | 0.7604 |
| 80% | 0.5485 |
| 100% | 0.6840 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.6094 |
| 5 | 0.5959 |
| 7 | 0.5911 |

**Feature Importance Concentration (Top 5):** 19.8%

**Feature Group Importance:**

- volatility: 18.3%
- signal: 5.2%
- other: 27.0%

**Top Features by Importance:**

- volatility_1d: 6.44%
- ofi_proxy: 4.46%
- kalman_trend_x_vol_ratio: 3.22%
- zigzag_bars_since_pivot: 2.97%
- signal_density_50: 2.73%
- high_dist_x_vol: 2.60%
- signal_rsi_distance_50: 2.48%
- event_mfe_mae_ratio_mean_last_50: 2.48%
- vol_price_corr: 2.48%
- log_ret: 2.35%

**Probe vs Deep Model AUC Improvement:** -1.4%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251204_220830.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251204_220830.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251204_220830.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251204_220830.md`

