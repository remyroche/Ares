# Meta-Labeling HPO Report

**Generated:** 2025-11-20 23:41:47 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 35
- **Total Trials:** 56
- **Best Edge:** 0.002532
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0000 | 24 |
| Stage 2 (Refinement) | medium | 0.0003 | 12 |
| Stage 3 (Production Proxy) | strong | 0.0025 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "profit_thr_base": 0.01927024080812633,
  "stop_to_profit_ratio": 0.4396437001150763,
  "min_event_spacing": 4,
  "iso_min_prob": 0.0969988993989568,
  "target_clip_high_q": 0.9366403376664056,
  "kalman_Q": 1.3964186061223476e-05,
  "kalman_R": 0.0539300837798634,
  "vol_baseline_window": 146,
  "profit_mult_min": 0.954891756119693,
  "profit_mult_max": 1.4030250753751377,
  "stop_mult_min": 0.7066757478551076,
  "stop_mult_max": 1.1598729943980903
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 0 | 127 | 3.098 | 0.03077 | -0.02814 | 0.000000 | 0.500 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** -0.2763
- **Profitability:** 16.7986
- **Mean AUC:** 0.5433
- **Sharpe (Winners):** 1.1395

```json
{
  "profit_thr_base": 0.01927024080812633,
  "stop_to_profit_ratio": 0.4396437001150763,
  "min_event_spacing": 4,
  "iso_min_prob": 0.0969988993989568,
  "target_clip_high_q": 0.9366403376664056,
  "kalman_Q": 1.3964186061223476e-05,
  "kalman_R": 0.0539300837798634,
  "vol_baseline_window": 146,
  "profit_mult_min": 0.954891756119693,
  "profit_mult_max": 1.4030250753751377,
  "stop_mult_min": 0.7066757478551076,
  "stop_mult_max": 1.1598729943980903,
  "horizon_bars": 32
}
```

## Underfit Diagnostics (Final Stage)

**✅ No Significant Underfit:** The model appears well-fitted.

### Learning Curves (Data Fractions)

| Data Fraction | AUC |
|---------------|-----|
| 40% | 0.5000 |
| 60% | 0.5919 |
| 80% | 0.5810 |
| 100% | 0.4236 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.5123 |
| 5 | 0.5167 |
| 7 | 0.5167 |

**Feature Importance Concentration (Top 5):** 34.5%

**Feature Group Importance:**

- volatility: 42.5%
- signal: 3.4%
- other: 34.0%

**Top Features by Importance:**

- event_r_multiple_mean_last_50: 9.47%
- volatility_4h_agg: 6.80%
- vol_of_vol: 6.55%
- event_win_rate_last_50: 6.07%
- volume_spike_ema: 5.58%
- rsi_kalman: 5.10%
- volume_trend: 4.61%
- high_dist_x_vol: 4.13%
- dist_from_recent_low_10: 3.88%
- volatility_1d: 3.64%

**Probe vs Deep Model AUC Improvement:** 0.4%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251120_234147.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251120_234147.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251120_234147.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251120_234147.md`

