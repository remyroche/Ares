# Meta-Labeling HPO Report

**Generated:** 2025-12-03 21:16:55 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 68
- **Total Trials:** 170
- **Best Edge:** 0.000642
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0005 | 100 |
| Stage 2 (Refinement) | medium | 0.0006 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0003 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "kalman_Q": 0.0004428133253697396,
  "kalman_R": 0.0011278100573736665,
  "vol_baseline_window": 65,
  "profit_mult_min": 0.9492511800702292,
  "profit_mult_max": 1.3972326859912196,
  "stop_mult_min": 0.7056787724808266,
  "stop_mult_max": 1.1575633129865195
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 3 | 139 | 0.127 | 0.01417 | -0.00989 | 0.000727 | 0.614 |
| 2 | 37 | 0.034 | 0.00965 | -0.00689 | 0.000241 | 0.614 |
| 1 | 39 | 0.036 | 0.01206 | -0.00704 | 0.000321 | 0.614 |
| 4 | 30 | 0.027 | 0.01133 | -0.00753 | 0.000262 | 0.614 |
| 0 | 68 | 0.062 | 0.00987 | -0.00697 | 0.000336 | 0.614 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.1835
- **Profitability:** 19.2076
- **Mean AUC:** 0.6139
- **Sharpe (Winners):** 0.4404

```json
{
  "kalman_Q": 0.0004428133253697396,
  "kalman_R": 0.0011278100573736665,
  "vol_baseline_window": 65,
  "profit_mult_min": 0.9492511800702292,
  "profit_mult_max": 1.3972326859912196,
  "stop_mult_min": 0.7056787724808266,
  "stop_mult_max": 1.1575633129865195,
  "horizon_bars": 8,
  "min_event_spacing": 4,
  "iso_min_prob": 0.1394574105385234,
  "target_clip_high_q": 0.9291377922364294,
  "econ_min_return_multiple": 2.2521250246939806,
  "label_low_q": 0.2911046606273069,
  "label_high_q": 0.8327081416754429,
  "signal_strength_scale_max": 1.5500859723616665
}
```

## Underfit Diagnostics (Final Stage)

**⚠️ Underfit Detected:** The model shows signs of underfitting.

Indicators:
- AUC still rising with more data

### Learning Curves (Data Fractions)

| Data Fraction | AUC |
|---------------|-----|
| 20% | 0.3559 |
| 40% | 0.6160 |
| 60% | 0.5645 |
| 80% | 0.4469 |
| 100% | 0.5533 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.5754 |
| 5 | 0.5420 |
| 7 | 0.5456 |

**Feature Importance Concentration (Top 5):** 21.1%

**Feature Group Importance:**

- volatility: 37.6%
- signal: 10.6%
- other: 6.8%

**Top Features by Importance:**

- volatility_1d: 4.89%
- vol_of_vol: 4.89%
- volume_spike_ema: 4.53%
- signal_trend_regime_x_macd_hist_abs: 3.94%
- close_range_20: 2.86%
- signal_density_50: 2.74%
- high_dist_x_vol: 2.63%
- volatility_ratio: 2.51%
- volatility_4h_agg: 2.51%
- hour: 2.39%

**Probe vs Deep Model AUC Improvement:** -3.0%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251203_211655.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251203_211655.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251203_211655.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251203_211655.md`

