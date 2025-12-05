# Meta-Labeling HPO Report

**Generated:** 2025-12-04 23:04:25 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 143
- **Total Trials:** 170
- **Best Edge:** 0.000622
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0004 | 100 |
| Stage 2 (Refinement) | medium | 0.0003 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0006 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "profit_thr_base": 0.016031095146617134,
  "stop_to_profit_ratio": 0.34698289448752156,
  "min_event_spacing": 3,
  "iso_min_prob": 0.13925897541895227,
  "target_clip_high_q": 0.9291265477527278,
  "econ_min_return_multiple": 1.9802786671811643,
  "label_low_q": 0.2870451524372904,
  "label_high_q": 0.8274703843488127,
  "signal_strength_scale_max": 1.5382874808060185,
  "kalman_Q": 0.0004455694748575598,
  "kalman_R": 0.0013063708746753335,
  "vol_baseline_window": 96,
  "profit_mult_min": 0.94925114817309,
  "profit_mult_max": 1.3972413387800793,
  "stop_mult_min": 0.6393964515543834,
  "stop_mult_max": 1.157563958395852
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 3 | 218 | 0.199 | 0.01737 | -0.01066 | 0.000659 | 0.638 |
| 1 | 68 | 0.062 | 0.01701 | -0.00851 | 0.000359 | 0.638 |
| 4 | 48 | 0.044 | 0.01489 | -0.00911 | 0.000256 | 0.638 |
| 0 | 104 | 0.095 | 0.01271 | -0.00861 | 0.000307 | 0.638 |
| 2 | 55 | 0.050 | 0.01546 | -0.00920 | 0.000287 | 0.638 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.3222
- **Profitability:** 73.2152
- **Mean AUC:** 0.6382
- **Sharpe (Winners):** 0.7210

```json
{
  "profit_thr_base": 0.016031095146617134,
  "stop_to_profit_ratio": 0.34698289448752156,
  "min_event_spacing": 3,
  "iso_min_prob": 0.13925897541895227,
  "target_clip_high_q": 0.9291265477527278,
  "econ_min_return_multiple": 1.9802786671811643,
  "label_low_q": 0.2870451524372904,
  "label_high_q": 0.8274703843488127,
  "signal_strength_scale_max": 1.5382874808060185,
  "kalman_Q": 0.0004455694748575598,
  "kalman_R": 0.0013063708746753335,
  "vol_baseline_window": 96,
  "profit_mult_min": 0.94925114817309,
  "profit_mult_max": 1.3972413387800793,
  "stop_mult_min": 0.6393964515543834,
  "stop_mult_max": 1.157563958395852,
  "horizon_bars": 20
}
```

## Underfit Diagnostics (Final Stage)

**⚠️ Underfit Detected:** The model shows signs of underfitting.

Indicators:
- AUC still rising with more data

### Learning Curves (Data Fractions)

| Data Fraction | AUC |
|---------------|-----|
| 20% | 0.5828 |
| 40% | 0.6155 |
| 60% | 0.6760 |
| 80% | 0.4141 |
| 100% | 0.6072 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.6637 |
| 5 | 0.6462 |
| 7 | 0.6308 |

**Feature Importance Concentration (Top 5):** 18.3%

**Feature Group Importance:**

- volatility: 15.3%
- signal: 7.5%
- other: 23.3%

**Top Features by Importance:**

- volatility_1d: 5.64%
- zigzag_swing_magnitude: 3.86%
- ofi_proxy: 3.26%
- hour_cos: 2.87%
- zigzag_swing_slope: 2.67%
- volatility_ratio: 2.47%
- event_r_multiple_mean_last_50: 2.18%
- signal_rsi_distance_50: 2.18%
- absorption_ratio: 1.98%
- regime_2_prob: 1.98%

**Probe vs Deep Model AUC Improvement:** -1.7%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251204_230425.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251204_230425.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251204_230425.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251204_230425.md`

