# Meta-Labeling HPO Report

**Generated:** 2025-11-20 23:11:49 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 35
- **Total Trials:** 56
- **Best Edge:** 0.001159
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0000 | 24 |
| Stage 2 (Refinement) | medium | 0.0004 | 12 |
| Stage 3 (Production Proxy) | strong | 0.0012 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "profit_thr_base": 0.01927024080812633,
  "stop_to_profit_ratio": 0.4396437001150763,
  "min_event_spacing": 4,
  "iso_min_prob": 0.09714688820262095,
  "target_clip_high_q": 0.9366439735630391,
  "kalman_Q": 1.3964186061223476e-05,
  "kalman_R": 0.0539300837798634,
  "vol_baseline_window": 146,
  "profit_mult_min": 0.9576836737275494,
  "profit_mult_max": 1.4141176043134405,
  "stop_mult_min": 0.7117622040575408,
  "stop_mult_max": 1.1639822678595368
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 0 | 127 | 3.098 | 0.03076 | -0.02816 | 0.000000 | 0.500 |

## Underfit Diagnostics (Final Stage)

**✅ No Significant Underfit:** The model appears well-fitted.

### Learning Curves (Data Fractions)

| Data Fraction | AUC |
|---------------|-----|
| 40% | 0.5000 |
| 60% | 0.5361 |
| 80% | 0.5907 |
| 100% | 0.3362 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.4910 |
| 5 | 0.4910 |
| 7 | 0.4910 |

**Feature Importance Concentration (Top 5):** 29.7%

**Probe vs Deep Model AUC Improvement:** 0.0%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251120_231149.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251120_231149.csv`
- **Pareto Frontier CSV:** `N/A`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251120_231149.md`

