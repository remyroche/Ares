# Meta-Labeling HPO Report

**Generated:** 2025-11-19 23:45:12 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 5
- **Total Trials:** 170
- **Best Edge:** 0.004183
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | -1000000000.0000 | 100 |
| Stage 2 (Refinement) | medium | -1000000000.0000 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0042 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "profit_thr_base": 0.02440964210271602,
  "stop_to_profit_ratio": 0.35825819423179067,
  "min_event_spacing": 2,
  "iso_min_prob": 0.07617808520142656,
  "target_clip_high_q": 0.9458023690426672,
  "kalman_Q": 2.0298732561683624e-05,
  "kalman_R": 0.09301822900268912,
  "vol_baseline_window": 108,
  "profit_mult_min": 0.8666478257222444,
  "profit_mult_max": 1.3976536730611517,
  "stop_mult_min": 0.7156790268518131,
  "stop_mult_max": 1.694468335372207
}
```

## Underfit Diagnostics (Final Stage)

**✅ No Significant Underfit:** The model appears well-fitted.

### Learning Curves (Data Fractions)

| Data Fraction | AUC |
|---------------|-----|
| 40% | 0.6029 |
| 60% | 0.4267 |
| 80% | 0.5280 |
| 100% | 0.4227 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.6271 |
| 5 | 0.6033 |
| 7 | 0.6033 |

**Feature Importance Concentration (Top 5):** 20.8%

**Probe vs Deep Model AUC Improvement:** -2.4%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251119_234512.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251119_234512.csv`
- **Pareto Frontier CSV:** `N/A`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251119_234512.md`

