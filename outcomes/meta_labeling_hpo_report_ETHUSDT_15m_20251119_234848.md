# Meta-Labeling HPO Report

**Generated:** 2025-11-19 23:48:48 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 117
- **Total Trials:** 170
- **Best Edge:** 0.004755
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0009 | 100 |
| Stage 2 (Refinement) | medium | 0.0020 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0048 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "profit_thr_base": 0.014763296294163692,
  "stop_to_profit_ratio": 0.4561604190991283,
  "min_event_spacing": 3,
  "iso_min_prob": 0.09903730209340252,
  "target_clip_high_q": 0.9487305378068354,
  "kalman_Q": 0.00039441640445944323,
  "kalman_R": 0.0033662672819167808,
  "vol_baseline_window": 154,
  "profit_mult_min": 0.99033766991545,
  "profit_mult_max": 1.0123507375439817,
  "stop_mult_min": 0.9436905947136536,
  "stop_mult_max": 1.2522306297505141
}
```

## Underfit Diagnostics (Final Stage)

**✅ No Significant Underfit:** The model appears well-fitted.

### Learning Curves (Data Fractions)

| Data Fraction | AUC |
|---------------|-----|
| 40% | 0.7588 |
| 60% | 0.7769 |
| 80% | 0.5855 |
| 100% | 0.4550 |

### Learning Curves (Model Depths)

| Depth | AUC |
|-------|-----|
| 3 | 0.5747 |
| 5 | 0.5772 |
| 7 | 0.5772 |

**Feature Importance Concentration (Top 5):** 29.2%

**Probe vs Deep Model AUC Improvement:** 0.3%

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251119_234848.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251119_234848.csv`
- **Pareto Frontier CSV:** `N/A`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251119_234848.md`

