# Meta-Labeling HPO Report

**Generated:** 2025-11-20 02:25:37 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 0
- **Total Trials:** 32
- **Best Edge:** -1000000000.000000
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | -1000000000.0000 | 8 |
| Stage 2 (Refinement) | medium | -1000000000.0000 | 4 |
| Stage 3 (Production Proxy) | strong | -1000000000.0000 | 20 |

## Best Parameters (Highest Edge)

```json
{
  "min_event_spacing": 3,
  "profit_mult_min": 0.9253678939994053,
  "profit_mult_max": 1.5342536582942952,
  "stop_mult_min": 0.6869593986055134,
  "stop_mult_max": 1.1204507079388881,
  "kalman_Q": 0.0004673952530772555,
  "kalman_R": 0.00162028797444862,
  "iso_min_prob": 0.09832393534856684,
  "target_clip_high_q": 0.9373533397100642
}
```

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251120_022537.json`
- **Candidate Pool CSV:** `N/A`
- **Pareto Frontier CSV:** `N/A`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251120_022537.md`

