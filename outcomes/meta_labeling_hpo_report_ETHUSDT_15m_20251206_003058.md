# Meta-Labeling HPO Report

**Generated:** 2025-12-06 00:30:58 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 163
- **Total Trials:** 200
- **Best Edge:** 0.006996
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0041 | 100 |
| Stage 2 (Refinement) | medium | 0.0070 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0024 | 20 |
| Stage 4 (Labeling Refinement) | strong | 0.0024 | 30 |

## Best Parameters (Highest Edge)

```json
{
  "kalman_Q": 0.00028735543976493875,
  "kalman_R": 0.0012108086139654153,
  "vol_baseline_window": 122,
  "profit_mult_min": 0.9857112000474277,
  "profit_mult_max": 1.569921395697537,
  "stop_mult_min": 0.5222248504722722,
  "stop_mult_max": 1.323448862343921
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 3 | 70 | 0.066 | 0.09234 | -0.04933 | 0.009909 | 0.797 |
| 0 | 40 | 0.037 | 0.07405 | -0.00999 | 0.005967 | 0.797 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** -1.1754
- **Profitability:** 116.5298
- **Mean AUC:** 0.7969
- **Sharpe (Winners):** 0.4010

```json
{
  "kalman_Q": 0.00028735543976493875,
  "kalman_R": 0.0012108086139654153,
  "vol_baseline_window": 122,
  "profit_mult_min": 0.9857112000474277,
  "profit_mult_max": 1.569921395697537,
  "stop_mult_min": 0.5222248504722722,
  "stop_mult_max": 1.323448862343921,
  "horizon_bars": 24,
  "cusum_threshold": 0.017490999264190156,
  "target_signal_density": 6.802857225639665,
  "min_event_spacing": 4,
  "trail_distance": 0.9591950905182219,
  "label_low_q": 0.3155551862459071,
  "label_high_q": 0.6876189197208056,
  "econ_min_return_multiple": 2.2337465809724426,
  "iso_min_prob": 0.10926785048733877,
  "target_clip_high_q": 0.9748654095847509,
  "signal_strength_scale_max": 1.5352501959391898,
  "r_multiple_pos_threshold": 0.7091125296375966,
  "transaction_cost_mult": 0.7998907636516395
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | nan |
| AUC (filtered labels) | 0.9908 |
| AUC inflation (filtered - full) | nan |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.0812 |
| Expected Calibration Error (ECE) | 0.1747 |
| Maximum Calibration Error (MCE) | 0.3636 |

### Mutual Information (Meta-Score vs Targets)

| Relationship | MI (bits) |
|--------------|-----------|
| Probabilities → Label | 0.8571 |
| Probabilities → Return sign | 0.6568 |

### Robustness & Class Overlap

| Metric | Value |
|--------|-------|
| Robust across folds | False |
| Worst fold AUC | 0.3421 |
| AUC CV std | 0.2214 |
| Easy problem detected | True |

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

## Gate Usage & Early-Exit Statistics

| Gate | Count |
|------|-------|
| rr_worst_rr | 25 |
| tto_hard | 12 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251206_003058.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251206_003058.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251206_003058.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251206_003058.md`

