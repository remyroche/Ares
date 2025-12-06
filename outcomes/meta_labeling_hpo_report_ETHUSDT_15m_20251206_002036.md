# Meta-Labeling HPO Report

**Generated:** 2025-12-06 00:20:36 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 155
- **Total Trials:** 200
- **Best Edge:** 0.004714
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0025 | 100 |
| Stage 2 (Refinement) | medium | 0.0029 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0046 | 20 |
| Stage 4 (Labeling Refinement) | strong | 0.0047 | 30 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.017491527538154895,
  "target_signal_density": 6.802225453729713,
  "label_low_q": 0.323191270305093,
  "label_high_q": 0.6890140686802342,
  "econ_min_return_multiple": 2.2840743976115214,
  "iso_min_prob": 0.11503480753276796,
  "target_clip_high_q": 0.979211911578666,
  "signal_strength_scale_max": 1.5733331067034204,
  "r_multiple_pos_threshold": 0.7350716317024795,
  "transaction_cost_mult": 0.782011056536069
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| -1 | 730 | 0.667 | 0.01329 | -0.01122 | 0.003356 | 0.765 |
| 1 | 132 | 0.121 | 0.01366 | -0.01090 | 0.001476 | 0.765 |
| 3 | 718 | 0.656 | 0.01130 | -0.01111 | 0.002724 | 0.765 |
| 0 | 55 | 0.050 | 0.01184 | -0.01126 | 0.000799 | 0.765 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** -1.4400
- **Profitability:** 23.1782
- **Mean AUC:** 0.8115
- **Sharpe (Winners):** 0.6099

```json
{
  "cusum_threshold": 0.01749080237694725,
  "target_signal_density": 6.802857225639665,
  "label_low_q": 0.32319014229958887,
  "label_high_q": 0.6882175048113278,
  "econ_min_return_multiple": 2.284273072893437,
  "iso_min_prob": 0.11501024210875525,
  "target_clip_high_q": 0.979257734780624,
  "signal_strength_scale_max": 1.5762805987230284,
  "r_multiple_pos_threshold": 0.7328061384971507,
  "transaction_cost_mult": 0.7826672065311674,
  "horizon_bars": 26,
  "min_event_spacing": 4,
  "trail_distance": 0.9591950905182219,
  "kalman_Q": 0.0004428133253697396,
  "kalman_R": 0.0011248189434353922,
  "vol_baseline_window": 113,
  "profit_mult_min": 0.9871824601234388,
  "profit_mult_max": 1.6567994392710221,
  "stop_mult_min": 0.5877880144686192,
  "stop_mult_max": 1.3442167703275445
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | nan |
| AUC (filtered labels) | 0.9935 |
| AUC inflation (filtered - full) | nan |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.0833 |
| Expected Calibration Error (ECE) | 0.2120 |
| Maximum Calibration Error (MCE) | 0.3936 |

### Mutual Information (Meta-Score vs Targets)

| Relationship | MI (bits) |
|--------------|-----------|
| Probabilities → Label | 0.8359 |
| Probabilities → Return sign | 0.8146 |

### Robustness & Class Overlap

| Metric | Value |
|--------|-------|
| Robust across folds | False |
| Worst fold AUC | 0.6065 |
| AUC CV std | 0.0910 |
| Easy problem detected | False |

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
| econ_gate | 1 |
| rr_worst_rr | 41 |
| tto_hard | 3 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251206_002036.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251206_002036.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251206_002036.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251206_002036.md`

