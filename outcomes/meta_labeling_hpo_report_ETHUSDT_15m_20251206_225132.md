# Meta-Labeling HPO Report

**Generated:** 2025-12-06 22:51:32 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 205
- **Total Trials:** 230
- **Best Edge:** 0.006320
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0000 | 100 |
| Stage 2 (Refinement) | medium | 0.0000 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0062 | 35 |
| Stage 4 (Labeling Refinement) | strong | 0.0063 | 45 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.01748731623431877,
  "target_signal_density": 6.8027685063867835,
  "label_low_q": 0.4298299261262667,
  "label_high_q": 0.6896596148645544,
  "econ_min_return_multiple": 1.873927330264368,
  "iso_min_prob": 0.0768056977274672,
  "target_clip_high_q": 0.9171555293908195,
  "signal_strength_scale_max": 1.4634566200816788,
  "r_multiple_pos_threshold": 0.6400793591841953,
  "transaction_cost_mult": 0.5603509967638031
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 2.0 | 48 | 0.044 | 0.02059 | -0.01015 | 0.002409 | 0.568 |
| 4.0 | 455 | 0.416 | 0.01362 | -0.01011 | 0.004683 | 0.568 |
| 0.0 | 59 | 0.054 | 0.01516 | -0.01004 | 0.001904 | 0.568 |
| 1.0 | 200 | 0.183 | 0.01216 | -0.00986 | 0.002724 | 0.568 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.5166
- **Profitability:** 45.4182
- **Mean AUC:** 0.5680
- **Sharpe (Winners):** 1.4996

```json
{
  "cusum_threshold": 0.01748731623431877,
  "target_signal_density": 6.8027685063867835,
  "label_low_q": 0.4298299261262667,
  "label_high_q": 0.6896596148645544,
  "econ_min_return_multiple": 1.873927330264368,
  "iso_min_prob": 0.0768056977274672,
  "target_clip_high_q": 0.9171555293908195,
  "signal_strength_scale_max": 1.4634566200816788,
  "r_multiple_pos_threshold": 0.6400793591841953,
  "transaction_cost_mult": 0.5603509967638031,
  "horizon_bars": 20,
  "min_event_spacing": 3,
  "trail_distance": 0.9592697638179813,
  "profit_mult_min": 0.9204109714708087,
  "profit_mult_max": 1.2511821491658754,
  "stop_mult_min": 0.5408535256602093,
  "stop_mult_max": 1.026699750678352,
  "kalman_Q": 0.00012167044063870076,
  "kalman_R": 0.0020441417287752,
  "vol_baseline_window": 97,
  "profit_thr_base": 0.017103603074793458,
  "stop_to_profit_ratio": 0.49576713759727403
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.7430 |
| AUC (filtered labels) | 0.3915 |
| AUC inflation (filtered - full) | -0.3515 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.7736 |
| AUC after dropping top-k features | 0.7243 |
| Delta AUC (baseline - dropped) | 0.0493 |
| God feature suspected | False |
| Top features | volatility_1d, trend_base, kalman_trend_x_vol_ratio, momentum_ema, close_range_50 |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.4193 |
| Expected Calibration Error (ECE) | 0.0000 |
| Maximum Calibration Error (MCE) | 0.0000 |

### Mutual Information (Meta-Score vs Targets)

| Relationship | MI (bits) |
|--------------|-----------|
| Probabilities → Label | 0.0000 |
| Probabilities → Return sign | 0.0000 |

### Robustness & Class Overlap

| Metric | Value |
|--------|-------|
| Robust across folds | False |
| Worst fold AUC | 0.6103 |
| AUC CV std | 0.0773 |
| Easy problem detected | False |

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.7736 |
| AUC (lag-1 features) | 0.7398 |
| AUC difference (base - lag1) | 0.0338 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.5833 |
| AUC (absolute, best side) | 0.5833 |
| Samples used | 14145 |

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

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251206_225132.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251206_225132.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251206_225132.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251206_225132.md`

