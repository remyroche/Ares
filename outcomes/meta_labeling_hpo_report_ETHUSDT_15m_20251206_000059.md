# Meta-Labeling HPO Report

**Generated:** 2025-12-06 00:00:59 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 12
- **Total Trials:** 86
- **Best Edge:** 0.005029
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | -1000000000.0000 | 24 |
| Stage 2 (Refinement) | medium | 0.0024 | 12 |
| Stage 3 (Production Proxy) | strong | 0.0046 | 20 |
| Stage 4 (Labeling Refinement) | strong | 0.0050 | 30 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.017364383171290264,
  "target_signal_density": 6.776630571024179,
  "label_low_q": 0.4994914146049511,
  "label_high_q": 0.6658614128807584,
  "econ_min_return_multiple": 1.4148226555926984,
  "iso_min_prob": 0.057231589113592975,
  "target_clip_high_q": 0.9363320181818054,
  "signal_strength_scale_max": 1.5811071017937053,
  "r_multiple_pos_threshold": 0.33259859860989166,
  "transaction_cost_mult": 0.6546078941275886
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 0 | 21 | 0.069 | 0.24004 | -0.04212 | 0.004068 | 0.544 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** -0.0462
- **Profitability:** 29.5407
- **Mean AUC:** 0.5436
- **Sharpe (Winners):** 0.6453

```json
{
  "cusum_threshold": 0.017364383171290264,
  "target_signal_density": 6.776630571024179,
  "label_low_q": 0.4994914146049511,
  "label_high_q": 0.6658614128807584,
  "econ_min_return_multiple": 1.4148226555926984,
  "iso_min_prob": 0.057231589113592975,
  "target_clip_high_q": 0.9363320181818054,
  "signal_strength_scale_max": 1.5811071017937053,
  "r_multiple_pos_threshold": 0.33259859860989166,
  "transaction_cost_mult": 0.6546078941275886,
  "horizon_bars": 32,
  "min_event_spacing": 4,
  "trail_distance": 0.9605059510856397,
  "profit_mult_min": 0.8776525201283789,
  "profit_mult_max": 1.5703880816224216,
  "stop_mult_min": 0.8966224755894199,
  "stop_mult_max": 1.179670319195723,
  "kalman_Q": 0.00043707294940093546,
  "kalman_R": 0.001612632019194478,
  "vol_baseline_window": 95,
  "profit_thr_base": 0.013047011707588,
  "stop_to_profit_ratio": 0.43294500998898666
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | nan |
| AUC (filtered labels) | 1.0000 |
| AUC inflation (filtered - full) | nan |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.1572 |
| Expected Calibration Error (ECE) | 0.3846 |
| Maximum Calibration Error (MCE) | 0.5000 |

### Mutual Information (Meta-Score vs Targets)

| Relationship | MI (bits) |
|--------------|-----------|
| Probabilities → Label | 0.9710 |
| Probabilities → Return sign | 0.7904 |

### Robustness & Class Overlap

| Metric | Value |
|--------|-------|
| Robust across folds | False |
| Worst fold AUC | nan |
| AUC CV std | nan |
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
| rr_worst_rr | 34 |
| tto_hard | 2 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251206_000059.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251206_000059.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251206_000059.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251206_000059.md`

