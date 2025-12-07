# Meta-Labeling HPO Report

**Generated:** 2025-12-06 22:16:20 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 205
- **Total Trials:** 230
- **Best Edge:** 0.006795
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0000 | 100 |
| Stage 2 (Refinement) | medium | 0.0000 | 50 |
| Stage 3 (Production Proxy) | strong | 0.0068 | 35 |
| Stage 4 (Labeling Refinement) | strong | 0.0067 | 45 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.017491123410749924,
  "target_signal_density": 6.80230429363441,
  "min_event_spacing": 3,
  "profit_thr_base": 0.016872560640588317,
  "stop_to_profit_ratio": 0.5045497373428888,
  "trail_distance": 0.9590923708028024,
  "vol_baseline_window": 107,
  "profit_mult_min": 0.9203911150379409,
  "profit_mult_max": 1.2519587516172928,
  "stop_mult_min": 0.5407016602147058,
  "stop_mult_max": 1.0264949862750758,
  "label_low_q": 0.42892281401508403,
  "label_high_q": 0.6769854116734872,
  "econ_min_return_multiple": 1.8698975626241372,
  "iso_min_prob": 0.07679581597356144,
  "target_clip_high_q": 0.9171110460158222,
  "signal_strength_scale_max": 1.4634420148557485,
  "r_multiple_pos_threshold": 0.6401089655453285,
  "transaction_cost_mult": 0.5618835469619745,
  "kalman_Q": 4.620747215934142e-05,
  "kalman_R": 0.0018427417116338776
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 2.0 | 52 | 0.047 | 0.02033 | -0.01015 | 0.002737 | 0.579 |
| 4.0 | 480 | 0.438 | 0.01308 | -0.01019 | 0.005085 | 0.579 |
| 0.0 | 62 | 0.057 | 0.01553 | -0.01037 | 0.002219 | 0.579 |
| 3.0 | 39 | 0.036 | 0.00778 | -0.01027 | 0.000775 | 0.579 |
| 1.0 | 204 | 0.186 | 0.01243 | -0.00986 | 0.003125 | 0.579 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.6101
- **Profitability:** 32.6643
- **Mean AUC:** 0.5827
- **Sharpe (Winners):** 1.0743

```json
{
  "cusum_threshold": 0.01749112317470988,
  "target_signal_density": 6.802361207399702,
  "min_event_spacing": 3,
  "profit_thr_base": 0.01739406479555386,
  "stop_to_profit_ratio": 0.45268629299841484,
  "trail_distance": 0.9591696523907508,
  "vol_baseline_window": 115,
  "profit_mult_min": 0.9203930057498643,
  "profit_mult_max": 1.2519777942547203,
  "stop_mult_min": 0.5407065697063501,
  "stop_mult_max": 1.0265064460141824,
  "label_low_q": 0.42923759853090426,
  "label_high_q": 0.6770869224491995,
  "econ_min_return_multiple": 1.8687350330005292,
  "iso_min_prob": 0.07680311541476327,
  "target_clip_high_q": 0.9171685263158537,
  "signal_strength_scale_max": 1.4634256075188168,
  "r_multiple_pos_threshold": 0.640063082630517,
  "transaction_cost_mult": 0.5613220132479143,
  "kalman_Q": 5.6718231432193384e-05,
  "kalman_R": 0.0019244822401182377,
  "horizon_bars": 22
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.7899 |
| AUC (filtered labels) | 0.3451 |
| AUC inflation (filtered - full) | -0.4448 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.8039 |
| AUC after dropping top-k features | 0.7469 |
| Delta AUC (baseline - dropped) | 0.0569 |
| God feature suspected | False |
| Top features | volatility_1d, trend_base, momentum_ema, mtf_conflict, trend_strength_4H |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | True |
| Brier score | 133001452861977234356996125457645210557381590949389454887935612337326947041280.0000 |
| Expected Calibration Error (ECE) | 0.0000 |
| Maximum Calibration Error (MCE) | 0.0000 |

### Mutual Information (Meta-Score vs Targets)

| Relationship | MI (bits) |
|--------------|-----------|
| Probabilities → Label | nan |
| Probabilities → Return sign | nan |

### Robustness & Class Overlap

| Metric | Value |
|--------|-------|
| Robust across folds | False |
| Worst fold AUC | 0.6258 |
| AUC CV std | 0.0791 |
| Easy problem detected | False |

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.8039 |
| AUC (lag-1 features) | 0.7753 |
| AUC difference (base - lag1) | 0.0286 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.6126 |
| AUC (absolute, best side) | 0.6126 |
| Samples used | 17308 |

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251206_221620.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251206_221620.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251206_221620.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251206_221620.md`

