# Meta-Labeling HPO Report

**Generated:** 2025-12-07 12:38:04 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 49
- **Total Trials:** 116
- **Best Edge:** 0.013347
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0000 | 24 |
| Stage 2 (Refinement) | medium | 0.0000 | 12 |
| Stage 3 (Production Proxy) | strong | 0.0124 | 35 |
| Stage 4 (Labeling Refinement) | strong | 0.0133 | 45 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.01748849807067399,
  "target_signal_density": 6.8023885722449045,
  "label_low_q": 0.42874742791359877,
  "label_high_q": 0.6952611666445333,
  "econ_min_return_multiple": 1.313814898636901,
  "iso_min_prob": 0.07680555333531935,
  "target_clip_high_q": 0.9171545777086133,
  "signal_strength_scale_max": 1.4634559380440704,
  "r_multiple_pos_threshold": 0.6400962704948743,
  "transaction_cost_mult": 0.5603218631910842
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 4 | 982 | 6.138 | 0.01527 | -0.00964 | 0.012909 | 0.578 |
| -1 | 104 | 0.650 | 0.01611 | -0.00973 | 0.004461 | 0.578 |
| 1 | 929 | 5.806 | 0.01268 | -0.00930 | 0.010157 | 0.578 |
| 3 | 377 | 2.356 | 0.01313 | -0.00966 | 0.006738 | 0.578 |
| 0 | 345 | 2.156 | 0.01299 | -0.00973 | 0.006367 | 0.578 |
| 2 | 104 | 0.650 | 0.01945 | -0.00967 | 0.005492 | 0.578 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.6064
- **Profitability:** -8.1868
- **Mean AUC:** 0.5777
- **Sharpe (Winners):** 2.4217

```json
{
  "cusum_threshold": 0.01748849807067399,
  "target_signal_density": 6.8023885722449045,
  "label_low_q": 0.42874742791359877,
  "label_high_q": 0.6952611666445333,
  "econ_min_return_multiple": 1.313814898636901,
  "iso_min_prob": 0.07680555333531935,
  "target_clip_high_q": 0.9171545777086133,
  "signal_strength_scale_max": 1.4634559380440704,
  "r_multiple_pos_threshold": 0.6400962704948743,
  "transaction_cost_mult": 0.5603218631910842,
  "horizon_bars": 22,
  "min_event_spacing": 3,
  "trail_distance": 0.9592474217434713,
  "profit_mult_min": 0.9204639290400661,
  "profit_mult_max": 1.2507737428625434,
  "stop_mult_min": 0.5409336007429606,
  "stop_mult_max": 1.0268376251109161,
  "kalman_Q": 0.00016048530193904452,
  "kalman_R": 0.0022316746877859605,
  "vol_baseline_window": 102,
  "profit_thr_base": 0.016400699759792612,
  "stop_to_profit_ratio": 0.4810674616337884
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.6645 |
| AUC (filtered labels) | 0.5208 |
| AUC inflation (filtered - full) | -0.1437 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.8591 |
| AUC after dropping top-k features | 0.8558 |
| Delta AUC (baseline - dropped) | 0.0033 |
| God feature suspected | False |
| Top features | bars_since_last_event, momentum_10_x_regime_high, momentum_5_x_regime_high, kalman_trend_x_vol_ratio, mtf_bearish_count |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.4156 |
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
| Worst fold AUC | 0.5540 |
| AUC CV std | 0.0820 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.5853 | 401 | 0.1352 | -0.0017 |
| 1 | 0.5789 | 401 | 0.0977 | -0.0005 |
| 2 | 0.6813 | 401 | 0.0782 | 0.0029 |
| 3 | 0.5540 | 401 | 0.1025 | -0.0014 |
| 4 | 0.7738 | 401 | 0.0751 | 0.0060 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.9491 | 794 |
| medium_vol | 0.9760 | 818 |
| high_vol | 0.9914 | 794 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.5000 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.8591 |
| AUC (lag-1 features) | 0.8522 |
| AUC difference (base - lag1) | 0.0069 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.6015 |
| AUC (absolute, best side) | 0.6015 |
| Samples used | 2406 |

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
| rr_worst_rr | 6 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251207_123804.json`
- Y-shuffle sanity tests to ensure no trivial leakage
- Robustness diagnostics across CV folds and volatility regimes
- Dummy volatility baseline AUC to benchmark meta-model value-add

## Recommended Next Step: Meta-Gated Backtest

To validate that the meta-labeling AUC and edge translate into tradable performance, run the meta-gated backtest using the meta_gating_config produced by feature_generation_meta_labeling_step. For example:

```bash
python3 src/launcher/ares_launcher.py \
  --step meta_gated_backtest \
  --symbol ETHUSDT --exchange binance --timeframe 15m --direction long --execution-mode full
```

The meta-gated backtest report (meta_gated_backtest_report_*.md) provides event-level P&L, trades-per-day, drawdowns, and cost stress tests for the diagnostic gate implied by the best HPO configuration.
