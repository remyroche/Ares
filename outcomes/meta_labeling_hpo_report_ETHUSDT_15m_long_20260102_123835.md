# Meta-Labeling HPO Report

**Generated:** 2026-01-02 12:38:35 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m | **Direction:** long

---

## Summary

- **Total Configurations Evaluated:** 22
- **Total Trials:** 78
- **Best Edge:** 0.007075
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Sample Count Screening) | fast | 0.0071 | 12 |
| Stage 2 (Sample Count Refinement) | medium | 0.0012 | 6 |
| Stage 3 (Edge Optimization) | strong | 0.0004 | 30 |
| Stage 4 (Edge Refinement) | strong | 0.0007 | 30 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.01936350297118406,
  "target_signal_density": 11.50714306409916,
  "min_event_spacing": 3,
  "trail_distance": 0.9591950905182219,
  "profit_mult_min": 0.7981417167433419,
  "profit_mult_max": 1.8591374909485978,
  "stop_mult_min": 0.8330451065490129,
  "stop_mult_max": 1.2705811061417018,
  "kalman_Q": 0.0004673952530772555,
  "kalman_R": 0.00162028797444862,
  "label_low_q": 0.1789876778744861,
  "label_high_q": 0.7568479206761065,
  "econ_min_return_multiple": 1.8561820485956244,
  "iso_min_prob": 0.11501024210875525,
  "target_clip_high_q": 0.979257734780624,
  "signal_strength_scale_max": 1.5762805987230284,
  "r_multiple_pos_threshold": 0.7328061384971507,
  "transaction_cost_mult": 1.0565334413062335
}
```

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.7097
- **Profitability:** 83.7030
- **Mean AUC:** 0.6497
- **Sharpe (Winners):** 2.7962

```json
{
  "cusum_threshold": 0.01936350297118406,
  "target_signal_density": 11.50714306409916,
  "min_event_spacing": 3,
  "trail_distance": 0.9591950905182219,
  "profit_mult_min": 0.7981417167433419,
  "profit_mult_max": 1.8591374909485978,
  "stop_mult_min": 0.8330451065490129,
  "stop_mult_max": 1.2705811061417018,
  "kalman_Q": 0.0004673952530772555,
  "kalman_R": 0.00162028797444862,
  "label_low_q": 0.1789876778744861,
  "label_high_q": 0.7568479206761065,
  "econ_min_return_multiple": 1.8561820485956244,
  "iso_min_prob": 0.11501024210875525,
  "target_clip_high_q": 0.979257734780624,
  "signal_strength_scale_max": 1.5762805987230284,
  "r_multiple_pos_threshold": 0.7328061384971507,
  "transaction_cost_mult": 1.0565334413062335,
  "horizon_bars": 22
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.5356 |
| AUC (filtered labels) | 0.3344 |
| AUC inflation (filtered - full) | -0.2012 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.9944 |
| AUC after dropping top-k features | 0.9909 |
| Delta AUC (baseline - dropped) | 0.0035 |
| God feature suspected | False |
| Top features | drawdown_100, vol_ratio_w100_vs_w400, climax_volume_flag_w20, range_per_volume_w10, trend_slope_stability_w150 |

### Raw Metrics (Uncalibrated)

| Metric | Value |
|--------|-------|
| Raw Probability Range | 1.0000 - 0.0000 |
| Raw Brier Score | 0.4150 |
| Degenerate Calibration | False |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.4150 |
| Expected Calibration Error (ECE) | 0.3735 |
| Maximum Calibration Error (MCE) | 0.7688 |

### Mutual Information (Meta-Score vs Targets)

| Relationship | MI (bits) |
|--------------|-----------|
| Probabilities → Label | 0.0700 |
| Probabilities → Return sign | 0.0700 |

### Robustness & Class Overlap

| Metric | Value |
|--------|-------|
| Robust across folds | False |
| Worst fold AUC | 0.5980 |
| AUC CV std | 0.0636 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.7247 | 228 | 0.1665 | 0.0034 |
| 1 | 0.6812 | 227 | 0.1623 | 0.0021 |
| 2 | 0.5980 | 227 | 0.3100 | -0.0061 |
| 3 | 0.6273 | 227 | 0.2642 | -0.0019 |
| 4 | 0.7733 | 227 | 0.1546 | 0.0041 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.4744 | 375 |
| medium_vol | 0.4429 | 386 |
| high_vol | 0.8433 | 375 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.5198 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.9944 |
| AUC (lag-1 features) | 0.9832 |
| AUC difference (base - lag1) | 0.0112 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.7566 |
| AUC (absolute, best side) | 0.7566 |
| Samples used | 1136 |

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
| rr_worst_rr | 7 |
| tto_hard | 3 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_long_20260102_123835.json`
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
