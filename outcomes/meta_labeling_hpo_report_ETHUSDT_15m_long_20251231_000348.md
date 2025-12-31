# Meta-Labeling HPO Report

**Generated:** 2025-12-31 00:03:48 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m | **Direction:** long

---

## Summary

- **Total Configurations Evaluated:** 9
- **Total Trials:** 66
- **Best Edge:** 0.002159
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Sample Count Screening) | fast | 0.0022 | 4 |
| Stage 2 (Sample Count Refinement) | medium | -0.0002 | 2 |
| Stage 3 (Edge Optimization) | strong | -0.0002 | 30 |
| Stage 4 (Edge Refinement) | strong | -0.0002 | 30 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.01936350297118406,
  "target_signal_density": 11.50714306409916,
  "min_event_spacing": 3,
  "trail_distance": 0.9591950905182219,
  "profit_mult_min": 0.7345163699169337,
  "profit_mult_max": 1.6090665392794814,
  "stop_mult_min": 0.5666954820929941,
  "stop_mult_max": 1.1202948099826744,
  "kalman_Q": 0.0004673952530772555,
  "kalman_R": 0.00162028797444862,
  "label_low_q": 0.3972528783689,
  "label_high_q": 0.7373861817111813,
  "econ_min_return_multiple": 1.2814473019550627,
  "iso_min_prob": 0.05772895660791727,
  "target_clip_high_q": 0.9355575596777442,
  "signal_strength_scale_max": 1.5782463758043883,
  "r_multiple_pos_threshold": 0.3339654006345121,
  "transaction_cost_mult": 1.0326648896416533
}
```

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.7170
- **Profitability:** 12.7605
- **Mean AUC:** 0.6015
- **Sharpe (Winners):** 0.6737

```json
{
  "cusum_threshold": 0.01936350297118406,
  "target_signal_density": 11.50714306409916,
  "min_event_spacing": 3,
  "trail_distance": 0.9591950905182219,
  "profit_mult_min": 0.7345163699169337,
  "profit_mult_max": 1.6090665392794814,
  "stop_mult_min": 0.5666954820929941,
  "stop_mult_max": 1.1202948099826744,
  "kalman_Q": 0.0004673952530772555,
  "kalman_R": 0.00162028797444862,
  "label_low_q": 0.3972528783689,
  "label_high_q": 0.7373861817111813,
  "econ_min_return_multiple": 1.2814473019550627,
  "iso_min_prob": 0.05772895660791727,
  "target_clip_high_q": 0.9355575596777442,
  "signal_strength_scale_max": 1.5782463758043883,
  "r_multiple_pos_threshold": 0.3339654006345121,
  "transaction_cost_mult": 1.0326648896416533,
  "horizon_bars": 22
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.4526 |
| AUC (filtered labels) | 0.4345 |
| AUC inflation (filtered - full) | -0.0181 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.9881 |
| AUC after dropping top-k features | 0.9842 |
| Delta AUC (baseline - dropped) | 0.0039 |
| God feature suspected | False |
| Top features | touch_count_near_price_w150, trend_duration_w20, cusum_rev_pos, donchian_width_w200, kaufman_efficiency_ratio |

### Raw Metrics (Uncalibrated)

| Metric | Value |
|--------|-------|
| Raw Probability Range | 1.0000 - 0.0000 |
| Raw Brier Score | 0.3679 |
| Degenerate Calibration | False |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.3679 |
| Expected Calibration Error (ECE) | 0.3421 |
| Maximum Calibration Error (MCE) | 0.5984 |

### Mutual Information (Meta-Score vs Targets)

| Relationship | MI (bits) |
|--------------|-----------|
| Probabilities → Label | 0.0378 |
| Probabilities → Return sign | 0.0481 |

### Robustness & Class Overlap

| Metric | Value |
|--------|-------|
| Robust across folds | False |
| Worst fold AUC | 0.3403 |
| AUC CV std | 0.1010 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.5575 | 61 | 0.3231 | N/A |
| 1 | 0.6024 | 61 | 0.2509 | 0.0051 |
| 2 | 0.3403 | 61 | 0.2273 | -0.0119 |
| 3 | 0.5806 | 61 | 0.2170 | -0.0017 |
| 4 | 0.6124 | 60 | 0.2468 | -0.0043 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.2769 | 100 |
| medium_vol | 0.5685 | 104 |
| high_vol | 0.4767 | 100 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.5000 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.9888 |
| AUC (lag-1 features) | 0.9863 |
| AUC difference (base - lag1) | 0.0025 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.7096 |
| AUC (absolute, best side) | 0.7096 |
| Samples used | 304 |

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
| rr_worst_rr | 3 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_long_20251231_000348.json`
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
