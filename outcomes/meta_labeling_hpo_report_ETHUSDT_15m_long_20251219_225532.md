# Meta-Labeling HPO Report

**Generated:** 2025-12-19 22:55:32 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m | **Direction:** long

---

## Summary

- **Total Configurations Evaluated:** 22
- **Total Trials:** 78
- **Best Edge:** 0.004098
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Sample Count Screening) | fast | 0.0035 | 12 |
| Stage 2 (Sample Count Refinement) | medium | 0.0041 | 6 |
| Stage 3 (Edge Optimization) | strong | 0.0012 | 30 |
| Stage 4 (Edge Refinement) | strong | 0.0013 | 30 |

## Best Parameters (Highest Edge)

```json
{
  "kalman_Q": 0.0002574582320731333,
  "kalman_R": 0.0010784593132166394,
  "vol_baseline_window": 170,
  "profit_mult_min": 0.7709990577859627,
  "profit_mult_max": 1.7205607938371443,
  "stop_mult_min": 0.652622506853014,
  "stop_mult_max": 1.2109207019122887
}
```

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.8103
- **Profitability:** 84.3717
- **Mean AUC:** 0.6234
- **Sharpe (Winners):** 2.7978

```json
{
  "kalman_Q": 0.0002574582320731333,
  "kalman_R": 0.0010784593132166394,
  "vol_baseline_window": 170,
  "profit_mult_min": 0.7709990577859627,
  "profit_mult_max": 1.7205607938371443,
  "stop_mult_min": 0.652622506853014,
  "stop_mult_max": 1.2109207019122887,
  "horizon_bars": 22,
  "cusum_threshold": 0.01936350297118406,
  "target_signal_density": 11.50714306409916,
  "min_event_spacing": 3,
  "trail_distance": 0.9591950905182219,
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

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.4998 |
| AUC (filtered labels) | 0.3691 |
| AUC inflation (filtered - full) | -0.1307 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.8870 |
| AUC after dropping top-k features | 0.8832 |
| Delta AUC (baseline - dropped) | 0.0039 |
| God feature suspected | False |
| Top features | vol_percentile_1d, return_std_24b, momentum_20_x_regime_medium, dist_from_recent_low_50, momentum_10_x_regime_medium |

### Raw Metrics (Uncalibrated)

| Metric | Value |
|--------|-------|
| Raw Probability Range | 1.0000 - 0.0000 |
| Raw Brier Score | 0.3992 |
| Degenerate Calibration | False |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.3992 |
| Expected Calibration Error (ECE) | 0.3457 |
| Maximum Calibration Error (MCE) | 0.6428 |

### Mutual Information (Meta-Score vs Targets)

| Relationship | MI (bits) |
|--------------|-----------|
| Probabilities → Label | 0.0433 |
| Probabilities → Return sign | 0.0102 |

### Robustness & Class Overlap

| Metric | Value |
|--------|-------|
| Robust across folds | False |
| Worst fold AUC | 0.5740 |
| AUC CV std | 0.0956 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.6023 | 499 | 0.1051 | -0.0011 |
| 1 | 0.6058 | 499 | 0.1298 | -0.0035 |
| 2 | 0.5740 | 499 | 0.1185 | -0.0041 |
| 3 | 0.7725 | 499 | 0.0715 | -0.0031 |
| 4 | 0.8014 | 499 | 0.0782 | -0.0037 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.5249 | 824 |
| medium_vol | 0.5209 | 847 |
| high_vol | 0.6032 | 824 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.4907 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.8870 |
| AUC (lag-1 features) | 0.8836 |
| AUC difference (base - lag1) | 0.0034 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.6026 |
| AUC (absolute, best side) | 0.6026 |
| Samples used | 2495 |

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

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_long_20251219_225532.json`
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
