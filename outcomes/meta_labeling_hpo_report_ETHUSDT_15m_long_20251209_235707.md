# Meta-Labeling HPO Report

**Generated:** 2025-12-09 23:57:07 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m | **Direction:** long

---

## Summary

- **Total Configurations Evaluated:** 17
- **Total Trials:** 92
- **Best Edge:** 0.004771
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | 0.0000 | 8 |
| Stage 2 (Refinement) | medium | 0.0035 | 4 |
| Stage 3 (Production Proxy) | strong | 0.0047 | 35 |
| Stage 4 (Labeling Refinement) | strong | 0.0048 | 45 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.017492453348824114,
  "target_signal_density": 6.8022341303705005,
  "label_low_q": 0.44420493583078763,
  "label_high_q": 0.7020876082223637,
  "econ_min_return_multiple": 1.3385979757928113,
  "iso_min_prob": 0.05752534173098487,
  "target_clip_high_q": 0.9355911262786637,
  "signal_strength_scale_max": 1.5717407346524526,
  "r_multiple_pos_threshold": 0.37431722443576965,
  "transaction_cost_mult": 0.6643451846287826
}
```

## Per-Regime Metrics (Best Edge Configuration)

| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |
|--------|----------|----------------|----------|----------|------|-----|
| 1 | 47 | 1.567 | 0.01125 | -0.00912 | 0.002459 | 0.535 |
| 4 | 306 | 10.200 | 0.01188 | -0.00877 | 0.005777 | 0.535 |

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.4306
- **Profitability:** -6.5494
- **Mean AUC:** 0.5345
- **Sharpe (Winners):** 1.3532

```json
{
  "cusum_threshold": 0.017492453348824114,
  "target_signal_density": 6.8022341303705005,
  "label_low_q": 0.44420493583078763,
  "label_high_q": 0.7020876082223637,
  "econ_min_return_multiple": 1.3385979757928113,
  "iso_min_prob": 0.05752534173098487,
  "target_clip_high_q": 0.9355911262786637,
  "signal_strength_scale_max": 1.5717407346524526,
  "r_multiple_pos_threshold": 0.37431722443576965,
  "transaction_cost_mult": 0.6643451846287826,
  "horizon_bars": 22,
  "min_event_spacing": 4,
  "trail_distance": 0.9592213077295703,
  "profit_mult_min": 0.9461078546111936,
  "profit_mult_max": 1.2763558928361536,
  "stop_mult_min": 0.6405216228335558,
  "stop_mult_max": 1.1283139120315007,
  "kalman_Q": 0.000284604457066743,
  "kalman_R": 0.0014101532422905977,
  "vol_baseline_window": 107,
  "profit_thr_base": 0.016495107477858855,
  "stop_to_profit_ratio": 0.38166964647275287
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.4409 |
| AUC (filtered labels) | 0.5282 |
| AUC inflation (filtered - full) | 0.0873 |
| Filtering is major contributor | True |
| AUC dominated by large moves | False |
| Precision collapse detected | True |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.9499 |
| AUC after dropping top-k features | 0.9474 |
| Delta AUC (baseline - dropped) | 0.0025 |
| God feature suspected | False |
| Top features | kalman_trend, volatility_1d, range_4h, close_range_10, close_min_50 |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.4041 |
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
| Worst fold AUC | 0.3713 |
| AUC CV std | 0.1292 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.3713 | 73 | 0.3508 | -0.0084 |
| 1 | 0.7372 | 73 | 0.3647 | 0.0064 |
| 2 | 0.4053 | 73 | 0.3166 | -0.0029 |
| 3 | 0.5200 | 73 | 0.2388 | -0.0057 |
| 4 | 0.4640 | 73 | 0.2526 | -0.0041 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.9998 | 145 |
| medium_vol | 1.0000 | 148 |
| high_vol | 0.9988 | 145 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.5000 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.9499 |
| AUC (lag-1 features) | 0.9501 |
| AUC difference (base - lag1) | -0.0002 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.5864 |
| AUC (absolute, best side) | 0.5864 |
| Samples used | 438 |

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
| rr_worst_rr | 2 |

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_long_20251209_235707.json`
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
