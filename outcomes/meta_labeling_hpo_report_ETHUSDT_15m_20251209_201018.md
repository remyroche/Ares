# Meta-Labeling HPO Report

**Generated:** 2025-12-09 20:10:18 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 0
- **Total Trials:** 116
- **Best Edge:** -1000000000.000000
- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration

## Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Screening) | fast | -1000000000.0000 | 24 |
| Stage 2 (Refinement) | medium | -1000000000.0000 | 12 |
| Stage 3 (Production Proxy) | strong | -1000000000.0000 | 35 |
| Stage 4 (Labeling Refinement) | strong | -1000000000.0000 | 45 |

## Best Parameters (Highest Edge)

```json
{
  "cusum_threshold": 0.017489123620356543,
  "target_signal_density": 6.931109719287836,
  "min_event_spacing": 4,
  "trail_distance": 0.9947015147631295,
  "profit_mult_min": 0.9421258996347891,
  "profit_mult_max": 1.242327588074971,
  "stop_mult_min": 0.6304281325962303,
  "stop_mult_max": 1.1241732191946008,
  "kalman_Q": 0.000209300327751675,
  "kalman_R": 0.0014910678814530696,
  "econ_min_return_multiple": 1.359614212729186,
  "iso_min_prob": 0.062259503612243014,
  "target_clip_high_q": 0.9357338777516935,
  "signal_strength_scale_max": 1.5713567218025828,
  "r_multiple_pos_threshold": 0.3954350613162536,
  "transaction_cost_mult": 0.6636382664571971
}
```

## Diagnostics Summary (Best Configuration)

### Filtering & AUC Inflation

| Metric | Value |
|--------|-------|
| AUC (full labels) | 0.5113 |
| AUC (filtered labels) | 0.5616 |
| AUC inflation (filtered - full) | 0.0503 |
| Filtering is major contributor | False |
| AUC dominated by large moves | False |
| Precision collapse detected | False |

### Permutation-Importance Leakage Diagnostics

| Metric | Value |
|--------|-------|
| Baseline AUC (probe) | 0.8899 |
| AUC after dropping top-k features | 0.8719 |
| Delta AUC (baseline - dropped) | 0.0180 |
| God feature suspected | False |
| Top features | momentum_10_x_regime_high, volatility_1d, signal_trend_regime_x_macd_hist_abs, kalman_trend_x_vol_ratio, dist_from_recent_high_50 |

### Calibration Diagnostics

| Metric | Value |
|--------|-------|
| Well calibrated | False |
| Brier score | 0.5000 |
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
| Worst fold AUC | 0.5985 |
| AUC CV std | 0.0742 |
| Easy problem detected | False |

#### Per-Fold AUC Summary

| Fold | AUC | n_test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.5985 | 326 | 0.1506 | -0.0006 |
| 1 | 0.6771 | 326 | 0.2023 | 0.0060 |
| 2 | 0.7715 | 326 | 0.0982 | 0.0041 |
| 3 | 0.6234 | 326 | 0.1754 | -0.0018 |
| 4 | 0.7791 | 326 | 0.0986 | 0.0048 |

#### AUC by Volatility Regime

| Regime | AUC | n_events |
|--------|-----|----------|
| low_vol | 0.8542 | 646 |
| medium_vol | 0.8538 | 666 |
| high_vol | 0.9718 | 646 |

### Y-Shuffle Sanity Test

| Metric | Value |
|--------|-------|
| AUC with shuffled labels | 0.5000 |

A well-behaved model should have AUC≈0.5 under label shuffling; any materially higher value would indicate leakage or mis-specification.

### Lag-1 Stress Test (Look-Ahead Bias)

| Metric | Value |
|--------|-------|
| AUC (base, t features) | 0.8899 |
| AUC (lag-1 features) | 0.8876 |
| AUC difference (base - lag1) | 0.0023 |
| Look-ahead suspected | False |

### Dummy-Rule Volatility Baseline

| Metric | Value |
|--------|-------|
| AUC (raw, signed) | 0.6989 |
| AUC (absolute, best side) | 0.6989 |
| Samples used | 1958 |

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
| exception | 49 |
| rr_worst_rr | 6 |

**Last exception (if any):** `min_periods 50 must be <= window 28`

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251209_201018.json`
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
