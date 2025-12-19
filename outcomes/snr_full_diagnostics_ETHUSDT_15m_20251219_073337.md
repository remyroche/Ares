# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 100.0% (labeled / total samples)
- Label positive rate: 0.0%
- Label economic SNR (post-filter, label=1): 0.000
- Label effect size (post-filter Cohen's d): N/A
- Aleatoric uncertainty fraction (|return| < cost): 7.6%

- Learnability mean CV AUC: N/A
- Learnability score (AUC - 0.5 * std): N/A
- Label balance (entropy score): 0.0000
- Combined label-quality score: N/A

- Probe model mean AUC: N/A
- Probe model stability score: 0.0000
- Probe model mean Brier score: N/A
- Probe global AUC (all folds combined): N/A
- Probe pseudo-R^2 (y vs predicted prob): -3.4138
- Probe permutation p-value (AUC): N/A
- Model-level SNR (p_hat pos vs neg): -3.4138

- Label-quality summary score: 0.400 (Rating: Pass)
- Learnability summary score: 1.000 (Rating: Great)
- Model-robustness summary score: 0.000 (Rating: Bad)

## Metric Definitions (brief)
- **Coverage**: share of events that receive a binary label.
- **Positive rate**: fraction of labeled events with label=1.
- **Cohen's d**: standardized difference in mean returns between positive and negative labels.
- **SNR (mean/std)**: mean positive-label return divided by its standard deviation.
- **Learnability AUC**: mean cross-validated ROC AUC from a shallow probe model.
- **Learnability score**: AUC penalized by instability (AUC - 0.5 * std).
- **Entropy balance**: how balanced labels are between 0 and 1; 1.0 is 50/50.
- **Combined score**: weighted average of learnability and balance.
- **Brier score**: mean squared error between predicted probabilities and true labels; lower is better.
- **Stability score**: 1 - std(AUC)/mean(AUC); higher indicates more stable performance across folds.

## Detailed Diagnostics

### Label-Quality
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Total samples: 3484
- Labeled samples: 3484 (coverage=100.0%)
- Positive labels: 0 (0.0%)
- Negative labels: 0

## Retention
- Pre-filter events (realized_return not NaN): 3484
- Pre-filter pos/neg (raw econ > cost): 1233 / 2251
- Post-filter labeled events: 3484
- Post-filter pos/neg (binary_label): 0 / 0
- Total retention: 100.0%
- Positive retention: 0.0%
- Negative retention: 0.0%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 0.85% / -0.96%
- Post-filter mean return (label=1/0): 0.00% / 0.00%
- Pre-filter Cohen's d: 4.016
- Post-filter Cohen's d: nan
- Pre-filter SNR (label=1): 2.274
- Post-filter SNR (label=1): 0.000

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.32%
- Mean return (label=1) minus cost: -0.30%
- Fraction of labeled events with |return| < cost: 7.6%
- Aleatoric uncertainty fraction (|return| < cost): 7.6%

## High-Probability Buckets (by meta_probability, isotonic expected returns)

## Enhanced Volatility Buckets (by volatility_1d)
- Vol low: n=1161, pos_rate=0.0%, mean_ret=-0.28%, Sharpe=-0.42, vol_range=[0.0027, 0.0037]
- Vol mid: n=1161, pos_rate=0.0%, mean_ret=-0.28%, Sharpe=-0.31, vol_range=[0.0027, 0.0037]
- Vol high: n=1162, pos_rate=0.0%, mean_ret=-0.41%, Sharpe=-0.32, vol_range=[0.0027, 0.0037]

## Interpretation Hints
- Coverage (100.0%): High coverage (>20%): many labeled events; check for redundancy or label noise.
- Post-filter effect size (Cohen's d=nan): Effect size not available (insufficient data).
- Post-filter SNR (label=1): 0.000 → Low SNR: positive-label returns are noisy relative to their mean.
- Retention (total=100.0%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.400
- Rating: Pass
- Summary: Mixed label quality; some usable signal but economic separation or coverage may be modest.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 3484
- Positive label rate: 40.3%

## Learnability
- Mean CV AUC: nan
- Learnability score (AUC - 0.5 * std): nan

## Entropy / Balance
- Balance score: 0.0000

## Combined Label-Quality Objective
- Combined score: nan

## Interpretation Hints
- Learnability (mean AUC=nan): Mean CV AUC ≥ 0.70 → strong learnability; labels are easy to learn.
- Balance (entropy score=0.0000): Entropy score < 0.5 → labels are highly imbalanced or dominated by one class.
- Combined score (nan): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 1.000
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=584, n_test=467, AUC=nan, Brier=nan, AP=nan
- Fold 2: n_train=1164, n_test=580, AUC=nan, Brier=nan, AP=nan
- Fold 3: n_train=1744, n_test=580, AUC=nan, Brier=nan, AP=nan
- Fold 4: n_train=2324, n_test=580, AUC=nan, Brier=nan, AP=nan
- Fold 5: n_train=2904, n_test=580, AUC=nan, Brier=nan, AP=nan

## Summary
- Mean AUC: nan (std=nan)
- Mean Brier: nan (std=nan)
- Mean AP: nan (std=nan)
- Stability score (1 - std(AUC)/mean(AUC)): 0.0000

## Interpretation Hints
- Mean AUC (nan): Mean CV AUC ≥ 0.70 → strong predictive power for the probe model.
- Stability score (0.0000): Stability score < 0.8 → performance is quite unstable across time splits.
- Mean Brier (nan): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): N/A
- Pseudo-R^2 (y vs predicted prob): -3.4138
- Pseudo-R^2 95% CI: [-3.6091, -3.2320]
- Permutation p-value for global AUC: N/A
- Model-level SNR (p_hat pos vs neg): N/A

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: N/A
- Shuffled std AUC: N/A
- Shuffled folds: 0

## Strict Forward Holdout
- Holdout AUC: N/A
- Holdout Brier: N/A
- Holdout AP: N/A
- Holdout train / test: 0 / 0

## Single-Feature Leakage Scan
- Max single-feature AUC: N/A
- AUC threshold for suspicion: N/A

## Naive Baseline Comparison (constant probability)
- Baseline AUC: N/A | Probe AUC: N/A | Delta: N/A
- Baseline Brier: N/A | Probe Brier: N/A | Delta (baseline - probe): N/A
- Baseline AP: N/A | Probe AP: N/A | Delta: N/A

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.0000
- Residual lag-1 autocorrelation: 0.5761

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: N/A | LogisticRegression: N/A
- Comment: Not applicable in label_based mode (no probe model training).

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Insufficient data for temporal AUC analysis

## Feature Importance Stability Analysis
- No feature importance data available

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 2787
- N mislabeled candidates (confident but wrong): 2787
- Estimated label noise rate: 100.000%
- False negative rate (confident): 0.000%
- False positive rate (confident): 100.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.000
- Rating: Bad
- Summary: Probe model is weak or unstable across folds.

### Trading-Simulation
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## Overview
- Date range: 286 days
- Valid samples: 2787

## Model Calibration
- Brier Score: nan
- Expected Calibration Error (ECE): nan
- Max Calibration Error (MCE): nan

### Calibration Interpretation
- Brier score not available.
- ECE not available.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 2787 (9.74/day)
- **Mean Return/Trade**: -0.2804%
- **PnL/Day**: -2.7323%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -15.440
- **Max Drawdown**: -99.97%
- **Final Equity**: 0.0004
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

### Threshold 0.60
- **Trades**: 2787 (9.74/day)
- **Mean Return/Trade**: -0.2804%
- **PnL/Day**: -2.7323%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -15.440
- **Max Drawdown**: -99.97%
- **Final Equity**: 0.0004
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

### Threshold 0.65
- **Trades**: 2787 (9.74/day)
- **Mean Return/Trade**: -0.2804%
- **PnL/Day**: -2.7323%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -15.440
- **Max Drawdown**: -99.97%
- **Final Equity**: 0.0004
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

### Threshold 0.70
- **Trades**: 2787 (9.74/day)
- **Mean Return/Trade**: -0.2804%
- **PnL/Day**: -2.7323%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -15.440
- **Max Drawdown**: -99.97%
- **Final Equity**: 0.0004
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

### Threshold 0.75
- **Trades**: 2787 (9.74/day)
- **Mean Return/Trade**: -0.2804%
- **PnL/Day**: -2.7323%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -15.440
- **Max Drawdown**: -99.97%
- **Final Equity**: 0.0004
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

### Threshold 0.80
- **Trades**: 2787 (9.74/day)
- **Mean Return/Trade**: -0.2804%
- **PnL/Day**: -2.7323%
- **Win Rate**: 0.0%
- **Sharpe Ratio**: -15.440
- **Max Drawdown**: -99.97%
- **Final Equity**: 0.0004
- **Max Consecutive Losses**: 0
- **Avg Consecutive Losses**: 0.00
- **Win-Rate Stability**: 1.000

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 2787 | 9.74 | -0.280% | -2.732% | 0.0% | -15.44 | -100.0% | 0 |
| 0.60 | 2787 | 9.74 | -0.280% | -2.732% | 0.0% | -15.44 | -100.0% | 0 |
| 0.65 | 2787 | 9.74 | -0.280% | -2.732% | 0.0% | -15.44 | -100.0% | 0 |
| 0.70 | 2787 | 9.74 | -0.280% | -2.732% | 0.0% | -15.44 | -100.0% | 0 |
| 0.75 | 2787 | 9.74 | -0.280% | -2.732% | 0.0% | -15.44 | -100.0% | 0 |
| 0.80 | 2787 | 9.74 | -0.280% | -2.732% | 0.0% | -15.44 | -100.0% | 0 |

## Label Quality, Learnability and Robustness Reference

### Label quality
1. Noise Ceiling (if multiple labelers / repeated labels). If you have multiple labelers, this can be combined with inter-rater reliability metrics (ICC, Cohen09s kappa).
> 0.6 b Labels are internally consistent; high R00 is achievable.
0.40.6 b Labels moderately noisy; realistic ceilings are limited.
< 0.4 b Labels are extremely noisy; even perfect models cannot perform well.

2. Aleatoric Uncertainty Fraction. Could link it to expected max R00; i.e., intrinsic unpredictability sets a ceiling for achievable performance
< 40% b Most error is model/feature-driven; improvement is possible.
4060% b Mixed noise and model limitations.
> 60% b Most unpredictability is intrinsic to the target.

### Label learnability vs noise
1. R00. Low R00 could be due to missing features or poor model choice, not just label noise
R00 > 0.40 b The target has a strong predictable signal; meaningful modeling gains are possible.
0.10 < R00 0.40 b The target has a weakbmoderate signal; features matter more than model choice.
R00 0.10 b The target is barely predictable; noise likely dominates.

2. SNR
SNR > 1 b Signal is stronger than noise; the target is learnable.
0.3 < SNR 1 b Weak but real signal exists; more features or nonlinear models may help.
SNR 0.3 b Noise overwhelms signal; predictability is fundamentally low.

3. Permutation p-value. If p is high, it may indicate noisy labels, but it could also reflect poor features or an underpowered model.
p < 0.01 b The model captures a real, statistically robust pattern.
0.01 c p 0.20 b There might be signal, but itb s weak or unstable.
p > 0.20 b The model performs no better than chance; label likely noisy.

4. Naive Baselines. A very simple predictive model used as a reference point. Establishes a floor for model performance & distinguish real signal from noise:
Model 4 baseline b low predictability, focus on labels or features
Model >> baseline b real signal exists, worth improving features/model (doesn't say we haven't reached the ceiling)

### Model & features robustness
1. Bootstrap R00 Confidence Interval. Helps assess stability and reliability of model performance, helps detect overfitting if the CI is very wide or unstable across bootstraps
CI does NOT include 0 b Performance is reliably above noise level.
CI barely clears 0 (lower bound < 0.05) b Signal is present but fragile.
CI spans below 0 b Model performance might be indistinguishable from noise.

2. Residual Structure. Residual structure tells you what signal your model/features are missing (and if there is a pattern), not directly about label noise.
Residuals look random b The model extracted essentially all available signal.
Residuals show patterns b There is remaining structure the model/features are missing.
Residuals differ strongly across subgroups b Predictability varies by segment (not globally noisy).

3. Residual Autocorrelation. Measures whether residuals are temporally or sequentially correlated (often lag-1 autocorrelation). Even if R00 looks okay, autocorrelated residuals indicate hidden structure your features/model missed.
Lag-1 autocorr < 0.10 b No missing temporal/ordered structure.
0.100.20 b Some time dependence is not modeled.
> 0.20 b Strong sequential structure missing; target not fully explained.

4. Model Family Comparison. Helps diagnose whether your model class is adequate and whether there09s remaining learnable signal
Nonlinear >> linear b There is real nonlinear structure not captured by simple models.
Linear >> nonlinear b Tree model overfitting.
All models perform similarly well b The problem is stable and well-posed.
All models perform similarly poorly b The target has low intrinsic predictability.
Ensembles significantly better b High model uncertainty; more data helps
