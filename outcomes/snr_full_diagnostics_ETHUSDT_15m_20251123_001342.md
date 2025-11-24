# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 18.4% (labeled / total samples)
- Label positive rate: 50.0%
- Label economic SNR (post-filter, label=1): 1.613
- Label effect size (post-filter Cohen's d): 3.490
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

- Learnability mean CV AUC: 0.7732
- Learnability score (AUC - 0.5 * std): 0.7335
- Label balance (entropy score): 1.0000
- Combined label-quality score: 0.8134

- Probe model mean AUC: 0.7718
- Probe model stability score: 0.9083
- Probe model mean Brier score: 0.1675
- Probe global AUC (all folds combined): 0.7849
- Probe pseudo-R^2 (y vs predicted prob): 0.3302
- Probe permutation p-value (AUC): 0.005
- Probe vs baseline ΔAUC: 0.2981, ΔBrier (baseline - probe): 0.0859, ΔAP: 0.3153

- Label-quality summary score: 0.912 (Rating: Great)
- Learnability summary score: 0.813 (Rating: Great)
- Model-robustness summary score: 1.000 (Rating: Great)

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
- Total samples: 173439
- Labeled samples: 31974 (coverage=18.4%)
- Positive labels: 15987 (50.0%)
- Negative labels: 15987

## Retention
- Pre-filter events (realized_return not NaN): 56502
- Pre-filter pos/neg (raw econ > cost): 15371 / 41131
- Post-filter labeled events: 31974
- Post-filter pos/neg (binary_label): 15987 / 15987
- Total retention: 56.6%
- Positive retention: 104.0%
- Negative retention: 38.9%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.63% / -0.76%
- Post-filter mean return (label=1/0): 1.48% / -0.80%
- Pre-filter Cohen's d: 6.106
- Post-filter Cohen's d: 3.490
- Pre-filter SNR (label=1): 2.443
- Post-filter SNR (label=1): 1.613

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 4.5%
- Transaction cost (approx per event): 0.150%
- Unconditional mean event return: 0.34%
- Mean return (label=1) minus cost: 1.33%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=1599, win_rate=100.0%, mean_exp_ret=0.71%, Sharpe_exp=1.25
- Top 10%: n=3198, win_rate=100.0%, mean_exp_ret=0.68%, Sharpe_exp=1.18
- Top 20%: n=6395, win_rate=100.0%, mean_exp_ret=0.60%, Sharpe_exp=1.03
- Top 30%: n=9592, win_rate=88.1%, mean_exp_ret=0.49%, Sharpe_exp=0.87
- Top 40%: n=12790, win_rate=75.5%, mean_exp_ret=0.36%, Sharpe_exp=0.69

## Volatility Buckets (by volatility_1d)
- Vol low: n=10658, pos_rate=27.1%, mean_ret=-0.13%, Sharpe=-0.13
- Vol mid: n=10658, pos_rate=35.6%, mean_ret=0.07%, Sharpe=0.05
- Vol high: n=10658, pos_rate=87.3%, mean_ret=1.09%, Sharpe=0.85

## Interpretation Hints
- Coverage (18.4%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=3.490): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 1.613 → High SNR: positive-label returns are well separated from noise.
- Retention (total=56.6%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.912
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 31974
- Positive label rate: 50.0%

## Learnability
- Mean CV AUC: 0.7732
- Learnability score (AUC - 0.5 * std): 0.7335

## Entropy / Balance
- Balance score: 1.0000

## Combined Label-Quality Objective
- Combined score: 0.8134

## Interpretation Hints
- Learnability (mean AUC=0.7732): Mean CV AUC ≥ 0.70 → strong learnability; labels are easy to learn.
- Balance (entropy score=1.0000): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.8134): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 0.813
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=5329, n_test=5329, AUC=0.7061, Brier=0.1875, AP=0.7337
- Fold 2: n_train=10658, n_test=5329, AUC=0.6685, Brier=0.1904, AP=0.6315
- Fold 3: n_train=15987, n_test=5329, AUC=0.8460, Brier=0.1415, AP=0.9242
- Fold 4: n_train=21316, n_test=5329, AUC=0.8153, Brier=0.1621, AP=0.8871
- Fold 5: n_train=26645, n_test=5329, AUC=0.8232, Brier=0.1557, AP=0.8762

## Summary
- Mean AUC: 0.7718 (std=0.0707)
- Mean Brier: 0.1675 (std=0.0188)
- Mean AP: 0.8105 (std=0.1105)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9083

## Interpretation Hints
- Mean AUC (0.7718): Mean CV AUC ≥ 0.70 → strong predictive power for the probe model.
- Stability score (0.9083): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1675): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.7849
- Pseudo-R^2 (y vs predicted prob): 0.3302
- Pseudo-R^2 95% CI: [0.3221, 0.3388]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 1.4046

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5016
- Shuffled std AUC: 0.0067
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.8230
- Holdout Brier: 0.1574
- Holdout AP: 0.8836
- Holdout train / test: 22381 / 9593

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.7972
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4738 | Probe AUC: 0.7718 | Delta: 0.2981
- Baseline Brier: 0.2534 | Probe Brier: 0.1675 | Delta (baseline - probe): 0.0859
- Baseline AP: 0.4952 | Probe AP: 0.8105 | Delta: 0.3153

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1166
- Residual lag-1 autocorrelation: 0.6368

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.7718 | LogisticRegression: 0.6517
- Comment: Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.6389
- Min rolling AUC: 0.0000
- Max rolling AUC: 1.0000
- AUC at start: 1.0000
- AUC at end: 0.9531
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251123_001340.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 4.0962
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 25: mean=101.8000, std=16.4365
  - Feature 3: mean=55.6000, std=13.1240
  - Feature 2: mean=52.4000, std=15.4997
  - Feature 5: mean=27.0000, std=9.4021
  - Feature 12: mean=18.4000, std=10.1311
  - Feature 9: mean=14.8000, std=2.9257
  - Feature 13: mean=14.2000, std=8.4475
  - Feature 14: mean=13.2000, std=5.2307
  - Feature 26: mean=11.0000, std=5.8652
  - Feature 4: mean=10.4000, std=2.4166
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 6494
- N mislabeled candidates (confident but wrong): 1
- Estimated label noise rate: 0.015%
- False negative rate (confident): 0.000%
- False positive rate (confident): 0.008%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 1.000
- Rating: Great
- Summary: Strong, stable probe model with consistent performance.

### Trading-Simulation
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1 days
- Valid samples: 31974

## Model Calibration
- Brier Score: 0.1675
- Expected Calibration Error (ECE): 0.0278
- Max Calibration Error (MCE): 0.3250

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 10733 (10733.00/day)
- **Mean Return/Trade**: 1.0101%
- **PnL/Day**: 10841.8142%
- **Win Rate**: 82.8%
- **Sharpe Ratio**: 82.095
- **Max Drawdown**: -25.97%
- **Final Equity**: 29937978755377191958696587085855137447408566272.0000
- **Max Consecutive Losses**: 21
- **Avg Consecutive Losses**: 4.78
- **Win-Rate Stability**: 0.826

### Threshold 0.60
- **Trades**: 9204 (9204.00/day)
- **Mean Return/Trade**: 1.1563%
- **PnL/Day**: 10643.0035%
- **Win Rate**: 90.1%
- **Sharpe Ratio**: 91.263
- **Max Drawdown**: -22.71%
- **Final Equity**: 4638236635893756133312625821161923066969718784.0000
- **Max Consecutive Losses**: 25
- **Avg Consecutive Losses**: 3.90
- **Win-Rate Stability**: 0.872

### Threshold 0.65
- **Trades**: 8301 (8301.00/day)
- **Mean Return/Trade**: 1.2495%
- **PnL/Day**: 10371.8081%
- **Win Rate**: 95.0%
- **Sharpe Ratio**: 97.679
- **Max Drawdown**: -22.71%
- **Final Equity**: 334632300207246927368161064643012619650727936.0000
- **Max Consecutive Losses**: 15
- **Avg Consecutive Losses**: 3.52
- **Win-Rate Stability**: 0.920

### Threshold 0.70
- **Trades**: 7806 (7806.00/day)
- **Mean Return/Trade**: 1.2982%
- **PnL/Day**: 10133.6119%
- **Win Rate**: 97.9%
- **Sharpe Ratio**: 101.333
- **Max Drawdown**: -22.71%
- **Final Equity**: 32604155612525662260637394069305863117471744.0000
- **Max Consecutive Losses**: 11
- **Avg Consecutive Losses**: 2.59
- **Win-Rate Stability**: 0.955

### Threshold 0.75
- **Trades**: 7494 (7494.00/day)
- **Mean Return/Trade**: 1.3177%
- **PnL/Day**: 9874.7198%
- **Win Rate**: 99.3%
- **Sharpe Ratio**: 101.959
- **Max Drawdown**: -22.71%
- **Final Equity**: 2542901407252045674542681447071352285560832.0000
- **Max Consecutive Losses**: 5
- **Avg Consecutive Losses**: 2.04
- **Win-Rate Stability**: 0.980

### Threshold 0.80
- **Trades**: 7278 (7278.00/day)
- **Mean Return/Trade**: 1.3198%
- **PnL/Day**: 9605.7829%
- **Win Rate**: 99.8%
- **Sharpe Ratio**: 100.378
- **Max Drawdown**: -22.71%
- **Final Equity**: 177536270149402265383287034438954954784768.0000
- **Max Consecutive Losses**: 4
- **Avg Consecutive Losses**: 2.00
- **Win-Rate Stability**: 0.989

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 10733 | 10733.00 | 1.010% | 10841.814% | 82.8% | 82.09 | -26.0% | 21 |
| 0.60 | 9204 | 9204.00 | 1.156% | 10643.003% | 90.1% | 91.26 | -22.7% | 25 |
| 0.65 | 8301 | 8301.00 | 1.249% | 10371.808% | 95.0% | 97.68 | -22.7% | 15 |
| 0.70 | 7806 | 7806.00 | 1.298% | 10133.612% | 97.9% | 101.33 | -22.7% | 11 |
| 0.75 | 7494 | 7494.00 | 1.318% | 9874.720% | 99.3% | 101.96 | -22.7% | 5 |
| 0.80 | 7278 | 7278.00 | 1.320% | 9605.783% | 99.8% | 100.38 | -22.7% | 4 |

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