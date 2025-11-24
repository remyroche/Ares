# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 18.4% (labeled / total samples)
- Label positive rate: 50.0%
- Label economic SNR (post-filter, label=1): 1.577
- Label effect size (post-filter Cohen's d): 3.409
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

- Learnability mean CV AUC: 0.7763
- Learnability score (AUC - 0.5 * std): 0.7385
- Label balance (entropy score): 1.0000
- Combined label-quality score: 0.8170

- Probe model mean AUC: 0.7728
- Probe model stability score: 0.9110
- Probe model mean Brier score: 0.1674
- Probe global AUC (all folds combined): 0.7857
- Probe pseudo-R^2 (y vs predicted prob): 0.3306
- Probe permutation p-value (AUC): 0.005
- Probe vs baseline ΔAUC: 0.2986, ΔBrier (baseline - probe): 0.0860, ΔAP: 0.3136

- Label-quality summary score: 0.912 (Rating: Great)
- Learnability summary score: 0.817 (Rating: Great)
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
- Labeled samples: 31972 (coverage=18.4%)
- Positive labels: 15986 (50.0%)
- Negative labels: 15986

## Retention
- Pre-filter events (realized_return not NaN): 56502
- Pre-filter pos/neg (raw econ > cost): 15307 / 41195
- Post-filter labeled events: 31972
- Post-filter pos/neg (binary_label): 15986 / 15986
- Total retention: 56.6%
- Positive retention: 104.4%
- Negative retention: 38.8%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.63% / -0.76%
- Post-filter mean return (label=1/0): 1.48% / -0.80%
- Pre-filter Cohen's d: 6.048
- Post-filter Cohen's d: 3.409
- Pre-filter SNR (label=1): 2.419
- Post-filter SNR (label=1): 1.577

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 4.7%
- Transaction cost (approx per event): 0.150%
- Unconditional mean event return: 0.34%
- Mean return (label=1) minus cost: 1.33%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=1599, win_rate=100.0%, mean_exp_ret=0.65%, Sharpe_exp=1.15
- Top 10%: n=3198, win_rate=100.0%, mean_exp_ret=0.64%, Sharpe_exp=1.13
- Top 20%: n=6395, win_rate=100.0%, mean_exp_ret=0.59%, Sharpe_exp=1.03
- Top 30%: n=9592, win_rate=88.5%, mean_exp_ret=0.49%, Sharpe_exp=0.90
- Top 40%: n=12789, win_rate=75.6%, mean_exp_ret=0.37%, Sharpe_exp=0.71

## Volatility Buckets (by volatility_1d)
- Vol low: n=10657, pos_rate=26.8%, mean_ret=-0.13%, Sharpe=-0.13
- Vol mid: n=10657, pos_rate=36.2%, mean_ret=0.08%, Sharpe=0.06
- Vol high: n=10658, pos_rate=87.1%, mean_ret=1.07%, Sharpe=0.83

## Interpretation Hints
- Coverage (18.4%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=3.409): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 1.577 → High SNR: positive-label returns are well separated from noise.
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
- Valid labeled samples: 31972
- Positive label rate: 50.0%

## Learnability
- Mean CV AUC: 0.7763
- Learnability score (AUC - 0.5 * std): 0.7385

## Entropy / Balance
- Balance score: 1.0000

## Combined Label-Quality Objective
- Combined score: 0.8170

## Interpretation Hints
- Learnability (mean AUC=0.7763): Mean CV AUC ≥ 0.70 → strong learnability; labels are easy to learn.
- Balance (entropy score=1.0000): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.8170): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 0.817
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=5332, n_test=5328, AUC=0.7090, Brier=0.1861, AP=0.7342
- Fold 2: n_train=10660, n_test=5328, AUC=0.6724, Brier=0.1907, AP=0.6297
- Fold 3: n_train=15988, n_test=5328, AUC=0.8453, Brier=0.1416, AP=0.9237
- Fold 4: n_train=21316, n_test=5328, AUC=0.8151, Brier=0.1630, AP=0.8859
- Fold 5: n_train=26644, n_test=5328, AUC=0.8223, Brier=0.1554, AP=0.8758

## Summary
- Mean AUC: 0.7728 (std=0.0688)
- Mean Brier: 0.1674 (std=0.0186)
- Mean AP: 0.8099 (std=0.1107)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9110

## Interpretation Hints
- Mean AUC (0.7728): Mean CV AUC ≥ 0.70 → strong predictive power for the probe model.
- Stability score (0.9110): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1674): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.7857
- Pseudo-R^2 (y vs predicted prob): 0.3306
- Pseudo-R^2 95% CI: [0.3211, 0.3404]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 1.4059

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5017
- Shuffled std AUC: 0.0056
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.8186
- Holdout Brier: 0.1584
- Holdout AP: 0.8811
- Holdout train / test: 22380 / 9592

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.7980
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4742 | Probe AUC: 0.7728 | Delta: 0.2986
- Baseline Brier: 0.2534 | Probe Brier: 0.1674 | Delta (baseline - probe): 0.0860
- Baseline AP: 0.4962 | Probe AP: 0.8099 | Delta: 0.3136

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1008
- Residual lag-1 autocorrelation: 0.6357

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.7728 | LogisticRegression: 0.6536
- Comment: Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.6382
- Min rolling AUC: 0.0000
- Max rolling AUC: 1.0000
- AUC at start: 1.0000
- AUC at end: 1.0000
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251122_232148.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 3.5317
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 25: mean=93.2000, std=4.9558
  - Feature 2: mean=57.4000, std=16.4268
  - Feature 3: mean=50.6000, std=9.4361
  - Feature 5: mean=24.6000, std=8.4285
  - Feature 12: mean=20.2000, std=9.5582
  - Feature 26: mean=17.6000, std=7.2829
  - Feature 13: mean=17.2000, std=12.9368
  - Feature 9: mean=13.6000, std=3.8781
  - Feature 14: mean=10.8000, std=3.0594
  - Feature 4: mean=10.4000, std=1.2000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 6503
- N mislabeled candidates (confident but wrong): 0
- Estimated label noise rate: 0.000%
- False negative rate (confident): 0.000%
- False positive rate (confident): 0.000%
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
- Valid samples: 31972

## Model Calibration
- Brier Score: 0.1674
- Expected Calibration Error (ECE): 0.0258
- Max Calibration Error (MCE): 0.1741

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 10668 (10668.00/day)
- **Mean Return/Trade**: 1.0136%
- **PnL/Day**: 10813.1582%
- **Win Rate**: 83.3%
- **Sharpe Ratio**: 81.792
- **Max Drawdown**: -26.14%
- **Final Equity**: 22432442177062489178181814493121661581677035520.0000
- **Max Consecutive Losses**: 22
- **Avg Consecutive Losses**: 4.76
- **Win-Rate Stability**: 0.831

### Threshold 0.60
- **Trades**: 9180 (9180.00/day)
- **Mean Return/Trade**: 1.1540%
- **PnL/Day**: 10593.4662%
- **Win Rate**: 90.6%
- **Sharpe Ratio**: 90.361
- **Max Drawdown**: -26.14%
- **Final Equity**: 2818107061893985696258161412142617482890313728.0000
- **Max Consecutive Losses**: 16
- **Avg Consecutive Losses**: 3.69
- **Win-Rate Stability**: 0.888

### Threshold 0.65
- **Trades**: 8302 (8302.00/day)
- **Mean Return/Trade**: 1.2288%
- **PnL/Day**: 10201.7868%
- **Win Rate**: 95.0%
- **Sharpe Ratio**: 94.575
- **Max Drawdown**: -26.14%
- **Final Equity**: 61311047164197078602288147106848091379073024.0000
- **Max Consecutive Losses**: 14
- **Avg Consecutive Losses**: 3.04
- **Win-Rate Stability**: 0.923

### Threshold 0.70
- **Trades**: 7840 (7840.00/day)
- **Mean Return/Trade**: 1.2691%
- **PnL/Day**: 9949.5834%
- **Win Rate**: 97.4%
- **Sharpe Ratio**: 96.887
- **Max Drawdown**: -26.14%
- **Final Equity**: 5172670206240348987974454284360168318697472.0000
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 2.61
- **Win-Rate Stability**: 0.951

### Threshold 0.75
- **Trades**: 7518 (7518.00/day)
- **Mean Return/Trade**: 1.2972%
- **PnL/Day**: 9752.5456%
- **Win Rate**: 99.1%
- **Sharpe Ratio**: 98.329
- **Max Drawdown**: -26.14%
- **Final Equity**: 745929359275024135625913036557615282257920.0000
- **Max Consecutive Losses**: 7
- **Avg Consecutive Losses**: 2.13
- **Win-Rate Stability**: 0.978

### Threshold 0.80
- **Trades**: 7266 (7266.00/day)
- **Mean Return/Trade**: 1.2980%
- **PnL/Day**: 9431.2911%
- **Win Rate**: 99.7%
- **Sharpe Ratio**: 96.478
- **Max Drawdown**: -26.14%
- **Final Equity**: 31069225538633575275715098042844359688192.0000
- **Max Consecutive Losses**: 2
- **Avg Consecutive Losses**: 1.73
- **Win-Rate Stability**: 0.991

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 10668 | 10668.00 | 1.014% | 10813.158% | 83.3% | 81.79 | -26.1% | 22 |
| 0.60 | 9180 | 9180.00 | 1.154% | 10593.466% | 90.6% | 90.36 | -26.1% | 16 |
| 0.65 | 8302 | 8302.00 | 1.229% | 10201.787% | 95.0% | 94.57 | -26.1% | 14 |
| 0.70 | 7840 | 7840.00 | 1.269% | 9949.583% | 97.4% | 96.89 | -26.1% | 10 |
| 0.75 | 7518 | 7518.00 | 1.297% | 9752.546% | 99.1% | 98.33 | -26.1% | 7 |
| 0.80 | 7266 | 7266.00 | 1.298% | 9431.291% | 99.7% | 96.48 | -26.1% | 2 |

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