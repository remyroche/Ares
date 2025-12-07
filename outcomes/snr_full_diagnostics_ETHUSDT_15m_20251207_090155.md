# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 15.1% (labeled / total samples)
- Label positive rate: 10.2%
- Label economic SNR (post-filter, label=1): 9.398
- Label effect size (post-filter Cohen's d): 2.801
- Aleatoric uncertainty fraction (|return| < cost): 14.2%

- Learnability mean CV AUC: 0.6224
- Learnability score (AUC - 0.5 * std): 0.6165
- Label balance (entropy score): 0.0000
- Combined label-quality score: 0.4316

- Probe model mean AUC: 0.6520
- Probe model stability score: 0.9095
- Probe model mean Brier score: 0.0860
- Probe global AUC (all folds combined): 0.6647
- Probe pseudo-R^2 (y vs predicted prob): 0.0270
- Probe permutation p-value (AUC): 0.005
- Probe vs baseline ΔAUC: 0.1963, ΔBrier (baseline - probe): 0.0029, ΔAP: 0.0811

- Label-quality summary score: 0.878 (Rating: Great)
- Learnability summary score: 0.432 (Rating: Pass)
- Model-robustness summary score: 0.893 (Rating: Great)

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
- Total samples: 140354
- Labeled samples: 21177 (coverage=15.1%)
- Positive labels: 2164 (10.2%)
- Negative labels: 19013

## Retention
- Pre-filter events (realized_return not NaN): 43472
- Pre-filter pos/neg (raw econ > cost): 11342 / 32130
- Post-filter labeled events: 21177
- Post-filter pos/neg (binary_label): 2164 / 19013
- Total retention: 48.7%
- Positive retention: 19.1%
- Negative retention: 59.2%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.31% / -0.84%
- Post-filter mean return (label=1/0): 1.74% / -0.52%
- Pre-filter Cohen's d: 4.391
- Post-filter Cohen's d: 2.801
- Pre-filter SNR (label=1): 2.392
- Post-filter SNR (label=1): 9.398

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 22.2%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.29%
- Mean return (label=1) minus cost: 1.44%
- Fraction of labeled events with |return| < cost: 14.2%
- Aleatoric uncertainty fraction (|return| < cost): 14.2%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=20267, win_rate=10.0%, mean_exp_ret=-0.31%, Sharpe_exp=-1.65
- Top 10%: n=20267, win_rate=10.0%, mean_exp_ret=-0.31%, Sharpe_exp=-1.65
- Top 20%: n=20267, win_rate=10.0%, mean_exp_ret=-0.31%, Sharpe_exp=-1.65
- Top 30%: n=20267, win_rate=10.0%, mean_exp_ret=-0.31%, Sharpe_exp=-1.65
- Top 40%: n=20267, win_rate=10.0%, mean_exp_ret=-0.31%, Sharpe_exp=-1.65

## Volatility Buckets (by volatility_1d)
- Vol low: n=7055, pos_rate=5.7%, mean_ret=-0.28%, Sharpe=-0.28
- Vol mid: n=7054, pos_rate=10.3%, mean_ret=-0.30%, Sharpe=-0.28
- Vol high: n=7055, pos_rate=14.7%, mean_ret=-0.29%, Sharpe=-0.27

## Interpretation Hints
- Coverage (15.1%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=2.801): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 9.398 → High SNR: positive-label returns are well separated from noise.
- Retention (total=48.7%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.878
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 21177
- Positive label rate: 10.2%

## Learnability
- Mean CV AUC: 0.6224
- Learnability score (AUC - 0.5 * std): 0.6165

## Entropy / Balance
- Balance score: 0.0000

## Combined Label-Quality Objective
- Combined score: 0.4316

## Interpretation Hints
- Learnability (mean AUC=0.6224): Mean CV AUC 0.60–0.70 → moderate learnability.
- Balance (entropy score=0.0000): Entropy score < 0.5 → labels are highly imbalanced or dominated by one class.
- Combined score (0.4316): Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.

## Overall Learnability Score
- Score (0-1): 0.432
- Rating: Pass
- Summary: Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=3532, n_test=3529, AUC=0.5491, Brier=0.1188, AP=0.1556
- Fold 2: n_train=7061, n_test=3529, AUC=0.7126, Brier=0.0513, AP=0.1290
- Fold 3: n_train=10590, n_test=3529, AUC=0.6504, Brier=0.0741, AP=0.1375
- Fold 4: n_train=14119, n_test=3529, AUC=0.6409, Brier=0.0857, AP=0.1613
- Fold 5: n_train=17648, n_test=3529, AUC=0.7067, Brier=0.0999, AP=0.2699

## Summary
- Mean AUC: 0.6520 (std=0.0590)
- Mean Brier: 0.0860 (std=0.0229)
- Mean AP: 0.1706 (std=0.0510)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9095

## Interpretation Hints
- Mean AUC (0.6520): Mean CV AUC 0.60–0.70 → moderate predictive power.
- Stability score (0.9095): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.0860): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.6647
- Pseudo-R^2 (y vs predicted prob): 0.0270
- Pseudo-R^2 95% CI: [0.0192, 0.0337]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.6050

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.4999
- Shuffled std AUC: 0.0201
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.6302
- Holdout Brier: 0.0965
- Holdout AP: 0.1734
- Holdout train / test: 14823 / 6354

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.6493
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4556 | Probe AUC: 0.6520 | Delta: 0.1963
- Baseline Brier: 0.0889 | Probe Brier: 0.0860 | Delta (baseline - probe): 0.0029
- Baseline AP: 0.0896 | Probe AP: 0.1706 | Delta: 0.0811

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.0486
- Residual lag-1 autocorrelation: 0.4566

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.6520 | LogisticRegression: 0.6054
- Comment: Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5802
- Min rolling AUC: 0.0000
- Max rolling AUC: 1.0000
- AUC at start: 0.4700
- AUC at end: 0.3688
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251207_090154.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 5.4572
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 17: mean=82.4000, std=4.7582
  - Feature 18: mean=62.6000, std=9.3509
  - Feature 1: mean=44.8000, std=9.6208
  - Feature 3: mean=37.2000, std=11.2143
  - Feature 4: mean=25.6000, std=11.2534
  - Feature 0: mean=24.0000, std=4.8990
  - Feature 2: mean=20.8000, std=7.3865
  - Feature 5: mean=14.4000, std=9.6042
  - Feature 9: mean=6.2000, std=12.4000
  - Feature 10: mean=3.2000, std=6.4000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 3081
- N mislabeled candidates (confident but wrong): 93
- Estimated label noise rate: 3.019%
- False negative rate (confident): 5.382%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.893
- Rating: Great
- Summary: Strong, stable probe model with consistent performance.

### Trading-Simulation
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1462 days
- Valid samples: 21177

## Model Calibration
- Brier Score: 0.0860
- Expected Calibration Error (ECE): 0.0144
- Max Calibration Error (MCE): 0.1439

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 671 (0.46/day)
- **Mean Return/Trade**: 0.5552%
- **PnL/Day**: 0.2548%
- **Win Rate**: 37.0%
- **Sharpe Ratio**: 11.187
- **Max Drawdown**: -21.91%
- **Final Equity**: 38.8650
- **Max Consecutive Losses**: 25
- **Avg Consecutive Losses**: 4.75
- **Win-Rate Stability**: 0.858

### Threshold 0.60
- **Trades**: 563 (0.39/day)
- **Mean Return/Trade**: 0.6303%
- **PnL/Day**: 0.2427%
- **Win Rate**: 39.6%
- **Sharpe Ratio**: 11.869
- **Max Drawdown**: -16.09%
- **Final Equity**: 32.8954
- **Max Consecutive Losses**: 21
- **Avg Consecutive Losses**: 4.66
- **Win-Rate Stability**: 0.871

### Threshold 0.65
- **Trades**: 453 (0.31/day)
- **Mean Return/Trade**: 0.7469%
- **PnL/Day**: 0.2314%
- **Win Rate**: 43.9%
- **Sharpe Ratio**: 13.206
- **Max Drawdown**: -11.07%
- **Final Equity**: 28.1681
- **Max Consecutive Losses**: 14
- **Avg Consecutive Losses**: 3.85
- **Win-Rate Stability**: 0.866

### Threshold 0.70
- **Trades**: 357 (0.24/day)
- **Mean Return/Trade**: 0.8717%
- **PnL/Day**: 0.2129%
- **Win Rate**: 47.9%
- **Sharpe Ratio**: 14.543
- **Max Drawdown**: -9.45%
- **Final Equity**: 21.6694
- **Max Consecutive Losses**: 11
- **Avg Consecutive Losses**: 3.26
- **Win-Rate Stability**: 0.879

### Threshold 0.75
- **Trades**: 289 (0.20/day)
- **Mean Return/Trade**: 0.9261%
- **PnL/Day**: 0.1831%
- **Win Rate**: 50.2%
- **Sharpe Ratio**: 14.505
- **Max Drawdown**: -9.45%
- **Final Equity**: 14.1151
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 2.94
- **Win-Rate Stability**: 0.893

### Threshold 0.80
- **Trades**: 212 (0.15/day)
- **Mean Return/Trade**: 0.9479%
- **PnL/Day**: 0.1375%
- **Win Rate**: 53.8%
- **Sharpe Ratio**: 12.898
- **Max Drawdown**: -8.20%
- **Final Equity**: 7.3014
- **Max Consecutive Losses**: 9
- **Avg Consecutive Losses**: 2.80
- **Win-Rate Stability**: 0.891

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 671 | 0.46 | 0.555% | 0.255% | 37.0% | 11.19 | -21.9% | 25 |
| 0.60 | 563 | 0.39 | 0.630% | 0.243% | 39.6% | 11.87 | -16.1% | 21 |
| 0.65 | 453 | 0.31 | 0.747% | 0.231% | 43.9% | 13.21 | -11.1% | 14 |
| 0.70 | 357 | 0.24 | 0.872% | 0.213% | 47.9% | 14.54 | -9.5% | 11 |
| 0.75 | 289 | 0.20 | 0.926% | 0.183% | 50.2% | 14.50 | -9.5% | 10 |
| 0.80 | 212 | 0.15 | 0.948% | 0.137% | 53.8% | 12.90 | -8.2% | 9 |

## Recommended Gating Threshold (from Trading Simulation)

- **Probability threshold**: 0.55
- **Trades**: 671 (0.459/day)
- **Mean return/trade**: 0.5552%
- **PnL/day**: 0.2548%
- **Sharpe (trades)**: 11.187
- **Max drawdown**: -21.91%
- **Final equity**: 38.8650

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