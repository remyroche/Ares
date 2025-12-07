# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 15.1% (labeled / total samples)
- Label positive rate: 17.6%
- Label economic SNR (post-filter, label=1): 3.118
- Label effect size (post-filter Cohen's d): 2.872
- Aleatoric uncertainty fraction (|return| < cost): 12.3%

- Learnability mean CV AUC: 0.5382
- Learnability score (AUC - 0.5 * std): 0.5165
- Label balance (entropy score): 0.0000
- Combined label-quality score: 0.3616

- Probe model mean AUC: 0.5799
- Probe model stability score: 0.9561
- Probe model mean Brier score: 0.1405
- Probe global AUC (all folds combined): 0.5931
- Probe pseudo-R^2 (y vs predicted prob): 0.0094
- Probe permutation p-value (AUC): 0.005
- Probe vs baseline ΔAUC: 0.1067, ΔBrier (baseline - probe): 0.0020, ΔAP: 0.0570

- Label-quality summary score: 0.841 (Rating: Great)
- Learnability summary score: 0.362 (Rating: Bad)
- Model-robustness summary score: 0.733 (Rating: Great)

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
- Positive labels: 3718 (17.6%)
- Negative labels: 17459

## Retention
- Pre-filter events (realized_return not NaN): 43471
- Pre-filter pos/neg (raw econ > cost): 11370 / 32101
- Post-filter labeled events: 21177
- Post-filter pos/neg (binary_label): 3718 / 17459
- Total retention: 48.7%
- Positive retention: 32.7%
- Negative retention: 54.4%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.32% / -0.85%
- Post-filter mean return (label=1/0): 1.37% / -0.65%
- Pre-filter Cohen's d: 4.884
- Post-filter Cohen's d: 2.872
- Pre-filter SNR (label=1): 2.675
- Post-filter SNR (label=1): 3.118

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 13.9%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.29%
- Mean return (label=1) minus cost: 1.07%
- Fraction of labeled events with |return| < cost: 12.3%
- Aleatoric uncertainty fraction (|return| < cost): 12.3%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=1059, win_rate=48.6%, mean_exp_ret=0.61%, Sharpe_exp=11.59
- Top 10%: n=19458, win_rate=13.6%, mean_exp_ret=-0.36%, Sharpe_exp=-1.37
- Top 20%: n=19458, win_rate=13.6%, mean_exp_ret=-0.36%, Sharpe_exp=-1.37
- Top 30%: n=19458, win_rate=13.6%, mean_exp_ret=-0.36%, Sharpe_exp=-1.37
- Top 40%: n=19458, win_rate=13.6%, mean_exp_ret=-0.36%, Sharpe_exp=-1.37

## Volatility Buckets (by volatility_1d)
- Vol low: n=7055, pos_rate=12.1%, mean_ret=-0.28%, Sharpe=-0.29
- Vol mid: n=7054, pos_rate=17.5%, mean_ret=-0.31%, Sharpe=-0.29
- Vol high: n=7055, pos_rate=23.0%, mean_ret=-0.30%, Sharpe=-0.28

## Interpretation Hints
- Coverage (15.1%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=2.872): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 3.118 → High SNR: positive-label returns are well separated from noise.
- Retention (total=48.7%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.841
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 21177
- Positive label rate: 17.6%

## Learnability
- Mean CV AUC: 0.5382
- Learnability score (AUC - 0.5 * std): 0.5165

## Entropy / Balance
- Balance score: 0.0000

## Combined Label-Quality Objective
- Combined score: 0.3616

## Interpretation Hints
- Learnability (mean AUC=0.5382): Mean CV AUC < 0.55 → very weak learnability; labels are close to random.
- Balance (entropy score=0.0000): Entropy score < 0.5 → labels are highly imbalanced or dominated by one class.
- Combined score (0.3616): Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds.

## Overall Learnability Score
- Score (0-1): 0.362
- Rating: Bad
- Summary: Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=3532, n_test=3529, AUC=0.5349, Brier=0.1645, AP=0.2208
- Fold 2: n_train=7061, n_test=3529, AUC=0.5984, Brier=0.0983, AP=0.1510
- Fold 3: n_train=10590, n_test=3529, AUC=0.5776, Brier=0.1398, AP=0.2098
- Fold 4: n_train=14119, n_test=3529, AUC=0.5794, Brier=0.1419, AP=0.2142
- Fold 5: n_train=17648, n_test=3529, AUC=0.6093, Brier=0.1578, AP=0.2852

## Summary
- Mean AUC: 0.5799 (std=0.0255)
- Mean Brier: 0.1405 (std=0.0231)
- Mean AP: 0.2162 (std=0.0426)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9561

## Interpretation Hints
- Mean AUC (0.5799): Mean CV AUC 0.55–0.60 → weak but potentially exploitable signal.
- Stability score (0.9561): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1405): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5931
- Pseudo-R^2 (y vs predicted prob): 0.0094
- Pseudo-R^2 95% CI: [0.0040, 0.0143]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.3208

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.4968
- Shuffled std AUC: 0.0132
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.5743
- Holdout Brier: 0.1535
- Holdout AP: 0.2327
- Holdout train / test: 14823 / 6354

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.5985
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4732 | Probe AUC: 0.5799 | Delta: 0.1067
- Baseline Brier: 0.1425 | Probe Brier: 0.1405 | Delta (baseline - probe): 0.0020
- Baseline AP: 0.1592 | Probe AP: 0.2162 | Delta: 0.0570

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.0860
- Residual lag-1 autocorrelation: 0.3951

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5799 | LogisticRegression: 0.5696
- Comment: All models perform similarly poorly; target has low intrinsic predictability.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5493
- Min rolling AUC: 0.0102
- Max rolling AUC: 1.0000
- AUC at start: 0.4058
- AUC at end: 0.5536
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251207_124726.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 5.1963
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 15: mean=82.6000, std=11.9766
  - Feature 16: mean=54.2000, std=9.9880
  - Feature 1: mean=50.0000, std=10.3537
  - Feature 3: mean=40.2000, std=1.9391
  - Feature 4: mean=28.4000, std=5.3889
  - Feature 0: mean=26.0000, std=4.6904
  - Feature 2: mean=14.8000, std=7.2222
  - Feature 6: mean=13.0000, std=3.5777
  - Feature 14: mean=4.0000, std=8.0000
  - Feature 11: mean=3.4000, std=6.8000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 143
- N mislabeled candidates (confident but wrong): 17
- Estimated label noise rate: 11.888%
- False negative rate (confident): 0.563%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.733
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
- Brier Score: 0.1405
- Expected Calibration Error (ECE): 0.0147
- Max Calibration Error (MCE): 0.1582

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 1565 (1.07/day)
- **Mean Return/Trade**: 0.2175%
- **PnL/Day**: 0.2328%
- **Win Rate**: 42.6%
- **Sharpe Ratio**: 6.760
- **Max Drawdown**: -41.76%
- **Final Equity**: 26.4187
- **Max Consecutive Losses**: 23
- **Avg Consecutive Losses**: 4.26
- **Win-Rate Stability**: 0.867

### Threshold 0.60
- **Trades**: 1449 (0.99/day)
- **Mean Return/Trade**: 0.2519%
- **PnL/Day**: 0.2497%
- **Win Rate**: 43.8%
- **Sharpe Ratio**: 7.575
- **Max Drawdown**: -41.51%
- **Final Equity**: 34.1283
- **Max Consecutive Losses**: 23
- **Avg Consecutive Losses**: 4.13
- **Win-Rate Stability**: 0.860

### Threshold 0.65
- **Trades**: 1330 (0.91/day)
- **Mean Return/Trade**: 0.3021%
- **PnL/Day**: 0.2748%
- **Win Rate**: 45.8%
- **Sharpe Ratio**: 8.742
- **Max Drawdown**: -38.50%
- **Final Equity**: 49.7172
- **Max Consecutive Losses**: 23
- **Avg Consecutive Losses**: 3.90
- **Win-Rate Stability**: 0.858

### Threshold 0.70
- **Trades**: 1173 (0.80/day)
- **Mean Return/Trade**: 0.3438%
- **PnL/Day**: 0.2758%
- **Win Rate**: 47.7%
- **Sharpe Ratio**: 9.416
- **Max Drawdown**: -37.09%
- **Final Equity**: 51.1420
- **Max Consecutive Losses**: 23
- **Avg Consecutive Losses**: 3.78
- **Win-Rate Stability**: 0.848

### Threshold 0.75
- **Trades**: 1029 (0.70/day)
- **Mean Return/Trade**: 0.4057%
- **PnL/Day**: 0.2855%
- **Win Rate**: 49.7%
- **Sharpe Ratio**: 10.504
- **Max Drawdown**: -30.35%
- **Final Equity**: 59.5907
- **Max Consecutive Losses**: 22
- **Avg Consecutive Losses**: 3.55
- **Win-Rate Stability**: 0.840

### Threshold 0.80
- **Trades**: 853 (0.58/day)
- **Mean Return/Trade**: 0.4335%
- **PnL/Day**: 0.2529%
- **Win Rate**: 51.2%
- **Sharpe Ratio**: 10.274
- **Max Drawdown**: -23.39%
- **Final Equity**: 37.5486
- **Max Consecutive Losses**: 20
- **Avg Consecutive Losses**: 3.56
- **Win-Rate Stability**: 0.856

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 1565 | 1.07 | 0.218% | 0.233% | 42.6% | 6.76 | -41.8% | 23 |
| 0.60 | 1449 | 0.99 | 0.252% | 0.250% | 43.8% | 7.57 | -41.5% | 23 |
| 0.65 | 1330 | 0.91 | 0.302% | 0.275% | 45.8% | 8.74 | -38.5% | 23 |
| 0.70 | 1173 | 0.80 | 0.344% | 0.276% | 47.7% | 9.42 | -37.1% | 23 |
| 0.75 | 1029 | 0.70 | 0.406% | 0.286% | 49.7% | 10.50 | -30.3% | 22 |
| 0.80 | 853 | 0.58 | 0.434% | 0.253% | 51.2% | 10.27 | -23.4% | 20 |

## Recommended Gating Threshold (from Trading Simulation)

- **Probability threshold**: 0.75
- **Trades**: 1029 (0.704/day)
- **Mean return/trade**: 0.4057%
- **PnL/day**: 0.2855%
- **Sharpe (trades)**: 10.504
- **Max drawdown**: -30.35%
- **Final equity**: 59.5907

## Regime-Specific Recommended Thresholds

- **Regime** `hmm_-1.0`:
  - prob_threshold = 0.65
  - trades/day ≈ 0.148
  - mean_return ≈ 0.7797%
  - Sharpe ≈ 4.404
  - n_trades = 50
- **Regime** `hmm_0.0`:
  - prob_threshold = 0.65
  - trades/day ≈ 0.399
  - mean_return ≈ 0.6141%
  - Sharpe ≈ 6.244
  - n_trades = 135
- **Regime** `hmm_1.0`:
  - prob_threshold = 0.75
  - trades/day ≈ 0.950
  - mean_return ≈ 0.3886%
  - Sharpe ≈ 5.760
  - n_trades = 320
- **Regime** `hmm_2.0`:
  - prob_threshold = 0.55
  - trades/day ≈ 0.298
  - mean_return ≈ 0.3803%
  - Sharpe ≈ 2.891
  - n_trades = 107
- **Regime** `hmm_3.0`:
  - prob_threshold = 0.80
  - trades/day ≈ 0.356
  - mean_return ≈ 0.1223%
  - Sharpe ≈ 1.094
  - n_trades = 120
- **Regime** `hmm_4.0`:
  - prob_threshold = 0.75
  - trades/day ≈ 1.053
  - mean_return ≈ 0.3911%
  - Sharpe ≈ 6.022
  - n_trades = 378

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