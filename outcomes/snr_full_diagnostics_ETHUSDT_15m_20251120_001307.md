# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 16.7% (labeled / total samples)
- Label positive rate: 50.0%
- Label economic SNR (post-filter, label=1): 4.370
- Label effect size (post-filter Cohen's d): 9.543
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

- Learnability mean CV AUC: 0.6761
- Learnability score (AUC - 0.5 * std): 0.6420
- Label balance (entropy score): 1.0000
- Combined label-quality score: 0.7494

- Probe model mean AUC: 0.6736
- Probe model stability score: 0.8894
- Probe model mean Brier score: 0.2096
- Probe global AUC (all folds combined): 0.6823
- Probe pseudo-R^2 (y vs predicted prob): 0.1555
- Probe permutation p-value (AUC): 0.005
- Probe vs baseline ΔAUC: 0.2420, ΔBrier (baseline - probe): 0.0609, ΔAP: 0.2601

- Label-quality summary score: 0.862 (Rating: Great)
- Learnability summary score: 0.749 (Rating: Great)
- Model-robustness summary score: 0.765 (Rating: Great)

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
- Total samples: 135775
- Labeled samples: 22658 (coverage=16.7%)
- Positive labels: 11329 (50.0%)
- Negative labels: 11329

## Retention
- Pre-filter events (realized_return not NaN): 44589
- Pre-filter pos/neg (raw econ > cost): 14255 / 30334
- Post-filter labeled events: 22658
- Post-filter pos/neg (binary_label): 11329 / 11329
- Total retention: 50.8%
- Positive retention: 79.5%
- Negative retention: 37.3%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.05% / -0.68%
- Post-filter mean return (label=1/0): 1.21% / -0.82%
- Pre-filter Cohen's d: 4.870
- Post-filter Cohen's d: 9.543
- Pre-filter SNR (label=1): 2.473
- Post-filter SNR (label=1): 4.370

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.150%
- Unconditional mean event return: 0.19%
- Mean return (label=1) minus cost: 1.06%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=1133, win_rate=100.0%, mean_exp_ret=0.50%, Sharpe_exp=0.92
- Top 10%: n=2266, win_rate=99.9%, mean_exp_ret=0.54%, Sharpe_exp=0.99
- Top 20%: n=4532, win_rate=89.1%, mean_exp_ret=0.42%, Sharpe_exp=0.85
- Top 30%: n=6798, win_rate=74.7%, mean_exp_ret=0.28%, Sharpe_exp=0.62
- Top 40%: n=11500, win_rate=67.4%, mean_exp_ret=0.17%, Sharpe_exp=0.44

## Volatility Buckets (by volatility_1d)
- Vol low: n=7553, pos_rate=33.7%, mean_ret=-0.15%, Sharpe=-0.17
- Vol mid: n=7552, pos_rate=35.7%, mean_ret=-0.11%, Sharpe=-0.12
- Vol high: n=7553, pos_rate=80.6%, mean_ret=0.84%, Sharpe=0.94

## Interpretation Hints
- Coverage (16.7%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=9.543): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 4.370 → High SNR: positive-label returns are well separated from noise.
- Retention (total=50.8%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.862
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 22658
- Positive label rate: 50.0%

## Learnability
- Mean CV AUC: 0.6761
- Learnability score (AUC - 0.5 * std): 0.6420

## Entropy / Balance
- Balance score: 1.0000

## Combined Label-Quality Objective
- Combined score: 0.7494

## Interpretation Hints
- Learnability (mean AUC=0.6761): Mean CV AUC 0.60–0.70 → moderate learnability.
- Balance (entropy score=1.0000): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.7494): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 0.749
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=3778, n_test=3776, AUC=0.6503, Brier=0.2346, AP=0.6656
- Fold 2: n_train=7554, n_test=3776, AUC=0.5574, Brier=0.2194, AP=0.4493
- Fold 3: n_train=11330, n_test=3776, AUC=0.6806, Brier=0.2074, AP=0.7097
- Fold 4: n_train=15106, n_test=3776, AUC=0.6905, Brier=0.2093, AP=0.7409
- Fold 5: n_train=18882, n_test=3776, AUC=0.7891, Brier=0.1774, AP=0.8683

## Summary
- Mean AUC: 0.6736 (std=0.0745)
- Mean Brier: 0.2096 (std=0.0188)
- Mean AP: 0.6868 (std=0.1366)
- Stability score (1 - std(AUC)/mean(AUC)): 0.8894

## Interpretation Hints
- Mean AUC (0.6736): Mean CV AUC 0.60–0.70 → moderate predictive power.
- Stability score (0.8894): Stability score 0.8–0.9 → moderate stability; some variation across folds.
- Mean Brier (0.2096): Mean Brier 0.18–0.25 → moderate calibration; room for improvement.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.6823
- Pseudo-R^2 (y vs predicted prob): 0.1555
- Pseudo-R^2 95% CI: [0.1464, 0.1645]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.8700

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4315 | Probe AUC: 0.6736 | Delta: 0.2420
- Baseline Brier: 0.2705 | Probe Brier: 0.2096 | Delta (baseline - probe): 0.0609
- Baseline AP: 0.4267 | Probe AP: 0.6868 | Delta: 0.2601

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1676
- Residual lag-1 autocorrelation: 0.5898

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.6736 | LogisticRegression: 0.5774
- Comment: Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present.

## Overall Model-Robustness Score
- Score (0-1): 0.765
- Rating: Great
- Summary: Strong, stable probe model with consistent performance.

### Trading-Simulation
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1462 days
- Valid samples: 22658

## Model Calibration
- Brier Score: 0.2096
- Expected Calibration Error (ECE): 0.0585
- Max Calibration Error (MCE): 0.2112

### Calibration Interpretation
- Brier 0.18-0.25 → Moderate calibration.
- ECE 0.05-0.15 → Moderate calibration error.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 6836 (4.68/day)
- **Mean Return/Trade**: 0.7119%
- **PnL/Day**: 3.3287%
- **Win Rate**: 74.5%
- **Sharpe Ratio**: 61.568
- **Max Drawdown**: -25.66%
- **Final Equity**: 843053470919304806400.0000
- **Max Consecutive Losses**: 22
- **Avg Consecutive Losses**: 3.84
- **Win-Rate Stability**: 0.802

### Threshold 0.60
- **Trades**: 5602 (3.83/day)
- **Mean Return/Trade**: 0.8574%
- **PnL/Day**: 3.2853%
- **Win Rate**: 81.1%
- **Sharpe Ratio**: 73.827
- **Max Drawdown**: -22.69%
- **Final Equity**: 478333861427060736000.0000
- **Max Consecutive Losses**: 17
- **Avg Consecutive Losses**: 3.41
- **Win-Rate Stability**: 0.827

### Threshold 0.65
- **Trades**: 4620 (3.16/day)
- **Mean Return/Trade**: 1.0177%
- **PnL/Day**: 3.2159%
- **Win Rate**: 88.4%
- **Sharpe Ratio**: 95.426
- **Max Drawdown**: -25.36%
- **Final Equity**: 183601233967937880064.0000
- **Max Consecutive Losses**: 19
- **Avg Consecutive Losses**: 3.13
- **Win-Rate Stability**: 0.866

### Threshold 0.70
- **Trades**: 4044 (2.77/day)
- **Mean Return/Trade**: 1.1254%
- **PnL/Day**: 3.1130%
- **Win Rate**: 93.2%
- **Sharpe Ratio**: 122.696
- **Max Drawdown**: -14.76%
- **Final Equity**: 42220174133234917376.0000
- **Max Consecutive Losses**: 16
- **Avg Consecutive Losses**: 2.97
- **Win-Rate Stability**: 0.902

### Threshold 0.75
- **Trades**: 3589 (2.45/day)
- **Mean Return/Trade**: 1.2106%
- **PnL/Day**: 2.9720%
- **Win Rate**: 97.0%
- **Sharpe Ratio**: 174.827
- **Max Drawdown**: -8.01%
- **Final Equity**: 5540287724817795072.0000
- **Max Consecutive Losses**: 9
- **Avg Consecutive Losses**: 2.42
- **Win-Rate Stability**: 0.947

### Threshold 0.80
- **Trades**: 3213 (2.20/day)
- **Mean Return/Trade**: 1.2523%
- **PnL/Day**: 2.7521%
- **Win Rate**: 98.6%
- **Sharpe Ratio**: 235.358
- **Max Drawdown**: -8.01%
- **Final Equity**: 228718024775217120.0000
- **Max Consecutive Losses**: 9
- **Avg Consecutive Losses**: 1.76
- **Win-Rate Stability**: 0.966

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 6836 | 4.68 | 0.712% | 3.329% | 74.5% | 61.57 | -25.7% | 22 |
| 0.60 | 5602 | 3.83 | 0.857% | 3.285% | 81.1% | 73.83 | -22.7% | 17 |
| 0.65 | 4620 | 3.16 | 1.018% | 3.216% | 88.4% | 95.43 | -25.4% | 19 |
| 0.70 | 4044 | 2.77 | 1.125% | 3.113% | 93.2% | 122.70 | -14.8% | 16 |
| 0.75 | 3589 | 2.45 | 1.211% | 2.972% | 97.0% | 174.83 | -8.0% | 9 |
| 0.80 | 3213 | 2.20 | 1.252% | 2.752% | 98.6% | 235.36 | -8.0% | 9 |

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