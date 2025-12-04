# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 12.3% (labeled / total samples)
- Label positive rate: 25.1%
- Label economic SNR (post-filter, label=1): 9.354
- Label effect size (post-filter Cohen's d): 4.217
- Aleatoric uncertainty fraction (|return| < cost): 5.7%

- Learnability mean CV AUC: 0.5862
- Learnability score (AUC - 0.5 * std): 0.5785
- Label balance (entropy score): 0.0000
- Combined label-quality score: 0.4049

- Probe model mean AUC: 0.5657
- Probe model stability score: 0.9580
- Probe model mean Brier score: 0.1978
- Probe global AUC (all folds combined): 0.5560
- Probe pseudo-R^2 (y vs predicted prob): -0.0339
- Probe permutation p-value (AUC): 0.005
- Probe vs baseline ΔAUC: 0.1134, ΔBrier (baseline - probe): -0.0052, ΔAP: 0.0699

- Label-quality summary score: 0.769 (Rating: Great)
- Learnability summary score: 0.405 (Rating: Pass)
- Model-robustness summary score: 0.617 (Rating: Pass)

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
- Total samples: 34135
- Labeled samples: 4187 (coverage=12.3%)
- Positive labels: 1051 (25.1%)
- Negative labels: 3136

## Retention
- Pre-filter events (realized_return not NaN): 8304
- Pre-filter pos/neg (raw econ > cost): 2670 / 5634
- Post-filter labeled events: 4187
- Post-filter pos/neg (binary_label): 1051 / 3136
- Total retention: 50.4%
- Positive retention: 39.4%
- Negative retention: 55.7%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 0.82% / -0.57%
- Post-filter mean return (label=1/0): 0.87% / -0.48%
- Pre-filter Cohen's d: 7.056
- Post-filter Cohen's d: 4.217
- Pre-filter SNR (label=1): 4.059
- Post-filter SNR (label=1): 9.354

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 8.6%
- Transaction cost (approx per event): 0.150%
- Unconditional mean event return: -0.14%
- Mean return (label=1) minus cost: 0.72%
- Fraction of labeled events with |return| < cost: 5.7%
- Aleatoric uncertainty fraction (|return| < cost): 5.7%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=210, win_rate=94.3%, mean_exp_ret=0.58%, Sharpe_exp=640.71
- Top 10%: n=419, win_rate=83.1%, mean_exp_ret=0.48%, Sharpe_exp=3.41
- Top 20%: n=838, win_rate=64.3%, mean_exp_ret=0.22%, Sharpe_exp=0.77
- Top 30%: n=3497, win_rate=24.3%, mean_exp_ret=-0.09%, Sharpe_exp=-0.39
- Top 40%: n=3497, win_rate=24.3%, mean_exp_ret=-0.09%, Sharpe_exp=-0.39

## Volatility Buckets (by volatility_1d)
- Vol low: n=1396, pos_rate=20.0%, mean_ret=-0.16%, Sharpe=-0.25
- Vol mid: n=1395, pos_rate=27.4%, mean_ret=-0.12%, Sharpe=-0.18
- Vol high: n=1396, pos_rate=27.9%, mean_ret=-0.13%, Sharpe=-0.19

## Interpretation Hints
- Coverage (12.3%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=4.217): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 9.354 → High SNR: positive-label returns are well separated from noise.
- Retention (total=50.4%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.769
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 4187
- Positive label rate: 25.1%

## Learnability
- Mean CV AUC: 0.5862
- Learnability score (AUC - 0.5 * std): 0.5785

## Entropy / Balance
- Balance score: 0.0000

## Combined Label-Quality Objective
- Combined score: 0.4049

## Interpretation Hints
- Learnability (mean AUC=0.5862): Mean CV AUC 0.55–0.60 → weak but potentially usable signal.
- Balance (entropy score=0.0000): Entropy score < 0.5 → labels are highly imbalanced or dominated by one class.
- Combined score (0.4049): Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.

## Overall Learnability Score
- Score (0-1): 0.405
- Rating: Pass
- Summary: Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=702, n_test=697, AUC=0.5444, Brier=0.2163, AP=0.3087
- Fold 2: n_train=1399, n_test=697, AUC=0.5956, Brier=0.1748, AP=0.2887
- Fold 3: n_train=2096, n_test=697, AUC=0.5854, Brier=0.2102, AP=0.3077
- Fold 4: n_train=2793, n_test=697, AUC=0.5331, Brier=0.2274, AP=0.3703
- Fold 5: n_train=3490, n_test=697, AUC=0.5700, Brier=0.1601, AP=0.2468

## Summary
- Mean AUC: 0.5657 (std=0.0238)
- Mean Brier: 0.1978 (std=0.0258)
- Mean AP: 0.3045 (std=0.0398)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9580

## Interpretation Hints
- Mean AUC (0.5657): Mean CV AUC 0.55–0.60 → weak but potentially exploitable signal.
- Stability score (0.9580): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1978): Mean Brier 0.18–0.25 → moderate calibration; room for improvement.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5560
- Pseudo-R^2 (y vs predicted prob): -0.0339
- Pseudo-R^2 95% CI: [-0.0529, -0.0126]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.1939

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5122
- Shuffled std AUC: 0.0404
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.5623
- Holdout Brier: 0.1918
- Holdout AP: 0.2954
- Holdout train / test: 2930 / 1257

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.5961
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4523 | Probe AUC: 0.5657 | Delta: 0.1134
- Baseline Brier: 0.1926 | Probe Brier: 0.1978 | Delta (baseline - probe): -0.0052
- Baseline AP: 0.2346 | Probe AP: 0.3045 | Delta: 0.0699

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.3033
- Residual lag-1 autocorrelation: 0.2519

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5657 | LogisticRegression: 0.5700
- Comment: All models perform similarly poorly; target has low intrinsic predictability.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5417
- Min rolling AUC: 0.2021
- Max rolling AUC: 0.9375
- AUC at start: 0.5559
- AUC at end: 0.6350
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251203_213032.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 4.5810
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 12: mean=51.4000, std=7.5525
  - Feature 13: mean=38.2000, std=6.3056
  - Feature 2: mean=33.4000, std=4.8826
  - Feature 7: mean=27.0000, std=9.6747
  - Feature 0: mean=26.2000, std=6.8527
  - Feature 10: mean=18.4000, std=7.7872
  - Feature 6: mean=15.6000, std=4.0792
  - Feature 9: mean=15.2000, std=5.7061
  - Feature 8: mean=13.0000, std=5.0990
  - Feature 4: mean=12.2000, std=4.4900
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 31
- N mislabeled candidates (confident but wrong): 4
- Estimated label noise rate: 12.903%
- False negative rate (confident): 0.445%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.617
- Rating: Pass
- Summary: Moderate robustness; some time variation or calibration issues.

### Trading-Simulation
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1489 days
- Valid samples: 4187

## Model Calibration
- Brier Score: 0.1978
- Expected Calibration Error (ECE): 0.0665
- Max Calibration Error (MCE): 0.3679

### Calibration Interpretation
- Brier 0.18-0.25 → Moderate calibration.
- ECE 0.05-0.15 → Moderate calibration error.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 925 (0.62/day)
- **Mean Return/Trade**: 0.3633%
- **PnL/Day**: 0.2257%
- **Win Rate**: 61.7%
- **Sharpe Ratio**: 15.507
- **Max Drawdown**: -8.69%
- **Final Equity**: 27.9579
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 2.20
- **Win-Rate Stability**: 0.862

### Threshold 0.60
- **Trades**: 797 (0.54/day)
- **Mean Return/Trade**: 0.4103%
- **PnL/Day**: 0.2196%
- **Win Rate**: 65.7%
- **Sharpe Ratio**: 16.691
- **Max Drawdown**: -7.43%
- **Final Equity**: 25.6496
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 2.01
- **Win-Rate Stability**: 0.872

### Threshold 0.65
- **Trades**: 677 (0.45/day)
- **Mean Return/Trade**: 0.4836%
- **PnL/Day**: 0.2199%
- **Win Rate**: 71.2%
- **Sharpe Ratio**: 19.226
- **Max Drawdown**: -5.86%
- **Final Equity**: 25.8385
- **Max Consecutive Losses**: 13
- **Avg Consecutive Losses**: 1.77
- **Win-Rate Stability**: 0.900

### Threshold 0.70
- **Trades**: 557 (0.37/day)
- **Mean Return/Trade**: 0.5517%
- **PnL/Day**: 0.2064%
- **Win Rate**: 75.8%
- **Sharpe Ratio**: 21.276
- **Max Drawdown**: -4.21%
- **Final Equity**: 21.1996
- **Max Consecutive Losses**: 7
- **Avg Consecutive Losses**: 1.50
- **Win-Rate Stability**: 0.931

### Threshold 0.75
- **Trades**: 453 (0.30/day)
- **Mean Return/Trade**: 0.6242%
- **PnL/Day**: 0.1899%
- **Win Rate**: 80.6%
- **Sharpe Ratio**: 23.881
- **Max Drawdown**: -2.72%
- **Final Equity**: 16.6442
- **Max Consecutive Losses**: 4
- **Avg Consecutive Losses**: 1.29
- **Win-Rate Stability**: 0.944

### Threshold 0.80
- **Trades**: 359 (0.24/day)
- **Mean Return/Trade**: 0.7122%
- **PnL/Day**: 0.1717%
- **Win Rate**: 86.4%
- **Sharpe Ratio**: 29.669
- **Max Drawdown**: -2.01%
- **Final Equity**: 12.7289
- **Max Consecutive Losses**: 3
- **Avg Consecutive Losses**: 1.17
- **Win-Rate Stability**: 0.956

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 925 | 0.62 | 0.363% | 0.226% | 61.7% | 15.51 | -8.7% | 10 |
| 0.60 | 797 | 0.54 | 0.410% | 0.220% | 65.7% | 16.69 | -7.4% | 10 |
| 0.65 | 677 | 0.45 | 0.484% | 0.220% | 71.2% | 19.23 | -5.9% | 13 |
| 0.70 | 557 | 0.37 | 0.552% | 0.206% | 75.8% | 21.28 | -4.2% | 7 |
| 0.75 | 453 | 0.30 | 0.624% | 0.190% | 80.6% | 23.88 | -2.7% | 4 |
| 0.80 | 359 | 0.24 | 0.712% | 0.172% | 86.4% | 29.67 | -2.0% | 3 |

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