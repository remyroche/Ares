# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 15.1% (labeled / total samples)
- Label positive rate: 18.3%
- Label economic SNR (post-filter, label=1): 2.859
- Label effect size (post-filter Cohen's d): 2.656
- Aleatoric uncertainty fraction (|return| < cost): 11.5%

- Learnability mean CV AUC: 0.5239
- Learnability score (AUC - 0.5 * std): 0.5131
- Label balance (entropy score): 0.0000
- Combined label-quality score: 0.3592

- Probe model mean AUC: 0.5589
- Probe model stability score: 0.9666
- Probe model mean Brier score: 0.1466
- Probe global AUC (all folds combined): 0.5732
- Probe pseudo-R^2 (y vs predicted prob): 0.0061
- Probe permutation p-value (AUC): 0.005
- Probe vs baseline ΔAUC: 0.0905, ΔBrier (baseline - probe): 0.0016, ΔAP: 0.0534

- Label-quality summary score: 0.851 (Rating: Great)
- Learnability summary score: 0.359 (Rating: Bad)
- Model-robustness summary score: 0.686 (Rating: Pass)

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
- Labeled samples: 21176 (coverage=15.1%)
- Positive labels: 3873 (18.3%)
- Negative labels: 17303

## Retention
- Pre-filter events (realized_return not NaN): 43470
- Pre-filter pos/neg (raw econ > cost): 11800 / 31670
- Post-filter labeled events: 21176
- Post-filter pos/neg (binary_label): 3873 / 17303
- Total retention: 48.7%
- Positive retention: 32.8%
- Negative retention: 54.6%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.46% / -0.93%
- Post-filter mean return (label=1/0): 1.46% / -0.68%
- Pre-filter Cohen's d: 4.786
- Post-filter Cohen's d: 2.656
- Pre-filter SNR (label=1): 2.506
- Post-filter SNR (label=1): 2.859

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 13.8%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.29%
- Mean return (label=1) minus cost: 1.16%
- Fraction of labeled events with |return| < cost: 11.5%
- Aleatoric uncertainty fraction (|return| < cost): 11.5%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=20060, win_rate=17.7%, mean_exp_ret=-0.15%, Sharpe_exp=-0.68
- Top 10%: n=20060, win_rate=17.7%, mean_exp_ret=-0.15%, Sharpe_exp=-0.68
- Top 20%: n=20060, win_rate=17.7%, mean_exp_ret=-0.15%, Sharpe_exp=-0.68
- Top 30%: n=20060, win_rate=17.7%, mean_exp_ret=-0.15%, Sharpe_exp=-0.68
- Top 40%: n=20060, win_rate=17.7%, mean_exp_ret=-0.15%, Sharpe_exp=-0.68

## Volatility Buckets (by volatility_1d)
- Vol low: n=7054, pos_rate=13.5%, mean_ret=-0.28%, Sharpe=-0.26
- Vol mid: n=7054, pos_rate=17.5%, mean_ret=-0.30%, Sharpe=-0.26
- Vol high: n=7055, pos_rate=23.9%, mean_ret=-0.29%, Sharpe=-0.24

## Interpretation Hints
- Coverage (15.1%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=2.656): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 2.859 → High SNR: positive-label returns are well separated from noise.
- Retention (total=48.7%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.851
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 21176
- Positive label rate: 18.3%

## Learnability
- Mean CV AUC: 0.5239
- Learnability score (AUC - 0.5 * std): 0.5131

## Entropy / Balance
- Balance score: 0.0000

## Combined Label-Quality Objective
- Combined score: 0.3592

## Interpretation Hints
- Learnability (mean AUC=0.5239): Mean CV AUC < 0.55 → very weak learnability; labels are close to random.
- Balance (entropy score=0.0000): Entropy score < 0.5 → labels are highly imbalanced or dominated by one class.
- Combined score (0.3592): Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds.

## Overall Learnability Score
- Score (0-1): 0.359
- Rating: Bad
- Summary: Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=3531, n_test=3529, AUC=0.5293, Brier=0.1678, AP=0.2284
- Fold 2: n_train=7060, n_test=3529, AUC=0.5799, Brier=0.1012, AP=0.1444
- Fold 3: n_train=10589, n_test=3529, AUC=0.5487, Brier=0.1458, AP=0.2080
- Fold 4: n_train=14118, n_test=3529, AUC=0.5597, Brier=0.1560, AP=0.2363
- Fold 5: n_train=17647, n_test=3529, AUC=0.5768, Brier=0.1620, AP=0.2799

## Summary
- Mean AUC: 0.5589 (std=0.0187)
- Mean Brier: 0.1466 (std=0.0238)
- Mean AP: 0.2194 (std=0.0442)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9666

## Interpretation Hints
- Mean AUC (0.5589): Mean CV AUC 0.55–0.60 → weak but potentially exploitable signal.
- Stability score (0.9666): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1466): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5732
- Pseudo-R^2 (y vs predicted prob): 0.0061
- Pseudo-R^2 95% CI: [-0.0006, 0.0115]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.2736

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.4998
- Shuffled std AUC: 0.0060
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.5408
- Holdout Brier: 0.1621
- Holdout AP: 0.2371
- Holdout train / test: 14823 / 6353

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.5903
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4684 | Probe AUC: 0.5589 | Delta: 0.0905
- Baseline Brier: 0.1481 | Probe Brier: 0.1466 | Delta (baseline - probe): 0.0016
- Baseline AP: 0.1660 | Probe AP: 0.2194 | Delta: 0.0534

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.0964
- Residual lag-1 autocorrelation: 0.4223

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5589 | LogisticRegression: 0.5470
- Comment: All models perform similarly poorly; target has low intrinsic predictability.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5246
- Min rolling AUC: 0.0000
- Max rolling AUC: 1.0000
- AUC at start: 0.2578
- AUC at end: 0.5492
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251209_225111.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 4.6653
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 15: mean=80.6000, std=3.8262
  - Feature 1: mean=55.4000, std=9.8102
  - Feature 16: mean=50.8000, std=11.5135
  - Feature 0: mean=36.4000, std=6.7112
  - Feature 3: mean=29.8000, std=5.8788
  - Feature 4: mean=21.4000, std=5.8515
  - Feature 2: mean=20.8000, std=2.0396
  - Feature 6: mean=15.4000, std=4.0792
  - Feature 14: mean=4.8000, std=9.6000
  - Feature 10: mean=2.8000, std=5.6000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 43
- N mislabeled candidates (confident but wrong): 5
- Estimated label noise rate: 11.628%
- False negative rate (confident): 0.158%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.686
- Rating: Pass
- Summary: Moderate robustness; some time variation or calibration issues.

### Trading-Simulation
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1462 days
- Valid samples: 21176

## Model Calibration
- Brier Score: 0.1466
- Expected Calibration Error (ECE): 0.0170
- Max Calibration Error (MCE): 0.1482

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 748 (0.51/day)
- **Mean Return/Trade**: 0.4511%
- **PnL/Day**: 0.2308%
- **Win Rate**: 49.3%
- **Sharpe Ratio**: 8.991
- **Max Drawdown**: -36.56%
- **Final Equity**: 27.0315
- **Max Consecutive Losses**: 13
- **Avg Consecutive Losses**: 3.38
- **Win-Rate Stability**: 0.842

### Threshold 0.60
- **Trades**: 675 (0.46/day)
- **Mean Return/Trade**: 0.5063%
- **PnL/Day**: 0.2337%
- **Win Rate**: 51.4%
- **Sharpe Ratio**: 9.657
- **Max Drawdown**: -31.40%
- **Final Equity**: 28.4016
- **Max Consecutive Losses**: 14
- **Avg Consecutive Losses**: 3.25
- **Win-Rate Stability**: 0.852

### Threshold 0.65
- **Trades**: 594 (0.41/day)
- **Mean Return/Trade**: 0.5654%
- **PnL/Day**: 0.2297%
- **Win Rate**: 53.9%
- **Sharpe Ratio**: 10.284
- **Max Drawdown**: -25.79%
- **Final Equity**: 27.0088
- **Max Consecutive Losses**: 12
- **Avg Consecutive Losses**: 3.11
- **Win-Rate Stability**: 0.862

### Threshold 0.70
- **Trades**: 531 (0.36/day)
- **Mean Return/Trade**: 0.5849%
- **PnL/Day**: 0.2124%
- **Win Rate**: 55.2%
- **Sharpe Ratio**: 10.157
- **Max Drawdown**: -22.38%
- **Final Equity**: 21.1223
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 2.98
- **Win-Rate Stability**: 0.887

### Threshold 0.75
- **Trades**: 469 (0.32/day)
- **Mean Return/Trade**: 0.6383%
- **PnL/Day**: 0.2047%
- **Win Rate**: 57.4%
- **Sharpe Ratio**: 10.527
- **Max Drawdown**: -20.60%
- **Final Equity**: 18.9876
- **Max Consecutive Losses**: 9
- **Avg Consecutive Losses**: 2.82
- **Win-Rate Stability**: 0.918

### Threshold 0.80
- **Trades**: 398 (0.27/day)
- **Mean Return/Trade**: 0.6377%
- **PnL/Day**: 0.1736%
- **Win Rate**: 59.8%
- **Sharpe Ratio**: 9.786
- **Max Drawdown**: -18.19%
- **Final Equity**: 12.1407
- **Max Consecutive Losses**: 8
- **Avg Consecutive Losses**: 2.62
- **Win-Rate Stability**: 0.899

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 748 | 0.51 | 0.451% | 0.231% | 49.3% | 8.99 | -36.6% | 13 |
| 0.60 | 675 | 0.46 | 0.506% | 0.234% | 51.4% | 9.66 | -31.4% | 14 |
| 0.65 | 594 | 0.41 | 0.565% | 0.230% | 53.9% | 10.28 | -25.8% | 12 |
| 0.70 | 531 | 0.36 | 0.585% | 0.212% | 55.2% | 10.16 | -22.4% | 10 |
| 0.75 | 469 | 0.32 | 0.638% | 0.205% | 57.4% | 10.53 | -20.6% | 9 |
| 0.80 | 398 | 0.27 | 0.638% | 0.174% | 59.8% | 9.79 | -18.2% | 8 |

## Recommended Gating Threshold (from Trading Simulation)

- **Probability threshold**: 0.60
- **Trades**: 675 (0.462/day)
- **Mean return/trade**: 0.5063%
- **PnL/day**: 0.2337%
- **Sharpe (trades)**: 9.657
- **Max drawdown**: -31.40%
- **Final equity**: 28.4016

## Regime-Specific Recommended Thresholds

- **Regime** `hmm_-1.0`:
  - prob_threshold = 0.60
  - trades/day ≈ 0.068
  - mean_return ≈ 0.9605%
  - Sharpe ≈ 3.769
  - n_trades = 23
- **Regime** `hmm_0.0`:
  - prob_threshold = 0.75
  - trades/day ≈ 0.133
  - mean_return ≈ 1.1881%
  - Sharpe ≈ 9.683
  - n_trades = 45
- **Regime** `hmm_1.0`:
  - prob_threshold = 0.55
  - trades/day ≈ 0.579
  - mean_return ≈ 0.5487%
  - Sharpe ≈ 5.683
  - n_trades = 195
- **Regime** `hmm_2.0`:
  - prob_threshold = 0.60
  - trades/day ≈ 0.201
  - mean_return ≈ 0.2479%
  - Sharpe ≈ 1.410
  - n_trades = 72
- **Regime** `hmm_3.0`:
  - prob_threshold = 0.80
  - trades/day ≈ 0.098
  - mean_return ≈ 0.8588%
  - Sharpe ≈ 4.800
  - n_trades = 33
- **Regime** `hmm_4.0`:
  - prob_threshold = 0.55
  - trades/day ≈ 0.833
  - mean_return ≈ 0.4453%
  - Sharpe ≈ 5.554
  - n_trades = 299

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