# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 15.1% (labeled / total samples)
- Label positive rate: 20.0%
- Label economic SNR (post-filter, label=1): 3.399
- Label effect size (post-filter Cohen's d): 4.276
- Aleatoric uncertainty fraction (|return| < cost): 11.2%

- Learnability mean CV AUC: 0.5073
- Learnability score (AUC - 0.5 * std): 0.4907
- Label balance (entropy score): 0.0000
- Combined label-quality score: 0.3435

- Probe model mean AUC: 0.5438
- Probe model stability score: 0.9404
- Probe model mean Brier score: 0.1606
- Probe global AUC (all folds combined): 0.5487
- Probe pseudo-R^2 (y vs predicted prob): -0.0126
- Probe permutation p-value (AUC): 0.005
- Probe vs baseline ΔAUC: 0.0739, ΔBrier (baseline - probe): -0.0015, ΔAP: 0.0338

- Label-quality summary score: 0.848 (Rating: Great)
- Learnability summary score: 0.343 (Rating: Bad)
- Model-robustness summary score: 0.667 (Rating: Pass)

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
- Positive labels: 4228 (20.0%)
- Negative labels: 16949

## Retention
- Pre-filter events (realized_return not NaN): 43471
- Pre-filter pos/neg (raw econ > cost): 10614 / 32857
- Post-filter labeled events: 21177
- Post-filter pos/neg (binary_label): 4228 / 16949
- Total retention: 48.7%
- Positive retention: 39.8%
- Negative retention: 51.6%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.34% / -0.81%
- Post-filter mean return (label=1/0): 1.43% / -0.72%
- Pre-filter Cohen's d: 5.276
- Post-filter Cohen's d: 4.276
- Pre-filter SNR (label=1): 2.668
- Post-filter SNR (label=1): 3.399

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 9.4%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.29%
- Mean return (label=1) minus cost: 1.13%
- Fraction of labeled events with |return| < cost: 11.2%
- Aleatoric uncertainty fraction (|return| < cost): 11.2%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=1059, win_rate=24.3%, mean_exp_ret=2.09%, Sharpe_exp=10.87
- Top 10%: n=2118, win_rate=24.1%, mean_exp_ret=2.09%, Sharpe_exp=10.67
- Top 20%: n=4236, win_rate=23.9%, mean_exp_ret=2.11%, Sharpe_exp=11.04
- Top 30%: n=6353, win_rate=24.0%, mean_exp_ret=2.11%, Sharpe_exp=11.14
- Top 40%: n=8471, win_rate=24.1%, mean_exp_ret=2.12%, Sharpe_exp=11.22

## Volatility Buckets (by volatility_1d)
- Vol low: n=7055, pos_rate=16.1%, mean_ret=-0.28%, Sharpe=-0.30
- Vol mid: n=7054, pos_rate=21.2%, mean_ret=-0.30%, Sharpe=-0.29
- Vol high: n=7055, pos_rate=22.6%, mean_ret=-0.30%, Sharpe=-0.29

## Interpretation Hints
- Coverage (15.1%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=4.276): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 3.399 → High SNR: positive-label returns are well separated from noise.
- Retention (total=48.7%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.848
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 21177
- Positive label rate: 20.0%

## Learnability
- Mean CV AUC: 0.5073
- Learnability score (AUC - 0.5 * std): 0.4907

## Entropy / Balance
- Balance score: 0.0000

## Combined Label-Quality Objective
- Combined score: 0.3435

## Interpretation Hints
- Learnability (mean AUC=0.5073): Mean CV AUC < 0.55 → very weak learnability; labels are close to random.
- Balance (entropy score=0.0000): Entropy score < 0.5 → labels are highly imbalanced or dominated by one class.
- Combined score (0.3435): Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds.

## Overall Learnability Score
- Score (0-1): 0.343
- Rating: Bad
- Summary: Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=3532, n_test=3529, AUC=0.4893, Brier=0.1845, AP=0.2144
- Fold 2: n_train=7061, n_test=3529, AUC=0.5743, Brier=0.1207, AP=0.1570
- Fold 3: n_train=10590, n_test=3529, AUC=0.5404, Brier=0.1628, AP=0.2200
- Fold 4: n_train=14119, n_test=3529, AUC=0.5355, Brier=0.1612, AP=0.2158
- Fold 5: n_train=17648, n_test=3529, AUC=0.5793, Brier=0.1736, AP=0.2791

## Summary
- Mean AUC: 0.5438 (std=0.0324)
- Mean Brier: 0.1606 (std=0.0216)
- Mean AP: 0.2173 (std=0.0387)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9404

## Interpretation Hints
- Mean AUC (0.5438): Mean CV AUC < 0.55 → robust models may still struggle; signal is weak.
- Stability score (0.9404): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1606): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5487
- Pseudo-R^2 (y vs predicted prob): -0.0126
- Pseudo-R^2 95% CI: [-0.0172, -0.0077]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.1442

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.4945
- Shuffled std AUC: 0.0039
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.5446
- Holdout Brier: 0.1700
- Holdout AP: 0.2416
- Holdout train / test: 14823 / 6354

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.5662
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4698 | Probe AUC: 0.5438 | Delta: 0.0739
- Baseline Brier: 0.1591 | Probe Brier: 0.1606 | Delta (baseline - probe): -0.0015
- Baseline AP: 0.1835 | Probe AP: 0.2173 | Delta: 0.0338

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1829
- Residual lag-1 autocorrelation: 0.4618

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5438 | LogisticRegression: 0.5191
- Comment: Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5327
- Min rolling AUC: 0.0199
- Max rolling AUC: 1.0000
- AUC at start: 0.3929
- AUC at end: 0.5415
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251210_002630.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 5.7311
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 14: mean=70.8000, std=9.5791
  - Feature 15: mean=53.6000, std=9.0244
  - Feature 1: mean=53.2000, std=7.6785
  - Feature 4: mean=32.8000, std=14.0485
  - Feature 3: mean=32.0000, std=3.1623
  - Feature 0: mean=32.0000, std=8.8769
  - Feature 2: mean=18.0000, std=6.5727
  - Feature 5: mean=17.4000, std=3.5553
  - Feature 13: mean=4.4000, std=8.8000
  - Feature 10: mean=2.8000, std=5.6000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 6
- N mislabeled candidates (confident but wrong): 0
- Estimated label noise rate: 0.000%
- False negative rate (confident): 0.000%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.667
- Rating: Pass
- Summary: Moderate robustness; some time variation or calibration issues.

### Trading-Simulation
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 1462 days
- Valid samples: 21177

## Model Calibration
- Brier Score: 0.1606
- Expected Calibration Error (ECE): 0.0285
- Max Calibration Error (MCE): 0.5611

### Calibration Interpretation
- Brier ≤ 0.18 → Well-calibrated probabilities.
- ECE ≤ 0.05 → Well-calibrated model.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- **Trades**: 8840 (6.05/day)
- **Mean Return/Trade**: -0.3190%
- **PnL/Day**: -1.9286%
- **Win Rate**: 24.1%
- **Sharpe Ratio**: -27.735
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 71
- **Avg Consecutive Losses**: 7.17
- **Win-Rate Stability**: 0.898

### Threshold 0.60
- **Trades**: 8149 (5.57/day)
- **Mean Return/Trade**: -0.3187%
- **PnL/Day**: -1.7763%
- **Win Rate**: 24.1%
- **Sharpe Ratio**: -26.602
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 70
- **Avg Consecutive Losses**: 7.04
- **Win-Rate Stability**: 0.896

### Threshold 0.65
- **Trades**: 7408 (5.07/day)
- **Mean Return/Trade**: -0.3222%
- **PnL/Day**: -1.6328%
- **Win Rate**: 24.1%
- **Sharpe Ratio**: -25.698
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 64
- **Avg Consecutive Losses**: 6.99
- **Win-Rate Stability**: 0.899

### Threshold 0.70
- **Trades**: 6613 (4.52/day)
- **Mean Return/Trade**: -0.3221%
- **PnL/Day**: -1.4568%
- **Win Rate**: 24.0%
- **Sharpe Ratio**: -24.263
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 62
- **Avg Consecutive Losses**: 7.05
- **Win-Rate Stability**: 0.902

### Threshold 0.75
- **Trades**: 5698 (3.90/day)
- **Mean Return/Trade**: -0.3211%
- **PnL/Day**: -1.2516%
- **Win Rate**: 23.9%
- **Sharpe Ratio**: -22.449
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 59
- **Avg Consecutive Losses**: 6.83
- **Win-Rate Stability**: 0.907

### Threshold 0.80
- **Trades**: 4735 (3.24/day)
- **Mean Return/Trade**: -0.3236%
- **PnL/Day**: -1.0481%
- **Win Rate**: 24.0%
- **Sharpe Ratio**: -20.667
- **Max Drawdown**: -100.00%
- **Final Equity**: 0.0000
- **Max Consecutive Losses**: 76
- **Avg Consecutive Losses**: 6.74
- **Win-Rate Stability**: 0.910

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.55 | 8840 | 6.05 | -0.319% | -1.929% | 24.1% | -27.74 | -100.0% | 71 |
| 0.60 | 8149 | 5.57 | -0.319% | -1.776% | 24.1% | -26.60 | -100.0% | 70 |
| 0.65 | 7408 | 5.07 | -0.322% | -1.633% | 24.1% | -25.70 | -100.0% | 64 |
| 0.70 | 6613 | 4.52 | -0.322% | -1.457% | 24.0% | -24.26 | -100.0% | 62 |
| 0.75 | 5698 | 3.90 | -0.321% | -1.252% | 23.9% | -22.45 | -100.0% | 59 |
| 0.80 | 4735 | 3.24 | -0.324% | -1.048% | 24.0% | -20.67 | -100.0% | 76 |

## Regime-Specific Recommended Thresholds

- **Regime** `hmm_-1.0`:
  - prob_threshold = 0.55
  - trades/day ≈ 0.293
  - mean_return ≈ 0.0955%
  - Sharpe ≈ 0.751
  - n_trades = 99
- **Regime** `hmm_2.0`:
  - prob_threshold = 0.80
  - trades/day ≈ 0.153
  - mean_return ≈ 0.1396%
  - Sharpe ≈ 0.779
  - n_trades = 55

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