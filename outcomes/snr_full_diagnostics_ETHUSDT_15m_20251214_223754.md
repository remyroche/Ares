# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 7.1% (labeled / total samples)
- Label positive rate: 27.4%
- Label economic SNR (post-filter, label=1): 1.622
- Label effect size (post-filter Cohen's d): 4.552
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

- Learnability mean CV AUC: 0.6512
- Learnability score (AUC - 0.5 * std): 0.6484
- Label balance (entropy score): 0.8472
- Combined label-quality score: 0.7081

- Probe model mean AUC: 0.6402
- Probe model stability score: 0.9692
- Probe model mean Brier score: 0.1947
- Probe global AUC (all folds combined): 0.6359
- Probe pseudo-R^2 (y vs predicted prob): 0.0262
- Probe permutation p-value (AUC): 0.005
- Probe vs baseline ΔAUC: 0.1589, ΔBrier (baseline - probe): 0.0056, ΔAP: 0.1510

- Label-quality summary score: 0.661 (Rating: Pass)
- Learnability summary score: 0.708 (Rating: Great)
- Model-robustness summary score: 0.797 (Rating: Great)

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
- Total samples: 34561
- Labeled samples: 2445 (coverage=7.1%)
- Positive labels: 670 (27.4%)
- Negative labels: 1775

## Retention
- Pre-filter events (realized_return not NaN): 6022
- Pre-filter pos/neg (raw econ > cost): 1691 / 4331
- Post-filter labeled events: 2445
- Post-filter pos/neg (binary_label): 670 / 1775
- Total retention: 40.6%
- Positive retention: 39.6%
- Negative retention: 41.0%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 0.42% / -0.40%
- Post-filter mean return (label=1/0): 0.43% / -0.42%
- Pre-filter Cohen's d: 4.083
- Post-filter Cohen's d: 4.552
- Pre-filter SNR (label=1): 1.619
- Post-filter SNR (label=1): 1.622

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.2%
- Transaction cost (approx per event): 0.100%
- Unconditional mean event return: -0.19%
- Mean return (label=1) minus cost: 0.33%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- meta_probability not available or insufficient data for bucket diagnostics.

## Enhanced Volatility Buckets (by volatility_1d)
- volatility_1d not available or insufficient data for volatility buckets.

## Interpretation Hints
- Coverage (7.1%): Moderate coverage (5–20%): typical for event-driven labeling.
- Post-filter effect size (Cohen's d=4.552): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 1.622 → High SNR: positive-label returns are well separated from noise.
- Retention (total=40.6%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.661
- Rating: Pass
- Summary: Mixed label quality; some usable signal but economic separation or coverage may be modest.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 2445
- Positive label rate: 27.4%

## Learnability
- Mean CV AUC: 0.6512
- Learnability score (AUC - 0.5 * std): 0.6484

## Entropy / Balance
- Balance score: 0.8472

## Combined Label-Quality Objective
- Combined score: 0.7081

## Interpretation Hints
- Learnability (mean AUC=0.6512): Mean CV AUC 0.60–0.70 → moderate learnability.
- Balance (entropy score=0.8472): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.7081): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 0.708
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=410, n_test=407, AUC=0.6328, Brier=0.2173, AP=0.4282
- Fold 2: n_train=817, n_test=407, AUC=0.6179, Brier=0.2048, AP=0.3823
- Fold 3: n_train=1224, n_test=407, AUC=0.6316, Brier=0.1842, AP=0.4012
- Fold 4: n_train=1631, n_test=407, AUC=0.6421, Brier=0.1819, AP=0.3480
- Fold 5: n_train=2038, n_test=407, AUC=0.6765, Brier=0.1852, AP=0.5496

## Summary
- Mean AUC: 0.6402 (std=0.0197)
- Mean Brier: 0.1947 (std=0.0140)
- Mean AP: 0.4219 (std=0.0690)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9692

## Interpretation Hints
- Mean AUC (0.6402): Mean CV AUC 0.60–0.70 → moderate predictive power.
- Stability score (0.9692): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1947): Mean Brier 0.18–0.25 → moderate calibration; room for improvement.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.6359
- Pseudo-R^2 (y vs predicted prob): 0.0262
- Pseudo-R^2 95% CI: [-0.0078, 0.0598]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.4989

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5275
- Shuffled std AUC: 0.0287
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.6603
- Holdout Brier: 0.1830
- Holdout AP: 0.4531
- Holdout train / test: 1711 / 734

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.3422
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4813 | Probe AUC: 0.6402 | Delta: 0.1589
- Baseline Brier: 0.2002 | Probe Brier: 0.1947 | Delta (baseline - probe): 0.0056
- Baseline AP: 0.2709 | Probe AP: 0.4219 | Delta: 0.1510

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.2815
- Residual lag-1 autocorrelation: 0.0363

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.6402 | LogisticRegression: 0.6520
- Comment: All models perform similarly well; problem is stable and well-posed.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.6395
- Min rolling AUC: 0.4743
- Max rolling AUC: 0.8190
- AUC at start: 0.8190
- AUC at end: 0.6655
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251214_223753.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 1.8417
- Importance concentration (top 20 features): 47.502%
- Top features (with stability):
  - Feature 43: mean=13.0000, std=3.7417
  - Feature 42: mean=9.6000, std=3.3823
  - Feature 87: mean=8.2000, std=7.3593
  - Feature 5: mean=7.0000, std=3.4059
  - Feature 35: mean=6.8000, std=5.4185
  - Feature 88: mean=6.6000, std=2.9394
  - Feature 1: mean=6.6000, std=3.0067
  - Feature 15: mean=6.6000, std=3.7736
  - Feature 20: mean=6.4000, std=4.3174
  - Feature 77: mean=6.0000, std=2.2804
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- Insufficient data for label noise analysis

## Overall Model-Robustness Score
- Score (0-1): 0.797
- Rating: Great
- Summary: Strong, stable probe model with consistent performance.

### Trading-Simulation
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long

## Overview
- Date range: 360 days
- Valid samples: 2064

## Model Calibration
- Brier Score: 0.1947
- Expected Calibration Error (ECE): 0.0512
- Max Calibration Error (MCE): 0.3455

### Calibration Interpretation
- Brier 0.18-0.25 → Moderate calibration.
- ECE 0.05-0.15 → Moderate calibration error.

## Trading Metrics by Probability Threshold

### Threshold 0.55
- Insufficient data (0 trades)

### Threshold 0.60
- Insufficient data (0 trades)

### Threshold 0.65
- Insufficient data (0 trades)

### Threshold 0.70
- Insufficient data (0 trades)

### Threshold 0.75
- Insufficient data (0 trades)

### Threshold 0.80
- Insufficient data (0 trades)

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|

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