# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 100.0% (labeled / total samples)
- Label positive rate: 26.6%
- Label economic SNR (post-filter, label=1): 3.378
- Label effect size (post-filter Cohen's d): 9.794
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

- Learnability mean CV AUC: 0.5640
- Learnability score (AUC - 0.5 * std): 0.5503
- Label balance (entropy score): 0.8351
- Combined label-quality score: 0.6357

- Probe model mean AUC: 0.5726
- Probe model stability score: 0.9467
- Probe model mean Brier score: 0.1988
- Probe global AUC (all folds combined): 0.5561
- Probe pseudo-R^2 (y vs predicted prob): -0.0004
- Probe permutation p-value (AUC): 0.005
- Model-level SNR (p_hat pos vs neg): -0.0004

- Label-quality summary score: 0.872 (Rating: Great)
- Learnability summary score: 0.636 (Rating: Great)
- Model-robustness summary score: 0.627 (Rating: Pass)

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
- Total samples: 1634
- Labeled samples: 1634 (coverage=100.0%)
- Positive labels: 434 (26.6%)
- Negative labels: 1200

## Retention
- Pre-filter events (realized_return not NaN): 1634
- Pre-filter pos/neg (raw econ > cost): 434 / 1200
- Post-filter labeled events: 1634
- Post-filter pos/neg (binary_label): 434 / 1200
- Total retention: 100.0%
- Positive retention: 100.0%
- Negative retention: 100.0%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.02% / -0.90%
- Post-filter mean return (label=1/0): 1.02% / -0.90%
- Pre-filter Cohen's d: 9.794
- Post-filter Cohen's d: 9.794
- Pre-filter SNR (label=1): 3.378
- Post-filter SNR (label=1): 3.378

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.39%
- Mean return (label=1) minus cost: 0.72%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)

## Enhanced Volatility Buckets (by volatility_1d)
- Vol low: n=545, pos_rate=28.4%, mean_ret=-0.34%, Sharpe=-0.45, vol_range=[0.0051, 0.0061]
- Vol mid: n=544, pos_rate=23.9%, mean_ret=-0.42%, Sharpe=-0.55, vol_range=[0.0051, 0.0061]
- Vol high: n=545, pos_rate=27.3%, mean_ret=-0.41%, Sharpe=-0.39, vol_range=[0.0051, 0.0061]

## Interpretation Hints
- Coverage (100.0%): High coverage (>20%): many labeled events; check for redundancy or label noise.
- Post-filter effect size (Cohen's d=9.794): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 3.378 → High SNR: positive-label returns are well separated from noise.
- Retention (total=100.0%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.872
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 1634
- Positive label rate: 26.6%

## Learnability
- Mean CV AUC: 0.5640
- Learnability score (AUC - 0.5 * std): 0.5503

## Entropy / Balance
- Balance score: 0.8351

## Combined Label-Quality Objective
- Combined score: 0.6357

## Interpretation Hints
- Learnability (mean AUC=0.5640): Mean CV AUC 0.55–0.60 → weak but potentially usable signal.
- Balance (entropy score=0.8351): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.6357): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 0.636
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=274, n_test=272, AUC=0.5371, Brier=0.1907, AP=0.2791
- Fold 2: n_train=546, n_test=272, AUC=0.5374, Brier=0.2216, AP=0.3322
- Fold 3: n_train=818, n_test=272, AUC=0.6098, Brier=0.1984, AP=0.3799
- Fold 4: n_train=1090, n_test=272, AUC=0.6000, Brier=0.2186, AP=0.4182
- Fold 5: n_train=1362, n_test=272, AUC=0.5786, Brier=0.1646, AP=0.2446

## Summary
- Mean AUC: 0.5726 (std=0.0305)
- Mean Brier: 0.1988 (std=0.0207)
- Mean AP: 0.3308 (std=0.0635)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9467

## Interpretation Hints
- Mean AUC (0.5726): Mean CV AUC 0.55–0.60 → weak but potentially exploitable signal.
- Stability score (0.9467): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1988): Mean Brier 0.18–0.25 → moderate calibration; room for improvement.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5561
- Pseudo-R^2 (y vs predicted prob): -0.0004
- Pseudo-R^2 95% CI: [-0.0208, 0.0181]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.1924

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.4991
- Shuffled std AUC: 0.0181
- Shuffled folds: 200

## Strict Forward Holdout
- Holdout AUC: 0.5433
- Holdout Brier: 0.1803
- Holdout AP: 0.2650
- Holdout train / test: 951 / 409

## Single-Feature Leakage Scan
- Max single-feature AUC: N/A
- AUC threshold for suspicion: N/A

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4778 | Probe AUC: 0.5726 | Delta: 0.0948
- Baseline Brier: 0.2000 | Probe Brier: 0.1988 | Delta (baseline - probe): 0.0012
- Baseline AP: 0.2582 | Probe AP: 0.3308 | Delta: 0.0726

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1813
- Residual lag-1 autocorrelation: 0.2815

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5726 | LogisticRegression: N/A
- Comment: Not applicable in label_based mode (no probe model training).

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5596
- Min rolling AUC: 0.3450
- Max rolling AUC: 0.9043
- AUC at start: 0.5488
- AUC at end: 0.4634
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251218_005729.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 3.1011
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 3: mean=91.8000, std=8.6116
  - Feature 0: mean=43.0000, std=11.5758
  - Feature 11: mean=39.2000, std=8.6810
  - Feature 4: mean=28.2000, std=4.2615
  - Feature 6: mean=18.6000, std=5.8856
  - Feature 1: mean=15.2000, std=4.4000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- Insufficient data for label noise analysis

## Overall Model-Robustness Score
- Score (0-1): 0.627
- Rating: Pass
- Summary: Moderate robustness; some time variation or calibration issues.

### Trading-Simulation
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## Overview
- Date range: 285 days
- Valid samples: 1634

## Model Calibration
- Brier Score: 0.1977
- Expected Calibration Error (ECE): 0.0638
- Max Calibration Error (MCE): 0.2036

### Calibration Interpretation
- Brier 0.18-0.25 → Moderate calibration.
- ECE 0.05-0.15 → Moderate calibration error.

## Trading Metrics by Probability Threshold

### Threshold 0.60
- Insufficient data (0 trades)

### Threshold 0.65
- Insufficient data (0 trades)

### Threshold 0.70
- Insufficient data (0 trades)

### Threshold 0.75
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
