# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 100.0% (labeled / total samples)
- Label positive rate: 38.8%
- Label economic SNR (post-filter, label=1): 3.568
- Label effect size (post-filter Cohen's d): 8.705
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

- Learnability mean CV AUC: 0.4726
- Learnability score (AUC - 0.5 * std): 0.4063
- Label balance (entropy score): 0.9635
- Combined label-quality score: 0.5735

- Probe model mean AUC: 0.4222
- Probe model stability score: 0.6413
- Probe model mean Brier score: 0.2595
- Probe global AUC (all folds combined): 0.5414
- Probe pseudo-R^2 (y vs predicted prob): -0.0773
- Probe permutation p-value (AUC): 0.114
- Model-level SNR (p_hat pos vs neg): -0.0773

- Label-quality summary score: 0.925 (Rating: Great)
- Learnability summary score: 0.573 (Rating: Pass)
- Model-robustness summary score: 0.000 (Rating: Bad)

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
- Total samples: 536
- Labeled samples: 536 (coverage=100.0%)
- Positive labels: 208 (38.8%)
- Negative labels: 328

## Retention
- Pre-filter events (realized_return not NaN): 536
- Pre-filter pos/neg (raw econ > cost): 208 / 328
- Post-filter labeled events: 536
- Post-filter pos/neg (binary_label): 208 / 328
- Total retention: 100.0%
- Positive retention: 100.0%
- Negative retention: 100.0%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.55% / -1.50%
- Post-filter mean return (label=1/0): 1.55% / -1.50%
- Pre-filter Cohen's d: 8.705
- Post-filter Cohen's d: 8.705
- Pre-filter SNR (label=1): 3.568
- Post-filter SNR (label=1): 3.568

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.300%
- Unconditional mean event return: -0.31%
- Mean return (label=1) minus cost: 1.25%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)

## Enhanced Volatility Buckets (by volatility_1d)
- Vol low: n=179, pos_rate=59.2%, mean_ret=0.31%, Sharpe=0.23, vol_range=[0.0040, 0.0048]
- Vol mid: n=178, pos_rate=31.5%, mean_ret=-0.51%, Sharpe=-0.37, vol_range=[0.0040, 0.0048]
- Vol high: n=179, pos_rate=25.7%, mean_ret=-0.74%, Sharpe=-0.45, vol_range=[0.0040, 0.0048]

## Interpretation Hints
- Coverage (100.0%): High coverage (>20%): many labeled events; check for redundancy or label noise.
- Post-filter effect size (Cohen's d=8.705): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 3.568 → High SNR: positive-label returns are well separated from noise.
- Retention (total=100.0%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.925
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 536
- Positive label rate: 38.8%

## Learnability
- Mean CV AUC: 0.4726
- Learnability score (AUC - 0.5 * std): 0.4063

## Entropy / Balance
- Balance score: 0.9635

## Combined Label-Quality Objective
- Combined score: 0.5735

## Interpretation Hints
- Learnability (mean AUC=0.4726): Mean CV AUC < 0.55 → very weak learnability; labels are close to random.
- Balance (entropy score=0.9635): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.5735): Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.

## Overall Learnability Score
- Score (0-1): 0.573
- Rating: Pass
- Summary: Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=91, n_test=89, AUC=0.2558, Brier=0.3216, AP=0.4077
- Fold 2: n_train=180, n_test=89, AUC=0.3010, Brier=0.2965, AP=0.3579
- Fold 3: n_train=269, n_test=89, AUC=0.6900, Brier=0.2277, AP=0.7335
- Fold 4: n_train=358, n_test=89, AUC=0.4180, Brier=0.2439, AP=0.2269
- Fold 5: n_train=447, n_test=89, AUC=0.4461, Brier=0.2079, AP=0.2679

## Summary
- Mean AUC: 0.4222 (std=0.1514)
- Mean Brier: 0.2595 (std=0.0428)
- Mean AP: 0.3988 (std=0.1791)
- Stability score (1 - std(AUC)/mean(AUC)): 0.6413

## Interpretation Hints
- Mean AUC (0.4222): Mean CV AUC < 0.55 → robust models may still struggle; signal is weak.
- Stability score (0.6413): Stability score < 0.8 → performance is quite unstable across time splits.
- Mean Brier (0.2595): Mean Brier > 0.25 → probabilities are poorly calibrated or close to random.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5414
- Pseudo-R^2 (y vs predicted prob): -0.0773
- Pseudo-R^2 95% CI: [-0.1446, -0.0217]
- Permutation p-value for global AUC: 0.1144
- Model-level SNR (p_hat pos vs neg): 0.1453

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5026
- Shuffled std AUC: 0.0287
- Shuffled folds: 200

## Strict Forward Holdout
- Holdout AUC: 0.5011
- Holdout Brier: 0.2077
- Holdout AP: 0.2558
- Holdout train / test: 311 / 134

## Single-Feature Leakage Scan
- Max single-feature AUC: N/A
- AUC threshold for suspicion: N/A

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4496 | Probe AUC: 0.4222 | Delta: -0.0274
- Baseline Brier: 0.2527 | Probe Brier: 0.2595 | Delta (baseline - probe): -0.0068
- Baseline AP: 0.3813 | Probe AP: 0.3988 | Delta: 0.0175

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.4460
- Residual lag-1 autocorrelation: 0.5876

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.4222 | LogisticRegression: N/A
- Comment: Not applicable in label_based mode (no probe model training).

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.4046
- Min rolling AUC: 0.1795
- Max rolling AUC: 0.7545
- AUC at start: 0.1859
- AUC at end: 0.5954
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251218_231643.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 1.7906
- Importance concentration (top 20 features): 80.233%
- Top features (with stability):
  - Feature 9: mean=26.8000, std=9.3680
  - Feature 4: mean=14.6000, std=4.7582
  - Feature 40: mean=11.8000, std=4.2615
  - Feature 53: mean=10.8000, std=4.0200
  - Feature 49: mean=9.4000, std=3.8781
  - Feature 50: mean=9.2000, std=2.9257
  - Feature 11: mean=8.4000, std=3.7736
  - Feature 41: mean=8.4000, std=2.9394
  - Feature 12: mean=8.0000, std=5.0200
  - Feature 58: mean=7.4000, std=3.3823
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 1
- N mislabeled candidates (confident but wrong): 0
- Estimated label noise rate: 0.000%
- False negative rate (confident): 0.000%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.000
- Rating: Bad
- Summary: Probe model is weak or unstable across folds.

### Trading-Simulation
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## Overview
- Date range: 287 days
- Valid samples: 536

## Model Calibration
- Brier Score: 0.2680
- Expected Calibration Error (ECE): 0.1275
- Max Calibration Error (MCE): 0.7732

### Calibration Interpretation
- Brier > 0.25 → Poorly calibrated probabilities.
- ECE 0.05-0.15 → Moderate calibration error.

## Trading Metrics by Probability Threshold

### Threshold 0.60
- **Trades**: 132 (0.46/day)
- **Mean Return/Trade**: -0.4959%
- **PnL/Day**: -0.2281%
- **Win Rate**: 31.1%
- **Sharpe Ratio**: -3.483
- **Max Drawdown**: -54.60%
- **Final Equity**: 0.5097
- **Max Consecutive Losses**: 13
- **Avg Consecutive Losses**: 5.69
- **Win-Rate Stability**: 0.865

### Threshold 0.65
- **Trades**: 71 (0.25/day)
- **Mean Return/Trade**: -0.6022%
- **PnL/Day**: -0.1490%
- **Win Rate**: 28.2%
- **Sharpe Ratio**: -2.983
- **Max Drawdown**: -36.40%
- **Final Equity**: 0.6446
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 5.67
- **Win-Rate Stability**: 0.844

### Threshold 0.70
- **Trades**: 28 (0.10/day)
- **Mean Return/Trade**: -0.8526%
- **PnL/Day**: -0.0832%
- **Win Rate**: 21.4%
- **Sharpe Ratio**: -2.690
- **Max Drawdown**: -23.23%
- **Final Equity**: 0.7837
- **Max Consecutive Losses**: 10
- **Avg Consecutive Losses**: 4.40
- **Win-Rate Stability**: 0.862

### Threshold 0.75
- **Trades**: 23 (0.08/day)
- **Mean Return/Trade**: -0.8361%
- **PnL/Day**: -0.0670%
- **Win Rate**: 21.7%
- **Sharpe Ratio**: -2.263
- **Max Drawdown**: -20.54%
- **Final Equity**: 0.8214
- **Max Consecutive Losses**: 11
- **Avg Consecutive Losses**: 4.50
- **Win-Rate Stability**: 0.811

## Summary Table

| Threshold | Trades | Trades/Day | Mean Return | PnL/Day | Win Rate | Sharpe | Max DD | Consec Losses |
|-----------|--------|------------|-------------|---------|----------|--------|--------|---------------|
| 0.60 | 132 | 0.46 | -0.496% | -0.228% | 31.1% | -3.48 | -54.6% | 13 |
| 0.65 | 71 | 0.25 | -0.602% | -0.149% | 28.2% | -2.98 | -36.4% | 10 |
| 0.70 | 28 | 0.10 | -0.853% | -0.083% | 21.4% | -2.69 | -23.2% | 10 |
| 0.75 | 23 | 0.08 | -0.836% | -0.067% | 21.7% | -2.26 | -20.5% | 11 |

## Regime-Specific Recommended Thresholds

- **Regime** `vol_low`:
  - prob_threshold = 0.60
  - trades/day ≈ 0.089
  - mean_return ≈ 0.6408%
  - Sharpe ≈ 2.398
  - n_trades = 25

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
