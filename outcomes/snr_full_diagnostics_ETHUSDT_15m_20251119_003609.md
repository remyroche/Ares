# Full SNR Diagnostics Report

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst

## High-Level Summary
- Label coverage: 38.8% (labeled / total samples)
- Label positive rate: 23.7%
- Label economic SNR (post-filter, label=1): 6.103
- Label effect size (post-filter Cohen's d): 10.526
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

- Learnability mean CV AUC: 0.5328
- Learnability score (AUC - 0.5 * std): 0.5123
- Label balance (entropy score): 0.0000
- Combined label-quality score: 0.3586

- Probe model mean AUC: 0.5406
- Probe model stability score: 0.9681
- Probe model mean Brier score: 0.1832
- Probe global AUC (all folds combined): 0.5036
- Probe pseudo-R^2 (y vs predicted prob): -0.0340
- Probe permutation p-value (AUC): 0.114
- Probe vs baseline ΔAUC: 0.0772, ΔBrier (baseline - probe): -0.0048, ΔAP: 0.0414

- Label-quality summary score: 0.897 (Rating: Great)
- Learnability summary score: 0.359 (Rating: Bad)
- Model-robustness summary score: 0.651 (Rating: Pass)

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
- Labeled samples: 52696 (coverage=38.8%)
- Positive labels: 12481 (23.7%)
- Negative labels: 40215

## Retention
- Pre-filter events (realized_return not NaN): 66522
- Pre-filter pos/neg (raw econ > cost): 21569 / 44953
- Post-filter labeled events: 52696
- Post-filter pos/neg (binary_label): 12481 / 40215
- Total retention: 79.2%
- Positive retention: 57.9%
- Negative retention: 89.5%

## Economic Separation and SNR
- Pre-filter mean return (label=1/0): 1.01% / -0.66%
- Post-filter mean return (label=1/0): 1.12% / -0.73%
- Pre-filter Cohen's d: 5.761
- Post-filter Cohen's d: 10.526
- Pre-filter SNR (label=1): 2.982
- Post-filter SNR (label=1): 6.103

## Label Overlap and Cost Metrics
- Label overlap (mis-signed P&L share): 0.0%
- Transaction cost (approx per event): 0.150%
- Unconditional mean event return: -0.29%
- Mean return (label=1) minus cost: 0.97%
- Fraction of labeled events with |return| < cost: 0.0%
- Aleatoric uncertainty fraction (|return| < cost): 0.0%

## High-Probability Buckets (by meta_probability, isotonic expected returns)
- Top  5%: n=2635, win_rate=41.0%, mean_exp_ret=0.00%, Sharpe_exp=0.20
- Top 10%: n=5270, win_rate=38.5%, mean_exp_ret=0.00%, Sharpe_exp=0.14
- Top 20%: n=10540, win_rate=35.3%, mean_exp_ret=0.00%, Sharpe_exp=0.10
- Top 30%: n=15809, win_rate=32.9%, mean_exp_ret=0.00%, Sharpe_exp=0.08
- Top 40%: n=26725, win_rate=30.7%, mean_exp_ret=0.00%, Sharpe_exp=0.06

## Volatility Buckets (by volatility_1d)
- Vol low: n=17558, pos_rate=19.2%, mean_ret=-0.33%, Sharpe=-0.48
- Vol mid: n=17557, pos_rate=23.4%, mean_ret=-0.31%, Sharpe=-0.39
- Vol high: n=17558, pos_rate=28.5%, mean_ret=-0.23%, Sharpe=-0.25

## Interpretation Hints
- Coverage (38.8%): High coverage (>20%): many labeled events; check for redundancy or label noise.
- Post-filter effect size (Cohen's d=10.526): Very large separation; labels are strongly aligned with economic outcomes.
- Post-filter SNR (label=1): 6.103 → High SNR: positive-label returns are well separated from noise.
- Retention (total=79.2%): Filters keep a substantial share of events; label density is relatively high.

## Overall Label-Quality Score
- Score (0-1): 0.897
- Rating: Great
- Summary: Strong label quality with good coverage, separation and economic margins.

### Label-Learnability
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 52696
- Positive label rate: 23.7%

## Learnability
- Mean CV AUC: 0.5328
- Learnability score (AUC - 0.5 * std): 0.5123

## Entropy / Balance
- Balance score: 0.0000

## Combined Label-Quality Objective
- Combined score: 0.3586

## Interpretation Hints
- Learnability (mean AUC=0.5328): Mean CV AUC < 0.55 → very weak learnability; labels are close to random.
- Balance (entropy score=0.0000): Entropy score < 0.5 → labels are highly imbalanced or dominated by one class.
- Combined score (0.3586): Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds.

## Overall Learnability Score
- Score (0-1): 0.359
- Rating: Bad
- Summary: Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds.

### Model-Robustness
**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=8786, n_test=8782, AUC=0.5622, Brier=0.2008, AP=0.3100
- Fold 2: n_train=17568, n_test=8782, AUC=0.5204, Brier=0.1665, AP=0.1700
- Fold 3: n_train=26350, n_test=8782, AUC=0.5591, Brier=0.1711, AP=0.2564
- Fold 4: n_train=35132, n_test=8782, AUC=0.5368, Brier=0.1854, AP=0.2618
- Fold 5: n_train=43914, n_test=8782, AUC=0.5246, Brier=0.1922, AP=0.2731

## Summary
- Mean AUC: 0.5406 (std=0.0172)
- Mean Brier: 0.1832 (std=0.0128)
- Mean AP: 0.2543 (std=0.0461)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9681

## Interpretation Hints
- Mean AUC (0.5406): Mean CV AUC < 0.55 → robust models may still struggle; signal is weak.
- Stability score (0.9681): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1832): Mean Brier 0.18–0.25 → moderate calibration; room for improvement.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5036
- Pseudo-R^2 (y vs predicted prob): -0.0340
- Permutation p-value for global AUC: 0.1144
- Model-level SNR (p_hat pos vs neg): 0.0008

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4634 | Probe AUC: 0.5406 | Delta: 0.0772
- Baseline Brier: 0.1784 | Probe Brier: 0.1832 | Delta (baseline - probe): -0.0048
- Baseline AP: 0.2129 | Probe AP: 0.2543 | Delta: 0.0414

## Overall Model-Robustness Score
- Score (0-1): 0.651
- Rating: Pass
- Summary: Moderate robustness; some time variation or calibration issues.

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