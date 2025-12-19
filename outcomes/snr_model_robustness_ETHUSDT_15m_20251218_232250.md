# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=91, n_test=89, AUC=0.2472, Brier=0.3164, AP=0.4168
- Fold 2: n_train=180, n_test=89, AUC=0.4295, Brier=0.2952, AP=0.4041
- Fold 3: n_train=269, n_test=89, AUC=0.6875, Brier=0.2259, AP=0.6785
- Fold 4: n_train=358, n_test=89, AUC=0.4529, Brier=0.2499, AP=0.2351
- Fold 5: n_train=447, n_test=89, AUC=0.4755, Brier=0.2055, AP=0.2359

## Summary
- Mean AUC: 0.4585 (std=0.1402)
- Mean Brier: 0.2586 (std=0.0416)
- Mean AP: 0.3941 (std=0.1623)
- Stability score (1 - std(AUC)/mean(AUC)): 0.6943

## Interpretation Hints
- Mean AUC (0.4585): Mean CV AUC < 0.55 → robust models may still struggle; signal is weak.
- Stability score (0.6943): Stability score < 0.8 → performance is quite unstable across time splits.
- Mean Brier (0.2586): Mean Brier > 0.25 → probabilities are poorly calibrated or close to random.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5537
- Pseudo-R^2 (y vs predicted prob): -0.0795
- Pseudo-R^2 95% CI: [-0.1562, -0.0158]
- Permutation p-value for global AUC: 0.0498
- Model-level SNR (p_hat pos vs neg): 0.1811

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5041
- Shuffled std AUC: 0.0287
- Shuffled folds: 200

## Strict Forward Holdout
- Holdout AUC: 0.4781
- Holdout Brier: 0.2140
- Holdout AP: 0.2436
- Holdout train / test: 311 / 134

## Single-Feature Leakage Scan
- Max single-feature AUC: N/A
- AUC threshold for suspicion: N/A

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4475 | Probe AUC: 0.4585 | Delta: 0.0111
- Baseline Brier: 0.2502 | Probe Brier: 0.2586 | Delta (baseline - probe): -0.0084
- Baseline AP: 0.3727 | Probe AP: 0.3941 | Delta: 0.0214

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.5468
- Residual lag-1 autocorrelation: 0.5864

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.4585 | LogisticRegression: N/A
- Comment: Not applicable in label_based mode (no probe model training).

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.4532
- Min rolling AUC: 0.2165
- Max rolling AUC: 0.7995
- AUC at start: 0.2165
- AUC at end: 0.6371
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251218_232249.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 1.8360
- Importance concentration (top 20 features): 81.132%
- Top features (with stability):
  - Feature 9: mean=29.2000, std=9.9880
  - Feature 4: mean=13.6000, std=4.2237
  - Feature 11: mean=12.0000, std=5.4037
  - Feature 40: mean=11.8000, std=4.1183
  - Feature 50: mean=10.8000, std=0.7483
  - Feature 53: mean=10.6000, std=3.6111
  - Feature 49: mean=10.2000, std=6.2097
  - Feature 12: mean=8.4000, std=5.6071
  - Feature 58: mean=8.2000, std=3.0594
  - Feature 41: mean=7.2000, std=2.1354
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 7
- N mislabeled candidates (confident but wrong): 2
- Estimated label noise rate: 28.571%
- False negative rate (confident): 0.000%
- False positive rate (confident): 0.746%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.000
- Rating: Bad
- Summary: Probe model is weak or unstable across folds.
