# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=5329, n_test=5329, AUC=0.7061, Brier=0.1875, AP=0.7337
- Fold 2: n_train=10658, n_test=5329, AUC=0.6685, Brier=0.1904, AP=0.6315
- Fold 3: n_train=15987, n_test=5329, AUC=0.8460, Brier=0.1415, AP=0.9242
- Fold 4: n_train=21316, n_test=5329, AUC=0.8153, Brier=0.1621, AP=0.8871
- Fold 5: n_train=26645, n_test=5329, AUC=0.8232, Brier=0.1557, AP=0.8762

## Summary
- Mean AUC: 0.7718 (std=0.0707)
- Mean Brier: 0.1675 (std=0.0188)
- Mean AP: 0.8105 (std=0.1105)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9083

## Interpretation Hints
- Mean AUC (0.7718): Mean CV AUC ≥ 0.70 → strong predictive power for the probe model.
- Stability score (0.9083): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1675): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.7849
- Pseudo-R^2 (y vs predicted prob): 0.3302
- Pseudo-R^2 95% CI: [0.3221, 0.3388]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 1.4046

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5016
- Shuffled std AUC: 0.0067
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.8230
- Holdout Brier: 0.1574
- Holdout AP: 0.8836
- Holdout train / test: 22381 / 9593

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.7972
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4738 | Probe AUC: 0.7718 | Delta: 0.2981
- Baseline Brier: 0.2534 | Probe Brier: 0.1675 | Delta (baseline - probe): 0.0859
- Baseline AP: 0.4952 | Probe AP: 0.8105 | Delta: 0.3153

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1166
- Residual lag-1 autocorrelation: 0.6368

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.7718 | LogisticRegression: 0.6517
- Comment: Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.6389
- Min rolling AUC: 0.0000
- Max rolling AUC: 1.0000
- AUC at start: 1.0000
- AUC at end: 0.9531
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251123_001340.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 4.0962
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 25: mean=101.8000, std=16.4365
  - Feature 3: mean=55.6000, std=13.1240
  - Feature 2: mean=52.4000, std=15.4997
  - Feature 5: mean=27.0000, std=9.4021
  - Feature 12: mean=18.4000, std=10.1311
  - Feature 9: mean=14.8000, std=2.9257
  - Feature 13: mean=14.2000, std=8.4475
  - Feature 14: mean=13.2000, std=5.2307
  - Feature 26: mean=11.0000, std=5.8652
  - Feature 4: mean=10.4000, std=2.4166
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 6494
- N mislabeled candidates (confident but wrong): 1
- Estimated label noise rate: 0.015%
- False negative rate (confident): 0.000%
- False positive rate (confident): 0.008%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 1.000
- Rating: Great
- Summary: Strong, stable probe model with consistent performance.