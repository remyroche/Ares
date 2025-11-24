# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=5332, n_test=5328, AUC=0.7090, Brier=0.1861, AP=0.7342
- Fold 2: n_train=10660, n_test=5328, AUC=0.6724, Brier=0.1907, AP=0.6297
- Fold 3: n_train=15988, n_test=5328, AUC=0.8453, Brier=0.1416, AP=0.9237
- Fold 4: n_train=21316, n_test=5328, AUC=0.8151, Brier=0.1630, AP=0.8859
- Fold 5: n_train=26644, n_test=5328, AUC=0.8223, Brier=0.1554, AP=0.8758

## Summary
- Mean AUC: 0.7728 (std=0.0688)
- Mean Brier: 0.1674 (std=0.0186)
- Mean AP: 0.8099 (std=0.1107)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9110

## Interpretation Hints
- Mean AUC (0.7728): Mean CV AUC ≥ 0.70 → strong predictive power for the probe model.
- Stability score (0.9110): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1674): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.7857
- Pseudo-R^2 (y vs predicted prob): 0.3306
- Pseudo-R^2 95% CI: [0.3211, 0.3404]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 1.4059

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5017
- Shuffled std AUC: 0.0056
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.8186
- Holdout Brier: 0.1584
- Holdout AP: 0.8811
- Holdout train / test: 22380 / 9592

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.7980
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4742 | Probe AUC: 0.7728 | Delta: 0.2986
- Baseline Brier: 0.2534 | Probe Brier: 0.1674 | Delta (baseline - probe): 0.0860
- Baseline AP: 0.4962 | Probe AP: 0.8099 | Delta: 0.3136

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1008
- Residual lag-1 autocorrelation: 0.6357

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.7728 | LogisticRegression: 0.6536
- Comment: Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.6382
- Min rolling AUC: 0.0000
- Max rolling AUC: 1.0000
- AUC at start: 1.0000
- AUC at end: 1.0000
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251122_232148.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 3.5317
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 25: mean=93.2000, std=4.9558
  - Feature 2: mean=57.4000, std=16.4268
  - Feature 3: mean=50.6000, std=9.4361
  - Feature 5: mean=24.6000, std=8.4285
  - Feature 12: mean=20.2000, std=9.5582
  - Feature 26: mean=17.6000, std=7.2829
  - Feature 13: mean=17.2000, std=12.9368
  - Feature 9: mean=13.6000, std=3.8781
  - Feature 14: mean=10.8000, std=3.0594
  - Feature 4: mean=10.4000, std=1.2000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 6503
- N mislabeled candidates (confident but wrong): 0
- Estimated label noise rate: 0.000%
- False negative rate (confident): 0.000%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 1.000
- Rating: Great
- Summary: Strong, stable probe model with consistent performance.