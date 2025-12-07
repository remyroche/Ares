# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=3532, n_test=3529, AUC=0.5491, Brier=0.1188, AP=0.1556
- Fold 2: n_train=7061, n_test=3529, AUC=0.7126, Brier=0.0513, AP=0.1290
- Fold 3: n_train=10590, n_test=3529, AUC=0.6504, Brier=0.0741, AP=0.1375
- Fold 4: n_train=14119, n_test=3529, AUC=0.6409, Brier=0.0857, AP=0.1613
- Fold 5: n_train=17648, n_test=3529, AUC=0.7067, Brier=0.0999, AP=0.2699

## Summary
- Mean AUC: 0.6520 (std=0.0590)
- Mean Brier: 0.0860 (std=0.0229)
- Mean AP: 0.1706 (std=0.0510)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9095

## Interpretation Hints
- Mean AUC (0.6520): Mean CV AUC 0.60–0.70 → moderate predictive power.
- Stability score (0.9095): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.0860): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.6647
- Pseudo-R^2 (y vs predicted prob): 0.0270
- Pseudo-R^2 95% CI: [0.0192, 0.0337]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.6050

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.4999
- Shuffled std AUC: 0.0201
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.6302
- Holdout Brier: 0.0965
- Holdout AP: 0.1734
- Holdout train / test: 14823 / 6354

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.6493
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4556 | Probe AUC: 0.6520 | Delta: 0.1963
- Baseline Brier: 0.0889 | Probe Brier: 0.0860 | Delta (baseline - probe): 0.0029
- Baseline AP: 0.0896 | Probe AP: 0.1706 | Delta: 0.0811

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.0486
- Residual lag-1 autocorrelation: 0.4566

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.6520 | LogisticRegression: 0.6054
- Comment: Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5802
- Min rolling AUC: 0.0000
- Max rolling AUC: 1.0000
- AUC at start: 0.4700
- AUC at end: 0.3688
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251207_090154.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 5.4572
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 17: mean=82.4000, std=4.7582
  - Feature 18: mean=62.6000, std=9.3509
  - Feature 1: mean=44.8000, std=9.6208
  - Feature 3: mean=37.2000, std=11.2143
  - Feature 4: mean=25.6000, std=11.2534
  - Feature 0: mean=24.0000, std=4.8990
  - Feature 2: mean=20.8000, std=7.3865
  - Feature 5: mean=14.4000, std=9.6042
  - Feature 9: mean=6.2000, std=12.4000
  - Feature 10: mean=3.2000, std=6.4000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 3081
- N mislabeled candidates (confident but wrong): 93
- Estimated label noise rate: 3.019%
- False negative rate (confident): 5.382%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.893
- Rating: Great
- Summary: Strong, stable probe model with consistent performance.