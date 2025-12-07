# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=3532, n_test=3529, AUC=0.5349, Brier=0.1645, AP=0.2208
- Fold 2: n_train=7061, n_test=3529, AUC=0.5984, Brier=0.0983, AP=0.1510
- Fold 3: n_train=10590, n_test=3529, AUC=0.5776, Brier=0.1398, AP=0.2098
- Fold 4: n_train=14119, n_test=3529, AUC=0.5794, Brier=0.1419, AP=0.2142
- Fold 5: n_train=17648, n_test=3529, AUC=0.6093, Brier=0.1578, AP=0.2852

## Summary
- Mean AUC: 0.5799 (std=0.0255)
- Mean Brier: 0.1405 (std=0.0231)
- Mean AP: 0.2162 (std=0.0426)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9561

## Interpretation Hints
- Mean AUC (0.5799): Mean CV AUC 0.55–0.60 → weak but potentially exploitable signal.
- Stability score (0.9561): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1405): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5931
- Pseudo-R^2 (y vs predicted prob): 0.0094
- Pseudo-R^2 95% CI: [0.0040, 0.0143]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.3208

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.4968
- Shuffled std AUC: 0.0132
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.5743
- Holdout Brier: 0.1535
- Holdout AP: 0.2327
- Holdout train / test: 14823 / 6354

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.5985
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4732 | Probe AUC: 0.5799 | Delta: 0.1067
- Baseline Brier: 0.1425 | Probe Brier: 0.1405 | Delta (baseline - probe): 0.0020
- Baseline AP: 0.1592 | Probe AP: 0.2162 | Delta: 0.0570

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.0860
- Residual lag-1 autocorrelation: 0.3951

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5799 | LogisticRegression: 0.5696
- Comment: All models perform similarly poorly; target has low intrinsic predictability.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5493
- Min rolling AUC: 0.0102
- Max rolling AUC: 1.0000
- AUC at start: 0.4058
- AUC at end: 0.5536
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251207_124726.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 5.1963
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 15: mean=82.6000, std=11.9766
  - Feature 16: mean=54.2000, std=9.9880
  - Feature 1: mean=50.0000, std=10.3537
  - Feature 3: mean=40.2000, std=1.9391
  - Feature 4: mean=28.4000, std=5.3889
  - Feature 0: mean=26.0000, std=4.6904
  - Feature 2: mean=14.8000, std=7.2222
  - Feature 6: mean=13.0000, std=3.5777
  - Feature 14: mean=4.0000, std=8.0000
  - Feature 11: mean=3.4000, std=6.8000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 143
- N mislabeled candidates (confident but wrong): 17
- Estimated label noise rate: 11.888%
- False negative rate (confident): 0.563%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.733
- Rating: Great
- Summary: Strong, stable probe model with consistent performance.