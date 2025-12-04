# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=702, n_test=697, AUC=0.5444, Brier=0.2163, AP=0.3087
- Fold 2: n_train=1399, n_test=697, AUC=0.5956, Brier=0.1748, AP=0.2887
- Fold 3: n_train=2096, n_test=697, AUC=0.5854, Brier=0.2102, AP=0.3077
- Fold 4: n_train=2793, n_test=697, AUC=0.5331, Brier=0.2274, AP=0.3703
- Fold 5: n_train=3490, n_test=697, AUC=0.5700, Brier=0.1601, AP=0.2468

## Summary
- Mean AUC: 0.5657 (std=0.0238)
- Mean Brier: 0.1978 (std=0.0258)
- Mean AP: 0.3045 (std=0.0398)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9580

## Interpretation Hints
- Mean AUC (0.5657): Mean CV AUC 0.55–0.60 → weak but potentially exploitable signal.
- Stability score (0.9580): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1978): Mean Brier 0.18–0.25 → moderate calibration; room for improvement.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5560
- Pseudo-R^2 (y vs predicted prob): -0.0339
- Pseudo-R^2 95% CI: [-0.0529, -0.0126]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.1939

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5122
- Shuffled std AUC: 0.0404
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.5623
- Holdout Brier: 0.1918
- Holdout AP: 0.2954
- Holdout train / test: 2930 / 1257

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.5961
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4523 | Probe AUC: 0.5657 | Delta: 0.1134
- Baseline Brier: 0.1926 | Probe Brier: 0.1978 | Delta (baseline - probe): -0.0052
- Baseline AP: 0.2346 | Probe AP: 0.3045 | Delta: 0.0699

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.3033
- Residual lag-1 autocorrelation: 0.2519

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5657 | LogisticRegression: 0.5700
- Comment: All models perform similarly poorly; target has low intrinsic predictability.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5417
- Min rolling AUC: 0.2021
- Max rolling AUC: 0.9375
- AUC at start: 0.5559
- AUC at end: 0.6350
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251203_213032.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 4.5810
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 12: mean=51.4000, std=7.5525
  - Feature 13: mean=38.2000, std=6.3056
  - Feature 2: mean=33.4000, std=4.8826
  - Feature 7: mean=27.0000, std=9.6747
  - Feature 0: mean=26.2000, std=6.8527
  - Feature 10: mean=18.4000, std=7.7872
  - Feature 6: mean=15.6000, std=4.0792
  - Feature 9: mean=15.2000, std=5.7061
  - Feature 8: mean=13.0000, std=5.0990
  - Feature 4: mean=12.2000, std=4.4900
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 31
- N mislabeled candidates (confident but wrong): 4
- Estimated label noise rate: 12.903%
- False negative rate (confident): 0.445%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.617
- Rating: Pass
- Summary: Moderate robustness; some time variation or calibration issues.