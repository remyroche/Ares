# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=3531, n_test=3529, AUC=0.5293, Brier=0.1678, AP=0.2284
- Fold 2: n_train=7060, n_test=3529, AUC=0.5799, Brier=0.1012, AP=0.1444
- Fold 3: n_train=10589, n_test=3529, AUC=0.5487, Brier=0.1458, AP=0.2080
- Fold 4: n_train=14118, n_test=3529, AUC=0.5597, Brier=0.1560, AP=0.2363
- Fold 5: n_train=17647, n_test=3529, AUC=0.5768, Brier=0.1620, AP=0.2799

## Summary
- Mean AUC: 0.5589 (std=0.0187)
- Mean Brier: 0.1466 (std=0.0238)
- Mean AP: 0.2194 (std=0.0442)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9666

## Interpretation Hints
- Mean AUC (0.5589): Mean CV AUC 0.55–0.60 → weak but potentially exploitable signal.
- Stability score (0.9666): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1466): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5732
- Pseudo-R^2 (y vs predicted prob): 0.0061
- Pseudo-R^2 95% CI: [-0.0006, 0.0115]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.2736

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.4998
- Shuffled std AUC: 0.0060
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.5408
- Holdout Brier: 0.1621
- Holdout AP: 0.2371
- Holdout train / test: 14823 / 6353

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.5903
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4684 | Probe AUC: 0.5589 | Delta: 0.0905
- Baseline Brier: 0.1481 | Probe Brier: 0.1466 | Delta (baseline - probe): 0.0016
- Baseline AP: 0.1660 | Probe AP: 0.2194 | Delta: 0.0534

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.0964
- Residual lag-1 autocorrelation: 0.4223

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5589 | LogisticRegression: 0.5470
- Comment: All models perform similarly poorly; target has low intrinsic predictability.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5246
- Min rolling AUC: 0.0000
- Max rolling AUC: 1.0000
- AUC at start: 0.2578
- AUC at end: 0.5492
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251209_225111.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 4.6653
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 15: mean=80.6000, std=3.8262
  - Feature 1: mean=55.4000, std=9.8102
  - Feature 16: mean=50.8000, std=11.5135
  - Feature 0: mean=36.4000, std=6.7112
  - Feature 3: mean=29.8000, std=5.8788
  - Feature 4: mean=21.4000, std=5.8515
  - Feature 2: mean=20.8000, std=2.0396
  - Feature 6: mean=15.4000, std=4.0792
  - Feature 14: mean=4.8000, std=9.6000
  - Feature 10: mean=2.8000, std=5.6000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 43
- N mislabeled candidates (confident but wrong): 5
- Estimated label noise rate: 11.628%
- False negative rate (confident): 0.158%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.686
- Rating: Pass
- Summary: Moderate robustness; some time variation or calibration issues.