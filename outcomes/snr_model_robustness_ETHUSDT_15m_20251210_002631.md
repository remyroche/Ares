# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=3532, n_test=3529, AUC=0.4893, Brier=0.1845, AP=0.2144
- Fold 2: n_train=7061, n_test=3529, AUC=0.5743, Brier=0.1207, AP=0.1570
- Fold 3: n_train=10590, n_test=3529, AUC=0.5404, Brier=0.1628, AP=0.2200
- Fold 4: n_train=14119, n_test=3529, AUC=0.5355, Brier=0.1612, AP=0.2158
- Fold 5: n_train=17648, n_test=3529, AUC=0.5793, Brier=0.1736, AP=0.2791

## Summary
- Mean AUC: 0.5438 (std=0.0324)
- Mean Brier: 0.1606 (std=0.0216)
- Mean AP: 0.2173 (std=0.0387)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9404

## Interpretation Hints
- Mean AUC (0.5438): Mean CV AUC < 0.55 → robust models may still struggle; signal is weak.
- Stability score (0.9404): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1606): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5487
- Pseudo-R^2 (y vs predicted prob): -0.0126
- Pseudo-R^2 95% CI: [-0.0172, -0.0077]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.1442

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.4945
- Shuffled std AUC: 0.0039
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.5446
- Holdout Brier: 0.1700
- Holdout AP: 0.2416
- Holdout train / test: 14823 / 6354

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.5662
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4698 | Probe AUC: 0.5438 | Delta: 0.0739
- Baseline Brier: 0.1591 | Probe Brier: 0.1606 | Delta (baseline - probe): -0.0015
- Baseline AP: 0.1835 | Probe AP: 0.2173 | Delta: 0.0338

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1829
- Residual lag-1 autocorrelation: 0.4618

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5438 | LogisticRegression: 0.5191
- Comment: Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5327
- Min rolling AUC: 0.0199
- Max rolling AUC: 1.0000
- AUC at start: 0.3929
- AUC at end: 0.5415
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251210_002630.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 5.7311
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 14: mean=70.8000, std=9.5791
  - Feature 15: mean=53.6000, std=9.0244
  - Feature 1: mean=53.2000, std=7.6785
  - Feature 4: mean=32.8000, std=14.0485
  - Feature 3: mean=32.0000, std=3.1623
  - Feature 0: mean=32.0000, std=8.8769
  - Feature 2: mean=18.0000, std=6.5727
  - Feature 5: mean=17.4000, std=3.5553
  - Feature 13: mean=4.4000, std=8.8000
  - Feature 10: mean=2.8000, std=5.6000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- N confident predictions (confidence ≥ 0.9): 6
- N mislabeled candidates (confident but wrong): 0
- Estimated label noise rate: 0.000%
- False negative rate (confident): 0.000%
- False positive rate (confident): 0.000%
**Interpretation**: High noise rate (>5%) suggests labels may be mislabeled; consider tightening TPSL geometry.

## Overall Model-Robustness Score
- Score (0-1): 0.667
- Rating: Pass
- Summary: Moderate robustness; some time variation or calibration issues.