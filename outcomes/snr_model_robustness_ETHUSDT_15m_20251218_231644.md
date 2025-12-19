# Model-Robustness Diagnostics (Probe LightGBM)

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
