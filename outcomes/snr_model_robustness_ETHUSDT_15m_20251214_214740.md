# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=410, n_test=407, AUC=0.6328, Brier=0.2173, AP=0.4282
- Fold 2: n_train=817, n_test=407, AUC=0.6179, Brier=0.2048, AP=0.3823
- Fold 3: n_train=1224, n_test=407, AUC=0.6316, Brier=0.1842, AP=0.4012
- Fold 4: n_train=1631, n_test=407, AUC=0.6421, Brier=0.1819, AP=0.3480
- Fold 5: n_train=2038, n_test=407, AUC=0.6765, Brier=0.1852, AP=0.5496

## Summary
- Mean AUC: 0.6402 (std=0.0197)
- Mean Brier: 0.1947 (std=0.0140)
- Mean AP: 0.4219 (std=0.0690)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9692

## Interpretation Hints
- Mean AUC (0.6402): Mean CV AUC 0.60–0.70 → moderate predictive power.
- Stability score (0.9692): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1947): Mean Brier 0.18–0.25 → moderate calibration; room for improvement.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.6359
- Pseudo-R^2 (y vs predicted prob): 0.0262
- Pseudo-R^2 95% CI: [-0.0078, 0.0598]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.4989

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5275
- Shuffled std AUC: 0.0287
- Shuffled folds: 5

## Strict Forward Holdout
- Holdout AUC: 0.6603
- Holdout Brier: 0.1830
- Holdout AP: 0.4531
- Holdout train / test: 1711 / 734

## Single-Feature Leakage Scan
- Max single-feature AUC: 0.3422
- AUC threshold for suspicion: 0.9000

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4813 | Probe AUC: 0.6402 | Delta: 0.1589
- Baseline Brier: 0.2002 | Probe Brier: 0.1947 | Delta (baseline - probe): 0.0056
- Baseline AP: 0.2709 | Probe AP: 0.4219 | Delta: 0.1510

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.2815
- Residual lag-1 autocorrelation: 0.0363

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.6402 | LogisticRegression: 0.6520
- Comment: All models perform similarly well; problem is stable and well-posed.

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.6395
- Min rolling AUC: 0.4743
- Max rolling AUC: 0.8190
- AUC at start: 0.8190
- AUC at end: 0.6655
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251214_214739.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 1.8417
- Importance concentration (top 20 features): 47.502%
- Top features (with stability):
  - Feature 43: mean=13.0000, std=3.7417
  - Feature 42: mean=9.6000, std=3.3823
  - Feature 87: mean=8.2000, std=7.3593
  - Feature 5: mean=7.0000, std=3.4059
  - Feature 35: mean=6.8000, std=5.4185
  - Feature 88: mean=6.6000, std=2.9394
  - Feature 1: mean=6.6000, std=3.0067
  - Feature 15: mean=6.6000, std=3.7736
  - Feature 20: mean=6.4000, std=4.3174
  - Feature 77: mean=6.0000, std=2.2804
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- Insufficient data for label noise analysis

## Overall Model-Robustness Score
- Score (0-1): 0.797
- Rating: Great
- Summary: Strong, stable probe model with consistent performance.