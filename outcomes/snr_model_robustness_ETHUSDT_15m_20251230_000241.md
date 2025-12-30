# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=1241, n_test=1236, AUC=0.5085, Brier=0.2466, AP=0.4205
- Fold 2: n_train=2477, n_test=1236, AUC=0.4987, Brier=0.2531, AP=0.5520
- Fold 3: n_train=3713, n_test=1236, AUC=0.4989, Brier=0.2563, AP=0.6197
- Fold 4: n_train=4949, n_test=1236, AUC=0.4973, Brier=0.2498, AP=0.4803
- Fold 5: n_train=6185, n_test=1236, AUC=0.5381, Brier=0.2445, AP=0.4118

## Summary
- Mean AUC: 0.5083 (std=0.0154)
- Mean Brier: 0.2501 (std=0.0042)
- Mean AP: 0.4969 (std=0.0793)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9697

## Interpretation Hints
- Mean AUC (0.5083): Mean CV AUC < 0.55 → robust models may still struggle; signal is weak.
- Stability score (0.9697): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.2501): Mean Brier > 0.25 → probabilities are poorly calibrated or close to random.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5019
- Pseudo-R^2 (y vs predicted prob): -0.0006
- Pseudo-R^2 95% CI: [-0.0023, 0.0002]
- Permutation p-value for global AUC: 0.2488
- Model-level SNR (p_hat pos vs neg): 0.0198

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.5003
- Shuffled std AUC: 0.0024
- Shuffled folds: 200

## Strict Forward Holdout
- Holdout AUC: 0.5266
- Holdout Brier: 0.2444
- Holdout AP: 0.3961
- Holdout train / test: 4326 / 1854

## Single-Feature Leakage Scan
- Max single-feature AUC: N/A
- AUC threshold for suspicion: N/A

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4939 | Probe AUC: 0.5083 | Delta: 0.0144
- Baseline Brier: 0.2539 | Probe Brier: 0.2501 | Delta (baseline - probe): 0.0038
- Baseline AP: 0.4850 | Probe AP: 0.4969 | Delta: 0.0119

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.0002
- Residual lag-1 autocorrelation: 0.7831

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5083 | LogisticRegression: N/A
- Comment: Not applicable in label_based mode (no probe model training).

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5087
- Min rolling AUC: 0.2200
- Max rolling AUC: 0.9574
- AUC at start: 0.2874
- AUC at end: 0.5000
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251230_000240.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 1.4219
- Importance concentration (top 20 features): 96.326%
- Top features (with stability):
  - Feature 0: mean=42.4000, std=6.1188
  - Feature 37: mean=35.2000, std=2.9257
  - Feature 42: mean=23.4000, std=4.2237
  - Feature 4: mean=21.8000, std=4.5343
  - Feature 41: mean=20.6000, std=3.6111
  - Feature 57: mean=20.4000, std=6.6513
  - Feature 39: mean=20.2000, std=4.3081
  - Feature 49: mean=17.2000, std=3.8678
  - Feature 2: mean=15.4000, std=7.1162
  - Feature 44: mean=14.8000, std=2.3152
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- Insufficient data for label noise analysis

## Overall Model-Robustness Score
- Score (0-1): 0.333
- Rating: Bad
- Summary: Probe model is weak or unstable across folds.
