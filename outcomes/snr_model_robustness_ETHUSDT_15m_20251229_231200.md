# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=1241, n_test=1236, AUC=nan, Brier=nan, AP=nan
- Fold 2: n_train=2477, n_test=1236, AUC=nan, Brier=nan, AP=nan
- Fold 3: n_train=3713, n_test=1236, AUC=nan, Brier=nan, AP=nan
- Fold 4: n_train=4949, n_test=1236, AUC=nan, Brier=nan, AP=nan
- Fold 5: n_train=6185, n_test=1236, AUC=nan, Brier=nan, AP=nan

## Summary
- Mean AUC: nan (std=nan)
- Mean Brier: nan (std=nan)
- Mean AP: nan (std=nan)
- Stability score (1 - std(AUC)/mean(AUC)): 0.0000

## Interpretation Hints
- Mean AUC (nan): Mean CV AUC ≥ 0.70 → strong predictive power for the probe model.
- Stability score (0.0000): Stability score < 0.8 → performance is quite unstable across time splits.
- Mean Brier (nan): Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): N/A
- Pseudo-R^2 (y vs predicted prob): -0.0492
- Pseudo-R^2 95% CI: [-0.0600, -0.0401]
- Permutation p-value for global AUC: N/A
- Model-level SNR (p_hat pos vs neg): N/A

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: N/A
- Shuffled std AUC: N/A
- Shuffled folds: 0

## Strict Forward Holdout
- Holdout AUC: N/A
- Holdout Brier: N/A
- Holdout AP: N/A
- Holdout train / test: 0 / 0

## Single-Feature Leakage Scan
- Max single-feature AUC: N/A
- AUC threshold for suspicion: N/A

## Naive Baseline Comparison (constant probability)
- Baseline AUC: N/A | Probe AUC: N/A | Delta: N/A
- Baseline Brier: N/A | Probe Brier: N/A | Delta (baseline - probe): N/A
- Baseline AP: N/A | Probe AP: N/A | Delta: N/A

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1162
- Residual lag-1 autocorrelation: 0.8445

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: N/A | LogisticRegression: N/A
- Comment: Not applicable in label_based mode (no probe model training).

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Insufficient data for temporal AUC analysis

## Feature Importance Stability Analysis
- No feature importance data available

## Label Noise Estimation (Confident Learning)
- Insufficient data for label noise analysis

## Overall Model-Robustness Score
- Score (0-1): 0.000
- Rating: Bad
- Summary: Probe model is weak or unstable across folds.
