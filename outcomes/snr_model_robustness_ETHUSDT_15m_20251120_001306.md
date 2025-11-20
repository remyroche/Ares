# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=3778, n_test=3776, AUC=0.6503, Brier=0.2346, AP=0.6656
- Fold 2: n_train=7554, n_test=3776, AUC=0.5574, Brier=0.2194, AP=0.4493
- Fold 3: n_train=11330, n_test=3776, AUC=0.6806, Brier=0.2074, AP=0.7097
- Fold 4: n_train=15106, n_test=3776, AUC=0.6905, Brier=0.2093, AP=0.7409
- Fold 5: n_train=18882, n_test=3776, AUC=0.7891, Brier=0.1774, AP=0.8683

## Summary
- Mean AUC: 0.6736 (std=0.0745)
- Mean Brier: 0.2096 (std=0.0188)
- Mean AP: 0.6868 (std=0.1366)
- Stability score (1 - std(AUC)/mean(AUC)): 0.8894

## Interpretation Hints
- Mean AUC (0.6736): Mean CV AUC 0.60–0.70 → moderate predictive power.
- Stability score (0.8894): Stability score 0.8–0.9 → moderate stability; some variation across folds.
- Mean Brier (0.2096): Mean Brier 0.18–0.25 → moderate calibration; room for improvement.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.6823
- Pseudo-R^2 (y vs predicted prob): 0.1555
- Pseudo-R^2 95% CI: [0.1464, 0.1645]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.8700

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4315 | Probe AUC: 0.6736 | Delta: 0.2420
- Baseline Brier: 0.2705 | Probe Brier: 0.2096 | Delta (baseline - probe): 0.0609
- Baseline AP: 0.4267 | Probe AP: 0.6868 | Delta: 0.2601

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1676
- Residual lag-1 autocorrelation: 0.5898

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.6736 | LogisticRegression: 0.5774
- Comment: Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present.

## Overall Model-Robustness Score
- Score (0-1): 0.765
- Rating: Great
- Summary: Strong, stable probe model with consistent performance.