# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=8786, n_test=8782, AUC=0.5622, Brier=0.2008, AP=0.3100
- Fold 2: n_train=17568, n_test=8782, AUC=0.5204, Brier=0.1665, AP=0.1700
- Fold 3: n_train=26350, n_test=8782, AUC=0.5591, Brier=0.1711, AP=0.2564
- Fold 4: n_train=35132, n_test=8782, AUC=0.5368, Brier=0.1854, AP=0.2618
- Fold 5: n_train=43914, n_test=8782, AUC=0.5246, Brier=0.1922, AP=0.2731

## Summary
- Mean AUC: 0.5406 (std=0.0172)
- Mean Brier: 0.1832 (std=0.0128)
- Mean AP: 0.2543 (std=0.0461)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9681

## Interpretation Hints
- Mean AUC (0.5406): Mean CV AUC < 0.55 → robust models may still struggle; signal is weak.
- Stability score (0.9681): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1832): Mean Brier 0.18–0.25 → moderate calibration; room for improvement.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5036
- Pseudo-R^2 (y vs predicted prob): -0.0340
- Permutation p-value for global AUC: 0.1144
- Model-level SNR (p_hat pos vs neg): 0.0008

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4634 | Probe AUC: 0.5406 | Delta: 0.0772
- Baseline Brier: 0.1784 | Probe Brier: 0.1832 | Delta (baseline - probe): -0.0048
- Baseline AP: 0.2129 | Probe AP: 0.2543 | Delta: 0.0414

## Overall Model-Robustness Score
- Score (0-1): 0.651
- Rating: Pass
- Summary: Moderate robustness; some time variation or calibration issues.