# Model-Robustness Diagnostics (Probe LightGBM)

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Fold Metrics
- Fold 1: n_train=274, n_test=272, AUC=0.5371, Brier=0.1907, AP=0.2791
- Fold 2: n_train=546, n_test=272, AUC=0.5374, Brier=0.2216, AP=0.3322
- Fold 3: n_train=818, n_test=272, AUC=0.6098, Brier=0.1984, AP=0.3799
- Fold 4: n_train=1090, n_test=272, AUC=0.6000, Brier=0.2186, AP=0.4182
- Fold 5: n_train=1362, n_test=272, AUC=0.5786, Brier=0.1646, AP=0.2446

## Summary
- Mean AUC: 0.5726 (std=0.0305)
- Mean Brier: 0.1988 (std=0.0207)
- Mean AP: 0.3308 (std=0.0635)
- Stability score (1 - std(AUC)/mean(AUC)): 0.9467

## Interpretation Hints
- Mean AUC (0.5726): Mean CV AUC 0.55–0.60 → weak but potentially exploitable signal.
- Stability score (0.9467): Stability score ≥ 0.9 → highly stable performance across folds.
- Mean Brier (0.1988): Mean Brier 0.18–0.25 → moderate calibration; room for improvement.

## Advanced Robustness Diagnostics
- Global AUC (all folds combined): 0.5561
- Pseudo-R^2 (y vs predicted prob): -0.0004
- Pseudo-R^2 95% CI: [-0.0208, 0.0181]
- Permutation p-value for global AUC: 0.0050
- Model-level SNR (p_hat pos vs neg): 0.1924

## Label-Shuffle CV Sanity Check
- Shuffled mean AUC: 0.4991
- Shuffled std AUC: 0.0181
- Shuffled folds: 200

## Strict Forward Holdout
- Holdout AUC: 0.5433
- Holdout Brier: 0.1803
- Holdout AP: 0.2650
- Holdout train / test: 951 / 409

## Single-Feature Leakage Scan
- Max single-feature AUC: N/A
- AUC threshold for suspicion: N/A

## Naive Baseline Comparison (constant probability)
- Baseline AUC: 0.4778 | Probe AUC: 0.5726 | Delta: 0.0948
- Baseline Brier: 0.2000 | Probe Brier: 0.1988 | Delta (baseline - probe): 0.0012
- Baseline AP: 0.2582 | Probe AP: 0.3308 | Delta: 0.0726

## Residual Diagnostics
- Residual pattern strength (max - min mean residual across probability deciles): 0.1813
- Residual lag-1 autocorrelation: 0.2815

## Model Family Comparison (LightGBM vs LogisticRegression)
- Mean AUC LightGBM: 0.5726 | LogisticRegression: N/A
- Comment: Not applicable in label_based mode (no probe model training).

## Regime-Specific AUC Breakdown
- No regime-specific breakdown available (volatility or HMM regimes not found)

## Temporal AUC Evolution
- Mean rolling AUC: 0.5596
- Min rolling AUC: 0.3450
- Max rolling AUC: 0.9043
- AUC at start: 0.5488
- AUC at end: 0.4634
- Plot saved: `outcomes/temporal_auc_ETHUSDT_15m_20251218_005729.png`
**Interpretation**: If rolling AUC declines over time, model performance degrades on recent data.

## Feature Importance Stability Analysis
- Feature importance std (across CV folds): 3.1011
- Importance concentration (top 20 features): 100.000%
- Top features (with stability):
  - Feature 3: mean=91.8000, std=8.6116
  - Feature 0: mean=43.0000, std=11.5758
  - Feature 11: mean=39.2000, std=8.6810
  - Feature 4: mean=28.2000, std=4.2615
  - Feature 6: mean=18.6000, std=5.8856
  - Feature 1: mean=15.2000, std=4.4000
**Interpretation**: High std_importance across folds suggests unstable features (overfitting risk).

## Label Noise Estimation (Confident Learning)
- Insufficient data for label noise analysis

## Overall Model-Robustness Score
- Score (0-1): 0.627
- Rating: Pass
- Summary: Moderate robustness; some time variation or calibration issues.
