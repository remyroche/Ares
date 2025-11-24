# Specialist Feature Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst
**Regime timeframe**: 1h
**Target column**: binary_label

## Overview
- Number of specialist features: 20
- Mean MI (CV-averaged): 0.0000
- Median MI (CV-averaged): 0.0000
- Mean R^2 (univariate): 0.0000
- Median R^2 (univariate): 0.0000
- High-MI features (MI>0.10): 0
- High-R^2 features (R^2>0.05): 0

### Probe model summary (LogReg / LGBM)
- Logistic Regression: AUC=0.500±0.000, Accuracy=0.422
- LightGBM: AUC=0.500±0.000, Accuracy=0.578

### Global stability (TimeSeriesSplit AUC)
- Mean AUC=0.500, std=0.000, stability score=1.000

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| risk_regime_0_prob | 0.0000 | 0.0000 | inf | nan | nan |
| risk_regime_1_prob | 0.0000 | 0.0000 | inf | nan | nan |
| risk_regime_2_prob | 0.0000 | 0.0000 | inf | nan | nan |
| risk_regime_3_prob | 0.0000 | 0.0000 | inf | nan | nan |
| risk_regime | 0.0000 | 0.0000 | inf | nan | nan |
| alpha_score_continuous | 0.0000 | 0.0000 | inf | nan | nan |
| alpha_score_continuous_ewm_3 | 0.0000 | 0.0000 | inf | nan | nan |
| alpha_score_continuous_ewm_5 | 0.0000 | 0.0000 | inf | nan | nan |
| liquidity_regime_0_prob | 0.0000 | 0.0000 | inf | nan | nan |
| liquidity_regime_1_prob | 0.0000 | 0.0000 | inf | nan | nan |
| liquidity_regime_2_prob | 0.0000 | 0.0000 | inf | nan | nan |
| liquidity_regime_3_prob | 0.0000 | 0.0000 | inf | nan | nan |
| liquidity_regime_4_prob | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_long_edge_score | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_regime_0_prob | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_regime_1_prob | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_regime_2_prob | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_short_edge_score | 0.0000 | 0.0000 | inf | nan | nan |
| is_resistance | 0.0000 | 0.0000 | inf | nan | nan |
| is_support | 0.0000 | 0.0000 | inf | nan | nan |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| risk_regime_0_prob | 0.0000 | 0.0000 | inf | nan | nan |
| risk_regime_1_prob | 0.0000 | 0.0000 | inf | nan | nan |
| risk_regime_2_prob | 0.0000 | 0.0000 | inf | nan | nan |
| risk_regime_3_prob | 0.0000 | 0.0000 | inf | nan | nan |
| risk_regime | 0.0000 | 0.0000 | inf | nan | nan |
| alpha_score_continuous | 0.0000 | 0.0000 | inf | nan | nan |
| alpha_score_continuous_ewm_3 | 0.0000 | 0.0000 | inf | nan | nan |
| alpha_score_continuous_ewm_5 | 0.0000 | 0.0000 | inf | nan | nan |
| liquidity_regime_0_prob | 0.0000 | 0.0000 | inf | nan | nan |
| liquidity_regime_1_prob | 0.0000 | 0.0000 | inf | nan | nan |
| liquidity_regime_2_prob | 0.0000 | 0.0000 | inf | nan | nan |
| liquidity_regime_3_prob | 0.0000 | 0.0000 | inf | nan | nan |
| liquidity_regime_4_prob | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_long_edge_score | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_regime_0_prob | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_regime_1_prob | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_regime_2_prob | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_short_edge_score | 0.0000 | 0.0000 | inf | nan | nan |
| is_resistance | 0.0000 | 0.0000 | inf | nan | nan |
| is_support | 0.0000 | 0.0000 | inf | nan | nan |

## Leakage diagnostics
- Suspicious features (|corr|>=0.95): 0
- Perfect-correlation features (|corr|>=0.99): 0

## Notable pairwise interactions (TreeSHAP)
- Interaction analysis unavailable: TreeSHAP interaction computation failed: 'TreeEnsemble' object has no attribute 'values'