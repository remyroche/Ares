# Specialist Feature Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst
**Regime timeframe**: 15m
**Target column**: binary_label

## Data Range Analysis
- Target start date: 2024-11-30 00:00:00
- Target end date: 2025-11-29 17:45:00
- Target duration: 364 days
- Target samples: 482

## Overview
- Number of specialist features: 14
- Mean MI (CV-averaged): 0.2443
- Median MI (CV-averaged): 0.3228
- Mean R^2 (univariate): 0.0682
- Median R^2 (univariate): 0.0622
- High-MI features (MI>0.10): 10
- High-R^2 features (R^2>0.05): 6

### Probe model summary (LogReg / LGBM)
- Logistic Regression: AUC=0.460±0.106, Accuracy=0.553
- LightGBM: AUC=0.510±0.161, Accuracy=0.506

### Per-specialist model reliability vs target (MI / R^2)
- liquidity: n_features=5, MI_mean=0.3231, R^2_mean=0.0729, high_MI=4, high_R^2=2
- breakout_bounce: n_features=4, MI_mean=0.2771, R^2_mean=0.0800, high_MI=3, high_R^2=3
- path_risk: n_features=1, MI_mean=0.1603, R^2_mean=0.0217, high_MI=1, high_R^2=0
- smc: n_features=1, MI_mean=0.2173, R^2_mean=0.0180, high_MI=1, high_R^2=0
- mean_reversion: n_features=1, MI_mean=0.0000, R^2_mean=0.0000, high_MI=0, high_R^2=0
- risk: n_features=1, MI_mean=0.3191, R^2_mean=0.1720, high_MI=1, high_R^2=1

### Per-specialist data coverage
*(Target samples: 482)*
- **breakout_bounce**: n=465 (96.5% coverage), range: 2024-12-30 23:30:00 → 2025-11-29 17:45:00 ⚠️ Starts late
- **liquidity**: n=335 (69.5% coverage), range: 2025-08-30 23:30:00 → 2025-11-29 17:45:00 ⚠️ Starts late
- **mean_reversion**: n=482 (100.0% coverage), range: 2024-11-30 00:00:00 → 2025-11-29 17:45:00
- **path_risk**: n=482 (100.0% coverage), range: 2024-11-30 00:00:00 → 2025-11-29 17:45:00
- **risk**: n=482 (100.0% coverage), range: 2024-11-30 00:00:00 → 2025-11-29 17:45:00
- **smc**: n=182 (37.8% coverage), range: 2025-10-05 23:30:00 → 2025-11-29 17:45:00 ⚠️ Starts late ⚠️ Low coverage (<50%)

### Pairwise relationships between specialist models (MI / R^2)

| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |
|---------|---------|---------------|---------------|---------:|----:|
| liquidity | risk | liquidity_regime_1_prob | risk_score | 0.2882 | 0.2899 |
| breakout_bounce | risk | support_scalar | risk_score | 0.1758 | 0.2355 |
| risk | smc | risk_score | smc_predicted | 0.1458 | 0.1700 |
| path_risk | risk | path_risk_score | risk_score | 0.0623 | 0.0179 |
| breakout_bounce | liquidity | support_scalar | liquidity_regime_1_prob | 0.0590 | 0.2912 |
| liquidity | smc | liquidity_regime_1_prob | smc_predicted | 0.0555 | 0.2090 |
| liquidity | path_risk | liquidity_regime_1_prob | path_risk_score | 0.0253 | 0.0021 |
| breakout_bounce | path_risk | support_scalar | path_risk_score | 0.0138 | 0.0003 |
| path_risk | smc | path_risk_score | smc_predicted | 0.0106 | 0.0009 |
| mean_reversion | path_risk | mr_probability_dense | path_risk_score | 0.0000 | nan |
| breakout_bounce | mean_reversion | support_scalar | mr_probability_dense | 0.0000 | nan |
| mean_reversion | risk | mr_probability_dense | risk_score | 0.0000 | nan |
| liquidity | mean_reversion | liquidity_regime_1_prob | mr_probability_dense | 0.0000 | nan |
| breakout_bounce | smc | support_scalar | smc_predicted | 0.0000 | 0.7221 |
| mean_reversion | smc | mr_probability_dense | smc_predicted | 0.0000 | nan |

### Global stability (TimeSeriesSplit AUC)
- Mean AUC=0.460, std=0.106, stability score=0.770

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| liquidity_regime_1_prob | 0.4173 | 0.5107 | 0.026 | -0.417 | 0.1742 |
| liquidity_regime_4_prob | 0.2819 | 0.3783 | 0.077 | -0.282 | 0.0795 |
| support_scalar | 0.2843 | 0.3735 | 0.000 | -0.284 | 0.0808 |
| resistance_scalar | 0.2820 | 0.3674 | 0.000 | -0.282 | 0.0795 |
| breakout_success_prob | 0.2820 | 0.3674 | 0.000 | -0.282 | 0.0795 |
| liquidity_regime_2_prob | 0.2120 | 0.3285 | 0.198 | -0.212 | 0.0449 |
| liquidity_regime_0_prob | 0.2080 | 0.3265 | 0.138 | -0.208 | 0.0433 |
| risk_score | 0.4147 | 0.3191 | 0.557 | -0.415 | 0.1720 |
| smc_predicted | 0.1341 | 0.2173 | 0.000 | -0.134 | 0.0180 |
| path_risk_score | 0.1473 | 0.1603 | 0.210 | -0.147 | 0.0217 |
| liquidity_regime_3_prob | 0.1506 | 0.0715 | 0.729 | -0.151 | 0.0227 |
| breakout_high_conf_signal | 0.0000 | 0.0000 | inf | nan | nan |
| vol_force_scalar | 0.0530 | 0.0000 | inf | 0.053 | 0.0028 |
| mr_probability_dense | 0.0000 | 0.0000 | inf | nan | nan |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| liquidity_regime_1_prob | 0.4173 | 0.5107 | 0.026 | -0.417 | 0.1742 |
| support_scalar | 0.2843 | 0.3735 | 0.000 | -0.284 | 0.0808 |
| resistance_scalar | 0.2820 | 0.3674 | 0.000 | -0.282 | 0.0795 |
| breakout_success_prob | 0.2820 | 0.3674 | 0.000 | -0.282 | 0.0795 |
| liquidity_regime_4_prob | 0.2819 | 0.3783 | 0.077 | -0.282 | 0.0795 |
| liquidity_regime_2_prob | 0.2120 | 0.3285 | 0.198 | -0.212 | 0.0449 |
| liquidity_regime_0_prob | 0.2080 | 0.3265 | 0.138 | -0.208 | 0.0433 |
| liquidity_regime_3_prob | 0.1506 | 0.0715 | 0.729 | -0.151 | 0.0227 |
| breakout_high_conf_signal | 0.0000 | 0.0000 | inf | nan | nan |
| path_risk_score | 0.1473 | 0.1603 | 0.210 | -0.147 | 0.0217 |
| smc_predicted | 0.1341 | 0.2173 | 0.000 | -0.134 | 0.0180 |
| vol_force_scalar | 0.0530 | 0.0000 | inf | 0.053 | 0.0028 |
| mr_probability_dense | 0.0000 | 0.0000 | inf | nan | nan |
| risk_score | 0.4147 | 0.3191 | 0.557 | -0.415 | 0.1720 |

## Leakage diagnostics
- Suspicious features (|corr|>=0.95): 0
- Perfect-correlation features (|corr|>=0.99): 0

## Notable pairwise interactions (TreeSHAP)
- Computed on 14 features, sample_size=482

| Feature i | Feature j | Interaction strength |
|----------|----------|---------------------:|
| smc_predicted | path_risk_score | 2.2028e-01 |
| risk_score | path_risk_score | 1.6546e-01 |
| liquidity_regime_1_prob | path_risk_score | 1.3340e-01 |
| liquidity_regime_0_prob | path_risk_score | 1.1714e-01 |
| liquidity_regime_0_prob | smc_predicted | 1.0499e-01 |
| liquidity_regime_0_prob | risk_score | 1.0426e-01 |
| liquidity_regime_4_prob | path_risk_score | 9.8916e-02 |
| liquidity_regime_1_prob | risk_score | 8.6412e-02 |
| liquidity_regime_2_prob | path_risk_score | 8.3413e-02 |
| support_scalar | path_risk_score | 8.3250e-02 |
| liquidity_regime_1_prob | liquidity_regime_4_prob | 7.8555e-02 |
| liquidity_regime_1_prob | smc_predicted | 7.3088e-02 |
| path_risk_score | liquidity_regime_3_prob | 6.3688e-02 |
| support_scalar | smc_predicted | 5.8354e-02 |
| smc_predicted | liquidity_regime_3_prob | 4.5779e-02 |
| liquidity_regime_1_prob | liquidity_regime_0_prob | 4.1957e-02 |
| liquidity_regime_2_prob | smc_predicted | 4.0804e-02 |
| resistance_scalar | smc_predicted | 3.8946e-02 |
| liquidity_regime_4_prob | smc_predicted | 3.6708e-02 |
| liquidity_regime_4_prob | risk_score | 3.4112e-02 |