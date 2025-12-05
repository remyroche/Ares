# Specialist Feature Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst
**Regime timeframe**: 15m
**Target column**: binary_label

## Data Range Analysis
- Target start date: 2024-10-30 23:15:00
- Target end date: 2025-10-13 14:00:00
- Target duration: 347 days
- Target samples: 258

## Overview
- Number of specialist features: 16
- Mean MI (CV-averaged): 0.1586
- Median MI (CV-averaged): 0.1446
- Mean R^2 (univariate): 0.0185
- Median R^2 (univariate): 0.0200
- High-MI features (MI>0.10): 13
- High-R^2 features (R^2>0.05): 0

### Probe model summary (LogReg / LGBM)
- Logistic Regression: AUC=0.440±0.096, Accuracy=0.516
- LightGBM: AUC=0.470±0.118, Accuracy=0.516

### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)

**Logistic Regression** (data range: 12 days, 192 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 60% | 35 | 2.92 | 22.9% | -0.5646% | -1.6467% | -50.13% | -23.84 |
| 70% | 28 | 2.33 | 17.9% | -0.6398% | -1.4928% | -45.44% | -28.18 |
| 80% | 26 | 2.17 | 15.4% | -0.6719% | -1.4558% | -44.32% | -30.18 |
| 90% | 15 | 1.25 | 20.0% | -0.5968% | -0.7460% | -22.71% | -18.35 |

**LightGBM** (data range: 12 days, 192 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 60% | 25 | 2.08 | 28.0% | -0.4853% | -1.0110% | -30.78% | -15.34 |
| 70% | 18 | 1.50 | 27.8% | -0.4837% | -0.7255% | -22.08% | -12.86 |
| 80% | 4 | 0.33 | 25.0% | -0.5266% | -0.1755% | -5.34% | -6.99 |
| 90% | 1 | 0.08 | 100.0% | 0.4993% | 0.0416% | 1.27% | 0.00 |


### Per-specialist model reliability vs target (MI / R^2)
- liquidity: n_features=5, MI_mean=0.2173, R^2_mean=0.0206, high_MI=4, high_R^2=0
- breakout_bounce: n_features=4, MI_mean=0.0806, R^2_mean=0.0096, high_MI=3, high_R^2=0
- path_risk: n_features=1, MI_mean=0.1819, R^2_mean=0.0127, high_MI=1, high_R^2=0
- smc: n_features=1, MI_mean=0.2344, R^2_mean=0.0322, high_MI=1, high_R^2=0
- mean_reversion: n_features=1, MI_mean=0.0000, R^2_mean=0.0000, high_MI=0, high_R^2=0
- risk: n_features=1, MI_mean=0.2662, R^2_mean=0.0472, high_MI=1, high_R^2=0

### Per-specialist data coverage
*(Target samples: 258)*
- **breakout_bounce**: n=248 (96.1% coverage), range: 2024-12-31 07:45:00 → 2025-10-13 14:00:00 ⚠️ Starts late
- **liquidity**: n=207 (80.2% coverage), range: 2025-09-30 03:15:00 → 2025-10-13 14:00:00 ⚠️ Starts late
- **mean_reversion**: n=258 (100.0% coverage), range: 2024-10-30 23:15:00 → 2025-10-13 14:00:00
- **path_risk**: n=248 (96.1% coverage), range: 2024-12-31 07:45:00 → 2025-10-13 14:00:00 ⚠️ Starts late
- **risk**: n=258 (100.0% coverage), range: 2024-10-30 23:15:00 → 2025-10-13 14:00:00
- **smc**: n=258 (100.0% coverage), range: 2024-10-30 23:15:00 → 2025-10-13 14:00:00

### Pairwise relationships between specialist models (MI / R^2)

| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |
|---------|---------|---------------|---------------|---------:|----:|
| liquidity | risk | liquidity_regime_1_prob | risk_score | 0.2441 | 0.4410 |
| breakout_bounce | smc | resistance_scalar | smc_predicted | 0.1363 | 0.0667 |
| risk | smc | risk_score | smc_predicted | 0.1021 | 0.0192 |
| path_risk | smc | path_risk_score | smc_predicted | 0.0876 | 0.0243 |
| breakout_bounce | risk | resistance_scalar | risk_score | 0.0830 | 0.1848 |
| liquidity | smc | liquidity_regime_1_prob | smc_predicted | 0.0744 | 0.0091 |
| liquidity | path_risk | liquidity_regime_1_prob | path_risk_score | 0.0593 | 0.0351 |
| path_risk | risk | path_risk_score | risk_score | 0.0382 | 0.0901 |
| mean_reversion | risk | mr_probability_dense | risk_score | 0.0000 | 0.0000 |
| mean_reversion | path_risk | mr_probability_dense | path_risk_score | 0.0000 | 0.0000 |
| liquidity | mean_reversion | liquidity_regime_1_prob | mr_probability_dense | 0.0000 | 0.0000 |
| breakout_bounce | liquidity | resistance_scalar | liquidity_regime_1_prob | 0.0000 | 0.0978 |
| breakout_bounce | path_risk | resistance_scalar | path_risk_score | 0.0000 | 0.6269 |
| breakout_bounce | mean_reversion | resistance_scalar | mr_probability_dense | 0.0000 | 0.0000 |
| mean_reversion | smc | mr_probability_dense | smc_predicted | 0.0000 | 0.0000 |

### Global stability (TimeSeriesSplit AUC)
- Mean AUC=0.440, std=0.096, stability score=0.783

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| liquidity_regime_1_prob | 0.1845 | 0.3217 | 0.234 | -0.184 | 0.0340 |
| liquidity_regime_4_prob | 0.1474 | 0.3034 | 0.136 | -0.147 | 0.0217 |
| liquidity_regime_0_prob | 0.1571 | 0.2778 | 0.099 | -0.157 | 0.0247 |
| risk_score | 0.2172 | 0.2662 | 0.423 | -0.217 | 0.0472 |
| smc_predicted | 0.1796 | 0.2344 | 0.234 | 0.180 | 0.0322 |
| path_risk_score | 0.1127 | 0.1819 | 0.243 | -0.113 | 0.0127 |
| vol_force_volatility | 0.1413 | 0.1569 | 0.626 | -0.141 | 0.0200 |
| vol_force_trend | 0.1413 | 0.1569 | 0.626 | -0.141 | 0.0200 |
| vol_force_breakout | 0.1181 | 0.1322 | 0.488 | -0.118 | 0.0139 |
| liquidity_regime_3_prob | 0.1435 | 0.1099 | 0.581 | -0.144 | 0.0206 |
| resistance_scalar | 0.0978 | 0.1075 | 0.524 | -0.098 | 0.0096 |
| support_scalar | 0.0978 | 0.1075 | 0.524 | -0.098 | 0.0096 |
| breakout_success_prob | 0.0978 | 0.1075 | 0.524 | -0.098 | 0.0096 |
| liquidity_regime_2_prob | 0.0446 | 0.0737 | 1.065 | -0.045 | 0.0020 |
| breakout_high_conf_signal | 0.0000 | 0.0000 | inf | nan | nan |
| mr_probability_dense | 0.0000 | 0.0000 | inf | 0.000 | 0.0000 |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| liquidity_regime_1_prob | 0.1845 | 0.3217 | 0.234 | -0.184 | 0.0340 |
| breakout_high_conf_signal | 0.0000 | 0.0000 | inf | nan | nan |
| risk_score | 0.2172 | 0.2662 | 0.423 | -0.217 | 0.0472 |
| smc_predicted | 0.1796 | 0.2344 | 0.234 | 0.180 | 0.0322 |
| liquidity_regime_0_prob | 0.1571 | 0.2778 | 0.099 | -0.157 | 0.0247 |
| liquidity_regime_4_prob | 0.1474 | 0.3034 | 0.136 | -0.147 | 0.0217 |
| liquidity_regime_3_prob | 0.1435 | 0.1099 | 0.581 | -0.144 | 0.0206 |
| vol_force_volatility | 0.1413 | 0.1569 | 0.626 | -0.141 | 0.0200 |
| vol_force_trend | 0.1413 | 0.1569 | 0.626 | -0.141 | 0.0200 |
| vol_force_breakout | 0.1181 | 0.1322 | 0.488 | -0.118 | 0.0139 |
| path_risk_score | 0.1127 | 0.1819 | 0.243 | -0.113 | 0.0127 |
| breakout_success_prob | 0.0978 | 0.1075 | 0.524 | -0.098 | 0.0096 |
| resistance_scalar | 0.0978 | 0.1075 | 0.524 | -0.098 | 0.0096 |
| support_scalar | 0.0978 | 0.1075 | 0.524 | -0.098 | 0.0096 |
| liquidity_regime_2_prob | 0.0446 | 0.0737 | 1.065 | -0.045 | 0.0020 |
| mr_probability_dense | 0.0000 | 0.0000 | inf | 0.000 | 0.0000 |

## Constant / Near-Constant Feature Check
⚠️ Found 7 constant features:
- resistance_scalar (val=0.6975)
- support_scalar (val=0.6975)
- breakout_success_prob (val=0.5000)
- breakout_high_conf_signal (val=0.0000)
- vol_force_volatility (val=0.0000)
- vol_force_trend (val=0.0000)
- mr_probability_dense (val=0.5615)

## Leakage diagnostics
- Suspicious features (|corr|>=0.95): 0
- Perfect-correlation features (|corr|>=0.99): 0

## Notable pairwise interactions (TreeSHAP)
- Computed on 16 features, sample_size=258

| Feature i | Feature j | Interaction strength |
|----------|----------|---------------------:|
| path_risk_score | vol_force_breakout | 1.2122e-01 |
| smc_predicted | path_risk_score | 1.2004e-01 |
| liquidity_regime_1_prob | path_risk_score | 1.1799e-01 |
| liquidity_regime_1_prob | smc_predicted | 1.0309e-01 |
| liquidity_regime_0_prob | smc_predicted | 9.5099e-02 |
| liquidity_regime_4_prob | path_risk_score | 6.7992e-02 |
| liquidity_regime_0_prob | path_risk_score | 6.6580e-02 |
| smc_predicted | liquidity_regime_2_prob | 4.7968e-02 |
| liquidity_regime_4_prob | smc_predicted | 4.7767e-02 |
| path_risk_score | liquidity_regime_2_prob | 4.7140e-02 |
| liquidity_regime_1_prob | vol_force_breakout | 3.1962e-02 |
| liquidity_regime_4_prob | liquidity_regime_2_prob | 3.0407e-02 |
| liquidity_regime_1_prob | liquidity_regime_2_prob | 2.7214e-02 |
| liquidity_regime_0_prob | vol_force_breakout | 2.6840e-02 |
| liquidity_regime_1_prob | liquidity_regime_0_prob | 2.5611e-02 |
| risk_score | path_risk_score | 2.4901e-02 |
| liquidity_regime_0_prob | liquidity_regime_2_prob | 2.4090e-02 |
| liquidity_regime_4_prob | liquidity_regime_0_prob | 2.3570e-02 |
| smc_predicted | vol_force_breakout | 2.0868e-02 |
| path_risk_score | liquidity_regime_3_prob | 2.0180e-02 |