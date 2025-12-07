# Specialist Feature Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst
**Regime timeframe**: 15m
**Target column**: binary_label

## Data Range Analysis
- Target start date: 2024-11-03 04:30:00
- Target end date: 2025-10-31 14:30:00
- Target duration: 362 days
- Target samples: 5513

## Overview
- Number of specialist features: 10
- Mean MI (CV-averaged): 0.0478
- Median MI (CV-averaged): 0.0300
- Mean R^2 (univariate): 0.0019
- Median R^2 (univariate): 0.0004
- High-MI features (MI>0.10): 0
- High-R^2 features (R^2>0.05): 0

### Probe model summary (LogReg / LGBM)
- Logistic Regression: AUC=0.498±0.021, Accuracy=0.485
- LightGBM: AUC=0.517±0.023, Accuracy=0.513

### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)

**Logistic Regression** (data range: 265 days, 4134 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 60% | 168 | 0.63 | 48.8% | 0.0402% | 0.0255% | 0.78% | 0.46 |
| 70% | 5 | 0.02 | 80.0% | 0.7583% | 0.0143% | 0.44% | 1.63 |
| 80% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 90% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |

**LightGBM** (data range: 265 days, 4134 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 60% | 1,430 | 5.40 | 51.0% | 0.0801% | 0.4325% | 13.16% | 2.70 |
| 70% | 791 | 2.98 | 50.3% | 0.0664% | 0.1982% | 6.03% | 1.66 |
| 80% | 400 | 1.51 | 51.0% | 0.0941% | 0.1421% | 4.32% | 1.65 |
| 90% | 70 | 0.26 | 52.9% | 0.1228% | 0.0324% | 0.99% | 0.91 |


### Per-specialist model reliability vs target (MI / R^2)
- liquidity: n_features=4, MI_mean=0.0263, R^2_mean=0.0003, high_MI=0, high_R^2=0
- sr_labeling_xgb: n_features=1, MI_mean=0.0284, R^2_mean=0.0002, high_MI=0, high_R^2=0
- volume_force_breakout: n_features=1, MI_mean=0.0957, R^2_mean=0.0054, high_MI=0, high_R^2=0
- volume_force_volatility: n_features=1, MI_mean=0.0978, R^2_mean=0.0057, high_MI=0, high_R^2=0
- volume_force_trend: n_features=1, MI_mean=0.0978, R^2_mean=0.0057, high_MI=0, high_R^2=0
- smc: n_features=1, MI_mean=0.0343, R^2_mean=0.0009, high_MI=0, high_R^2=0
- risk: n_features=1, MI_mean=0.0182, R^2_mean=0.0000, high_MI=0, high_R^2=0

### Per-specialist data coverage
*(Target samples: 5513)*
- **liquidity**: n=5513 (100.0% coverage), range: 2024-11-03 04:30:00 → 2025-10-31 14:30:00
- **mean_reversion**: n=5513 (100.0% coverage), range: 2024-11-03 04:30:00 → 2025-10-31 14:30:00
- **risk**: n=5508 (99.9% coverage), range: 2024-11-05 23:15:00 → 2025-10-31 14:30:00
- **smc**: n=5513 (100.0% coverage), range: 2024-11-03 04:30:00 → 2025-10-31 14:30:00
- **sr_labeling_xgb**: n=5513 (100.0% coverage), range: 2024-11-03 04:30:00 → 2025-10-31 14:30:00
- **volume_force_breakout**: n=5195 (94.2% coverage), range: 2024-12-01 02:30:00 → 2025-10-31 14:30:00 ⚠️ Starts late
- **volume_force_trend**: n=5195 (94.2% coverage), range: 2024-12-01 02:30:00 → 2025-10-31 14:30:00 ⚠️ Starts late
- **volume_force_volatility**: n=5195 (94.2% coverage), range: 2024-12-01 02:30:00 → 2025-10-31 14:30:00 ⚠️ Starts late

### Pairwise relationships between specialist models (MI / R^2)

| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |
|---------|---------|---------------|---------------|---------:|----:|
| liquidity | volume_force_breakout | liquidity_regime_3_prob | vol_force_breakout | 0.0717 | 0.0000 |
| smc | sr_labeling_xgb | smc_predicted | sr_labeling_xgb_prob | 0.0608 | 0.0026 |
| liquidity | smc | liquidity_regime_3_prob | smc_predicted | 0.0537 | 0.0050 |
| liquidity | risk | liquidity_regime_3_prob | risk_score | 0.0431 | 0.0163 |
| sr_labeling_xgb | volume_force_trend | sr_labeling_xgb_prob | vol_force_trend | 0.0339 | 0.0058 |
| sr_labeling_xgb | volume_force_volatility | sr_labeling_xgb_prob | vol_force_volatility | 0.0339 | 0.0058 |
| sr_labeling_xgb | volume_force_breakout | sr_labeling_xgb_prob | vol_force_breakout | 0.0339 | 0.0058 |
| smc | volume_force_breakout | smc_predicted | vol_force_breakout | 0.0282 | 0.0027 |
| smc | volume_force_trend | smc_predicted | vol_force_trend | 0.0261 | 0.0023 |
| smc | volume_force_volatility | smc_predicted | vol_force_volatility | 0.0261 | 0.0023 |
| risk | volume_force_breakout | risk_score | vol_force_breakout | 0.0145 | 0.0011 |
| liquidity | sr_labeling_xgb | liquidity_regime_3_prob | sr_labeling_xgb_prob | 0.0132 | 0.0007 |
| risk | sr_labeling_xgb | risk_score | sr_labeling_xgb_prob | 0.0077 | 0.0000 |
| risk | volume_force_trend | risk_score | vol_force_trend | 0.0061 | 0.0010 |
| risk | volume_force_volatility | risk_score | vol_force_volatility | 0.0061 | 0.0010 |
| risk | smc | risk_score | smc_predicted | 0.0017 | 0.0000 |
| liquidity | volume_force_trend | liquidity_regime_3_prob | vol_force_trend | 0.0001 | 0.0002 |
| liquidity | volume_force_volatility | liquidity_regime_3_prob | vol_force_volatility | 0.0001 | 0.0002 |
| volume_force_breakout | volume_force_trend | vol_force_breakout | vol_force_trend | 0.0000 | 0.9985 |
| volume_force_breakout | volume_force_volatility | vol_force_breakout | vol_force_volatility | 0.0000 | 0.9985 |
| volume_force_trend | volume_force_volatility | vol_force_trend | vol_force_volatility | 0.0000 | 1.0000 |

### LGBM interaction probes (specialist groups)

| Groups | n_features | AUC | n_oof_samples |
|--------|-----------:|----:|--------------:|
| liquidity|risk | 5 | 0.533 | 4134 |
| liquidity|risk|smc | 6 | 0.530 | 4134 |
| risk | 1 | 0.530 | 4134 |
| liquidity | 4 | 0.530 | 4134 |
| liquidity|risk|volume_force_breakout | 6 | 0.526 | 4134 |
| liquidity|risk|sr_labeling_xgb | 6 | 0.526 | 4134 |
| liquidity|volume_force_breakout | 5 | 0.525 | 4134 |
| liquidity|volume_force_breakout|volume_force_trend | 6 | 0.525 | 4134 |
| liquidity|volume_force_breakout|volume_force_volatility | 6 | 0.525 | 4134 |
| liquidity|smc | 5 | 0.524 | 4134 |
| liquidity|sr_labeling_xgb | 5 | 0.523 | 4134 |
| liquidity|volume_force_trend | 5 | 0.523 | 4134 |
| liquidity|volume_force_volatility | 5 | 0.523 | 4134 |
| liquidity|volume_force_trend|volume_force_volatility | 6 | 0.523 | 4134 |
| liquidity|risk|volume_force_trend | 6 | 0.522 | 4134 |
| liquidity|risk|volume_force_volatility | 6 | 0.522 | 4134 |
| liquidity|mean_reversion|volume_force_breakout | 6 | 0.522 | 4134 |
| risk|sr_labeling_xgb | 2 | 0.521 | 4134 |
| liquidity|sr_labeling_xgb|volume_force_trend | 6 | 0.520 | 4134 |
| liquidity|sr_labeling_xgb|volume_force_volatility | 6 | 0.520 | 4134 |

### Global stability (TimeSeriesSplit AUC)
- Mean AUC=0.498, std=0.021, stability score=0.958

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| vol_force_volatility | 0.0754 | 0.0978 | 0.177 | -0.075 | 0.0057 |
| vol_force_trend | 0.0754 | 0.0978 | 0.177 | -0.075 | 0.0057 |
| vol_force_breakout | 0.0734 | 0.0957 | 0.183 | -0.073 | 0.0054 |
| smc_predicted | 0.0305 | 0.0343 | 0.133 | 0.030 | 0.0009 |
| liquidity_regime_3_prob | 0.0129 | 0.0315 | 0.384 | 0.013 | 0.0002 |
| sr_labeling_xgb_prob | 0.0148 | 0.0284 | 0.639 | 0.015 | 0.0002 |
| liquidity_regime_2_prob | 0.0147 | 0.0274 | 0.132 | -0.015 | 0.0002 |
| liquidity_regime_0_prob | 0.0245 | 0.0259 | 0.342 | -0.024 | 0.0006 |
| liquidity_regime_1_prob | 0.0086 | 0.0205 | 0.158 | -0.009 | 0.0001 |
| risk_score | 0.0062 | 0.0182 | 0.739 | 0.006 | 0.0000 |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| vol_force_volatility | 0.0754 | 0.0978 | 0.177 | -0.075 | 0.0057 |
| vol_force_trend | 0.0754 | 0.0978 | 0.177 | -0.075 | 0.0057 |
| vol_force_breakout | 0.0734 | 0.0957 | 0.183 | -0.073 | 0.0054 |
| smc_predicted | 0.0305 | 0.0343 | 0.133 | 0.030 | 0.0009 |
| liquidity_regime_0_prob | 0.0245 | 0.0259 | 0.342 | -0.024 | 0.0006 |
| sr_labeling_xgb_prob | 0.0148 | 0.0284 | 0.639 | 0.015 | 0.0002 |
| liquidity_regime_2_prob | 0.0147 | 0.0274 | 0.132 | -0.015 | 0.0002 |
| liquidity_regime_3_prob | 0.0129 | 0.0315 | 0.384 | 0.013 | 0.0002 |
| liquidity_regime_1_prob | 0.0086 | 0.0205 | 0.158 | -0.009 | 0.0001 |
| risk_score | 0.0062 | 0.0182 | 0.739 | 0.006 | 0.0000 |

## Constant / Near-Constant Feature Check
⚠️ Found 2 constant features:
- vol_force_volatility (val=0.0000)
- vol_force_trend (val=0.0000)

## Leakage diagnostics
- Suspicious features (|corr|>=0.95): 0
- Perfect-correlation features (|corr|>=0.99): 0

## Notable pairwise interactions (TreeSHAP)
- Interaction analysis unavailable: lightgbm or shap not available: No module named 'shap'