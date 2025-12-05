# Specialist Feature Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst
**Regime timeframe**: 1h
**Target column**: target_long

## Data Range Analysis
- Target start date: 2024-11-29 23:00:00
- Target end date: 2025-11-29 22:00:00
- Target duration: 364 days
- Target samples: 2440

## Overview
- Number of specialist features: 17
- Mean MI (CV-averaged): 0.0352
- Median MI (CV-averaged): 0.0285
- Mean R^2 (univariate): 0.0017
- Median R^2 (univariate): 0.0004
- High-MI features (MI>0.10): 1
- High-R^2 features (R^2>0.05): 0

### Probe model summary (LogReg / LGBM)
- Logistic Regression: not available
- LightGBM: not available

### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)

**Logistic Regression** (data range: 243 days, 465 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 60% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 70% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 80% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 90% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |

**LightGBM** (data range: 243 days, 465 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 60% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 70% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 80% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 90% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |


### Per-specialist model reliability vs target (MI / R^2)
- liquidity: n_features=5, MI_mean=0.0324, R^2_mean=0.0010, high_MI=0, high_R^2=0
- breakout_bounce: n_features=4, MI_mean=0.0061, R^2_mean=0.0001, high_MI=0, high_R^2=0
- path_risk: n_features=1, MI_mean=0.0333, R^2_mean=0.0023, high_MI=0, high_R^2=0
- macro_trend: n_features=1, MI_mean=0.0921, R^2_mean=0.0026, high_MI=0, high_R^2=0
- smc: n_features=1, MI_mean=0.1107, R^2_mean=0.0109, high_MI=1, high_R^2=0
- mean_reversion: n_features=1, MI_mean=0.0385, R^2_mean=0.0006, high_MI=0, high_R^2=0
- risk: n_features=1, MI_mean=0.0531, R^2_mean=0.0032, high_MI=0, high_R^2=0

### Per-specialist data coverage
*(Target samples: 2440)*
- **breakout_bounce**: n=2342 (96.0% coverage), range: 2024-12-30 23:15:00 → 2025-11-29 22:00:00 ⚠️ Starts late
- **liquidity**: n=1567 (64.2% coverage), range: 2025-08-30 22:00:00 → 2025-11-29 22:00:00 ⚠️ Starts late
- **macro_trend**: n=2440 (100.0% coverage), range: 2024-11-29 23:00:00 → 2025-11-29 22:00:00
- **mean_reversion**: n=2440 (100.0% coverage), range: 2024-11-29 23:00:00 → 2025-11-29 22:00:00
- **path_risk**: n=2342 (96.0% coverage), range: 2024-12-30 23:15:00 → 2025-11-29 22:00:00 ⚠️ Starts late
- **risk**: n=2440 (100.0% coverage), range: 2024-11-29 23:00:00 → 2025-11-29 22:00:00
- **smc**: n=2440 (100.0% coverage), range: 2024-11-29 23:00:00 → 2025-11-29 22:00:00

### Pairwise relationships between specialist models (MI / R^2)

| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |
|---------|---------|---------------|---------------|---------:|----:|
| breakout_bounce | mean_reversion | support_scalar | mr_probability_dense | 0.3480 | 0.5022 |
| mean_reversion | path_risk | mr_probability_dense | path_risk_score | 0.2829 | 0.3359 |
| path_risk | risk | path_risk_score | risk_score | 0.2066 | 0.0923 |
| mean_reversion | risk | mr_probability_dense | risk_score | 0.1807 | 0.0367 |
| macro_trend | mean_reversion | macro_trend_score_continuous | mr_probability_dense | 0.1578 | 0.0257 |
| liquidity | risk | liquidity_regime_2_prob | risk_score | 0.1554 | 0.1065 |
| macro_trend | path_risk | macro_trend_score_continuous | path_risk_score | 0.1140 | 0.0523 |
| breakout_bounce | macro_trend | support_scalar | macro_trend_score_continuous | 0.1037 | 0.0435 |
| liquidity | smc | liquidity_regime_2_prob | smc_predicted | 0.0947 | 0.0415 |
| macro_trend | risk | macro_trend_score_continuous | risk_score | 0.0742 | 0.0214 |
| breakout_bounce | risk | support_scalar | risk_score | 0.0661 | 0.0223 |
| liquidity | macro_trend | liquidity_regime_2_prob | macro_trend_score_continuous | 0.0620 | 0.0128 |
| liquidity | path_risk | liquidity_regime_2_prob | path_risk_score | 0.0509 | 0.0327 |
| liquidity | mean_reversion | liquidity_regime_2_prob | mr_probability_dense | 0.0495 | 0.0105 |
| breakout_bounce | smc | support_scalar | smc_predicted | 0.0472 | 0.0141 |
| risk | smc | risk_score | smc_predicted | 0.0373 | 0.0053 |
| path_risk | smc | path_risk_score | smc_predicted | 0.0302 | 0.0086 |
| breakout_bounce | path_risk | support_scalar | path_risk_score | 0.0225 | 0.3926 |
| mean_reversion | smc | mr_probability_dense | smc_predicted | 0.0156 | 0.0006 |
| macro_trend | smc | macro_trend_score_continuous | smc_predicted | 0.0072 | 0.0000 |
| breakout_bounce | liquidity | support_scalar | liquidity_regime_2_prob | 0.0000 | 0.0210 |

### LGBM interaction probes (specialist groups)

| Groups | n_features | AUC | n_oof_samples |
|--------|-----------:|----:|--------------:|
| liquidity|risk | 6 | 0.597 | 2030 |
| liquidity|mean_reversion|risk | 7 | 0.590 | 2030 |
| breakout_bounce|liquidity|risk | 10 | 0.589 | 2030 |
| liquidity|macro_trend|risk | 7 | 0.549 | 2030 |
| risk | 1 | 0.543 | 2030 |
| mean_reversion|risk | 2 | 0.543 | 2030 |
| liquidity|risk|smc | 7 | 0.536 | 2030 |
| liquidity|mean_reversion|path_risk | 7 | 0.534 | 2030 |
| breakout_bounce|liquidity|path_risk | 10 | 0.534 | 2030 |
| liquidity|path_risk | 6 | 0.534 | 2030 |
| breakout_bounce|liquidity | 9 | 0.531 | 2030 |
| breakout_bounce|liquidity|mean_reversion | 10 | 0.531 | 2030 |
| macro_trend|risk | 2 | 0.529 | 2030 |
| liquidity|mean_reversion | 6 | 0.528 | 2030 |
| liquidity | 5 | 0.514 | 2030 |
| breakout_bounce|liquidity|smc | 10 | 0.513 | 2030 |
| liquidity|smc | 6 | 0.512 | 2030 |
| liquidity|mean_reversion|smc | 7 | 0.511 | 2030 |
| liquidity|path_risk|risk | 7 | 0.511 | 2030 |
| breakout_bounce|risk|smc | 6 | 0.504 | 2030 |

### Global stability (TimeSeriesSplit AUC)
- Stability analysis unavailable: Insufficient folds for stability analysis

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| smc_predicted | 0.1046 | 0.1107 | 0.072 | 0.105 | 0.0109 |
| macro_trend_score_continuous | 0.0505 | 0.0921 | 0.268 | 0.051 | 0.0026 |
| liquidity_regime_2_prob | 0.0600 | 0.0730 | 0.307 | -0.060 | 0.0036 |
| risk_score | 0.0563 | 0.0531 | 0.441 | -0.056 | 0.0032 |
| mr_probability_dense | 0.0254 | 0.0385 | 0.412 | -0.025 | 0.0006 |
| liquidity_regime_1_prob | 0.0211 | 0.0377 | 0.520 | -0.021 | 0.0004 |
| path_risk_score | 0.0475 | 0.0333 | 0.388 | -0.047 | 0.0023 |
| vol_force_volatility | 0.0006 | 0.0285 | 0.774 | 0.001 | 0.0000 |
| vol_force_trend | 0.0006 | 0.0285 | 0.774 | 0.001 | 0.0000 |
| liquidity_regime_3_prob | 0.0292 | 0.0268 | 0.403 | -0.029 | 0.0009 |
| vol_force_breakout | 0.0052 | 0.0265 | 0.771 | 0.005 | 0.0000 |
| liquidity_regime_0_prob | 0.0141 | 0.0158 | 0.178 | -0.014 | 0.0002 |
| support_scalar | 0.0113 | 0.0123 | 0.856 | -0.011 | 0.0001 |
| breakout_success_prob | 0.0113 | 0.0123 | 0.856 | -0.011 | 0.0001 |
| liquidity_regime_4_prob | 0.0101 | 0.0087 | 0.407 | 0.010 | 0.0001 |
| resistance_scalar | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_high_conf_signal | 0.0000 | 0.0000 | inf | nan | nan |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| resistance_scalar | 0.0000 | 0.0000 | inf | nan | nan |
| breakout_high_conf_signal | 0.0000 | 0.0000 | inf | nan | nan |
| smc_predicted | 0.1046 | 0.1107 | 0.072 | 0.105 | 0.0109 |
| liquidity_regime_2_prob | 0.0600 | 0.0730 | 0.307 | -0.060 | 0.0036 |
| risk_score | 0.0563 | 0.0531 | 0.441 | -0.056 | 0.0032 |
| macro_trend_score_continuous | 0.0505 | 0.0921 | 0.268 | 0.051 | 0.0026 |
| path_risk_score | 0.0475 | 0.0333 | 0.388 | -0.047 | 0.0023 |
| liquidity_regime_3_prob | 0.0292 | 0.0268 | 0.403 | -0.029 | 0.0009 |
| mr_probability_dense | 0.0254 | 0.0385 | 0.412 | -0.025 | 0.0006 |
| liquidity_regime_1_prob | 0.0211 | 0.0377 | 0.520 | -0.021 | 0.0004 |
| liquidity_regime_0_prob | 0.0141 | 0.0158 | 0.178 | -0.014 | 0.0002 |
| support_scalar | 0.0113 | 0.0123 | 0.856 | -0.011 | 0.0001 |
| breakout_success_prob | 0.0113 | 0.0123 | 0.856 | -0.011 | 0.0001 |
| liquidity_regime_4_prob | 0.0101 | 0.0087 | 0.407 | 0.010 | 0.0001 |
| vol_force_breakout | 0.0052 | 0.0265 | 0.771 | 0.005 | 0.0000 |
| vol_force_volatility | 0.0006 | 0.0285 | 0.774 | 0.001 | 0.0000 |
| vol_force_trend | 0.0006 | 0.0285 | 0.774 | 0.001 | 0.0000 |

## Constant / Near-Constant Feature Check
⚠️ Found 5 constant features:
- support_scalar (val=0.5297)
- breakout_success_prob (val=0.5000)
- breakout_high_conf_signal (val=0.0000)
- vol_force_volatility (val=0.0000)
- vol_force_trend (val=0.0000)

## Leakage diagnostics
- Suspicious features (|corr|>=0.95): 0
- Perfect-correlation features (|corr|>=0.99): 0

## Notable pairwise interactions (TreeSHAP)
- Interaction analysis unavailable: Insufficient data for TreeSHAP interactions