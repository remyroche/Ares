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
- Number of specialist features: 12
- Mean MI (CV-averaged): 0.0793
- Median MI (CV-averaged): 0.0737
- Mean R^2 (univariate): 0.0012
- Median R^2 (univariate): 0.0005
- High-MI features (MI>0.10): 1
- High-R^2 features (R^2>0.05): 0

### Probe model summary (LogReg / LGBM)
- Linear Regression (Ridge): RMSE=0.0007±0.0001, R2=-0.0598
- LightGBM Regressor: RMSE=0.0008±0.0001, R2=-0.1779

### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)

**Linear Regression** (data range: 243 days, 465 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 0% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |

**LightGBM Regressor** (data range: 243 days, 465 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 0% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |


### Per-specialist model reliability vs target (MI / R^2)
- liquidity: n_features=5, MI_mean=0.0734, R^2_mean=0.0010, high_MI=0, high_R^2=0
- path_risk: n_features=1, MI_mean=0.0145, R^2_mean=0.0025, high_MI=0, high_R^2=0
- macro_trend: n_features=1, MI_mean=0.0119, R^2_mean=0.0026, high_MI=0, high_R^2=0
- smc: n_features=1, MI_mean=0.2586, R^2_mean=0.0000, high_MI=1, high_R^2=0
- risk: n_features=1, MI_mean=0.0999, R^2_mean=0.0032, high_MI=0, high_R^2=0

### Per-specialist data coverage
*(Target samples: 2440)*
- **breakout_bounce**: n=2342 (96.0% coverage), range: 2024-12-30 23:15:00 → 2025-11-29 22:00:00 ⚠️ Starts late
- **liquidity**: n=1567 (64.2% coverage), range: 2025-08-30 22:00:00 → 2025-11-29 22:00:00 ⚠️ Starts late
- **macro_trend**: n=2440 (100.0% coverage), range: 2024-11-29 23:00:00 → 2025-11-29 22:00:00
- **mean_reversion**: n=2440 (100.0% coverage), range: 2024-11-29 23:00:00 → 2025-11-29 22:00:00
- **path_risk**: n=2439 (100.0% coverage), range: 2024-11-29 23:15:00 → 2025-11-29 22:00:00
- **risk**: n=2440 (100.0% coverage), range: 2024-11-29 23:00:00 → 2025-11-29 22:00:00
- **smc**: n=2440 (100.0% coverage), range: 2024-11-29 23:00:00 → 2025-11-29 22:00:00

### Pairwise relationships between specialist models (MI / R^2)

| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |
|---------|---------|---------------|---------------|---------:|----:|
| liquidity | risk | liquidity_regime_1_prob | risk_score | 0.3019 | 0.3389 |
| liquidity | macro_trend | liquidity_regime_1_prob | macro_trend_score_continuous | 0.0985 | 0.0381 |
| macro_trend | smc | macro_trend_score_continuous | smc_predicted | 0.0821 | 0.0089 |
| macro_trend | risk | macro_trend_score_continuous | risk_score | 0.0742 | 0.0214 |
| path_risk | risk | path_risk_score | risk_score | 0.0668 | 0.0201 |
| path_risk | smc | path_risk_score | smc_predicted | 0.0387 | 0.0039 |
| liquidity | path_risk | liquidity_regime_1_prob | path_risk_score | 0.0306 | 0.0038 |
| liquidity | smc | liquidity_regime_1_prob | smc_predicted | 0.0283 | 0.0037 |
| risk | smc | risk_score | smc_predicted | 0.0148 | 0.0007 |
| macro_trend | path_risk | macro_trend_score_continuous | path_risk_score | 0.0036 | 0.0001 |

### LGBM interaction probes (specialist groups)

| Groups | n_features | AUC | n_oof_samples |
|--------|-----------:|----:|--------------:|
| liquidity|risk | 6 | 0.597 | 2030 |
| liquidity|mean_reversion|risk | 7 | 0.591 | 2030 |
| breakout_bounce|liquidity|risk | 10 | 0.589 | 2030 |
| liquidity|risk|smc | 7 | 0.581 | 2030 |
| breakout_bounce|liquidity|path_risk | 10 | 0.567 | 2030 |
| breakout_bounce|path_risk|risk | 6 | 0.566 | 2030 |
| macro_trend|risk|smc | 3 | 0.560 | 2030 |
| breakout_bounce|liquidity|smc | 10 | 0.555 | 2030 |
| breakout_bounce|risk|smc | 6 | 0.553 | 2030 |
| liquidity|mean_reversion|smc | 7 | 0.552 | 2030 |
| mean_reversion|risk | 2 | 0.551 | 2030 |
| liquidity|smc | 6 | 0.550 | 2030 |
| liquidity|macro_trend|risk | 7 | 0.549 | 2030 |
| risk | 1 | 0.543 | 2030 |
| liquidity|mean_reversion | 6 | 0.540 | 2030 |
| mean_reversion|risk|smc | 3 | 0.539 | 2030 |
| breakout_bounce|liquidity|mean_reversion | 10 | 0.537 | 2030 |
| macro_trend|mean_reversion|risk | 3 | 0.536 | 2030 |
| breakout_bounce|liquidity | 9 | 0.531 | 2030 |
| liquidity|path_risk|risk | 7 | 0.530 | 2030 |

### Global stability (TimeSeriesSplit AUC)
- Stability analysis unavailable: Insufficient folds for stability analysis

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| smc_predicted | 0.0084 | 0.2586 | 0.233 | -0.001 | 0.0000 |
| risk_score | 0.1092 | 0.0999 | 1.226 | -0.056 | 0.0032 |
| liquidity_regime_1_prob | 0.0000 | 0.0992 | 0.842 | -0.021 | 0.0004 |
| vol_force_volatility | 0.0276 | 0.0853 | 0.488 | -0.020 | 0.0004 |
| vol_force_trend | 0.0276 | 0.0853 | 0.488 | -0.020 | 0.0004 |
| liquidity_regime_2_prob | 0.0361 | 0.0795 | 0.944 | -0.060 | 0.0036 |
| liquidity_regime_3_prob | 0.1072 | 0.0678 | 1.048 | -0.029 | 0.0009 |
| liquidity_regime_0_prob | 0.0000 | 0.0617 | 0.826 | -0.014 | 0.0002 |
| liquidity_regime_4_prob | 0.0000 | 0.0589 | 0.882 | 0.010 | 0.0001 |
| vol_force_breakout | 0.0000 | 0.0287 | 2.000 | -0.023 | 0.0005 |
| path_risk_score | 0.0628 | 0.0145 | 2.000 | -0.050 | 0.0025 |
| macro_trend_score_continuous | 0.0126 | 0.0119 | 0.969 | 0.051 | 0.0026 |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| liquidity_regime_2_prob | 0.0361 | 0.0795 | 0.944 | -0.060 | 0.0036 |
| risk_score | 0.1092 | 0.0999 | 1.226 | -0.056 | 0.0032 |
| macro_trend_score_continuous | 0.0126 | 0.0119 | 0.969 | 0.051 | 0.0026 |
| path_risk_score | 0.0628 | 0.0145 | 2.000 | -0.050 | 0.0025 |
| liquidity_regime_3_prob | 0.1072 | 0.0678 | 1.048 | -0.029 | 0.0009 |
| vol_force_breakout | 0.0000 | 0.0287 | 2.000 | -0.023 | 0.0005 |
| liquidity_regime_1_prob | 0.0000 | 0.0992 | 0.842 | -0.021 | 0.0004 |
| vol_force_volatility | 0.0276 | 0.0853 | 0.488 | -0.020 | 0.0004 |
| vol_force_trend | 0.0276 | 0.0853 | 0.488 | -0.020 | 0.0004 |
| liquidity_regime_0_prob | 0.0000 | 0.0617 | 0.826 | -0.014 | 0.0002 |
| liquidity_regime_4_prob | 0.0000 | 0.0589 | 0.882 | 0.010 | 0.0001 |
| smc_predicted | 0.0084 | 0.2586 | 0.233 | -0.001 | 0.0000 |

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