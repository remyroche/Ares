# Specialist Feature Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst
**Regime timeframe**: 1h
**Target column**: binary_label_long

## Data Range Analysis
- Target start date: 2021-10-31 20:45:00
- Target end date: 2025-12-10 09:30:00
- Target duration: 1500 days
- Target samples: 19729

## Overview
- Number of specialist features: 10
- Mean MI (CV-averaged): 0.0042
- Median MI (CV-averaged): 0.0041
- Mean R^2 (univariate): 0.0000
- Median R^2 (univariate): 0.0000
- High-MI features (MI>0.10): 0
- High-R^2 features (R^2>0.05): 0

### Probe model summary (LogReg / LGBM)
- Logistic Regression: not available
- LightGBM: not available

### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)

**Logistic Regression** (data range: 1245 days, 16440 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 0.500 | 3,194 | 2.57 | 0.0% | -0.4738% | -1.2155% | -37.00% | -21.09 |
| 0.550 | 3,194 | 2.57 | 0.0% | -0.4738% | -1.2155% | -37.00% | -21.09 |
| 0.600 | 3,194 | 2.57 | 0.0% | -0.4738% | -1.2155% | -37.00% | -21.09 |

**LightGBM** (data range: 1245 days, 16440 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 0.500 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 0.550 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 0.600 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |


### Per-specialist model reliability vs target (MI / R^2)
- xgb_macro: n_features=1, MI_mean=0.0098, R^2_mean=0.0000, high_MI=0, high_R^2=0
- volume: n_features=3, MI_mean=0.0044, R^2_mean=0.0000, high_MI=0, high_R^2=0
- smc: n_features=1, MI_mean=0.0047, R^2_mean=0.0000, high_MI=0, high_R^2=0
- liquidity: n_features=1, MI_mean=0.0004, R^2_mean=0.0000, high_MI=0, high_R^2=0
- microstructure: n_features=1, MI_mean=0.0034, R^2_mean=0.0000, high_MI=0, high_R^2=0
- spectral: n_features=1, MI_mean=0.0030, R^2_mean=0.0000, high_MI=0, high_R^2=0
- risk: n_features=1, MI_mean=0.0042, R^2_mean=0.0000, high_MI=0, high_R^2=0

### Per-specialist data coverage
*(Target samples: 19729)*
- **liquidity**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **microstructure**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **risk**: n=5144 (26.1% coverage), range: 2024-11-30 00:15:00 → 2025-12-10 09:30:00 ⚠️ Starts late ⚠️ Low coverage (<50%)
- **smc**: n=1383 (7.0% coverage), range: 2022-12-11 16:45:00 → 2023-03-22 14:00:00 ⚠️ Starts late ⚠️ Ends early ⚠️ Low coverage (<50%)
- **spectral**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **volume**: n=13285 (67.3% coverage), range: 2023-03-12 17:30:00 → 2025-12-10 09:30:00 ⚠️ Starts late
- **xgb_macro**: n=19272 (97.7% coverage), range: 2021-12-01 04:45:00 → 2025-12-10 09:30:00 ⚠️ Starts late

### Pairwise relationships between specialist models (MI / R^2)

| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |
|---------|---------|---------------|---------------|---------:|----:|
| microstructure | spectral | microstructure_score | spectral_score | 0.5392 | 0.2907 |
| smc | volume | smc_predicted | vol_force_volatility | 0.3677 | 0.1352 |
| liquidity | smc | liquidity_score | smc_predicted | 0.2631 | 0.0692 |
| risk | volume | path_risk_score | vol_force_volatility | 0.2562 | 0.0656 |
| volume | xgb_macro | vol_force_volatility | macro_trend_score_continuous | 0.2518 | 0.0634 |
| liquidity | volume | liquidity_score | vol_force_volatility | 0.1996 | 0.0398 |
| microstructure | smc | microstructure_score | smc_predicted | 0.1368 | 0.0187 |
| liquidity | xgb_macro | liquidity_score | macro_trend_score_continuous | 0.1204 | 0.0145 |
| smc | spectral | smc_predicted | spectral_score | 0.1200 | 0.0144 |
| liquidity | risk | liquidity_score | path_risk_score | 0.0926 | 0.0086 |
| risk | smc | path_risk_score | smc_predicted | 0.0776 | 0.0060 |
| risk | xgb_macro | path_risk_score | macro_trend_score_continuous | 0.0642 | 0.0041 |
| spectral | xgb_macro | spectral_score | macro_trend_score_continuous | 0.0558 | 0.0031 |
| spectral | volume | spectral_score | vol_force_volatility | 0.0364 | 0.0013 |
| microstructure | volume | microstructure_score | vol_force_volatility | 0.0360 | 0.0013 |
| smc | xgb_macro | smc_predicted | macro_trend_score_continuous | 0.0282 | 0.0008 |
| liquidity | spectral | liquidity_score | spectral_score | 0.0271 | 0.0007 |
| liquidity | microstructure | liquidity_score | microstructure_score | 0.0258 | 0.0007 |
| microstructure | risk | microstructure_score | path_risk_score | 0.0196 | 0.0004 |
| microstructure | xgb_macro | microstructure_score | macro_trend_score_continuous | 0.0043 | 0.0000 |
| risk | spectral | path_risk_score | spectral_score | 0.0031 | 0.0000 |

### LGBM interaction probes (specialist groups)

| Groups | n_features | AUC | n_oof_samples |
|--------|-----------:|----:|--------------:|
| liquidity | 1 | 0.500 | 16440 |
| microstructure | 1 | 0.500 | 16440 |
| risk | 1 | 0.500 | 16440 |
| smc | 1 | 0.500 | 16440 |
| spectral | 1 | 0.500 | 16440 |
| volume | 3 | 0.500 | 16440 |
| xgb_macro | 1 | 0.500 | 16440 |
| liquidity|microstructure | 2 | 0.500 | 16440 |
| liquidity|risk | 2 | 0.500 | 16440 |
| liquidity|smc | 2 | 0.500 | 16440 |
| liquidity|spectral | 2 | 0.500 | 16440 |
| liquidity|volume | 4 | 0.500 | 16440 |
| liquidity|xgb_macro | 2 | 0.500 | 16440 |
| microstructure|risk | 2 | 0.500 | 16440 |
| microstructure|smc | 2 | 0.500 | 16440 |
| microstructure|spectral | 2 | 0.500 | 16440 |
| microstructure|volume | 4 | 0.500 | 16440 |
| risk|smc | 2 | 0.500 | 16440 |
| microstructure|xgb_macro | 2 | 0.500 | 16440 |
| risk|spectral | 2 | 0.500 | 16440 |

### Global stability (TimeSeriesSplit AUC)
- Stability analysis unavailable: Insufficient folds for stability analysis

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| macro_trend_score_continuous | 0.0049 | 0.0098 | 0.477 | 0.000 | 0.0000 |
| vol_force_volatility | 0.0000 | 0.0049 | 1.429 | -0.000 | 0.0000 |
| smc_predicted | 0.0000 | 0.0047 | 1.493 | 0.000 | 0.0000 |
| vol_force_breakout | 0.0000 | 0.0044 | 1.610 | -0.000 | 0.0000 |
| path_risk_score | 0.0079 | 0.0042 | 1.683 | -0.000 | 0.0000 |
| vol_force_trend | 0.0000 | 0.0040 | 1.814 | 0.000 | 0.0000 |
| candlestick_score | 0.0022 | 0.0036 | 0.488 | -0.000 | 0.0000 |
| microstructure_score | 0.0016 | 0.0034 | 0.541 | -0.000 | 0.0000 |
| spectral_score | 0.0019 | 0.0030 | 0.539 | -0.001 | 0.0000 |
| liquidity_score | 0.0000 | 0.0004 | 1.371 | -0.003 | 0.0000 |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| liquidity_score | 0.0000 | 0.0004 | 1.371 | -0.003 | 0.0000 |
| spectral_score | 0.0019 | 0.0030 | 0.539 | -0.001 | 0.0000 |
| macro_trend_score_continuous | 0.0049 | 0.0098 | 0.477 | 0.000 | 0.0000 |
| microstructure_score | 0.0016 | 0.0034 | 0.541 | -0.000 | 0.0000 |
| vol_force_breakout | 0.0000 | 0.0044 | 1.610 | -0.000 | 0.0000 |
| vol_force_volatility | 0.0000 | 0.0049 | 1.429 | -0.000 | 0.0000 |
| candlestick_score | 0.0022 | 0.0036 | 0.488 | -0.000 | 0.0000 |
| vol_force_trend | 0.0000 | 0.0040 | 1.814 | 0.000 | 0.0000 |
| smc_predicted | 0.0000 | 0.0047 | 1.493 | 0.000 | 0.0000 |
| path_risk_score | 0.0079 | 0.0042 | 1.683 | -0.000 | 0.0000 |

## Constant / Near-Constant Feature Check
- No constant features found (std < 1e-9).

## Leakage diagnostics
- Suspicious features (|corr|>=0.95): 0
- Perfect-correlation features (|corr|>=0.99): 0

## Notable pairwise interactions (TreeSHAP)
- Computed on 10 features, sample_size=999

| Feature i | Feature j | Interaction strength |
|----------|----------|---------------------:|
| macro_trend_score_continuous | microstructure_score | 2.3607e-02 |
| macro_trend_score_continuous | liquidity_score | 1.3519e-02 |
| vol_force_volatility | microstructure_score | 1.0576e-02 |
| vol_force_volatility | liquidity_score | 8.8318e-03 |
| vol_force_volatility | candlestick_score | 8.2497e-03 |
| macro_trend_score_continuous | candlestick_score | 6.7814e-03 |
| vol_force_breakout | microstructure_score | 5.6242e-03 |
| vol_force_trend | microstructure_score | 3.9294e-03 |