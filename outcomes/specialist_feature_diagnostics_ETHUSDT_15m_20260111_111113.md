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
- Number of specialist features: 11
- Mean MI (CV-averaged): 0.0039
- Median MI (CV-averaged): 0.0040
- Mean R^2 (univariate): 0.0000
- Median R^2 (univariate): 0.0000
- High-MI features (MI>0.10): 0
- High-R^2 features (R^2>0.05): 0

### TV-VAR System Metrics
- Stability Score: 0.000
- TV-VAR Samples: 19729
- Market Regimes Detected: 5

### Probe model summary (LogReg / LGBM)
- Logistic Regression: not available
- LightGBM: not available

### Per-regime probe models (TimeSeriesSplit within each regime)

| Regime | n_samples | pos_frac | LogReg AUC | LGBM AUC |
|--------|----------:|---------:|----------:|---------:|
| HIGH_VOLATILITY | 1826 | 0.000 | nan | nan |
| LIQUIDITY_REGIME | 896 | 0.000 | nan | nan |
| LOW_VOLATILITY | 5336 | 0.004 | nan | nan |
| NEUTRAL | 11661 | 0.000 | nan | nan |
| STRESS_REGIME | 10 |  |  |  |

### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)

**Logistic Regression**: Insufficient LogReg predictions

**LightGBM**: Insufficient LGBM predictions


### Per-specialist model reliability vs target (MI / R^2)
- momentum: n_features=1, MI_mean=0.0011, R^2_mean=0.0000, high_MI=0, high_R^2=0
- xgb_macro: n_features=1, MI_mean=0.0098, R^2_mean=0.0000, high_MI=0, high_R^2=0
- volume: n_features=3, MI_mean=0.0044, R^2_mean=0.0000, high_MI=0, high_R^2=0
- smc: n_features=1, MI_mean=0.0047, R^2_mean=0.0000, high_MI=0, high_R^2=0
- liquidity: n_features=1, MI_mean=0.0004, R^2_mean=0.0000, high_MI=0, high_R^2=0
- microstructure: n_features=1, MI_mean=0.0034, R^2_mean=0.0000, high_MI=0, high_R^2=0
- spectral: n_features=1, MI_mean=0.0030, R^2_mean=0.0000, high_MI=0, high_R^2=0
- candlestick: n_features=1, MI_mean=0.0036, R^2_mean=0.0000, high_MI=0, high_R^2=0
- risk: n_features=1, MI_mean=0.0042, R^2_mean=0.0000, high_MI=0, high_R^2=0

### Per-specialist data coverage
*(Target samples: 19729)*
- **candlestick**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **liquidity**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **microstructure**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **momentum**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **risk**: n=5144 (26.1% coverage), range: 2024-11-30 00:15:00 → 2025-12-10 09:30:00 ⚠️ Starts late ⚠️ Low coverage (<50%)
- **smc**: n=1383 (7.0% coverage), range: 2022-12-11 16:45:00 → 2023-03-22 14:00:00 ⚠️ Starts late ⚠️ Ends early ⚠️ Low coverage (<50%)
- **spectral**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **volume**: n=13285 (67.3% coverage), range: 2023-03-12 17:30:00 → 2025-12-10 09:30:00 ⚠️ Starts late
- **xgb_macro**: n=19272 (97.7% coverage), range: 2021-12-01 04:45:00 → 2025-12-10 09:30:00 ⚠️ Starts late

### Pairwise relationships between specialist models (MI / R^2)

| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |
|---------|---------|---------------|---------------|---------:|----:|
| microstructure | spectral | microstructure_score | spectral_score | 0.5392 | 0.2907 |
| candlestick | microstructure | candlestick_score | microstructure_score | 0.4478 | 0.2006 |
| liquidity | momentum | liquidity_score | ml_momentum_persistence_specialist_probability | 0.3920 | 0.1537 |
| smc | volume | smc_predicted | vol_force_volatility | 0.3677 | 0.1352 |
| candlestick | spectral | candlestick_score | spectral_score | 0.2779 | 0.0772 |
| liquidity | smc | liquidity_score | smc_predicted | 0.2631 | 0.0692 |
| risk | volume | path_risk_score | vol_force_volatility | 0.2562 | 0.0656 |
| volume | xgb_macro | vol_force_volatility | macro_trend_score_continuous | 0.2518 | 0.0634 |
| momentum | risk | ml_momentum_persistence_specialist_probability | path_risk_score | 0.2494 | 0.0622 |
| momentum | xgb_macro | ml_momentum_persistence_specialist_probability | macro_trend_score_continuous | 0.2462 | 0.0606 |
| liquidity | volume | liquidity_score | vol_force_volatility | 0.1996 | 0.0398 |
| microstructure | smc | microstructure_score | smc_predicted | 0.1368 | 0.0187 |
| candlestick | volume | candlestick_score | vol_force_volatility | 0.1289 | 0.0166 |
| microstructure | momentum | microstructure_score | ml_momentum_persistence_specialist_probability | 0.1272 | 0.0162 |
| liquidity | xgb_macro | liquidity_score | macro_trend_score_continuous | 0.1204 | 0.0145 |
| smc | spectral | smc_predicted | spectral_score | 0.1200 | 0.0144 |
| momentum | volume | ml_momentum_persistence_specialist_probability | vol_force_volatility | 0.1187 | 0.0141 |
| momentum | smc | ml_momentum_persistence_specialist_probability | smc_predicted | 0.1136 | 0.0129 |
| candlestick | xgb_macro | candlestick_score | macro_trend_score_continuous | 0.1101 | 0.0121 |
| liquidity | risk | liquidity_score | path_risk_score | 0.0926 | 0.0086 |
| risk | smc | path_risk_score | smc_predicted | 0.0776 | 0.0060 |
| risk | xgb_macro | path_risk_score | macro_trend_score_continuous | 0.0642 | 0.0041 |
| spectral | xgb_macro | spectral_score | macro_trend_score_continuous | 0.0558 | 0.0031 |
| candlestick | smc | candlestick_score | smc_predicted | 0.0555 | 0.0031 |
| momentum | spectral | ml_momentum_persistence_specialist_probability | spectral_score | 0.0404 | 0.0016 |
| spectral | volume | spectral_score | vol_force_volatility | 0.0364 | 0.0013 |
| microstructure | volume | microstructure_score | vol_force_volatility | 0.0360 | 0.0013 |
| candlestick | risk | candlestick_score | path_risk_score | 0.0307 | 0.0009 |
| smc | xgb_macro | smc_predicted | macro_trend_score_continuous | 0.0282 | 0.0008 |
| liquidity | spectral | liquidity_score | spectral_score | 0.0271 | 0.0007 |
| liquidity | microstructure | liquidity_score | microstructure_score | 0.0258 | 0.0007 |
| candlestick | liquidity | candlestick_score | liquidity_score | 0.0226 | 0.0005 |
| microstructure | risk | microstructure_score | path_risk_score | 0.0196 | 0.0004 |
| microstructure | xgb_macro | microstructure_score | macro_trend_score_continuous | 0.0043 | 0.0000 |
| risk | spectral | path_risk_score | spectral_score | 0.0031 | 0.0000 |
| candlestick | momentum | candlestick_score | ml_momentum_persistence_specialist_probability | 0.0013 | 0.0000 |

### LGBM interaction probes (specialist groups)

| Groups | n_features | AUC | n_oof_samples |
|--------|-----------:|----:|--------------:|
| candlestick | 1 | 0.500 | 16440 |
| liquidity | 1 | 0.500 | 16440 |
| microstructure | 1 | 0.500 | 16440 |
| momentum | 3 | 0.500 | 16440 |
| risk | 1 | 0.500 | 16440 |
| smc | 1 | 0.500 | 16440 |
| spectral | 1 | 0.500 | 16440 |
| volume | 3 | 0.500 | 16440 |
| xgb_macro | 1 | 0.500 | 16440 |
| candlestick|liquidity | 2 | 0.500 | 16440 |
| candlestick|microstructure | 2 | 0.500 | 16440 |
| candlestick|momentum | 4 | 0.500 | 16440 |
| candlestick|risk | 2 | 0.500 | 16440 |
| candlestick|smc | 2 | 0.500 | 16440 |
| candlestick|spectral | 2 | 0.500 | 16440 |
| candlestick|volume | 4 | 0.500 | 16440 |
| candlestick|xgb_macro | 2 | 0.500 | 16440 |
| liquidity|microstructure | 2 | 0.500 | 16440 |
| liquidity|momentum | 4 | 0.500 | 16440 |
| liquidity|risk | 2 | 0.500 | 16440 |

### Global stability (TimeSeriesSplit AUC)
- Stability analysis unavailable: Insufficient folds for stability analysis

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| macro_trend_score_continuous | 0.0056 | 0.0098 | 0.477 | 0.000 | 0.0000 |
| vol_force_volatility | 0.0029 | 0.0049 | 1.429 | -0.000 | 0.0000 |
| smc_predicted | 0.0000 | 0.0047 | 1.493 | 0.000 | 0.0000 |
| vol_force_breakout | 0.0056 | 0.0044 | 1.610 | -0.000 | 0.0000 |
| path_risk_score | 0.0000 | 0.0042 | 1.683 | -0.000 | 0.0000 |
| vol_force_trend | 0.0000 | 0.0040 | 1.814 | 0.000 | 0.0000 |
| candlestick_score | 0.0016 | 0.0036 | 0.488 | -0.000 | 0.0000 |
| microstructure_score | 0.0017 | 0.0034 | 0.541 | -0.000 | 0.0000 |
| spectral_score | 0.0016 | 0.0030 | 0.539 | -0.001 | 0.0000 |
| ml_momentum_persistence_specialist_probability | 0.0003 | 0.0011 | 0.652 | 0.001 | 0.0000 |
| liquidity_score | 0.0000 | 0.0004 | 1.371 | -0.003 | 0.0000 |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| liquidity_score | 0.0000 | 0.0004 | 1.371 | -0.003 | 0.0000 |
| ml_momentum_persistence_specialist_probability | 0.0003 | 0.0011 | 0.652 | 0.001 | 0.0000 |
| spectral_score | 0.0016 | 0.0030 | 0.539 | -0.001 | 0.0000 |
| macro_trend_score_continuous | 0.0056 | 0.0098 | 0.477 | 0.000 | 0.0000 |
| microstructure_score | 0.0017 | 0.0034 | 0.541 | -0.000 | 0.0000 |
| vol_force_breakout | 0.0056 | 0.0044 | 1.610 | -0.000 | 0.0000 |
| vol_force_volatility | 0.0029 | 0.0049 | 1.429 | -0.000 | 0.0000 |
| candlestick_score | 0.0016 | 0.0036 | 0.488 | -0.000 | 0.0000 |
| vol_force_trend | 0.0000 | 0.0040 | 1.814 | 0.000 | 0.0000 |
| smc_predicted | 0.0000 | 0.0047 | 1.493 | 0.000 | 0.0000 |
| path_risk_score | 0.0000 | 0.0042 | 1.683 | -0.000 | 0.0000 |

## Constant / Near-Constant Feature Check
⚠️ Found 1 constant features:
- ml_momentum_persistence_specialist_prediction (val=1.0000)

## Leakage diagnostics
- Suspicious features (|corr|>=0.95): 0
- Perfect-correlation features (|corr|>=0.99): 0

## Notable pairwise interactions (TreeSHAP)
- Computed on 11 features, sample_size=999

| Feature i | Feature j | Interaction strength |
|----------|----------|---------------------:|
| microstructure_score | ml_momentum_persistence_specialist_probability | 1.6893e-02 |
| macro_trend_score_continuous | microstructure_score | 8.7044e-03 |
| macro_trend_score_continuous | candlestick_score | 8.0945e-03 |
| vol_force_volatility | microstructure_score | 7.0110e-03 |
| ml_momentum_persistence_specialist_probability | liquidity_score | 4.8250e-03 |
| spectral_score | ml_momentum_persistence_specialist_probability | 2.9512e-03 |
| candlestick_score | ml_momentum_persistence_specialist_probability | 1.9128e-03 |
| macro_trend_score_continuous | liquidity_score | 1.4199e-03 |
| candlestick_score | microstructure_score | 6.2653e-04 |
| vol_force_breakout | microstructure_score | 5.0039e-04 |
| microstructure_score | spectral_score | 4.8454e-04 |
| candlestick_score | spectral_score | 2.8303e-04 |