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
- Number of specialist features: 13
- Mean MI (CV-averaged): 0.0033
- Median MI (CV-averaged): 0.0036
- Mean R^2 (univariate): 0.0000
- Median R^2 (univariate): 0.0000
- High-MI features (MI>0.10): 0
- High-R^2 features (R^2>0.05): 0

## Feature Engineering Recommendations

- Engineer non-linear transforms (log, diff, z-scores) and volatility-scaled features.
- Add regime/context signals (trend strength, volatility regime, liquidity stress) to lift MI.
- Introduce interaction features (ratios, spreads, cross-timeframe blends) to capture non-linear structure.
- Create horizon-aligned targets/features (multi-horizon returns, realized volatility, drawdown).
- Incorporate event-driven features (breakouts, mean-reversion markers, imbalance shocks).
- Repair or regenerate constant features (e.g., momentum prediction) upstream before training.
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
| HIGH_VOLATILITY | 2100 | 0.000 | nan | nan |
| LIQUIDITY_REGIME | 516 | 0.000 | nan | nan |
| LOW_VOLATILITY | 6828 | 0.004 | nan | nan |
| NEUTRAL | 10217 | 0.000 | nan | nan |
| STRESS_REGIME | 68 |  |  |  |

### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)

**Logistic Regression**: Insufficient LogReg predictions

**LightGBM**: Insufficient LGBM predictions


### Per-specialist model reliability vs target (MI / R^2)
- momentum: n_features=1, MI_mean=0.0011, R^2_mean=0.0000, high_MI=0, high_R^2=0
- volatility: n_features=2, MI_mean=0.0021, R^2_mean=0.0000, high_MI=0, high_R^2=0
- path: n_features=1, MI_mean=0.0042, R^2_mean=0.0000, high_MI=0, high_R^2=0
- xgb_macro: n_features=1, MI_mean=0.0098, R^2_mean=0.0000, high_MI=0, high_R^2=0
- smc: n_features=1, MI_mean=0.0047, R^2_mean=0.0000, high_MI=0, high_R^2=0
- liquidity: n_features=1, MI_mean=0.0004, R^2_mean=0.0000, high_MI=0, high_R^2=0
- volume: n_features=1, MI_mean=0.0006, R^2_mean=0.0000, high_MI=0, high_R^2=0
- microstructure: n_features=1, MI_mean=0.0034, R^2_mean=0.0000, high_MI=0, high_R^2=0
- spectral: n_features=1, MI_mean=0.0030, R^2_mean=0.0000, high_MI=0, high_R^2=0
- candlestick: n_features=1, MI_mean=0.0036, R^2_mean=0.0000, high_MI=0, high_R^2=0
- risk: n_features=1, MI_mean=0.0042, R^2_mean=0.0000, high_MI=0, high_R^2=0

### Per-specialist data coverage
*(Target samples: 19729)*
- **candlestick**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **causal**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **liquidity**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **microstructure**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **momentum**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **path**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **reversion**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **risk**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **smc**: n=1383 (7.0% coverage), range: 2022-12-11 16:45:00 → 2023-03-22 14:00:00 ⚠️ Starts late ⚠️ Ends early ⚠️ Low coverage (<50%)
- **spectral**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **volatility**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **volume**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **xgb_macro**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **xgb_meso**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00

### Pairwise relationships between specialist models (MI / R^2)

| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |
|---------|---------|---------------|---------------|---------:|----:|
| momentum | volume | ml_momentum_persistence_specialist_probability | volume_force_score | 0.9462 | 0.8952 |
| microstructure | spectral | microstructure_score | spectral_score | 0.5392 | 0.2907 |
| candlestick | microstructure | candlestick_score | microstructure_score | 0.4478 | 0.2006 |
| liquidity | momentum | liquidity_score | ml_momentum_persistence_specialist_probability | 0.3920 | 0.1537 |
| liquidity | volume | liquidity_score | volume_force_score | 0.3444 | 0.1186 |
| candlestick | spectral | candlestick_score | spectral_score | 0.2779 | 0.0772 |
| risk | volatility | path_risk_score | ml_breakout_bounce_regime_volatility_zscore_w10 | 0.2746 | 0.0754 |
| liquidity | smc | liquidity_score | smc_predicted | 0.2631 | 0.0692 |
| momentum | risk | ml_momentum_persistence_specialist_probability | path_risk_score | 0.2494 | 0.0622 |
| momentum | xgb_macro | ml_momentum_persistence_specialist_probability | macro_trend_score_continuous | 0.2462 | 0.0606 |
| volume | xgb_macro | volume_force_score | macro_trend_score_continuous | 0.2178 | 0.0474 |
| volatility | volume | ml_breakout_bounce_regime_volatility_zscore_w10 | volume_force_score | 0.2077 | 0.0432 |
| momentum | volatility | ml_momentum_persistence_specialist_probability | ml_breakout_bounce_regime_volatility_zscore_w10 | 0.1912 | 0.0366 |
| risk | volume | path_risk_score | volume_force_score | 0.1895 | 0.0359 |
| path | volume | ml_path_regime_vwap_zscore_w50 | volume_force_score | 0.1778 | 0.0316 |
| momentum | path | ml_momentum_persistence_specialist_probability | ml_path_regime_vwap_zscore_w50 | 0.1711 | 0.0293 |
| liquidity | volatility | liquidity_score | ml_breakout_bounce_regime_volatility_zscore_w10 | 0.1565 | 0.0245 |
| microstructure | volume | microstructure_score | volume_force_score | 0.1386 | 0.0192 |
| microstructure | smc | microstructure_score | smc_predicted | 0.1368 | 0.0187 |
| microstructure | momentum | microstructure_score | ml_momentum_persistence_specialist_probability | 0.1272 | 0.0162 |
| liquidity | xgb_macro | liquidity_score | macro_trend_score_continuous | 0.1204 | 0.0145 |
| smc | spectral | smc_predicted | spectral_score | 0.1200 | 0.0144 |
| smc | volatility | smc_predicted | ml_breakout_bounce_regime_volatility_zscore_w10 | 0.1181 | 0.0140 |
| momentum | smc | ml_momentum_persistence_specialist_probability | smc_predicted | 0.1136 | 0.0129 |
| candlestick | xgb_macro | candlestick_score | macro_trend_score_continuous | 0.1101 | 0.0121 |
| volatility | xgb_macro | ml_breakout_bounce_regime_volatility_zscore_w10 | macro_trend_score_continuous | 0.1058 | 0.0112 |
| smc | volume | smc_predicted | volume_force_score | 0.1055 | 0.0111 |
| liquidity | risk | liquidity_score | path_risk_score | 0.0926 | 0.0086 |
| path | risk | ml_path_regime_vwap_zscore_w50 | path_risk_score | 0.0857 | 0.0073 |
| risk | smc | path_risk_score | smc_predicted | 0.0776 | 0.0060 |
| candlestick | path | candlestick_score | ml_path_regime_vwap_zscore_w50 | 0.0693 | 0.0048 |
| path | spectral | ml_path_regime_vwap_zscore_w50 | spectral_score | 0.0649 | 0.0042 |
| risk | xgb_macro | path_risk_score | macro_trend_score_continuous | 0.0642 | 0.0041 |
| spectral | xgb_macro | spectral_score | macro_trend_score_continuous | 0.0558 | 0.0031 |
| candlestick | smc | candlestick_score | smc_predicted | 0.0555 | 0.0031 |
| spectral | volume | spectral_score | volume_force_score | 0.0533 | 0.0028 |
| microstructure | path | microstructure_score | ml_path_regime_vwap_zscore_w50 | 0.0493 | 0.0024 |
| path | volatility | ml_path_regime_vwap_zscore_w50 | ml_breakout_bounce_regime_volatility_zscore_w10 | 0.0475 | 0.0023 |
| liquidity | path | liquidity_score | ml_path_regime_vwap_zscore_w50 | 0.0434 | 0.0019 |
| momentum | spectral | ml_momentum_persistence_specialist_probability | spectral_score | 0.0404 | 0.0016 |
| candlestick | volatility | candlestick_score | ml_breakout_bounce_regime_volatility_zscore_w10 | 0.0397 | 0.0016 |
| path | smc | ml_path_regime_vwap_zscore_w50 | smc_predicted | 0.0310 | 0.0010 |
| candlestick | risk | candlestick_score | path_risk_score | 0.0307 | 0.0009 |
| path | xgb_macro | ml_path_regime_vwap_zscore_w50 | macro_trend_score_continuous | 0.0304 | 0.0009 |
| smc | xgb_macro | smc_predicted | macro_trend_score_continuous | 0.0282 | 0.0008 |
| liquidity | spectral | liquidity_score | spectral_score | 0.0271 | 0.0007 |
| liquidity | microstructure | liquidity_score | microstructure_score | 0.0258 | 0.0007 |
| candlestick | liquidity | candlestick_score | liquidity_score | 0.0226 | 0.0005 |
| microstructure | risk | microstructure_score | path_risk_score | 0.0196 | 0.0004 |
| microstructure | volatility | microstructure_score | ml_breakout_bounce_regime_volatility_zscore_w10 | 0.0172 | 0.0003 |
| candlestick | volume | candlestick_score | volume_force_score | 0.0148 | 0.0002 |
| spectral | volatility | spectral_score | ml_breakout_bounce_regime_volatility_zscore_w10 | 0.0146 | 0.0002 |
| microstructure | xgb_macro | microstructure_score | macro_trend_score_continuous | 0.0043 | 0.0000 |
| risk | spectral | path_risk_score | spectral_score | 0.0031 | 0.0000 |
| candlestick | momentum | candlestick_score | ml_momentum_persistence_specialist_probability | 0.0013 | 0.0000 |

### LGBM interaction probes (specialist groups)

| Groups | n_features | AUC | n_oof_samples |
|--------|-----------:|----:|--------------:|
| causal | 4 | 0.500 | 16440 |
| candlestick | 1 | 0.500 | 16440 |
| liquidity | 1 | 0.500 | 16440 |
| microstructure | 1 | 0.500 | 16440 |
| path | 5 | 0.500 | 16440 |
| momentum | 3 | 0.500 | 16440 |
| reversion | 4 | 0.500 | 16440 |
| risk | 6 | 0.500 | 16440 |
| smc | 1 | 0.500 | 16440 |
| spectral | 1 | 0.500 | 16440 |
| volatility | 4 | 0.500 | 16440 |
| volume | 1 | 0.500 | 16440 |
| xgb_meso | 5 | 0.500 | 16440 |
| xgb_macro | 6 | 0.500 | 16440 |
| candlestick|causal | 5 | 0.500 | 16440 |
| candlestick|liquidity | 2 | 0.500 | 16440 |
| candlestick|microstructure | 2 | 0.500 | 16440 |
| candlestick|momentum | 4 | 0.500 | 16440 |
| candlestick|path | 6 | 0.500 | 16440 |
| candlestick|reversion | 5 | 0.500 | 16440 |

### Global stability (TimeSeriesSplit AUC)
- Stability analysis unavailable: Insufficient folds for stability analysis

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| macro_trend_score_continuous | 0.0038 | 0.0098 | 0.477 | 0.000 | 0.0000 |
| smc_predicted | 0.0099 | 0.0047 | 1.493 | 0.000 | 0.0000 |
| path_risk_score | 0.0000 | 0.0042 | 1.683 | -0.000 | 0.0000 |
| ml_breakout_bounce_regime_volatility_zscore_w10 | 0.0000 | 0.0042 | 1.715 | -0.000 | 0.0000 |
| ml_breakout_bounce_regime_vol_trend_conflict_w50 | 0.0026 | 0.0042 | 1.715 | -0.000 | 0.0000 |
| ml_path_regime_vwap_zscore_w50 | 0.0049 | 0.0042 | 1.715 | -0.000 | 0.0000 |
| candlestick_score | 0.0016 | 0.0036 | 0.488 | -0.000 | 0.0000 |
| microstructure_score | 0.0014 | 0.0034 | 0.541 | -0.000 | 0.0000 |
| spectral_score | 0.0016 | 0.0030 | 0.539 | -0.001 | 0.0000 |
| ml_momentum_persistence_specialist_probability | 0.0003 | 0.0011 | 0.652 | 0.001 | 0.0000 |
| volume_force_score | 0.0001 | 0.0006 | 0.356 | 0.002 | 0.0000 |
| liquidity_score | 0.0000 | 0.0004 | 1.371 | -0.003 | 0.0000 |
| volatility_burst_score | 0.0001 | 0.0001 | 2.000 | 0.002 | 0.0000 |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| liquidity_score | 0.0000 | 0.0004 | 1.371 | -0.003 | 0.0000 |
| volume_force_score | 0.0001 | 0.0006 | 0.356 | 0.002 | 0.0000 |
| volatility_burst_score | 0.0001 | 0.0001 | 2.000 | 0.002 | 0.0000 |
| ml_momentum_persistence_specialist_probability | 0.0003 | 0.0011 | 0.652 | 0.001 | 0.0000 |
| spectral_score | 0.0016 | 0.0030 | 0.539 | -0.001 | 0.0000 |
| macro_trend_score_continuous | 0.0038 | 0.0098 | 0.477 | 0.000 | 0.0000 |
| microstructure_score | 0.0014 | 0.0034 | 0.541 | -0.000 | 0.0000 |
| candlestick_score | 0.0016 | 0.0036 | 0.488 | -0.000 | 0.0000 |
| ml_breakout_bounce_regime_vol_trend_conflict_w50 | 0.0026 | 0.0042 | 1.715 | -0.000 | 0.0000 |
| smc_predicted | 0.0099 | 0.0047 | 1.493 | 0.000 | 0.0000 |
| ml_path_regime_vwap_zscore_w50 | 0.0049 | 0.0042 | 1.715 | -0.000 | 0.0000 |
| ml_breakout_bounce_regime_volatility_zscore_w10 | 0.0000 | 0.0042 | 1.715 | -0.000 | 0.0000 |
| path_risk_score | 0.0000 | 0.0042 | 1.683 | -0.000 | 0.0000 |

## Constant / Near-Constant Feature Check
⚠️ Removed 3 constant/near-constant features before scoring:
- ml_momentum_persistence_specialist_prediction
- breakout_long_edge_score
- breakout_short_edge_score

⚠️ Found 1 constant features:
- ml_momentum_persistence_specialist_prediction (val=1.0000)

## Leakage diagnostics
- Suspicious features (|corr|>=0.95): 0
- Perfect-correlation features (|corr|>=0.99): 0

## Notable pairwise interactions (TreeSHAP)
- Computed on 13 features, sample_size=999

| Feature i | Feature j | Interaction strength |
|----------|----------|---------------------:|
| microstructure_score | volatility_burst_score | 2.3852e-02 |
| spectral_score | volatility_burst_score | 1.1392e-02 |
| ml_momentum_persistence_specialist_probability | volume_force_score | 9.2668e-03 |
| volume_force_score | liquidity_score | 6.1982e-03 |
| microstructure_score | volume_force_score | 5.3346e-03 |
| candlestick_score | volume_force_score | 3.4970e-04 |