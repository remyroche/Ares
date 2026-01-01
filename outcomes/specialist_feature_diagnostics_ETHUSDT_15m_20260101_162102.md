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
- Number of specialist features: 6
- Mean MI (CV-averaged): 0.0635
- Median MI (CV-averaged): 0.0393
- Mean R^2 (univariate): 0.0093
- Median R^2 (univariate): 0.0019
- High-MI features (MI>0.10): 1
- High-R^2 features (R^2>0.05): 0

### Probe model summary (LogReg / LGBM)
- Logistic Regression: not available
- LightGBM: not available

### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)

**Logistic Regression** (data range: 1245 days, 16440 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 60% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 70% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 80% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 90% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |

**LightGBM** (data range: 1245 days, 16440 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 60% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 70% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 80% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 90% | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |


### Per-specialist model reliability vs target (MI / R^2)
- risk: n_features=1, MI_mean=0.0131, R^2_mean=0.0003, high_MI=0, high_R^2=0
- macro_trend: n_features=1, MI_mean=0.2183, R^2_mean=0.0486, high_MI=1, high_R^2=0
- volume_force_breakout: n_features=1, MI_mean=0.0339, R^2_mean=0.0025, high_MI=0, high_R^2=0
- smc: n_features=1, MI_mean=0.0446, R^2_mean=0.0030, high_MI=0, high_R^2=0
- liquidity: n_features=1, MI_mean=0.0626, R^2_mean=0.0013, high_MI=0, high_R^2=0
- path_risk: n_features=1, MI_mean=0.0082, R^2_mean=0.0001, high_MI=0, high_R^2=0

### Per-specialist data coverage
*(Target samples: 19729)*
- **liquidity**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **macro_trend**: n=19272 (97.7% coverage), range: 2021-12-01 04:45:00 → 2025-12-10 09:30:00 ⚠️ Starts late
- **mean_reversion**: n=4982 (25.3% coverage), range: 2024-12-11 03:15:00 → 2025-12-10 09:30:00 ⚠️ Starts late ⚠️ Low coverage (<50%)
- **path_risk**: n=5144 (26.1% coverage), range: 2024-11-30 00:15:00 → 2025-12-10 09:30:00 ⚠️ Starts late ⚠️ Low coverage (<50%)
- **risk**: n=5144 (26.1% coverage), range: 2024-11-30 00:15:00 → 2025-12-10 09:30:00 ⚠️ Starts late ⚠️ Low coverage (<50%)
- **smc**: n=14500 (73.5% coverage), range: 2022-12-11 16:45:00 → 2025-12-10 09:30:00 ⚠️ Starts late
- **volume_force_breakout**: n=13285 (67.3% coverage), range: 2023-03-12 17:30:00 → 2025-12-10 09:30:00 ⚠️ Starts late

### Pairwise relationships between specialist models (MI / R^2)

| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |
|---------|---------|---------------|---------------|---------:|----:|
| smc | volume_force_breakout | smc_predicted | vol_force_breakout | 0.7949 | 0.6318 |
| risk | volume_force_breakout | risk_score | vol_force_breakout | 0.4043 | 0.1635 |
| path_risk | risk | path_risk_score | risk_score | 0.3875 | 0.1502 |
| risk | smc | risk_score | smc_predicted | 0.3177 | 0.1009 |
| macro_trend | smc | macro_trend_score_continuous | smc_predicted | 0.2526 | 0.0638 |
| macro_trend | volume_force_breakout | macro_trend_score_continuous | vol_force_breakout | 0.2309 | 0.0533 |
| path_risk | volume_force_breakout | path_risk_score | vol_force_breakout | 0.2217 | 0.0491 |
| liquidity | risk | liquidity_score | risk_score | 0.1904 | 0.0362 |
| liquidity | volume_force_breakout | liquidity_score | vol_force_breakout | 0.1890 | 0.0357 |
| path_risk | smc | path_risk_score | smc_predicted | 0.1753 | 0.0307 |
| liquidity | macro_trend | liquidity_score | macro_trend_score_continuous | 0.1348 | 0.0182 |
| macro_trend | risk | macro_trend_score_continuous | risk_score | 0.1106 | 0.0122 |
| liquidity | path_risk | liquidity_score | path_risk_score | 0.0926 | 0.0086 |
| macro_trend | path_risk | macro_trend_score_continuous | path_risk_score | 0.0580 | 0.0034 |
| liquidity | smc | liquidity_score | smc_predicted | 0.0180 | 0.0003 |

### LGBM interaction probes (specialist groups)
- Group LGBM probes unavailable

### Global stability (TimeSeriesSplit AUC)
- Stability analysis unavailable: Insufficient folds for stability analysis

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| macro_trend_score_continuous | 0.2204 | 0.2183 | 0.015 | -0.220 | 0.0486 |
| liquidity_score | 0.0361 | 0.0626 | 0.232 | 0.036 | 0.0013 |
| smc_predicted | 0.0547 | 0.0446 | 0.198 | -0.055 | 0.0030 |
| vol_force_breakout | 0.0500 | 0.0339 | 0.447 | -0.050 | 0.0025 |
| risk_score | 0.0184 | 0.0131 | 0.000 | -0.018 | 0.0003 |
| path_risk_score | 0.0102 | 0.0082 | 0.000 | -0.010 | 0.0001 |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| macro_trend_score_continuous | 0.2204 | 0.2183 | 0.015 | -0.220 | 0.0486 |
| smc_predicted | 0.0547 | 0.0446 | 0.198 | -0.055 | 0.0030 |
| vol_force_breakout | 0.0500 | 0.0339 | 0.447 | -0.050 | 0.0025 |
| liquidity_score | 0.0361 | 0.0626 | 0.232 | 0.036 | 0.0013 |
| risk_score | 0.0184 | 0.0131 | 0.000 | -0.018 | 0.0003 |
| path_risk_score | 0.0102 | 0.0082 | 0.000 | -0.010 | 0.0001 |

## Constant / Near-Constant Feature Check
- No constant features found (std < 1e-9).

## Leakage diagnostics
- Leakage detection unavailable: 'FinalFeatureSelectionComponent' object has no attribute 'detect_potential_leakage'

## Notable pairwise interactions (TreeSHAP)
- Interaction analysis unavailable: Insufficient data for TreeSHAP interactions