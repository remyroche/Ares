# Specialist Feature Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m
**Direction**: long
**Model**: analyst
**Regime timeframe**: 1h
**Target column**: binary_label_long

## Data Range Analysis
- Target start date: 2021-10-30 23:15:00
- Target end date: 2025-10-31 17:30:00
- Target duration: 1461 days
- Target samples: 21177

## Overview
- Number of specialist features: 3
- Mean MI (CV-averaged): 0.0035
- Median MI (CV-averaged): 0.0035
- Mean R^2 (univariate): 0.0006
- Median R^2 (univariate): 0.0002
- High-MI features (MI>0.10): 0
- High-R^2 features (R^2>0.05): 0

### Probe model summary (LogReg / LGBM)
- Logistic Regression: AUC=0.503±0.007, Accuracy=0.829
- LightGBM: AUC=0.503±0.006, Accuracy=0.828

### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)

**Logistic Regression** (data range: 1224 days, 17645 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 0.500 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 0.550 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 0.600 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |

**LightGBM** (data range: 1224 days, 17645 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 0.500 | 18 | 0.01 | 16.7% | -0.6730% | -0.0099% | -0.30% | -10.83 |
| 0.550 | 6 | 0.00 | 0.0% | -1.0811% | -0.0053% | -0.16% | -39.14 |
| 0.600 | 1 | 0.00 | 0.0% | -1.2602% | -0.0010% | -0.03% | 0.00 |


### Per-specialist model reliability vs target (MI / R^2)
- volume: n_features=3, MI_mean=0.0035, R^2_mean=0.0006, high_MI=0, high_R^2=0

### Per-specialist data coverage
*(Target samples: 21177)*
- **volume**: n=21177 (100.0% coverage), range: 2021-10-30 23:15:00 → 2025-10-31 17:30:00

### Pairwise relationships between specialist models (MI / R^2)
- Pairwise model analysis unavailable: Not enough specialist model groups for pairwise analysis

### LGBM interaction probes (specialist groups)

| Groups | n_features | AUC | n_oof_samples |
|--------|-----------:|----:|--------------:|
| volume | 4 | 0.479 | 17645 |

### Global stability (TimeSeriesSplit AUC)
- Mean AUC=0.503, std=0.007, stability score=0.986

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| ml_volume_force_vol_zscore | 0.0000 | 0.0035 | 1.255 | -0.011 | 0.0001 |
| ml_volume_force_amihud_validity | 0.0000 | 0.0035 | 1.255 | -0.037 | 0.0014 |
| ml_volume_force_cvd_15m_zscore_over_1d | 0.0020 | 0.0035 | 1.255 | -0.013 | 0.0002 |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| ml_volume_force_amihud_validity | 0.0000 | 0.0035 | 1.255 | -0.037 | 0.0014 |
| ml_volume_force_cvd_15m_zscore_over_1d | 0.0020 | 0.0035 | 1.255 | -0.013 | 0.0002 |
| ml_volume_force_vol_zscore | 0.0000 | 0.0035 | 1.255 | -0.011 | 0.0001 |

## Constant / Near-Constant Feature Check
- No constant features found (std < 1e-9).

## Leakage diagnostics
- Suspicious features (|corr|>=0.95): 0
- Perfect-correlation features (|corr|>=0.99): 0

## Notable pairwise interactions (TreeSHAP)
- Computed on 3 features, sample_size=999

| Feature i | Feature j | Interaction strength |
|----------|----------|---------------------:|
| ml_volume_force_amihud_validity | ml_volume_force_cvd_15m_zscore_over_1d | 1.1449e-03 |
| ml_volume_force_vol_zscore | ml_volume_force_amihud_validity | 1.8587e-04 |