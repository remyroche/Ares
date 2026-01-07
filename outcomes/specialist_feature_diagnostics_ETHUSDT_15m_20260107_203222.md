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
- Number of specialist features: 3
- Mean MI (CV-averaged): 0.0074
- Median MI (CV-averaged): 0.0063
- Mean R^2 (univariate): 0.0082
- Median R^2 (univariate): 0.0043
- High-MI features (MI>0.10): 0
- High-R^2 features (R^2>0.05): 0

### TV-VAR System Metrics
- Stability Score: 0.500
- TV-VAR Samples: 19729
- Market Regimes Detected: 5

### Probe model summary (LogReg / LGBM)
- Logistic Regression: not available
- LightGBM: not available

### Per-regime probe models (TimeSeriesSplit within each regime)

| Regime | n_samples | pos_frac | LogReg AUC | LGBM AUC |
|--------|----------:|---------:|----------:|---------:|
| HIGH_VOLATILITY | 1192 | 0.000 | nan | nan |
| LIQUIDITY_REGIME | 2645 | 0.000 | nan | nan |
| LOW_VOLATILITY | 9239 | 0.003 | nan | nan |
| NEUTRAL | 6648 | 0.000 | nan | nan |
| STRESS_REGIME | 5 |  |  |  |

### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)

**Logistic Regression** (data range: 1245 days, 16440 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 0.500 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 0.550 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 0.600 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |

**LightGBM** (data range: 1245 days, 16440 samples)

| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |
|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|
| 0.500 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 0.550 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |
| 0.600 | 0 | 0.00 | 0.0% | 0.0000% | 0.0000% | 0.00% | 0.00 |


### Per-specialist model reliability vs target (MI / R^2)
- microstructure: n_features=1, MI_mean=0.0060, R^2_mean=0.0034, high_MI=0, high_R^2=0
- spectral: n_features=1, MI_mean=0.0100, R^2_mean=0.0170, high_MI=0, high_R^2=0

### Per-specialist data coverage
*(Target samples: 19729)*
- **microstructure**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00
- **spectral**: n=19729 (100.0% coverage), range: 2021-10-31 20:45:00 → 2025-12-10 09:30:00

### Pairwise relationships between specialist models (MI / R^2)

| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |
|---------|---------|---------------|---------------|---------:|----:|
| microstructure | spectral | microstructure_score | spectral_score | 0.5392 | 0.2907 |

### LGBM interaction probes (specialist groups)

| Groups | n_features | AUC | n_oof_samples |
|--------|-----------:|----:|--------------:|
| spectral | 1 | 0.500 | 16440 |
| microstructure | 1 | 0.500 | 16440 |
| microstructure|spectral | 2 | 0.500 | 16440 |

### Global stability (TimeSeriesSplit AUC)
- Stability analysis unavailable: Insufficient folds for stability analysis

## Top Features by MI Proxy (CV-averaged)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| spectral_score | 0.0045 | 0.0100 | 0.494 | 0.130 | 0.0170 |
| candlestick_score | 0.0028 | 0.0063 | 0.485 | 0.065 | 0.0043 |
| microstructure_score | 0.0026 | 0.0060 | 0.541 | 0.058 | 0.0034 |

## Top Features by R^2 (Univariate)

| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |
|---------|--------:|--------:|------:|-----:|----:|
| spectral_score | 0.0045 | 0.0100 | 0.494 | 0.130 | 0.0170 |
| candlestick_score | 0.0028 | 0.0063 | 0.485 | 0.065 | 0.0043 |
| microstructure_score | 0.0026 | 0.0060 | 0.541 | 0.058 | 0.0034 |

## Constant / Near-Constant Feature Check
- No constant features found (std < 1e-9).

## Leakage diagnostics
- Suspicious features (|corr|>=0.95): 0
- Perfect-correlation features (|corr|>=0.99): 0

## Notable pairwise interactions (TreeSHAP)
- Computed on 3 features, sample_size=999

| Feature i | Feature j | Interaction strength |
|----------|----------|---------------------:|
| candlestick_score | microstructure_score | 4.2222e-02 |