# Layer 3 Features (Exhaustive List)

This document lists the exhaustive set of features used in Layer 3 (`src/training/steps/labeling/label_based_layer_3.py`), specifically within the active `layer3_analyst_lgbm` pipeline (delegating to `src/training/steps/labeling/layer3/core.py`).

## 1. Base Features
*Derived from Input `oof_df`*
- **Base Model Predictions**: Columns specified in `base_model_cols` (e.g., `lgbm_cls`, `xgb_reg`, etc.).
- **Market Data**: `close`, `high`, `low`, `open`, `volume` (if available in input or integrated from market data).
- **Regime Label**: `regime_label` (if present).

## 2. Entropy Features
*Source: `src/utils/entropy_bars.py` & `src/utils/entropy_optimized.py` (via `integrate_entropy_bars_into_layer3`)*

### OHLCV (Resampled to Entropy Bars)
- `entropy_open`
- `entropy_high`
- `entropy_low`
- `entropy_close`
- `entropy_volume`
- `entropy_n_minutes`
- `entropy_entropy_contribution`

### Specialized Entropy Features
- `entropy_rolling_20` (Shannon entropy on close prices, window 20)
- `entropy_rolling_40`
- `entropy_rolling_60`
- `entropy_rolling_100`
- `lz_complexity` (Lempel-Ziv complexity)
- `trend_conviction_index` (Delta Entropy / Delta Time)
- `entropy_ma` (Rolling mean of entropy contribution)
- `entropy_std` (Rolling std of entropy contribution)
- `entropy_zscore` (Z-score of entropy contribution)
- `staleness_seconds`
- `staleness_minutes`
- `staleness_adjusted_drift`
- `drift_proxy`

## 3. Layer 3 Specific Features
*Source: `src/feature_generation/categories/layer3_specific_features.py` (via `generate_layer3_features`)*

### Regime Alignment
- `regime_prob_Quiet`
- `regime_prob_Trending`
- `regime_prob_Chaos`
- `regime_id_*` (One-Hot Encoded regime labels)

### Gate Regime Features (Price/Returns Based)
- `slope_short`
- `adx_proxy`
- `momentum_short`
- `snr` (Signal-to-Noise Ratio)
- `choppiness_index`
- `variance_ratio`
- `permutation_entropy`
- `efficiency_ratio`
- `time_since_last_vol_spike`
- `time_since_last_large_candle`

### Gate Regime Features (FracDiff Based)
- `frac_vol_12`
- `frac_slope_12`
- `frac_efficiency_ratio`
- `frac_choppiness`
- `frac_snr`

### Gate Regime Features (Innovation Based)
- `innov_vol_12`
- `innov_slope_12`
- `innov_efficiency_ratio`
- `innov_choppiness`
- `innov_snr`

### Time Features
- `day_of_week`
- `hour_sin`
- `hour_cos`
- `is_weekend`

### Cross-Timeframe Momentum Agreement
- `momentum_agreement`
- `momentum_agreement_abs`
- `momentum_weighted_agreement`
- `trend_consistency_12`

### Price-Denoised Features (if `kalman_price` available)
- `market_stretch`
- `price_deviation_abs`
- `price_deviation_pct`
- `raw_denoised_ratio`

### Kalman Information Features (if `kalman_price` available)
- `kalman_velocity`
- `kalman_acceleration`
- `kalman_deviation`
- `kalman_deviation_pct`
- `kalman_trend_strength`
- `kalman_vol_ratio`

### Position Features
- `price_position_in_range`

## 4. Core Regime-Aware Features
*Source: `src/training/steps/labeling/layer3/core.py` (via `generate_regime_aware_features`)*
- `regime_vol_z` (Robust Z-Score of Volatility)
- `regime_prob_high_vol` (Sigmoid probability of High Volatility)
- `regime_prob_low_vol` (Sigmoid probability of Low Volatility)
- `meta_disagreement` (Standard deviation of base model probabilities, if available)
- `meta_conservatism` (Min Prob / Mean Prob, if available)

## 5. Optimized / Unified Features (Optional/Modular)
*Source: `src/training/steps/labeling/layer3/feature_engineering.py` (via `enhance_layer3_features_optimized`)*
*Note: These are available in the modular Layer 3 architecture but may not be active in all pipeline configurations.*

- **Unified Price**: `unified_price_momentum`, `unified_price_strength`, `unified_volatility_adj`, `unified_regime_confidence`.
- **Adaptive Filtering**: `adaptive_filter_momentum`, `adaptive_filter_distance`, `adaptive_filter_regime`.
- **Noise Reduction**: `noise_reduction_momentum`, `noise_reduction_smoothness`, `noise_reduction_stability`.
- **Consensus**: `filter_consensus_score`.
- **Advanced Noise**: `price_disorder_score` (Vol 20 / Vol 100).
- **Layer 1 Weights**: `layer1_weight_momentum`, `layer1_weight_volatility_adj`, `weight_confidence_score`, `weight_regime_indicator`.
- **Anti-Explosion**: `log_returns`, `vol_adjusted_returns`, `fracdiff_log_price`, `causal_denoised_returns`, `rolling_volatility_20`, `rolling_volatility_50`, `rolling_momentum_{10,20,50}`, `rolling_skew_50`, `rolling_kurtosis_50`, `drawdown_100`, `vol_adj_tail_20`, `denoised_divergence`, `fracdiff_zscore_50`.
