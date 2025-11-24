# Liquidity Cluster Quality Report

**Symbol:** ETHUSDT  \n**Assessment time:** 2025-11-23T13:54:15.958312

## Overall Quality

- Overall quality score: **0.2905**

## Regime Separation Algorithm

The regime classification uses a hierarchical decision tree based on two key metrics:

**Key Metrics:**
- **RVOL (Relative Volume):** Volume / Average Volume (20-bar lookback)
- **VER (Volume-Efficiency Ratio):** Volume / Candle Range (High - Low)

### Phase 1: The "Energy" Filter (Volume)

First, check if the market is awake using RVOL:

**If RVOL < 0.8 (Low Energy):**
- Check Price Range:
  - **Small Range** → **Apathy** (Dead Zone)
  - **Large Range** → **Ghost** (Liquidity Gap / Trap)

### Phase 2: The "Conflict" Filter (High Volume)

**If RVOL > 1.2 (High Energy):** Big players are active. Check Efficiency (VER):

- **Is Range Small relative to Volume? (Low VER)**
  - → **Absorption** (The harder they push, the less it moves)
- **Is Range Large relative to Volume? (High VER)**
  - → **Valid Trend** (Efficient price discovery)

### Phase 3: The "Anomaly" Filter (The Steamroller)

This is the outlier regime:

**If RVOL > 3.0 AND Range > 3x ATR (Average True Range):**
- → **Steamroller**
- Even though liquidity is thick, buying pressure is so immense it clears the book instantly
- Represents initiative momentum with low liquidity risk

---

## CoV-based Separation

- Effort/Result CoV separation score: 0.7616
- Returns CoV separation score: 0.7616

## Effort vs Result Separation

- Effort/Result separation score: 0.0000
- Ghost vs Valid contrast: 0.0000
- Absorption vs Valid contrast: 0.0000

## Trap / Ghost Behavior

- Ghost reversal rate: 0.0000
- Ghost false-trend rate: 0.0000

## Absorption Behavior

- Absorption reversal rate: 0.4989
- Absorption follow-through rate: 0.2390

## Trend Confirmation & Apathy

- Valid trend follow-through (mean fwd return): 0.000000
- Apathy noise fraction: 0.0000

## Class Balance

- Class balance score: 0.0011
- Number of regimes: 2
- Number of samples: 27579

## Per-Regime Metrics

### Regime 2

- n_samples: 27577.000000
- ghost_ratio_mean: 0.190610
- ghost_ratio_std: 0.100077
- ghost_ratio_cov: 0.525039
- absorption_ratio_mean: 6.908894
- absorption_ratio_std: 3.863598
- absorption_ratio_cov: 0.559221
- rvol_24_mean: 0.959320
- rvol_24_std: 0.552194
- rvol_24_cov: 0.575610
- rvol_20_mean: 0.960235
- rvol_20_std: 0.672291
- rvol_20_cov: 0.700132
- volume_efficiency_ratio_mean: 982.962669
- volume_efficiency_ratio_std: 689.042265
- volume_efficiency_ratio_cov: 0.700985
- intraday_close_ratio_mean: 13051.793356
- intraday_close_ratio_std: 20556.981982
- intraday_close_ratio_cov: 1.575031
- amihud_spike_ratio_scaled_mean: -0.001471
- amihud_spike_ratio_scaled_std: 1.005504
- amihud_spike_ratio_scaled_cov: 683.778325
- rvol_168_scaled_mean: -0.102668
- rvol_168_scaled_std: 0.911777
- rvol_168_scaled_cov: 8.880847
- cumulative_delta_divergence_mean: 0.958873
- cumulative_delta_divergence_std: 0.727001
- cumulative_delta_divergence_cov: 0.758182
- volume_direction_conviction_mean: 0.497132
- volume_direction_conviction_std: 0.283183
- volume_direction_conviction_cov: 0.569633
- volume_direction_imbalance_mean: 0.024656
- volume_direction_imbalance_std: 0.571606
- volume_direction_imbalance_cov: 23.183062
- trend_confirmation_6h_mean: 0.229055
- trend_confirmation_6h_std: 0.169650
- trend_confirmation_6h_cov: 0.740652
- momentum_persistence_3h_mean: 31.754486
- momentum_persistence_3h_std: 11945.907828
- momentum_persistence_3h_cov: 376.195910
- vol_momentum_sync_mean: 0.130124
- vol_momentum_sync_std: 0.279222
- vol_momentum_sync_cov: 2.145821
- range_momentum_divergence_mean: 0.999808
- range_momentum_divergence_std: 0.000174
- range_momentum_divergence_cov: 0.000174
- volume_concentration_ratio_3h_mean: 0.328526
- volume_concentration_ratio_3h_std: 0.171719
- volume_concentration_ratio_3h_cov: 0.522696
- pressure_ratio_mean: 753512349696.000000
- pressure_ratio_std: 31965273128960.000000
- pressure_ratio_cov: 42.421698
- kyle_lambda_proxy_mean: 12415385.348228
- kyle_lambda_proxy_std: 12535153.364310
- kyle_lambda_proxy_cov: 1.009647
- reversal_intensity_mean: 0.002130
- reversal_intensity_std: 0.003761
- reversal_intensity_cov: 1.765931
- whipsaw_count_mean: 6.401596
- whipsaw_count_std: 1.666409
- whipsaw_count_cov: 0.260311
- vol_clustering_mean: 0.402159
- vol_clustering_std: 0.185177
- vol_clustering_cov: 0.460456
- vol_regime_change_mean: -0.079705
- vol_regime_change_std: 0.388158
- vol_regime_change_cov: 4.869939
- efficiency_ratio_mean: 1000.955375
- efficiency_ratio_std: 805.906054
- efficiency_ratio_cov: 0.805137
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000033
- forward_return_std: 0.006945
- forward_return_cov: 208.698173

### Regime 4

- n_samples: 2.000000
- ghost_ratio_mean: 0.000000
- ghost_ratio_std: 0.000000
- ghost_ratio_cov: 0.000000
- absorption_ratio_mean: 0.000000
- absorption_ratio_std: 0.000000
- absorption_ratio_cov: 0.000000
- rvol_24_mean: 0.000000
- rvol_24_std: 0.000000
- rvol_24_cov: 0.000000
- intraday_close_ratio_mean: 0.000000
- intraday_close_ratio_std: 0.000000
- intraday_close_ratio_cov: 0.000000
- volume_direction_conviction_mean: 0.000000
- volume_direction_conviction_std: 0.000000
- volume_direction_conviction_cov: 0.000000
- trend_confirmation_mean: 0.000000
- trend_confirmation_std: 0.000000
- trend_confirmation_cov: 0.000000
- range_momentum_divergence_mean: 0.000000
- range_momentum_divergence_std: 0.000000
- range_momentum_divergence_cov: 0.000000
- forward_return_mean: 0.000000
- forward_return_std: 0.000000
- forward_return_cov: 0.000000

