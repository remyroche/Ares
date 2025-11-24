# Liquidity Cluster Quality Report

**Symbol:** ETHUSDT  \n**Assessment time:** 2025-11-23T23:48:26.860371

## Overall Quality

- Overall quality score: **0.2900**

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

- Absorption reversal rate: 0.4975
- Absorption follow-through rate: 0.2467

## Trend Confirmation & Apathy

- Valid trend follow-through (mean fwd return): 0.000000
- Apathy noise fraction: 0.0000

## Class Balance

- Class balance score: 0.0011
- Number of regimes: 2
- Number of samples: 28413

## Per-Regime Metrics

### Regime 2

- n_samples: 28411.000000
- ghost_ratio_mean: 0.196844
- ghost_ratio_std: 0.102864
- ghost_ratio_cov: 0.522564
- absorption_ratio_mean: 6.724455
- absorption_ratio_std: 3.833478
- absorption_ratio_cov: 0.570080
- rvol_24_mean: 1.002769
- rvol_24_std: 0.596202
- rvol_24_cov: 0.594556
- rvol_20_mean: 1.018562
- rvol_20_std: 0.938863
- rvol_20_cov: 0.921753
- volume_efficiency_ratio_mean: 992.515924
- volume_efficiency_ratio_std: 704.280251
- volume_efficiency_ratio_cov: 0.709591
- intraday_close_ratio_mean: 12791.025585
- intraday_close_ratio_std: 20303.072705
- intraday_close_ratio_cov: 1.587290
- amihud_spike_ratio_scaled_mean: 0.015647
- amihud_spike_ratio_scaled_std: 1.009246
- amihud_spike_ratio_scaled_cov: 64.498977
- rvol_168_scaled_mean: -0.015757
- rvol_168_scaled_std: 1.012879
- rvol_168_scaled_cov: 64.282430
- cumulative_delta_divergence_mean: 0.964589
- cumulative_delta_divergence_std: 0.731785
- cumulative_delta_divergence_cov: 0.758650
- volume_direction_conviction_mean: 0.498940
- volume_direction_conviction_std: 0.282609
- volume_direction_conviction_cov: 0.566419
- volume_direction_imbalance_mean: 0.024025
- volume_direction_imbalance_std: 0.572923
- volume_direction_imbalance_cov: 23.846593
- trend_confirmation_6h_mean: 0.232478
- trend_confirmation_6h_std: 0.142199
- trend_confirmation_6h_cov: 0.611669
- momentum_persistence_3h_mean: -2.036265
- momentum_persistence_3h_std: 262.885212
- momentum_persistence_3h_cov: 129.101685
- vol_momentum_sync_mean: 0.130039
- vol_momentum_sync_std: 0.279387
- vol_momentum_sync_cov: 2.148482
- range_momentum_divergence_mean: 0.999803
- range_momentum_divergence_std: 0.000176
- range_momentum_divergence_cov: 0.000176
- volume_concentration_ratio_3h_mean: 0.422677
- volume_concentration_ratio_3h_std: 0.152009
- volume_concentration_ratio_3h_cov: 0.359634
- pressure_ratio_mean: 807211565056.000000
- pressure_ratio_std: 37034664132608.000000
- pressure_ratio_cov: 45.879749
- kyle_lambda_proxy_mean: 12666066.666179
- kyle_lambda_proxy_std: 7403953.070346
- kyle_lambda_proxy_cov: 0.584550
- reversal_intensity_mean: 0.002326
- reversal_intensity_std: 0.004293
- reversal_intensity_cov: 1.846002
- whipsaw_count_mean: 6.392728
- whipsaw_count_std: 1.668653
- whipsaw_count_cov: 0.261024
- vol_clustering_mean: 0.368371
- vol_clustering_std: 0.106942
- vol_clustering_cov: 0.290309
- vol_regime_change_mean: -0.031151
- vol_regime_change_std: 0.249797
- vol_regime_change_cov: 8.019002
- efficiency_ratio_mean: 544.863422
- efficiency_ratio_std: 473.719658
- efficiency_ratio_cov: 0.869428
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000010
- forward_return_std: 0.007440
- forward_return_cov: 763.624186

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

