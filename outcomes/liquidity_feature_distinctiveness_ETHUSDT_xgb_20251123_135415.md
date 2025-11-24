# Feature Distinctiveness Report

**Symbol:** ETHUSDT
**Assessment time:** 2025-11-23T13:54:15.958312
**Number of regimes:** 2
**Number of samples:** 27579


====================================================================================================
FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)
====================================================================================================

## Core Dimension WCoV (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.8578          5.5644          0.1542         
Volume Long (rvol_168_scaled)                 0.8108          4.4404          0.1826         
Delta Regime Signal (delta_regime_signal_scaled) 0.9800          95.1859         0.0103         
Delta Align 3h (delta_alignment_3h)           0.2113          0.3414          0.6188         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.9754          342.5963        0.0028         

## Top Overall Features for Regime Distinction (Between/Within CoV)

Rank   Feature                                  Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
1      rvol_20                                  0.9800          0.3501          2.7995         
2      session_vol_percentile_ewm6              0.2957          0.1240          2.3840         
3      vol_ratio_3h_6h                          0.7247          0.3054          2.3727         
4      breakout_failure_rate                    0.9800          0.4985          1.9658         
5      volume_concentration_ratio_3h_ewm6       0.2034          0.1443          1.4088         
6      realized_vol_3h                          0.7393          0.5519          1.3394         
7      momentum_vol_alignment_3h_ewm3           0.6398          0.5013          1.2761         
8      price_impact_ratio_ewm6                  0.3542          0.2895          1.2237         
9      reversal_intensity_ewm3                  0.6576          0.5471          1.2019         
10     volume_price_trend_sync_ewm6             1.6617          1.4618          1.1368         


## Best Features for Each Regime Pair (Separation Score)


### Absorption vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      vol_regime_change                        2.8445         
2      vol_ratio_3h_6h                          2.8445         
3      session_vol_percentile_ewm6              2.6428         
4      volume_concentration_ratio_3h_ewm6       2.5654         
5      volume_price_trend_sync_ewm6             2.3801         
6      volume_price_trend_sync                  2.2830         
7      rvol_20                                  2.0199         
8      rvol_24_scaled                           1.7834         
9      whipsaw_count_ewm6                       1.6995         
10     rvol_168_scaled                          1.5259         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.8753          5.5644          0.1573         
Volume Long (rvol_168_scaled)                 0.8273          4.4404          0.1863         
Delta Regime Signal (delta_regime_signal_scaled) 1.0000          95.1859         0.0105         
Delta Align 3h (delta_alignment_3h)           0.2156          0.3414          0.6314         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.9953          342.5963        0.0029         