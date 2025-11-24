# Feature Distinctiveness Report

**Symbol:** ETHUSDT
**Assessment time:** 2025-11-23T23:48:26.860371
**Number of regimes:** 2
**Number of samples:** 28413


====================================================================================================
FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)
====================================================================================================

## Core Dimension WCoV (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.9639          48.4032         0.0199         
Volume Long (rvol_168_scaled)                 0.9520          32.1412         0.0296         
Delta Regime Signal (delta_regime_signal_scaled) 0.9800          72.6561         0.0135         
Delta Align 3h (delta_alignment_3h)           0.2094          0.1688          1.2403         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.0303          32.9566         0.0313         

## Top Overall Features for Regime Distinction (Between/Within CoV)

Rank   Feature                                  Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
1      vol_ratio_3h_6h                          0.3448          0.1327          2.5985         
2      session_vol_percentile_ewm6              0.3351          0.1430          2.3424         
3      rvol_20                                  0.9800          0.4609          2.1264         
4      vol_clustering_ewm6                      0.2785          0.1380          2.0183         
5      breakout_failure_rate                    0.9800          0.4975          1.9699         
6      momentum_vol_alignment_3h_ewm3           0.5152          0.2671          1.9285         
7      vol_clustering                           0.2720          0.1452          1.8735         
8      delta_alignment_3h                       0.2094          0.1688          1.2403         
9      consecutive_direction_ratio_3h_ewm3      0.1762          0.1461          1.2054         
10     price_impact_ratio_ewm6                  0.3503          0.2916          1.2015         


## Best Features for Each Regime Pair (Separation Score)


### Absorption vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      vol_regime_change                        2.8547         
2      vol_ratio_3h_6h                          2.8547         
3      session_vol_percentile_ewm6              2.5191         
4      vol_clustering_ewm6                      2.2970         
5      delta_alignment_3h                       2.2760         
6      vol_clustering                           2.1164         
7      momentum_vol_alignment_3h_ewm3           1.8240         
8      rvol_24_scaled                           1.7547         
9      whipsaw_count_ewm6                       1.7054         
10     volume_price_trend_sync                  1.6096         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.9836          48.4032         0.0203         
Volume Long (rvol_168_scaled)                 0.9714          32.1412         0.0302         
Delta Regime Signal (delta_regime_signal_scaled) 1.0000          72.6561         0.0138         
Delta Align 3h (delta_alignment_3h)           0.2136          0.1688          1.2656         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.0513          32.9566         0.0319         