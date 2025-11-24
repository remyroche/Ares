# Feature Distinctiveness Report

**Symbol:** ETHUSDT
**Assessment time:** 2025-11-24T23:12:07.364623
**Number of regimes:** 5
**Number of samples:** 720


====================================================================================================
FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)
====================================================================================================

## Core Dimension WCoV (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       5.8476          1.7522          3.3374         
Volume Long (rvol_168_scaled)                 5.5929          0.8902          6.2824         
Delta Regime Signal (delta_regime_signal_scaled) 27.0290         15.6504         1.7270         
Delta Align 3h (delta_alignment_3h)           0.0407          0.2787          0.1461         
Volume Direction Conviction                   0.0576          0.5563          0.1035         
Cumulative Delta Divergence                   0.1215          0.7665          0.1586         
Amihud Illiquidity (amihud_spike_ratio_scaled) 611.7450        3.5753          171.1047       

## Top Overall Features for Regime Distinction (Between/Within CoV)

Rank   Feature                                  Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
1      amihud_spike_ratio_scaled                611.7450        3.5753          171.1047       
2      rvol_168_scaled                          5.5929          0.8902          6.2824         
3      rvol_24_scaled                           5.8476          1.7522          3.3374         
4      volume_depth_ratio                       0.8569          0.4090          2.0949         
5      delta_regime_signal_scaled               27.0290         15.6504         1.7270         
6      rvol_20                                  0.7346          0.4417          1.6633         
7      volume_efficiency_ratio                  0.3520          0.2570          1.3697         
8      price_impact_ratio                       0.4116          0.3667          1.1225         
9      intra_bar_vol_estimate                   0.6064          0.5766          1.0516         
10     parkinsons_volatility                    0.3847          0.4874          0.7894         


## Best Features for Each Regime Pair (Separation Score)


### Apathy vs Valid Trend

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       2.7729         
2      intra_bar_vol_estimate                   1.7661         
3      rvol_168_scaled                          1.3183         
4      rvol_20                                  1.1637         
5      realized_vol_1h                          0.9544         
6      parkinsons_volatility                    0.8986         
7      rvol_24_scaled                           0.8617         
8      session_vol_percentile                   0.7497         
9      momentum_vol_alignment_3h_ewm3           0.7248         
10     momentum_vol_alignment_3h                0.7184         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.6205          3.3550          0.4830         
Volume Long (rvol_168_scaled)                 2.6124          1.4816          1.7633         
Delta Regime Signal (delta_regime_signal_scaled) 0.1733          8.6575          0.0200         
Delta Align 3h (delta_alignment_3h)           0.0287          0.2862          0.1003         
Volume Direction Conviction                   0.0159          0.5718          0.0278         
Cumulative Delta Divergence                   0.0331          0.7738          0.0428         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.4077          2.6456          0.1541         

### Apathy vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      rvol_168_scaled                          2.6473         
2      rvol_24_scaled                           1.8213         
3      rvol_20                                  1.5421         
4      volume_depth_ratio                       1.3907         
5      momentum_vol_alignment_3h_ewm3           1.3730         
6      realized_vol_1h                          1.2544         
7      session_vol_percentile_ewm6              1.2424         
8      session_vol_percentile                   1.2320         
9      vol_ratio_1h_6h                          1.1057         
10     momentum_vol_alignment_3h                1.0836         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.1637          2.9080          0.4002         
Volume Long (rvol_168_scaled)                 1.2406          1.0740          1.1550         
Delta Regime Signal (delta_regime_signal_scaled) 0.3091          15.4200         0.0200         
Delta Align 3h (delta_alignment_3h)           0.0097          0.2548          0.0380         
Volume Direction Conviction                   0.0310          0.5689          0.0545         
Cumulative Delta Divergence                   0.0224          0.7653          0.0293         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.1738          1.7390          0.0999         

### Apathy vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  3.0392         
2      volume_depth_ratio                       2.1009         
3      rvol_168_scaled                          1.9769         
4      rvol_20                                  1.9462         
5      amihud_spike_ratio_scaled                1.4319         
6      price_impact_ratio_ewm6                  1.1243         
7      rvol_24_scaled                           1.1164         
8      price_impact_ratio                       0.9270         
9      trap_score                               0.7511         
10     whipsaw_count                            0.6074         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.7046          2.8216          0.2497         
Volume Long (rvol_168_scaled)                 0.5842          0.9536          0.6127         
Delta Regime Signal (delta_regime_signal_scaled) 1.9998          6.9617          0.2872         
Delta Align 3h (delta_alignment_3h)           0.0332          0.2797          0.1188         
Volume Direction Conviction                   0.0848          0.5339          0.1588         
Cumulative Delta Divergence                   0.1616          0.7418          0.2178         
Amihud Illiquidity (amihud_spike_ratio_scaled) 3.0386          1.3104          2.3188         

### Apathy vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       2.4702         
2      volume_efficiency_ratio                  1.8957         
3      rvol_20                                  0.9568         
4      rvol_168_scaled                          0.9452         
5      rvol_24_scaled                           0.8245         
6      volume_depth_ratio                       0.7801         
7      price_impact_ratio_ewm6                  0.7530         
8      amihud_spike_ratio_scaled                0.7442         
9      trap_score                               0.4817         
10     intra_bar_vol_estimate                   0.4244         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.6397          2.9070          0.2200         
Volume Long (rvol_168_scaled)                 0.4175          1.0885          0.3836         
Delta Regime Signal (delta_regime_signal_scaled) 1.9414          23.1228         0.0840         
Delta Align 3h (delta_alignment_3h)           0.0135          0.2849          0.0473         
Volume Direction Conviction                   0.0454          0.5566          0.0815         
Cumulative Delta Divergence                   0.1232          0.7448          0.1654         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.5350          5.1678          0.2970         

### Valid Trend vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      rvol_168_scaled                          1.6349         
2      rvol_20                                  1.1552         
3      rvol_24_scaled                           1.0607         
4      volume_depth_ratio                       1.0475         
5      price_impact_ratio                       0.9389         
6      volume_efficiency_ratio                  0.9118         
7      realized_vol_1h                          0.7513         
8      session_vol_percentile_ewm6              0.7288         
9      momentum_vol_alignment_3h_ewm3           0.6671         
10     vol_ratio_1h_6h                          0.6105         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.5158          1.1889          0.4339         
Volume Long (rvol_168_scaled)                 0.6122          0.9742          0.6284         
Delta Regime Signal (delta_regime_signal_scaled) 0.4579          14.0535         0.0326         
Delta Align 3h (delta_alignment_3h)           0.0190          0.2684          0.0709         
Volume Direction Conviction                   0.0151          0.5804          0.0261         
Cumulative Delta Divergence                   0.0107          0.7995          0.0134         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.2517          3.1015          0.0812         

### Valid Trend vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       4.2339         
2      rvol_168_scaled                          2.5655         
3      volume_efficiency_ratio                  2.4600         
4      rvol_20                                  2.3589         
5      rvol_24_scaled                           1.8478         
6      intra_bar_vol_estimate                   1.5212         
7      parkinsons_volatility                    1.2346         
8      amihud_spike_ratio_scaled                1.0813         
9      price_impact_ratio                       0.8793         
10     price_impact_ratio_ewm6                  0.8340         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       6.4549          1.1025          5.8550         
Volume Long (rvol_168_scaled)                 3.8541          0.8538          4.5140         
Delta Regime Signal (delta_regime_signal_scaled) 2.7953          5.5952          0.4996         
Delta Align 3h (delta_alignment_3h)           0.0619          0.2934          0.2109         
Volume Direction Conviction                   0.0690          0.5454          0.1265         
Cumulative Delta Divergence                   0.1292          0.7760          0.1665         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.5394          2.6729          0.5759         

### Valid Trend vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       3.2518         
2      rvol_168_scaled                          1.9338         
3      rvol_20                                  1.7797         
4      price_impact_ratio                       1.6450         
5      rvol_24_scaled                           1.6145         
6      volume_efficiency_ratio                  1.4732         
7      intra_bar_vol_estimate                   1.3382         
8      momentum_vol_alignment_3h_ewm3           0.7515         
9      parkinsons_volatility                    0.6803         
10     session_vol_percentile_ewm6              0.5263         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       26.7984         1.1879          22.5593        
Volume Long (rvol_168_scaled)                 24.1772         0.9887          24.4545        
Delta Regime Signal (delta_regime_signal_scaled) 1.5823          21.7563         0.0727         
Delta Align 3h (delta_alignment_3h)           0.0153          0.2986          0.0511         
Volume Direction Conviction                   0.0295          0.5681          0.0519         
Cumulative Delta Divergence                   0.0904          0.7790          0.1161         
Amihud Illiquidity (amihud_spike_ratio_scaled) 3.0126          6.5304          0.4613         

### Absorption vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      rvol_168_scaled                          3.4232         
2      rvol_24_scaled                           2.5969         
3      volume_efficiency_ratio                  2.1550         
4      rvol_20                                  1.8779         
5      volume_depth_ratio                       1.5692         
6      price_impact_ratio_ewm6                  1.5516         
7      session_vol_percentile_ewm6              1.4315         
8      amihud_spike_ratio_scaled                1.2379         
9      price_impact_ratio                       1.0602         
10     momentum_vol_alignment_3h_ewm3           1.0348         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.5495          0.6554          3.8898         
Volume Long (rvol_168_scaled)                 2.3846          0.4462          5.3438         
Delta Regime Signal (delta_regime_signal_scaled) 1.4269          12.3577         0.1155         
Delta Align 3h (delta_alignment_3h)           0.0429          0.2620          0.1638         
Volume Direction Conviction                   0.0539          0.5425          0.0994         
Cumulative Delta Divergence                   0.1397          0.7675          0.1820         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.1023          1.7663          1.1903         

### Absorption vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      rvol_168_scaled                          3.0354         
2      price_impact_ratio                       2.7399         
3      rvol_24_scaled                           2.4238         
4      rvol_20                                  1.7197         
5      price_impact_ratio_ewm6                  1.6902         
6      volume_efficiency_ratio                  1.6840         
7      volume_depth_ratio                       1.4604         
8      momentum_vol_alignment_3h_ewm3           1.3925         
9      session_vol_percentile_ewm6              1.2021         
10     vol_momentum_sync_ewm3                   1.0065         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.0496          0.7409          2.7665         
Volume Long (rvol_168_scaled)                 1.7074          0.5811          2.9382         
Delta Regime Signal (delta_regime_signal_scaled) 4.0817          28.5188         0.1431         
Delta Align 3h (delta_alignment_3h)           0.0038          0.2671          0.0142         
Volume Direction Conviction                   0.0144          0.5652          0.0254         
Cumulative Delta Divergence                   0.1010          0.7705          0.1311         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.8565          5.6237          0.3301         

### Ghost vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  2.4341         
2      volume_depth_ratio                       1.2007         
3      rvol_168_scaled                          1.0202         
4      rvol_20                                  1.0080         
5      amihud_spike_ratio_scaled                0.7662         
6      price_impact_ratio_ewm6                  0.7322         
7      price_impact_ratio                       0.6204         
8      parkinsons_volatility                    0.5454         
9      efficiency_ratio_ewm6                    0.4713         
10     vol_clustering_ewm6                      0.4036         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.1183          0.6545          0.1807         
Volume Long (rvol_168_scaled)                 0.2205          0.4607          0.4786         
Delta Regime Signal (delta_regime_signal_scaled) 0.8072          20.0605         0.0402         
Delta Align 3h (delta_alignment_3h)           0.0467          0.2921          0.1598         
Volume Direction Conviction                   0.0396          0.5302          0.0747         
Cumulative Delta Divergence                   0.0392          0.7470          0.0525         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.8074          5.1951          0.1554         