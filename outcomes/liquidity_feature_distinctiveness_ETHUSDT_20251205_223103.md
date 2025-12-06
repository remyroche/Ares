# Feature Distinctiveness Report

**Symbol:** ETHUSDT
**Assessment time:** 2025-12-05T22:31:02.100844
**Number of regimes:** 5
**Number of samples:** 34135


====================================================================================================
FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)
====================================================================================================

## Core Dimension WCoV (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.8847          3.2897          0.5729         
Volume Long (rvol_168_scaled)                 1.7791          2.6628          0.6681         
Delta Regime Signal (delta_regime_signal_scaled) 9.7560          75.6196         0.1290         
Delta Align 3h (delta_alignment_3h)           0.0156          0.3005          0.0518         
Volume Direction Conviction                   0.0617          0.5472          0.1128         
Cumulative Delta Divergence                   0.0622          0.7405          0.0841         
Amihud Illiquidity (amihud_spike_ratio_scaled) 5.4825          4.9321          1.1116         

## Top Overall Features for Regime Distinction (Between/Within CoV)

Rank   Feature                                  Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
1      price_impact_ratio                       0.7353          0.3104          2.3688         
2      volume_efficiency_ratio                  0.6089          0.3265          1.8645         
3      price_impact_ratio_ewm6                  0.6421          0.3911          1.6419         
4      volume_depth_ratio                       0.8533          0.6467          1.3194         
5      intra_bar_vol_estimate                   0.6313          0.5460          1.1562         
6      amihud_spike_ratio_scaled                5.4825          4.9321          1.1116         
7      kyle_lambda_proxy                        0.6608          0.6949          0.9509         
8      kyle_lambda_proxy_ewm6                   0.6467          0.7142          0.9055         
9      parkinsons_volatility                    0.4055          0.5093          0.7962         
10     rvol_168_scaled                          1.7791          2.6628          0.6681         


## Best Features for Each Regime Pair (Separation Score)


### Apathy vs Valid Trend

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      intra_bar_vol_estimate                   1.8528         
2      session_vol_percentile                   1.4699         
3      parkinsons_volatility                    1.4167         
4      rvol_168_scaled                          1.3205         
5      vol_ratio_1h_6h                          1.2591         
6      rvol_24_scaled                           1.2460         
7      momentum_vol_alignment_3h                1.2352         
8      vol_ratio_1h_3h                          1.2352         
9      momentum_vol_alignment_3h_ewm3           1.2308         
10     session_vol_percentile_ewm6              1.2211         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.3740          3.0521          0.4502         
Volume Long (rvol_168_scaled)                 1.3365          2.9792          0.4486         
Delta Regime Signal (delta_regime_signal_scaled) 2.5855          9.9037          0.2611         
Delta Align 3h (delta_alignment_3h)           0.0181          0.3056          0.0592         
Volume Direction Conviction                   0.0668          0.5374          0.1242         
Cumulative Delta Divergence                   0.0861          0.7298          0.1180         
Amihud Illiquidity (amihud_spike_ratio_scaled) 6.5126          4.1824          1.5572         

### Apathy vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       2.6405         
2      volume_efficiency_ratio                  1.5795         
3      price_impact_ratio_ewm6                  1.5377         
4      volume_depth_ratio                       1.4534         
5      kyle_lambda_proxy                        1.2362         
6      kyle_lambda_proxy_ewm6                   1.1759         
7      intra_bar_vol_estimate                   1.1039         
8      parkinsons_volatility                    0.9476         
9      rvol_168_scaled                          0.6571         
10     rvol_20                                  0.5755         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       3.6292          4.1943          0.8653         
Volume Long (rvol_168_scaled)                 2.2812          3.6219          0.6298         
Delta Regime Signal (delta_regime_signal_scaled) 0.0238          14.0508         0.0017         
Delta Align 3h (delta_alignment_3h)           0.0055          0.3150          0.0174         
Volume Direction Conviction                   0.0124          0.5890          0.0211         
Cumulative Delta Divergence                   0.0294          0.7480          0.0393         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.3500          3.1238          0.1120         

### Apathy vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  4.3010         
2      price_impact_ratio_ewm6                  2.7846         
3      price_impact_ratio                       2.5323         
4      volume_depth_ratio                       2.0366         
5      kyle_lambda_proxy                        1.4977         
6      kyle_lambda_proxy_ewm6                   1.4487         
7      efficiency_ratio_ewm6                    0.5271         
8      efficiency_ratio                         0.4476         
9      amihud_spike_ratio_scaled                0.4063         
10     momentum_vol_alignment_3h                0.3367         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.1833          4.2063          0.0436         
Volume Long (rvol_168_scaled)                 0.3128          3.5589          0.0879         
Delta Regime Signal (delta_regime_signal_scaled) 2.3647          24.0398         0.0984         
Delta Align 3h (delta_alignment_3h)           0.0195          0.3054          0.0640         
Volume Direction Conviction                   0.0583          0.5608          0.1040         
Cumulative Delta Divergence                   0.0052          0.7517          0.0069         
Amihud Illiquidity (amihud_spike_ratio_scaled) 45.1593         4.8838          9.2467         

### Apathy vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       2.6710         
2      volume_efficiency_ratio                  2.6624         
3      price_impact_ratio_ewm6                  1.5907         
4      intra_bar_vol_estimate                   0.7673         
5      kyle_lambda_proxy                        0.7051         
6      kyle_lambda_proxy_ewm6                   0.7002         
7      rvol_168_scaled                          0.5562         
8      rvol_20                                  0.5231         
9      session_vol_percentile                   0.5163         
10     vol_ratio_1h_6h                          0.5136         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       3.8921          4.3614          0.8924         
Volume Long (rvol_168_scaled)                 2.7677          3.9420          0.7021         
Delta Regime Signal (delta_regime_signal_scaled) 0.9154          161.7321        0.0057         
Delta Align 3h (delta_alignment_3h)           0.0176          0.3055          0.0577         
Volume Direction Conviction                   0.0325          0.5696          0.0571         
Cumulative Delta Divergence                   0.0446          0.7444          0.0599         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.3167          6.8286          0.0464         

### Valid Trend vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       1.6156         
2      volume_efficiency_ratio                  1.3798         
3      intra_bar_vol_estimate                   1.3708         
4      price_impact_ratio_ewm6                  1.1239         
5      session_vol_percentile                   1.0825         
6      vol_ratio_1h_6h                          1.0205         
7      momentum_vol_alignment_3h                0.9980         
8      vol_ratio_1h_3h                          0.9980         
9      realized_vol_1h                          0.9528         
10     momentum_vol_alignment_3h_ewm3           0.9159         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.5657          2.1865          0.2587         
Volume Long (rvol_168_scaled)                 0.4611          1.6378          0.2815         
Delta Regime Signal (delta_regime_signal_scaled) 2.4580          10.1696         0.2417         
Delta Align 3h (delta_alignment_3h)           0.0126          0.3004          0.0420         
Volume Direction Conviction                   0.0791          0.5339          0.1482         
Cumulative Delta Divergence                   0.0568          0.7294          0.0779         
Amihud Illiquidity (amihud_spike_ratio_scaled) 4.8169          2.8472          1.6918         

### Valid Trend vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio_ewm6                  2.6185         
2      volume_efficiency_ratio                  2.4388         
3      price_impact_ratio                       2.4309         
4      intra_bar_vol_estimate                   1.7768         
5      rvol_168_scaled                          1.5569         
6      kyle_lambda_proxy                        1.4741         
7      parkinsons_volatility                    1.4660         
8      kyle_lambda_proxy_ewm6                   1.4299         
9      volume_depth_ratio                       1.3795         
10     rvol_24_scaled                           1.3416         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.5916          2.1985          0.7239         
Volume Long (rvol_168_scaled)                 1.7590          1.5748          1.1169         
Delta Regime Signal (delta_regime_signal_scaled) 0.6958          20.1586         0.0345         
Delta Align 3h (delta_alignment_3h)           0.0014          0.2908          0.0050         
Volume Direction Conviction                   0.0085          0.5057          0.0168         
Cumulative Delta Divergence                   0.0810          0.7331          0.1105         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.1319          4.6072          0.0286         

### Valid Trend vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       1.9020         
2      volume_efficiency_ratio                  1.5934         
3      intra_bar_vol_estimate                   1.4744         
4      price_impact_ratio_ewm6                  1.3458         
5      volume_depth_ratio                       1.2209         
6      kyle_lambda_proxy                        1.2083         
7      kyle_lambda_proxy_ewm6                   1.1636         
8      parkinsons_volatility                    1.1495         
9      session_vol_percentile                   0.8443         
10     realized_vol_1h                          0.8350         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.5792          2.3536          0.2461         
Volume Long (rvol_168_scaled)                 0.5303          1.9579          0.2709         
Delta Regime Signal (delta_regime_signal_scaled) 1.0399          157.8509        0.0066         
Delta Align 3h (delta_alignment_3h)           0.0005          0.2909          0.0016         
Volume Direction Conviction                   0.0343          0.5145          0.0667         
Cumulative Delta Divergence                   0.0417          0.7258          0.0574         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.2301          6.5520          0.3404         

### Absorption vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio_ewm6                  3.4051         
2      price_impact_ratio                       2.9227         
3      volume_efficiency_ratio                  2.7466         
4      kyle_lambda_proxy                        1.9998         
5      kyle_lambda_proxy_ewm6                   1.9579         
6      volume_depth_ratio                       1.8586         
7      parkinsons_volatility                    1.0338         
8      intra_bar_vol_estimate                   0.8935         
9      rvol_168_scaled                          0.8850         
10     efficiency_ratio_ewm6                    0.7833         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       10.2973         3.3407          3.0824         
Volume Long (rvol_168_scaled)                 6.8709          2.2175          3.0985         
Delta Regime Signal (delta_regime_signal_scaled) 2.4806          24.3057         0.1021         
Delta Align 3h (delta_alignment_3h)           0.0141          0.3002          0.0468         
Volume Direction Conviction                   0.0707          0.5573          0.1269         
Cumulative Delta Divergence                   0.0242          0.7513          0.0323         
Amihud Illiquidity (amihud_spike_ratio_scaled) 3.0265          3.5487          0.8529         

### Absorption vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       4.3389         
2      price_impact_ratio_ewm6                  2.7331         
3      volume_efficiency_ratio                  2.3258         
4      kyle_lambda_proxy                        1.6435         
5      kyle_lambda_proxy_ewm6                   1.6009         
6      volume_depth_ratio                       1.5492         
7      efficiency_ratio_ewm6                    0.7311         
8      efficiency_ratio                         0.6658         
9      parkinsons_volatility                    0.5130         
10     amihud_spike_ratio_scaled                0.3642         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.0200          3.4958          0.0057         
Volume Long (rvol_168_scaled)                 0.0916          2.6006          0.0352         
Delta Regime Signal (delta_regime_signal_scaled) 0.9114          161.9981        0.0056         
Delta Align 3h (delta_alignment_3h)           0.0121          0.3002          0.0405         
Volume Direction Conviction                   0.0449          0.5660          0.0794         
Cumulative Delta Divergence                   0.0152          0.7441          0.0204         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.6001          5.4934          0.1092         

### Ghost vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  2.8062         
2      price_impact_ratio                       1.8095         
3      price_impact_ratio_ewm6                  1.7182         
4      volume_depth_ratio                       1.6372         
5      kyle_lambda_proxy                        0.9422         
6      kyle_lambda_proxy_ewm6                   0.8805         
7      rvol_168_scaled                          0.7659         
8      rvol_20                                  0.7319         
9      intra_bar_vol_estimate                   0.5863         
10     rvol_24_scaled                           0.5641         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       12.9482         3.5078          3.6912         
Volume Long (rvol_168_scaled)                 18.2786         2.5376          7.2031         
Delta Regime Signal (delta_regime_signal_scaled) 1.2445          171.9871        0.0072         
Delta Align 3h (delta_alignment_3h)           0.0019          0.2907          0.0066         
Volume Direction Conviction                   0.0258          0.5378          0.0481         
Cumulative Delta Divergence                   0.0394          0.7477          0.0527         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.9723          7.2534          0.4098         