# Feature Distinctiveness Report

**Symbol:** ETHUSDT
**Assessment time:** 2026-01-02T19:35:25.857031
**Number of regimes:** 5
**Number of samples:** 142619


====================================================================================================
FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)
====================================================================================================

## Core Dimension WCoV (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.4459          3.7000          0.6611         
Volume Long (rvol_168_scaled)                 2.4188          3.2734          0.7389         
Delta Regime Signal (delta_regime_signal_scaled) 3.2995          44.6084         0.0740         
Delta Align 3h (delta_alignment_3h)           0.0163          0.3123          0.0522         
Volume Direction Conviction                   0.0389          0.5548          0.0700         
Cumulative Delta Divergence                   0.0291          0.7484          0.0388         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.5503          7.3677          0.3461         

## Top Overall Features for Regime Distinction (Between/Within CoV)

Rank   Feature                                  Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
1      price_impact_ratio                       0.6311          0.2783          2.2678         
2      volume_efficiency_ratio                  0.5839          0.3359          1.7383         
3      volume_depth_ratio                       0.8658          0.5846          1.4811         
4      price_impact_ratio_ewm6                  0.5159          0.3524          1.4638         
5      kyle_lambda_proxy                        0.6057          0.7157          0.8463         
6      kyle_lambda_proxy_ewm6                   0.5820          0.7373          0.7893         
7      intra_bar_vol_estimate                   0.4744          0.6298          0.7534         
8      rvol_168_scaled                          2.4188          3.2734          0.7389         
9      rvol_24_scaled                           2.4459          3.7000          0.6611         
10     parkinsons_volatility                    0.3521          0.5714          0.6163         


## Best Features for Each Regime Pair (Separation Score)


### Apathy vs Valid Trend

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       2.2718         
2      intra_bar_vol_estimate                   2.0526         
3      parkinsons_volatility                    1.6334         
4      volume_efficiency_ratio                  1.3964         
5      realized_vol_1h                          1.3502         
6      realized_vol_3h                          1.3209         
7      rvol_168_scaled                          1.3192         
8      price_impact_ratio                       1.2780         
9      realized_vol_6h                          1.1865         
10     rvol_24_scaled                           1.1709         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.4935          2.5926          0.5761         
Volume Long (rvol_168_scaled)                 1.4784          2.3013          0.6424         
Delta Regime Signal (delta_regime_signal_scaled) 1.6882          51.3624         0.0329         
Delta Align 3h (delta_alignment_3h)           0.0183          0.3128          0.0586         
Volume Direction Conviction                   0.0055          0.5555          0.0099         
Cumulative Delta Divergence                   0.0144          0.7495          0.0192         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.5715          6.2795          0.0910         

### Apathy vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       1.8262         
2      rvol_168_scaled                          1.1624         
3      volume_depth_ratio                       1.1556         
4      price_impact_ratio_ewm6                  1.0659         
5      rvol_24_scaled                           1.0500         
6      kyle_lambda_proxy                        1.0361         
7      parkinsons_volatility                    1.0284         
8      intra_bar_vol_estimate                   0.9886         
9      kyle_lambda_proxy_ewm6                   0.9811         
10     realized_vol_3h                          0.9356         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.5502          2.6829          0.5778         
Volume Long (rvol_168_scaled)                 1.5355          2.3957          0.6409         
Delta Regime Signal (delta_regime_signal_scaled) 0.3464          15.7647         0.0220         
Delta Align 3h (delta_alignment_3h)           0.0014          0.3277          0.0042         
Volume Direction Conviction                   0.0374          0.5791          0.0646         
Cumulative Delta Divergence                   0.0097          0.7470          0.0130         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.2071          1.8690          0.1108         

### Apathy vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  4.3880         
2      price_impact_ratio                       2.7626         
3      price_impact_ratio_ewm6                  2.7457         
4      volume_depth_ratio                       1.2422         
5      intra_bar_vol_estimate                   0.9665         
6      kyle_lambda_proxy                        0.8797         
7      kyle_lambda_proxy_ewm6                   0.8512         
8      realized_vol_1h                          0.7680         
9      amihud_spike_ratio_scaled                0.7259         
10     range_momentum_divergence                0.6391         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.3674          2.8350          0.1296         
Volume Long (rvol_168_scaled)                 0.3722          2.4993          0.1489         
Delta Regime Signal (delta_regime_signal_scaled) 6.9896          18.7808         0.3722         
Delta Align 3h (delta_alignment_3h)           0.0190          0.3144          0.0603         
Volume Direction Conviction                   0.0254          0.5511          0.0461         
Cumulative Delta Divergence                   0.0288          0.7495          0.0384         
Amihud Illiquidity (amihud_spike_ratio_scaled) 10.5367         2.6661          3.9521         

### Apathy vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       3.1975         
2      volume_efficiency_ratio                  2.7810         
3      price_impact_ratio_ewm6                  1.5683         
4      intra_bar_vol_estimate                   0.9031         
5      realized_vol_1h                          0.6625         
6      parkinsons_volatility                    0.5975         
7      realized_vol_3h                          0.4956         
8      realized_vol_6h                          0.4598         
9      efficiency_ratio_ewm6                    0.4541         
10     reversal_intensity_ewm3                  0.4473         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.4040          7.0947          0.0569         
Volume Long (rvol_168_scaled)                 0.3957          6.2655          0.0632         
Delta Regime Signal (delta_regime_signal_scaled) 0.6298          57.7063         0.0109         
Delta Align 3h (delta_alignment_3h)           0.0135          0.3187          0.0423         
Volume Direction Conviction                   0.0043          0.5644          0.0076         
Cumulative Delta Divergence                   0.0051          0.7487          0.0068         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.7418          11.1320         0.0666         

### Valid Trend vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       2.3656         
2      price_impact_ratio_ewm6                  1.5441         
3      volume_efficiency_ratio                  1.1125         
4      volume_depth_ratio                       0.7068         
5      kyle_lambda_proxy                        0.6199         
6      kyle_lambda_proxy_ewm6                   0.5849         
7      amihud_spike_ratio_scaled                0.5060         
8      efficiency_ratio_ewm6                    0.4767         
9      session_vol_percentile                   0.4575         
10     efficiency_ratio                         0.4223         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.0431          1.3052          0.0330         
Volume Long (rvol_168_scaled)                 0.0450          1.1781          0.0382         
Delta Regime Signal (delta_regime_signal_scaled) 1.2838          45.7316         0.0281         
Delta Align 3h (delta_alignment_3h)           0.0170          0.3120          0.0544         
Volume Direction Conviction                   0.0319          0.5592          0.0570         
Cumulative Delta Divergence                   0.0047          0.7473          0.0063         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.6962          5.7968          0.1201         

### Valid Trend vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       3.1713         
2      volume_efficiency_ratio                  3.1526         
3      price_impact_ratio                       2.2664         
4      price_impact_ratio_ewm6                  2.0987         
5      rvol_168_scaled                          1.6079         
6      rvol_24_scaled                           1.4261         
7      intra_bar_vol_estimate                   1.3850         
8      kyle_lambda_proxy                        1.3078         
9      kyle_lambda_proxy_ewm6                   1.2030         
10     parkinsons_volatility                    1.1731         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.4955          1.4574          1.7123         
Volume Long (rvol_168_scaled)                 2.4596          1.2817          1.9190         
Delta Regime Signal (delta_regime_signal_scaled) 0.6780          48.7477         0.0139         
Delta Align 3h (delta_alignment_3h)           0.0006          0.2987          0.0021         
Volume Direction Conviction                   0.0309          0.5312          0.0581         
Cumulative Delta Divergence                   0.0432          0.7498          0.0576         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.5819          6.5940          0.2399         

### Valid Trend vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       2.5100         
2      intra_bar_vol_estimate                   1.4720         
3      volume_efficiency_ratio                  1.2835         
4      rvol_168_scaled                          1.1612         
5      parkinsons_volatility                    1.1387         
6      price_impact_ratio                       1.0617         
7      rvol_24_scaled                           1.0305         
8      realized_vol_1h                          0.9836         
9      kyle_lambda_proxy                        0.9306         
10     realized_vol_3h                          0.9208         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.1834          5.7171          0.2070         
Volume Long (rvol_168_scaled)                 1.1824          5.0478          0.2342         
Delta Regime Signal (delta_regime_signal_scaled) 16.7325         87.6732         0.1909         
Delta Align 3h (delta_alignment_3h)           0.0049          0.3030          0.0161         
Volume Direction Conviction                   0.0098          0.5445          0.0179         
Cumulative Delta Divergence                   0.0195          0.7491          0.0261         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.2956          15.0599         0.0196         

### Absorption vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio_ewm6                  3.2112         
2      price_impact_ratio                       3.1245         
3      volume_efficiency_ratio                  1.5909         
4      rvol_168_scaled                          1.4313         
5      kyle_lambda_proxy                        1.3349         
6      rvol_24_scaled                           1.2920         
7      volume_depth_ratio                       1.2891         
8      kyle_lambda_proxy_ewm6                   1.2848         
9      amihud_spike_ratio_scaled                0.9248         
10     rvol_20                                  0.9225         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.7479          1.5477          1.7755         
Volume Long (rvol_168_scaled)                 2.7150          1.3761          1.9730         
Delta Regime Signal (delta_regime_signal_scaled) 4.6738          13.1499         0.3554         
Delta Align 3h (delta_alignment_3h)           0.0176          0.3135          0.0561         
Volume Direction Conviction                   0.0627          0.5548          0.1130         
Cumulative Delta Divergence                   0.0385          0.7473          0.0515         
Amihud Illiquidity (amihud_spike_ratio_scaled) 8.7384          2.1834          4.0022         

### Absorption vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       4.5499         
2      price_impact_ratio_ewm6                  2.4013         
3      volume_efficiency_ratio                  1.3142         
4      volume_depth_ratio                       1.1869         
5      kyle_lambda_proxy                        1.1553         
6      kyle_lambda_proxy_ewm6                   1.1048         
7      rvol_168_scaled                          1.0181         
8      rvol_24_scaled                           0.9187         
9      rvol_20                                  0.7551         
10     parkinsons_volatility                    0.7156         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.2016          5.8074          0.2069         
Volume Long (rvol_168_scaled)                 1.2013          5.1422          0.2336         
Delta Regime Signal (delta_regime_signal_scaled) 0.8014          52.0754         0.0154         
Delta Align 3h (delta_alignment_3h)           0.0121          0.3178          0.0381         
Volume Direction Conviction                   0.0416          0.5680          0.0733         
Cumulative Delta Divergence                   0.0148          0.7466          0.0199         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.8225          10.6493         0.0772         

### Ghost vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  3.0520         
2      price_impact_ratio                       1.9313         
3      price_impact_ratio_ewm6                  1.7249         
4      volume_depth_ratio                       1.0887         
5      kyle_lambda_proxy                        0.6765         
6      kyle_lambda_proxy_ewm6                   0.6401         
7      rvol_168_scaled                          0.4891         
8      rvol_24_scaled                           0.4321         
9      amihud_spike_ratio_scaled                0.4121         
10     rvol_20                                  0.3994         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.6717          5.9595          0.1127         
Volume Long (rvol_168_scaled)                 0.6693          5.2459          0.1276         
Delta Regime Signal (delta_regime_signal_scaled) 1.4105          55.0915         0.0256         
Delta Align 3h (delta_alignment_3h)           0.0055          0.3045          0.0180         
Volume Direction Conviction                   0.0211          0.5400          0.0391         
Cumulative Delta Divergence                   0.0237          0.7491          0.0316         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.2793          11.4465         0.1118         