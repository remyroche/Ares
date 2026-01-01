# Feature Distinctiveness Report

**Symbol:** ETHUSDT
**Assessment time:** 2025-12-31T17:22:13.938812
**Number of regimes:** 5
**Number of samples:** 2880


====================================================================================================
FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)
====================================================================================================

## Core Dimension WCoV (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       3.5252          1.5924          2.2138         
Volume Long (rvol_168_scaled)                 2.9050          1.0695          2.7161         
Delta Regime Signal (delta_regime_signal_scaled) 3.7758          11.6223         0.3249         
Delta Align 3h (delta_alignment_3h)           0.0235          0.2989          0.0785         
Volume Direction Conviction                   0.0532          0.5606          0.0948         
Cumulative Delta Divergence                   0.1297          0.7441          0.1743         
Amihud Illiquidity (amihud_spike_ratio_scaled) 3.9312          2.4573          1.5998         

## Top Overall Features for Regime Distinction (Between/Within CoV)

Rank   Feature                                  Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
1      rvol_168_scaled                          2.9050          1.0695          2.7161         
2      vwap_distance                            21.2607         8.7076          2.4416         
3      rvol_24_scaled                           3.5252          1.5924          2.2138         
4      volume_depth_ratio                       0.8756          0.4368          2.0045         
5      price_impact_ratio                       0.4585          0.2383          1.9243         
6      amihud_spike_ratio_scaled                3.9312          2.4573          1.5998         
7      rvol_20                                  0.6823          0.5048          1.3516         
8      volume_efficiency_ratio                  0.4639          0.3469          1.3374         
9      intra_bar_vol_estimate                   0.5608          0.4276          1.3115         
10     price_impact_ratio_ewm6                  0.2745          0.2398          1.1449         


## Best Features for Each Regime Pair (Separation Score)


### Apathy vs Valid Trend

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      rvol_168_scaled                          3.5318         
2      rvol_24_scaled                           2.9435         
3      price_impact_ratio                       2.8833         
4      parkinsons_volatility                    2.6646         
5      intra_bar_vol_estimate                   2.5429         
6      volume_depth_ratio                       2.3053         
7      price_impact_ratio_ewm6                  2.0514         
8      realized_vol_3h                          1.8820         
9      session_vol_percentile_ewm6              1.8002         
10     rvol_20                                  1.7925         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.9238          0.6898          2.7889         
Volume Long (rvol_168_scaled)                 1.7570          0.5709          3.0776         
Delta Regime Signal (delta_regime_signal_scaled) 4.7488          14.9187         0.3183         
Delta Align 3h (delta_alignment_3h)           0.0150          0.2974          0.0506         
Volume Direction Conviction                   0.0524          0.5444          0.0963         
Cumulative Delta Divergence                   0.1509          0.7299          0.2067         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.3481          2.3192          0.1501         

### Apathy vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       4.3706         
2      volume_depth_ratio                       2.3207         
3      rvol_168_scaled                          2.0586         
4      price_impact_ratio_ewm6                  1.8671         
5      rvol_24_scaled                           1.7363         
6      rvol_20                                  1.6466         
7      volume_efficiency_ratio                  1.4558         
8      intra_bar_vol_estimate                   1.0236         
9      amihud_spike_ratio_scaled                1.0116         
10     kyle_lambda_proxy                        0.9301         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       25.2057         1.0908          23.1071        
Volume Long (rvol_168_scaled)                 9.1070          0.8912          10.2190        
Delta Regime Signal (delta_regime_signal_scaled) 0.5466          11.9127         0.0459         
Delta Align 3h (delta_alignment_3h)           0.0169          0.3091          0.0546         
Volume Direction Conviction                   0.0169          0.5973          0.0283         
Cumulative Delta Divergence                   0.0650          0.7355          0.0883         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.5700          1.8795          0.3033         

### Apathy vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  2.6352         
2      price_impact_ratio                       2.1343         
3      price_impact_ratio_ewm6                  1.1448         
4      amihud_spike_ratio_scaled                1.0881         
5      intra_bar_vol_estimate                   0.8364         
6      realized_vol_1h                          0.7490         
7      session_vol_percentile                   0.6617         
8      momentum_vol_alignment_3h                0.6320         
9      vol_ratio_1h_3h                          0.6320         
10     vol_ratio_1h_6h                          0.6242         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.1080          0.7710          0.1401         
Volume Long (rvol_168_scaled)                 0.1104          0.6546          0.1687         
Delta Regime Signal (delta_regime_signal_scaled) 3.8265          14.6390         0.2614         
Delta Align 3h (delta_alignment_3h)           0.0164          0.3002          0.0546         
Volume Direction Conviction                   0.0443          0.5595          0.0792         
Cumulative Delta Divergence                   0.0381          0.7174          0.0532         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.6202          2.3493          0.6897         

### Apathy vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       2.1461         
2      rvol_168_scaled                          1.6206         
3      intra_bar_vol_estimate                   1.5313         
4      rvol_24_scaled                           1.1972         
5      rvol_20                                  1.1824         
6      parkinsons_volatility                    1.1587         
7      volume_efficiency_ratio                  1.0527         
8      price_impact_ratio                       0.9939         
9      realized_vol_1h                          0.9814         
10     realized_vol_3h                          0.9233         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.8220          2.6294          0.6929         
Volume Long (rvol_168_scaled)                 2.5287          1.5550          1.6262         
Delta Regime Signal (delta_regime_signal_scaled) 0.1741          15.6159         0.0111         
Delta Align 3h (delta_alignment_3h)           0.0033          0.3126          0.0106         
Volume Direction Conviction                   0.0058          0.5840          0.0100         
Cumulative Delta Divergence                   0.0404          0.7207          0.0561         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.2303          4.5344          0.0508         

### Valid Trend vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      intra_bar_vol_estimate                   2.1172         
2      parkinsons_volatility                    1.6472         
3      volume_depth_ratio                       1.5600         
4      rvol_168_scaled                          1.5222         
5      realized_vol_1h                          1.3261         
6      session_vol_percentile_ewm6              1.2857         
7      session_vol_percentile                   1.2695         
8      rvol_24_scaled                           1.2443         
9      realized_vol_3h                          1.1187         
10     rvol_20                                  1.1039         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.4903          0.9805          0.5000         
Volume Long (rvol_168_scaled)                 0.4900          0.7968          0.6149         
Delta Regime Signal (delta_regime_signal_scaled) 2.6332          8.1444          0.3233         
Delta Align 3h (delta_alignment_3h)           0.0319          0.2919          0.1093         
Volume Direction Conviction                   0.0693          0.5526          0.1254         
Cumulative Delta Divergence                   0.0868          0.7700          0.1127         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.2768          0.9058          0.3056         

### Valid Trend vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      rvol_168_scaled                          3.6157         
2      price_impact_ratio                       3.6079         
3      rvol_24_scaled                           3.0363         
4      price_impact_ratio_ewm6                  2.9600         
5      volume_depth_ratio                       2.3082         
6      parkinsons_volatility                    2.2912         
7      intra_bar_vol_estimate                   2.0635         
8      rvol_20                                  1.8397         
9      volume_efficiency_ratio                  1.6113         
10     realized_vol_3h                          1.5990         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.2919          0.6607          3.4689         
Volume Long (rvol_168_scaled)                 2.0430          0.5603          3.6464         
Delta Regime Signal (delta_regime_signal_scaled) 0.0537          10.8707         0.0049         
Delta Align 3h (delta_alignment_3h)           0.0013          0.2829          0.0048         
Volume Direction Conviction                   0.0082          0.5148          0.0158         
Cumulative Delta Divergence                   0.1879          0.7518          0.2500         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.9181          1.3756          2.1213         

### Valid Trend vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      rvol_168_scaled                          2.2318         
2      volume_depth_ratio                       1.8513         
3      rvol_24_scaled                           1.8083         
4      price_impact_ratio                       1.7242         
5      intra_bar_vol_estimate                   1.5145         
6      rvol_20                                  1.3965         
7      parkinsons_volatility                    1.3844         
8      price_impact_ratio_ewm6                  1.2064         
9      volume_efficiency_ratio                  1.0148         
10     realized_vol_3h                          0.9112         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.8315          2.5192          0.3301         
Volume Long (rvol_168_scaled)                 0.7874          1.4606          0.5391         
Delta Regime Signal (delta_regime_signal_scaled) 26.4059         11.8476         2.2288         
Delta Align 3h (delta_alignment_3h)           0.0117          0.2953          0.0397         
Volume Direction Conviction                   0.0466          0.5393          0.0864         
Cumulative Delta Divergence                   0.1111          0.7551          0.1472         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.5355          3.5607          0.1504         

### Absorption vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       4.1178         
2      price_impact_ratio_ewm6                  2.8057         
3      volume_depth_ratio                       2.2801         
4      rvol_168_scaled                          2.1942         
5      rvol_24_scaled                           1.8714         
6      volume_efficiency_ratio                  1.8303         
7      amihud_spike_ratio_scaled                1.8030         
8      rvol_20                                  1.7624         
9      kyle_lambda_proxy                        0.8017         
10     trap_score                               0.6903         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       14.5750         1.0617          13.7276        
Volume Long (rvol_168_scaled)                 1549.0331       0.8806          1759.1406      
Delta Regime Signal (delta_regime_signal_scaled) 3.0044          7.8648          0.3820         
Delta Align 3h (delta_alignment_3h)           0.0332          0.2947          0.1128         
Volume Direction Conviction                   0.0612          0.5677          0.1077         
Cumulative Delta Divergence                   0.1029          0.7574          0.1358         
Amihud Illiquidity (amihud_spike_ratio_scaled) 13.7448         0.9360          14.6850        

### Absorption vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       2.6055         
2      volume_efficiency_ratio                  1.2292         
3      amihud_spike_ratio_scaled                1.0713         
4      price_impact_ratio_ewm6                  1.0069         
5      intra_bar_vol_estimate                   0.8055         
6      volume_depth_ratio                       0.7544         
7      realized_vol_1h                          0.6720         
8      rvol_168_scaled                          0.6575         
9      rvol_20                                  0.6009         
10     rvol_24_scaled                           0.5650         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.5760          2.9202          0.1972         
Volume Long (rvol_168_scaled)                 0.4842          1.7809          0.2719         
Delta Regime Signal (delta_regime_signal_scaled) 0.4117          8.8417          0.0466         
Delta Align 3h (delta_alignment_3h)           0.0202          0.3071          0.0658         
Volume Direction Conviction                   0.0227          0.5922          0.0384         
Cumulative Delta Divergence                   0.0246          0.7607          0.0324         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.7075          3.1211          0.2267         

### Ghost vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  3.0263         
2      price_impact_ratio                       2.5823         
3      volume_depth_ratio                       2.0517         
4      price_impact_ratio_ewm6                  1.9297         
5      rvol_168_scaled                          1.7885         
6      rvol_24_scaled                           1.3618         
7      rvol_20                                  1.3302         
8      amihud_spike_ratio_scaled                0.9867         
9      parkinsons_volatility                    0.8491         
10     kyle_lambda_proxy                        0.7667         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.6127          2.6003          0.6202         
Volume Long (rvol_168_scaled)                 2.0630          1.5444          1.3358         
Delta Regime Signal (delta_regime_signal_scaled) 10.9418         11.5679         0.9459         
Delta Align 3h (delta_alignment_3h)           0.0131          0.2981          0.0438         
Volume Direction Conviction                   0.0385          0.5544          0.0694         
Cumulative Delta Divergence                   0.0784          0.7426          0.1056         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.3477          3.5909          0.3753         