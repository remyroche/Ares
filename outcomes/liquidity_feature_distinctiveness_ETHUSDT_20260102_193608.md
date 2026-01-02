# Feature Distinctiveness Report

**Symbol:** ETHUSDT
**Assessment time:** 2026-01-02T19:36:07.005376
**Number of regimes:** 5
**Number of samples:** 142619


====================================================================================================
FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)
====================================================================================================

## Core Dimension WCoV (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.4624          3.6362          0.6772         
Volume Long (rvol_168_scaled)                 2.4346          3.2165          0.7569         
Delta Regime Signal (delta_regime_signal_scaled) 3.1319          45.0857         0.0695         
Delta Align 3h (delta_alignment_3h)           0.0162          0.3123          0.0520         
Volume Direction Conviction                   0.0389          0.5548          0.0700         
Cumulative Delta Divergence                   0.0294          0.7484          0.0392         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.5506          7.4449          0.3426         

## Top Overall Features for Regime Distinction (Between/Within CoV)

Rank   Feature                                  Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
1      price_impact_ratio                       0.6311          0.2785          2.2658         
2      volume_efficiency_ratio                  0.5835          0.3359          1.7370         
3      volume_depth_ratio                       0.8679          0.5840          1.4860         
4      price_impact_ratio_ewm6                  0.5158          0.3526          1.4628         
5      kyle_lambda_proxy                        0.6066          0.7150          0.8483         
6      kyle_lambda_proxy_ewm6                   0.5828          0.7365          0.7913         
7      rvol_168_scaled                          2.4346          3.2165          0.7569         
8      intra_bar_vol_estimate                   0.4734          0.6294          0.7522         
9      rvol_24_scaled                           2.4624          3.6362          0.6772         
10     parkinsons_volatility                    0.3517          0.5711          0.6159         


## Best Features for Each Regime Pair (Separation Score)


### Apathy vs Valid Trend

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       2.2675         
2      intra_bar_vol_estimate                   2.0489         
3      parkinsons_volatility                    1.6305         
4      volume_efficiency_ratio                  1.4017         
5      realized_vol_1h                          1.3462         
6      realized_vol_3h                          1.3185         
7      rvol_168_scaled                          1.3148         
8      price_impact_ratio                       1.2832         
9      realized_vol_6h                          1.1840         
10     rvol_24_scaled                           1.1670         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.5060          2.5687          0.5863         
Volume Long (rvol_168_scaled)                 1.4900          2.2813          0.6531         
Delta Regime Signal (delta_regime_signal_scaled) 1.6218          51.5247         0.0315         
Delta Align 3h (delta_alignment_3h)           0.0184          0.3129          0.0586         
Volume Direction Conviction                   0.0055          0.5555          0.0099         
Cumulative Delta Divergence                   0.0151          0.7497          0.0202         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.5740          6.3091          0.0910         

### Apathy vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       1.8071         
2      rvol_168_scaled                          1.1678         
3      volume_depth_ratio                       1.1583         
4      price_impact_ratio_ewm6                  1.0567         
5      rvol_24_scaled                           1.0552         
6      kyle_lambda_proxy                        1.0379         
7      parkinsons_volatility                    1.0329         
8      intra_bar_vol_estimate                   0.9931         
9      kyle_lambda_proxy_ewm6                   0.9828         
10     realized_vol_3h                          0.9402         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.5563          2.6529          0.5866         
Volume Long (rvol_168_scaled)                 1.5409          2.3704          0.6501         
Delta Regime Signal (delta_regime_signal_scaled) 0.3160          15.1674         0.0208         
Delta Align 3h (delta_alignment_3h)           0.0017          0.3277          0.0051         
Volume Direction Conviction                   0.0374          0.5792          0.0646         
Cumulative Delta Divergence                   0.0102          0.7469          0.0136         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.2055          1.8673          0.1101         

### Apathy vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  4.3842         
2      price_impact_ratio                       2.7637         
3      price_impact_ratio_ewm6                  2.7471         
4      volume_depth_ratio                       1.2381         
5      intra_bar_vol_estimate                   0.9722         
6      kyle_lambda_proxy                        0.8790         
7      kyle_lambda_proxy_ewm6                   0.8507         
8      realized_vol_1h                          0.7713         
9      amihud_spike_ratio_scaled                0.7275         
10     range_momentum_divergence                0.6394         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.3624          2.8070          0.1291         
Volume Long (rvol_168_scaled)                 0.3675          2.4757          0.1485         
Delta Regime Signal (delta_regime_signal_scaled) 9.8825          18.3417         0.5388         
Delta Align 3h (delta_alignment_3h)           0.0191          0.3145          0.0607         
Volume Direction Conviction                   0.0253          0.5512          0.0460         
Cumulative Delta Divergence                   0.0285          0.7496          0.0381         
Amihud Illiquidity (amihud_spike_ratio_scaled) 10.7526         2.6613          4.0404         

### Apathy vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       3.2056         
2      volume_efficiency_ratio                  2.7831         
3      price_impact_ratio_ewm6                  1.5723         
4      intra_bar_vol_estimate                   0.9051         
5      realized_vol_1h                          0.6640         
6      parkinsons_volatility                    0.5998         
7      realized_vol_3h                          0.4985         
8      realized_vol_6h                          0.4619         
9      efficiency_ratio_ewm6                    0.4553         
10     reversal_intensity_ewm3                  0.4479         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.3982          6.9327          0.0574         
Volume Long (rvol_168_scaled)                 0.3893          6.1210          0.0636         
Delta Regime Signal (delta_regime_signal_scaled) 0.6512          57.8405         0.0113         
Delta Align 3h (delta_alignment_3h)           0.0136          0.3188          0.0426         
Volume Direction Conviction                   0.0043          0.5644          0.0077         
Cumulative Delta Divergence                   0.0052          0.7488          0.0069         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.7460          11.2925         0.0661         

### Valid Trend vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       2.3646         
2      price_impact_ratio_ewm6                  1.5404         
3      volume_efficiency_ratio                  1.1109         
4      volume_depth_ratio                       0.7141         
5      kyle_lambda_proxy                        0.6241         
6      kyle_lambda_proxy_ewm6                   0.5888         
7      amihud_spike_ratio_scaled                0.5053         
8      efficiency_ratio_ewm6                    0.4740         
9      session_vol_percentile                   0.4493         
10     efficiency_ratio                         0.4189         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.0374          1.3077          0.0286         
Volume Long (rvol_168_scaled)                 0.0393          1.1802          0.0333         
Delta Regime Signal (delta_regime_signal_scaled) 1.2812          46.5853         0.0275         
Delta Align 3h (delta_alignment_3h)           0.0167          0.3119          0.0535         
Volume Direction Conviction                   0.0319          0.5591          0.0570         
Cumulative Delta Divergence                   0.0050          0.7472          0.0066         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.6973          5.8311          0.1196         

### Valid Trend vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       3.1722         
2      volume_efficiency_ratio                  3.1612         
3      price_impact_ratio                       2.2680         
4      price_impact_ratio_ewm6                  2.1012         
5      rvol_168_scaled                          1.6003         
6      rvol_24_scaled                           1.4191         
7      intra_bar_vol_estimate                   1.3732         
8      kyle_lambda_proxy                        1.3064         
9      kyle_lambda_proxy_ewm6                   1.2023         
10     parkinsons_volatility                    1.1637         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.5174          1.4618          1.7221         
Volume Long (rvol_168_scaled)                 2.4813          1.2855          1.9302         
Delta Regime Signal (delta_regime_signal_scaled) 0.6756          49.7596         0.0136         
Delta Align 3h (delta_alignment_3h)           0.0007          0.2987          0.0025         
Volume Direction Conviction                   0.0309          0.5311          0.0581         
Cumulative Delta Divergence                   0.0436          0.7499          0.0582         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.5793          6.6250          0.2384         

### Valid Trend vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       2.5090         
2      intra_bar_vol_estimate                   1.4635         
3      volume_efficiency_ratio                  1.2888         
4      rvol_168_scaled                          1.1563         
5      parkinsons_volatility                    1.1323         
6      price_impact_ratio                       1.0691         
7      rvol_24_scaled                           1.0259         
8      realized_vol_1h                          0.9767         
9      kyle_lambda_proxy                        0.9300         
10     realized_vol_3h                          0.9144         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.1903          5.5876          0.2130         
Volume Long (rvol_168_scaled)                 1.1894          4.9308          0.2412         
Delta Regime Signal (delta_regime_signal_scaled) 17.2765         89.2585         0.1936         
Delta Align 3h (delta_alignment_3h)           0.0048          0.3030          0.0158         
Volume Direction Conviction                   0.0098          0.5444          0.0181         
Cumulative Delta Divergence                   0.0203          0.7491          0.0271         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.3008          15.2563         0.0197         

### Absorption vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio_ewm6                  3.2083         
2      price_impact_ratio                       3.1231         
3      volume_efficiency_ratio                  1.5890         
4      rvol_168_scaled                          1.4336         
5      kyle_lambda_proxy                        1.3356         
6      rvol_24_scaled                           1.2942         
7      volume_depth_ratio                       1.2912         
8      kyle_lambda_proxy_ewm6                   1.2853         
9      amihud_spike_ratio_scaled                0.9252         
10     rvol_20                                  0.9242         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.7381          1.5460          1.7710         
Volume Long (rvol_168_scaled)                 2.7058          1.3747          1.9683         
Delta Regime Signal (delta_regime_signal_scaled) 4.5073          13.4024         0.3363         
Delta Align 3h (delta_alignment_3h)           0.0174          0.3134          0.0556         
Volume Direction Conviction                   0.0627          0.5548          0.1130         
Cumulative Delta Divergence                   0.0387          0.7471          0.0518         
Amihud Illiquidity (amihud_spike_ratio_scaled) 8.7166          2.1833          3.9924         

### Absorption vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       4.5391         
2      price_impact_ratio_ewm6                  2.3969         
3      volume_efficiency_ratio                  1.3127         
4      volume_depth_ratio                       1.1897         
5      kyle_lambda_proxy                        1.1570         
6      kyle_lambda_proxy_ewm6                   1.1063         
7      rvol_168_scaled                          1.0232         
8      rvol_24_scaled                           0.9233         
9      rvol_20                                  0.7580         
10     parkinsons_volatility                    0.7197         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.2067          5.6718          0.2128         
Volume Long (rvol_168_scaled)                 1.2065          5.0199          0.2403         
Delta Regime Signal (delta_regime_signal_scaled) 0.8021          52.9012         0.0152         
Delta Align 3h (delta_alignment_3h)           0.0119          0.3178          0.0375         
Volume Direction Conviction                   0.0417          0.5681          0.0735         
Cumulative Delta Divergence                   0.0153          0.7464          0.0205         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.8250          10.8145         0.0763         

### Ghost vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  3.0516         
2      price_impact_ratio                       1.9308         
3      price_impact_ratio_ewm6                  1.7241         
4      volume_depth_ratio                       1.0834         
5      kyle_lambda_proxy                        0.6742         
6      kyle_lambda_proxy_ewm6                   0.6380         
7      rvol_168_scaled                          0.4863         
8      rvol_24_scaled                           0.4296         
9      amihud_spike_ratio_scaled                0.4115         
10     rvol_20                                  0.3969         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.6647          5.8259          0.1141         
Volume Long (rvol_168_scaled)                 0.6621          5.1253          0.1292         
Delta Regime Signal (delta_regime_signal_scaled) 1.4166          56.0755         0.0253         
Delta Align 3h (delta_alignment_3h)           0.0055          0.3045          0.0181         
Volume Direction Conviction                   0.0210          0.5401          0.0389         
Cumulative Delta Divergence                   0.0234          0.7491          0.0312         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.2746          11.6085         0.1098         