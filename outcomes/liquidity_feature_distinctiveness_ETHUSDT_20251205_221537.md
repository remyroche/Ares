# Feature Distinctiveness Report

**Symbol:** ETHUSDT
**Assessment time:** 2025-12-05T22:15:36.398414
**Number of regimes:** 5
**Number of samples:** 8564


====================================================================================================
FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)
====================================================================================================

## Core Dimension WCoV (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       3.5440          4.8656          0.7284         
Volume Long (rvol_168_scaled)                 3.4266          3.5144          0.9750         
Delta Regime Signal (delta_regime_signal_scaled) 4.6493          26.7671         0.1737         
Delta Align 3h (delta_alignment_3h)           0.0248          0.3159          0.0785         
Volume Direction Conviction                   0.0479          0.5657          0.0847         
Cumulative Delta Divergence                   0.0690          0.7692          0.0897         
Amihud Illiquidity (amihud_spike_ratio_scaled) 3.6934          18.7763         0.1967         

## Top Overall Features for Regime Distinction (Between/Within CoV)

Rank   Feature                                  Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
1      volume_depth_ratio                       0.9881          0.3883          2.5449         
2      vwap_distance                            33.2757         16.3091         2.0403         
3      volume_efficiency_ratio                  0.5669          0.3609          1.5708         
4      price_impact_ratio                       0.6551          0.4295          1.5254         
5      kyle_lambda_proxy                        0.7501          0.5451          1.3763         
6      kyle_lambda_proxy_ewm6                   0.7367          0.5588          1.3185         
7      price_impact_ratio_ewm6                  0.5627          0.4632          1.2149         
8      rvol_168_scaled                          3.4266          3.5144          0.9750         
9      intra_bar_vol_estimate                   0.4684          0.5199          0.9008         
10     parkinsons_volatility                    0.3259          0.4446          0.7330         


## Best Features for Each Regime Pair (Separation Score)


### Apathy vs Valid Trend

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       2.2161         
2      kyle_lambda_proxy                        2.0718         
3      volume_depth_ratio                       2.0626         
4      kyle_lambda_proxy_ewm6                   2.0025         
5      volume_efficiency_ratio                  1.7313         
6      parkinsons_volatility                    1.7026         
7      intra_bar_vol_estimate                   1.6394         
8      price_impact_ratio_ewm6                  1.6051         
9      rvol_168_scaled                          1.4801         
10     rvol_24_scaled                           1.2809         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       3.5565          1.5946          2.2303         
Volume Long (rvol_168_scaled)                 3.3995          1.3727          2.4764         
Delta Regime Signal (delta_regime_signal_scaled) 2.9739          11.2702         0.2639         
Delta Align 3h (delta_alignment_3h)           0.0337          0.3178          0.1059         
Volume Direction Conviction                   0.0323          0.5802          0.0556         
Cumulative Delta Divergence                   0.0702          0.7814          0.0898         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.8065          29.7096         0.0271         

### Apathy vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       3.6221         
2      kyle_lambda_proxy                        1.6071         
3      kyle_lambda_proxy_ewm6                   1.5445         
4      intra_bar_vol_estimate                   1.4405         
5      price_impact_ratio                       1.3433         
6      volume_efficiency_ratio                  1.3257         
7      parkinsons_volatility                    1.2942         
8      price_impact_ratio_ewm6                  1.0256         
9      rvol_168_scaled                          0.6555         
10     rvol_20                                  0.6497         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.3628          8.9099          0.1529         
Volume Long (rvol_168_scaled)                 1.4812          6.1131          0.2423         
Delta Regime Signal (delta_regime_signal_scaled) 0.2048          8.8646          0.0231         
Delta Align 3h (delta_alignment_3h)           0.0104          0.3143          0.0330         
Volume Direction Conviction                   0.0033          0.5835          0.0057         
Cumulative Delta Divergence                   0.0126          0.7803          0.0161         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.0835          7.1980          0.0116         

### Apathy vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  2.6117         
2      price_impact_ratio_ewm6                  1.8659         
3      price_impact_ratio                       1.4409         
4      intra_bar_vol_estimate                   1.1017         
5      kyle_lambda_proxy                        1.0200         
6      volume_depth_ratio                       1.0142         
7      kyle_lambda_proxy_ewm6                   1.0124         
8      parkinsons_volatility                    0.6098         
9      session_vol_percentile                   0.5659         
10     vol_of_vol                               0.5303         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.1939          2.3688          0.0819         
Volume Long (rvol_168_scaled)                 0.1690          1.8606          0.0908         
Delta Regime Signal (delta_regime_signal_scaled) 1.1991          44.7815         0.0268         
Delta Align 3h (delta_alignment_3h)           0.0192          0.3100          0.0621         
Volume Direction Conviction                   0.0593          0.5686          0.1042         
Cumulative Delta Divergence                   0.0002          0.7733          0.0003         
Amihud Illiquidity (amihud_spike_ratio_scaled) 7.6771          7.6414          1.0047         

### Apathy vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       2.6152         
2      intra_bar_vol_estimate                   1.7988         
3      parkinsons_volatility                    1.1713         
4      rvol_20                                  1.1159         
5      rvol_168_scaled                          1.1097         
6      session_vol_percentile                   1.0815         
7      momentum_vol_alignment_3h_ewm3           0.9893         
8      rvol_24_scaled                           0.9837         
9      session_vol_percentile_ewm6              0.9547         
10     vol_ratio_1h_6h                          0.8871         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       9.0235          1.9723          4.5752         
Volume Long (rvol_168_scaled)                 8.0163          1.7280          4.6391         
Delta Regime Signal (delta_regime_signal_scaled) 2.4732          12.4542         0.1986         
Delta Align 3h (delta_alignment_3h)           0.0308          0.3012          0.1024         
Volume Direction Conviction                   0.0509          0.5691          0.0894         
Cumulative Delta Divergence                   0.0591          0.7718          0.0766         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.8456          12.8621         0.1435         

### Valid Trend vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       1.4667         
2      kyle_lambda_proxy                        1.0805         
3      kyle_lambda_proxy_ewm6                   1.0243         
4      volume_efficiency_ratio                  0.9296         
5      intra_bar_vol_estimate                   0.8831         
6      parkinsons_volatility                    0.8133         
7      price_impact_ratio                       0.7626         
8      rvol_168_scaled                          0.7599         
9      rvol_24_scaled                           0.6568         
10     price_impact_ratio_ewm6                  0.6275         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.8414          8.7168          0.0965         
Volume Long (rvol_168_scaled)                 0.8087          5.9602          0.1357         
Delta Regime Signal (delta_regime_signal_scaled) 7.0832          13.1662         0.5380         
Delta Align 3h (delta_alignment_3h)           0.0233          0.3296          0.0707         
Volume Direction Conviction                   0.0289          0.5723          0.0506         
Cumulative Delta Divergence                   0.0827          0.7725          0.1071         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.7752          29.9274         0.0259         

### Valid Trend vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio_ewm6                  3.0414         
2      volume_efficiency_ratio                  2.5499         
3      kyle_lambda_proxy                        2.3265         
4      kyle_lambda_proxy_ewm6                   2.2665         
5      volume_depth_ratio                       2.1337         
6      price_impact_ratio                       1.8609         
7      parkinsons_volatility                    1.3599         
8      rvol_168_scaled                          1.3193         
9      intra_bar_vol_estimate                   1.2648         
10     rvol_24_scaled                           1.0878         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.2196          2.1757          1.0202         
Volume Long (rvol_168_scaled)                 2.2662          1.7078          1.3270         
Delta Regime Signal (delta_regime_signal_scaled) 0.6916          49.0831         0.0141         
Delta Align 3h (delta_alignment_3h)           0.0144          0.3254          0.0443         
Volume Direction Conviction                   0.0271          0.5573          0.0486         
Cumulative Delta Divergence                   0.0704          0.7655          0.0920         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.3234          30.3708         0.0436         

### Valid Trend vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       2.2974         
2      volume_efficiency_ratio                  2.0157         
3      price_impact_ratio_ewm6                  1.9529         
4      kyle_lambda_proxy                        1.8896         
5      kyle_lambda_proxy_ewm6                   1.8320         
6      volume_depth_ratio                       1.8186         
7      parkinsons_volatility                    0.8197         
8      intra_bar_vol_estimate                   0.7126         
9      efficiency_ratio_ewm6                    0.6487         
10     efficiency_ratio                         0.5789         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.1758          1.7792          0.0988         
Volume Long (rvol_168_scaled)                 0.1759          1.5751          0.1117         
Delta Regime Signal (delta_regime_signal_scaled) 0.0788          16.7558         0.0047         
Delta Align 3h (delta_alignment_3h)           0.0028          0.3165          0.0089         
Volume Direction Conviction                   0.0186          0.5578          0.0334         
Cumulative Delta Divergence                   0.0111          0.7640          0.0145         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.1271          35.5915         0.0598         

### Absorption vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       4.0308         
2      volume_efficiency_ratio                  2.9746         
3      price_impact_ratio_ewm6                  2.6515         
4      kyle_lambda_proxy                        2.1093         
5      kyle_lambda_proxy_ewm6                   2.0515         
6      price_impact_ratio                       1.7061         
7      efficiency_ratio_ewm6                    0.7562         
8      parkinsons_volatility                    0.7533         
9      intra_bar_vol_estimate                   0.6388         
10     efficiency_ratio                         0.6349         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.5887          9.4910          0.1674         
Volume Long (rvol_168_scaled)                 1.7505          6.4481          0.2715         
Delta Regime Signal (delta_regime_signal_scaled) 1.3180          46.6775         0.0282         
Delta Align 3h (delta_alignment_3h)           0.0089          0.3219          0.0276         
Volume Direction Conviction                   0.0560          0.5607          0.0998         
Cumulative Delta Divergence                   0.0124          0.7644          0.0162         
Amihud Illiquidity (amihud_spike_ratio_scaled) 21.1759         7.8592          2.6944         

### Absorption vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       1.9753         
2      volume_efficiency_ratio                  1.8685         
3      price_impact_ratio                       1.7499         
4      price_impact_ratio_ewm6                  1.4049         
5      kyle_lambda_proxy                        1.2569         
6      kyle_lambda_proxy_ewm6                   1.2260         
7      rvol_168_scaled                          0.4771         
8      momentum_vol_alignment_3h_ewm3           0.4554         
9      efficiency_ratio_ewm6                    0.4489         
10     session_vol_percentile                   0.4463         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.7811          9.0945          0.0859         
Volume Long (rvol_168_scaled)                 0.7377          6.3155          0.1168         
Delta Regime Signal (delta_regime_signal_scaled) 4.5967          14.3503         0.3203         
Delta Align 3h (delta_alignment_3h)           0.0205          0.3130          0.0655         
Volume Direction Conviction                   0.0476          0.5612          0.0847         
Cumulative Delta Divergence                   0.0717          0.7629          0.0939         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.0833          13.0799         0.1593         

### Ghost vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       3.3000         
2      volume_efficiency_ratio                  1.9270         
3      price_impact_ratio_ewm6                  1.5166         
4      kyle_lambda_proxy                        1.1504         
5      price_impact_ratio                       1.1464         
6      kyle_lambda_proxy_ewm6                   1.0899         
7      rvol_20                                  0.9803         
8      intra_bar_vol_estimate                   0.9688         
9      rvol_168_scaled                          0.9631         
10     rvol_24_scaled                           0.8123         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       3.3520          2.5534          1.3128         
Volume Long (rvol_168_scaled)                 3.4757          2.0630          1.6848         
Delta Regime Signal (delta_regime_signal_scaled) 0.6482          50.2671         0.0129         
Delta Align 3h (delta_alignment_3h)           0.0116          0.3088          0.0376         
Volume Direction Conviction                   0.0084          0.5463          0.0154         
Cumulative Delta Divergence                   0.0594          0.7559          0.0785         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.4428          13.5233         0.0327         