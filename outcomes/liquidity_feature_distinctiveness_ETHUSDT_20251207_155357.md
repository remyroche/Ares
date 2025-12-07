# Feature Distinctiveness Report

**Symbol:** ETHUSDT
**Assessment time:** 2025-12-07T15:53:57.137060
**Number of regimes:** 5
**Number of samples:** 140354


====================================================================================================
FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)
====================================================================================================

## Core Dimension WCoV (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.4242          4.8180          0.5032         
Volume Long (rvol_168_scaled)                 2.4226          3.4580          0.7006         
Delta Regime Signal (delta_regime_signal_scaled) 3.4035          36.0035         0.0945         
Delta Align 3h (delta_alignment_3h)           0.0170          0.3120          0.0544         
Volume Direction Conviction                   0.0394          0.5543          0.0711         
Cumulative Delta Divergence                   0.0266          0.7482          0.0356         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.6657          9.2489          0.2882         

## Top Overall Features for Regime Distinction (Between/Within CoV)

Rank   Feature                                  Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
1      price_impact_ratio                       0.6269          0.2807          2.2336         
2      volume_efficiency_ratio                  0.5778          0.3360          1.7196         
3      volume_depth_ratio                       0.8609          0.5811          1.4816         
4      price_impact_ratio_ewm6                  0.5151          0.3533          1.4579         
5      kyle_lambda_proxy                        0.5990          0.7138          0.8391         
6      kyle_lambda_proxy_ewm6                   0.5752          0.7355          0.7820         
7      intra_bar_vol_estimate                   0.4812          0.6257          0.7690         
8      rvol_168_scaled                          2.4226          3.4580          0.7006         
9      parkinsons_volatility                    0.3548          0.5720          0.6204         
10     rvol_20                                  0.4132          0.6785          0.6090         


## Best Features for Each Regime Pair (Separation Score)


### Apathy vs Valid Trend

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       2.2245         
2      intra_bar_vol_estimate                   2.0901         
3      parkinsons_volatility                    1.6154         
4      volume_efficiency_ratio                  1.5593         
5      price_impact_ratio                       1.3908         
6      realized_vol_1h                          1.3775         
7      realized_vol_3h                          1.3016         
8      realized_vol_6h                          1.1700         
9      rvol_168_scaled                          1.1690         
10     session_vol_percentile                   1.1272         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.4443          3.9277          0.3677         
Volume Long (rvol_168_scaled)                 1.4786          2.6397          0.5601         
Delta Regime Signal (delta_regime_signal_scaled) 2.2634          38.7990         0.0583         
Delta Align 3h (delta_alignment_3h)           0.0197          0.3123          0.0631         
Volume Direction Conviction                   0.0039          0.5544          0.0071         
Cumulative Delta Divergence                   0.0145          0.7481          0.0193         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.6947          9.0464          0.0768         

### Apathy vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       1.7299         
2      volume_depth_ratio                       1.1533         
3      rvol_168_scaled                          1.0544         
4      price_impact_ratio_ewm6                  1.0252         
5      parkinsons_volatility                    1.0239         
6      kyle_lambda_proxy                        1.0203         
7      intra_bar_vol_estimate                   0.9903         
8      kyle_lambda_proxy_ewm6                   0.9645         
9      realized_vol_3h                          0.9292         
10     realized_vol_6h                          0.9147         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.5055          4.0297          0.3736         
Volume Long (rvol_168_scaled)                 1.5313          2.7241          0.5621         
Delta Regime Signal (delta_regime_signal_scaled) 0.3599          16.0578         0.0224         
Delta Align 3h (delta_alignment_3h)           0.0015          0.3275          0.0045         
Volume Direction Conviction                   0.0374          0.5788          0.0646         
Cumulative Delta Divergence                   0.0072          0.7466          0.0096         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.2040          1.8822          0.1084         

### Apathy vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  4.4163         
2      price_impact_ratio_ewm6                  2.7044         
3      price_impact_ratio                       2.7040         
4      volume_depth_ratio                       1.2642         
5      intra_bar_vol_estimate                   0.9337         
6      kyle_lambda_proxy                        0.9136         
7      kyle_lambda_proxy_ewm6                   0.8850         
8      realized_vol_1h                          0.7539         
9      amihud_spike_ratio_scaled                0.7145         
10     range_momentum_divergence                0.6641         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.3886          4.4971          0.0864         
Volume Long (rvol_168_scaled)                 0.3650          2.9134          0.1253         
Delta Regime Signal (delta_regime_signal_scaled) 7.1071          19.3618         0.3671         
Delta Align 3h (delta_alignment_3h)           0.0194          0.3144          0.0616         
Volume Direction Conviction                   0.0267          0.5504          0.0485         
Cumulative Delta Divergence                   0.0262          0.7485          0.0350         
Amihud Illiquidity (amihud_spike_ratio_scaled) 12.2954         2.7106          4.5360         

### Apathy vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       3.1834         
2      volume_efficiency_ratio                  2.7786         
3      price_impact_ratio_ewm6                  1.5426         
4      intra_bar_vol_estimate                   0.8439         
5      realized_vol_1h                          0.6293         
6      parkinsons_volatility                    0.5434         
7      realized_vol_3h                          0.4488         
8      efficiency_ratio_ewm6                    0.4354         
9      reversal_intensity_ewm3                  0.4238         
10     realized_vol_6h                          0.4131         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.2782          9.0368          0.0308         
Volume Long (rvol_168_scaled)                 0.3384          6.4705          0.0523         
Delta Regime Signal (delta_regime_signal_scaled) 0.5485          48.7868         0.0112         
Delta Align 3h (delta_alignment_3h)           0.0135          0.3186          0.0425         
Volume Direction Conviction                   0.0043          0.5644          0.0076         
Cumulative Delta Divergence                   0.0049          0.7486          0.0065         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.7760          13.0178         0.0596         

### Valid Trend vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       2.3774         
2      price_impact_ratio_ewm6                  1.5672         
3      volume_efficiency_ratio                  1.1269         
4      volume_depth_ratio                       0.7159         
5      kyle_lambda_proxy                        0.6262         
6      kyle_lambda_proxy_ewm6                   0.5898         
7      amihud_spike_ratio_scaled                0.5314         
8      efficiency_ratio_ewm6                    0.4814         
9      session_vol_percentile                   0.4773         
10     efficiency_ratio                         0.4250         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.0521          1.6597          0.0314         
Volume Long (rvol_168_scaled)                 0.0417          1.2954          0.0322         
Delta Regime Signal (delta_regime_signal_scaled) 1.4457          32.8590         0.0440         
Delta Align 3h (delta_alignment_3h)           0.0182          0.3113          0.0585         
Volume Direction Conviction                   0.0334          0.5584          0.0599         
Cumulative Delta Divergence                   0.0073          0.7472          0.0098         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.7871          8.5720          0.0918         

### Valid Trend vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       3.2122         
2      volume_efficiency_ratio                  3.0567         
3      price_impact_ratio                       2.1676         
4      price_impact_ratio_ewm6                  2.0075         
5      intra_bar_vol_estimate                   1.4306         
6      rvol_168_scaled                          1.4154         
7      kyle_lambda_proxy                        1.3002         
8      kyle_lambda_proxy_ewm6                   1.1978         
9      parkinsons_volatility                    1.1747         
10     rvol_20                                  1.1517         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.4062          2.1272          1.1312         
Volume Long (rvol_168_scaled)                 2.4194          1.4846          1.6296         
Delta Regime Signal (delta_regime_signal_scaled) 0.5484          36.1629         0.0152         
Delta Align 3h (delta_alignment_3h)           0.0003          0.2982          0.0012         
Volume Direction Conviction                   0.0306          0.5300          0.0578         
Cumulative Delta Divergence                   0.0407          0.7490          0.0543         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.3615          9.4004          0.1448         

### Valid Trend vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       2.5515         
2      intra_bar_vol_estimate                   1.5579         
3      parkinsons_volatility                    1.1764         
4      volume_efficiency_ratio                  1.1053         
5      rvol_168_scaled                          1.0484         
6      realized_vol_1h                          1.0395         
7      realized_vol_3h                          0.9496         
8      rvol_20                                  0.9298         
9      kyle_lambda_proxy                        0.9117         
10     realized_vol_6h                          0.8543         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.2288          6.6669          0.1843         
Volume Long (rvol_168_scaled)                 1.2110          5.0418          0.2402         
Delta Regime Signal (delta_regime_signal_scaled) 7.1019          65.5880         0.1083         
Delta Align 3h (delta_alignment_3h)           0.0062          0.3024          0.0204         
Volume Direction Conviction                   0.0082          0.5440          0.0151         
Cumulative Delta Divergence                   0.0193          0.7491          0.0258         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.1764          19.7076         0.0090         

### Absorption vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio_ewm6                  3.1297         
2      price_impact_ratio                       3.0383         
3      volume_efficiency_ratio                  1.5877         
4      kyle_lambda_proxy                        1.3378         
5      volume_depth_ratio                       1.2956         
6      rvol_168_scaled                          1.2894         
7      kyle_lambda_proxy_ewm6                   1.2869         
8      rvol_24_scaled                           0.9583         
9      rvol_20                                  0.9201         
10     amihud_spike_ratio_scaled                0.9080         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.6916          2.2292          1.2075         
Volume Long (rvol_168_scaled)                 2.6447          1.5691          1.6854         
Delta Regime Signal (delta_regime_signal_scaled) 4.3311          13.4217         0.3227         
Delta Align 3h (delta_alignment_3h)           0.0179          0.3135          0.0570         
Volume Direction Conviction                   0.0640          0.5543          0.1155         
Cumulative Delta Divergence                   0.0334          0.7476          0.0447         
Amihud Illiquidity (amihud_spike_ratio_scaled) 8.0158          2.2363          3.5844         

### Absorption vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       4.4227         
2      price_impact_ratio_ewm6                  2.3313         
3      volume_efficiency_ratio                  1.2996         
4      volume_depth_ratio                       1.1945         
5      kyle_lambda_proxy                        1.1525         
6      kyle_lambda_proxy_ewm6                   1.1007         
7      rvol_168_scaled                          0.9410         
8      rvol_20                                  0.7666         
9      parkinsons_volatility                    0.7434         
10     intra_bar_vol_estimate                   0.7265         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.2572          6.7689          0.1857         
Volume Long (rvol_168_scaled)                 1.2315          5.1262          0.2402         
Delta Regime Signal (delta_regime_signal_scaled) 0.7586          42.8468         0.0177         
Delta Align 3h (delta_alignment_3h)           0.0121          0.3176          0.0380         
Volume Direction Conviction                   0.0416          0.5684          0.0733         
Cumulative Delta Divergence                   0.0120          0.7477          0.0161         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.8461          12.5435         0.0675         

### Ghost vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  3.0724         
2      price_impact_ratio                       1.9131         
3      price_impact_ratio_ewm6                  1.7350         
4      volume_depth_ratio                       1.0847         
5      kyle_lambda_proxy                        0.6898         
6      kyle_lambda_proxy_ewm6                   0.6545         
7      rvol_168_scaled                          0.4044         
8      amihud_spike_ratio_scaled                0.3935         
9      range_momentum_divergence                0.3756         
10     rvol_20                                  0.3690         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.6018          7.2364          0.0832         
Volume Long (rvol_168_scaled)                 0.6261          5.3155          0.1178         
Delta Regime Signal (delta_regime_signal_scaled) 1.5629          46.1507         0.0339         
Delta Align 3h (delta_alignment_3h)           0.0058          0.3045          0.0191         
Volume Direction Conviction                   0.0224          0.5400          0.0415         
Cumulative Delta Divergence                   0.0214          0.7495          0.0285         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.2400          13.3719         0.0927         