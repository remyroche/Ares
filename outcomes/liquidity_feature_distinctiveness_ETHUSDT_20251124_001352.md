# Feature Distinctiveness Report

**Symbol:** ETHUSDT
**Assessment time:** 2025-11-24T00:13:52.649451
**Number of regimes:** 5
**Number of samples:** 33947


====================================================================================================
FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)
====================================================================================================

## Core Dimension WCoV (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       2.2780          2.6490          0.8599         
Volume Long (rvol_168_scaled)                 2.2897          2.1191          1.0805         
Delta Regime Signal (delta_regime_signal_scaled) 24.3770         212.5890        0.1147         
Delta Align 3h (delta_alignment_3h)           0.0189          0.3355          0.0563         
Volume Direction Conviction                   0.0430          0.5644          0.0762         
Cumulative Delta Divergence                   0.0359          0.7572          0.0475         
Amihud Illiquidity (amihud_spike_ratio_scaled) 13.2608         11.0256         1.2027         

## Top Overall Features for Regime Distinction (Between/Within CoV)

Rank   Feature                                  Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
1      volume_depth_ratio                       0.8908          0.4286          2.0783         
2      volume_efficiency_ratio                  0.5392          0.3490          1.5450         
3      price_impact_ratio                       0.5212          0.3982          1.3089         
4      amihud_spike_ratio_scaled                13.2608         11.0256         1.2027         
5      price_impact_ratio_ewm6                  0.4534          0.3874          1.1703         
6      rvol_168_scaled                          2.2897          2.1191          1.0805         
7      intra_bar_vol_estimate                   0.4900          0.4770          1.0272         
8      rvol_24_scaled                           2.2780          2.6490          0.8599         
9      rvol_20                                  0.4457          0.6052          0.7365         
10     parkinsons_volatility                    0.3364          0.4753          0.7078         


## Best Features for Each Regime Pair (Separation Score)


### Apathy vs Valid Trend

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       2.8326         
2      intra_bar_vol_estimate                   2.0839         
3      realized_vol_1h                          1.3762         
4      parkinsons_volatility                    1.3747         
5      rvol_168_scaled                          1.3157         
6      session_vol_percentile                   1.1729         
7      rvol_20                                  1.1700         
8      momentum_vol_alignment_3h_ewm3           1.0562         
9      realized_vol_3h                          1.0549         
10     realized_vol_6h                          1.0490         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.6320          2.5381          0.6430         
Volume Long (rvol_168_scaled)                 1.6281          1.8702          0.8706         
Delta Regime Signal (delta_regime_signal_scaled) 4.6461          13.5471         0.3430         
Delta Align 3h (delta_alignment_3h)           0.0245          0.3387          0.0723         
Volume Direction Conviction                   0.0309          0.5666          0.0545         
Cumulative Delta Divergence                   0.0401          0.7557          0.0531         
Amihud Illiquidity (amihud_spike_ratio_scaled) 1.8641          12.2238         0.1525         

### Apathy vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       1.5632         
2      intra_bar_vol_estimate                   1.3186         
3      parkinsons_volatility                    1.3154         
4      rvol_168_scaled                          1.2938         
5      price_impact_ratio                       1.1931         
6      realized_vol_6h                          1.1852         
7      realized_vol_3h                          1.1421         
8      volume_efficiency_ratio                  1.1098         
9      rvol_24_scaled                           1.0170         
10     realized_vol_1h                          0.9705         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.6266          2.5648          0.6342         
Volume Long (rvol_168_scaled)                 1.5702          1.8831          0.8339         
Delta Regime Signal (delta_regime_signal_scaled) 0.1599          12.5316         0.0128         
Delta Align 3h (delta_alignment_3h)           0.0222          0.3427          0.0648         
Volume Direction Conviction                   0.0115          0.5881          0.0195         
Cumulative Delta Divergence                   0.0324          0.7602          0.0426         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.1662          6.4507          0.0258         

### Apathy vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  2.6407         
2      price_impact_ratio                       2.2774         
3      intra_bar_vol_estimate                   2.2278         
4      price_impact_ratio_ewm6                  2.0689         
5      realized_vol_1h                          1.3167         
6      parkinsons_volatility                    1.2499         
7      realized_vol_6h                          0.9386         
8      realized_vol_3h                          0.9159         
9      session_vol_percentile                   0.8676         
10     vol_ratio_1h_6h                          0.8379         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       8.5340          4.6223          1.8463         
Volume Long (rvol_168_scaled)                 4.1096          3.8001          1.0815         
Delta Regime Signal (delta_regime_signal_scaled) 8.9830          9.5980          0.9359         
Delta Align 3h (delta_alignment_3h)           0.0255          0.3356          0.0760         
Volume Direction Conviction                   0.0498          0.5627          0.0885         
Cumulative Delta Divergence                   0.0179          0.7548          0.0237         
Amihud Illiquidity (amihud_spike_ratio_scaled) 5.3864          7.1247          0.7560         

### Apathy vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  2.5649         
2      price_impact_ratio_ewm6                  1.9758         
3      volume_depth_ratio                       1.8314         
4      price_impact_ratio                       1.6052         
5      rvol_168_scaled                          0.6634         
6      rvol_20                                  0.6499         
7      kyle_lambda_proxy                        0.5153         
8      kyle_lambda_proxy_ewm6                   0.5023         
9      rvol_24_scaled                           0.4378         
10     efficiency_ratio_ewm6                    0.4363         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.4083          2.3615          0.1729         
Volume Long (rvol_168_scaled)                 0.4074          1.5801          0.2578         
Delta Regime Signal (delta_regime_signal_scaled) 1.0218          511.5710        0.0020         
Delta Align 3h (delta_alignment_3h)           0.0119          0.3436          0.0347         
Volume Direction Conviction                   0.0212          0.5841          0.0363         
Cumulative Delta Divergence                   0.0072          0.7592          0.0094         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.7797          9.7523          0.2850         

### Valid Trend vs Absorption

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       1.6986         
2      volume_efficiency_ratio                  1.6800         
3      price_impact_ratio_ewm6                  1.3812         
4      volume_depth_ratio                       1.0797         
5      kyle_lambda_proxy_ewm6                   0.4260         
6      kyle_lambda_proxy                        0.4249         
7      efficiency_ratio_ewm6                    0.3652         
8      range_momentum_divergence                0.3637         
9      session_vol_percentile                   0.3077         
10     efficiency_ratio                         0.3044         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.0032          1.4602          0.0022         
Volume Long (rvol_168_scaled)                 0.0372          1.1962          0.0311         
Delta Regime Signal (delta_regime_signal_scaled) 17.4379         15.5619         1.1206         
Delta Align 3h (delta_alignment_3h)           0.0023          0.3335          0.0068         
Volume Direction Conviction                   0.0423          0.5611          0.0754         
Cumulative Delta Divergence                   0.0077          0.7580          0.0102         
Amihud Illiquidity (amihud_spike_ratio_scaled) 2.4600          13.3496         0.1843         

### Valid Trend vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       3.2252         
2      volume_efficiency_ratio                  1.9640         
3      price_impact_ratio                       1.6365         
4      price_impact_ratio_ewm6                  1.4236         
5      rvol_168_scaled                          0.8680         
6      rvol_20                                  0.8021         
7      intra_bar_vol_estimate                   0.7446         
8      rvol_24_scaled                           0.6421         
9      range_momentum_divergence                0.5566         
10     realized_vol_1h                          0.4617         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.6810          3.5176          0.1936         
Volume Long (rvol_168_scaled)                 0.7460          3.1132          0.2396         
Delta Regime Signal (delta_regime_signal_scaled) 0.3189          12.6283         0.0253         
Delta Align 3h (delta_alignment_3h)           0.0010          0.3265          0.0032         
Volume Direction Conviction                   0.0190          0.5357          0.0354         
Cumulative Delta Divergence                   0.0222          0.7527          0.0295         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.3896          14.0236         0.0278         

### Valid Trend vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_depth_ratio                       4.9030         
2      intra_bar_vol_estimate                   2.0577         
3      volume_efficiency_ratio                  1.8756         
4      rvol_168_scaled                          1.7997         
5      rvol_20                                  1.5442         
6      price_impact_ratio_ewm6                  1.4254         
7      rvol_24_scaled                           1.4240         
8      parkinsons_volatility                    1.3644         
9      realized_vol_1h                          1.2946         
10     session_vol_percentile_ewm6              1.2423         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       3.6680          1.2568          2.9185         
Volume Long (rvol_168_scaled)                 3.6253          0.8932          4.0588         
Delta Regime Signal (delta_regime_signal_scaled) 0.9671          514.6013        0.0019         
Delta Align 3h (delta_alignment_3h)           0.0125          0.3344          0.0375         
Volume Direction Conviction                   0.0097          0.5570          0.0173         
Cumulative Delta Divergence                   0.0473          0.7570          0.0624         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.2189          16.6512         0.0131         

### Absorption vs Ghost

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      price_impact_ratio                       2.6715         
2      price_impact_ratio_ewm6                  2.5521         
3      volume_efficiency_ratio                  2.5443         
4      volume_depth_ratio                       1.5852         
5      rvol_168_scaled                          0.8815         
6      range_momentum_divergence                0.8511         
7      kyle_lambda_proxy_ewm6                   0.7840         
8      kyle_lambda_proxy                        0.7691         
9      rvol_20                                  0.7287         
10     efficiency_ratio_ewm6                    0.6308         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       0.6828          3.5444          0.1926         
Volume Long (rvol_168_scaled)                 0.7621          3.1261          0.2438         
Delta Regime Signal (delta_regime_signal_scaled) 3.7532          11.6129         0.3232         
Delta Align 3h (delta_alignment_3h)           0.0033          0.3305          0.0100         
Volume Direction Conviction                   0.0612          0.5573          0.1099         
Cumulative Delta Divergence                   0.0145          0.7571          0.0191         
Amihud Illiquidity (amihud_spike_ratio_scaled) 49.7646         8.2505          6.0317         

### Absorption vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      volume_efficiency_ratio                  2.5227         
2      price_impact_ratio_ewm6                  2.3955         
3      price_impact_ratio                       1.8681         
4      volume_depth_ratio                       1.7756         
5      rvol_168_scaled                          1.7273         
6      rvol_24_scaled                           1.3784         
7      parkinsons_volatility                    1.2976         
8      intra_bar_vol_estimate                   1.2934         
9      rvol_20                                  1.1759         
10     realized_vol_6h                          1.1372         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       3.6280          1.2836          2.8264         
Volume Long (rvol_168_scaled)                 3.2274          0.9061          3.5619         
Delta Regime Signal (delta_regime_signal_scaled) 1.0302          513.5858        0.0020         
Delta Align 3h (delta_alignment_3h)           0.0103          0.3384          0.0304         
Volume Direction Conviction                   0.0327          0.5786          0.0564         
Cumulative Delta Divergence                   0.0396          0.7614          0.0519         
Amihud Illiquidity (amihud_spike_ratio_scaled) 4.8571          10.8781         0.4465         

### Ghost vs Steamroller

Rank   Feature                                  Separation Score
-----------------------------------------------------------------
1      intra_bar_vol_estimate                   2.2303         
2      volume_depth_ratio                       2.1360         
3      parkinsons_volatility                    1.2498         
4      rvol_168_scaled                          1.2458         
5      rvol_20                                  1.1884         
6      realized_vol_1h                          1.1778         
7      session_vol_percentile                   0.9212         
8      realized_vol_3h                          0.8785         
9      session_vol_percentile_ewm6              0.8779         
10     rvol_24_scaled                           0.8758         

#### Core Dimension WCoV for This Pair (Between/Within CoV Ratios)

Feature                                       Between-CoV     Within-CoV      Distinctiveness
-----------------------------------------------------------------------------------------------
Volume (rvol_24_scaled)                       1.9940          3.3410          0.5968         
Volume Long (rvol_168_scaled)                 1.6891          2.8231          0.5983         
Delta Regime Signal (delta_regime_signal_scaled) 0.9829          510.6522        0.0019         
Delta Align 3h (delta_alignment_3h)           0.0136          0.3314          0.0410         
Volume Direction Conviction                   0.0286          0.5532          0.0517         
Cumulative Delta Divergence                   0.0251          0.7561          0.0332         
Amihud Illiquidity (amihud_spike_ratio_scaled) 0.1866          11.5521         0.0161         