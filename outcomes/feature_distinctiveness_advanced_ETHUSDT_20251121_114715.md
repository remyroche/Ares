
========================================================================================================================
ADVANCED FEATURE DISTINCTIVENESS VALIDATION
ETHUSDT Liquidity Regimes - Within/Between CoV Analysis
========================================================================================================================

This report shows:
  1. Within-Regime CoV: How consistent each feature is within a regime
  2. Between-Regime CoV: How much the feature varies across regimes
  3. Distinctiveness Score: Between/Within ratio (higher = better separation)


------------------------------------------------------------------------------------------------------------------------
PART 1: TOP FEATURES FOR OVERALL REGIME DISTINCTION
------------------------------------------------------------------------------------------------------------------------

Rank Feature                                       Between-CoV     Within-CoV      Distinctiveness
---- --------------------------------------------- --------------- --------------- ---------------
1    rvol_24                                       0.4414          0.3125          1.4124         
2    ghost_ratio                                   0.3768          0.2710          1.3903         
3    absorption_ratio                              0.4082          0.3128          1.3050         
4    intraday_close_ratio                          0.1892          2.9349          0.0645         
5    forward_return                                2.2879          411.1091        0.0056         

------------------------------------------------------------------------------------------------------------------------
PART 2: PER-REGIME ANALYSIS - FEATURE CONSISTENCY & DISTINCTIVENESS
------------------------------------------------------------------------------------------------------------------------


📊 REGIME 1: VALID TREND
   Description: Strong directional flow with trend persistence
   Expected high features: volume_direction_conviction, trend_confirmation_6h, momentum_persistence_3h
   Expected low features: whipsaw_count, reversal_intensity, ghost_ratio

   ✓ Most Consistent Features (Low Within-CoV = Regime-defining):
     Feature                                       Within-CoV      Mean            Distinctiveness
     --------------------------------------------- --------------- --------------- ---------------
        absorption_ratio                            0.2762          3.553235        1.3050         
     ✅ ghost_ratio                                 0.3117          0.306585        1.3903         
        rvol_24                                     0.4639          1.921439        1.4124         
        intraday_close_ratio                        3.2146          13239.922222    0.0645         
        forward_return                              376.3596        0.000024        0.0056         

   ✓ Most Distinctive Features (Best separate this regime from others):
     Feature                                       Distinctiveness Within-CoV      Mean           
     --------------------------------------------- --------------- --------------- ---------------
        rvol_24                                     1.4124          0.4639          1.921439       
     ✅ ghost_ratio                                 1.3903          0.3117          0.306585       
        absorption_ratio                            1.3050          0.2762          3.553235       
        intraday_close_ratio                        0.0645          3.2146          13239.922222   
        forward_return                              0.0056          376.3596        0.000024       


📊 REGIME 3: GHOST
   Description: Whipsaws and false moves without momentum backing
   Expected high features: whipsaw_count, range_momentum_divergence, ghost_ratio
   Expected low features: vol_momentum_sync, trend_confirmation_6h, volume_direction_conviction

   ✓ Most Consistent Features (Low Within-CoV = Regime-defining):
     Feature                                       Within-CoV      Mean            Distinctiveness
     --------------------------------------------- --------------- --------------- ---------------
        absorption_ratio                            0.1587          4.214075        1.3050         
     ✅ ghost_ratio                                 0.1871          0.244342        1.3903         
        rvol_24                                     0.1929          0.786155        1.4124         
        intraday_close_ratio                        2.9392          12625.366410    0.0645         
        forward_return                              886.4196        -0.000009       0.0056         

   ✓ Most Distinctive Features (Best separate this regime from others):
     Feature                                       Distinctiveness Within-CoV      Mean           
     --------------------------------------------- --------------- --------------- ---------------
        rvol_24                                     1.4124          0.1929          0.786155       
     ✅ ghost_ratio                                 1.3903          0.1871          0.244342       
        absorption_ratio                            1.3050          0.1587          4.214075       
        intraday_close_ratio                        0.0645          2.9392          12625.366410   
        forward_return                              0.0056          886.4196        -0.000009      


📊 REGIME 2: ABSORPTION
   Description: High participation with limited follow-through
   Expected high features: reversal_intensity, pressure_ratio, absorption_ratio
   Expected low features: vol_momentum_sync, ghost_ratio

   ✓ Most Consistent Features (Low Within-CoV = Regime-defining):
     Feature                                       Within-CoV      Mean            Distinctiveness
     --------------------------------------------- --------------- --------------- ---------------
     ✅ ghost_ratio                                 0.2360          0.142610        1.3903         
        rvol_24                                     0.2811          1.359067        1.4124         
     ✅ absorption_ratio                            0.3528          7.578629        1.3050         
        intraday_close_ratio                        2.7222          18903.792962    0.0645         
        forward_return                              42.3913         -0.000182       0.0056         

   ✓ Most Distinctive Features (Best separate this regime from others):
     Feature                                       Distinctiveness Within-CoV      Mean           
     --------------------------------------------- --------------- --------------- ---------------
        rvol_24                                     1.4124          0.2811          1.359067       
     ✅ ghost_ratio                                 1.3903          0.2360          0.142610       
     ✅ absorption_ratio                            1.3050          0.3528          7.578629       
        intraday_close_ratio                        0.0645          2.7222          18903.792962   
        forward_return                              0.0056          42.3913         -0.000182      


📊 REGIME 0: APATHY
   Description: Low signal, noisy, random-like behavior
   Expected high features: ghost_ratio, intraday_close_ratio
   Expected low features: volume_direction_conviction, momentum_persistence_3h

   ✓ Most Consistent Features (Low Within-CoV = Regime-defining):
     Feature                                       Within-CoV      Mean            Distinctiveness
     --------------------------------------------- --------------- --------------- ---------------
        rvol_24                                     0.3121          0.607615        1.4124         
     ✅ ghost_ratio                                 0.3492          0.117719        1.3903         
        absorption_ratio                            0.4637          9.912943        1.3050         
     ✅ intraday_close_ratio                        2.8636          18997.180497    0.0645         
        forward_return                              339.2659        0.000019        0.0056         

   ✓ Most Distinctive Features (Best separate this regime from others):
     Feature                                       Distinctiveness Within-CoV      Mean           
     --------------------------------------------- --------------- --------------- ---------------
        rvol_24                                     1.4124          0.3121          0.607615       
     ✅ ghost_ratio                                 1.3903          0.3492          0.117719       
        absorption_ratio                            1.3050          0.4637          9.912943       
     ✅ intraday_close_ratio                        0.0645          2.8636          18997.180497   
        forward_return                              0.0056          339.2659        0.000019       


------------------------------------------------------------------------------------------------------------------------
PART 3: REGIME-PAIR SEPARATION - WHICH FEATURES BEST DISTINGUISH PAIRS
------------------------------------------------------------------------------------------------------------------------

Valid Trend vs Ghost:
Feature                                       Mean(A)         Mean(B)         Separation     
--------------------------------------------- --------------- --------------- ---------------
intraday_close_ratio                          13239.922222    12625.366410    614.555812     
rvol_24                                       1.921439        0.786155        1.135284       
absorption_ratio                              3.553235        4.214075        0.660840       
ghost_ratio                                   0.306585        0.244342        0.062243       
forward_return                                0.000024        -0.000009       0.000033       

Valid Trend vs Absorption:
Feature                                       Mean(A)         Mean(B)         Separation     
--------------------------------------------- --------------- --------------- ---------------
intraday_close_ratio                          13239.922222    18903.792962    5663.870740    
absorption_ratio                              3.553235        7.578629        4.025394       
rvol_24                                       1.921439        1.359067        0.562372       
ghost_ratio                                   0.306585        0.142610        0.163975       
forward_return                                0.000024        -0.000182       0.000206       

Valid Trend vs Apathy:
Feature                                       Mean(A)         Mean(B)         Separation     
--------------------------------------------- --------------- --------------- ---------------
intraday_close_ratio                          13239.922222    18997.180497    5757.258275    
absorption_ratio                              3.553235        9.912943        6.359708       
rvol_24                                       1.921439        0.607615        1.313824       
ghost_ratio                                   0.306585        0.117719        0.188866       
forward_return                                0.000024        0.000019        0.000005       

Ghost vs Absorption:
Feature                                       Mean(A)         Mean(B)         Separation     
--------------------------------------------- --------------- --------------- ---------------
intraday_close_ratio                          12625.366410    18903.792962    6278.426552    
absorption_ratio                              4.214075        7.578629        3.364554       
rvol_24                                       0.786155        1.359067        0.572912       
ghost_ratio                                   0.244342        0.142610        0.101732       
forward_return                                -0.000009       -0.000182       0.000173       

Ghost vs Apathy:
Feature                                       Mean(A)         Mean(B)         Separation     
--------------------------------------------- --------------- --------------- ---------------
intraday_close_ratio                          12625.366410    18997.180497    6371.814087    
absorption_ratio                              4.214075        9.912943        5.698868       
rvol_24                                       0.786155        0.607615        0.178540       
ghost_ratio                                   0.244342        0.117719        0.126623       
forward_return                                -0.000009       0.000019        0.000028       

Absorption vs Apathy:
Feature                                       Mean(A)         Mean(B)         Separation     
--------------------------------------------- --------------- --------------- ---------------
intraday_close_ratio                          18903.792962    18997.180497    93.387535      
absorption_ratio                              7.578629        9.912943        2.334314       
rvol_24                                       1.359067        0.607615        0.751452       
ghost_ratio                                   0.142610        0.117719        0.024891       
forward_return                                -0.000182       0.000019        0.000201       


========================================================================================================================
SUMMARY & INTERPRETATION
========================================================================================================================

Within-Regime CoV Interpretation:
  - LOW CoV (<0.3): Feature is consistent within regime → GOOD (defines regime)
  - MEDIUM CoV (0.3-0.7): Feature varies moderately → OK
  - HIGH CoV (>0.7): Feature varies widely → POOR (not regime-defining)

Between-Regime CoV Interpretation:
  - HIGH Between-CoV: Feature means differ across regimes → GOOD (distinguishes regimes)
  - LOW Between-CoV: Feature means similar across regimes → POOR (not discriminative)

Distinctiveness Score = Between-CoV / Within-CoV:
  - SCORE > 2.0: Excellent feature (consistent within, varies between)
  - SCORE 1.0-2.0: Good feature
  - SCORE < 1.0: Poor feature (varies within as much as between)

Expected Characteristics Matching (✅):
  ✅ = Feature behavior matches expectation for this regime
  ❌ = Feature behavior contradicts expectation

