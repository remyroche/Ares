
====================================================================================================
FEATURE DISTINCTIVENESS VALIDATION REPORT
ETHUSDT Liquidity Regimes
====================================================================================================

This report validates that regime characteristics match expected behavior patterns
based on within-regime and between-regime coefficient of variation (CoV) analysis.


----------------------------------------------------------------------------------------------------
REGIME 1: VALID TREND
----------------------------------------------------------------------------------------------------

📋 Description: Strong directional flow with trend persistence

Sample count: 8163

✓ Per-Regime Metrics (Mean, Std, CoV):
Metric                                           Mean          Std          CoV
---------------------------------------- ------------ ------------ ------------
absorption_ratio                               3.5532       0.9813       0.2762
forward_return                                 0.0000       0.0092     376.3596
ghost_ratio                                    0.3066       0.0956       0.3117
intraday_close_ratio                       13239.9222   42560.4880       3.2146
rvol_24                                        1.9214       0.8913       0.4639

✓ Expected High Features:
  (No data available)

✓ Expected Low Features:
  ✅ ghost_ratio_cov                                 0.3117 (expected: low)
  ✅ ghost_ratio_mean                                0.3066 (expected: low)
  ✅ ghost_ratio_std                                 0.0956 (expected: low)

✓ Validation Score: 100.0% (3/3 metrics matched)

----------------------------------------------------------------------------------------------------
REGIME 3: GHOST
----------------------------------------------------------------------------------------------------

📋 Description: Whipsaws and false moves without momentum backing

Sample count: 3375

✓ Per-Regime Metrics (Mean, Std, CoV):
Metric                                           Mean          Std          CoV
---------------------------------------- ------------ ------------ ------------
absorption_ratio                               4.2141       0.6686       0.1587
forward_return                                -0.0000       0.0079     886.4196
ghost_ratio                                    0.2443       0.0457       0.1871
intraday_close_ratio                       12625.3664   37108.6274       2.9392
rvol_24                                        0.7862       0.1517       0.1929

✓ Expected High Features:
  ✅ ghost_ratio_cov                                 0.1871 (expected: high)
  ✅ ghost_ratio_mean                                0.2443 (expected: high)
  ⚠️ ghost_ratio_std                                 0.0457 (expected: high)

✓ Expected Low Features:
  (No data available)

✓ Validation Score: 66.7% (2/3 metrics matched)

----------------------------------------------------------------------------------------------------
REGIME 2: ABSORPTION
----------------------------------------------------------------------------------------------------

📋 Description: High participation with limited follow-through (absorption)

Sample count: 3377

✓ Per-Regime Metrics (Mean, Std, CoV):
Metric                                           Mean          Std          CoV
---------------------------------------- ------------ ------------ ------------
absorption_ratio                               7.5786       2.6736       0.3528
forward_return                                -0.0002       0.0077      42.3913
ghost_ratio                                    0.1426       0.0337       0.2360
intraday_close_ratio                       18903.7930   51460.4740       2.7222
rvol_24                                        1.3591       0.3820       0.2811

✓ Expected High Features:
  ✅ absorption_ratio_cov                            0.3528 (expected: high)
  ✅ absorption_ratio_mean                           7.5786 (expected: high)
  ⚠️ absorption_ratio_std                            2.6736 (expected: high)

✓ Expected Low Features:
  ✅ ghost_ratio_cov                                 0.2360 (expected: low)
  ✅ ghost_ratio_mean                                0.1426 (expected: low)
  ✅ ghost_ratio_std                                 0.0337 (expected: low)

✓ Validation Score: 83.3% (5/6 metrics matched)

----------------------------------------------------------------------------------------------------
REGIME 0: APATHY
----------------------------------------------------------------------------------------------------

📋 Description: Low signal, noisy, random-like behavior

Sample count: 19032

✓ Per-Regime Metrics (Mean, Std, CoV):
Metric                                           Mean          Std          CoV
---------------------------------------- ------------ ------------ ------------
absorption_ratio                               9.9129       4.5970       0.4637
forward_return                                 0.0000       0.0063     339.2659
ghost_ratio                                    0.1177       0.0411       0.3492
intraday_close_ratio                       18997.1805   54399.5121       2.8636
rvol_24                                        0.6076       0.1896       0.3121

✓ Expected High Features:
  ✅ ghost_ratio_cov                                 0.3492 (expected: high)
  ✅ ghost_ratio_mean                                0.1177 (expected: high)
  ⚠️ ghost_ratio_std                                 0.0411 (expected: high)
  ✅ intraday_close_ratio_cov                        2.8636 (expected: high)
  ✅ intraday_close_ratio_mean                     18997.1805 (expected: high)
  ⚠️ intraday_close_ratio_std                      54399.5121 (expected: high)

✓ Expected Low Features:
  (No data available)

✓ Validation Score: 66.7% (4/6 metrics matched)

====================================================================================================
INTERPRETATION GUIDE
====================================================================================================

✅ High Match (80%+): Regime behavior strongly matches expected characteristics
   → Regime is well-defined and distinct

⚠️  Partial Match (50-80%): Some expected characteristics present
   → Regime shows primary behavior but with some contamination

❌ Low Match (<50%): Few expected characteristics present
   → Regime may be poorly defined or overlapping with others

Within-Regime CoV Interpretation:
  - High CoV (>0.5): Feature varies widely within regime → Less distinctive
  - Low CoV (<0.2): Feature is consistent within regime → More distinctive

This indicates the regime is "pure" with consistent behavior across samples.

