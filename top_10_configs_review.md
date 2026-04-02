# Detailed Review of Top 10 Configurations

This document presents a detailed review of the top 10 configurations found in `final_rule_registry.csv` (sorted descending by `ridge_profitable_daily_sortino`).

## Overall Summary
The top 10 configurations show remarkable consistency across most key metrics. Interestingly, all top 10 rules share identical performance profiles, suggesting they are capturing highly overlapping or nearly identical subsets of the underlying data, albeit using slightly different structural or threshold features.

All rules share the following core characteristics:
- **Side:** Long
- **Target Variable:** `target_vame`
- **Horizon:** 3
- **Mean Net Return:** ~0.2869
- **Standard Deviation of Net Return:** ~0.0121
- **Mean Support Percentage:** ~16.24%
- **Ridge Profitable Daily Sortino:** ~2,696,477,413.18
- **Take Profit (TP) Rate:** ~31.14%
- **Stop Loss (SL) Rate:** ~65.15%
- **Conditional Win Rate:** ~32.34%
- **Ridge Profitable Win Rate:** ~97.06%
- **Ridge Profitable Avg Net Return:** ~0.0266
- **Ridge Profitable Total Net PnL:** ~105.16

## Configuration Details

### Top 1
- **Canonical Key:** `(*)|(*)|(reg_atr_change_rate_ts_band25_75==0&reg_atr_change_rate_ts_top80==1&reg_ema50_gt_ema200_raw==1)`
- **Review:** This config targets the top 80th percentile and excludes the 25-75th percentile band for the ATR change rate, while also requiring the EMA50 to be strictly greater than EMA200. It effectively targets high volatility expansion regimes in an overall uptrend.

### Top 2
- **Canonical Key:** `(*)|(loc_dist_ema20_atr_ts_band30_70==0)|(reg_atr_change_rate_ts_band20_80==0&reg_ema20_gt_ema50_raw==1)`
- **Review:** This config utilizes location filtering (excluding the 30-70 band of EMA20 distance from ATR) combined with an extreme volatility regime filter (excluding the 20-80 band of ATR change rate) and a shorter-term trend filter (EMA20 > EMA50).

### Top 3
- **Canonical Key:** `(*)|(loc_dist_ema20_atr_ts_top70==1)|(reg_atr_change_rate_ts_top80==1&reg_ema20_gt_ema50_raw==1)`
- **Review:** Features strong overlap with Top 1, but utilizes a location constraint (EMA20 distance top 70) alongside the ATR expansion and shorter trend alignment (EMA20 > EMA50).

### Top 4
- **Canonical Key:** `(*)|(loc_dist_ema20_atr_ts_top70==1)|(reg_atr_change_rate_ts_top80==1&reg_ema50_gt_ema200_raw==1)`
- **Review:** A direct variation of Top 3, swapping out the shorter-term trend filter (EMA20 > EMA50) for the longer-term trend filter (EMA50 > EMA200) present in Top 1.

### Top 5
- **Canonical Key:** `(*)|(loc_dist_ema20_atr_ts_top80==1&loc_dist_ema50_atr_ts_top70==1)|(reg_ema20_gt_ema50_raw==1)`
- **Review:** This rule removes the explicit volatility regime filter, relying entirely on complex location distance filters (both EMA20 and EMA50 top percentiles) combined with a simple trend filter.

### Top 6
- **Canonical Key:** `(*)|(loc_dist_ema20_atr_ts_band25_75==0)|(reg_atr_change_rate_ts_top80==1&reg_ema20_gt_ema50_raw==1)`
- **Review:** Similar to Top 2 and 3, using a location exclusion band (25-75) rather than a top percentile filter, combined with the standard volatility expansion and short-term trend filters.

### Top 7
- **Canonical Key:** `(*)|(loc_dist_ema20_atr_ts_band25_75==0&loc_dist_ema20_atr_ts_bot80==0)|(reg_ema20_gt_ema50_raw==1)`
- **Review:** Uses highly specific location exclusions (band 25-75 and bottom 80) paired simply with a short-term trend filter. The location filters implicitly capture the extreme top percentiles.

### Top 8
- **Canonical Key:** `(*)|(loc_dist_ema20_atr_ts_band20_80==0)|(reg_atr_change_rate_ts_band40_60==0&reg_ema20_gt_ema50_raw==1)`
- **Review:** Excludes the middle 60% (20-80 band) for location and excludes the very middle 20% (40-60 band) of volatility expansion, combined with the short-term trend.

### Top 9
- **Canonical Key:** `(*)|(loc_dist_ema20_atr_ts_band30_70==0&loc_dist_ema20_atr_ts_top80==1)|(reg_ema20_gt_ema50_raw==1)`
- **Review:** Combines a location band exclusion with a location top percentile requirement, effectively isolating very specific upper-percentile location states within a short-term uptrend.

### Top 10
- **Canonical Key:** `(*)|(loc_dist_ema20_atr_ts_top80==1)|(reg_atr_change_rate_ts_band25_75==0&reg_ema20_gt_ema50_raw==1)`
- **Review:** Uses a straightforward location top 80% filter combined with a volatility regime band exclusion and short-term trend filter.

## Conclusion
The highly consistent performance metrics across varying logical combinations suggest these rules have discovered a robust, fundamental market state characterized by trend alignment (either EMA20 > EMA50 or EMA50 > EMA200), high volatility expansion (ATR change rate upper percentiles), and significant positive location offsets. While the exact canonical keys differ, their structural impact on the dataset is effectively identical.
