# Feature/Trigger Gap Review

## Summary

- Current feature inventory items: 67
- Current trigger inventory items: 72
- Target feature items: 59
- Target trigger items: 14
- Missing feature items: 5
- Missing trigger items: 0

## Current Feature Inventory

| feature_name | source_file | source_function | parameterization | formula_or_description |
| --- | --- | --- | --- | --- |
| open | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | raw open |
| high | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | raw high |
| low | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | raw low |
| close | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | raw close |
| volume | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | raw volume |
| range | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | high - low |
| true_range | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | max(high-low, |high-prev_close|, |low-prev_close|) |
| atr_14 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 14 | EWMA true range alpha=1/14 |
| atr_100 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 100 | EWMA true range alpha=1/100 |
| range_atr | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 14 | range / atr_14 |
| compression_ratio | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 14/100 | atr_14 / atr_100 |
| rolling_range_5 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 5 | shifted rolling max(high,5) - shifted rolling min(low,5) |
| rolling_range_10 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 10 | shifted rolling max(high,10) - shifted rolling min(low,10) |
| rolling_range_20 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 20 | shifted rolling max(high,20) - shifted rolling min(low,20) |
| body | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | abs(close-open) |
| body_ratio | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | abs(close-open)/range |
| upper_wick | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | high - max(open, close) |
| lower_wick | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | min(open, close) - low |
| upper_wick_ratio | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | upper_wick / range |
| lower_wick_ratio | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | lower_wick / range |
| close_location_in_bar | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | (close - low) / range |
| open_location_in_bar | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | (open - low) / range |
| signed_body_ratio | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | (close-open) / range |
| ema_10 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 10 | EMA(close,10) |
| ema_20 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 20 | EMA(close,20) |
| ema_30 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 30 | EMA(close,30) |
| ema_50 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 50 | EMA(close,50) |
| ema_slope_ema20_3 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 3 | ema_20 - ema_20.shift(3) |
| ema_slope_ema20_5 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 5 | ema_20 - ema_20.shift(5) |
| distance_to_ema10 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 10 | close - ema_10 |
| distance_to_ema20 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 20 | close - ema_20 |
| distance_to_ema30 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 30 | close - ema_30 |
| distance_to_ema20_atr | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 20 | (close - ema_20) / atr_14 |
| distance_to_ema50_atr | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 50 | (close - ema_50) / atr_14 |
| trend_alignment_ema20_gt_ema50 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 20/50 | ema_20 > ema_50 |
| returns_1 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 1 | close.pct_change(1) |
| returns_3 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 3 | close.pct_change(3) |
| returns_5 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 5 | close.pct_change(5) |
| returns_10 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 10 | close.pct_change(10) |
| acceleration_close | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | close - 2*close.shift(1) + close.shift(2) |
| acceleration_close_atr | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | acceleration_close / atr_14 |
| volume_ma_20 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 20 | rolling mean(volume,20) |
| volume_spike | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 20 | volume / volume_ma_20 |
| rolling_high_5 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 5 | shifted rolling max(high,5) |
| rolling_high_10 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 10 | shifted rolling max(high,10) |
| rolling_high_20 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 20 | shifted rolling max(high,20) |
| rolling_low_5 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 5 | shifted rolling min(low,5) |
| rolling_low_10 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 10 | shifted rolling min(low,10) |
| rolling_low_20 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 20 | shifted rolling min(low,20) |
| close_gt_rolling_high_5 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 5 | close > shifted rolling high 5 |
| close_lt_rolling_low_5 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 5 | close < shifted rolling low 5 |
| high_gt_rolling_high_5 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 5 | high > shifted rolling high 5 |
| low_lt_rolling_low_5 | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | 5 | low < shifted rolling low 5 |
| bullish_bar | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | close > open |
| bearish_bar | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | close < open |
| prior_bullish_bar | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | bullish_bar shifted by 1 |
| prior_bearish_bar | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | bearish_bar shifted by 1 |
| inside_bar | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | high <= prev_high and low >= prev_low |
| outside_bar | extreme_price_movements/trigger_discovery.py | build_trigger_feature_frame | None | high >= prev_high and low <= prev_low |
| hl_range | extreme_price_movements/mask_optimiser.py | _compute_z_cache | None | rolling robust z-score of high-low range |
| intrabar_range_atr | extreme_price_movements/mask_optimiser.py | _compute_z_cache | None | intrabar range normalized by approximate ATR |
| compression_expansion_transition | extreme_price_movements/mask_optimiser.py | _compute_z_cache | None | range spike divided by rolling bollinger width proxy |
| distance_from_ema_atr | extreme_price_movements/mask_optimiser.py | _compute_z_cache | None | distance from SMA/EMA proxy normalized by ATR proxy |
| volume_robust_z | extreme_price_movements/mask_optimiser.py | _compute_z_cache | None | rolling robust z-score of volume |
| breakout_distance_up_atr | extreme_price_movements/mask_optimiser.py | _compute_z_cache | None | distance from shifted trailing high normalized by ATR proxy |
| breakout_distance_down_atr | extreme_price_movements/mask_optimiser.py | _compute_z_cache | None | distance from shifted trailing low normalized by ATR proxy |
| true_range_percentile | extreme_price_movements/ridge_regime_event_assessment.py | build_regime_features | 168 | rolling percentile rank of true range |

## Current Trigger Inventory

| trigger_family | trigger_name | source_file | source_function | params | semantic_description |
| --- | --- | --- | --- | --- | --- |
| pullback_recovery | close_crosses_above_ema | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'ema_len': 10} | close[t-1] <= ema_10[t-1] and close[t] > ema_10[t] |
| pullback_recovery | ema_reclaim_touch | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'ema_len': 10} | bar touches ema_10 intrabar and closes back through it |
| pullback_recovery | close_crosses_above_ema | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'ema_len': 20} | close[t-1] <= ema_20[t-1] and close[t] > ema_20[t] |
| pullback_recovery | ema_reclaim_touch | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'ema_len': 20} | bar touches ema_20 intrabar and closes back through it |
| pullback_recovery | close_crosses_above_ema | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'ema_len': 30} | close[t-1] <= ema_30[t-1] and close[t] > ema_30[t] |
| pullback_recovery | ema_reclaim_touch | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'ema_len': 30} | bar touches ema_30 intrabar and closes back through it |
| pullback_recovery | reclaim_after_opposite_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'body_ratio_min': 0.4} | close reclaims prior bar extreme after opposite candle |
| pullback_recovery | reclaim_after_opposite_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'body_ratio_min': 0.6} | close reclaims prior bar extreme after opposite candle |
| pullback_recovery | close_in_extreme_of_range | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'close_location_min': 0.7} | close finishes in the directional extreme of the bar |
| pullback_recovery | close_in_extreme_of_range | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'close_location_min': 0.8} | close finishes in the directional extreme of the bar |
| pullback_recovery | close_in_extreme_of_range | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'close_location_min': 0.9} | close finishes in the directional extreme of the bar |
| breakout | simple_close_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5} | close breaks the prior rolling extreme |
| breakout | simple_close_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10} | close breaks the prior rolling extreme |
| breakout | simple_close_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 20} | close breaks the prior rolling extreme |
| breakout | close_gt_rolling_extreme | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'close_location_min': 0.7} | close breaks the prior rolling extreme with a strong close |
| breakout | close_gt_rolling_extreme | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'close_location_min': 0.8} | close breaks the prior rolling extreme with a strong close |
| breakout | close_gt_rolling_extreme | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10, 'close_location_min': 0.7} | close breaks the prior rolling extreme with a strong close |
| breakout | close_gt_rolling_extreme | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10, 'close_location_min': 0.8} | close breaks the prior rolling extreme with a strong close |
| breakout | close_gt_rolling_extreme | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 20, 'close_location_min': 0.7} | close breaks the prior rolling extreme with a strong close |
| breakout | close_gt_rolling_extreme | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 20, 'close_location_min': 0.8} | close breaks the prior rolling extreme with a strong close |
| breakout | high_break_close_near_extreme | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'near_high_threshold': 0.8} | high takes the rolling extreme and close holds near the directional extreme |
| breakout | high_break_close_near_extreme | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'near_high_threshold': 0.9} | high takes the rolling extreme and close holds near the directional extreme |
| breakout | high_break_close_near_extreme | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10, 'near_high_threshold': 0.8} | high takes the rolling extreme and close holds near the directional extreme |
| breakout | high_break_close_near_extreme | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10, 'near_high_threshold': 0.9} | high takes the rolling extreme and close holds near the directional extreme |
| breakout | expansion_body_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'body_ratio_min': 0.6, 'range_atr_min': 1.2} | breakout bar expands in both range and body |
| breakout | expansion_body_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'body_ratio_min': 0.6, 'range_atr_min': 1.5} | breakout bar expands in both range and body |
| breakout | expansion_body_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'body_ratio_min': 0.7, 'range_atr_min': 1.2} | breakout bar expands in both range and body |
| breakout | expansion_body_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'body_ratio_min': 0.7, 'range_atr_min': 1.5} | breakout bar expands in both range and body |
| breakout | expansion_body_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 15, 'body_ratio_min': 0.6, 'range_atr_min': 1.2} | breakout bar expands in both range and body |
| breakout | expansion_body_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 15, 'body_ratio_min': 0.6, 'range_atr_min': 1.5} | breakout bar expands in both range and body |
| breakout | expansion_body_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 15, 'body_ratio_min': 0.7, 'range_atr_min': 1.2} | breakout bar expands in both range and body |
| breakout | expansion_body_breakout | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 15, 'body_ratio_min': 0.7, 'range_atr_min': 1.5} | breakout bar expands in both range and body |
| expansion_impulse | expansion_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'range_atr_min': 1.2} | directional expansion bar with large range |
| expansion_impulse | expansion_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'range_atr_min': 1.5} | directional expansion bar with large range |
| expansion_impulse | expansion_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'range_atr_min': 2.0} | directional expansion bar with large range |
| expansion_impulse | impulse_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'range_atr_min': 1.2, 'body_ratio_min': 0.5} | directional expansion bar with large range and strong body |
| expansion_impulse | impulse_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'range_atr_min': 1.2, 'body_ratio_min': 0.6} | directional expansion bar with large range and strong body |
| expansion_impulse | impulse_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'range_atr_min': 1.5, 'body_ratio_min': 0.5} | directional expansion bar with large range and strong body |
| expansion_impulse | impulse_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'range_atr_min': 1.5, 'body_ratio_min': 0.6} | directional expansion bar with large range and strong body |
| expansion_impulse | impulse_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'range_atr_min': 2.0, 'body_ratio_min': 0.5} | directional expansion bar with large range and strong body |
| expansion_impulse | impulse_bar | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'range_atr_min': 2.0, 'body_ratio_min': 0.6} | directional expansion bar with large range and strong body |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'wick_min': 0.4, 'body_ratio_max': 0.6} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'wick_min': 0.4, 'body_ratio_max': 0.8} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | relaxed_sweep | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'wick_min': 0.4} | bar sweeps the prior extreme and leaves a large rejection wick |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'wick_min': 0.6, 'body_ratio_max': 0.6} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'wick_min': 0.6, 'body_ratio_max': 0.8} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | relaxed_sweep | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 5, 'wick_min': 0.6} | bar sweeps the prior extreme and leaves a large rejection wick |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10, 'wick_min': 0.4, 'body_ratio_max': 0.6} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10, 'wick_min': 0.4, 'body_ratio_max': 0.8} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | relaxed_sweep | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10, 'wick_min': 0.4} | bar sweeps the prior extreme and leaves a large rejection wick |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10, 'wick_min': 0.6, 'body_ratio_max': 0.6} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10, 'wick_min': 0.6, 'body_ratio_max': 0.8} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | relaxed_sweep | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 10, 'wick_min': 0.6} | bar sweeps the prior extreme and leaves a large rejection wick |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 20, 'wick_min': 0.4, 'body_ratio_max': 0.6} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 20, 'wick_min': 0.4, 'body_ratio_max': 0.8} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | relaxed_sweep | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 20, 'wick_min': 0.4} | bar sweeps the prior extreme and leaves a large rejection wick |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 20, 'wick_min': 0.6, 'body_ratio_max': 0.6} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | sweep_reversal | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 20, 'wick_min': 0.6, 'body_ratio_max': 0.8} | bar sweeps the prior extreme then rejects back into range |
| sweep_reversal | relaxed_sweep | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'lookback': 20, 'wick_min': 0.6} | bar sweeps the prior extreme and leaves a large rejection wick |
| compression_release | compression_release | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.5, 'range_atr_min': 1.2} | compression state releases into a directional expansion bar |
| compression_release | compression_release | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.5, 'range_atr_min': 1.5} | compression state releases into a directional expansion bar |
| compression_release | compression_release | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.6, 'range_atr_min': 1.2} | compression state releases into a directional expansion bar |
| compression_release | compression_release | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.6, 'range_atr_min': 1.5} | compression state releases into a directional expansion bar |
| compression_release | compressed_breakout_up_down | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.5, 'lookback': 5} | compression state breaks the prior rolling extreme |
| compression_release | compressed_breakout_up_down | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.5, 'lookback': 10} | compression state breaks the prior rolling extreme |
| compression_release | compressed_breakout_up_down | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.5, 'lookback': 15} | compression state breaks the prior rolling extreme |
| compression_release | compressed_breakout_up_down | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.6, 'lookback': 5} | compression state breaks the prior rolling extreme |
| compression_release | compressed_breakout_up_down | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.6, 'lookback': 10} | compression state breaks the prior rolling extreme |
| compression_release | compressed_breakout_up_down | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.6, 'lookback': 15} | compression state breaks the prior rolling extreme |
| compression_release | compressed_breakout_up_down | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.7, 'lookback': 5} | compression state breaks the prior rolling extreme |
| compression_release | compressed_breakout_up_down | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.7, 'lookback': 10} | compression state breaks the prior rolling extreme |
| compression_release | compressed_breakout_up_down | extreme_price_movements/trigger_discovery.py | generate_trigger_templates | {'compression_ratio_max': 0.7, 'lookback': 15} | compression state breaks the prior rolling extreme |

## Target Feature List

| target_name | family | optional |
| --- | --- | --- |
| range | volatility_range |  |
| true_range | volatility_range |  |
| atr_14 | volatility_range |  |
| atr_100 | volatility_range |  |
| range_atr | volatility_range |  |
| compression_ratio | volatility_range |  |
| rolling_range_5 | volatility_range |  |
| rolling_range_10 | volatility_range |  |
| rolling_range_20 | volatility_range |  |
| body | candle_geometry |  |
| body_ratio | candle_geometry |  |
| upper_wick | candle_geometry |  |
| lower_wick | candle_geometry |  |
| upper_wick_ratio | candle_geometry |  |
| lower_wick_ratio | candle_geometry |  |
| close_location_in_bar | candle_geometry |  |
| open_location_in_bar | candle_geometry | True |
| signed_body_ratio | candle_geometry | True |
| ema_10 | trend_distance |  |
| ema_20 | trend_distance |  |
| ema_30 | trend_distance |  |
| ema_50 | trend_distance |  |
| ema_slope_ema20_3 | trend_distance |  |
| ema_slope_ema20_5 | trend_distance |  |
| ema_slope_ema50_3 | trend_distance | True |
| distance_to_ema10 | trend_distance |  |
| distance_to_ema20 | trend_distance |  |
| distance_to_ema30 | trend_distance |  |
| distance_to_ema20_atr | trend_distance |  |
| distance_to_ema50_atr | trend_distance |  |
| trend_alignment_ema20_gt_ema50 | trend_distance |  |
| returns_1 | momentum |  |
| returns_3 | momentum |  |
| returns_5 | momentum |  |
| returns_10 | momentum |  |
| acceleration_close | momentum |  |
| acceleration_close_atr | momentum |  |
| momentum_sign_N | momentum | True |
| macd_histogram | momentum | True |
| rsi_14 | momentum | True |
| volume_ma_20 | volume |  |
| volume_spike | volume |  |
| volume_zscore_rolling | volume | True |
| rolling_high_5 | structure |  |
| rolling_high_10 | structure |  |
| rolling_high_20 | structure |  |
| rolling_low_5 | structure |  |
| rolling_low_10 | structure |  |
| rolling_low_20 | structure |  |
| close_gt_rolling_high_5 | structure | True |
| close_lt_rolling_low_5 | structure | True |
| high_gt_rolling_high_5 | structure |  |
| low_lt_rolling_low_5 | structure |  |
| bullish_bar | bar_state |  |
| bearish_bar | bar_state |  |
| prior_bullish_bar | bar_state |  |
| prior_bearish_bar | bar_state |  |
| inside_bar | bar_state | True |
| outside_bar | bar_state | True |

## Target Trigger List

| target_name | family |
| --- | --- |
| close_crosses_above_ema | pullback_recovery |
| ema_reclaim_touch | pullback_recovery |
| reclaim_after_opposite_bar | pullback_recovery |
| close_in_extreme_of_range | pullback_recovery |
| simple_close_breakout | breakout |
| close_gt_rolling_extreme | breakout |
| high_break_close_near_extreme | breakout |
| expansion_body_breakout | breakout |
| expansion_bar | expansion_impulse |
| impulse_bar | expansion_impulse |
| sweep_reversal | sweep_reversal |
| relaxed_sweep | sweep_reversal |
| compression_release | compression_release |
| compressed_breakout_up_down | compression_release |

## Match Table For Features

| target_name | status | matched_current_name | notes |
| --- | --- | --- | --- |
| range | exact_match | range |  |
| true_range | exact_match | true_range |  |
| atr_14 | exact_match | atr_14 |  |
| atr_100 | exact_match | atr_100 |  |
| range_atr | exact_match | range_atr |  |
| compression_ratio | exact_match | compression_ratio |  |
| rolling_range_5 | exact_match | rolling_range_5 |  |
| rolling_range_10 | exact_match | rolling_range_10 |  |
| rolling_range_20 | exact_match | rolling_range_20 |  |
| body | exact_match | body |  |
| body_ratio | exact_match | body_ratio |  |
| upper_wick | exact_match | upper_wick |  |
| lower_wick | exact_match | lower_wick |  |
| upper_wick_ratio | exact_match | upper_wick_ratio |  |
| lower_wick_ratio | exact_match | lower_wick_ratio |  |
| close_location_in_bar | exact_match | close_location_in_bar |  |
| open_location_in_bar | exact_match | open_location_in_bar |  |
| signed_body_ratio | exact_match | signed_body_ratio |  |
| ema_10 | exact_match | ema_10 |  |
| ema_20 | exact_match | ema_20 |  |
| ema_30 | exact_match | ema_30 |  |
| ema_50 | exact_match | ema_50 |  |
| ema_slope_ema20_3 | exact_match | ema_slope_ema20_3 |  |
| ema_slope_ema20_5 | exact_match | ema_slope_ema20_5 |  |
| ema_slope_ema50_3 | missing | None | Optional target not implemented. |
| distance_to_ema10 | exact_match | distance_to_ema10 |  |
| distance_to_ema20 | exact_match | distance_to_ema20 |  |
| distance_to_ema30 | exact_match | distance_to_ema30 |  |
| distance_to_ema20_atr | exact_match | distance_to_ema20_atr |  |
| distance_to_ema50_atr | exact_match | distance_to_ema50_atr |  |
| trend_alignment_ema20_gt_ema50 | exact_match | trend_alignment_ema20_gt_ema50 |  |
| returns_1 | exact_match | returns_1 |  |
| returns_3 | exact_match | returns_3 |  |
| returns_5 | exact_match | returns_5 |  |
| returns_10 | exact_match | returns_10 |  |
| acceleration_close | exact_match | acceleration_close |  |
| acceleration_close_atr | exact_match | acceleration_close_atr |  |
| momentum_sign_N | missing | None | Optional target not implemented. |
| macd_histogram | missing | None | Optional target not implemented. |
| rsi_14 | missing | None | Optional target not implemented. |
| volume_ma_20 | exact_match | volume_ma_20 |  |
| volume_spike | exact_match | volume_spike |  |
| volume_zscore_rolling | missing | None | Optional target not implemented. |
| rolling_high_5 | exact_match | rolling_high_5 |  |
| rolling_high_10 | exact_match | rolling_high_10 |  |
| rolling_high_20 | exact_match | rolling_high_20 |  |
| rolling_low_5 | exact_match | rolling_low_5 |  |
| rolling_low_10 | exact_match | rolling_low_10 |  |
| rolling_low_20 | exact_match | rolling_low_20 |  |
| close_gt_rolling_high_5 | exact_match | close_gt_rolling_high_5 |  |
| close_lt_rolling_low_5 | exact_match | close_lt_rolling_low_5 |  |
| high_gt_rolling_high_5 | exact_match | high_gt_rolling_high_5 |  |
| low_lt_rolling_low_5 | exact_match | low_lt_rolling_low_5 |  |
| bullish_bar | exact_match | bullish_bar |  |
| bearish_bar | exact_match | bearish_bar |  |
| prior_bullish_bar | exact_match | prior_bullish_bar |  |
| prior_bearish_bar | exact_match | prior_bearish_bar |  |
| inside_bar | exact_match | inside_bar |  |
| outside_bar | exact_match | outside_bar |  |

## Match Table For Triggers

| target_name | status | matched_current_name | notes |
| --- | --- | --- | --- |
| close_crosses_above_ema | exact_match | close_crosses_above_ema |  |
| ema_reclaim_touch | exact_match | ema_reclaim_touch |  |
| reclaim_after_opposite_bar | exact_match | reclaim_after_opposite_bar |  |
| close_in_extreme_of_range | exact_match | close_in_extreme_of_range |  |
| simple_close_breakout | exact_match | simple_close_breakout |  |
| close_gt_rolling_extreme | exact_match | close_gt_rolling_extreme |  |
| high_break_close_near_extreme | exact_match | high_break_close_near_extreme |  |
| expansion_body_breakout | exact_match | expansion_body_breakout |  |
| expansion_bar | exact_match | expansion_bar |  |
| impulse_bar | exact_match | impulse_bar |  |
| sweep_reversal | exact_match | sweep_reversal |  |
| relaxed_sweep | exact_match | relaxed_sweep |  |
| compression_release | exact_match | compression_release |  |
| compressed_breakout_up_down | exact_match | compressed_breakout_up_down |  |

## Missing Items

- Features: ['ema_slope_ema50_3', 'momentum_sign_N', 'macd_histogram', 'rsi_14', 'volume_zscore_rolling']
- Triggers: None

## Implementation Plan

- Inventory current feature/trigger sources from the trigger discovery and regime search stack.
- Compare against the target reference lists using exact and approximate matching rules.
- Extend the canonical trigger feature frame with missing OHLCV primitives.
- Add missing primitive trigger templates with config-driven toggles and long/short symmetry.
- Regenerate the review artifacts after implementation.

## Post-Implementation Status

- Added features: ['range', 'atr_14', 'atr_100', 'rolling_range_5', 'rolling_range_10', 'rolling_range_20', 'body', 'open_location_in_bar', 'signed_body_ratio', 'ema_50', 'ema_slope_ema20_3', 'ema_slope_ema20_5', 'distance_to_ema10', 'distance_to_ema20', 'distance_to_ema30', 'distance_to_ema20_atr', 'distance_to_ema50_atr', 'trend_alignment_ema20_gt_ema50', 'returns_1', 'returns_3', 'returns_5', 'returns_10', 'acceleration_close', 'acceleration_close_atr', 'volume_ma_20', 'bullish_bar', 'bearish_bar', 'prior_bullish_bar', 'prior_bearish_bar', 'inside_bar', 'outside_bar']
- Added triggers: ['ema_reclaim_touch', 'simple_close_breakout', 'expansion_bar', 'impulse_bar', 'relaxed_sweep', 'compression_release', 'compressed_breakout_up_down']
- Remaining missing features: ['ema_slope_ema50_3', 'momentum_sign_N', 'macd_histogram', 'rsi_14', 'volume_zscore_rolling']
- Remaining missing triggers: None
