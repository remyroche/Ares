# Correlation Analysis

Source: `/Users/remyroche/Documents/Ares/outcomes/meta_labeling_hpo_candidate_pool_ETHUSDT_15m_long_20251210_233144.csv`

Rows: 96

## Correlations with `trades_per_day`

| Feature | Correlation |
| --- | --- |
| r_multiple_pos_threshold | 0.6761 |
| vol_baseline_window | 0.5730 |
| iso_min_prob | 0.3769 |
| kalman_R | 0.3191 |
| stop_to_profit_ratio | 0.1948 |
| profit_mult_min | 0.1112 |
| min_event_spacing | 0.1080 |
| profit_mult_max | 0.0966 |
| stop_mult_max | 0.0838 |
| trail_distance | 0.0762 |
| cusum_threshold | 0.0404 |
| stop_mult_min | -0.0071 |
| target_clip_high_q | -0.0554 |
| profit_thr_base | -0.0681 |
| scale_pos_weight | -0.0859 |
| kalman_Q | -0.3264 |
| label_low_q | -0.5257 |
| econ_min_return_multiple | -0.6990 |
| signal_strength_scale_max | -0.7905 |
| transaction_cost_mult | -0.8295 |
| label_high_q | -0.9057 |
| horizon_bars | nan |
| target_signal_density | nan |


## Correlations with `pnl_per_day`

| Feature | Correlation |
| --- | --- |
| vol_baseline_window | 0.5773 |
| r_multiple_pos_threshold | 0.4219 |
| target_clip_high_q | 0.3209 |
| iso_min_prob | 0.2990 |
| stop_to_profit_ratio | 0.2003 |
| econ_min_return_multiple | 0.0639 |
| kalman_Q | 0.0371 |
| profit_thr_base | -0.0494 |
| transaction_cost_mult | -0.0725 |
| kalman_R | -0.0749 |
| scale_pos_weight | -0.0831 |
| cusum_threshold | -0.1070 |
| label_high_q | -0.1260 |
| signal_strength_scale_max | -0.1650 |
| stop_mult_max | -0.1651 |
| profit_mult_min | -0.1820 |
| trail_distance | -0.2242 |
| profit_mult_max | -0.2530 |
| stop_mult_min | -0.2852 |
| min_event_spacing | -0.4113 |
| label_low_q | -0.4606 |
| horizon_bars | nan |
| target_signal_density | nan |


## Correlations with `pnl_avg_ret_per_trade`

| Feature | Correlation |
| --- | --- |
| kalman_Q | 0.5190 |
| vol_baseline_window | 0.4983 |
| label_high_q | 0.4611 |
| transaction_cost_mult | 0.4307 |
| econ_min_return_multiple | 0.3996 |
| signal_strength_scale_max | 0.3990 |
| stop_to_profit_ratio | 0.2250 |
| label_low_q | 0.1683 |
| profit_thr_base | 0.1217 |
| target_clip_high_q | 0.1216 |
| scale_pos_weight | -0.0522 |
| iso_min_prob | -0.1248 |
| cusum_threshold | -0.2476 |
| r_multiple_pos_threshold | -0.2668 |
| stop_mult_max | -0.3796 |
| profit_mult_min | -0.4929 |
| stop_mult_min | -0.4964 |
| trail_distance | -0.5014 |
| kalman_R | -0.5810 |
| profit_mult_max | -0.6015 |
| min_event_spacing | -0.8564 |
| horizon_bars | nan |
| target_signal_density | nan |


## Correlations with `edge`

| Feature | Correlation |
| --- | --- |
| vol_baseline_window | 0.5882 |
| stop_to_profit_ratio | 0.3812 |
| kalman_Q | 0.2033 |
| signal_strength_scale_max | 0.1567 |
| target_clip_high_q | 0.1294 |
| transaction_cost_mult | 0.0552 |
| econ_min_return_multiple | 0.0528 |
| iso_min_prob | 0.0494 |
| label_high_q | 0.0211 |
| label_low_q | 0.0000 |
| r_multiple_pos_threshold | -0.0000 |
| profit_thr_base | -0.0650 |
| cusum_threshold | -0.2353 |
| scale_pos_weight | -0.2372 |
| kalman_R | -0.2570 |
| stop_mult_max | -0.2901 |
| profit_mult_min | -0.3110 |
| trail_distance | -0.3816 |
| profit_mult_max | -0.3871 |
| stop_mult_min | -0.4059 |
| min_event_spacing | -0.5941 |
| horizon_bars | nan |
| target_signal_density | nan |


## Correlations with `combined`

| Feature | Correlation |
| --- | --- |
| vol_baseline_window | 0.6083 |
| kalman_Q | 0.3540 |
| stop_to_profit_ratio | 0.3355 |
| signal_strength_scale_max | 0.1189 |
| target_clip_high_q | 0.0922 |
| profit_thr_base | 0.0569 |
| econ_min_return_multiple | 0.0512 |
| iso_min_prob | 0.0397 |
| label_high_q | 0.0284 |
| transaction_cost_mult | 0.0224 |
| r_multiple_pos_threshold | 0.0000 |
| label_low_q | -0.0000 |
| scale_pos_weight | -0.0938 |
| cusum_threshold | -0.1618 |
| profit_mult_min | -0.2925 |
| stop_mult_max | -0.3459 |
| stop_mult_min | -0.3502 |
| profit_mult_max | -0.3643 |
| trail_distance | -0.3694 |
| kalman_R | -0.3893 |
| min_event_spacing | -0.6311 |
| horizon_bars | nan |
| target_signal_density | nan |


## Correlations with `mean_auc`

| Feature | Correlation |
| --- | --- |
| min_event_spacing | 0.2126 |
| stop_mult_max | 0.1022 |
| kalman_R | 0.0974 |
| profit_mult_max | 0.0732 |
| stop_mult_min | 0.0691 |
| trail_distance | 0.0639 |
| profit_mult_min | 0.0504 |
| stop_to_profit_ratio | 0.0436 |
| label_high_q | 0.0352 |
| profit_thr_base | 0.0319 |
| econ_min_return_multiple | 0.0225 |
| iso_min_prob | 0.0213 |
| transaction_cost_mult | 0.0124 |
| r_multiple_pos_threshold | 0.0000 |
| label_low_q | -0.0000 |
| cusum_threshold | -0.0002 |
| signal_strength_scale_max | -0.0247 |
| target_clip_high_q | -0.0397 |
| scale_pos_weight | -0.0498 |
| kalman_Q | -0.1072 |
| vol_baseline_window | -0.3155 |
| horizon_bars | nan |
| target_signal_density | nan |

