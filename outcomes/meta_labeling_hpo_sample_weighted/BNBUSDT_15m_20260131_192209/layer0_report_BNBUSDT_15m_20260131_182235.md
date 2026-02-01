# Layer0 Report
- timestamp: 20260131_182235
- symbol: BNBUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/BNBUSDT_15m_20260131_192209/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 17278
- date_range: 2025-07-06 20:45:00 -> 2026-01-02 20:45:00

## Best Params
- kalman_Q: 0.0005324600975689569
- kalman_R: 0.020316027766016837
- volume_weight: 0.30000000000000004
- volume_adaptive: False

## Loss Components
- smoothness_penalty: 3.896863411087554
- tracking_penalty: 0.44514921560208975
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 4.375917077813363

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 2131.113386
- volume_kalman_noise_reduction: 0.999531
- volume_kalman_smoothness_ratio: 1.942463
- volume_kalman_tracking_rmse: 2.782572

### Moving_Average Filter
- moving_average_snr_improvement: 176.834278
- moving_average_noise_reduction: 0.994365
- moving_average_smoothness_ratio: 0.002475
- moving_average_tracking_rmse: 9.650225

### Fisher Filter
- fisher_snr_improvement: 0.000083
- fisher_noise_reduction: -0.000008
- fisher_smoothness_ratio: 0.034175
- fisher_tracking_rmse: 990.829813

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.964329
- fisher_low_freq_preservation: 0.000071
- moving_average_high_freq_reduction: 0.898984
- moving_average_low_freq_preservation: 0.996786
- volume_kalman_high_freq_reduction: -1.000000
- volume_kalman_low_freq_preservation: 0.999957
