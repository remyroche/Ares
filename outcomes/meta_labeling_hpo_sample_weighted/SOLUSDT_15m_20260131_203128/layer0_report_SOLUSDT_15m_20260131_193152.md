# Layer0 Report
- timestamp: 20260131_193152
- symbol: SOLUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/SOLUSDT_15m_20260131_203128/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 17281
- date_range: 2025-07-15 00:45:00 -> 2026-01-11 00:45:00

## Best Params
- kalman_Q: 0.0005324600975689569
- kalman_R: 0.020316027766016837
- volume_weight: 0.15000000000000002
- volume_adaptive: False

## Loss Components
- smoothness_penalty: 73.34307814711158
- tracking_penalty: 12.384429180113171
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 85.76141177834847

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 176.345001
- volume_kalman_noise_reduction: 0.994291
- volume_kalman_smoothness_ratio: 35.831887
- volume_kalman_tracking_rmse: 2.550823

### Moving_Average Filter
- moving_average_snr_improvement: 366.931607
- moving_average_noise_reduction: 0.997274
- moving_average_smoothness_ratio: 0.002430
- moving_average_tracking_rmse: 1.761918

### Fisher Filter
- fisher_snr_improvement: 0.001287
- fisher_noise_reduction: -0.000839
- fisher_smoothness_ratio: 1.139855
- fisher_tracking_rmse: 162.221711

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.348662
- fisher_low_freq_preservation: 0.001112
- moving_average_high_freq_reduction: 0.514540
- moving_average_low_freq_preservation: 1.000451
- volume_kalman_high_freq_reduction: -1.000000
- volume_kalman_low_freq_preservation: 1.002295
