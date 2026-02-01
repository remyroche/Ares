# Layer0 Report
- timestamp: 20260131_182441
- symbol: ETHUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/ETHUSDT_15m_20260131_192416/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 17281
- date_range: 2025-07-14 12:00:00 -> 2026-01-10 12:00:00

## Best Params
- kalman_Q: 0.0005324600975689569
- kalman_R: 0.020316027766016837
- volume_weight: 0.15000000000000002
- volume_adaptive: False

## Loss Components
- smoothness_penalty: 3.8553051385708175
- tracking_penalty: 0.4622970816821136
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 4.351506671376651

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 3691.462811
- volume_kalman_noise_reduction: 0.999729
- volume_kalman_smoothness_ratio: 1.939248
- volume_kalman_tracking_rmse: 8.656841

### Moving_Average Filter
- moving_average_snr_improvement: 260.413601
- moving_average_noise_reduction: 0.996165
- moving_average_smoothness_ratio: 0.002502
- moving_average_tracking_rmse: 32.562383

### Fisher Filter
- fisher_snr_improvement: 0.000005
- fisher_noise_reduction: -0.000019
- fisher_smoothness_ratio: 0.003961
- fisher_tracking_rmse: 3480.976937

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.997052
- fisher_low_freq_preservation: 0.000004
- moving_average_high_freq_reduction: 0.668122
- moving_average_low_freq_preservation: 0.998823
- volume_kalman_high_freq_reduction: -0.805880
- volume_kalman_low_freq_preservation: 1.000676
