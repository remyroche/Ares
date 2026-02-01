# Layer0 Report
- timestamp: 20260131_095510
- symbol: SOLUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/SOLUSDT_15m_20260131_105446/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 874
- date_range: 2026-01-01 00:00:00 -> 2026-01-11 00:45:00

## Best Params
- kalman_Q: 0.0005324600975689569
- kalman_R: 0.020316027766016837
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 12.510942840855458
- tracking_penalty: 1.929332173030757
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 14.474179465009934

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 62.661566
- volume_kalman_noise_reduction: 0.983679
- volume_kalman_smoothness_ratio: 6.029828
- volume_kalman_tracking_rmse: 0.571161

### Moving_Average Filter
- moving_average_snr_improvement: 22.549855
- moving_average_noise_reduction: 0.954694
- moving_average_smoothness_ratio: 0.002478
- moving_average_tracking_rmse: 0.925148

### Fisher Filter
- fisher_snr_improvement: 0.069419
- fisher_noise_reduction: -0.071112
- fisher_smoothness_ratio: 3.274277
- fisher_tracking_rmse: 134.516950

### Frequency Domain Analysis
- fisher_high_freq_reduction: -0.939076
- fisher_low_freq_preservation: 0.064201
- moving_average_high_freq_reduction: 0.550606
- moving_average_low_freq_preservation: 1.025522
- volume_kalman_high_freq_reduction: -1.000000
- volume_kalman_low_freq_preservation: 1.011843
