# Layer0 Report
- timestamp: 20260131_101914
- symbol: BTCUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/BTCUSDT_15m_20260131_111801/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 13828
- date_range: 2025-07-14 10:00:00 -> 2026-01-10 10:00:00

## Best Params
- kalman_Q: 0.0005324600975689569
- kalman_R: 0.020316027766016837
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 109.9643985785577
- tracking_penalty: 17.636154718421327
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 127.63445774810275

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 123.332375
- volume_kalman_noise_reduction: 0.991814
- volume_kalman_smoothness_ratio: 55.167179
- volume_kalman_tracking_rmse: 1060.041926

### Moving_Average Filter
- moving_average_snr_improvement: 349.003709
- moving_average_noise_reduction: 0.997141
- moving_average_smoothness_ratio: 0.002561
- moving_average_tracking_rmse: 627.067809

### Fisher Filter
- fisher_snr_improvement: 0.000000
- fisher_noise_reduction: -0.000001
- fisher_smoothness_ratio: 0.000010
- fisher_tracking_rmse: 102701.169691

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.999993
- fisher_low_freq_preservation: 0.000000
- moving_average_high_freq_reduction: 0.668753
- moving_average_low_freq_preservation: 0.998063
- volume_kalman_high_freq_reduction: -1.000000
- volume_kalman_low_freq_preservation: 1.003364
