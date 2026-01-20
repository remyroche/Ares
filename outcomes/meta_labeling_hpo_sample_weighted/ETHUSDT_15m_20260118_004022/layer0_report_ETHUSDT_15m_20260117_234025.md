# Layer0 Report
- timestamp: 20260117_234025
- symbol: ETHUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/ETHUSDT_15m_20260118_004022/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 2881
- date_range: 2025-11-10 14:45:00 -> 2025-12-10 14:45:00

## Best Params
- kalman_Q: 1e-08
- kalman_R: 1e-06
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 2.4679374256714075
- tracking_penalty: 0.2370000256245586
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.033979224634790015
- total_loss: 2.738916675930756

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 1072.586819
- volume_kalman_noise_reduction: 0.999067
- volume_kalman_smoothness_ratio: 1.205183
- volume_kalman_tracking_rmse: 6.143665

### Moving_Average Filter
- moving_average_snr_improvement: 35.354619
- moving_average_noise_reduction: 0.971754
- moving_average_smoothness_ratio: 0.002485
- moving_average_tracking_rmse: 33.306370

### Fisher Filter
- fisher_snr_improvement: 0.000038
- fisher_noise_reduction: 0.000170
- fisher_smoothness_ratio: 0.003782
- fisher_tracking_rmse: 3071.943101

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.996339
- fisher_low_freq_preservation: 0.000033
- moving_average_high_freq_reduction: 0.917077
- moving_average_low_freq_preservation: 1.000143
- volume_kalman_high_freq_reduction: -0.182854
- volume_kalman_low_freq_preservation: 1.000435
