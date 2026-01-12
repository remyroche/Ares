# Layer0 Report
- timestamp: 20260112_231953
- symbol: ETHUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/ETHUSDT_15m_20260113_001936/layer0_kalman_bundle.joblib
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
- moving_average_snr_improvement: 0.000000
- moving_average_noise_reduction: nan
- moving_average_smoothness_ratio: nan
- moving_average_tracking_rmse: nan

### Fisher Filter
- fisher_snr_improvement: 0.000000
- fisher_noise_reduction: nan
- fisher_smoothness_ratio: nan
- fisher_tracking_rmse: nan

### Frequency Domain Analysis
- fisher_high_freq_reduction: nan
- fisher_low_freq_preservation: nan
- moving_average_high_freq_reduction: nan
- moving_average_low_freq_preservation: nan
- volume_kalman_high_freq_reduction: -0.182854
- volume_kalman_low_freq_preservation: 1.000435
