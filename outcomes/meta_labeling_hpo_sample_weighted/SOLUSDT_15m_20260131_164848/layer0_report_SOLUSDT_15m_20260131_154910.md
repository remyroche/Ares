# Layer0 Report
- timestamp: 20260131_154910
- symbol: SOLUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/SOLUSDT_15m_20260131_164848/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 13855
- date_range: 2025-07-15 00:45:00 -> 2026-01-11 00:45:00

## Best Params
- kalman_Q: 0.0005324600975689569
- kalman_R: 0.020316027766016837
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 41.808494848427564
- tracking_penalty: 6.925873660477042
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 48.76827296002833

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 346.386405
- volume_kalman_noise_reduction: 0.997107
- volume_kalman_smoothness_ratio: 20.454839
- volume_kalman_tracking_rmse: 2.104599

### Moving_Average Filter
- moving_average_snr_improvement: 406.125794
- moving_average_noise_reduction: 0.997541
- moving_average_smoothness_ratio: 0.002481
- moving_average_tracking_rmse: 1.941778

### Fisher Filter
- fisher_snr_improvement: 0.000950
- fisher_noise_reduction: -0.001611
- fisher_smoothness_ratio: 0.928351
- fisher_tracking_rmse: 174.663205

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.385312
- fisher_low_freq_preservation: 0.000821
- moving_average_high_freq_reduction: 0.583875
- moving_average_low_freq_preservation: 0.998952
- volume_kalman_high_freq_reduction: -1.000000
- volume_kalman_low_freq_preservation: 1.000057
