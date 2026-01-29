# Layer0 Report
- timestamp: 20260129_092239
- symbol: ETHUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/ETHUSDT_15m_20260129_092118/layer0_kalman_bundle.joblib
- loaded_from:
- n_bars: 17281
- date_range: 2025-08-02 09:16:31.927405 -> 2026-01-29 09:16:31.927405

## Best Params
- kalman_Q: 0.005000009499999998
- kalman_R: 0.1
- volume_weight: 0.1499999999999999
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 73300015.64707105
- tracking_penalty: 12214418.05695973
- volume_weight: 1.35
- volume_adaptive: False
- parameter_regularization: 0.02346787433130229
- total_loss: 85514433.72749865

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 1.000061
- volume_kalman_noise_reduction: -1.000000
- volume_kalman_smoothness_ratio: 100.000000
- volume_kalman_tracking_rmse: 4276.218500

### Moving_Average Filter
- moving_average_snr_improvement: 196.368556
- moving_average_noise_reduction: 0.994949
- moving_average_smoothness_ratio: 0.002121
- moving_average_tracking_rmse: 2.443343

### Fisher Filter
- fisher_snr_improvement: 0.001104
- fisher_noise_reduction: -0.001466
- fisher_smoothness_ratio: 0.445157
- fisher_tracking_rmse: 984.209169

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.704993
- fisher_low_freq_preservation: 0.000907
- moving_average_high_freq_reduction: 0.633240
- moving_average_low_freq_preservation: 0.992380
- volume_kalman_high_freq_reduction: -1.000000
- volume_kalman_low_freq_preservation: 10.000000
