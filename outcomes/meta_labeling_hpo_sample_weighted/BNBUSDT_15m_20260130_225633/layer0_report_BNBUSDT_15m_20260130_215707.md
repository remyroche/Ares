# Layer0 Report
- timestamp: 20260130_215707
- symbol: BNBUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/BNBUSDT_15m_20260130_225633/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 35345
- date_range: 2025-07-06 20:45:00 -> 2026-01-02 20:45:00

## Best Params
- kalman_Q: 0.0017782794100389228
- kalman_R: 0.1
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 2543.8974271839284
- tracking_penalty: 425.0314356158637
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.026619764744595622
- total_loss: 2968.955482564537

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 1.584189
- volume_kalman_noise_reduction: -0.675091
- volume_kalman_smoothness_ratio: 100.000000
- volume_kalman_tracking_rmse: 40.659429

### Moving_Average Filter
- moving_average_snr_improvement: 57.899137
- moving_average_noise_reduction: 0.982779
- moving_average_smoothness_ratio: 0.001981
- moving_average_tracking_rmse: 4.100472

### Fisher Filter
- fisher_snr_improvement: 0.001335
- fisher_noise_reduction: -0.001972
- fisher_smoothness_ratio: 0.171164
- fisher_tracking_rmse: 875.359988

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.831781
- fisher_low_freq_preservation: 0.001112
- moving_average_high_freq_reduction: 0.920068
- moving_average_low_freq_preservation: 0.998420
- volume_kalman_high_freq_reduction: -1.000000
- volume_kalman_low_freq_preservation: 1.401100
