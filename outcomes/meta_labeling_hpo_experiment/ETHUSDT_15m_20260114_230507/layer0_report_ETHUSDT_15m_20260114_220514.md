# Layer0 Report
- timestamp: 20260114_220514
- symbol: ETHUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_experiment/ETHUSDT_15m_20260114_230507/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 105121
- date_range: 2023-01-04 23:30:00 -> 2026-01-03 23:30:00

## Best Params
- kalman_Q: 3.1622776601683795e-05
- kalman_R: 0.00031622776601683794
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 2.5761675396865105
- tracking_penalty: 0.2670889983039135
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 2.8771609891141434

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 6557.832253
- volume_kalman_noise_reduction: 0.999847
- volume_kalman_smoothness_ratio: 1.290742
- volume_kalman_tracking_rmse: 6.713951

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
- volume_kalman_high_freq_reduction: -0.156868
- volume_kalman_low_freq_preservation: 1.000570
