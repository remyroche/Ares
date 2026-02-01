# Layer0 Report
- timestamp: 20260130_104510
- symbol: SOLUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/SOLUSDT_15m_20260130_114503/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 964
- date_range: 2026-01-01 00:00:00 -> 2026-01-11 00:45:00

## Best Params
- kalman_Q: 3.1622776601683795e-05
- kalman_R: 0.00031622776601683794
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 1.2564060138676238
- tracking_penalty: 0.03806496334932988
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 1.3283754283406735

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 3365.871156
- volume_kalman_noise_reduction: 0.999703
- volume_kalman_smoothness_ratio: 0.638918
- volume_kalman_tracking_rmse: 0.075923

### Moving_Average Filter
- moving_average_snr_improvement: 23.199704
- moving_average_noise_reduction: 0.955945
- moving_average_smoothness_ratio: 0.002548
- moving_average_tracking_rmse: 0.897929

### Fisher Filter
- fisher_snr_improvement: 0.070653
- fisher_noise_reduction: -0.062776
- fisher_smoothness_ratio: 3.552516
- fisher_tracking_rmse: 134.706485

### Frequency Domain Analysis
- fisher_high_freq_reduction: -0.974909
- fisher_low_freq_preservation: 0.065322
- moving_average_high_freq_reduction: 0.538658
- moving_average_low_freq_preservation: 1.025669
- volume_kalman_high_freq_reduction: 0.239935
- volume_kalman_low_freq_preservation: 1.000845
