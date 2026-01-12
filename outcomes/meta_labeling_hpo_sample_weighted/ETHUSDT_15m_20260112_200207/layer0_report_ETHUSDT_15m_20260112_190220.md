# Layer0 Report
- timestamp: 20260112_190220
- symbol: ETHUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/ETHUSDT_15m_20260112_200207/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 104508
- date_range: 2022-12-11 14:45:00 -> 2025-12-10 14:45:00

## Best Params
- kalman_Q: 3.1622776601683795e-05
- kalman_R: 0.00031622776601683794
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 1.5275507512904902
- tracking_penalty: 0.07505497840054734
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 1.6365101808147573

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 24706.288443
- volume_kalman_noise_reduction: 0.999960
- volume_kalman_smoothness_ratio: 0.759970
- volume_kalman_tracking_rmse: 3.670436

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
- volume_kalman_high_freq_reduction: 0.172698
- volume_kalman_low_freq_preservation: 1.000079
