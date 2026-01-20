# Layer0 Report
- timestamp: 20260119_192953
- symbol: ETHUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/ETHUSDT_15m_20260119_202944/layer0_kalman_bundle.joblib
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
- moving_average_snr_improvement: 292.133377
- moving_average_noise_reduction: 0.996580
- moving_average_smoothness_ratio: 0.002501
- moving_average_tracking_rmse: 33.735597

### Fisher Filter
- fisher_snr_improvement: 0.000004
- fisher_noise_reduction: 0.000065
- fisher_smoothness_ratio: 0.003513
- fisher_tracking_rmse: 3877.435048

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.997727
- fisher_low_freq_preservation: 0.000004
- moving_average_high_freq_reduction: 0.518917
- moving_average_low_freq_preservation: 0.999157
- volume_kalman_high_freq_reduction: 0.172698
- volume_kalman_low_freq_preservation: 1.000079
