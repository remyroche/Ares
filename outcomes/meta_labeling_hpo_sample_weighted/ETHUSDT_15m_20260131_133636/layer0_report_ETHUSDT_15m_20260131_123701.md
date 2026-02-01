# Layer0 Report
- timestamp: 20260131_123701
- symbol: ETHUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/ETHUSDT_15m_20260131_133636/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 1597
- date_range: 2025-05-30 22:00:00 -> 2025-10-31 23:30:00

## Best Params
- kalman_Q: 0.0005324600975689569
- kalman_R: 0.020316027766016837
- volume_weight: 0.1499999999999999
- volume_adaptive: False

## Loss Components
- smoothness_penalty: 2746.3194884675354
- tracking_penalty: 453.84789089925147
- volume_weight: 1.35
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 3200.2012838179107

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 1.550786
- volume_kalman_noise_reduction: -0.788881
- volume_kalman_smoothness_ratio: 100.000000
- volume_kalman_tracking_rmse: 847.665895

### Moving_Average Filter
- moving_average_snr_improvement: 36.471422
- moving_average_noise_reduction: 0.971933
- moving_average_smoothness_ratio: 0.002530
- moving_average_tracking_rmse: 103.214053

### Fisher Filter
- fisher_snr_improvement: 0.000004
- fisher_noise_reduction: -0.000054
- fisher_smoothness_ratio: 0.000398
- fisher_tracking_rmse: 4135.150909

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.999727
- fisher_low_freq_preservation: 0.000004
- moving_average_high_freq_reduction: 0.590826
- moving_average_low_freq_preservation: 1.025386
- volume_kalman_high_freq_reduction: -1.000000
- volume_kalman_low_freq_preservation: 1.432004
