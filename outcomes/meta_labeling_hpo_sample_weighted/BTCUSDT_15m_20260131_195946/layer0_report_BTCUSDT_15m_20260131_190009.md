# Layer0 Report
- timestamp: 20260131_190009
- symbol: BTCUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/BTCUSDT_15m_20260131_195946/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 17281
- date_range: 2025-07-14 21:15:00 -> 2026-01-10 21:15:00

## Best Params
- kalman_Q: 0.0005324600975689569
- kalman_R: 0.020316027766016837
- volume_weight: 0.15000000000000002
- volume_adaptive: False

## Loss Components
- smoothness_penalty: 13.558510007015347
- tracking_penalty: 2.0696570975287774
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 15.662071555667843

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 1138.316510
- volume_kalman_noise_reduction: 0.999121
- volume_kalman_smoothness_ratio: 6.806898
- volume_kalman_tracking_rmse: 349.335701

### Moving_Average Filter
- moving_average_snr_improvement: 381.832652
- moving_average_noise_reduction: 0.997384
- moving_average_smoothness_ratio: 0.002542
- moving_average_tracking_rmse: 602.980228

### Fisher Filter
- fisher_snr_improvement: 0.000000
- fisher_noise_reduction: 0.000001
- fisher_smoothness_ratio: 0.000011
- fisher_tracking_rmse: 100077.724753

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.999993
- fisher_low_freq_preservation: 0.000000
- moving_average_high_freq_reduction: 0.614542
- moving_average_low_freq_preservation: 0.999122
- volume_kalman_high_freq_reduction: -1.000000
- volume_kalman_low_freq_preservation: 1.000168
