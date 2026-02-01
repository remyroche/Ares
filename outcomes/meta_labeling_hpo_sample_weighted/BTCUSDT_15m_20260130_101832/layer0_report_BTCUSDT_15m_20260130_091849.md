# Layer0 Report
- timestamp: 20260130_091849
- symbol: BTCUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/BTCUSDT_15m_20260130_101832/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 788
- date_range: 2026-01-01 00:00:00 -> 2026-01-10 10:00:00

## Best Params
- kalman_Q: 0.0017782794100389228
- kalman_R: 0.1
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 6.302060257943034
- tracking_penalty: 1.1775452914860116
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.026619764744595622
- total_loss: 7.5062253141736415

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 83.847945
- volume_kalman_noise_reduction: 0.988018
- volume_kalman_smoothness_ratio: 3.074928
- volume_kalman_tracking_rmse: 189.474356

### Moving_Average Filter
- moving_average_snr_improvement: 20.751206
- moving_average_noise_reduction: 0.950948
- moving_average_smoothness_ratio: 0.002416
- moving_average_tracking_rmse: 372.857112

### Fisher Filter
- fisher_snr_improvement: 0.000000
- fisher_noise_reduction: -0.000025
- fisher_smoothness_ratio: 0.000022
- fisher_tracking_rmse: 91184.352423

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.999981
- fisher_low_freq_preservation: 0.000000
- moving_average_high_freq_reduction: 0.755246
- moving_average_low_freq_preservation: 1.022778
- volume_kalman_high_freq_reduction: -1.000000
- volume_kalman_low_freq_preservation: 0.999754
