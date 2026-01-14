# Layer0 Report
- timestamp: 20260114_220827
- symbol: ETHUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_experiment/ETHUSDT_15m_20260114_230824/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 2881
- date_range: 2025-12-04 23:30:00 -> 2026-01-03 23:30:00

## Best Params
- kalman_Q: 1e-08
- kalman_R: 1e-06
- volume_weight: 0.6000000000000001
- volume_adaptive: False

## Loss Components
- smoothness_penalty: 2.9239087303293227
- tracking_penalty: 0.2873216478107868
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.033979224634790015
- total_loss: 3.2452096027748993

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 514.541240
- volume_kalman_noise_reduction: 0.998051
- volume_kalman_smoothness_ratio: 1.467261
- volume_kalman_tracking_rmse: 4.982333

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
- volume_kalman_high_freq_reduction: -0.573933
- volume_kalman_low_freq_preservation: 1.002239
