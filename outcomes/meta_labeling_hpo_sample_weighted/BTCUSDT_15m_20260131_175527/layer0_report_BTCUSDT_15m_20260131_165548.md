# Layer0 Report
- timestamp: 20260131_165548
- symbol: BTCUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/BTCUSDT_15m_20260131_175527/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 17281
- date_range: 2025-07-14 21:15:00 -> 2026-01-10 21:15:00

## Best Params
- kalman_Q: 0.0005324600975689569
- kalman_R: 0.020316027766016837
- volume_weight: 0.15000000000000002
- volume_adaptive: False

## Loss Components
- smoothness_penalty: 1.917604504686617
- tracking_penalty: 0.14045033722678735
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 2.0919592930371236

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 16641.052008
- volume_kalman_noise_reduction: 0.999940
- volume_kalman_smoothness_ratio: 0.954758
- volume_kalman_tracking_rmse: 91.334739

### Moving_Average Filter
- moving_average_snr_improvement: 381.476131
- moving_average_noise_reduction: 0.997381
- moving_average_smoothness_ratio: 0.002507
- moving_average_tracking_rmse: 603.249050

### Fisher Filter
- fisher_snr_improvement: 0.000000
- fisher_noise_reduction: 0.000001
- fisher_smoothness_ratio: 0.000011
- fisher_tracking_rmse: 100077.627674

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.999993
- fisher_low_freq_preservation: 0.000000
- moving_average_high_freq_reduction: 0.619988
- moving_average_low_freq_preservation: 0.999121
- volume_kalman_high_freq_reduction: 0.056099
- volume_kalman_low_freq_preservation: 0.999939
