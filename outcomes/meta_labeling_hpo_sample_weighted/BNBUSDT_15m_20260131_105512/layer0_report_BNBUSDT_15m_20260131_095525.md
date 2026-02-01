# Layer0 Report
- timestamp: 20260131_095525
- symbol: BNBUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/BNBUSDT_15m_20260131_105512/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 180
- date_range: 2026-01-01 00:00:00 -> 2026-01-02 20:45:00

## Best Params
- kalman_Q: 0.0005324600975689569
- kalman_R: 0.020316027766016837
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 1.2076520979097838
- tracking_penalty: 0.03364048345358186
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 1.2751970324870854

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 1010.548183
- volume_kalman_noise_reduction: 0.999009
- volume_kalman_smoothness_ratio: 0.602077
- volume_kalman_tracking_rmse: 0.236878

### Moving_Average Filter
- moving_average_snr_improvement: 3.442627
- moving_average_noise_reduction: 0.814032
- moving_average_smoothness_ratio: 0.002293
- moving_average_tracking_rmse: 3.474794

### Fisher Filter
- fisher_snr_improvement: 0.021316
- fisher_noise_reduction: 0.001653
- fisher_smoothness_ratio: 0.475303
- fisher_tracking_rmse: 866.173875

### Frequency Domain Analysis
- fisher_high_freq_reduction: 0.806125
- fisher_low_freq_preservation: 0.017359
- moving_average_high_freq_reduction: 0.556335
- moving_average_low_freq_preservation: 0.644264
- volume_kalman_high_freq_reduction: 0.291067
- volume_kalman_low_freq_preservation: 1.004103
