# Layer0 Report
- timestamp: 20260120_221442
- symbol: BTCUSDT
- timeframe: 15m
- run_optimization: True
- bundle_path: outcomes/meta_labeling_hpo_sample_weighted/BTCUSDT_15m_20260120_231439/layer0_kalman_bundle.joblib
- loaded_from: 
- n_bars: 788
- date_range: 2026-01-01 00:00:00 -> 2026-01-10 10:00:00

## Best Params
- kalman_Q: 3.1622776601683795e-05
- kalman_R: 0.00031622776601683794
- volume_weight: 0.0
- volume_adaptive: True

## Loss Components
- smoothness_penalty: 3.4715259290879854
- tracking_penalty: 0.5408607266474758
- volume_weight: 0.6000000000000001
- volume_adaptive: False
- parameter_regularization: 0.03390445112371963
- total_loss: 4.04629110685918

## Filter Diagnostics

### Volume_Kalman Filter
- volume_kalman_snr_improvement: 183.240399
- volume_kalman_noise_reduction: 0.994519
- volume_kalman_smoothness_ratio: 1.693841
- volume_kalman_tracking_rmse: 128.411628

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
- volume_kalman_high_freq_reduction: -0.716550
- volume_kalman_low_freq_preservation: 1.002161
