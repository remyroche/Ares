# Layer3 Report
- timestamp: 20251229_012229
- symbol: ETHUSDT
- timeframe: 15m
- n_rows_input: 15353
- n_rows_after_target_dropna: 15353
- n_base_models: 0
- winner_scheme: S1_L1
- winner_score: -625.8470185988145

## Winner Metrics (OOF)
- Log Loss: 9.21034
- AUC: 0.50000
- PR AUC: nan
- IC: 0.00000
- ECE: 0.99990
- MCE: 0.99990
- Brier: 0.99980

## Honest Holdout Metrics (Forward Tail)
- n_train: 13010
- n_holdout: 2303
- honest_auc: nan
- honest_pr_auc: nan
- honest_logloss: 0.00010000500033334731
- honest_ece: 0.00010000000000320863
- honest_brier: 9.999999999997798e-09
- honest_temperature: 1.0
- honest_prob_clip_low: 0.0001
- honest_prob_clip_high: 0.9999

## Unified Comparison (Schemes, Models, Goals)
| Category | Name | Goal | Target | Score | AUC | LogLoss | Details |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Weighting Scheme | S1_L1 | Cross Entropy (Default) | l2_consensus_target | -625.8470 | 0.5000 | 9.2103 | Rating: Toxic, Top30_Win: 1.0 |
| Weighting Scheme | S2_L1_L2 | Cross Entropy (Default) | l2_consensus_target | -625.8470 | 0.5000 | 9.2103 | Rating: Toxic, Top30_Win: 1.0 |
| Weighting Scheme | S3_L2 | Cross Entropy (Default) | l2_consensus_target | -625.8470 | 0.5000 | 9.2103 | Rating: Toxic, Top30_Win: 1.0 |


## Weighting Scheme Comparison (Detailed)
| Scheme | Score | AUC | PR_AUC | LogLoss | ECE | Top30_TPD | Top30_Win | Rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_L1 | -625.8470 | 0.5000 | nan | 9.2103 | 0.9999 | 37.3 | 1.0000 | Toxic |
| S2_L1_L2 | -625.8470 | 0.5000 | nan | 9.2103 | 0.9999 | 37.3 | 1.0000 | Toxic |
| S3_L2 | -625.8470 | 0.5000 | nan | 9.2103 | 0.9999 | 37.3 | 1.0000 | Toxic |


## SHAP Feature Importance (Global)
### Top 20 Features
| Feature | Mean |SHAP| |
| --- | --- |
| nn_embed_0 | 0.000000 |
| nn_embed_1 | 0.000000 |
| ens_prediction_range | 0.000000 |
| ens_avg_divergence | 0.000000 |
| ens_max_confidence | 0.000000 |
| ens_disagreement_rate | 0.000000 |
| ens_snr_internal | 0.000000 |
| ens_snr_consensus | 0.000000 |
| slope_short | 0.000000 |
| adx_proxy | 0.000000 |
| momentum_short | 0.000000 |
| snr | 0.000000 |
| time_since_last_vol_spike | 0.000000 |
| time_since_last_large_candle | 0.000000 |
| choppiness_index | 0.000000 |
| variance_ratio | 0.000000 |
| permutation_entropy | 0.000000 |
| day_of_week | 0.000000 |
| hour_sin | 0.000000 |
| hour_cos | 0.000000 |



## Target Definition Comparison
| Target_Type     |   Score | Used_For_Production   |
|:----------------|--------:|:----------------------|
| Economic (Soft) |     nan | True                  |
| Binary (L2)     |     nan | False                 |
