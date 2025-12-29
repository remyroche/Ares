# Layer3 Report
- timestamp: 20251229_012645
- symbol: ETHUSDT
- timeframe: 15m
- n_rows_input: 15353
- n_rows_after_target_dropna: 15353
- n_base_models: 0
- winner_scheme: S1_L1
- winner_score: -100.00735902799727

## Winner Metrics (OOF)
- Log Loss: 0.69315
- AUC: 0.50000
- PR AUC: nan
- IC: 0.00000
- ECE: 0.50000
- MCE: 0.50000
- Brier: 0.00000

## Honest Holdout Metrics (Forward Tail)
- n_train: 13002
- n_holdout: 2303
- honest_auc: nan
- honest_pr_auc: nan
- honest_logloss: 0.6931471805599453
- honest_ece: 0.5
- honest_brier: 0.25
- honest_temperature: nan
- honest_prob_clip_low: 0.0001
- honest_prob_clip_high: 0.9999

## Unified Comparison (Schemes, Models, Goals)
| Category | Name | Goal | Target | Score | AUC | LogLoss | Details |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model Architecture | LGBM Regressor | Regression (Soft) | Soft | -0.0074 | 0.5000 | 0.6931 | Model Race (Holdout) |
| Model Architecture | LGBM Classifier | Binary Classification | Binary | -15.3500 | 0.0000 | 0.0000 | Model Race (Holdout) |
| Weighting Scheme | S1_L1 | Cross Entropy (Default) | l2_consensus_target | -100.0074 | 0.5000 | 0.6931 | Rating: Toxic, Top30_Win: 0.0 |
| Weighting Scheme | S2_L1_L2 | Cross Entropy (Default) | l2_consensus_target | -100.0074 | 0.5000 | 0.6931 | Rating: Toxic, Top30_Win: 0.0 |
| Weighting Scheme | S3_L2 | Cross Entropy (Default) | l2_consensus_target | -100.0074 | 0.5000 | 0.6931 | Rating: Toxic, Top30_Win: 0.0 |


## Weighting Scheme Comparison (Detailed)
| Scheme | Score | AUC | PR_AUC | LogLoss | ECE | Top30_TPD | Top30_Win | Rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_L1 | -100.0074 | 0.5000 | nan | 0.6931 | 0.5000 | 37.3 | 0.0000 | Toxic |
| S2_L1_L2 | -100.0074 | 0.5000 | nan | 0.6931 | 0.5000 | 37.3 | 0.0000 | Toxic |
| S3_L2 | -100.0074 | 0.5000 | nan | 0.6931 | 0.5000 | 37.3 | 0.0000 | Toxic |


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

