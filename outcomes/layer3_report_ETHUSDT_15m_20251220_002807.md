# Layer3 Report
- timestamp: 20251220_002807
- symbol: ETHUSDT
- timeframe: 15m
- n_rows_input: 12614
- n_rows_after_target_dropna: 8778
- n_base_models: 10
- winner_scheme: S3_L2
- winner_score: -13.691473305621438

## Winner Metrics (OOF)
- Log Loss: 0.63095
- AUC: 0.52869
- PR AUC: 0.29398
- IC: 0.00791
- ECE: 0.10213
- MCE: 0.90506
- Brier: 0.06950

## Honest Holdout Metrics (Forward Tail)
- n_train: 7267
- n_holdout: 1317
- honest_auc: nan
- honest_pr_auc: nan
- honest_logloss: nan
- honest_ece: nan
- honest_brier: nan
- honest_temperature: 1.0
- honest_prob_clip_low: 0.0001
- honest_prob_clip_high: 0.9999

## Weighting Scheme Comparison
| Scheme | Score | AUC | PR_AUC | LogLoss | ECE | Top30_TPD | Top30_Win | Rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S3_L2 | -13.6915 | 0.5287 | 0.2940 | 0.6309 | 0.1021 | 8.8 | 0.3075 | Toxic |
| S12_StabilityWeighted | -13.7715 | 0.5249 | 0.2894 | 0.6314 | 0.0994 | 8.8 | 0.3000 | Toxic |
| S2_L1_L2 | -14.0149 | 0.5290 | 0.2909 | 0.6323 | 0.1028 | 8.8 | 0.3080 | Toxic |
