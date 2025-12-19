# Layer3 Report
- timestamp: 20251218_232229
- symbol: ETHUSDT
- timeframe: 15m
- n_rows_input: 11607
- n_rows_after_target_dropna: 536
- n_base_models: 9
- winner_scheme: S2_L1_L2
- winner_score: -30.30867720561985

## Winner Metrics (OOF)
- AUC: 0.52782
- PR AUC: 0.39321
- Log Loss: 0.73815
- ECE: 0.15685
- IC: 0.04685
- MCE: 0.63322
- Brier: 0.26601

## Honest Holdout Metrics (Forward Tail)
- n_train: 356
- n_holdout: 81
- honest_auc: 0.5038200339558574
- honest_pr_auc: 0.24038257603628735
- honest_logloss: 1.1922853336989958
- honest_ece: 0.5147533043689275
- honest_brier: 0.4500894299490801

## Weighting Scheme Comparison
| Scheme | Score | AUC | PR_AUC | LogLoss | ECE | Top30_TPD | Top30_Win | Rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2_L1_L2 | -30.3087 | 0.5278 | 0.3932 | 0.7382 | 0.1569 | 0.6 | 0.3765 | Toxic |
| S3_L2 | -31.5082 | 0.5382 | 0.4030 | 0.8070 | 0.1533 | 0.6 | 0.4024 | Toxic |
| S1_L1 | -46.3372 | 0.5414 | 0.3977 | 0.8121 | 0.2264 | 0.6 | 0.4110 | Toxic |
