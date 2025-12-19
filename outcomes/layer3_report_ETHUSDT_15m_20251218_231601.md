# Layer3 Report
- timestamp: 20251218_231601
- symbol: ETHUSDT
- timeframe: 15m
- n_rows_input: 11607
- n_rows_after_target_dropna: 536
- n_base_models: 9
- winner_scheme: S3_L2
- winner_score: -25.633838703326777

## Winner Metrics (OOF)
- AUC: 0.52941
- PR AUC: 0.38438
- Log Loss: 0.75085
- ECE: 0.12749
- IC: 0.04965
- MCE: 0.77316
- Brier: 0.26797

## Honest Holdout Metrics (Forward Tail)
- n_train: 356
- n_holdout: 81
- honest_auc: 0.515704584040747
- honest_pr_auc: 0.25443552893836485
- honest_logloss: 1.13583677920694
- honest_ece: 0.4651918663226501
- honest_brier: 0.4187500711497267

## Weighting Scheme Comparison
| Scheme | Score | AUC | PR_AUC | LogLoss | ECE | Top30_TPD | Top30_Win | Rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S3_L2 | -25.6338 | 0.5294 | 0.3844 | 0.7509 | 0.1275 | 0.6 | 0.3374 | Toxic |
| S2_L1_L2 | -48.6176 | 0.5123 | 0.3942 | 0.8155 | 0.2202 | 0.6 | 0.3515 | Toxic |
| S1_L1 | -53.7358 | 0.5229 | 0.3951 | 0.8664 | 0.2385 | 0.6 | 0.3889 | Toxic |
