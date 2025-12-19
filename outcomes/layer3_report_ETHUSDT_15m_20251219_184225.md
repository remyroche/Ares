# Layer3 Report
- timestamp: 20251219_184225
- symbol: ETHUSDT
- timeframe: 15m
- n_rows_input: 12614
- n_rows_after_target_dropna: 3499
- n_base_models: 10
- winner_scheme: S1_L1
- winner_score: -20.604870089774533

## Winner Metrics (OOF)
- Log Loss: 0.72718
- AUC: 0.52955
- PR AUC: 0.48184
- IC: 0.05116
- ECE: 0.11655
- MCE: 0.81438
- Brier: 0.26442

## Honest Holdout Metrics (Forward Tail)
- n_train: 2924
- n_holdout: 525
- honest_auc: 0.48268514426415804
- honest_pr_auc: 0.34667507352046706
- honest_logloss: 0.7159982600629353
- honest_ece: 0.10723099923486525
- honest_brier: 0.24916204180620158
- honest_temperature: 1.0
- honest_prob_clip_low: 0.0025
- honest_prob_clip_high: 0.9975

## Weighting Scheme Comparison
| Scheme | Score | AUC | PR_AUC | LogLoss | ECE | Top30_TPD | Top30_Win | Rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_L1 | -20.6049 | 0.5295 | 0.4818 | 0.7272 | 0.1165 | 3.1 | 0.5254 | Toxic |
| S10_L1_Qual | -20.6049 | 0.5295 | 0.4818 | 0.7272 | 0.1165 | 3.1 | 0.5254 | Toxic |
| S11_Qual | -20.8663 | 0.5296 | 0.4847 | 0.7389 | 0.1157 | 2.9 | 0.5419 | Toxic |


## Target Definition Comparison
| Target_Type     |    Score | Used_For_Production   |
|:----------------|---------:|:----------------------|
| Economic (Soft) | -25.6988 | True                  |
| Binary (L2)     | -22.0637 | False                 |
