# Layer3 Report
- timestamp: 20251218_224822
- symbol: ETHUSDT
- timeframe: 15m
- n_rows_input: 11607
- n_rows_after_target_dropna: 536
- n_base_models: 9
- winner_scheme: S1_L1
- winner_score: -5.216341611323999

## Winner Metrics (OOF)
- AUC: 0.58649
- PR AUC: 0.49104
- Log Loss: 0.67781
- ECE: 0.09701
- IC: 0.14654
- MCE: 0.16039
- Brier: 0.23983

## Honest Holdout Metrics (Forward Tail)
- n_train: 356
- n_holdout: 81
- honest_auc: 0.639344262295082
- honest_pr_auc: 0.357335713865054
- honest_logloss: 0.778560613822925
- honest_ece: 0.3380697360770104
- honest_brier: 0.2905961548964931

## Weighting Scheme Comparison
| Scheme | Score | AUC | PR_AUC | LogLoss | ECE | Top30_TPD | Top30_Win | Rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_L1 | -5.2163 | 0.5865 | 0.4910 | 0.6778 | 0.0970 | 0.6 | 0.4720 | Toxic |
| S2_L1_L2 | -23.7552 | 0.6250 | 0.5105 | 0.7800 | 0.1883 | 0.6 | 0.5155 | Toxic |
| S9_ClassBalanced | -31.6819 | 0.6164 | 0.4926 | 0.8570 | 0.1999 | 0.6 | 0.5093 | Toxic |
