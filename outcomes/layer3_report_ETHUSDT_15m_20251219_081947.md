# Layer3 Report
- timestamp: 20251219_081947
- symbol: ETHUSDT
- timeframe: 15m
- n_rows_input: 12614
- n_rows_after_target_dropna: 3484
- n_base_models: 10
- winner_scheme: S1_L1
- winner_score: -259.59038021679896

## Winner Metrics (OOF)
- AUC: 0.50000
- PR AUC: 0.41012
- Log Loss: 3.53528
- ECE: 0.58738
- IC: 0.00000
- MCE: 0.58738
- Brier: 0.43823

## Honest Holdout Metrics (Forward Tail)
- n_train: 2769
- n_holdout: 523
- honest_auc: nan
- honest_pr_auc: nan
- honest_logloss: nan
- honest_ece: nan
- honest_brier: nan
- honest_temperature: 1.0
- honest_prob_clip_low: 0.0025
- honest_prob_clip_high: 0.9975

## Weighting Scheme Comparison
| Scheme | Score | AUC | PR_AUC | LogLoss | ECE | Top30_TPD | Top30_Win | Rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_L1 | -259.5904 | 0.5000 | 0.4101 | 3.5353 | 0.5874 | 9.7 | 0.4101 | Toxic |
| S2_L1_L2 | -259.5904 | 0.5000 | 0.4101 | 3.5353 | 0.5874 | 9.7 | 0.4101 | Toxic |
| S3_L2 | -259.5904 | 0.5000 | 0.4101 | 3.5353 | 0.5874 | 9.7 | 0.4101 | Toxic |
