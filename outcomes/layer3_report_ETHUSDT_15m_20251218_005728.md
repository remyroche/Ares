# Layer3 Report
- timestamp: 20251218_005728
- symbol: ETHUSDT
- timeframe: 15m
- n_rows_input: 18780
- n_rows_after_target_dropna: 1634
- n_base_models: 3
- winner_scheme: S3_L2
- winner_score: 1.025978958237328

## Winner Metrics (OOF)
- AUC: 0.54393
- Log Loss: 0.58632
- ECE: 0.04350
- IC: 0.06722
- MCE: 0.20356
- Brier: 0.19771

## Honest Holdout Metrics (Forward Tail)
- n_train: 1288
- n_holdout: 246
- honest_auc: 0.5665008291873963
- honest_logloss: 0.5047231069424098
- honest_ece: 0.11768841649256574
- honest_brier: 0.1607584159594704
