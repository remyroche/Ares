# Label-Based Unified Report
- generated_at: 2025-12-30T00:12:00.606913
- csv_path: outcomes/label_based_unified_report_20251230_001200.csv
- json_path: outcomes/label_based_unified_report_20251230_001200.json

## Context
- direction: long
- exchange: binance
- outcomes_dir: outcomes
- symbol: ETHUSDT
- timeframe: 15m

## Layer2 Metrics
- coverage: 1
- mean_return: 0.000804241
- mean_weight: 1
- median_return: -0.00356679
- median_weight: 0.105943
- n_labeled: 7421
- n_total: 7421
- pos_rate: 0.549252
- std_return: 0.0148235
- std_weight: 5.70703
- weight_entropy_norm: 0.72993

## Layer3 Metrics
- auc: 0.473693
- brier: 0.261031
- ece: 0.126322
- log_loss: 0.715373
- meta_prob_nan: 0
- meta_prob_nan_pct: 0
- n_eval: 7421
- prob_mean: 0.544864
- prob_std: 0.0305629
- size_trade_auc: 0.476844
- size_trade_brier: 0.26203
- size_trade_ece: 0.135361
- size_trade_log_loss: 0.717382
- size_trade_n: 6884
- size_trade_n_eval: 6884
- size_trade_prob_mean: 0.550527
- size_trade_prob_std: 0.0203785
- size_trade_target_is_binary_like: False
- target_is_binary_like: False

## Layer4 Metrics
- layer4_enabled: False
- reason: train_layer4_oof() got an unexpected keyword argument 'l3_quantile_thresholds'

## Layer5 Metrics
- AUC: 0.473693
- Avg Trade PnL: 2.01656e-05
- Maximum Drawdown: 0.224135
- Net Sortino: 0.654008
- Runtime: {'n_rows': 7421, 'total_ms': 2.1242500079097226}
- Total PnL: 0.13882
- Total Return: 0.145034
- Trade Count: 6884
- Turnover Estimate: 896.021
- pmin_sweep_csv: outcomes/layer5_pmin_sweep_20251230_001200.csv
