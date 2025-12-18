# Label-Based Unified Report
- generated_at: 2025-12-18T01:57:28.915060
- csv_path: outcomes/label_based_unified_report_20251218_015728.csv
- json_path: outcomes/label_based_unified_report_20251218_015728.json

## Context
- direction: long
- exchange: binance
- outcomes_dir: outcomes
- symbol: ETHUSDT
- timeframe: 15m

## Layer2 Metrics
- coverage: 0.798136
- mean_return: -0.00297732
- mean_weight: 1.0006
- median_return: -0.0052118
- median_weight: 1
- n_labeled: 14989
- n_total: 18780
- pos_rate: 0.173194
- std_return: 0.00487528
- std_weight: 0.00107604
- weight_entropy_norm: 1

## Layer3 Metrics
- auc: 0.543926
- brier: 0.197705
- ece: 0.043502
- gate_auc: 0.5
- gate_brier: 0.255252
- gate_ece: 0.203562
- gate_log_loss: 0.703503
- gate_n: 6
- gate_n_eval: 6
- gate_prob_mean: 0.536895
- gate_prob_std: 0.0191572
- gate_target_is_binary_like: True
- log_loss: 0.586324
- n_eval: 1634
- prob_mean: 0.270685
- prob_std: 0.083395
- size_trade_auc: 0.5
- size_trade_brier: 0.255252
- size_trade_ece: 0.203562
- size_trade_log_loss: 0.703503
- size_trade_n: 6
- size_trade_n_eval: 6
- size_trade_prob_mean: 0.536895
- size_trade_prob_std: 0.0191572
- size_trade_target_is_binary_like: True
- target_is_binary_like: True

## Layer4 Metrics
- Average Size: 0.0588464
- Avg Trade PnL: 1.67831e-06
- Bet Utilization Efficiency: 0
- Edge Monotonicity: {'correlation': 0.0, 'bins': {'(0.0, 0.1]': nan, '(0.1, 0.2]': nan, '(0.2, 0.30000000000000004]': nan, '(0.30000000000000004, 0.4]': nan, '(0.4, 0.5]': nan, '(0.5, 0.6000000000000001]': nan, '(0.6000000000000001, 0.7000000000000001]': nan, '(0.7000000000000001, 0.8]': nan, '(0.8, 0.9]': nan, '(0.9, 1.0]': nan}}
- Expectancy: 1.67831e-06
- Exposure: 0.00367197
- Gate Count: 6
- Gate Mode: p_min
- Gate Threshold: 0.5
- Gross Loss: 0.0017894
- Gross Profit: 0.00179947
- Maximum Drawdown: 0.00115418
- Median Size: 0.0452382
- Net Sortino: 0.00256687
- Parameters: {'gamma': 1.2, 'p_min': 0.5, 'p_max': 0.9, 'gate_mode': 'p_min', 'gate_quantile': None, 'gate_top_k': None, 'gate_top_k_per_day': None, 'gate_search_q_low': None, 'gate_search_q_high': None, 'gate_search_min_range': None, 'gate_search_max_iter': None, 'gate_threshold': 0.5}
- Payoff Ratio: 2.01126
- PnL Mean (Traded): 1.67831e-06
- PnL Q10 (Traded): -0.000482703
- PnL Q50 (Traded): -0.000411997
- PnL Q90 (Traded): 0.000899735
- PnL Std (Traded): 0.000654072
- Prob Mean: 0.270685
- Prob Q50: 0.278526
- Prob Q90: 0.369647
- Prob Q99: 0.457931
- Prob Std: 0.083395
- Prob>=p_min Count: 6
- Profit Factor: 1.00563
- Return / Drawdown Ratio: 0.00872472
- Runtime: {'n_rows': 1634, 'sizing_ms': 0.0964579958235845, 'edge_monotonicity_ms': 1.2813329958589748, 'total_ms': 1.3777909916825593}
- Sharpe-like (Traded): 0.00628525
- Tail Loss Amplification: 0.000180996
- Total PnL: 1.00699e-05
- Trade Count: 6
- Win Rate: 0.333333
- pmin_sweep_csv: outcomes/layer4_pmin_sweep_20251218_015728.csv
