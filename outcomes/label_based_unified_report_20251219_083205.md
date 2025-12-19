# Label-Based Unified Report
- generated_at: 2025-12-19T08:32:05.278180
- csv_path: outcomes/label_based_unified_report_20251219_083205.csv
- json_path: outcomes/label_based_unified_report_20251219_083205.json

## Context
- direction: long
- exchange: binance
- outcomes_dir: outcomes
- symbol: ETHUSDT
- timeframe: 15m

## Layer2 Metrics
- coverage: 0.276201
- mean_return: -0.00321908
- mean_weight: 2.14813
- median_return: -0.00749886
- median_weight: 0.550892
- n_labeled: 3484
- n_total: 12614
- pos_rate: 0.420207
- std_return: 0.00978386
- std_weight: 6.63643
- weight_entropy_norm: 0.829351

## Layer3 Metrics
- auc: 0.5
- brier: 0.586938
- ece: 0.587382
- gate_auc: 0.5
- gate_brier: 0.69291
- gate_ece: 0.693886
- gate_log_loss: 4.17313
- gate_n: 415
- gate_n_eval: 415
- gate_prob_mean: 0.9975
- gate_prob_std: 3.33067e-16
- gate_target_is_binary_like: False
- log_loss: 3.53528
- meta_prob_nan: 697
- meta_prob_nan_pct: 0.200057
- n_eval: 2787
- prob_mean: 0.9975
- prob_std: 3.33067e-16
- size_trade_auc: 0.5
- size_trade_brier: 0.69218
- size_trade_ece: 0.693152
- size_trade_log_loss: 4.16874
- size_trade_n: 414
- size_trade_n_eval: 414
- size_trade_prob_mean: 0.9975
- size_trade_prob_std: 3.33067e-16
- size_trade_target_is_binary_like: False
- target_is_binary_like: False

## Layer4 Metrics
- final_score_mean: 0.385338
- final_score_std: 0.0150577
- layer4_enabled: True
- layer4_final_score_formula: dynamic
- layer4_n_samples: 3484
- layer4_prob_mean: 0.38776
- layer4_prob_std: 0.0138333
- layer4_quantile_threshold: 0.6
- oof: {'n_samples': 3484, 'purge_minutes': 2880, 'embargo_minutes': 1440, 'decision_threshold': 0.5, 'min_trades': 50, 'n_boot': 500, 'l4_oof_coverage': 1.0, 'l4_oof_mean': 0.387759616211812, 'l4_oof_std': 0.013833285536259081, 'grid_results': [{'l3_keep_fraction': 0.7, 'l3_quantile': 0.7, 'l3_threshold': 0.9975, 'formula': 'logit_avg', 'n_samples': 2787, 'n_trades': 2787, 'decision_threshold': 0.5, 'auc': 0.4629038918796951, 'brier': 0.5234566794524677, 'total_pnl': -7.814421681784979, 'avg_pnl': -0.002803882914167556, 'avg_pnl_ci_low': -0.0031358184942508124, 'avg_pnl_ci_high': -0.0024279474793087786, 'win_rate': 0.4101184068891281, 'profit_factor': 0.5351823089199232, 'trades_per_day': 12.413101976431289, 'total_return': -0.9996487328329356, 'max_drawdown': 0.9996655346799691, 'calmar_like': -0.9999831925304509}, {'l3_keep_fraction': 0.6, 'l3_quantile': 0.6, 'l3_threshold': 0.9975, 'formula': 'logit_avg', 'n_samples': 2787, 'n_trades': 2787, 'decision_threshold': 0.5, 'auc': 0.4629038918796951, 'brier': 0.5234566794524677, 'total_pnl': -7.814421681784979, 'avg_pnl': -0.002803882914167556, 'avg_pnl_ci_low': -0.0031358184942508124, 'avg_pnl_ci_high': -0.0024279474793087786, 'win_rate': 0.4101184068891281, 'profit_factor': 0.5351823089199232, 'trades_per_day': 12.413101976431289, 'total_return': -0.9996487328329356, 'max_drawdown': 0.9996655346799691, 'calmar_like': -0.9999831925304509}, {'l3_keep_fraction': 0.5, 'l3_quantile': 0.5, 'l3_threshold': 0.9975, 'formula': 'logit_avg', 'n_samples': 2787, 'n_trades': 2787, 'decision_threshold': 0.5, 'auc': 0.4629038918796951, 'brier': 0.5234566794524677, 'total_pnl': -7.814421681784979, 'avg_pnl': -0.002803882914167556, 'avg_pnl_ci_low': -0.0031358184942508124, 'avg_pnl_ci_high': -0.0024279474793087786, 'win_rate': 0.4101184068891281, 'profit_factor': 0.5351823089199232, 'trades_per_day': 12.413101976431289, 'total_return': -0.9996487328329356, 'max_drawdown': 0.9996655346799691, 'calmar_like': -0.9999831925304509}, {'l3_keep_fraction': 0.4, 'l3_quantile': 0.4, 'l3_threshold': 0.9975, 'formula': 'logit_avg', 'n_samples': 2787, 'n_trades': 2787, 'decision_threshold': 0.5, 'auc': 0.4629038918796951, 'brier': 0.5234566794524677, 'total_pnl': -7.814421681784979, 'avg_pnl': -0.002803882914167556, 'avg_pnl_ci_low': -0.0031358184942508124, 'avg_pnl_ci_high': -0.0024279474793087786, 'win_rate': 0.4101184068891281, 'profit_factor': 0.5351823089199232, 'trades_per_day': 12.413101976431289, 'total_return': -0.9996487328329356, 'max_drawdown': 0.9996655346799691, 'calmar_like': -0.9999831925304509}, {'l3_keep_fraction': 0.3, 'l3_quantile': 0.3, 'l3_threshold': 0.9975, 'formula': 'logit_avg', 'n_samples': 2787, 'n_trades': 2787, 'decision_threshold': 0.5, 'auc': 0.4629038918796951, 'brier': 0.5234566794524677, 'total_pnl': -7.814421681784979, 'avg_pnl': -0.002803882914167556, 'avg_pnl_ci_low': -0.0031358184942508124, 'avg_pnl_ci_high': -0.0024279474793087786, 'win_rate': 0.4101184068891281, 'profit_factor': 0.5351823089199232, 'trades_per_day': 12.413101976431289, 'total_return': -0.9996487328329356, 'max_drawdown': 0.9996655346799691, 'calmar_like': -0.9999831925304509}], 'best_by_auc': {'l3_keep_fraction': 0.7, 'l3_quantile': 0.7, 'l3_threshold': 0.9975, 'formula': 'logit_avg', 'n_samples': 2787, 'n_trades': 2787, 'decision_threshold': 0.5, 'auc': 0.4629038918796951, 'brier': 0.5234566794524677, 'total_pnl': -7.814421681784979, 'avg_pnl': -0.002803882914167556, 'avg_pnl_ci_low': -0.0031358184942508124, 'avg_pnl_ci_high': -0.0024279474793087786, 'win_rate': 0.4101184068891281, 'profit_factor': 0.5351823089199232, 'trades_per_day': 12.413101976431289, 'total_return': -0.9996487328329356, 'max_drawdown': 0.9996655346799691, 'calmar_like': -0.9999831925304509}, 'best_by_pf': {'l3_keep_fraction': 0.7, 'l3_quantile': 0.7, 'l3_threshold': 0.9975, 'formula': 'logit_avg', 'n_samples': 2787, 'n_trades': 2787, 'decision_threshold': 0.5, 'auc': 0.4629038918796951, 'brier': 0.5234566794524677, 'total_pnl': -7.814421681784979, 'avg_pnl': -0.002803882914167556, 'avg_pnl_ci_low': -0.0031358184942508124, 'avg_pnl_ci_high': -0.0024279474793087786, 'win_rate': 0.4101184068891281, 'profit_factor': 0.5351823089199232, 'trades_per_day': 12.413101976431289, 'total_return': -0.9996487328329356, 'max_drawdown': 0.9996655346799691, 'calmar_like': -0.9999831925304509}, 'best': {'l3_keep_fraction': 0.7, 'l3_quantile': 0.7, 'l3_threshold': 0.9975, 'formula': 'logit_avg', 'n_samples': 2787, 'n_trades': 2787, 'decision_threshold': 0.5, 'auc': 0.4629038918796951, 'brier': 0.5234566794524677, 'total_pnl': -7.814421681784979, 'avg_pnl': -0.002803882914167556, 'avg_pnl_ci_low': -0.0031358184942508124, 'avg_pnl_ci_high': -0.0024279474793087786, 'win_rate': 0.4101184068891281, 'profit_factor': 0.5351823089199232, 'trades_per_day': 12.413101976431289, 'total_return': -0.9996487328329356, 'max_drawdown': 0.9996655346799691, 'calmar_like': -0.9999831925304509}}

## Layer5 Metrics
- Average Size: 0.0488513
- Avg Trade PnL: -0.000229834
- Bet Utilization Efficiency: 0
- Configured p_min: 0.9975
- Edge Monotonicity: {'correlation': 0.0, 'bins': {'(0.0, 0.1]': nan, '(0.1, 0.2]': nan, '(0.2, 0.30000000000000004]': nan, '(0.30000000000000004, 0.4]': -0.16749681682978365, '(0.4, 0.5]': -0.26302323200468847, '(0.5, 0.6000000000000001]': nan, '(0.6000000000000001, 0.7000000000000001]': nan, '(0.7000000000000001, 0.8]': nan, '(0.8, 0.9]': nan, '(0.9, 1.0]': nan}}
- Effective Gate Threshold: 0.347047
- Expectancy: -0.000229834
- Exposure: 0.118829
- Gate Count: 415
- Gate Mode: top_k_per_day
- Gate Threshold: 0.347047
- Gross Loss: 0.146075
- Gross Profit: 0.050924
- Maximum Drawdown: 0.0908613
- Median Size: 0.0556126
- Min Trades Reliable: 50
- Net Sortino: -6.74088
- Parameters: {'gamma': 1.2, 'p_min': 0.9975, 'p_max': 0.9, 'gate_mode': 'top_k_per_day', 'gate_quantile': None, 'gate_top_k': None, 'gate_top_k_per_day': 2, 'gate_search_q_low': None, 'gate_search_q_high': None, 'gate_search_min_range': None, 'gate_search_max_iter': None, 'gate_threshold': 0.34704678679281575}
- Payoff Ratio: 0.796833
- PnL Mean (Traded): -0.000229834
- PnL Q10 (Traded): -0.000760397
- PnL Q50 (Traded): -0.000371787
- PnL Q90 (Traded): 0.000448033
- PnL Std (Traded): 0.000488732
- Prob Mean: 0.385338
- Prob Q50: 0.393717
- Prob Q90: 0.39962
- Prob Q99: 0.402515
- Prob Std: 0.0150577
- Prob>=gate_threshold Count: 2754
- Prob>=p_min Count: 0
- Profit Factor: 0.348614
- Return / Drawdown Ratio: -0.999541
- Runtime: {'n_rows': 3484, 'sizing_ms': 4.621625004801899, 'edge_monotonicity_ms': 8.202124998206273, 'total_ms': 12.823750003008172}
- Sharpe-like (Traded): -9.56851
- Tail Loss Amplification: 0.0908623
- Total PnL: -0.0951514
- Total Return: -0.0908196
- Trade Count: 414
- Trades Reliable: True
- Win Rate: 0.304348
- pmin_sweep_csv: outcomes/layer5_pmin_sweep_20251219_083205.csv
