# Label-Based Unified Report
- generated_at: 2025-12-19T00:22:48.926880
- csv_path: outcomes/label_based_unified_report_20251219_002248.csv
- json_path: outcomes/label_based_unified_report_20251219_002248.json

## Context
- direction: long
- exchange: binance
- outcomes_dir: outcomes
- symbol: ETHUSDT
- timeframe: 15m

## Layer2 Metrics
- coverage: 0.291117
- mean_return: -0.00292183
- mean_weight: 2.1036
- median_return: -0.00648006
- median_weight: 0.556055
- n_labeled: 3379
- n_total: 11607
- pos_rate: 0.371708
- std_return: 0.00908171
- std_weight: 6.74808
- weight_entropy_norm: 0.824589

## Layer3 Metrics
- auc: 0.527824
- brier: 0.266011
- ece: 0.156855
- gate_auc: 0.8
- gate_brier: 0.591944
- gate_ece: 0.688255
- gate_log_loss: 1.78034
- gate_n: 6
- gate_n_eval: 6
- gate_prob_mean: 0.854922
- gate_prob_std: 0.0800224
- gate_target_is_binary_like: True
- log_loss: 0.738152
- n_eval: 536
- prob_mean: 0.476149
- prob_std: 0.171859
- size_trade_auc: 0.8
- size_trade_brier: 0.591944
- size_trade_ece: 0.688255
- size_trade_log_loss: 1.78034
- size_trade_n: 6
- size_trade_n_eval: 6
- size_trade_prob_mean: 0.854922
- size_trade_prob_std: 0.0800224
- size_trade_target_is_binary_like: True
- target_is_binary_like: True

## Layer4 Metrics
- final_score_mean: 0.189629
- final_score_std: 0.0723618
- layer4_enabled: True
- layer4_final_score_formula: dynamic
- layer4_n_samples: 536
- layer4_prob_mean: 0.406149
- layer4_prob_std: 0.0733113
- layer4_quantile_threshold: 0.6
- oof: {'n_samples': 536, 'purge_minutes': 60, 'embargo_minutes': 30, 'decision_threshold': 0.5, 'min_trades': 50, 'n_boot': 500, 'l4_oof_coverage': 1.0, 'l4_oof_mean': 0.4061494634865468, 'l4_oof_std': 0.07331128688908066, 'grid_results': [{'l3_keep_fraction': 0.7, 'l3_quantile': 0.7, 'l3_threshold': 0.3965864304813981, 'formula': 'logit_avg', 'n_samples': 376, 'n_trades': 112, 'decision_threshold': 0.5, 'auc': 0.3578706591807029, 'brier': 0.2629231817441764, 'total_pnl': -0.6741351776380924, 'avg_pnl': -0.006019064086054396, 'avg_pnl_ci_low': -0.008693220196265929, 'avg_pnl_ci_high': -0.0028406757813494057, 'win_rate': 0.2767857142857143, 'profit_factor': 0.4597350233596941, 'trades_per_day': 0.4260073695471294, 'total_return': -0.4985715221806848, 'max_drawdown': 0.5066708175512419, 'calmar_like': -0.9840146795691032}, {'l3_keep_fraction': 0.6, 'l3_quantile': 0.6, 'l3_threshold': 0.4457852833984175, 'formula': 'logit_avg', 'n_samples': 322, 'n_trades': 135, 'decision_threshold': 0.5, 'auc': 0.3125598086124402, 'brier': 0.2740698529862388, 'total_pnl': -0.8224639846308062, 'avg_pnl': -0.006092325812080046, 'avg_pnl_ci_low': -0.008941488702531171, 'avg_pnl_ci_high': -0.00367581144486378, 'win_rate': 0.2740740740740741, 'profit_factor': 0.449639231772551, 'trades_per_day': 0.4822863947603453, 'total_return': -0.5688581246824496, 'max_drawdown': 0.5745980711443177, 'calmar_like': -0.9900105016861143}, {'l3_keep_fraction': 0.5, 'l3_quantile': 0.5, 'l3_threshold': 0.4795077623043944, 'formula': 'min', 'n_samples': 268, 'n_trades': 80, 'decision_threshold': 0.5, 'auc': 0.30475698628564873, 'brier': 0.29955485214938926, 'total_pnl': -0.5393368784298389, 'avg_pnl': -0.0067417109803729865, 'avg_pnl_ci_low': -0.009847943218418384, 'avg_pnl_ci_high': -0.0032020230906309125, 'win_rate': 0.25, 'profit_factor': 0.4232170626373714, 'trades_per_day': 1.8677042801556418, 'total_return': -0.4237783480781624, 'max_drawdown': 0.459824090429244, 'calmar_like': -0.9216097131440969}, {'l3_keep_fraction': 0.5, 'l3_quantile': 0.5, 'l3_threshold': 0.4795077623043944, 'formula': 'logit_avg', 'n_samples': 268, 'n_trades': 177, 'decision_threshold': 0.5, 'auc': 0.29511677282377924, 'brier': 0.2867313236313904, 'total_pnl': -1.1100581637710665, 'avg_pnl': -0.006271515049554048, 'avg_pnl_ci_low': -0.008614385516003959, 'avg_pnl_ci_high': -0.004130083617251259, 'win_rate': 0.2824858757062147, 'profit_factor': 0.43334043082571627, 'trades_per_day': 0.6194451532937189, 'total_return': -0.678253529676107, 'max_drawdown': 0.6825370558647751, 'calmar_like': -0.9937241118956763}, {'l3_keep_fraction': 0.4, 'l3_quantile': 0.4, 'l3_threshold': 0.5569671405724586, 'formula': 'min', 'n_samples': 215, 'n_trades': 69, 'decision_threshold': 0.5, 'auc': 0.3031924380238987, 'brier': 0.2936297462772733, 'total_pnl': -0.33766984128849564, 'avg_pnl': -0.004893765815775299, 'avg_pnl_ci_low': -0.008368247072976991, 'avg_pnl_ci_high': -0.0007194985472123996, 'win_rate': 0.2898550724637681, 'profit_factor': 0.5395896901078107, 'trades_per_day': 1.610894941634241, 'total_return': -0.29368930173283747, 'max_drawdown': 0.33787280883403215, 'calmar_like': -0.8692303554863231}, {'l3_keep_fraction': 0.4, 'l3_quantile': 0.4, 'l3_threshold': 0.5569671405724586, 'formula': 'logit_avg', 'n_samples': 215, 'n_trades': 158, 'decision_threshold': 0.5, 'auc': 0.29061886927055464, 'brier': 0.28637577424188976, 'total_pnl': -0.8200987792474828, 'avg_pnl': -0.00519049860283217, 'avg_pnl_ci_low': -0.007902488795629947, 'avg_pnl_ci_high': -0.0025058448688876924, 'win_rate': 0.3037974683544304, 'profit_factor': 0.5010347676335958, 'trades_per_day': 0.6009746820397005, 'total_return': -0.5686158912265867, 'max_drawdown': 0.5755837988538425, 'calmar_like': -0.9878941908335176}, {'l3_keep_fraction': 0.3, 'l3_quantile': 0.3, 'l3_threshold': 0.5895266158683357, 'formula': 'min', 'n_samples': 162, 'n_trades': 63, 'decision_threshold': 0.5, 'auc': 0.3205648433695828, 'brier': 0.2919199451107009, 'total_pnl': -0.31867756003025494, 'avg_pnl': -0.005058373968734205, 'avg_pnl_ci_low': -0.008705974359512916, 'avg_pnl_ci_high': -0.000859380138926453, 'win_rate': 0.2857142857142857, 'profit_factor': 0.5275092652142943, 'trades_per_day': 1.470817120622568, 'total_return': -0.27948854093280795, 'max_drawdown': 0.32456038147874555, 'calmar_like': -0.8611295675046823}, {'l3_keep_fraction': 0.3, 'l3_quantile': 0.3, 'l3_threshold': 0.5895266158683357, 'formula': 'logit_avg', 'n_samples': 162, 'n_trades': 127, 'decision_threshold': 0.5, 'auc': 0.31910404155169614, 'brier': 0.29356774138638775, 'total_pnl': -0.7526557553056638, 'avg_pnl': -0.005926423270123337, 'avg_pnl_ci_low': -0.008662567629995961, 'avg_pnl_ci_high': -0.0032880488483055473, 'win_rate': 0.2755905511811024, 'profit_factor': 0.45663855224548494, 'trades_per_day': 0.48306192796862, 'total_return': -0.5369493890832423, 'max_drawdown': 0.5444287881106809, 'calmar_like': -0.9862619332559904}], 'best_by_auc': {'l3_keep_fraction': 0.7, 'l3_quantile': 0.7, 'l3_threshold': 0.3965864304813981, 'formula': 'logit_avg', 'n_samples': 376, 'n_trades': 112, 'decision_threshold': 0.5, 'auc': 0.3578706591807029, 'brier': 0.2629231817441764, 'total_pnl': -0.6741351776380924, 'avg_pnl': -0.006019064086054396, 'avg_pnl_ci_low': -0.008693220196265929, 'avg_pnl_ci_high': -0.0028406757813494057, 'win_rate': 0.2767857142857143, 'profit_factor': 0.4597350233596941, 'trades_per_day': 0.4260073695471294, 'total_return': -0.4985715221806848, 'max_drawdown': 0.5066708175512419, 'calmar_like': -0.9840146795691032}, 'best_by_pf': {'l3_keep_fraction': 0.4, 'l3_quantile': 0.4, 'l3_threshold': 0.5569671405724586, 'formula': 'min', 'n_samples': 215, 'n_trades': 69, 'decision_threshold': 0.5, 'auc': 0.3031924380238987, 'brier': 0.2936297462772733, 'total_pnl': -0.33766984128849564, 'avg_pnl': -0.004893765815775299, 'avg_pnl_ci_low': -0.008368247072976991, 'avg_pnl_ci_high': -0.0007194985472123996, 'win_rate': 0.2898550724637681, 'profit_factor': 0.5395896901078107, 'trades_per_day': 1.610894941634241, 'total_return': -0.29368930173283747, 'max_drawdown': 0.33787280883403215, 'calmar_like': -0.8692303554863231}, 'best': {'l3_keep_fraction': 0.4, 'l3_quantile': 0.4, 'l3_threshold': 0.5569671405724586, 'formula': 'min', 'n_samples': 215, 'n_trades': 69, 'decision_threshold': 0.5, 'auc': 0.3031924380238987, 'brier': 0.2936297462772733, 'total_pnl': -0.33766984128849564, 'avg_pnl': -0.004893765815775299, 'avg_pnl_ci_low': -0.008368247072976991, 'avg_pnl_ci_high': -0.0007194985472123996, 'win_rate': 0.2898550724637681, 'profit_factor': 0.5395896901078107, 'trades_per_day': 1.610894941634241, 'total_return': -0.29368930173283747, 'max_drawdown': 0.33787280883403215, 'calmar_like': -0.8692303554863231}}

## Layer5 Metrics
- Average Size: 0.024006
- Avg Trade PnL: -0.000145648
- Bet Utilization Efficiency: 0
- Edge Monotonicity: {'correlation': 0.0, 'bins': {'(0.0, 0.1]': nan, '(0.1, 0.2]': nan, '(0.2, 0.30000000000000004]': nan, '(0.30000000000000004, 0.4]': -0.418492728745719, '(0.4, 0.5]': nan, '(0.5, 0.6000000000000001]': nan, '(0.6000000000000001, 0.7000000000000001]': nan, '(0.7000000000000001, 0.8]': nan, '(0.8, 0.9]': nan, '(0.9, 1.0]': nan}}
- Expectancy: -0.000145648
- Exposure: 0.011194
- Gate Count: 6
- Gate Mode: quantile
- Gate Threshold: 0.362611
- Gross Loss: 0.00154716
- Gross Profit: 0.00067327
- Maximum Drawdown: 0.00146244
- Median Size: 0.0115527
- Net Sortino: -0.247496
- Parameters: {'gamma': 1.2, 'p_min': 0.5569671405724586, 'p_max': 0.9, 'gate_mode': 'quantile', 'gate_quantile': None, 'gate_top_k': None, 'gate_top_k_per_day': None, 'gate_search_q_low': None, 'gate_search_q_high': None, 'gate_search_min_range': None, 'gate_search_max_iter': None, 'gate_threshold': 0.36261116729374415}
- Payoff Ratio: 2.17583
- PnL Mean (Traded): -0.000145648
- PnL Q10 (Traded): -0.000645595
- PnL Q50 (Traded): -0.00012583
- PnL Q90 (Traded): 0.000334482
- PnL Std (Traded): 0.000523615
- Prob Mean: 0.189629
- Prob Q50: 0.187837
- Prob Q90: 0.279086
- Prob Q99: 0.362611
- Prob Std: 0.0723618
- Prob>=p_min Count: 0
- Profit Factor: 0.435166
- Return / Drawdown Ratio: -0.597898
- Runtime: {'n_rows': 536, 'sizing_ms': 0.32958301017060876, 'edge_monotonicity_ms': 4.610083007719368, 'total_ms': 4.9396660178899765}
- Sharpe-like (Traded): -0.681346
- Tail Loss Amplification: 0.00173351
- Total PnL: -0.000873888
- Total Return: -0.000874392
- Trade Count: 6
- Win Rate: 0.166667
- pmin_sweep_csv: outcomes/layer5_pmin_sweep_20251219_002248.csv
