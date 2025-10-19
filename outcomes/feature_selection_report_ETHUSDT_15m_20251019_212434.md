# Feature Selection Report

**Generated:** 2025-10-19T21:24:34.230621

## 📌 Configuration

- Symbol: ETHUSDT
- Timeframe: 15m
- Direction: DirectionType.LONGS
- Strategy: sophisticated_multi_stage

## 📊 Summary

- Rows: 1,162,368
- Columns (input): 354
- Columns (selected): 1
- Reduction: 99.7%

## 🧱 Selected Features

- vectorbt_rogers_satchell_volatility_50

## 🎯 Selection Metrics

- advanced_metrics: {'average_score': 0.8489717872417654, 'max_score': 0.8489717872417654, 'min_score': 0.8489717872417654, 'average_variance': 558469.94, 'average_correlation': 0.9999999999999999, 'average_information_content': 0.9948594646609732, 'average_uniqueness': 1.1102230246251565e-16, 'score_std': 0.0, 'total_features': 1}
- multi_objective_metrics: {}
- economic_metrics: {}
- vectorbt_metrics: {'interaction_generation': [{'name': 'polynomial_vectorbt_rogers_satchell_volatility_50', 'feature_series': 0                 0.0
1                 0.0
2                 0.0
3                 0.0
4                 0.0
              ...    
1162363    12916322.0
1162364    12916322.0
1162365    12916524.0
1162366    12916631.0
1162367    12915478.0
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float32, 'parent_features': ['vectorbt_rogers_satchell_volatility_50'], 'interaction_type': 'polynomial', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'rolling_mean_vectorbt_rogers_satchell_volatility_50', 'feature_series': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4                  NaN
              ...     
1162363    3594.091736
1162364    3594.043481
1162365    3594.009106
1162366    3593.987854
1162367    3593.960059
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'parent_features': ['vectorbt_rogers_satchell_volatility_50'], 'interaction_type': 'rolling_mean', 'utility_score': 0.9999769269921569, 'vectorbt_optimized': True}, {'name': 'rolling_std_vectorbt_rogers_satchell_volatility_50', 'feature_series': 0               NaN
1               NaN
2               NaN
3               NaN
4               NaN
             ...   
1162363    0.366531
1162364    0.315372
1162365    0.282189
1162366    0.267080
1162367    0.254166
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'parent_features': ['vectorbt_rogers_satchell_volatility_50'], 'interaction_type': 'rolling_std', 'utility_score': 0.1920398324401831, 'vectorbt_optimized': True}], 'lookback_analysis': {'vectorbt_rogers_satchell_volatility_50': {5: {'lookback': 5, 'rolling_mean': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4             0.000000
              ...     
1162363    3593.841406
1162364    3593.889111
1162365    3593.922607
1162366    3593.948682
1162367    3593.919189
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
1               NaN
2               NaN
3               NaN
4          0.000000
             ...   
1162363    0.108364
1162364    0.071117
1162365    0.047848
1162366    0.019321
1162367    0.063235
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4             0.000000
              ...     
1162363    3593.689941
1162364    3593.789062
1162365    3593.841064
1162366    3593.928467
1162367    3593.811035
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4             0.000000
              ...     
1162363    3593.958496
1162364    3593.958496
1162365    3593.958496
1162366    3593.971436
1162367    3593.971436
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'stability': 5859.799531782981, 'predictability': 9659.342958432804, 'information_content': 0.00016972110987314399, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0, 'stationarity': 180650.94153583987, 'data_length': 1162368, 'lookback_coverage': 232473.6}}, 10: {'lookback': 10, 'rolling_mean': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4                  NaN
              ...     
1162363    3593.793091
1162364    3593.789648
1162365    3593.795508
1162366    3593.812598
1162367    3593.838062
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
1               NaN
2               NaN
3               NaN
4               NaN
             ...   
1162363    0.162245
1162364    0.158565
1162365    0.164001
1162366    0.173228
1162367    0.148306
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4                  NaN
              ...     
1162363    3593.506104
1162364    3593.506104
1162365    3593.506104
1162366    3593.506104
1162367    3593.506104
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4                  NaN
              ...     
1162363    3593.962891
1162364    3593.958496
1162365    3593.958496
1162366    3593.971436
1162367    3593.971436
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'stability': 4736.294489563332, 'predictability': 5862.156378083525, 'information_content': 0.0003191463706868475, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0, 'stationarity': 147710.94306083003, 'data_length': 1162368, 'lookback_coverage': 116236.8}}, 20: {'lookback': 20, 'rolling_mean': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4                  NaN
              ...     
1162363    3594.091736
1162364    3594.043481
1162365    3594.009106
1162366    3593.987854
1162367    3593.960059
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
1               NaN
2               NaN
3               NaN
4               NaN
             ...   
1162363    0.366531
1162364    0.315372
1162365    0.282189
1162366    0.267080
1162367    0.254166
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4                  NaN
              ...     
1162363    3593.506104
1162364    3593.506104
1162365    3593.506104
1162366    3593.506104
1162367    3593.506104
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4                  NaN
              ...     
1162363    3594.893555
1162364    3594.644043
1162365    3594.460938
1162366    3594.460938
1162367    3594.460938
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'stability': 3013.2958434032653, 'predictability': 3363.100734921331, 'information_content': 0.0005985469617846599, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0026231364246022495, 'stationarity': 121922.0272088319, 'data_length': 1162368, 'lookback_coverage': 58118.4}}, 50: {'lookback': 50, 'rolling_mean': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4                  NaN
              ...     
1162363    3594.430366
1162364    3594.376167
1162365    3594.333506
1162366    3594.298086
1162367    3594.268545
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
1               NaN
2               NaN
3               NaN
4               NaN
             ...   
1162363    0.626145
1162364    0.542863
1162365    0.486334
1162366    0.444302
1162367    0.425856
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4                  NaN
              ...     
1162363    3593.506104
1162364    3593.506104
1162365    3593.506104
1162366    3593.506104
1162367    3593.506104
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
1                  NaN
2                  NaN
3                  NaN
4                  NaN
              ...     
1162363    3596.638428
1162364    3596.089600
1162365    3595.742432
1162366    3595.288086
1162367    3594.987061
Name: vectorbt_rogers_satchell_volatility_50, Length: 1162368, dtype: float64, 'stability': 431.0451334287431, 'predictability': 434.15048410201825, 'information_content': 0.0012547792024956992, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.018840782952012313, 'stationarity': 97959.85163381853, 'data_length': 1162368, 'lookback_coverage': 23247.36}}}}}

## 🧪 Quality Metrics

- average_score: 0.8489717872417654
- max_score: 0.8489717872417654
- min_score: 0.8489717872417654
- average_variance: 558469.9375
- average_correlation: 0.9999999999999999
- average_information_content: 0.9948594646609732
- average_uniqueness: 1.1102230246251565e-16
- score_std: 0.0
- total_features: 1

## 🌈 Diversity Metrics

- category_diversity: 1
- aspect_diversity: 1
- average_uniqueness: 1.1102230246251565e-16
- min_uniqueness: 1.1102230246251565e-16
- max_uniqueness: 1.1102230246251565e-16

## 🔁 Stability Metrics

- average_stability: 3013.2958434032653
- min_stability: 3013.2958434032653
- max_stability: 3013.2958434032653
- average_predictability: 0.9999989430957075

## 📐 Multi-Objective

- selected_features: ['vectorbt_rogers_satchell_volatility_50']
- objective_values: {}
- pareto_front: []
- optimization_metadata: {}
- is_valid: True
- feature_scores: {}
- success: False
- error_message: 'M1CPUOptimizer' object has no attribute 'optimize_dataframe_cpu'

## 💰 Economic Validation

- economic_scores: {}
- validation_metrics: {}
- performance_stats: {'total_evaluations': 0, 'successful_evaluations': 0, 'failed_evaluations': 0, 'total_execution_time': 0.00017714500427246094, 'backtest_operations': 0, 'vectorbt_operations': 0}
- success: False
- error_message: No price data available for economic validation

## ⚡ VectorBT Optimizations

- performance_stats: {'total_operations': 2, 'vectorbt_operations': 3, 'fallback_operations': 0, 'gpu_operations': 0, 'total_execution_time': 3.2588508129119873, 'memory_usage': 0.0}
- success: True
- error_message: None
