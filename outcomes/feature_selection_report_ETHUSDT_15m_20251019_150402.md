# Feature Selection Report

**Generated:** 2025-10-19T15:04:02.743220

## 📌 Configuration

- Symbol: ETHUSDT
- Timeframe: 15m
- Direction: DirectionType.LONGS
- Strategy: sophisticated_multi_stage

## 📊 Summary

- Rows: 1,162,368
- Columns (input): 354
- Columns (selected): 4
- Reduction: 98.9%

## 🧱 Selected Features

- volume_momentum_5, pfe_12_returns_vwap, support_level_2_20_price_returns, lempel_ziv_complexity_20

## 🎯 Selection Metrics

- advanced_metrics: {'average_score': 0.8489717872417655, 'max_score': 0.8489717872417654, 'min_score': 0.8489717872417654, 'average_variance': 558469.94, 'average_correlation': 0.9999999999999999, 'average_information_content': 0.9948594646609732, 'average_uniqueness': 1.1102230246251565e-16, 'score_std': 1.1102230246251565e-16, 'total_features': 60}
- multi_objective_metrics: {'out_of_sample_sharpe': 47.197853446206445, 'drawdown': 0.0, 'turnover': 0.24815156, 'stability': 0.0, 'diversity': 0.0, 'mutual_information': 12.711746860047837, 'profit_centered': 2221.8862549087416}
- economic_metrics: {'target_mean': 2221.886474609375, 'target_std': 747.3084716796875, 'target_skew': 0.601192057132721, 'target_kurtosis': -0.7654091119766235}
- vectorbt_metrics: {'interaction_generation': [{'name': 'product_volume_momentum_5_pfe_12_returns_vwap', 'feature_series': 0                 0.0
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
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'pfe_12_returns_vwap'], 'interaction_type': 'product', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'product_volume_momentum_5_support_level_2_20_price_returns', 'feature_series': 0                 0.0
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
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'support_level_2_20_price_returns'], 'interaction_type': 'product', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'product_volume_momentum_5_lempel_ziv_complexity_20', 'feature_series': 0                 0.0
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
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'lempel_ziv_complexity_20'], 'interaction_type': 'product', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'product_pfe_12_returns_vwap_support_level_2_20_price_returns', 'feature_series': 0                 0.0
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
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'support_level_2_20_price_returns'], 'interaction_type': 'product', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'product_pfe_12_returns_vwap_lempel_ziv_complexity_20', 'feature_series': 0                 0.0
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
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'lempel_ziv_complexity_20'], 'interaction_type': 'product', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'product_support_level_2_20_price_returns_lempel_ziv_complexity_20', 'feature_series': 0                 0.0
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
Length: 1162368, dtype: float32, 'parent_features': ['support_level_2_20_price_returns', 'lempel_ziv_complexity_20'], 'interaction_type': 'product', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'ratio_volume_momentum_5_pfe_12_returns_vwap', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    1.0
1162364    1.0
1162365    1.0
1162366    1.0
1162367    1.0
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'pfe_12_returns_vwap'], 'interaction_type': 'ratio', 'utility_score': 0.012020728508308846, 'vectorbt_optimized': True}, {'name': 'ratio_volume_momentum_5_support_level_2_20_price_returns', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    1.0
1162364    1.0
1162365    1.0
1162366    1.0
1162367    1.0
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'support_level_2_20_price_returns'], 'interaction_type': 'ratio', 'utility_score': 0.012020728508308846, 'vectorbt_optimized': True}, {'name': 'ratio_volume_momentum_5_lempel_ziv_complexity_20', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    1.0
1162364    1.0
1162365    1.0
1162366    1.0
1162367    1.0
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'lempel_ziv_complexity_20'], 'interaction_type': 'ratio', 'utility_score': 0.012020728508308846, 'vectorbt_optimized': True}, {'name': 'ratio_pfe_12_returns_vwap_support_level_2_20_price_returns', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    1.0
1162364    1.0
1162365    1.0
1162366    1.0
1162367    1.0
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'support_level_2_20_price_returns'], 'interaction_type': 'ratio', 'utility_score': 0.012020728508308846, 'vectorbt_optimized': True}, {'name': 'ratio_pfe_12_returns_vwap_lempel_ziv_complexity_20', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    1.0
1162364    1.0
1162365    1.0
1162366    1.0
1162367    1.0
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'lempel_ziv_complexity_20'], 'interaction_type': 'ratio', 'utility_score': 0.012020728508308846, 'vectorbt_optimized': True}, {'name': 'ratio_support_level_2_20_price_returns_lempel_ziv_complexity_20', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    1.0
1162364    1.0
1162365    1.0
1162366    1.0
1162367    1.0
Length: 1162368, dtype: float32, 'parent_features': ['support_level_2_20_price_returns', 'lempel_ziv_complexity_20'], 'interaction_type': 'ratio', 'utility_score': 0.012020728508308846, 'vectorbt_optimized': True}, {'name': 'difference_volume_momentum_5_pfe_12_returns_vwap', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'pfe_12_returns_vwap'], 'interaction_type': 'difference', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'difference_volume_momentum_5_support_level_2_20_price_returns', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'support_level_2_20_price_returns'], 'interaction_type': 'difference', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'difference_volume_momentum_5_lempel_ziv_complexity_20', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'lempel_ziv_complexity_20'], 'interaction_type': 'difference', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'difference_pfe_12_returns_vwap_support_level_2_20_price_returns', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'support_level_2_20_price_returns'], 'interaction_type': 'difference', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'difference_pfe_12_returns_vwap_lempel_ziv_complexity_20', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'lempel_ziv_complexity_20'], 'interaction_type': 'difference', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'difference_support_level_2_20_price_returns_lempel_ziv_complexity_20', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['support_level_2_20_price_returns', 'lempel_ziv_complexity_20'], 'interaction_type': 'difference', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'sum_volume_momentum_5_pfe_12_returns_vwap', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    7187.856934
1162364    7187.856934
1162365    7187.913086
1162366    7187.942871
1162367    7187.622070
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'pfe_12_returns_vwap'], 'interaction_type': 'sum', 'utility_score': 0.9999999999999999, 'vectorbt_optimized': True}, {'name': 'sum_volume_momentum_5_support_level_2_20_price_returns', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    7187.856934
1162364    7187.856934
1162365    7187.913086
1162366    7187.942871
1162367    7187.622070
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'support_level_2_20_price_returns'], 'interaction_type': 'sum', 'utility_score': 0.9999999999999999, 'vectorbt_optimized': True}, {'name': 'sum_volume_momentum_5_lempel_ziv_complexity_20', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    7187.856934
1162364    7187.856934
1162365    7187.913086
1162366    7187.942871
1162367    7187.622070
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'lempel_ziv_complexity_20'], 'interaction_type': 'sum', 'utility_score': 0.9999999999999999, 'vectorbt_optimized': True}, {'name': 'sum_pfe_12_returns_vwap_support_level_2_20_price_returns', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    7187.856934
1162364    7187.856934
1162365    7187.913086
1162366    7187.942871
1162367    7187.622070
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'support_level_2_20_price_returns'], 'interaction_type': 'sum', 'utility_score': 0.9999999999999999, 'vectorbt_optimized': True}, {'name': 'sum_pfe_12_returns_vwap_lempel_ziv_complexity_20', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    7187.856934
1162364    7187.856934
1162365    7187.913086
1162366    7187.942871
1162367    7187.622070
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'lempel_ziv_complexity_20'], 'interaction_type': 'sum', 'utility_score': 0.9999999999999999, 'vectorbt_optimized': True}, {'name': 'sum_support_level_2_20_price_returns_lempel_ziv_complexity_20', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    7187.856934
1162364    7187.856934
1162365    7187.913086
1162366    7187.942871
1162367    7187.622070
Length: 1162368, dtype: float32, 'parent_features': ['support_level_2_20_price_returns', 'lempel_ziv_complexity_20'], 'interaction_type': 'sum', 'utility_score': 0.9999999999999999, 'vectorbt_optimized': True}, {'name': 'log_product_volume_momentum_5_pfe_12_returns_vwap', 'feature_series': 0          339.321503
1          339.321503
2          339.321503
3          339.321503
4          339.321503
              ...    
1162363     67.026993
1162364     67.026993
1162365     67.027115
1162366     67.027176
1162367     67.026459
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'pfe_12_returns_vwap'], 'interaction_type': 'log_product', 'utility_score': 0.9642425160256505, 'vectorbt_optimized': True}, {'name': 'log_product_volume_momentum_5_support_level_2_20_price_returns', 'feature_series': 0          339.321503
1          339.321503
2          339.321503
3          339.321503
4          339.321503
              ...    
1162363     67.026993
1162364     67.026993
1162365     67.027115
1162366     67.027176
1162367     67.026459
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'support_level_2_20_price_returns'], 'interaction_type': 'log_product', 'utility_score': 0.9642425160256505, 'vectorbt_optimized': True}, {'name': 'log_product_volume_momentum_5_lempel_ziv_complexity_20', 'feature_series': 0          339.321503
1          339.321503
2          339.321503
3          339.321503
4          339.321503
              ...    
1162363     67.026993
1162364     67.026993
1162365     67.027115
1162366     67.027176
1162367     67.026459
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'lempel_ziv_complexity_20'], 'interaction_type': 'log_product', 'utility_score': 0.9642425160256505, 'vectorbt_optimized': True}, {'name': 'log_product_pfe_12_returns_vwap_support_level_2_20_price_returns', 'feature_series': 0          339.321503
1          339.321503
2          339.321503
3          339.321503
4          339.321503
              ...    
1162363     67.026993
1162364     67.026993
1162365     67.027115
1162366     67.027176
1162367     67.026459
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'support_level_2_20_price_returns'], 'interaction_type': 'log_product', 'utility_score': 0.9642425160256505, 'vectorbt_optimized': True}, {'name': 'log_product_pfe_12_returns_vwap_lempel_ziv_complexity_20', 'feature_series': 0          339.321503
1          339.321503
2          339.321503
3          339.321503
4          339.321503
              ...    
1162363     67.026993
1162364     67.026993
1162365     67.027115
1162366     67.027176
1162367     67.026459
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'lempel_ziv_complexity_20'], 'interaction_type': 'log_product', 'utility_score': 0.9642425160256505, 'vectorbt_optimized': True}, {'name': 'log_product_support_level_2_20_price_returns_lempel_ziv_complexity_20', 'feature_series': 0          339.321503
1          339.321503
2          339.321503
3          339.321503
4          339.321503
              ...    
1162363     67.026993
1162364     67.026993
1162365     67.027115
1162366     67.027176
1162367     67.026459
Length: 1162368, dtype: float32, 'parent_features': ['support_level_2_20_price_returns', 'lempel_ziv_complexity_20'], 'interaction_type': 'log_product', 'utility_score': 0.9642425160256505, 'vectorbt_optimized': True}, {'name': 'log_ratio_volume_momentum_5_pfe_12_returns_vwap', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'pfe_12_returns_vwap'], 'interaction_type': 'log_ratio', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'log_ratio_volume_momentum_5_support_level_2_20_price_returns', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'support_level_2_20_price_returns'], 'interaction_type': 'log_ratio', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'log_ratio_volume_momentum_5_lempel_ziv_complexity_20', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5', 'lempel_ziv_complexity_20'], 'interaction_type': 'log_ratio', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'log_ratio_pfe_12_returns_vwap_support_level_2_20_price_returns', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'support_level_2_20_price_returns'], 'interaction_type': 'log_ratio', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'log_ratio_pfe_12_returns_vwap_lempel_ziv_complexity_20', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap', 'lempel_ziv_complexity_20'], 'interaction_type': 'log_ratio', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'log_ratio_support_level_2_20_price_returns_lempel_ziv_complexity_20', 'feature_series': 0          0.0
1          0.0
2          0.0
3          0.0
4          0.0
          ... 
1162363    0.0
1162364    0.0
1162365    0.0
1162366    0.0
1162367    0.0
Length: 1162368, dtype: float32, 'parent_features': ['support_level_2_20_price_returns', 'lempel_ziv_complexity_20'], 'interaction_type': 'log_ratio', 'utility_score': 0.0, 'vectorbt_optimized': True}, {'name': 'polynomial_volume_momentum_5', 'feature_series': 0                 0.0
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
Name: volume_momentum_5, Length: 1162368, dtype: float32, 'parent_features': ['volume_momentum_5'], 'interaction_type': 'polynomial', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'polynomial_pfe_12_returns_vwap', 'feature_series': 0                 0.0
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float32, 'parent_features': ['pfe_12_returns_vwap'], 'interaction_type': 'polynomial', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'polynomial_support_level_2_20_price_returns', 'feature_series': 0                 0.0
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float32, 'parent_features': ['support_level_2_20_price_returns'], 'interaction_type': 'polynomial', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'polynomial_lempel_ziv_complexity_20', 'feature_series': 0                 0.0
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float32, 'parent_features': ['lempel_ziv_complexity_20'], 'interaction_type': 'polynomial', 'utility_score': 0.98996932033324, 'vectorbt_optimized': True}, {'name': 'conditional_volume_momentum_5_pfe_12_returns_vwap', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    3593.928467
1162364    3593.928467
1162365    3593.956543
1162366    3593.971436
1162367    3593.811035
Length: 1162368, dtype: float64, 'parent_features': ['volume_momentum_5', 'pfe_12_returns_vwap'], 'interaction_type': 'conditional', 'utility_score': 0.9329225030929743, 'vectorbt_optimized': True}, {'name': 'conditional_volume_momentum_5_support_level_2_20_price_returns', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    3593.928467
1162364    3593.928467
1162365    3593.956543
1162366    3593.971436
1162367    3593.811035
Length: 1162368, dtype: float64, 'parent_features': ['volume_momentum_5', 'support_level_2_20_price_returns'], 'interaction_type': 'conditional', 'utility_score': 0.9329225030929743, 'vectorbt_optimized': True}, {'name': 'conditional_volume_momentum_5_lempel_ziv_complexity_20', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    3593.928467
1162364    3593.928467
1162365    3593.956543
1162366    3593.971436
1162367    3593.811035
Length: 1162368, dtype: float64, 'parent_features': ['volume_momentum_5', 'lempel_ziv_complexity_20'], 'interaction_type': 'conditional', 'utility_score': 0.9329225030929743, 'vectorbt_optimized': True}, {'name': 'conditional_pfe_12_returns_vwap_support_level_2_20_price_returns', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    3593.928467
1162364    3593.928467
1162365    3593.956543
1162366    3593.971436
1162367    3593.811035
Length: 1162368, dtype: float64, 'parent_features': ['pfe_12_returns_vwap', 'support_level_2_20_price_returns'], 'interaction_type': 'conditional', 'utility_score': 0.9329225030929743, 'vectorbt_optimized': True}, {'name': 'conditional_pfe_12_returns_vwap_lempel_ziv_complexity_20', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    3593.928467
1162364    3593.928467
1162365    3593.956543
1162366    3593.971436
1162367    3593.811035
Length: 1162368, dtype: float64, 'parent_features': ['pfe_12_returns_vwap', 'lempel_ziv_complexity_20'], 'interaction_type': 'conditional', 'utility_score': 0.9329225030929743, 'vectorbt_optimized': True}, {'name': 'conditional_support_level_2_20_price_returns_lempel_ziv_complexity_20', 'feature_series': 0             0.000000
1             0.000000
2             0.000000
3             0.000000
4             0.000000
              ...     
1162363    3593.928467
1162364    3593.928467
1162365    3593.956543
1162366    3593.971436
1162367    3593.811035
Length: 1162368, dtype: float64, 'parent_features': ['support_level_2_20_price_returns', 'lempel_ziv_complexity_20'], 'interaction_type': 'conditional', 'utility_score': 0.9329225030929743, 'vectorbt_optimized': True}, {'name': 'rolling_mean_volume_momentum_5', 'feature_series': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'parent_features': ['volume_momentum_5'], 'interaction_type': 'rolling_mean', 'utility_score': 0.9999769269921569, 'vectorbt_optimized': True}, {'name': 'rolling_mean_pfe_12_returns_vwap', 'feature_series': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'parent_features': ['pfe_12_returns_vwap'], 'interaction_type': 'rolling_mean', 'utility_score': 0.9999769269921569, 'vectorbt_optimized': True}, {'name': 'rolling_mean_support_level_2_20_price_returns', 'feature_series': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'parent_features': ['support_level_2_20_price_returns'], 'interaction_type': 'rolling_mean', 'utility_score': 0.9999769269921569, 'vectorbt_optimized': True}, {'name': 'rolling_mean_lempel_ziv_complexity_20', 'feature_series': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'parent_features': ['lempel_ziv_complexity_20'], 'interaction_type': 'rolling_mean', 'utility_score': 0.9999769269921569, 'vectorbt_optimized': True}, {'name': 'rolling_std_volume_momentum_5', 'feature_series': 0               NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'parent_features': ['volume_momentum_5'], 'interaction_type': 'rolling_std', 'utility_score': 0.1920398324401831, 'vectorbt_optimized': True}, {'name': 'rolling_std_pfe_12_returns_vwap', 'feature_series': 0               NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'parent_features': ['pfe_12_returns_vwap'], 'interaction_type': 'rolling_std', 'utility_score': 0.1920398324401831, 'vectorbt_optimized': True}, {'name': 'rolling_std_support_level_2_20_price_returns', 'feature_series': 0               NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'parent_features': ['support_level_2_20_price_returns'], 'interaction_type': 'rolling_std', 'utility_score': 0.1920398324401831, 'vectorbt_optimized': True}, {'name': 'rolling_std_lempel_ziv_complexity_20', 'feature_series': 0               NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'parent_features': ['lempel_ziv_complexity_20'], 'interaction_type': 'rolling_std', 'utility_score': 0.1920398324401831, 'vectorbt_optimized': True}, {'name': 'correlation_volume_momentum_5_pfe_12_returns_vwap', 'feature_series': 0               NaN
1               NaN
2               NaN
3               NaN
4               NaN
             ...   
1162363    1.000002
1162364    1.000003
1162365    1.000004
1162366    1.000004
1162367    1.000004
Length: 1162368, dtype: float64, 'parent_features': ['volume_momentum_5', 'pfe_12_returns_vwap'], 'interaction_type': 'correlation', 'utility_score': 0.07887431520024052, 'vectorbt_optimized': True}, {'name': 'correlation_volume_momentum_5_support_level_2_20_price_returns', 'feature_series': 0               NaN
1               NaN
2               NaN
3               NaN
4               NaN
             ...   
1162363    1.000002
1162364    1.000003
1162365    1.000004
1162366    1.000004
1162367    1.000004
Length: 1162368, dtype: float64, 'parent_features': ['volume_momentum_5', 'support_level_2_20_price_returns'], 'interaction_type': 'correlation', 'utility_score': 0.07887431520024052, 'vectorbt_optimized': True}, {'name': 'correlation_volume_momentum_5_lempel_ziv_complexity_20', 'feature_series': 0               NaN
1               NaN
2               NaN
3               NaN
4               NaN
             ...   
1162363    1.000002
1162364    1.000003
1162365    1.000004
1162366    1.000004
1162367    1.000004
Length: 1162368, dtype: float64, 'parent_features': ['volume_momentum_5', 'lempel_ziv_complexity_20'], 'interaction_type': 'correlation', 'utility_score': 0.07887431520024052, 'vectorbt_optimized': True}, {'name': 'correlation_pfe_12_returns_vwap_support_level_2_20_price_returns', 'feature_series': 0               NaN
1               NaN
2               NaN
3               NaN
4               NaN
             ...   
1162363    1.000002
1162364    1.000003
1162365    1.000004
1162366    1.000004
1162367    1.000004
Length: 1162368, dtype: float64, 'parent_features': ['pfe_12_returns_vwap', 'support_level_2_20_price_returns'], 'interaction_type': 'correlation', 'utility_score': 0.07887431520024052, 'vectorbt_optimized': True}, {'name': 'correlation_pfe_12_returns_vwap_lempel_ziv_complexity_20', 'feature_series': 0               NaN
1               NaN
2               NaN
3               NaN
4               NaN
             ...   
1162363    1.000002
1162364    1.000003
1162365    1.000004
1162366    1.000004
1162367    1.000004
Length: 1162368, dtype: float64, 'parent_features': ['pfe_12_returns_vwap', 'lempel_ziv_complexity_20'], 'interaction_type': 'correlation', 'utility_score': 0.07887431520024052, 'vectorbt_optimized': True}, {'name': 'correlation_support_level_2_20_price_returns_lempel_ziv_complexity_20', 'feature_series': 0               NaN
1               NaN
2               NaN
3               NaN
4               NaN
             ...   
1162363    1.000002
1162364    1.000003
1162365    1.000004
1162366    1.000004
1162367    1.000004
Length: 1162368, dtype: float64, 'parent_features': ['support_level_2_20_price_returns', 'lempel_ziv_complexity_20'], 'interaction_type': 'correlation', 'utility_score': 0.07887431520024052, 'vectorbt_optimized': True}], 'lookback_analysis': {'volume_momentum_5': {5: {'lookback': 5, 'rolling_mean': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'stability': 5859.799531782981, 'predictability': 9659.342958432804, 'information_content': 0.00016972110987314399, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0, 'stationarity': 180650.94153583987, 'data_length': 1162368, 'lookback_coverage': 232473.6}}, 10: {'lookback': 10, 'rolling_mean': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'stability': 4736.294489563332, 'predictability': 5862.156378083525, 'information_content': 0.0003191463706868475, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0, 'stationarity': 147710.94306083003, 'data_length': 1162368, 'lookback_coverage': 116236.8}}, 20: {'lookback': 20, 'rolling_mean': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'stability': 3013.2958434032653, 'predictability': 3363.100734921331, 'information_content': 0.0005985469617846599, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0026231364246022495, 'stationarity': 121922.0272088319, 'data_length': 1162368, 'lookback_coverage': 58118.4}}, 50: {'lookback': 50, 'rolling_mean': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: volume_momentum_5, Length: 1162368, dtype: float64, 'stability': 431.0451334287431, 'predictability': 434.15048410201825, 'information_content': 0.0012547792024956992, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.018840782952012313, 'stationarity': 97959.85163381853, 'data_length': 1162368, 'lookback_coverage': 23247.36}}}, 'pfe_12_returns_vwap': {5: {'lookback': 5, 'rolling_mean': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'stability': 5859.799531782981, 'predictability': 9659.342958432804, 'information_content': 0.00016972110987314399, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0, 'stationarity': 180650.94153583987, 'data_length': 1162368, 'lookback_coverage': 232473.6}}, 10: {'lookback': 10, 'rolling_mean': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'stability': 4736.294489563332, 'predictability': 5862.156378083525, 'information_content': 0.0003191463706868475, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0, 'stationarity': 147710.94306083003, 'data_length': 1162368, 'lookback_coverage': 116236.8}}, 20: {'lookback': 20, 'rolling_mean': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'stability': 3013.2958434032653, 'predictability': 3363.100734921331, 'information_content': 0.0005985469617846599, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0026231364246022495, 'stationarity': 121922.0272088319, 'data_length': 1162368, 'lookback_coverage': 58118.4}}, 50: {'lookback': 50, 'rolling_mean': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: pfe_12_returns_vwap, Length: 1162368, dtype: float64, 'stability': 431.0451334287431, 'predictability': 434.15048410201825, 'information_content': 0.0012547792024956992, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.018840782952012313, 'stationarity': 97959.85163381853, 'data_length': 1162368, 'lookback_coverage': 23247.36}}}, 'support_level_2_20_price_returns': {5: {'lookback': 5, 'rolling_mean': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'stability': 5859.799531782981, 'predictability': 9659.342958432804, 'information_content': 0.00016972110987314399, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0, 'stationarity': 180650.94153583987, 'data_length': 1162368, 'lookback_coverage': 232473.6}}, 10: {'lookback': 10, 'rolling_mean': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'stability': 4736.294489563332, 'predictability': 5862.156378083525, 'information_content': 0.0003191463706868475, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0, 'stationarity': 147710.94306083003, 'data_length': 1162368, 'lookback_coverage': 116236.8}}, 20: {'lookback': 20, 'rolling_mean': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'stability': 3013.2958434032653, 'predictability': 3363.100734921331, 'information_content': 0.0005985469617846599, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0026231364246022495, 'stationarity': 121922.0272088319, 'data_length': 1162368, 'lookback_coverage': 58118.4}}, 50: {'lookback': 50, 'rolling_mean': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: support_level_2_20_price_returns, Length: 1162368, dtype: float64, 'stability': 431.0451334287431, 'predictability': 434.15048410201825, 'information_content': 0.0012547792024956992, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.018840782952012313, 'stationarity': 97959.85163381853, 'data_length': 1162368, 'lookback_coverage': 23247.36}}}, 'lempel_ziv_complexity_20': {5: {'lookback': 5, 'rolling_mean': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'stability': 5859.799531782981, 'predictability': 9659.342958432804, 'information_content': 0.00016972110987314399, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0, 'stationarity': 180650.94153583987, 'data_length': 1162368, 'lookback_coverage': 232473.6}}, 10: {'lookback': 10, 'rolling_mean': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'stability': 4736.294489563332, 'predictability': 5862.156378083525, 'information_content': 0.0003191463706868475, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0, 'stationarity': 147710.94306083003, 'data_length': 1162368, 'lookback_coverage': 116236.8}}, 20: {'lookback': 20, 'rolling_mean': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'stability': 3013.2958434032653, 'predictability': 3363.100734921331, 'information_content': 0.0005985469617846599, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.0026231364246022495, 'stationarity': 121922.0272088319, 'data_length': 1162368, 'lookback_coverage': 58118.4}}, 50: {'lookback': 50, 'rolling_mean': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_std': 0               NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_min': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'rolling_max': 0                  NaN
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
Name: lempel_ziv_complexity_20, Length: 1162368, dtype: float64, 'stability': 431.0451334287431, 'predictability': 434.15048410201825, 'information_content': 0.0012547792024956992, 'data_quality': {'missing_ratio': 0.0, 'outlier_ratio': 0.018840782952012313, 'stationarity': 97959.85163381853, 'data_length': 1162368, 'lookback_coverage': 23247.36}}}}}

## 🧪 Quality Metrics

- average_score: 0.8489717872417655
- max_score: 0.8489717872417654
- min_score: 0.8489717872417654
- average_variance: 558469.9375
- average_correlation: 0.9999999999999999
- average_information_content: 0.9948594646609732
- average_uniqueness: 1.1102230246251565e-16
- score_std: 1.1102230246251565e-16
- total_features: 60

## 🌈 Diversity Metrics

- category_diversity: 9
- aspect_diversity: 4
- average_uniqueness: 1.1102230246251565e-16
- min_uniqueness: 1.1102230246251565e-16
- max_uniqueness: 1.1102230246251565e-16

## 🔁 Stability Metrics

- average_stability: 3013.2958434032653
- min_stability: 3013.2958434032653
- max_stability: 3013.2958434032653
- average_predictability: 0.9999989430957075

## 📐 Multi-Objective

- selected_features: ['volume_momentum_5', 'pfe_12_returns_vwap', 'support_level_2_20_price_returns', 'lempel_ziv_complexity_20']
- pareto_front: []
- optimization_metadata: {'n_features_selected': 4, 'n_features_total': 60, 'selection_ratio': 0.06666666666666667, 'hardware_optimization_used': True, 'vectorbt_optimization_used': True, 'bayesian_tpe_used': True, 'mean_correlation': 0.9999999999999999, 'max_correlation': 0.9999999999999999, 'min_correlation': 0.9999999999999999}
- is_valid: True
- feature_scores: {'volume_momentum_5': 0.9999999999999999, 'pfe_12_returns_vwap': 0.9999999999999999, 'support_level_2_20_price_returns': 0.9999999999999999, 'lempel_ziv_complexity_20': 0.9999999999999999, 'candlestick_three_white_soldiers_pattern': 0.9999999999999999, 'momentum_endpoints_sma_20': 0.9999999999999999, 'log_returns_1_price_returns': 0.9999999999999999, 'vectorbt_parkinson_volatility_50': 0.9999999999999999, 'stochastic_30_3_price_returns': 0.9999999999999999, 'vectorbt_acceleration_volatility_10_10_price_returns': 0.9999999999999999, 'vectorbt_acceleration_trend_strength_5_20_price_returns': 0.9999999999999999, 'candlestick_engulfing_pattern': 0.9999999999999999, 'resistance_level_4_5_price_returns': 0.9999999999999999, 'sma_100_returns_vwap': 0.9999999999999999, 'resistance_level_3_5_price_returns': 0.9999999999999999, 'vectorbt_acceleration_correlation_20_price_returns': 0.9999999999999999, 'volume_sma_10': 0.9999999999999999, 'vectorbt_enhanced_obv_10': 0.9999999999999999, 'vectorbt_volatility_comprehensive_20': 0.9999999999999999, 'shannon_entropy_20_10': 0.9999999999999999, 'macd_delta_12_26_9': 0.9999999999999999, 'vectorbt_acceleration_regime_5_20_price_returns': 0.9999999999999999, 'vectorbt_acceleration_consistency_5_20_price_returns': 0.9999999999999999, 'cmf_20': 0.9999999999999999, 'vectorbt_bbands_14_1.5': 0.9999999999999999, 'analyst_volume_trend': 0.9999999999999999, 'price_entropy_10_price_returns': 0.9999999999999999, 'tema_21_price_returns': 0.9999999999999999, 'resistance_level_5_20_price_returns': 0.9999999999999999, 'support_level_1_10_price_returns': 0.9999999999999999, 'ctf_15m_trend_price_returns': 0.9999999999999999, 'resistance_level_4_20_price_returns': 0.9999999999999999, 'price_volume_oscillator_5_15': 0.9999999999999999, 'ema_12_returns_vwap': 0.9999999999999999, 'kama_30_2_30_returns_vwap': 0.9999999999999999, 'fibonacci_0.236_10_price_returns': 0.9999999999999999, 'vectorbt_bbands_20_1.5': 0.9999999999999999, 'mama_21_0.05_price_returns': 0.9999999999999999, 'fibonacci_0.236_5_price_returns': 0.9999999999999999, 'resistance_level_5_10_price_returns': 0.9999999999999999, 'support_level_1_5_price_returns': 0.9999999999999999, 'williams_r_30_price_returns': 0.9999999999999999, 'log_returns_10_price_returns': 0.9999999999999999, 'sample_entropy_20_2_0.2': 0.9999999999999999, 'ema_50_returns_vwap': 0.9999999999999999, 'volume_momentum_10': 0.9999999999999999, 'volume_roc_20': 0.9999999999999999, 'sma_20_returns_vwap': 0.9999999999999999, 'ultimate_oscillator_7_14_28_returns_vwap': 0.9999999999999999, 'fibonacci_0.236_20_price_returns': 0.9999999999999999, 'vectorbt_acceleration_momentum_10_10_price_returns': 0.9999999999999999, 'vectorbt_acceleration_10_price_returns': 0.9999999999999999, 'vectorbt_acceleration_consistency_10_10_price_returns': 0.9999999999999999, 'order_flow_imbalance_20': 0.9999999999999999, 'vectorbt_momentum_acceleration_10_10_price_returns': 0.9999999999999999, 'vwma_20_price_returns': 0.9999999999999999, 'vectorbt_garman_klass_volatility_14': 0.9999999999999999, 'pivot_point_10_price_returns': 0.9999999999999999, 'vectorbt_atr_20': 0.9999999999999999, 'vectorbt_multi_timeframe_acceleration_5_20_price_returns': 0.9999999999999999}
- success: True
- error_message: None

## 💰 Economic Validation

- economic_scores: {'volume_momentum_5': 0.9999999999999999, 'pfe_12_returns_vwap': 0.9999999999999999, 'support_level_2_20_price_returns': 0.9999999999999999, 'lempel_ziv_complexity_20': 0.9999999999999999}
- validation_metrics: {'target_mean': 2221.886474609375, 'target_std': 747.3084716796875, 'target_skew': 0.601192057132721, 'target_kurtosis': -0.7654091119766235}
- performance_stats: {'total_evaluations': 0, 'successful_evaluations': 0, 'failed_evaluations': 0, 'total_execution_time': 0.09732198715209961, 'backtest_operations': 0, 'vectorbt_operations': 0}
- success: True
- error_message: None

## ⚡ VectorBT Optimizations

- performance_stats: {'total_operations': 5, 'vectorbt_operations': 6, 'fallback_operations': 0, 'gpu_operations': 0, 'total_execution_time': 15.480539083480835, 'memory_usage': 0.0}
- success: True
- error_message: None
