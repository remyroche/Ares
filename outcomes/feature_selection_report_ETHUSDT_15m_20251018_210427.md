# Feature Selection Report

**Generated:** 2025-10-18T21:04:27.496975

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

- candlestick_harami_cross_pattern

## 🎯 Selection Metrics

- advanced_metrics: {}
- multi_objective_metrics: {}
- economic_metrics: {}
- vectorbt_metrics: {'interaction_generation': [], 'lookback_analysis': {'candlestick_harami_cross_pattern': {5: {'lookback': 5, 'error': 'VectorBT not available', 'fallback': True}, 10: {'lookback': 10, 'error': 'VectorBT not available', 'fallback': True}, 20: {'lookback': 20, 'error': 'VectorBT not available', 'fallback': True}, 50: {'lookback': 50, 'error': 'VectorBT not available', 'fallback': True}}}}

## 📐 Multi-Objective

- selected_features: ['candlestick_harami_cross_pattern']
- objective_values: {}
- pareto_front: []
- optimization_metadata: {}
- is_valid: True
- feature_scores: {}
- success: False
- error_message: 'M1MemoryOptimizer' object has no attribute 'optimize_series_memory'

## 💰 Economic Validation

- economic_scores: {}
- validation_metrics: {}
- performance_stats: {'total_evaluations': 0, 'successful_evaluations': 0, 'failed_evaluations': 0, 'total_execution_time': 0.0001652240753173828, 'backtest_operations': 0, 'vectorbt_operations': 0}
- success: False
- error_message: No price data available for economic validation

## ⚡ VectorBT Optimizations

- performance_stats: {'total_operations': 1, 'vectorbt_operations': 1, 'fallback_operations': 0, 'gpu_operations': 0, 'total_execution_time': 0.005192279815673828, 'memory_usage': 0.0}
- success: True
- error_message: None
