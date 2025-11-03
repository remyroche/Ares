# SR Workflow Summary Report

**Generated:** 2025-11-02 19:40:53
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 61.43 seconds
- **Steps Completed:** 5/5
- **Steps Failed:** 0/5
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-02 19:40:53.100559

## Steps Completed

✅ ml_model_training
✅ ml_model_validation
✅ sr_parameter_optimization
✅ sr_detection
✅ sr_filtering

## Artifacts Created

### ml_training

- **training_data_path:** `data_cache/sr_ml_training/sr_quality_training_data.parquet`
- **model_path:** `models/sr_quality_model.lgb`
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 67, 'val_samples': 67, 'train_rmse': 0.17836138202741603, 'val_rmse': 0.14260087506290825, 'train_r2': -0.1820146521328807, 'val_r2': -0.01912624216427905, 'train_mae': 0.1536263824612734, 'val_mae': 0.12028435836619499, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 134, 'val_samples': 67, 'train_rmse': 0.17135075093245516, 'val_rmse': 0.17373473355890626, 'train_r2': -0.15424603613240984, 'val_r2': -0.07260592523195464, 'train_mae': 0.14038133293225455, 'val_mae': 0.1409449454277341, 'num_boost_rounds': 1}, {'fold': 2, 'train_samples': 201, 'val_samples': 67, 'train_rmse': 0.17392584236545486, 'val_rmse': 0.167832526272362, 'train_r2': -0.14557761120924018, 'val_r2': -0.19940361241605475, 'train_mae': 0.1398291307034885, 'val_mae': 0.13270038178523047, 'num_boost_rounds': 4}], 'best_fold': 0, 'avg_metrics': {'avg_val_rmse': 0.16138937829805886, 'avg_val_r2': -0.09704525993742948, 'avg_val_mae': 0.13130989519305317, 'std_val_rmse': 0.013502219646568784, 'std_val_r2': 0.07559957392674743}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': 6, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 38, 'min_gain_to_split': 0.3, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'log_lambda_l1': 2.73584334142081, 'log_lambda_l2': 0.42984567536050944, 'raw_learning_rate': 5.354865650929355, 'raw_min_gain_to_split': 0.906901684827254, 'raw_feature_fraction': 2.733852293213708, 'raw_bagging_fraction': -1.0226267225276438}, 'feature_importance': [{'feature': 'feature_strength', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_touch_count', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_age_bars', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_consistency', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_avg_bounce_ratio', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_max_bounce_ratio', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_volume_confirmation', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_bounce_consistency', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_recency_weighted_strength', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_touch_quality_score', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_price_zscore', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_distance_to_current_pct', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_is_support', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_market_trend', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_is_uptrend', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_quality_tier', 'importance': 0.0, 'importance_pct': nan}], 'hpo_results': {'best_params': {'num_leaves': 31, 'max_depth': 6, 'log_lambda_l1': 2.73584334142081, 'log_lambda_l2': 0.42984567536050944, 'min_data_in_leaf': 38, 'raw_learning_rate': 5.354865650929355, 'raw_min_gain_to_split': 0.906901684827254, 'raw_feature_fraction': 2.733852293213708, 'raw_bagging_fraction': -1.0226267225276438, 'bagging_freq': 5}, 'best_score': -0.032026863194349615, 'n_trials': 5, 'optimization_curve': [-0.032026863194349615, -0.032026863194349615, -0.032026863194349615, -0.032026863194349615, -0.032026863194349615], 'parameter_importance': {}}, 'hpo_best_params': {'num_leaves': 31, 'max_depth': 6, 'log_lambda_l1': 2.73584334142081, 'log_lambda_l2': 0.42984567536050944, 'min_data_in_leaf': 38, 'raw_learning_rate': 5.354865650929355, 'raw_min_gain_to_split': 0.906901684827254, 'raw_feature_fraction': 2.733852293213708, 'raw_bagging_fraction': -1.0226267225276438, 'bagging_freq': 5}}`
- **shap_report:** `None`

### ml_validation

- **validation_results:** `{'separation': {'mean_strong': 0.5533787987949503, 'mean_weak': 0.5533787987949504, 'median_strong': 0.5533787987949504, 'median_weak': 0.5533787987949504, 'separation': -1.1102230246251565e-16, 'weak_above_strong_median_pct': 100.0, 'strong_below_weak_median_pct': 100.0}, 'future_generalization': {'r2': None}, 'sample_size_check': {'total_samples': 304, 'strong_samples': 42}}`
- **tests_passed:** `0`
- **total_tests:** `1`
- **success_rate:** `0.0`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251102_193951.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_193951.json`

### filtering

- **filtered_sr_levels:** `158`
- **removed_weak_levels:** `2`
- **strength_threshold:** `0.4`

## Metrics Summary

```json
{
  "ml_validation": {
    "tests_passed": 0,
    "total_tests": 1,
    "success_rate": 0.0,
    "precision_at_10": null,
    "spearman_rho": null,
    "separation": -1.1102230246251565e-16
  },
  "optimization": {
    "data_points": 105092,
    "optimization_time": 31.484783172607422,
    "best_score": 0.9487210233079597,
    "total_combinations_tested": 76891,
    "performance_improvements": {
      "vectorbt_speedup": 1.0,
      "hardware_optimization_gains": {
        "cpu_optimization": 1.0,
        "memory_optimization": 1.0,
        "gpu_acceleration": 1.0
      },
      "bayesian_efficiency": 0.0
    }
  },
  "detection": {
    "total_levels": 160,
    "support_levels": 94,
    "resistance_levels": 66,
    "ml_model_used": true
  },
  "filtering": {
    "total_levels_before": 160,
    "total_levels_after": 158,
    "weak_levels_removed": 2,
    "retention_rate": 0.9875
  }
}
```

## Individual Step Reports

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251102_193951.md)
- [ml_model_validation](outcomes/sr_workflow_ETHUSDT_15m/ml_model_validation_ETHUSDT_15m_20251102_193951.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251102_193951.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_193951.md)
