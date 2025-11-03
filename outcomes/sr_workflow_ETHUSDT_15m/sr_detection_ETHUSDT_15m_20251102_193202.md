# SR Workflow Summary Report

**Generated:** 2025-11-02 19:34:56
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 174.32 seconds
- **Steps Completed:** 5/5
- **Steps Failed:** 0/5
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-02 19:34:56.891065

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
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 67, 'val_samples': 67, 'train_rmse': 0.17836138202741603, 'val_rmse': 0.14260087506290825, 'train_r2': -0.1820146521328807, 'val_r2': -0.01912624216427905, 'train_mae': 0.1536263824612734, 'val_mae': 0.12028435836619499, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 134, 'val_samples': 67, 'train_rmse': 0.17135075093245516, 'val_rmse': 0.17373473355890626, 'train_r2': -0.15424603613240984, 'val_r2': -0.07260592523195464, 'train_mae': 0.14038133293225455, 'val_mae': 0.1409449454277341, 'num_boost_rounds': 1}, {'fold': 2, 'train_samples': 201, 'val_samples': 67, 'train_rmse': 0.17475715696684507, 'val_rmse': 0.17060069035966213, 'train_r2': -0.15655483373597456, 'val_r2': -0.23929488588555747, 'train_mae': 0.14123835080721045, 'val_mae': 0.1350134567982117, 'num_boost_rounds': 1}], 'best_fold': 0, 'avg_metrics': {'avg_val_rmse': 0.16231209966049223, 'avg_val_r2': -0.11034235109393038, 'avg_val_mae': 0.13208092019738024, 'std_val_rmse': 0.01399654333626655, 'std_val_r2': 0.09376063956388844}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': 6, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 191, 'min_gain_to_split': 0.3, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'log_lambda_l1': 3.088579719829568, 'log_lambda_l2': -0.5004423350521505, 'raw_learning_rate': -1.695175442317491, 'raw_min_gain_to_split': 0.03552758863383976, 'raw_feature_fraction': -4.517199626818565, 'raw_bagging_fraction': 1.1746529296428765}, 'feature_importance': [{'feature': 'feature_strength', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_touch_count', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_age_bars', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_consistency', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_avg_bounce_ratio', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_max_bounce_ratio', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_volume_confirmation', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_bounce_consistency', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_recency_weighted_strength', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_touch_quality_score', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_price_zscore', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_distance_to_current_pct', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_is_support', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_market_trend', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_is_uptrend', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_quality_tier', 'importance': 0.0, 'importance_pct': nan}], 'hpo_results': {'best_params': {'num_leaves': 31, 'max_depth': 6, 'log_lambda_l1': 3.088579719829568, 'log_lambda_l2': -0.5004423350521505, 'min_data_in_leaf': 191, 'raw_learning_rate': -1.695175442317491, 'raw_min_gain_to_split': 0.03552758863383976, 'raw_feature_fraction': -4.517199626818565, 'raw_bagging_fraction': 1.1746529296428765, 'bagging_freq': 5}, 'best_score': -0.032026863194349615, 'n_trials': 5, 'optimization_curve': [-0.032026863194349615, -0.032026863194349615, -0.032026863194349615, -0.032026863194349615, -0.032026863194349615], 'parameter_importance': {}}, 'hpo_best_params': {'num_leaves': 31, 'max_depth': 6, 'log_lambda_l1': 3.088579719829568, 'log_lambda_l2': -0.5004423350521505, 'min_data_in_leaf': 191, 'raw_learning_rate': -1.695175442317491, 'raw_min_gain_to_split': 0.03552758863383976, 'raw_feature_fraction': -4.517199626818565, 'raw_bagging_fraction': 1.1746529296428765, 'bagging_freq': 5}}`
- **shap_report:** `None`

### ml_validation

- **validation_results:** `{'separation': {'mean_strong': 0.5533787987949503, 'mean_weak': 0.5533787987949504, 'median_strong': 0.5533787987949504, 'median_weak': 0.5533787987949504, 'separation': -1.1102230246251565e-16, 'weak_above_strong_median_pct': 100.0, 'strong_below_weak_median_pct': 100.0}, 'future_generalization': {'r2': None}, 'sample_size_check': {'total_samples': 304, 'strong_samples': 42}}`
- **tests_passed:** `0`
- **total_tests:** `1`
- **success_rate:** `0.0`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251102_193201.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_193202.json`

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
    "optimization_time": 91.83472490310669,
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

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251102_193202.md)
- [ml_model_validation](outcomes/sr_workflow_ETHUSDT_15m/ml_model_validation_ETHUSDT_15m_20251102_193202.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251102_193202.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_193202.md)
