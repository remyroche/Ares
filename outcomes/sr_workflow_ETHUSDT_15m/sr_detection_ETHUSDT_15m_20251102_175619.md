# SR Workflow Summary Report

**Generated:** 2025-11-02 17:57:44
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 85.41 seconds
- **Steps Completed:** 4/4
- **Steps Failed:** 0/4
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-02 17:57:44.546343

## Steps Completed

✅ ml_model_training
✅ sr_parameter_optimization
✅ sr_detection
✅ sr_filtering

## Artifacts Created

### ml_training

- **training_data_path:** `data_cache/sr_ml_training/sr_quality_training_data.parquet`
- **model_path:** `models/sr_quality_model.lgb`
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 43, 'val_samples': 40, 'train_rmse': 0.1492316829382799, 'val_rmse': 0.19216190572986233, 'train_r2': -0.10885068818370214, 'val_r2': -0.45595082490512784, 'train_mae': 0.11904021410825134, 'val_mae': 0.1471251270150068, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 83, 'val_samples': 40, 'train_rmse': 0.16250574614107804, 'val_rmse': 0.16116937535831072, 'train_r2': -0.12136196402511557, 'val_r2': -0.024108047212835215, 'train_mae': 0.13002088187930846, 'val_mae': 0.1302413063646622, 'num_boost_rounds': 1}, {'fold': 2, 'train_samples': 123, 'val_samples': 40, 'train_rmse': 0.16545369843298108, 'val_rmse': 0.1504379855317096, 'train_r2': -0.1255608020660215, 'val_r2': -0.035625222889660524, 'train_mae': 0.13107008629482397, 'val_mae': 0.11205010198623196, 'num_boost_rounds': 1}, {'fold': 3, 'train_samples': 163, 'val_samples': 40, 'train_rmse': 0.16332000461044316, 'val_rmse': 0.1809932584486758, 'train_r2': -0.11819078745634881, 'val_r2': -0.46584843312596225, 'train_mae': 0.12659801455539382, 'val_mae': 0.13020768058166493, 'num_boost_rounds': 1}, {'fold': 4, 'train_samples': 203, 'val_samples': 40, 'train_rmse': 0.16329926005753806, 'val_rmse': 0.1783738764838958, 'train_r2': -0.1140730647236472, 'val_r2': -0.15965227136237248, 'train_mae': 0.12615189797123416, 'val_mae': 0.12293760496699715, 'num_boost_rounds': 1}], 'best_fold': 2, 'avg_metrics': {'avg_val_rmse': 0.17262728031049085, 'avg_val_r2': -0.22823695989919165, 'avg_val_mae': 0.12851236418291262, 'std_val_rmse': 0.014888001455781626, 'std_val_r2': 0.19584921253605972}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': 6, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 92, 'min_gain_to_split': 0.3, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'raw_lambda_l1': 0.8838136419028563, 'raw_lambda_l2': 0.5358933201225126, 'raw_learning_rate': 1.5178533030450705, 'raw_min_gain_to_split': -2.7636966675530217, 'raw_feature_fraction': -1.6116228212588446, 'raw_bagging_fraction': 3.625943614523706}, 'hpo_results': {'best_params': {'num_leaves': 31, 'max_depth': 6, 'raw_lambda_l1': 0.8838136419028563, 'raw_lambda_l2': 0.5358933201225126, 'min_data_in_leaf': 92, 'raw_learning_rate': 1.5178533030450705, 'raw_min_gain_to_split': -2.7636966675530217, 'raw_feature_fraction': -1.6116228212588446, 'raw_bagging_fraction': 3.625943614523706}, 'best_score': -0.03421177056805032, 'n_trials': 5, 'optimization_curve': [-0.03421177056805032, -0.03421177056805032, -0.03421177056805032, -0.03421177056805032, -0.03421177056805032], 'parameter_importance': {}}, 'hpo_best_params': {'num_leaves': 31, 'max_depth': 6, 'raw_lambda_l1': 0.8838136419028563, 'raw_lambda_l2': 0.5358933201225126, 'min_data_in_leaf': 92, 'raw_learning_rate': 1.5178533030450705, 'raw_min_gain_to_split': -2.7636966675530217, 'raw_feature_fraction': -1.6116228212588446, 'raw_bagging_fraction': 3.625943614523706}}`
- **shap_report:** `None`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251102_175618.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_175619.json`

### filtering

- **filtered_sr_levels:** `158`
- **removed_weak_levels:** `2`
- **strength_threshold:** `0.4`

## Metrics Summary

```json
{
  "optimization": {
    "data_points": 105092,
    "optimization_time": 43.00192999839783,
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

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251102_175619.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251102_175619.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_175619.md)
