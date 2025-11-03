# SR Workflow Summary Report

**Generated:** 2025-11-01 15:55:02
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 258.98 seconds
- **Steps Completed:** 4/4
- **Steps Failed:** 0/4
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-01 15:55:02.476296

## Steps Completed

✅ ml_model_training
✅ sr_parameter_optimization
✅ sr_detection
✅ sr_filtering

## Artifacts Created

### ml_training

- **training_data_path:** `data_cache/sr_ml_training/sr_quality_training_data.parquet`
- **model_path:** `models/sr_quality_model.lgb`
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 646, 'val_samples': 646, 'train_rmse': 0.1973227424948091, 'val_rmse': 0.23574112726420382, 'train_r2': 0.3113084768479838, 'val_r2': 0.09004051007726455, 'train_mae': 0.17101992876909128, 'val_mae': 0.21283513072240462, 'num_boost_rounds': 98}, {'fold': 1, 'train_samples': 1292, 'val_samples': 646, 'train_rmse': 0.19438876643757147, 'val_rmse': 0.22551613255547312, 'train_r2': 0.35857580396814115, 'val_r2': 0.14431867441903534, 'train_mae': 0.1695540070479609, 'val_mae': 0.19584762530488184, 'num_boost_rounds': 140}, {'fold': 2, 'train_samples': 1938, 'val_samples': 646, 'train_rmse': 0.2062220337739724, 'val_rmse': 0.22333778787184205, 'train_r2': 0.2806012045760937, 'val_r2': 0.14764351097264905, 'train_mae': 0.18136098010994406, 'val_mae': 0.19653256867463795, 'num_boost_rounds': 89}, {'fold': 3, 'train_samples': 2584, 'val_samples': 646, 'train_rmse': 0.20864068475242878, 'val_rmse': 0.2396421902999061, 'train_r2': 0.2659530127996329, 'val_r2': 0.17422733459974582, 'train_mae': 0.18196825087946325, 'val_mae': 0.20574260767269803, 'num_boost_rounds': 80}, {'fold': 4, 'train_samples': 3230, 'val_samples': 646, 'train_rmse': 0.19605968226611195, 'val_rmse': 0.22262206009877417, 'train_r2': 0.3737259394917255, 'val_r2': 0.19059932484841335, 'train_mae': 0.16729552787783517, 'val_mae': 0.19211344588194762, 'num_boost_rounds': 210}], 'best_fold': 4, 'avg_metrics': {'avg_val_rmse': 0.22937185961803985, 'avg_val_r2': 0.14936587098342163, 'avg_val_mae': 0.200614275651314, 'std_val_rmse': 0.006969701704275524, 'std_val_r2': 0.034252676551166816}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 15, 'max_depth': 4, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 50, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True}}`
- **shap_report:** `None`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251101_155043.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_155043.json`

### filtering

- **filtered_sr_levels:** `151`
- **removed_weak_levels:** `9`
- **strength_threshold:** `0.5`

## Metrics Summary

```json
{
  "optimization": {
    "data_points": 105092,
    "optimization_time": 47.26648926734924,
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
    "total_levels_after": 151,
    "weak_levels_removed": 9,
    "retention_rate": 0.94375
  }
}
```

## Individual Step Reports

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251101_155043.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251101_155043.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_155043.md)
