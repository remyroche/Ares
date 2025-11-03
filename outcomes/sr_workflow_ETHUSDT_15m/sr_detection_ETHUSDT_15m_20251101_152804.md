# SR Workflow Summary Report

**Generated:** 2025-11-01 15:32:37
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 272.19 seconds
- **Steps Completed:** 4/4
- **Steps Failed:** 0/4
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-01 15:32:37.092968

## Steps Completed

✅ ml_model_training
✅ sr_parameter_optimization
✅ sr_detection
✅ sr_filtering

## Artifacts Created

### ml_training

- **training_data_path:** `data_cache/sr_ml_training/sr_quality_training_data.parquet`
- **model_path:** `models/sr_quality_model.lgb`
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 1313, 'val_samples': 1308, 'train_rmse': 0.17636550050288113, 'val_rmse': 0.23573092487477657, 'train_r2': 0.522843493835491, 'val_r2': 0.20449879036870988, 'train_mae': 0.13288152064909822, 'val_mae': 0.18187410070587337, 'num_boost_rounds': 31}, {'fold': 1, 'train_samples': 2621, 'val_samples': 1308, 'train_rmse': 0.2072684249887262, 'val_rmse': 0.24377599852414197, 'train_r2': 0.3652117611034039, 'val_r2': 0.09549128646368321, 'train_mae': 0.16208969376601473, 'val_mae': 0.18208880990842446, 'num_boost_rounds': 20}, {'fold': 2, 'train_samples': 3929, 'val_samples': 1308, 'train_rmse': 0.2206652152370078, 'val_rmse': 0.23890759742438683, 'train_r2': 0.273446736095033, 'val_r2': 0.08210592450128262, 'train_mae': 0.17301130745990914, 'val_mae': 0.17273171685221567, 'num_boost_rounds': 15}, {'fold': 3, 'train_samples': 5237, 'val_samples': 1308, 'train_rmse': 0.20695653833477295, 'val_rmse': 0.24512169416122567, 'train_r2': 0.3491944451192409, 'val_r2': 0.17214232395039675, 'train_mae': 0.15800210594429848, 'val_mae': 0.18325185406730993, 'num_boost_rounds': 30}, {'fold': 4, 'train_samples': 6545, 'val_samples': 1308, 'train_rmse': 0.21277103314326448, 'val_rmse': 0.23907102117609885, 'train_r2': 0.3260817642152183, 'val_r2': 0.1257904650604179, 'train_mae': 0.1626239584071963, 'val_mae': 0.19680103676763788, 'num_boost_rounds': 30}], 'best_fold': 0, 'avg_metrics': {'avg_val_rmse': 0.24052144723212598, 'avg_val_r2': 0.13600575806889809, 'avg_val_mae': 0.18334950366029226, 'std_val_rmse': 0.0034470778152882135, 'std_val_r2': 0.046147835369920585}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'max_depth': 6, 'min_data_in_leaf': 20, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'seed': 42, 'force_col_wise': True}}`
- **shap_report:** `None`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251101_152804.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_152804.json`

### filtering

- **filtered_sr_levels:** `151`
- **removed_weak_levels:** `9`
- **strength_threshold:** `0.5`

## Metrics Summary

```json
{
  "optimization": {
    "data_points": 105092,
    "optimization_time": 48.815253019332886,
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

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251101_152804.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251101_152804.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_152804.md)
