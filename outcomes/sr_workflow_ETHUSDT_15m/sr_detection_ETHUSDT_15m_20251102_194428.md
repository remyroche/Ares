# SR Workflow Summary Report

**Generated:** 2025-11-02 19:46:06
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 98.02 seconds
- **Steps Completed:** 5/5
- **Steps Failed:** 0/5
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-02 19:46:06.721731

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
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 68, 'val_samples': 67, 'train_rmse': 0.12789962556998644, 'val_rmse': 0.19981305395468293, 'train_r2': -0.10998231220203247, 'val_r2': -0.0004294583498773985, 'train_mae': 0.10539858439647834, 'val_mae': 0.17447402245947913, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 135, 'val_samples': 67, 'train_rmse': 0.18171419608646155, 'val_rmse': 0.14592619616926225, 'train_r2': -0.19830164532430983, 'val_r2': -0.024703449314032122, 'train_mae': 0.15528510219021532, 'val_mae': 0.11713398933185869, 'num_boost_rounds': 1}, {'fold': 2, 'train_samples': 202, 'val_samples': 67, 'train_rmse': 0.1811032450058978, 'val_rmse': 0.17428571175448543, 'train_r2': -0.19805799391996914, 'val_r2': -0.029570600672157, 'train_mae': 0.14997566481801045, 'val_mae': 0.14147033031639317, 'num_boost_rounds': 1}], 'best_fold': 1, 'avg_metrics': {'avg_val_rmse': 0.17334165395947687, 'avg_val_r2': -0.01823450277868884, 'avg_val_mae': 0.14435944736924367, 'std_val_rmse': 0.022009343384674546, 'std_val_r2': 0.012745901195278065}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': 6, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 115, 'min_gain_to_split': 0.3, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'log_lambda_l1': 1.893996939982785, 'log_lambda_l2': 1.1345526674883848, 'raw_learning_rate': -2.7594016464194673, 'raw_min_gain_to_split': 0.8493088619436976, 'raw_feature_fraction': -1.1429196233796635, 'raw_bagging_fraction': 5.415009397607694}, 'feature_importance': [{'feature': 'feature_strength', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_touch_quality_score', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_is_uptrend', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_hour_of_day', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_market_trend', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_is_support', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_distance_to_current_pct', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_price_zscore', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_recency_weighted_strength', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_touch_count', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_bounce_consistency', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_volume_confirmation', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_max_bounce_ratio', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_avg_bounce_ratio', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_consistency', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_age_bars', 'importance': 0.0, 'importance_pct': nan}, {'feature': 'feature_quality_tier', 'importance': 0.0, 'importance_pct': nan}], 'hpo_results': {'best_params': {'num_leaves': 31, 'max_depth': 6, 'log_lambda_l1': 1.893996939982785, 'log_lambda_l2': 1.1345526674883848, 'min_data_in_leaf': 115, 'raw_learning_rate': -2.7594016464194673, 'raw_min_gain_to_split': 0.8493088619436976, 'raw_feature_fraction': -1.1429196233796635, 'raw_bagging_fraction': 5.415009397607694, 'bagging_freq': 5}, 'best_score': -0.04097021915992593, 'n_trials': 5, 'optimization_curve': [-0.04097021915992593, -0.04097021915992593, -0.04097021915992593, -0.04097021915992593, -0.04097021915992593], 'parameter_importance': {}}, 'hpo_best_params': {'num_leaves': 31, 'max_depth': 6, 'log_lambda_l1': 1.893996939982785, 'log_lambda_l2': 1.1345526674883848, 'min_data_in_leaf': 115, 'raw_learning_rate': -2.7594016464194673, 'raw_min_gain_to_split': 0.8493088619436976, 'raw_feature_fraction': -1.1429196233796635, 'raw_bagging_fraction': 5.415009397607694, 'bagging_freq': 5}}`
- **shap_report:** `None`

### ml_validation

- **validation_results:** `{'precision_at_k': {5: 1.0, 10: 1.0, 20: 1.0, 50: 1.0}, 'spearman': nan, 'spearman_pvalue': nan, 'separation': {'mean_strong': 0.552714777496176, 'mean_weak': 0.5527147774961759, 'median_strong': 0.552714777496176, 'median_weak': 0.552714777496176, 'separation': 1.1102230246251565e-16, 'weak_above_strong_median_pct': 100.0, 'strong_below_weak_median_pct': 100.0}, 'future_generalization': {'r2': None}, 'sample_size_check': {'total_samples': 317, 'strong_samples': 50}}`
- **tests_passed:** `4`
- **total_tests:** `6`
- **success_rate:** `66.66666666666666`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251102_194428.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_194428.json`

### filtering

- **filtered_sr_levels:** `158`
- **removed_weak_levels:** `2`
- **strength_threshold:** `0.4`

## Metrics Summary

```json
{
  "ml_validation": {
    "tests_passed": 4,
    "total_tests": 6,
    "success_rate": 66.66666666666666,
    "precision_at_10": 1.0,
    "spearman_rho": NaN,
    "separation": 1.1102230246251565e-16
  },
  "optimization": {
    "data_points": 105092,
    "optimization_time": 49.234522104263306,
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

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251102_194428.md)
- [ml_model_validation](outcomes/sr_workflow_ETHUSDT_15m/ml_model_validation_ETHUSDT_15m_20251102_194428.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251102_194428.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_194428.md)
