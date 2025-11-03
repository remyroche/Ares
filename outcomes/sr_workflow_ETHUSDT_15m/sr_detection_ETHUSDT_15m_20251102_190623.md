# SR Workflow Summary Report

**Generated:** 2025-11-02 19:08:39
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 135.97 seconds
- **Steps Completed:** 5/5
- **Steps Failed:** 0/5
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-02 19:08:39.716431

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
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 50, 'val_samples': 47, 'train_rmse': 0.24790368475730648, 'val_rmse': 0.2013041677333296, 'train_r2': -0.3380227745437172, 'val_r2': -0.059732738813661035, 'train_mae': 0.2241891758275071, 'val_mae': 0.16817852983909337, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 97, 'val_samples': 47, 'train_rmse': 0.14588836136400174, 'val_rmse': 0.1281516963096964, 'train_r2': 0.5127130673273779, 'val_r2': 0.3954151395762927, 'train_mae': 0.1218020697089006, 'val_mae': 0.10850220602204223, 'num_boost_rounds': 47}, {'fold': 2, 'train_samples': 144, 'val_samples': 47, 'train_rmse': 0.10678836710529935, 'val_rmse': 0.11712773068668017, 'train_r2': 0.702795112510405, 'val_r2': 0.6326516150974697, 'train_mae': 0.0897686769786408, 'val_mae': 0.08961583486206105, 'num_boost_rounds': 51}, {'fold': 3, 'train_samples': 191, 'val_samples': 47, 'train_rmse': 0.1026652260426122, 'val_rmse': 0.08626972815227922, 'train_r2': 0.7235596641449618, 'val_r2': 0.6596133939637993, 'train_mae': 0.084325493355182, 'val_mae': 0.058922948797485754, 'num_boost_rounds': 52}, {'fold': 4, 'train_samples': 238, 'val_samples': 47, 'train_rmse': 0.09348442628244756, 'val_rmse': 0.09608938729617743, 'train_r2': 0.7497374528962218, 'val_r2': 0.5493907386312986, 'train_mae': 0.07388188810415046, 'val_mae': 0.07007060124822105, 'num_boost_rounds': 59}], 'best_fold': 3, 'avg_metrics': {'avg_val_rmse': 0.12578854203563256, 'avg_val_r2': 0.4354676296910398, 'avg_val_mae': 0.0990580241537807, 'std_val_rmse': 0.04056348189898168, 'std_val_r2': 0.2641699942932127}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 23, 'max_depth': 5, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 30, 'min_gain_to_split': 0.3, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'log_lambda_l1': -0.051341460715933485, 'log_lambda_l2': 0.47842007482163607, 'raw_learning_rate': -0.3979674221745455, 'raw_min_gain_to_split': 0.0016648897268919521, 'raw_feature_fraction': 0.9371596409151206, 'raw_bagging_fraction': 1.0986803229677458}, 'feature_importance': [{'feature': 'rejection_speed', 'importance': 28.809488773345947, 'importance_pct': 43.06545966045947}, {'feature': 'hold_quality', 'importance': 22.886512756347656, 'importance_pct': 34.21158215028648}, {'feature': 'speed_quality', 'importance': 9.156093150377274, 'importance_pct': 13.686857247525753}, {'feature': 'bounce_quality', 'importance': 6.044877976179123, 'importance_pct': 9.03610094172829}, {'feature': 'feature_volume_regime', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_regime_volatility', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_market_momentum', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_rejection_velocity', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_strength_in_ranging', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_prominence', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_distance_x_volatility', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_momentum_adjusted_distance', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_median_bounce_ratio', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_market_trend', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_day_of_week', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_recent_touch_rate', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_relative_strength_rank', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_success_rate', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_bars_since_last_touch', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_distance_to_nearest_level', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_avg_time_between_touches', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_regime_trend_strength', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_mtf_x_prominence', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_recency_x_strength', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_quality_tier', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_level_age_days', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_strength_x_volume', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_touch_count', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_touch_x_consistency', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'trade_quality', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'max_bounce_strength', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'volume_quality', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_price_zscore', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_width', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_price_position', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_strength_x_momentum', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_distance_to_current_pct', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_avg_bounce_ratio', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_distance_x_velocity', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_trend_aligned_strength', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_failure_count', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_touch_frequency', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_cluster_density', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_multi_tf_score', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_success_x_strength', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_recency_weighted_strength', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_volume_x_trend', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_strength', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_touch_quality_ratio', 'importance': 0.0, 'importance_pct': 0.0}, {'feature': 'feature_quality_composite', 'importance': 0.0, 'importance_pct': 0.0}], 'hpo_results': {'best_params': {'num_leaves': 23, 'max_depth': 5, 'log_lambda_l1': -0.051341460715933485, 'log_lambda_l2': 0.47842007482163607, 'min_data_in_leaf': 30, 'raw_learning_rate': -0.3979674221745455, 'raw_min_gain_to_split': 0.0016648897268919521, 'raw_feature_fraction': 0.9371596409151206, 'raw_bagging_fraction': 1.0986803229677458, 'bagging_freq': 10}, 'best_score': -0.020033563102522505, 'n_trials': 100, 'optimization_curve': [-0.0385574954532656, -0.03915835115855752, -0.04009350482512517, -0.04009350482512517, -0.032121258501744215, -0.03780321437854565, -0.03635447331292488, -0.01838167460202778, -0.023820657607227687, -0.023820657607227687, -0.031125940226644312, -0.031072194349278028, -0.030452769715289006, -0.030605490742926473, -0.023820657607227687, -0.03773194277762218, -0.012672718612638715, -0.014261666921660676, -0.014595185331649324, -0.029345195271193357, -0.01251984246488711, -0.031398241031665095, -0.030301322769552587, -0.033636815902901816, -0.027865615776404822, -0.01274732268276805, -0.04414539424776822, -0.026514170439657737, -0.032638595726413144, -0.03233470805502013, -0.02826790559655549, -0.0257924783413572, -0.0263181584775788, -0.031638229056740796, -0.03250758241029174, -0.03159621798858232, -0.024776624449106686, -0.025199906423866752, -0.027433462300428906, -0.02786662568306082, -0.02818963375750558, -0.026640717136535978, -0.024810434730886736, -0.022217424397317324, -0.022987368557443807, -0.0228985777947184, -0.021797300280744926, -0.022177552637278775, -0.023340560715833496, -0.02428964179093625, -0.030692741878726608, -0.010737126870583256, -0.02306335212252985, -0.04414539424776822, -0.04414539424776822, -0.02770766656789193, -0.028043717521149314, -0.027223621278265837, -0.03224037548700648, -0.028747718242617498, -0.022054999030915966, -0.02308262667951557, -0.012187634760551189, -0.025118007625192153, -0.030580247819484586, -0.025548936936908996, -0.04414539424776822, -0.010877770997392087, -0.025513489361978687, -0.026499624592737202, -0.04414539424776822, -0.009163439225686442, -0.022247725548242177, -0.022380140089268953, -0.026110935071323636, -0.02450063447705792, -0.009400337374412485, -0.026834074894863613, -0.023738560408161313, -0.04414539424776822, -0.023622720332533466, -0.010168373624741163, -0.02267194639283037, -0.02343803728756305, -0.024388075814053483, -0.02013101920410506, -0.011427337163037394, -0.02650641552976024, -0.020702547405416064, -0.02506580518715189, -0.022116564304340818, -0.009607440878056098, -0.020033563102522505, -0.02061661226836132, -0.022832674119653874, -0.024350062434859147, -0.02092049703323484, -0.021707729622119375, -0.021155171719861383, -0.024822647063323964], 'parameter_importance': {'log_lambda_l1': 0.26461948814989716, 'min_data_in_leaf': 0.2611367095365916, 'raw_min_gain_to_split': 0.24134244569597715, 'raw_learning_rate': 0.0483684012581519, 'log_lambda_l2': 0.04573404412928432, 'bagging_freq': 0.04280612303386816, 'raw_feature_fraction': 0.03378800894324343, 'raw_bagging_fraction': 0.03249332218391769, 'num_leaves': 0.027347423279653636, 'max_depth': 0.002364033789415076}}, 'hpo_best_params': {'num_leaves': 23, 'max_depth': 5, 'log_lambda_l1': -0.051341460715933485, 'log_lambda_l2': 0.47842007482163607, 'min_data_in_leaf': 30, 'raw_learning_rate': -0.3979674221745455, 'raw_min_gain_to_split': 0.0016648897268919521, 'raw_feature_fraction': 0.9371596409151206, 'raw_bagging_fraction': 1.0986803229677458, 'bagging_freq': 10}}`
- **shap_report:** `None`

### ml_validation

- **validation_results:** `{'precision_at_k': {5: 1.0, 10: 1.0, 20: 1.0, 50: 1.0}, 'spearman': 0.8522441826579998, 'spearman_pvalue': 6.169953778223678e-24, 'separation': {'mean_strong': 0.7860932250173134, 'mean_weak': 0.5482404888925924, 'median_strong': 0.7720786777563549, 'median_weak': 0.5473239770935154, 'separation': 0.23785273612472102, 'weak_above_strong_median_pct': 0.0, 'strong_below_weak_median_pct': 0.0}, 'future_generalization': {'r2': None}, 'sample_size_check': {'total_samples': 319, 'strong_samples': 81}}`
- **tests_passed:** `5`
- **total_tests:** `6`
- **success_rate:** `83.33333333333334`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251102_190623.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_190623.json`

### filtering

- **filtered_sr_levels:** `158`
- **removed_weak_levels:** `2`
- **strength_threshold:** `0.4`

## Metrics Summary

```json
{
  "ml_validation": {
    "tests_passed": 5,
    "total_tests": 6,
    "success_rate": 83.33333333333334,
    "precision_at_10": 1.0,
    "spearman_rho": 0.8522441826579998,
    "separation": 0.23785273612472102
  },
  "optimization": {
    "data_points": 105092,
    "optimization_time": 62.587427854537964,
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

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251102_190623.md)
- [ml_model_validation](outcomes/sr_workflow_ETHUSDT_15m/ml_model_validation_ETHUSDT_15m_20251102_190623.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251102_190623.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_190623.md)
