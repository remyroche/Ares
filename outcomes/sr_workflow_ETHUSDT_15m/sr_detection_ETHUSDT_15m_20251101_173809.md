# SR Workflow Summary Report

**Generated:** 2025-11-01 17:40:47
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 157.79 seconds
- **Steps Completed:** 4/4
- **Steps Failed:** 0/4
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-01 17:40:47.381662

## Steps Completed

✅ ml_model_training
✅ sr_parameter_optimization
✅ sr_detection
✅ sr_filtering

## Artifacts Created

### ml_training

- **training_data_path:** `data_cache/sr_ml_training/sr_quality_training_data.parquet`
- **model_path:** `models/sr_quality_model.lgb`
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 151, 'val_samples': 150, 'train_rmse': 0.2479955974550698, 'val_rmse': 0.29280291058763835, 'train_r2': -0.239034073920112, 'val_r2': -0.5177891153802414, 'train_mae': 0.20084099775178152, 'val_mae': 0.2632049016941577, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 301, 'val_samples': 150, 'train_rmse': 0.2517315904883694, 'val_rmse': 0.27091174310991184, 'train_r2': -0.17319483805291402, 'val_r2': -0.28416742416644114, 'train_mae': 0.21735107397959347, 'val_mae': 0.24215443862798328, 'num_boost_rounds': 70}, {'fold': 2, 'train_samples': 451, 'val_samples': 150, 'train_rmse': 0.2473483899526449, 'val_rmse': 0.26123144891521355, 'train_r2': -0.1074873258451885, 'val_r2': -0.18392945584621012, 'train_mae': 0.21570415883070843, 'val_mae': 0.23486793698609232, 'num_boost_rounds': 105}, {'fold': 3, 'train_samples': 601, 'val_samples': 150, 'train_rmse': 0.23855640763132518, 'val_rmse': 0.28647144113756895, 'train_r2': -0.017323008878193713, 'val_r2': -0.36803936701843876, 'train_mae': 0.20787920424581866, 'val_mae': 0.26074372180634164, 'num_boost_rounds': 100}, {'fold': 4, 'train_samples': 751, 'val_samples': 150, 'train_rmse': 0.25008425766618136, 'val_rmse': 0.2636065317862647, 'train_r2': -0.10032524575437574, 'val_r2': -0.10667463408101052, 'train_mae': 0.22195560795179226, 'val_mae': 0.24027804433819283, 'num_boost_rounds': 60}], 'best_fold': 2, 'avg_metrics': {'avg_val_rmse': 0.27500481510731944, 'avg_val_r2': -0.2921199992984684, 'avg_val_mae': 0.24824980869055358, 'std_val_rmse': 0.012527054183238667, 'std_val_r2': 0.1434215611213953}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 17, 'max_depth': 5, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 98, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'raw_lambda_l1': 0.3130842721896675, 'raw_lambda_l2': 0.3852303603783908, 'raw_learning_rate': 0.9738189819816396, 'raw_feature_fraction': -2.6668567105042964, 'raw_bagging_fraction': -2.485068310945122}, 'hpo_results': {'best_params': {'num_leaves': 17, 'max_depth': 5, 'raw_lambda_l1': 0.3130842721896675, 'raw_lambda_l2': 0.3852303603783908, 'min_data_in_leaf': 98, 'raw_learning_rate': 0.9738189819816396, 'raw_feature_fraction': -2.6668567105042964, 'raw_bagging_fraction': -2.485068310945122}, 'best_score': -0.058812036139464084, 'n_trials': 100, 'optimization_curve': [-0.059554908626314206, -0.06037434767675069, -0.060681316724706426, -0.06025665149306714, -0.05998247322389535, -0.06003649966514344, -0.0598544633939283, -0.06026529656754113, -0.05959676344046905, -0.06071560334880306, -0.059351819515706804, -0.059340929416042784, -0.0592311091672737, -0.05951435038702417, -0.05908829213815814, -0.059291129900723084, -0.060171124940890774, -0.059166762713079336, -0.059620941320607156, -0.059540425217980884, -0.05928637861205921, -0.05918669111150197, -0.05912746432210382, -0.059628874064556504, -0.05924962271207823, -0.05974534388446603, -0.059185216563053554, -0.05911547202082603, -0.058812036139464084, -0.05965681978190156, -0.0594680602382938, -0.05914507136038033, -0.05914257551022182, -0.059353690809497386, -0.05983253647941165, -0.061160287759960366, -0.06005055232299197, -0.05946139519668876, -0.05939287037256469, -0.05925406271520567, -0.06073043556927775, -0.05954557519877914, -0.05987068065391381, -0.0589222868989693, -0.059641545184038434, -0.05934202611599506, -0.05948806115485865, -0.05989154064893768, -0.05947840131026003, -0.06059013185079052, -0.06002059807879524, -0.058948593011286465, -0.05924113753209583, -0.05934383194079783, -0.05967270009838357, -0.059512562044597095, -0.05924937826027381, -0.061942660269936015, -0.05955027959085422, -0.059219424865281665, -0.05942176653266431, -0.059018928650302414, -0.0594306330176532, -0.058986119222349906, -0.0593115357088377, -0.0594027974755718, -0.05935711292904331, -0.05898730596244125, -0.05914272935510727, -0.059360801093402284, -0.059715262430279845, -0.05919617063621622, -0.05922658594579098, -0.05913201355437601, -0.059194812265090956, -0.06006485878106842, -0.05893310690337176, -0.058891152187047435, -0.05890053818249996, -0.05908646242535921, -0.059216571477384364, -0.05891028793499185, -0.058953926480497465, -0.05910933674029596, -0.059032275374792, -0.05898285041340869, -0.0598443030693621, -0.05927065508503606, -0.058963257872529776, -0.059225583694552345, -0.05915451830936717, -0.059120960210737296, -0.059028232577937455, -0.05899128037692815, -0.05949569512529919, -0.05927710525850638, -0.059205387374171324, -0.05941871926830526, -0.05896432253873858, -0.05945031952750142], 'parameter_importance': {'min_data_in_leaf': 0.3947910798536834, 'raw_bagging_fraction': 0.29029274347269785, 'raw_feature_fraction': 0.15541824760623862, 'raw_learning_rate': 0.05763155380227991, 'raw_lambda_l1': 0.041277863082433044, 'raw_lambda_l2': 0.02467044260673993, 'max_depth': 0.019835209763138337, 'num_leaves': 0.016082859812788928}}, 'hpo_best_params': {'num_leaves': 17, 'max_depth': 5, 'raw_lambda_l1': 0.3130842721896675, 'raw_lambda_l2': 0.3852303603783908, 'min_data_in_leaf': 98, 'raw_learning_rate': 0.9738189819816396, 'raw_feature_fraction': -2.6668567105042964, 'raw_bagging_fraction': -2.485068310945122}, 'ranking_metrics': {'precision_at_k': 0.1, 'spearman_rho': -0.2652121253111717, 'spearman_p_value': 1.089326397542946e-30, 'ndcg_at_k': 0.26851688184375333, 'r2_score': -1.853526685573208, 'rmse': 0.43234835706164904, 'k': 10, 'quality_threshold': 0.7, 'total_samples': 1821}}`
- **shap_report:** `None`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251101_173809.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_173809.json`

### filtering

- **filtered_sr_levels:** `158`
- **removed_weak_levels:** `2`
- **strength_threshold:** `0.4`

## Metrics Summary

```json
{
  "optimization": {
    "data_points": 105092,
    "optimization_time": 49.41963005065918,
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

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251101_173809.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251101_173809.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_173809.md)
