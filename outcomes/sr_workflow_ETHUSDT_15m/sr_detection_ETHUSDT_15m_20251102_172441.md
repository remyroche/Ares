# SR Workflow Summary Report

**Generated:** 2025-11-02 17:26:52
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 130.70 seconds
- **Steps Completed:** 4/4
- **Steps Failed:** 0/4
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-02 17:26:52.111417

## Steps Completed

✅ ml_model_training
✅ sr_parameter_optimization
✅ sr_detection
✅ sr_filtering

## Artifacts Created

### ml_training

- **training_data_path:** `data_cache/sr_ml_training/sr_quality_training_data.parquet`
- **model_path:** `models/sr_quality_model.lgb`
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 151, 'val_samples': 150, 'train_rmse': 0.2280699869844332, 'val_rmse': 0.27934947263851306, 'train_r2': -0.04792833446268885, 'val_r2': -0.3815174240058732, 'train_mae': 0.18619598515142874, 'val_mae': 0.25183738933011945, 'num_boost_rounds': 43}, {'fold': 1, 'train_samples': 301, 'val_samples': 150, 'train_rmse': 0.19480655044994366, 'val_rmse': 0.25011127381346315, 'train_r2': 0.2974102394646567, 'val_r2': -0.09454223352935665, 'train_mae': 0.16369687436759373, 'val_mae': 0.22496292951683045, 'num_boost_rounds': 230}, {'fold': 2, 'train_samples': 451, 'val_samples': 150, 'train_rmse': 0.21624316188358414, 'val_rmse': 0.242700699758286, 'train_r2': 0.15354211457401812, 'val_r2': -0.02192015421430482, 'train_mae': 0.1891944010937921, 'val_mae': 0.2175563257045961, 'num_boost_rounds': 64}, {'fold': 3, 'train_samples': 601, 'val_samples': 150, 'train_rmse': 0.23954167397470746, 'val_rmse': 0.2802477873424571, 'train_r2': -0.025743692359724957, 'val_r2': -0.30924316092461646, 'train_mae': 0.21010174918955488, 'val_mae': 0.2570648019726417, 'num_boost_rounds': 28}, {'fold': 4, 'train_samples': 751, 'val_samples': 150, 'train_rmse': 0.24037365941078437, 'val_rmse': 0.2627094079219256, 'train_r2': -0.01653449055315237, 'val_r2': -0.09915482951651367, 'train_mae': 0.21293490173783433, 'val_mae': 0.2393641945110314, 'num_boost_rounds': 30}], 'best_fold': 2, 'avg_metrics': {'avg_val_rmse': 0.263023728294929, 'avg_val_r2': -0.18127556043813295, 'avg_val_mae': 0.23815712820704382, 'std_val_rmse': 0.015119880326105884, 'std_val_r2': 0.13866011729107738}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 11, 'max_depth': 6, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 31, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'raw_lambda_l1': 0.08921504761655664, 'raw_lambda_l2': 0.19013618781637237, 'raw_learning_rate': -0.0728406836083933, 'raw_feature_fraction': -2.0764272162485917, 'raw_bagging_fraction': -0.43285273256631757}, 'hpo_results': {'best_params': {'num_leaves': 11, 'max_depth': 6, 'raw_lambda_l1': 0.08921504761655664, 'raw_lambda_l2': 0.19013618781637237, 'min_data_in_leaf': 31, 'raw_learning_rate': -0.0728406836083933, 'raw_feature_fraction': -2.0764272162485917, 'raw_bagging_fraction': -0.43285273256631757}, 'best_score': -0.049384323798472504, 'n_trials': 100, 'optimization_curve': [-0.052918799370953326, -0.05121480184577072, -0.05325959629429974, -0.050677233568832115, -0.053013164393795734, -0.05159620477928752, -0.05251795094089988, -0.05252643277375796, -0.052358947757788175, -0.05206361825826996, -0.05143309851151082, -0.051158569962971676, -0.05163055820735335, -0.05134432750333632, -0.05068792663117536, -0.051135709683991504, -0.05065589783603277, -0.05224731507404452, -0.050941607654711354, -0.05105380728764557, -0.05285167402046835, -0.05062597659644976, -0.050637774172981084, -0.05138579257496525, -0.05087537373032959, -0.050501113282955666, -0.050889571816727666, -0.05168000351980613, -0.0520187817676225, -0.052229395761362804, -0.05122317719597017, -0.05026743841640181, -0.05062314689766909, -0.051209814028785314, -0.05053868077759292, -0.05084318854501106, -0.050705562826702834, -0.05068036643845921, -0.05177438536688117, -0.05135639830263876, -0.05095059806161854, -0.05066013767589671, -0.05053423429441164, -0.05118323468650417, -0.05096844069632567, -0.05122036695929503, -0.050751056952905396, -0.05195001647723133, -0.05276144448518709, -0.05018670252996917, -0.0512122973579522, -0.05046710274496378, -0.0499591097894066, -0.05034053035108638, -0.049957004350563286, -0.05029154071458317, -0.050610358285507084, -0.05205248424748733, -0.05091755183124315, -0.05079478164087155, -0.05101663442946145, -0.05061480427302032, -0.05121491807983223, -0.05029650533102248, -0.05035853447275831, -0.05061476656096183, -0.050733311962162644, -0.05106041639480783, -0.05190101866699467, -0.05036172573786278, -0.05179311170065896, -0.05058702142340739, -0.05107201978450347, -0.051084460104366604, -0.0498863706738388, -0.051361597997429696, -0.0495669162019708, -0.04976084980766776, -0.04970689121935988, -0.049856710813725794, -0.049628057551081094, -0.04987580452329371, -0.049384323798472504, -0.050374906339477624, -0.050093456625266375, -0.0500917141298734, -0.05080510139893406, -0.050387912185540604, -0.05109333955554084, -0.050696060594651926, -0.0511942139255247, -0.05011251556692279, -0.049752209313043964, -0.05000949238499343, -0.05023064252657725, -0.04999901548982407, -0.049755172899073094, -0.04987702098093113, -0.050321634104150394, -0.05091291046881515], 'parameter_importance': {'raw_learning_rate': 0.32808693091917407, 'min_data_in_leaf': 0.3206945695384069, 'raw_lambda_l1': 0.1580318541691762, 'num_leaves': 0.0665229730083367, 'raw_bagging_fraction': 0.06592558014544853, 'raw_lambda_l2': 0.03548600297509098, 'raw_feature_fraction': 0.01845974993007203, 'max_depth': 0.0067923393142946716}}, 'hpo_best_params': {'num_leaves': 11, 'max_depth': 6, 'raw_lambda_l1': 0.08921504761655664, 'raw_lambda_l2': 0.19013618781637237, 'min_data_in_leaf': 31, 'raw_learning_rate': -0.0728406836083933, 'raw_feature_fraction': -2.0764272162485917, 'raw_bagging_fraction': -0.43285273256631757}}`
- **shap_report:** `None`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251102_172441.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_172441.json`

### filtering

- **filtered_sr_levels:** `158`
- **removed_weak_levels:** `2`
- **strength_threshold:** `0.4`

## Metrics Summary

```json
{
  "optimization": {
    "data_points": 105092,
    "optimization_time": 43.63427782058716,
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

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251102_172441.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251102_172441.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_172441.md)
