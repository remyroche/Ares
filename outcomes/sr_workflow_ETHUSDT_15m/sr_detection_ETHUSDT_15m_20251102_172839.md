# SR Workflow Summary Report

**Generated:** 2025-11-02 17:30:04
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 85.21 seconds
- **Steps Completed:** 4/4
- **Steps Failed:** 0/4
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-02 17:30:04.610665

## Steps Completed

✅ ml_model_training
✅ sr_parameter_optimization
✅ sr_detection
✅ sr_filtering

## Artifacts Created

### ml_training

- **training_data_path:** `data_cache/sr_ml_training/sr_quality_training_data.parquet`
- **model_path:** `models/sr_quality_model.lgb`
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 76, 'val_samples': 71, 'train_rmse': 0.3088997032774735, 'val_rmse': 0.30906595391477953, 'train_r2': -0.4305494555711187, 'val_r2': -0.5871155567356396, 'train_mae': 0.29079772018439, 'val_mae': 0.2799621809394978, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 147, 'val_samples': 71, 'train_rmse': 0.2551242689753884, 'val_rmse': 0.2697627262505686, 'train_r2': -0.022762916730645344, 'val_r2': -0.12252499330802302, 'train_mae': 0.23498611036308475, 'val_mae': 0.24743681594040018, 'num_boost_rounds': 40}, {'fold': 2, 'train_samples': 218, 'val_samples': 71, 'train_rmse': 0.29516409399189936, 'val_rmse': 0.28560671560669953, 'train_r2': -0.3603196912742568, 'val_r2': -0.3667442980419109, 'train_mae': 0.27118635924255413, 'val_mae': 0.25988941176364505, 'num_boost_rounds': 2}, {'fold': 3, 'train_samples': 289, 'val_samples': 71, 'train_rmse': 0.2180689677100009, 'val_rmse': 0.23230593442285372, 'train_r2': 0.24499154128025347, 'val_r2': 0.08522461758885058, 'train_mae': 0.1894385378535504, 'val_mae': 0.20234375892320472, 'num_boost_rounds': 201}, {'fold': 4, 'train_samples': 360, 'val_samples': 71, 'train_rmse': 0.25134513903129424, 'val_rmse': 0.26936997669834595, 'train_r2': -0.009734109677995262, 'val_r2': -0.19455501138596398, 'train_mae': 0.2282600674277275, 'val_mae': 0.2431910780383604, 'num_boost_rounds': 40}], 'best_fold': 3, 'avg_metrics': {'avg_val_rmse': 0.27322226137864947, 'avg_val_r2': -0.23714304837653738, 'avg_val_mae': 0.2465646491210216, 'std_val_rmse': 0.025056333703338957, 'std_val_r2': 0.22722200246646876}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 25, 'max_depth': 4, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 42, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'raw_lambda_l1': 0.4702829280689116, 'raw_lambda_l2': 0.2018861634336565, 'raw_learning_rate': 3.1015616035044182, 'raw_feature_fraction': -4.894895916741076, 'raw_bagging_fraction': 4.097629960574553}, 'hpo_results': {'best_params': {'num_leaves': 25, 'max_depth': 4, 'raw_lambda_l1': 0.4702829280689116, 'raw_lambda_l2': 0.2018861634336565, 'min_data_in_leaf': 42, 'raw_learning_rate': 3.1015616035044182, 'raw_feature_fraction': -4.894895916741076, 'raw_bagging_fraction': 4.097629960574553}, 'best_score': -0.051799720959656036, 'n_trials': 100, 'optimization_curve': [-0.054397050052924556, -0.05530606687811783, -0.05448447231158786, -0.054393933085932857, -0.051799720959656036, -0.05399729211103159, -0.05950363562883477, -0.04589268929383277, -0.04475332480631758, -0.045135868595717304, -0.05322199102387519, -0.044491933287632304, -0.053401012199552844, -0.04315644235902084, -0.05652644702400732, -0.05340767485922517, -0.04489133358740102, -0.0538114298104682, -0.04461853148182073, -0.054017058169473164, -0.053849108100815925, -0.04528725777354595, -0.05384970340787974, -0.045112352362284315, -0.043809071449416204, -0.05616521453288005, -0.05327012499300378, -0.056880873855625326, -0.04413925228670368, -0.055416085125973735, -0.05406900306342527, -0.05343546063451373, -0.0530415661567004, -0.052397297951188256, -0.05234536791489867, -0.05311179918110568, -0.054622306377776104, -0.05191302932661009, -0.053899746857161625, -0.04337286471287033, -0.05100883369224684, -0.05316653121715494, -0.04360509487065997, -0.052797824677518826, -0.044539684537220306, -0.05408384260954221, -0.05068869524786406, -0.051134588895876285, -0.053611954322422065, -0.04280219074065931, -0.050971153995389346, -0.05372130472242216, -0.05116276867910471, -0.04299790642100176, -0.052057535866419785, -0.04393125025360167, -0.0433284472973078, -0.050912149793918414, -0.05063583620948347, -0.04522951733454458, -0.05331894909396762, -0.043709723667373135, -0.04388422400442323, -0.043454226441430245, -0.04555978857717913, -0.042586801527175075, -0.0526720901530602, -0.0557892346184472, -0.05401835539896991, -0.04392532222026257, -0.05282104208233897, -0.050571586403034566, -0.042794530614394644, -0.05239728758245669, -0.0434648561080322, -0.04343967550082976, -0.05241716786207361, -0.0426128107731873, -0.052478139455720475, -0.04290102541875856, -0.052503891429598645, -0.043356424023655575, -0.05253448834461623, -0.04276450899231222, -0.0526754297557234, -0.052554283132439115, -0.052718813861856416, -0.04299225199425089, -0.0523861555314898, -0.04306247213931451, -0.04309450115743833, -0.05031587610900569, -0.04306457891858039, -0.04259740128583295, -0.0424497040309019, -0.04281005703284555, -0.04384366617968177, -0.04301788728045385, -0.05065633233776001, -0.05275921991834172], 'parameter_importance': {'raw_bagging_fraction': 0.35718165153199, 'min_data_in_leaf': 0.2550947155657128, 'raw_lambda_l1': 0.13662742773783723, 'raw_feature_fraction': 0.079674711858801, 'num_leaves': 0.07251712373225445, 'raw_learning_rate': 0.05015851804771536, 'max_depth': 0.035288961434319664, 'raw_lambda_l2': 0.013456890091369514}}, 'hpo_best_params': {'num_leaves': 25, 'max_depth': 4, 'raw_lambda_l1': 0.4702829280689116, 'raw_lambda_l2': 0.2018861634336565, 'min_data_in_leaf': 42, 'raw_learning_rate': 3.1015616035044182, 'raw_feature_fraction': -4.894895916741076, 'raw_bagging_fraction': 4.097629960574553}}`
- **shap_report:** `None`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251102_172839.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_172839.json`

### filtering

- **filtered_sr_levels:** `158`
- **removed_weak_levels:** `2`
- **strength_threshold:** `0.4`

## Metrics Summary

```json
{
  "optimization": {
    "data_points": 105092,
    "optimization_time": 36.405720710754395,
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

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251102_172839.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251102_172839.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251102_172839.md)
