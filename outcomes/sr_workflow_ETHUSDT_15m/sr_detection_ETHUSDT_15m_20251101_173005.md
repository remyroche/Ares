# SR Workflow Summary Report

**Generated:** 2025-11-01 17:31:45
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 100.07 seconds
- **Steps Completed:** 4/4
- **Steps Failed:** 0/4
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-01 17:31:45.165609

## Steps Completed

✅ ml_model_training
✅ sr_parameter_optimization
✅ sr_detection
✅ sr_filtering

## Artifacts Created

### ml_training

- **training_data_path:** `data_cache/sr_ml_training/sr_quality_training_data.parquet`
- **model_path:** `models/sr_quality_model.lgb`
- **metrics:** `{'cv_scores': [{'fold': 0, 'train_samples': 74, 'val_samples': 72, 'train_rmse': 0.36370333894401113, 'val_rmse': 0.3449583837139582, 'train_r2': -0.8317411547091587, 'val_r2': -1.0418337217823193, 'train_mae': 0.3146734465696282, 'val_mae': 0.2956463059971887, 'num_boost_rounds': 1}, {'fold': 1, 'train_samples': 146, 'val_samples': 72, 'train_rmse': 0.30052674835273485, 'val_rmse': 0.32245729491931846, 'train_r2': -0.38218374102251684, 'val_r2': -0.6609091267908376, 'train_mae': 0.25372961776096575, 'val_mae': 0.2770060038648681, 'num_boost_rounds': 85}, {'fold': 2, 'train_samples': 218, 'val_samples': 72, 'train_rmse': 0.32723602684295616, 'val_rmse': 0.30711514672922874, 'train_r2': -0.659827676155428, 'val_r2': -0.8189420445257638, 'train_mae': 0.2808081830573632, 'val_mae': 0.2542807435948143, 'num_boost_rounds': 10}, {'fold': 3, 'train_samples': 290, 'val_samples': 72, 'train_rmse': 0.2872150540986009, 'val_rmse': 0.2832666156691814, 'train_r2': -0.34386983131201077, 'val_r2': -0.23804437510156462, 'train_mae': 0.24359170739299782, 'val_mae': 0.24608499811899356, 'num_boost_rounds': 52}, {'fold': 4, 'train_samples': 362, 'val_samples': 72, 'train_rmse': 0.2753500920615514, 'val_rmse': 0.2860327859045711, 'train_r2': -0.2203961860895598, 'val_r2': -0.3796152726410127, 'train_mae': 0.23196842968674133, 'val_mae': 0.2494744580924153, 'num_boost_rounds': 70}], 'best_fold': 3, 'avg_metrics': {'avg_val_rmse': 0.30876604538725155, 'avg_val_r2': -0.6278689081682995, 'avg_val_mae': 0.264498501933656, 'std_val_rmse': 0.023095814063751312, 'std_val_r2': 0.2907096491442299}, 'config': {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': 6, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_data_in_leaf': 34, 'learning_rate': 0.03, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'seed': 42, 'force_col_wise': True, 'raw_lambda_l1': 0.571360581907433, 'raw_lambda_l2': 0.3655857828141976, 'raw_learning_rate': 5.476284247416853, 'raw_feature_fraction': 4.660954426185938, 'raw_bagging_fraction': 3.3583670087412028}, 'hpo_results': {'best_params': {'num_leaves': 31, 'max_depth': 6, 'raw_lambda_l1': 0.571360581907433, 'raw_lambda_l2': 0.3655857828141976, 'min_data_in_leaf': 34, 'raw_learning_rate': 5.476284247416853, 'raw_feature_fraction': 4.660954426185938, 'raw_bagging_fraction': 3.3583670087412028}, 'best_score': -0.05295847246153342, 'n_trials': 100, 'optimization_curve': [-0.05295847246153342, -0.05583048276494409, -0.05357813048420831, -0.05530526144489415, -0.05454053294073173, -0.05416947097504758, -0.05399686836435093, -0.060616911610265584, -0.05126622808091061, -0.05362094827058104, -0.052964308167680656, -0.05329865612474192, -0.05331015126987322, -0.05421541122165737, -0.061173489272784334, -0.053433216860291066, -0.057586152266963495, -0.04904406285906546, -0.055961183067622794, -0.05515029575253047, -0.05417185227684039, -0.05006401880504332, -0.050455455167032835, -0.05108132082097636, -0.05004107357664148, -0.051543904392129664, -0.05030274307195229, -0.049584316129387754, -0.05081382365780266, -0.05743378657129567, -0.05172557506437279, -0.050037249501818676, -0.049280892562003464, -0.05077197838377269, -0.054195388312931095, -0.05056740176624721, -0.05156346590957457, -0.04883357807950961, -0.053380307805779556, -0.05439003992191825, -0.050624180964774335, -0.05354198083545807, -0.05377189470608731, -0.05055804082403158, -0.05040203515588141, -0.05481411441339299, -0.05298839119541, -0.050838311815447845, -0.0513223196106298, -0.04969640400645123, -0.051157699456840046, -0.04890656672422903, -0.053460587580428334, -0.04938146566019078, -0.05325669661124823, -0.05332820460503142, -0.05028047935109127, -0.053478138628715324, -0.05014473978492196, -0.05576981429898669, -0.04960194404427333, -0.05329502103337791, -0.049914702095928805, -0.055489633623752185, -0.05373206069258744, -0.04965656769141099, -0.05053005506128497, -0.056042177826049096, -0.053047594865144425, -0.05093171172888784, -0.04914884758325996, -0.04981762086902297, -0.049371494001772986, -0.05013984694989472, -0.05523903112919837, -0.050206006607584536, -0.050456154060577005, -0.04996922242417808, -0.049937758198257125, -0.049739823012201395, -0.0535129967656192, -0.05371288306748957, -0.0488877330723256, -0.053276529087783984, -0.05344695540669817, -0.053272213148256364, -0.053261173534398396, -0.0533002069469665, -0.053287173841505595, -0.05311183775383319, -0.053450002082963986, -0.05319241089084555, -0.05348654758628255, -0.05329223094140702, -0.053150588381129446, -0.05313061028859517, -0.05355020948908888, -0.05308979705846438, -0.04874759744618941, -0.05381797349946441], 'parameter_importance': {'raw_lambda_l1': 0.3676183232535317, 'min_data_in_leaf': 0.2998661307481971, 'raw_bagging_fraction': 0.11574321128540975, 'raw_lambda_l2': 0.08584117460196539, 'raw_feature_fraction': 0.05687508989125668, 'raw_learning_rate': 0.05046035131173359, 'num_leaves': 0.016037605997634936, 'max_depth': 0.007558112910270867}}, 'hpo_best_params': {'num_leaves': 31, 'max_depth': 6, 'raw_lambda_l1': 0.571360581907433, 'raw_lambda_l2': 0.3655857828141976, 'min_data_in_leaf': 34, 'raw_learning_rate': 5.476284247416853, 'raw_feature_fraction': 4.660954426185938, 'raw_bagging_fraction': 3.3583670087412028}, 'ranking_metrics': {'precision_at_k': 0.4, 'spearman_rho': -0.18293095959637956, 'spearman_p_value': 1.4542680560771202e-08, 'ndcg_at_k': 0.5433129070308852, 'r2_score': -2.5387335546256877, 'rmse': 0.488468477022725, 'k': 10, 'quality_threshold': 0.7, 'total_samples': 946}}`
- **shap_report:** `None`

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251101_173004.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_173005.json`

### filtering

- **filtered_sr_levels:** `158`
- **removed_weak_levels:** `2`
- **strength_threshold:** `0.4`

## Metrics Summary

```json
{
  "optimization": {
    "data_points": 105092,
    "optimization_time": 46.89857196807861,
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

- [ml_model_training](outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_ETHUSDT_15m_20251101_173005.md)
- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251101_173005.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_173005.md)
