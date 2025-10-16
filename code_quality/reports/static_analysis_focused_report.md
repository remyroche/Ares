# Static Analysis Report: Missing Imports & Undefined Names

**Generated:** 2025-10-16 07:41:23

## Executive Summary

- **Total files analyzed:** 1530
- **Files with issues:** 1530 (100% of files have issues)
- **Total undefined names found:** 70,729

⚠️ **Critical Finding:** Every single Python file in the codebase has undefined name issues, indicating widespread import problems.

## Most Common Undefined Names

These are the most frequently occurring undefined names across the entire codebase:

1. **e** - appears in 14436 locations
2. **i** - appears in 6412 locations
3. **col** - appears in 3863 locations
4. **kwargs** - appears in 3621 locations
5. **f** - appears in 1524 locations
6. **key** - appears in 1060 locations
7. **args** - appears in 888 locations
8. **symbol** - appears in 881 locations
9. **x** - appears in 861 locations
10. **j** - appears in 715 locations
11. **value** - appears in 683 locations
12. **feature** - appears in 674 locations
13. **name** - appears in 598 locations
14. **r** - appears in 562 locations
15. **item** - appears in 550 locations
16. **k** - appears in 549 locations
17. **v** - appears in 548 locations
18. **t** - appears in 499 locations
19. **regime** - appears in 469 locations
20. **p** - appears in 402 locations


## Top 20 Files with Most Issues

1. **src/utils/ml_common/feature_selection.py** - 577 issues
2. **src/training/steps/pre_training/unified_data_driven_pipeline/consolidated_pipeline.py** - 529 issues
3. **src/training/steps/market_analysis/optimal_regime_clustering_backup/orchestrator.py** - 437 issues
4. **research/candle_based_features/advanced_candle_features.py** - 372 issues
5. **research/candle_ml_patterns/advanced_candle_features.py** - 372 issues
6. **src/training/steps/backtesting/final_parameters_optimization.py** - 337 issues
7. **exchanges/binance.py** - 337 issues
8. **src/trading/reporting/performance_reporter.py** - 304 issues
9. **src/monitoring/csv_export_manager.py** - 286 issues
10. **exchanges/okx.py** - 270 issues
11. **src/training/steps/market_analysis/clusters/iterative_optimization.py** - 269 issues
12. **src/training/steps/market_analysis/tas_regime/core/tas_regime_detector.py** - 267 issues
13. **src/training/steps/pre_training/sub_pipeline.py** - 262 issues
14. **src/feature_generation/utils/vectorbt_rolling_optimizer.py** - 252 issues
15. **src/training/steps/pre_training/multi_horizon_profit_labeler.py** - 252 issues
16. **src/utils/feature_output_validator.py** - 251 issues
17. **src/training/steps/model_training/tactician_ensemble_training.py** - 246 issues
18. **src/training/steps/pre_training/unified_data_driven_pipeline/consolidated_pipeline_runner.py** - 244 issues
19. **src/training/steps/model_training/__init__.py** - 244 issues
20. **src/training/steps/market_analysis/components/regime_models_training.py** - 239 issues


## Recommendations

### Immediate Actions Required

1. **Import Audit**: The fact that 100% of files have undefined name issues suggests a systematic problem with imports
2. **Dependency Management**: Review and fix missing dependencies
3. **Code Organization**: Many undefined names may be due to incorrect module paths

### Priority Files to Fix

Based on the analysis, focus on these high-impact files first:

1. **src/utils/ml_common/feature_selection.py** (577 issues)
2. **src/training/steps/pre_training/unified_data_driven_pipeline/consolidated_pipeline.py** (529 issues)
3. **src/training/steps/market_analysis/optimal_regime_clustering_backup/orchestrator.py** (437 issues)
4. **research/candle_based_features/advanced_candle_features.py** (372 issues)
5. **research/candle_ml_patterns/advanced_candle_features.py** (372 issues)
6. **src/training/steps/backtesting/final_parameters_optimization.py** (337 issues)
7. **exchanges/binance.py** (337 issues)
8. **src/trading/reporting/performance_reporter.py** (304 issues)
9. **src/monitoring/csv_export_manager.py** (286 issues)
10. **exchanges/okx.py** (270 issues)


### Common Issues to Address

1. **Missing imports** for commonly used libraries
2. **Incorrect import paths** for internal modules
3. **Unused imports** that should be removed
4. **Circular import dependencies**

## Detailed Analysis

The following sections provide detailed analysis for the most problematic files:

### 1. src/utils/ml_common/feature_selection.py

**Total Issues:** 577

**Top 10 Most Common Undefined Names in this file:**

- **e** (155 occurrences)
- **i** (144 occurrences)
- **f** (46 occurrences)
- **j** (32 occurrences)
- **idx** (14 occurrences)
- **s** (12 occurrences)
- **start_idx** (11 occurrences)
- **args** (11 occurrences)
- **kwargs** (10 occurrences)
- **combination** (7 occurrences)

**Sample Issues:**

- Line 6702, Column 27: 
- Line 9478, Column 22: 
- Line 9487, Column 30: 
- Line 2523, Column 22: 
- Line 2655, Column 22: 
- Line 2966, Column 28: 
- Line 3414, Column 41: 
- Line 4347, Column 16: 
- Line 5134, Column 30: 
- Line 5164, Column 32: 
- ... and 567 more issues

---

### 2. src/training/steps/pre_training/unified_data_driven_pipeline/consolidated_pipeline.py

**Total Issues:** 529

**Top 10 Most Common Undefined Names in this file:**

- **e** (234 occurrences)
- **col** (95 occurrences)
- **period** (29 occurrences)
- **i** (25 occurrences)
- **column** (12 occurrences)
- **f** (9 occurrences)
- **feat** (9 occurrences)
- **feature_name** (8 occurrences)
- **x** (7 occurrences)
- **symbol** (6 occurrences)

**Sample Issues:**

- Line 42, Column 76: 
- Line 85, Column 88: 
- Line 108, Column 85: 
- Line 315, Column 87: 
- Line 330, Column 86: 
- Line 348, Column 105: 
- Line 359, Column 85: 
- Line 376, Column 85: 
- Line 4789, Column 15: 
- Line 5764, Column 27: 
- ... and 519 more issues

---

### 3. src/training/steps/market_analysis/optimal_regime_clustering_backup/orchestrator.py

**Total Issues:** 437

**Top 10 Most Common Undefined Names in this file:**

- **e** (73 occurrences)
- **c** (50 occurrences)
- **i** (45 occurrences)
- **cluster** (37 occurrences)
- **r** (25 occurrences)
- **cluster_metrics** (24 occurrences)
- **v** (23 occurrences)
- **p** (21 occurrences)
- **s** (17 occurrences)
- **m** (16 occurrences)

**Sample Issues:**

- Line 3900, Column 102: 
- Line 3944, Column 102: 
- Line 3988, Column 102: 
- Line 4035, Column 94: 
- Line 57, Column 30: 
- Line 1811, Column 32: 
- Line 1812, Column 32: 
- Line 2055, Column 32: 
- Line 2056, Column 35: 
- Line 2057, Column 33: 
- ... and 427 more issues

---

### 4. research/candle_based_features/advanced_candle_features.py

**Total Issues:** 372

**Top 10 Most Common Undefined Names in this file:**

- **i** (363 occurrences)
- **tf_period** (4 occurrences)
- **pattern** (2 occurrences)
- **level** (2 occurrences)
- **col** (1 occurrences)

**Sample Issues:**

- Line 315, Column 16: 
- Line 324, Column 31: 
- Line 583, Column 28: 
- Line 596, Column 15: 
- Line 628, Column 36: 
- Line 656, Column 27: 
- Line 674, Column 23: 
- Line 695, Column 23: 
- Line 905, Column 27: 
- Line 908, Column 49: 
- ... and 362 more issues

---

### 5. research/candle_ml_patterns/advanced_candle_features.py

**Total Issues:** 372

**Top 10 Most Common Undefined Names in this file:**

- **i** (363 occurrences)
- **tf_period** (4 occurrences)
- **pattern** (2 occurrences)
- **level** (2 occurrences)
- **col** (1 occurrences)

**Sample Issues:**

- Line 315, Column 16: 
- Line 324, Column 31: 
- Line 583, Column 28: 
- Line 596, Column 15: 
- Line 628, Column 36: 
- Line 656, Column 27: 
- Line 674, Column 23: 
- Line 695, Column 23: 
- Line 905, Column 27: 
- Line 908, Column 49: 
- ... and 362 more issues

---

### 6. src/training/steps/backtesting/final_parameters_optimization.py

**Total Issues:** 337

**Top 10 Most Common Undefined Names in this file:**

- **param_config** (79 occurrences)
- **e** (72 occurrences)
- **param_name** (45 occurrences)
- **key** (19 occurrences)
- **i** (17 occurrences)
- **value** (15 occurrences)
- **param** (15 occurrences)
- **v** (8 occurrences)
- **symbol** (7 occurrences)
- **exchange** (7 occurrences)

**Sample Issues:**

- Line 4012, Column 25: 
- Line 4041, Column 15: 
- Line 4046, Column 62: 
- Line 4047, Column 46: 
- Line 4792, Column 65: 
- Line 4792, Column 73: 
- Line 4792, Column 83: 
- Line 4805, Column 68: 
- Line 4805, Column 76: 
- Line 4805, Column 86: 
- ... and 327 more issues

---

### 7. exchanges/binance.py

**Total Issues:** 337

**Top 10 Most Common Undefined Names in this file:**

- **response** (74 occurrences)
- **symbol** (65 occurrences)
- **item** (46 occurrences)
- **limit** (28 occurrences)
- **instrument** (17 occurrences)
- **e** (17 occurrences)
- **interval** (11 occurrences)
- **price** (10 occurrences)
- **order_id** (10 occurrences)
- **side** (8 occurrences)

**Sample Issues:**

- Line 670, Column 11: 
- Line 748, Column 11: 
- Line 479, Column 15: 
- Line 496, Column 15: 
- Line 515, Column 15: 
- Line 534, Column 15: 
- Line 536, Column 15: 
- Line 544, Column 24: 
- Line 566, Column 15: 
- Line 568, Column 15: 
- ... and 327 more issues

---

### 8. src/trading/reporting/performance_reporter.py

**Total Issues:** 304

**Top 10 Most Common Undefined Names in this file:**

- **t** (87 occurrences)
- **trades** (74 occurrences)
- **trade** (25 occurrences)
- **model_id** (17 occurrences)
- **e** (15 occurrences)
- **metrics** (12 occurrences)
- **report_name** (11 occurrences)
- **data** (11 occurrences)
- **value** (9 occurrences)
- **feature** (8 occurrences)

**Sample Issues:**

- Line 106, Column 27: 
- Line 141, Column 25: 
- Line 190, Column 25: 
- Line 195, Column 25: 
- Line 242, Column 25: 
- Line 294, Column 25: 
- Line 363, Column 25: 
- Line 414, Column 25: 
- Line 520, Column 25: 
- Line 837, Column 68: 
- ... and 294 more issues

---

### 9. src/monitoring/csv_export_manager.py

**Total Issues:** 286

**Top 10 Most Common Undefined Names in this file:**

- **decision** (59 occurrences)
- **summary** (35 occurrences)
- **model_decision** (17 occurrences)
- **perf** (17 occurrences)
- **trade_decisions** (14 occurrences)
- **metric** (13 occurrences)
- **e** (13 occurrences)
- **i** (13 occurrences)
- **Any** (12 occurrences)
- **indicator** (12 occurrences)

**Sample Issues:**

- Line 22, Column 1: 
- Line 34, Column 1: 
- Line 43, Column 23: 
- Line 207, Column 24: 
- Line 850, Column 57: 
- Line 871, Column 34: 
- Line 61, Column 29: 
- Line 62, Column 31: 
- Line 79, Column 15: 
- Line 96, Column 28: 
- ... and 276 more issues

---

### 10. exchanges/okx.py

**Total Issues:** 270

**Top 10 Most Common Undefined Names in this file:**

- **symbol** (67 occurrences)
- **response** (49 occurrences)
- **e** (36 occurrences)
- **order_id** (18 occurrences)
- **instrument** (17 occurrences)
- **position** (14 occurrences)
- **k** (12 occurrences)
- **limit** (10 occurrences)
- **order_type** (9 occurrences)
- **side** (7 occurrences)

**Sample Issues:**

- Line 882, Column 15: 
- Line 977, Column 35: 
- Line 1012, Column 12: 
- Line 1012, Column 20: 
- Line 1012, Column 26: 
- Line 1022, Column 66: 
- Line 1060, Column 51: 
- Line 502, Column 15: 
- Line 644, Column 16: 
- Line 644, Column 24: 
- ... and 260 more issues

---

## Remaining Files (1520 files)

| File | Issues |
|------|--------|
| src/training/steps/market_analysis/clusters/iterative_optimization.py | 269 |
| src/training/steps/market_analysis/tas_regime/core/tas_regime_detector.py | 267 |
| src/training/steps/pre_training/sub_pipeline.py | 262 |
| src/feature_generation/utils/vectorbt_rolling_optimizer.py | 252 |
| src/training/steps/pre_training/multi_horizon_profit_labeler.py | 252 |
| src/utils/feature_output_validator.py | 251 |
| src/training/steps/model_training/tactician_ensemble_training.py | 246 |
| src/training/steps/pre_training/unified_data_driven_pipeline/consolidated_pipeline_runner.py | 244 |
| src/training/steps/model_training/__init__.py | 244 |
| src/training/steps/market_analysis/components/regime_models_training.py | 239 |
| src/utils/ml_common/optimization/bayesian_tpe_optimizer.py | 238 |
| src/training/steps/model_training/analyst_models_training_refactored.py | 234 |
| src/training/steps/model_training/tactician_models_training_refactored.py | 232 |
| src/training/steps/model_training/tactician_pre_ml_orchestrator.py | 226 |
| exchanges/mexc.py | 222 |
| src/trading/reporting/daily_recorder.py | 216 |
| GUI/api_server.py | 216 |
| src/analyst/ml_confidence_predictor.py | 212 |
| src/trading/reporting/trade_analyzer.py | 209 |
| src/utils/ml_common/optimization/hpo_utils.py | 208 |
| src/feature_generation/categories/volume.py | 200 |
| src/feature_generation/utils/feature_generators.py | 198 |
| src/training/steps/pre_training/final_feature_selection_step.py | 195 |
| src/feature_generation/categories/trend.py | 192 |
| src/training/steps/market_analysis/shared_utils/balanced_feature_extractor.py | 187 |
| src/training/steps/market_analysis/clusters/nas_tas_clustering_refactored.py | 179 |
| src/utils/ml_common/models/model_factory.py | 179 |
| src/trading/reporting/dashboard_generator.py | 177 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/hybrid_orchestrator.py | 176 |
| src/utils/data/quality/data_quality.py | 176 |
| src/trading/monitoring/comprehensive_trade_monitor.py | 175 |
| src/training/steps/data_collection/data_consolidation_manager.py | 172 |
| src/training/model_interpretability/model_explainer.py | 171 |
| research/clusters/advanced_feature_engineering.py | 171 |
| src/launcher/ares_launcher.py | 170 |
| src/training/steps/data_collection/klines_downloading_processing.py | 170 |
| research/cluster_analysis/market_factor_analysis/factor_extraction.py | 170 |
| src/training/steps/data_collection/data_preparation/validate_and_fix_aggtrades_format.py | 168 |
| research/clusters/economic_metrics.py | 167 |
| src/training/steps/models_training/analyst_models_training.py | 166 |
| src/training/steps/market_analysis/multi_horizon_sub_pipeline_adapter.py | 163 |
| research/cluster_analysis/economic_relevance/trading_significance.py | 163 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_perfect_nas_regime_detector.py | 162 |
| src/training/steps/pre_training/unified_data_driven_pipeline/statistical_analysis/statistical_framework.py | 160 |
| src/utils/ml_common/optimization/pareto.py | 159 |
| src/utils/common_operations.py | 158 |
| src/utils/data/cli.py | 156 |
| research/cluster_analysis/market_factor_analysis/dimension_discovery.py | 155 |
| research/clusters/dimension_analyzer.py | 155 |
| src/feature_generation/utils/cross_timeframe_analysis_pipeline.py | 153 |
| src/training/steps/data_collection/data_preparation/enhanced_data_quality_manager.py | 152 |
| src/training/steps/data_collection/klines_downloading_processing_enhanced.py | 151 |
| research/clusters/comprehensive_feature_integration.py | 151 |
| src/feature_generation/utils/enhanced_sr_feature_extractor.py | 150 |
| src/monitoring/enhanced_ml_monitoring.py | 148 |
| src/research/crypto_analysis/automated_crypto_processor.py | 146 |
| research/crypto_analysis/automated_crypto_processor.py | 146 |
| src/training/steps/backtesting/abc_testing/results_visualization.py | 144 |
| src/training/steps/pre_training/unified_data_driven_pipeline/feature_selection/multi_objective_selector.py | 144 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_feature_selection.py | 142 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/feature_engineering_pipeline.py | 142 |
| src/utils/vectorbt_batch_processor.py | 142 |
| src/research/crypto_analysis/data_analyzer.py | 140 |
| research/crypto_analysis/data_analyzer.py | 140 |
| src/feature_generation/utils/unified_vectorization_manager.py | 139 |
| src/utils/ml_common/optimization/shared_utils/evolutionary_search.py | 138 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/hyperparameter_optimization.py | 135 |
| src/utils/matrix_operations/unified_operations.py | 135 |
| src/training/steps/market_analysis/clusters/step8_validation.py | 134 |
| src/trading/execution/exchange_interface.py | 133 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/enhanced_statistical_framework.py | 133 |
| src/training/model_interpretability/shap_analyzer.py | 133 |
| exchanges/base_exchange/base_exchange.py | 133 |
| src/utils/parquet_utils.py | 132 |
| research/clusters/automated_feature_engineering.py | 132 |
| exchanges/bingx_production.py | 132 |
| src/monitoring/trading_integration.py | 130 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_performance_monitor.py | 130 |
| src/utils/ml_common/validation/enhanced_overfitting_detection.py | 129 |
| src/monitoring/daily_summary_tracker.py | 128 |
| src/research/profit_labeling/contextual_feature_labeling.py | 128 |
| src/training/steps/model_training/analyst_ensemble_training.py | 128 |
| src/monitoring/gui/enhanced_dashboard.py | 127 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/search_strategies.py | 127 |
| src/training/utils/feature_selection/data_validation.py | 126 |
| exchanges/bingx_fixed.py | 126 |
| exchanges/shared/klines_downloading_processing.py | 126 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/enhanced_feature_generator.py | 125 |
| src/core/decorators/validate.py | 125 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/vectorbt_optimizer.py | 124 |
| src/training/steps/market_analysis/components/nas_tas_regime_discovery.py | 124 |
| src/explainability/explainability_orchestrator.py | 123 |
| src/training/steps/model_training/sub_pipeline.py | 123 |
| src/training/utils/feature_selection/partial_information_decompositor.py | 123 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/detailed_pipeline_reporter.py | 122 |
| src/utils/ml_common/ensembles/ensemble_manager.py | 122 |
| src/monitoring/gui/monitoring_dashboard.py | 121 |
| src/feature_generation/categories/oscillator.py | 120 |
| src/explainability/visualization_tools.py | 120 |
| src/nas_tas/data/data_processor.py | 120 |
| src/monitoring/enhanced_monitoring_orchestrator.py | 119 |
| src/training/steps/data_collection/data_preparation/comprehensive_gap_filler.py | 119 |
| src/training/steps/backtesting/real_parameters_optimization.py | 119 |
| src/training/steps/market_analysis/components/regime_ensemble_training.py | 119 |
| src/utils/ml_common/explainability/model_interpretability.py | 116 |
| src/training/steps/market_analysis/tas_regime/regime_analysis/unsupervised_regime_detection.py | 114 |
| src/training/steps/model_training/model_validation.py | 114 |
| src/utils/sr_clustering/parameter_optimization_engine.py | 114 |
| src/utils/ml_common/utils/lookahead_protection.py | 114 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/lightgbm_featuretools_generator.py | 113 |
| research/cluster_analysis/market_factor_analysis/feature_clustering.py | 113 |
| src/training/steps/data_collection/validators/pipeline_validators.py | 112 |
| src/training/steps/backtesting/real_reporting_engine.py | 112 |
| src/monitoring/ensemble_monitor.py | 111 |
| src/trading/integration/data_integration.py | 111 |
| src/feature_generation/utils/optimization/unified_optimizer.py | 111 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/htf_template_system.py | 111 |
| src/training/model_interpretability/interpretability_visualizer.py | 111 |
| src/analyst/unified_regime_classifier_sr_focused.py | 111 |
| src/training/steps/market_analysis/clusters/step10_comprehensive_reporting.py | 110 |
| src/utils/ml_common/confidence_metrics.py | 110 |
| exchanges/bingx.py | 110 |
| exchanges/phemex.py | 109 |
| src/feature_generation/utils/multi_timeframe_training_analysis.py | 107 |
| src/training/steps/data_collection/sub_pipeline.py | 107 |
| src/utils/model_performance_monitor.py | 107 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/data_validation.py | 106 |
| src/research/price_patterns/run_complete_pattern_discovery.py | 105 |
| src/training/steps/market_analysis/components/hybrid_nas_tas_regime_discovery.py | 105 |
| src/utils/matrix_operations/vectorbt_optimizations.py | 105 |
| research/price_patterns/run_complete_pattern_discovery.py | 105 |
| research/clusters/production_feature_integration.py | 104 |
| src/supervisor/dynamic_weighter.py | 102 |
| src/feature_engineering_roadmap/transforms.py | 102 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/enhanced_clustering_improvements.py | 102 |
| src/research/price_patterns/run_pure_pattern_discovery.py | 101 |
| src/explainability/sr_explainer.py | 101 |
| src/tactician/ml_tactics_manager.py | 101 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/evolutionary_algorithms.py | 101 |
| src/training/steps/model_training/tactician_training_step.py | 101 |
| research/price_patterns/run_pure_pattern_discovery.py | 101 |
| research/vectorbt_optimizations/feature_comparison_optimizer.py | 101 |
| src/monitoring/shap_lime_integration.py | 100 |
| src/utils/ml_common/validation/enhanced_overfitting_detection_with_learning_curves.py | 100 |
| src/training/steps/data_collection/data_preparation_components/data_integrity_checker.py | 99 |
| src/training/steps/data_collection/utils/data_operations_utils.py | 99 |
| src/utils/ml_common/models/enhanced_model_trainer.py | 99 |
| src/utils/ml_common/training/vectorized_training_manager.py | 99 |
| data_quality/mapping/data_flow.py | 99 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_search_algorithms.py | 98 |
| research/profit_labeling/contextual_feature_labeling.py | 98 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/model_validation.py | 97 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_multi_objective_optimizer.py | 97 |
| src/training/steps/market_analysis/clusters/features/analyzer.py | 97 |
| src/training/utils/feature_selection/main_framework.py | 97 |
| exchanges/shared/high_level_wrappers_typed.py | 97 |
| src/monitoring/explainability_integration.py | 96 |
| src/feature_generation/core/feature_bank.py | 96 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/economic_clustering.py | 96 |
| src/training/model_interpretability/interpretability_reporter.py | 96 |
| src/utils/data/quality/data_cleaning.py | 96 |
| src/analyst/meta_labeling_system.py | 96 |
| src/trading/signal_generation/signal_pipeline.py | 95 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/shared_validation.py | 95 |
| src/training/steps/models_training/analyst_ensemble_training.py | 94 |
| src/training/utils/feature_selection/selection_methods.py | 94 |
| src/utils/sr_clustering/weight_optimization_engine.py | 94 |
| src/research/price_patterns/pattern_discovery_framework.py | 93 |
| src/training/steps/market_analysis/optimized_process_engines.py | 93 |
| src/utils/ml_common/optimization/hierarchical_hpo.py | 93 |
| research/price_patterns/pattern_discovery_framework.py | 93 |
| src/explainability/integration_decorators.py | 91 |
| src/training/steps/market_analysis/clusters/step9_results_consolidation.py | 91 |
| src/utils/data/feature_engineer.py | 91 |
| research/candle_based_features/interpretability_analysis.py | 91 |
| research/candle_ml_patterns/interpretability_analysis.py | 91 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/metrics_evolution_report.py | 90 |
| live_trading/config_manager.py | 90 |
| src/monitoring/trade_decision_capture.py | 89 |
| src/feature_generation/utils/step06_enhanced_feature_engineering_step.py | 89 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/feature_collection.py | 89 |
| src/training/model_interpretability/lime_analyzer.py | 89 |
| src/feature_generation/utils/step06_enhanced_feature_engineering.py | 88 |
| src/training/steps/data_collection/data_preparation_components/quality_metrics_calculator.py | 88 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/randomforest_feature_generator.py | 88 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/template_interaction_generator.py | 88 |
| src/feature_selection/advanced/enhanced_multi_stage_rfe.py | 87 |
| src/training/steps/market_analysis/sub_pipeline.py | 87 |
| src/utils/ml_common/evaluation/unified_evaluator.py | 87 |
| src/utils/ml_common/validation/cv.py | 87 |
| src/feature_generation/categories/returns.py | 86 |
| src/training/steps/models_training/tactician_ensemble_training.py | 86 |
| src/training/steps/market_analysis/tas_regime/components/micro_regime_detector.py | 86 |
| src/training/utils/feature_selection/temporal_analysis.py | 86 |
| research/clusters/visualization.py | 86 |
| exchanges/gateio.py | 86 |
| src/training/steps/data_collection/data_preparation/missing_data_downloader_and_gap_filler.py | 85 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_lookback_optimizer.py | 85 |
| src/utils/ml_common/unified_vectorization_manager.py | 85 |
| src/utils/ml_common/utils/memory_integration.py | 85 |
| src/utils/data/klines_parquet.py | 85 |
| src/utils/config/security.py | 85 |
| src/database/precomputed_features_manager.py | 85 |
| src/trading/execution/live_trader.py | 84 |
| src/training/steps/data_collection/decorators/step_decorators.py | 84 |
| src/training/steps/backtesting/sub_pipeline.py | 84 |
| src/training/steps/market_analysis/coverage_constrained_clustering/clusterer.py | 84 |
| src/training/utils/feature_selection/stability_analysis.py | 84 |
| src/utils/validation/unified_framework.py | 84 |
| src/utils/hardware/m1_gpu_utils.py | 84 |
| src/analyst/enhanced_regime_predictor.py | 84 |
| src/analyst/data_utils.py | 84 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/common_feature_logic.py | 83 |
| src/training/steps/market_analysis/tas_regime/search/multi_objective_search.py | 83 |
| src/core/decorators/auth.py | 83 |
| src/trading/regime/regime_analyzer.py | 82 |
| src/feature_generation/core/feature_generator.py | 82 |
| src/feature_generation/categories/momentum.py | 81 |
| src/research/profit_labeling/dynamic_target_optimizer.py | 81 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/performance_estimators.py | 81 |
| src/training/utils/feature_selection partial_information_decomposition.py | 81 |
| src/utils/ml_common/models/multi_output_models.py | 81 |
| src/utils/matrix_operations/vectorized_core.py | 81 |
| research/profit_labeling/dynamic_target_optimizer.py | 81 |
| src/feature_generation/matrix_integration/matrix_processor.py | 80 |
| src/training/steps/data_collection/data_preparation/step02_5_financial_logging.py | 80 |
| live_trading/data_streamer.py | 80 |
| src/monitoring/monitoring_orchestrator.py | 79 |
| src/tactician/async_order_executor.py | 79 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/intelligent_feature_selector.py | 79 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/clustering.py | 79 |
| src/training/steps/market_analysis/model_persistence_components/model_persistence_step.py | 79 |
| data_quality/unified_quality_orchestrator.py | 79 |
| src/trading/monitoring/performance_tracker.py | 78 |
| src/trading/execution/paper_trader.py | 78 |
| src/feature_generation/core/vectorbt_feature_generator.py | 77 |
| src/research/price_patterns/pure_price_action_patterns.py | 77 |
| src/training/steps/data_collection/enhanced_append_data_downloader.py | 77 |
| src/training/steps/backtesting/nas_tas_deprecated/walk_forward_analyzer.py | 77 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_search_strategies.py | 77 |
| src/features_common/transforms/vectorbt_scaler.py | 77 |
| src/utils/sr_clustering/predictive_sr_engine.py | 77 |
| src/analyst/unified_regime_classifier_fractal_simplified.py | 77 |
| research/price_patterns/pure_price_action_patterns.py | 77 |
| research/cluster_analysis/price_patterns/pure_price_patterns.py | 77 |
| src/research/mixed_factor_analysis/ml_pattern_discovery.py | 76 |
| src/training/steps/data_collection/enhanced_api_agnostic_data_collector.py | 76 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_clustering_algorithms.py | 76 |
| src/analyst/location_classifier_improvements.py | 76 |
| research/mixed_factor_analysis/ml_pattern_discovery.py | 76 |
| research/cluster_analysis/price_patterns/ml_discovery/anomaly_discovery.py | 76 |
| src/interfaces/enhanced_event_bus.py | 75 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/math_validation_integration.py | 75 |
| src/utils/ml_common/validation/validation_utils.py | 75 |
| data_quality/mapping/dead_code.py | 75 |
| src/feature_engineering_roadmap/disagreement_meta_features.py | 74 |
| src/training/steps/market_analysis/regime_handler.py | 74 |
| src/utils/sr_clustering/trading_ml_integration.py | 74 |
| src/utils/ml_common/utils/base_safeguards.py | 74 |
| src/utils/data/gap_detector.py | 74 |
| src/nas_tas/unified_pipeline.py | 74 |
| src/research/profit_labeling/parameter_optimizer.py | 73 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/coherent_regime_modeling.py | 73 |
| research/profit_labeling/parameter_optimizer.py | 73 |
| exchanges/order_router.py | 73 |
| exchanges/trading_receiver.py | 73 |
| src/research/profit_labeling/ensemble_labeling_system.py | 72 |
| src/research/mixed_factor_analysis/economic_relevance_research_framework.py | 72 |
| src/tactician/tactics_orchestrator.py | 72 |
| src/training/steps/main_training_pipeline.py | 72 |
| src/training/steps/market_analysis/tas_regime/adaptation/dynamic_optimization.py | 72 |
| src/training/steps/market_analysis/clusters/step1_feature_preparation.py | 72 |
| src/utils/async_utils.py | 72 |
| research/profit_labeling/ensemble_labeling_system.py | 72 |
| research/mixed_factor_analysis/economic_relevance_research_framework.py | 72 |
| research/cluster_analysis/economic_relevance/causal_analysis.py | 72 |
| exchanges/shared/high_level_wrappers.py | 72 |
| src/explainability/tactician_explainer.py | 71 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/common_lookback_optimizer.py | 71 |
| src/core/domain/decorators_extended.py | 71 |
| research/clusters/trading_calibration.py | 71 |
| src/supervisor/enhanced_prediction_service.py | 70 |
| src/training/steps/market_analysis/shared_utils/features.py | 70 |
| src/utils/report_manager.py | 70 |
| src/utils/hardware/adaptive_optimization_engine.py | 70 |
| src/analyst/unified_regime_classifier_fractal_enhanced.py | 70 |
| src/research/price_patterns/ml_pure_price_pattern_discovery.py | 69 |
| src/training/steps/data_collection/data_quality_components/anomaly_detector.py | 69 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_data_loading.py | 69 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/multi_objective_optimizer.py | 69 |
| src/analyst/candlestick_pattern_analyzer.py | 69 |
| live_trading/trading_engine.py | 69 |
| research/price_patterns/ml_pure_price_pattern_discovery.py | 69 |
| research/cluster_analysis/price_patterns/ml_discovery/clustering_discovery.py | 69 |
| research/vectorbt_optimizations/price_patterns_optimizer.py | 69 |
| src/research/crypto_analysis/optimized_crypto_processor.py | 68 |
| src/training/steps/market_analysis/model_persistence_components/metadata_tracker.py | 68 |
| research/crypto_analysis/optimized_crypto_processor.py | 68 |
| src/trading/integration/training_integration.py | 67 |
| src/supervisor/performance_reporter.py | 67 |
| src/feature_generation/categories/vectorbt_acceleration.py | 67 |
| src/feature_selection/advanced/native_validation.py | 67 |
| src/utils/enhanced_mlflow_integration.py | 67 |
| research/cluster_analysis/economic_relevance/market_state_relevance.py | 67 |
| research/clusters/dimension_economic_relevance.py | 67 |
| data_quality/mapping/call_graph.py | 67 |
| src/trading/utils/helpers.py | 66 |
| src/research/price_patterns/matrix_profile_discovery.py | 66 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/enhanced_optimized_clustering.py | 66 |
| research/candle_based_features/ml_neural_indicators.py | 66 |
| research/candle_ml_patterns/ml_neural_indicators.py | 66 |
| research/feature_comparison/feature_acceleration_dilation_enhanced.py | 66 |
| research/price_patterns/matrix_profile_discovery.py | 66 |
| research/cluster_analysis/price_patterns/ml_discovery/matrix_profile_discovery.py | 66 |
| research/cluster_analysis/clustering/validation_metrics.py | 66 |
| research/clusters/validation_metrics.py | 66 |
| src/launcher/validation_utilities.py | 65 |
| src/training/steps/pre_training/tactician_entry_labeler.py | 65 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/comprehensive_validator.py | 65 |
| src/utils/nas_tas/core/tas_engine.py | 65 |
| research/feature_comparison/multi_target_system.py | 65 |
| research/vectorbt_optimizations/clustering_optimizer.py | 65 |
| exchanges/base_exchange/response_handler.py | 65 |
| src/supervisor/main.py | 64 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/enhanced_hybrid_orchestrator.py | 64 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/hybrid_regime_detector.py | 64 |
| src/training/steps/market_analysis/clusters/feature_service.py | 64 |
| src/utils/tprint.py | 64 |
| src/utils/hardware/m1_cpu_optimizer.py | 64 |
| research/vectorbt_optimizations/profit_labeling_optimizer.py | 64 |
| src/training/steps/backtesting/abc_testing/statistical_analysis.py | 63 |
| src/utils/data/historical_data_pipeline.py | 63 |
| src/analyst/feature_engineering_utils.py | 63 |
| research/feature_comparison/standardized_features.py | 63 |
| research/cluster_analysis/clustering/regime_discovery.py | 63 |
| research/clusters/regime_clusterer.py | 63 |
| src/trading/monitoring/regime_monitor.py | 62 |
| src/feature_generation/categories/volatility.py | 62 |
| src/feature_generation/utils/fractional_differentiation_pipeline.py | 62 |
| src/feature_generation/utils/enhanced_matrix_accelerator.py | 62 |
| src/research/profit_labeling/backtesting_integrated_validator.py | 62 |
| src/research/profit_labeling/advanced_statistical_validator.py | 62 |
| src/research/price_patterns/advanced_pattern_definitions.py | 62 |
| src/explainability/hmm_explainer.py | 62 |
| src/training/steps/pre_training/analyst_profit_labeler.py | 62 |
| src/utils/ml_common/data_processing/data_labeling.py | 62 |
| src/utils/common_ml/backtesting/model_saver.py | 62 |
| research/candle_based_features/ml_candle_pattern_indicators.py | 62 |
| research/candle_based_features/enhanced_consensus_system.py | 62 |
| research/candle_ml_patterns/ml_candle_pattern_indicators.py | 62 |
| research/candle_ml_patterns/enhanced_consensus_system.py | 62 |
| research/profit_labeling/advanced_statistical_validator.py | 62 |
| research/price_patterns/advanced_pattern_definitions.py | 62 |
| research/clusters/feature_importance.py | 62 |
| GUI/api_server_simple.py | 62 |
| src/feature_generation/categories/microstructure_features.py | 61 |
| src/training/steps/data_collection/data_preparation_components/data_cleaner.py | 61 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/utils.py | 61 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/feature_engine_integration.py | 61 |
| src/training/steps/market_analysis/regime_analysis/label_fusion.py | 61 |
| src/utils/validator_orchestrator.py | 61 |
| src/utils/ml_common/optimization/shared_utils/feature_engineering.py | 61 |
| src/utils/ml_common/validation/data_leakage_detector.py | 61 |
| research/feature_comparison/enhanced_comparison_runner.py | 61 |
| src/research/mixed_factor_analysis/microstructure_impact_research.py | 60 |
| src/tactician/position_sizer.py | 60 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/enhanced_schema_validation.py | 60 |
| src/training/steps/market_analysis/regime_data_splitting/nas_tas_regime_data_splitting.py | 60 |
| src/utils/ml_common/validation/stability.py | 60 |
| research/mixed_factor_analysis/microstructure_impact_research.py | 60 |
| src/research/profit_labeling/bonus_penalty_optimizer.py | 59 |
| src/research/profit_labeling/labeling_visualizer.py | 59 |
| src/research/mixed_factor_analysis/volatility_impact_research.py | 59 |
| src/training/steps/data_collection/data_preparation/data_quality_dashboard.py | 59 |
| src/training/steps/backtesting/abc_testing/multi_model_orchestrator.py | 59 |
| src/training/steps/backtesting/abc_testing/performance_monitoring.py | 59 |
| src/training/steps/models_training/corrected_ml_entry_timing_labeler.py | 59 |
| src/training/steps/market_analysis/clusters/data_validator.py | 59 |
| src/training/steps/market_analysis/clusters/optimizer.py | 59 |
| src/utils/model_manager.py | 59 |
| src/utils/ml_common/training/enhanced_training_utils.py | 59 |
| src/nas_tas/error_handling.py | 59 |
| src/analyst/predictive_ensembles/ensemble_orchestrator.py | 59 |
| src/database/sqlite_manager.py | 59 |
| research/profit_labeling/labeling_visualizer.py | 59 |
| research/feature_comparison/analyst_labeler_integration.py | 59 |
| research/mixed_factor_analysis/volatility_impact_research.py | 59 |
| src/feature_selection/advanced/validation_framework.py | 58 |
| src/training/steps/backtesting/real_monte_carlo_engine.py | 58 |
| src/training/steps/market_analysis/tas_regime/search/evolutionary_search.py | 58 |
| src/utils/decorator_registry.py | 58 |
| src/nas_tas/training/training_orchestrator.py | 58 |
| exchanges/shared/unified_exchange_interface.py | 58 |
| src/trading/utils/validation.py | 57 |
| src/feature_generation/core/vectorbt_optimization_mixin.py | 57 |
| src/feature_generation/utils/vectorization_optimizer.py | 57 |
| src/feature_generation/utils/step06_utility_container.py | 57 |
| src/feature_generation/utils/optimization_validator.py | 57 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_performance_monitoring.py | 57 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/enhanced_data_integration.py | 57 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_regime_analyzer.py | 57 |
| src/training/steps/market_analysis/components/component_factory.py | 57 |
| src/training/steps/market_analysis/nas_clustering/core/essential_nas_clusterer.py | 57 |
| src/training/steps/market_analysis/nas_regime/core/nas_search.py | 57 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_matrix_operations.py | 57 |
| src/core/config_service.py | 57 |
| src/analyst/enhanced_prediction_integrator.py | 57 |
| research/candle_based_features/model_comparison_pipeline.py | 57 |
| research/candle_ml_patterns/model_comparison_pipeline.py | 57 |
| research/profit_labeling/bonus_penalty_optimizer.py | 57 |
| research/feature_comparison/stability_metrics.py | 57 |
| research/clusters/lookahead_bias_prevention.py | 57 |
| exchanges/shared/high_level_wrappers_typed_part2.py | 57 |
| src/training/steps/backtesting/unified_config.py | 56 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_architecture_compression.py | 56 |
| src/utils/ml_common/ensembles/enhanced_oof_stacking_with_confidence.py | 56 |
| src/analyst/regime_expert_orchestrator.py | 56 |
| src/monitoring/regime_monitoring_dashboard.py | 55 |
| src/feature_generation/utils/data_driven_feature_selector.py | 55 |
| src/research/profit_labeling/labeling_validator.py | 55 |
| src/research/mixed_factor_analysis/pattern_ml_integration.py | 55 |
| src/feature_selection/advanced/enhanced_ensemble_selector.py | 55 |
| src/training/steps/standardized_parquet_handler.py | 55 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/financial_performance_metrics.py | 55 |
| src/training/steps/market_analysis/tas_regime/adaptation/performance_tracking.py | 55 |
| src/training/utils/feature_selection/performance_monitoring.py | 55 |
| src/training/simplified_architecture/enhanced_pipeline_orchestrator.py | 55 |
| src/utils/fallback_monitoring.py | 55 |
| src/utils/parallel_processing_optimizer.py | 55 |
| src/utils/data/processing/data_processing.py | 55 |
| src/utils/data/processing/transformers.py | 55 |
| research/profit_labeling/labeling_validator.py | 55 |
| research/mixed_factor_analysis/pattern_ml_integration.py | 55 |
| research/cluster_analysis/economic_relevance/pattern_dimension_analysis.py | 55 |
| src/monitoring/fractional_system_monitor.py | 54 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/modular_architecture.py | 54 |
| src/training/steps/market_analysis/sr_detection.py | 54 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_evaluation_framework.py | 54 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_architecture_config.py | 54 |
| src/utils/common_utilities.py | 54 |
| src/utils/ml_common/models/model_training.py | 54 |
| src/utils/ml_common/optimization/grid_utils.py | 54 |
| src/utils/data/historical_data_downloader.py | 54 |
| src/analyst/multi_timeframe_feature_engineering.py | 54 |
| src/monitoring/regime_performance_tracker.py | 53 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/config.py | 53 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/search_spaces.py | 53 |
| src/training/steps/market_analysis/clusters/performance_monitor.py | 53 |
| src/core/decorators.py | 53 |
| src/utils/ml_common/pipeline_orchestrator.py | 53 |
| src/utils/ml_common/ensembles/vectorbt_ensemble_optimizer.py | 53 |
| src/utils/matrix_operations/error_handling.py | 53 |
| src/feature_generation/categories/autoencoder.py | 52 |
| src/tactician/enhanced_scenario_based_predictor.py | 52 |
| src/feature_selection/vectorbt/vectorbt_feature_selector.py | 52 |
| src/training/steps/backtesting/abc_testing/configuration_management.py | 52 |
| src/training/steps/market_analysis/enhanced_validation_framework.py | 52 |
| src/training/steps/market_analysis/tas_regime/backtesting/risk_analysis.py | 52 |
| src/training/steps/market_analysis/model_persistence_components/model_registry.py | 52 |
| src/training/steps/market_analysis/clusters/optimization_service.py | 52 |
| src/features_common/transforms/categorical_encoding.py | 52 |
| src/utils/ml_common/optimization/enhanced_hpo_monitor.py | 52 |
| src/utils/ml_common/utils/data_quality.py | 52 |
| src/database/efficient_features_database.py | 52 |
| research/clusters/constraints.py | 52 |
| exchanges/shared/monitoring_dashboard.py | 52 |
| src/trading/execution/order_manager.py | 51 |
| src/feature_selection/dimensionality/vif_module.py | 51 |
| src/training/steps/market_analysis/clusters/metrics.py | 51 |
| src/core/decorators/logging.py | 51 |
| src/utils/decorators.py | 51 |
| src/utils/regime_aware_financial_logging_decorator.py | 51 |
| src/utils/ml_common/post_training/model_persistence.py | 51 |
| src/utils/ml_common/utils/memory_optimization.py | 51 |
| src/utils/ml_common/validation/model_complexity_analysis.py | 51 |
| src/utils/nas_tas/core/nas_engine.py | 51 |
| research/feature_comparison/compute_aware_optimizer.py | 51 |
| research/clusters/ml_integration_framework.py | 51 |
| exchanges/base_exchange/message_handler.py | 51 |
| src/feature_generation/utils/vectorbt_memory_optimizer.py | 50 |
| src/research/profit_labeling/real_time_monitor.py | 50 |
| src/research/profit_labeling/adaptive_labeling_strategy.py | 50 |
| src/feature_selection/vectorbt/vectorbt_mrmr_selector.py | 50 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/architecture_encoders.py | 50 |
| src/training/steps/market_analysis/model_persistence_components/model_serializer.py | 50 |
| src/training/utils/feature_selection/causal_analysis.py | 50 |
| src/features_common/transforms/scaling_normalization.py | 50 |
| src/utils/ml_common/reporting/enhanced_reporting_system.py | 50 |
| src/utils/ml_common/explainability/model_explanations.py | 50 |
| research/profit_labeling/real_time_monitor.py | 50 |
| research/profit_labeling/adaptive_labeling_strategy.py | 50 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_economic_evaluator.py | 49 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/shared_metrics.py | 49 |
| src/training/steps/market_analysis/tas_regime/adaptation/real_time_adaptation.py | 49 |
| src/core/decorators/retry_timeout.py | 49 |
| src/utils/comprehensive_function_logger.py | 49 |
| src/utils/ml_common/optimization/regime_specific_tpsl_optimizer.py | 49 |
| src/utils/ml_common/data_processing/sr_feature_integration.py | 49 |
| src/analyst/predictive_ensembles.py | 49 |
| research/feature_comparison/comparison_report.py | 49 |
| research/clusters/integration_layer.py | 49 |
| data_quality/simple_quality_orchestrator.py | 49 |
| exchanges/shared/performance_monitor.py | 49 |
| src/trading/integration/exchange_integration.py | 48 |
| src/feature_generation/utils/optimized_cross_timeframe_analysis_integration.py | 48 |
| src/training/steps/data_collection/data_preparation/data_quality_monitor.py | 48 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_caching.py | 48 |
| src/training/steps/market_analysis/cluster_constraints.py | 48 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/enhanced_economic_evaluator.py | 48 |
| src/training/steps/market_analysis/tas_regime/search/rl_search.py | 48 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_ml_common_integration.py | 48 |
| src/utils/regime_probability_analyzer.py | 48 |
| src/utils/ml_common/post_training/model_validation.py | 48 |
| src/utils/ml_common/validation/data_leakage_prevention.py | 48 |
| src/nas_tas/results/result_manager.py | 48 |
| src/analyst/sr_relevance_optimizer.py | 48 |
| research/feature_comparison/relevance_analyzer.py | 48 |
| research/clusters/dimension_discovery_pipeline.py | 48 |
| src/monitoring/gui/data_visualization.py | 47 |
| src/feature_selection/advanced/multi_stage_rfe.py | 47 |
| src/feature_selection/advanced/enhanced_advanced_selector.py | 47 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/clustering_quality_metrics.py | 47 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/data_normalization.py | 47 |
| src/training/utils/debug_utilities.py | 47 |
| src/utils/function_call_monitor.py | 47 |
| src/utils/ml_common/ensembles/stacking_confidence_calibration.py | 47 |
| research/vectorbt_optimizations/crypto_analysis_optimizer.py | 47 |
| data_quality/mapping/cli.py | 47 |
| exchanges/exchange_registry.py | 47 |
| src/trading/signal_generation/analyst_signals.py | 46 |
| src/trading/execution/trading_orchestrator.py | 46 |
| src/feature_generation/categories/spectral_features.py | 46 |
| src/feature_generation/core/vectorbt_batch_processor.py | 46 |
| src/training/steps/market_analysis/components/tas_regime_discovery.py | 46 |
| src/utils/ml_common/training/per_regime_training_step.py | 46 |
| src/utils/ml_common/validation/unified_cv.py | 46 |
| src/utils/data/basic_returns_engineer.py | 46 |
| src/utils/matrix_operations/computation_toolbox.py | 46 |
| research/feature_comparison/diagnostics.py | 46 |
| research/clusters/ml_enhanced_discovery.py | 46 |
| src/trading/monitoring/trade_monitor.py | 45 |
| src/trading/data/data_validator.py | 45 |
| src/explainability/analyst_explainer.py | 45 |
| src/training/steps/data_collection/data_quality_components/validation_decorators.py | 45 |
| src/training/steps/data_collection/data_preparation_components/data_format_converter.py | 45 |
| src/training/utils/feature_selection/base_framework.py | 45 |
| src/utils/kline_parquet.py | 45 |
| src/utils/ml_common/training/ensemble_training_step.py | 45 |
| src/utils/matrix_operations/batch_operations.py | 45 |
| research/feature_comparison/pre_screening_pipeline.py | 45 |
| research/cluster_analysis/clustering/similarity_clustering.py | 45 |
| research/clusters/similarity_matrix_clustering.py | 45 |
| exchanges/data_aggregator.py | 45 |
| src/components/modular_supervisor.py | 44 |
| src/tactician/tactician.py | 44 |
| src/feature_selection/vectorbt/vectorbt_memory_optimizer.py | 44 |
| src/training/steps/data_collection/monitoring/pipeline_monitor.py | 44 |
| src/training/steps/backtesting/nas_tas_deprecated/performance_attribution.py | 44 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/advanced_search_strategies.py | 44 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/performance_estimator.py | 44 |
| src/training/steps/market_analysis/tas_regime/search/advanced_search.py | 44 |
| src/training/simplified_architecture/enhanced_config_system.py | 44 |
| src/core/sr_error_handlers.py | 44 |
| src/core/domain/decorators.py | 44 |
| src/core/decorators/cache.py | 44 |
| src/utils/ml_common/training/training_integration.py | 44 |
| src/nas_tas/evaluation/unified_evaluator.py | 44 |
| research/clusters/empirical_threshold_discovery.py | 44 |
| src/trading/signal_generation/tactician_signals.py | 43 |
| src/trading/integration/model_integration.py | 43 |
| src/supervisor/performance_monitor.py | 43 |
| src/custom_types/validation.py | 43 |
| src/tactician/sr_levels/sr_modules/sr_metrics_calculator.py | 43 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/regime_alignment_manager.py | 43 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/analysis_components.py | 43 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/architecture_encoder.py | 43 |
| src/training/steps/market_analysis/monitoring/performance_monitor.py | 43 |
| src/training/steps/market_analysis/tas_regime/utils/tree_utils.py | 43 |
| src/training/steps/market_analysis/regime_data_splitting/validator.py | 43 |
| src/core/decorators/enhanced_error_handling.py | 43 |
| src/utils/mlflow_utils.py | 43 |
| src/analyst/predictive_ensembles/multi_timeframe_ensemble.py | 43 |
| exchanges/shared/enhanced_unified_exchange_interface.py | 43 |
| src/models/stacker_lgbm_gate.py | 42 |
| src/trading/signal_generation/signal_combiner.py | 42 |
| src/feature_selection/advanced/prefiltering.py | 42 |
| src/training/common/artifact_persistence.py | 42 |
| src/training/steps/data_collection/data_preparation/gap_filler_pipeline.py | 42 |
| src/training/steps/data_collection/data_quality_components/validation_strategies.py | 42 |
| src/training/steps/market_analysis/monitoring/function_call_monitor.py | 42 |
| src/training/steps/market_analysis/clusters/weighted_category_pca.py | 42 |
| src/utils/ml_common/optimization/tree_architecture_search.py | 42 |
| src/utils/ml_common/optimization/specialized_trading_trees.py | 42 |
| live_trading/order_manager.py | 42 |
| live_trading/trading_orchestrator.py | 42 |
| research/clusters/enhanced_price_action_analysis.py | 42 |
| data_quality/generate_unified_report.py | 42 |
| exchanges/shared/market/instrument_manager.py | 42 |
| src/trading/regime/regime_classifier.py | 41 |
| src/feature_generation/utils/enhanced_data_driven_interaction_generator.py | 41 |
| src/research/profit_labeling/heuristic_analyzer.py | 41 |
| src/tactician/scenario_based_predictor.py | 41 |
| src/feature_selection/vectorbt/vectorbt_mutual_information.py | 41 |
| src/training/steps/models_training/analyst_training_pipeline.py | 41 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_validation.py | 41 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_search_space_evolution.py | 41 |
| src/training/steps/market_analysis/components/nas_regime_discovery.py | 41 |
| src/training/steps/market_analysis/clusters/clustering_utils.py | 41 |
| src/features_common/mixins/vectorbt_mixin.py | 41 |
| src/utils/ml_common/utils/enhanced_error_handling.py | 41 |
| src/utils/ml_common/validation/hpo_overfitting_prevention.py | 41 |
| research/profit_labeling/heuristic_analyzer.py | 41 |
| research/feature_comparison/feature_consolidation.py | 41 |
| exchanges/shared/auth/subaccount_manager.py | 41 |
| src/components/modular_analyst.py | 40 |
| src/trading/sizing/position_sizer.py | 40 |
| src/feature_generation/categories/advanced_statistical.py | 40 |
| src/feature_generation/categories/time.py | 40 |
| src/feature_generation/utils/optimization_config.py | 40 |
| src/training/steps/data_collection/data_preparation/step1_orchestrator.py | 40 |
| src/training/steps/data_collection/utils/common_operations.py | 40 |
| src/training/steps/models_training/analyst_pre_ml_orchestration.py | 40 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/financial_optimizers.py | 40 |
| src/training/steps/market_analysis/tas_regime/trading/trading_engine.py | 40 |
| src/training/steps/market_analysis/nas_clustering/core/evaluation/multi_objective.py | 40 |
| src/training/steps/market_analysis/nas_regime/core/hybrid_architecture.py | 40 |
| src/core/decorators/function_monitor.py | 40 |
| src/features_common/transforms/base_scaler.py | 40 |
| src/analyst/meta_label_relevance.py | 40 |
| research/feature_comparison/feature_scorecard.py | 40 |
| src/trading/config/regime_config.py | 39 |
| src/feature_generation/utils/vectorbt_performance_benchmark.py | 39 |
| src/feature_selection/vectorbt/vectorbt_correlation_filter.py | 39 |
| src/training/steps/data_collection/data_preparation_components/training_validation_config.py | 39 |
| src/training/steps/models_training/tactician_training_pipeline.py | 39 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/economic_evaluator.py | 39 |
| src/training/steps/market_analysis/components/artifact_manager.py | 39 |
| src/training/simplified_architecture/dependency_injection.py | 39 |
| src/features_common/mixins/optimization_mixin.py | 39 |
| src/utils/enhanced_data_quality_validator.py | 39 |
| src/utils/financial_metrics_logger.py | 39 |
| src/utils/ml_common/evaluation/enhanced_bootstrap_confidence_intervals.py | 39 |
| src/utils/data/quality/advanced_quality_metrics.py | 39 |
| GUI/launcher_integration.py | 39 |
| src/research/profit_labeling/ml_label_quality_assessor.py | 38 |
| src/training/steps/feature_engineering/price_action/close_location_value.py | 38 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/economic_evaluation.py | 38 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/regime_aware_search.py | 38 |
| src/utils/enhanced_financial_metrics_logger.py | 38 |
| src/utils/enhanced_step_optimizations.py | 38 |
| src/utils/validation.py | 38 |
| src/utils/ml_common/optimization/adaptive_regime_nas.py | 38 |
| src/utils/ml_common/optimization/neural_architecture_search.py | 38 |
| src/utils/ml_common/data_processing/feature_preparation.py | 38 |
| src/strategist/strategist.py | 38 |
| research/profit_labeling/ml_label_quality_assessor.py | 38 |
| research/feature_comparison/run_comparison.py | 38 |
| exchanges/shared/config_manager.py | 38 |
| src/components/modular_tactician.py | 37 |
| src/training/steps/data_collection/unified_gap_filler.py | 37 |
| src/training/steps/feature_engineering/trend/trend_coherence.py | 37 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_error_handling.py | 37 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_trading_viability_evaluator.py | 37 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/nas_financial_optimizer.py | 37 |
| src/training/steps/market_analysis/tas_regime/components/advanced_tree_models.py | 37 |
| src/training/steps/market_analysis/nas_regime/core/advanced_neural_architectures.py | 37 |
| src/utils/data_access_protection.py | 37 |
| src/utils/pipeline_standards.py | 37 |
| src/utils/ml_common/vectorbt_portfolio_optimization.py | 37 |
| src/utils/ml_common/post_training/model_evaluation.py | 37 |
| src/utils/data/monthly_data_downloader.py | 37 |
| src/utils/data/quality/data_qualification_imports.py | 37 |
| src/validation/regime_consensus_validator.py | 37 |
| live_trading/unified_trading_system.py | 37 |
| research/candle_based_features/ml_indicator_training_pipeline.py | 37 |
| research/candle_ml_patterns/ml_indicator_training_pipeline.py | 37 |
| research/clusters/adaptive_clustering.py | 37 |
| exchanges/shared/unified_ohlcv_standardizer.py | 37 |
| exchanges/shared/unified_exchange_standardizer.py | 37 |
| src/ci/validators.py | 36 |
| src/components/modular_strategist.py | 36 |
| src/trading/regime/regime_weights.py | 36 |
| src/research/profit_labeling/enhanced_multi_horizon_labeler.py | 36 |
| src/research/price_patterns/pattern_discovery_example.py | 36 |
| src/feature_engineering_roadmap/assembly_dag.py | 36 |
| src/tactician/comprehensive_enhanced_scenario_predictor.py | 36 |
| src/training/steps/backtesting/vectorbt_unified_manager.py | 36 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/modular_architecture.py | 36 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/shared_training.py | 36 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/financial_search_strategies.py | 36 |
| src/training/steps/market_analysis/optimized_multi_horizon_optimizer/grid_bayesian_optimizer.py | 36 |
| src/training/steps/market_analysis/tas_regime/search/bayesian_search.py | 36 |
| src/config/sr_config_loader.py | 36 |
| src/utils/matrix_operations.py | 36 |
| src/utils/ml_common/monitoring/enhanced_error_detector.py | 36 |
| src/utils/hmm/hardware_integration.py | 36 |
| src/utils/nas_tas/optimization/architecture_search.py | 36 |
| research/candle_based_features/consensus_indicator_system.py | 36 |
| research/candle_ml_patterns/consensus_indicator_system.py | 36 |
| research/profit_labeling/enhanced_multi_horizon_labeler.py | 36 |
| src/trading/data/market_data_provider.py | 35 |
| src/explainability/base_explainer.py | 35 |
| src/tactician/step17_optimized_tactician.py | 35 |
| src/feature_selection/analysis/feature_importance_analyzer.py | 35 |
| src/feature_selection/parallel/parallel_feature_selector.py | 35 |
| src/training/steps/backtesting/nas_tas_deprecated/validation_orchestrator.py | 35 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_artifact_management.py | 35 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/clustering_cross_validation.py | 35 |
| src/training/core/decorators.py | 35 |
| src/training/utils/feature_selection/quality_metrics.py | 35 |
| src/utils/enhanced_step_wrapper.py | 35 |
| src/utils/ml_common/validation/overfitting_monitoring.py | 35 |
| src/utils/data/quality/quality_alert_system.py | 35 |
| research/feature_comparison/enhanced_relevance_analyzer.py | 35 |
| exchanges/exchange_dispatcher.py | 35 |
| exchanges/shared/pricing/enhanced_ohlcv_manager.py | 35 |
| src/monitoring/surrogate_optimization_monitor.py | 34 |
| src/launcher/enhanced_trading_launcher.py | 34 |
| src/trading/monitoring/alert_manager.py | 34 |
| src/feature_generation/categories/entropy.py | 34 |
| src/feature_generation/utils/step06_labeling_components/optimized_triple_barrier_labeling_improved.py | 34 |
| src/research/price_patterns/core_patterns.py | 34 |
| src/feature_selection/vectorbt/vectorbt_unified_framework.py | 34 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/clustering_quality_analyzer.py | 34 |
| src/utils/ml_common/data_drift_detector.py | 34 |
| src/utils/matrix_operations/hardware_integration.py | 34 |
| src/utils/nas_tas/optimization/strategy_search.py | 34 |
| src/utils/hardware/m1_memory_optimizer.py | 34 |
| src/utils/common_ml/backtesting/analytics_reporter.py | 34 |
| research/feature_comparison/feature_acceleration_dilation.py | 34 |
| research/price_patterns/core_patterns.py | 34 |
| research/cluster_analysis/price_patterns/mathematical_definitions.py | 34 |
| research/clusters/refined_ml_discovery.py | 34 |
| src/feature_generation/utils/limited_microstructure_features.py | 33 |
| src/feature_engineering_roadmap/data_contracts.py | 33 |
| src/tactician/position_division_strategy.py | 33 |
| src/feature_selection/dimensionality/pca_module.py | 33 |
| src/feature_selection/advanced/improved_mrmr.py | 33 |
| src/feature_selection/chunked/chunked_processor.py | 33 |
| src/training/steps/data_collection/enhanced_data_validation_framework.py | 33 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/consensus_validator.py | 33 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_hardware_manager.py | 33 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_validation_system.py | 33 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/automatic_training/regime_training_pipeline.py | 33 |
| src/training/steps/market_analysis/model_persistence_components/version_manager.py | 33 |
| src/training/steps/market_analysis/clusters/hardware_service.py | 33 |
| src/training/steps/market_analysis/nas_regime/core/nas_shared_utils_integration.py | 33 |
| src/core/decorators/trace.py | 33 |
| src/utils/performance_utils.py | 33 |
| src/utils/ml_common/optimization/regime_hpo_wrapper.py | 33 |
| src/utils/data/optimized_parquet_storage.py | 33 |
| src/database/migration_utils.py | 33 |
| exchanges/shared/tests/verify_type_coverage.py | 33 |
| src/feature_generation/utils/enhanced_matrix_operations.py | 32 |
| src/feature_generation/utils/optimized_feature_pipeline.py | 32 |
| src/tactician/sr_levels/sr_modules/sr_feature_extractor.py | 32 |
| src/feature_selection/error_handling/enhanced_error_handler.py | 32 |
| src/feature_selection/advanced/advanced_selector.py | 32 |
| src/feature_selection/vectorbt/vectorbt_rfe_selector.py | 32 |
| src/training/steps/market_analysis/logging_standards.py | 32 |
| src/training/steps/market_analysis/shared_utils/metrics.py | 32 |
| src/training/steps/market_analysis/tas_regime/tree_cvlSA_demo.py | 32 |
| src/training/steps/market_analysis/tas_regime/backtesting/data_manager.py | 32 |
| src/features_common/vectorbt/unified_manager.py | 32 |
| src/utils/core/file_operations.py | 32 |
| src/utils/hardware/unified_hardware_manager.py | 32 |
| src/strategist/enhanced_regime_classifier.py | 32 |
| src/validation/walkforward_validation.py | 32 |
| live_trading/risk_manager.py | 32 |
| src/models/patch_gru.py | 31 |
| src/trading/execution/paper_trading_integration.py | 31 |
| src/feature_generation/utils/math_validation.py | 31 |
| src/research/profit_labeling/research_runner.py | 31 |
| src/research/price_patterns/lstm_discovery.py | 31 |
| src/integration/paper_trading_integration.py | 31 |
| src/nas_tas_integration/unified_regime_training_pipeline.py | 31 |
| src/training/steps/data_collection/data_quality_components/data_utils.py | 31 |
| src/training/steps/backtesting/abc_testing/risk_management.py | 31 |
| src/training/steps/feature_engineering/volatility/atr_volatility_ratio.py | 31 |
| src/training/steps/market_analysis/optimization_monitor.py | 31 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/data_caching.py | 31 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/advanced_search_strategies.py | 31 |
| src/training/steps/market_analysis/shared_utils/feature_filters.py | 31 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_nas_integration.py | 31 |
| src/training/steps/model_training/random_survival_forest_tactician.py | 31 |
| src/training/steps/model_training/bayesian_optimization_msm.py | 31 |
| src/training/simplified_architecture/config_driven_architecture.py | 31 |
| src/core/domain/__init__.py | 31 |
| src/utils/ml_common/utils/thread_guard.py | 31 |
| src/utils/ml_common/validation/model_enhancement_guide.py | 31 |
| live_trading/api_client.py | 31 |
| research/profit_labeling/research_runner.py | 31 |
| research/price_patterns/lstm_discovery.py | 31 |
| research/cluster_analysis/price_patterns/ml_discovery/lstm_discovery.py | 31 |
| research/clusters/metric_orthogonalization.py | 31 |
| src/models/enhanced_patchtst.py | 30 |
| src/models/vectorbt_enhanced_models.py | 30 |
| src/trading/execution/live_trading_scheduler.py | 30 |
| src/trading/sizing/risk_calculator.py | 30 |
| src/research/profit_labeling/bonus_penalty_integration_example.py | 30 |
| src/research/profit_labeling/example_usage.py | 30 |
| src/feature_selection/memory/memory_efficient_selector.py | 30 |
| src/training/steps/data_collection/data_preparation/data_gap_detector.py | 30 |
| src/training/steps/data_collection/data_preparation_components/aggtrades_data_formatting.py | 30 |
| src/training/steps/models_training/ml_based_entry_timing_labeler.py | 30 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/feature_bank_integration.py | 30 |
| src/training/steps/market_analysis/tas_regime/core/tas_engine.py | 30 |
| src/training/steps/market_analysis/clusters/clustering_service.py | 30 |
| src/features_common/mixins/caching_mixin.py | 30 |
| src/utils/error_handler.py | 30 |
| src/utils/unified_cache.py | 30 |
| src/utils/import_standardizer.py | 30 |
| research/feature_comparison/family_diverse_features.py | 30 |
| src/trading/regime/regime_detector.py | 29 |
| src/feature_generation/categories/candlestick_pattern.py | 29 |
| src/research/profit_labeling/enhanced_example_usage.py | 29 |
| src/research/price_patterns/gradient_targets.py | 29 |
| src/feature_selection/vectorbt/vectorbt_regularization.py | 29 |
| src/training/steps/data_collection/data_download_monitor.py | 29 |
| src/training/steps/feature_engineering/price_action/bar_efficiency_ratio.py | 29 |
| src/training/steps/pre_training/standardized_labeling_interface.py | 29 |
| src/training/steps/pre_training/artifacts/manifest.py | 29 |
| src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_data_validation_step.py | 29 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/economic_evaluator.py | 29 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/data_pipeline.py | 29 |
| src/training/steps/market_analysis/tas_regime/meta_learning/tree_meta_learning.py | 29 |
| src/training/steps/market_analysis/clustering/main_component.py | 29 |
| src/utils/trading_decorators.py | 29 |
| src/utils/ml_common/models/model_registry.py | 29 |
| src/utils/ml_common/validation/cv_utils.py | 29 |
| src/utils/ml_common/ensembles/oof_stacking_ensemble_manager.py | 29 |
| src/utils/hardware/advanced_cpu_optimizer.py | 29 |
| src/utils/hardware/m1_optimizations.py | 29 |
| src/analyst/predictive_ensembles/regime_ensembles/volatile_regime_ensemble.py | 29 |
| research/candle_based_features/ml_indicator_integration.py | 29 |
| research/candle_ml_patterns/ml_indicator_integration.py | 29 |
| research/price_patterns/gradient_targets.py | 29 |
| src/monitoring/fractional_performance_tracker.py | 28 |
| src/models/lgbm_gru_embedding.py | 28 |
| src/feature_generation/utils/step06_enhanced_validation_framework.py | 28 |
| src/feature_generation/utils/optimized_cross_timeframe_analysis.py | 28 |
| src/training/steps/data_collection/data_preparation/step02_data_reading.py | 28 |
| src/training/steps/models_training/negative_learning_training_patches.py | 28 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/enhanced_caching_integration.py | 28 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/enhanced_ml_integration.py | 28 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/regime_model_mapping/data_driven_model_selector.py | 28 |
| src/training/steps/market_analysis/regime_analysis/metrics.py | 28 |
| src/training/steps/market_analysis/tas_regime/backtesting/performance_attribution.py | 28 |
| src/training/steps/market_analysis/tas_regime/backtesting/monte_carlo.py | 28 |
| src/training/steps/market_analysis/tas_regime/evaluation/tas_evaluator.py | 28 |
| src/training/steps/market_analysis/regime_model_mapping/data_driven_model_selector.py | 28 |
| src/training/steps/market_analysis/clusters/memory_manager.py | 28 |
| src/training/steps/market_analysis/clusters/cv_enhancement_strategies.py | 28 |
| src/training/steps/market_analysis/clusters/step2_initial_clustering.py | 28 |
| src/features_common/mixins/validation_mixin.py | 28 |
| src/utils/decorators/errors.py | 28 |
| src/utils/ml_common/models/multiscale_nbeats.py | 28 |
| src/utils/ml_common/integration/enhanced_ml_pipeline_integration.py | 28 |
| src/utils/ml_common/training/universal_validation_integration.py | 28 |
| src/utils/ml_common/utils/logging_utils.py | 28 |
| src/utils/hardware/advanced_memory_optimizer.py | 28 |
| src/nas_tas/evaluation/performance_monitor.py | 28 |
| research/cluster_analysis/market_factor_analysis/statistical_analysis.py | 28 |
| research/clusters/statistical_dimension_analysis.py | 28 |
| src/monitoring/retrain_monitoring.py | 27 |
| src/feature_generation/test_tprint_logging.py | 27 |
| src/feature_generation/categories/representation_learning.py | 27 |
| src/feature_generation/core/factory.py | 27 |
| src/feature_generation/utils/centralized_logging.py | 27 |
| src/training/steps/data_collection/unified_data_loader.py | 27 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_ensemble_search_space.py | 27 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/enhanced_economic_clustering.py | 27 |
| src/training/steps/market_analysis/monitoring/error_handler.py | 27 |
| src/training/steps/market_analysis/components/nas_ensemble_training.py | 27 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/data_storage.py | 27 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/pipeline_orchestrator.py | 27 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_nas_modeling_integration.py | 27 |
| src/training/steps/market_analysis/nas_regime/core/perfect_nas_regime_detector.py | 27 |
| src/utils/enhanced_artifact_manager.py | 27 |
| src/utils/error_recovery/advanced_error_recovery.py | 27 |
| src/utils/data/quality/comprehensive_duplicate_analyzer.py | 27 |
| research/feature_comparison/time_series_validation.py | 27 |
| research/cluster_analysis/economic_relevance/__init__.py | 27 |
| src/monitoring/auto_monitoring_launcher.py | 26 |
| src/monitoring/trading_mode_monitoring_integration.py | 26 |
| src/common/config/loader.py | 26 |
| src/trading/data/live_data_collector.py | 26 |
| src/feature_generation/utils/vectorbt_optimization_integration.py | 26 |
| src/feature_selection/advanced/dynamic_selection.py | 26 |
| src/feature_selection/vectorbt/vectorbt_rolling_operations.py | 26 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/gpu_optimizations.py | 26 |
| src/training/steps/market_analysis/enhanced_validation.py | 26 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/position_aware_trading.py | 26 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/dynamic_search_space.py | 26 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/architecture_signal_generator.py | 26 |
| src/training/steps/market_analysis/shared_utils/feature_importance_pipeline_utils.py | 26 |
| src/training/steps/market_analysis/nas_clustering/core/micro_regime_detector.py | 26 |
| src/training/steps/market_analysis/nas_regime/validate_implementations.py | 26 |
| src/core/error_classes.py | 26 |
| src/utils/ml_common/integrated_analysis_pipeline.py | 26 |
| src/utils/ml_common/optimization/creative_tree_models.py | 26 |
| src/utils/ml_common/examples/universal_validation_demo.py | 26 |
| src/utils/ml_common/training/enhanced_early_stopping.py | 26 |
| src/utils/ml_common/ensembles/ensembling.py | 26 |
| src/utils/data/quality/data_qualification_error_handler.py | 26 |
| src/utils/matrix_operations/convenience.py | 26 |
| examples/partial_bar_nowcasting_demo.py | 26 |
| src/models/enhanced_tft.py | 25 |
| src/trading/execution/partial_bar_nowcasting.py | 25 |
| src/feature_generation/test_backward_compatibility.py | 25 |
| src/feature_generation/core/optimization_mixin.py | 25 |
| src/feature_generation/utils/unified_optimization_system.py | 25 |
| src/tactician/dynamic_barrier_calculator.py | 25 |
| src/tactician/ml_target_validator.py | 25 |
| src/feature_selection/advanced/confidence_scoring.py | 25 |
| src/training/steps/data_collection/data_collection_orchestrator.py | 25 |
| src/training/steps/data_collection/data_preparation/data_resampler.py | 25 |
| src/training/steps/market_analysis/regime_processing_decorator.py | 25 |
| src/training/steps/market_analysis/shared_utils/feature_importance_integration.py | 25 |
| src/training/steps/market_analysis/nas_clustering/core/nas_clusterer.py | 25 |
| src/training/steps/market_analysis/nas_modeling/core/advanced_preprocessing.py | 25 |
| src/training/steps/market_analysis/clusters/features/preprocessor.py | 25 |
| src/core/errors/base.py | 25 |
| src/utils/pipeline_enhancement_integration.py | 25 |
| src/utils/report_collector.py | 25 |
| src/utils/ml_common/training/quick_integration.py | 25 |
| src/utils/ml_common/training/base_training_step.py | 25 |
| src/utils/ml_common/validation/universal_temporal_validation.py | 25 |
| research/clusters/core_regime_discovery.py | 25 |
| exchanges/shared/monitoring_api.py | 25 |
| exchanges/shared/data_validation_suite.py | 25 |
| exchanges/shared/orders/order_manager.py | 25 |
| scripts/diagnose_regime_data_leakage.py | 24 |
| src/ares_pipeline.py | 24 |
| src/launcher/pipeline_managers.py | 24 |
| src/trading/sizing/leverage_manager.py | 24 |
| src/feature_generation/core/generator_factory.py | 24 |
| src/feature_generation/utils/optimization_metrics.py | 24 |
| src/feature_generation/utils/error_handling.py | 24 |
| src/feature_generation/utils/unified_optimization_wrapper.py | 24 |
| src/feature_selection/vectorbt/vectorbt_stability_selection.py | 24 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/constraint_systems.py | 24 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/advanced_clustering.py | 24 |
| src/training/steps/market_analysis/components/imports.py | 24 |
| src/training/steps/market_analysis/tas_regime/trading/signal_generator.py | 24 |
| src/training/steps/market_analysis/clusters/risk_mitigation.py | 24 |
| src/utils/input_validation.py | 24 |
| src/utils/step_validation_system.py | 24 |
| src/utils/ml_common/examples/automatic_validation_demo.py | 24 |
| research/feature_comparison/optimized_feature_versions.py | 24 |
| src/interfaces/event_bus.py | 23 |
| src/trading/utils/error_handling.py | 23 |
| src/supervisor/model_behavior_tracker.py | 23 |
| src/training/steps/data_collection/unified_resampler.py | 23 |
| src/training/steps/data_collection/utils/monitoring.py | 23 |
| src/training/steps/pre_training/unified_data_driven_pipeline/time_series_cv/purged_embargoed_cv.py | 23 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/enhanced_walk_forward_validation.py | 23 |
| src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_feature_generation_step.py | 23 |
| src/training/steps/market_analysis/nas_tas_comparison_analysis.py | 23 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/enhanced_analysis.py | 23 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/multi_objective_optimizer.py | 23 |
| src/training/steps/market_analysis/tas_regime/components/neural_architecture.py | 23 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/data_ingestion.py | 23 |
| src/training/steps/market_analysis/clusters/m1_optimizer.py | 23 |
| src/training/steps/market_analysis/clusters/gpu_manager.py | 23 |
| src/training/utils/regime_feature_utils.py | 23 |
| src/features_common/normalization.py | 23 |
| src/utils/confidence.py | 23 |
| src/utils/validated_step_factory.py | 23 |
| src/utils/logger.py | 23 |
| src/utils/ml_common/utils/feature_selection.py | 23 |
| src/utils/data/validation/validators.py | 23 |
| src/utils/hardware/memory_optimization.py | 23 |
| src/analyst/di_analyst.py | 23 |
| exchanges/shared/reliability/rate_limit_manager.py | 23 |
| src/trading/model_selection/model_selector_service.py | 22 |
| src/supervisor/dependency_container.py | 22 |
| src/supervisor/exchange_volume_adapter.py | 22 |
| src/feature_selection/specialized/adaptive_selector.py | 22 |
| src/training/common/component_result.py | 22 |
| src/training/steps/market_analysis/triple_barrier_validator.py | 22 |
| src/training/steps/market_analysis/coverage_constrained_clustering/utils.py | 22 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/performance_benchmark.py | 22 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/unified_architecture_search_engine.py | 22 |
| src/training/steps/market_analysis/regime_analysis/data_access.py | 22 |
| src/training/steps/market_analysis/components/hardware_setup.py | 22 |
| src/training/steps/market_analysis/components/sr_parameter_optimization.py | 22 |
| src/training/steps/market_analysis/tas_regime/optimization/enhanced_hardware_optimization.py | 22 |
| src/training/steps/market_analysis/tas_regime/core/search_space.py | 22 |
| src/training/steps/model_training/auto_step_trigger.py | 22 |
| src/features_common/optimization/cv_base.py | 22 |
| src/utils/error_handling_template.py | 22 |
| src/utils/step_validation_updater.py | 22 |
| src/utils/ml_common/models/hpo_enhancement_guide.py | 22 |
| src/utils/ml_common/optimization/shared_utils/advanced_metrics.py | 22 |
| src/utils/ml_common/training/training_utils.py | 22 |
| src/utils/hardware/enhanced_gpu_manager.py | 22 |
| src/feature_generation/auto_optimization_examples.py | 21 |
| src/feature_generation/core/auto_optimized_feature_generator.py | 21 |
| src/feature_selection/specialized/entropy_balancer.py | 21 |
| src/training/steps/data_collection/data_quality_components/quality_metrics_calculator.py | 21 |
| src/training/steps/market_analysis/labeling_components.py | 21 |
| src/training/steps/market_analysis/enhanced_market_analysis_with_triple_barrier.py | 21 |
| src/training/steps/market_analysis/clusters/engine.py | 21 |
| src/training/steps/market_analysis/clusters/clustering_orchestrator.py | 21 |
| src/features_common/mixins/monitoring_mixin.py | 21 |
| src/utils/step_validation_initializer.py | 21 |
| src/utils/ml_common/optimization/overfitting_prevention.py | 21 |
| src/utils/data/quality/comprehensive_quality_scorer.py | 21 |
| src/utils/matrix_operations/enhanced_operations.py | 21 |
| src/utils/common_ml/backtesting/monte_carlo_engine.py | 21 |
| exchanges/shared/wallet/balance_manager.py | 21 |
| src/feature_generation/test_default_auto_optimization.py | 20 |
| src/feature_generation/convenience/convenience_functions.py | 20 |
| src/feature_generation/core/optimization_strategies.py | 20 |
| src/feature_engineering_roadmap/feature_registry.py | 20 |
| src/tactician/position_closing.py | 20 |
| src/feature_selection/advanced/permutation_importance.py | 20 |
| src/training/steps/data_collection/data_quality_components/data_preprocessor.py | 20 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/enhanced_utility_integration.py | 20 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/regime_aware_training.py | 20 |
| src/training/steps/market_analysis/shared_utils/config.py | 20 |
| src/training/steps/market_analysis/shared_utils/characteristics.py | 20 |
| src/features_common/demo_extensive_logging.py | 20 |
| src/features_common/utils.py | 20 |
| src/utils/standardized_model_manager.py | 20 |
| src/utils/dependency_injection.py | 20 |
| src/utils/sr_clustering/sr_backtesting_engine.py | 20 |
| src/utils/ml_common/math_validation.py | 20 |
| src/utils/core/common.py | 20 |
| src/analyst/regime_runtime.py | 20 |
| src/monitoring/csv_exporter.py | 19 |
| src/models/stacker_lgbm_calibrated.py | 19 |
| src/supervisor/risk_allocator.py | 19 |
| src/tactician/ml_target_updater.py | 19 |
| src/tactician/leverage_sizer.py | 19 |
| src/training/steps/data_collection/unified_data_downloader.py | 19 |
| src/training/steps/market_analysis/components/sr_clustering.py | 19 |
| src/training/steps/market_analysis/nas_regime/core/adaptive_threshold_learning.py | 19 |
| src/training/steps/model_training/tactician_trainer.py | 19 |
| src/training/simplified_architecture/enhanced_interfaces.py | 19 |
| src/utils/ml_common/vectorbt_backtesting_engine.py | 19 |
| src/utils/ml_common/optimization/shared_utils/evaluation_metrics.py | 19 |
| src/utils/ml_common/reporting/validation_reporting_integration.py | 19 |
| src/deployment/rollout_plan.py | 19 |
| src/trading/monitoring/unified_trailing_manager.py | 18 |
| src/trading/cross_asset/trade_gate.py | 18 |
| src/feature_generation/categories/support_resistance.py | 18 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/metrics_reporting.py | 18 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/multi_timeframe_sync.py | 18 |
| src/training/steps/market_analysis/shared_utils/data_preprocessing.py | 18 |
| src/training/steps/market_analysis/tas_regime/utils/visualization.py | 18 |
| src/training/steps/market_analysis/nas_clustering/core/nas_regime_analyzer.py | 18 |
| src/training/steps/market_analysis/nas_modeling/core/nas_evaluator.py | 18 |
| src/training/steps/market_analysis/nas_modeling/core/neural_odes.py | 18 |
| src/training/steps/model_training/analyst_training_validation.py | 18 |
| src/training/steps/model_validation/tactician_validator.py | 18 |
| src/training/core/training_manager.py | 18 |
| src/core/examples/decorator_usage.py | 18 |
| src/features_common/error_handling.py | 18 |
| src/features_common/factories/scaler_factory.py | 18 |
| src/features_common/mixins/performance_mixin.py | 18 |
| src/utils/nonlinear_optimization_helpers.py | 18 |
| src/utils/cross_step_validation.py | 18 |
| src/utils/ml_common/vectorbt_performance_monitor.py | 18 |
| src/utils/ml_common/optimization/pure_tree_nas.py | 18 |
| src/utils/ml_common/evaluation/evaluation_utils.py | 18 |
| src/utils/ml_common/validation/enhanced_validation.py | 18 |
| research/feature_comparison/feature_versions.py | 18 |
| src/launcher/step_orchestrator_wrapper.py | 17 |
| src/trading/model_selection/trading_model_manager.py | 17 |
| src/supervisor/coordinator/system_coordinator.py | 17 |
| src/feature_generation/core/feature_cache.py | 17 |
| src/feature_generation/base_calculations/base_calculator.py | 17 |
| src/tactician/fully_migrated_tactician.py | 17 |
| src/training/steps/market_analysis/components/base_component.py | 17 |
| src/training/steps/market_analysis/nas_clustering/core/nas_search/evolutionary_search.py | 17 |
| src/training/utils/feature_selection/partial_information_decomposition.py | 17 |
| src/core/decorators/compose.py | 17 |
| src/utils/regime_ensemble_utils.py | 17 |
| src/utils/dependency_manager.py | 17 |
| src/utils/ml_common/optimization/tree_based_architecture_search.py | 17 |
| src/utils/ml_common/optimization/shared_utils/integration_verification.py | 17 |
| src/utils/ml_common/validation/underfitting_detection.py | 17 |
| src/analyst/order_book_analyzer.py | 17 |
| src/analyst/liquidation_risk_model.py | 17 |
| src/analyst/dynamic_regime_mapper.py | 17 |
| src/monitoring/enhanced_monitoring_launcher.py | 16 |
| src/launcher/configuration_manager.py | 16 |
| src/supervisor/coordinator/online_learning_manager.py | 16 |
| src/feature_generation/test_auto_optimization_integration.py | 16 |
| src/feature_generation/utils/optimized_feature_factory.py | 16 |
| src/training/steps/pre_training/column_naming.py | 16 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/optimized_clustering.py | 16 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/vectorized_operations.py | 16 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/automatic_training/regime_hpo_integration.py | 16 |
| src/training/steps/market_analysis/components/memory_manager.py | 16 |
| src/training/steps/market_analysis/optimized_multi_horizon_optimizer/enhanced_validation.py | 16 |
| src/training/steps/market_analysis/optimized_multi_horizon_optimizer/optimized_timeframe_optimizer.py | 16 |
| src/training/steps/market_analysis/clusters/features/selector.py | 16 |
| src/training/steps/model_training/patchtst_wrapper.py | 16 |
| src/utils/regime_data_access.py | 16 |
| src/utils/ml_common/optimization/hybrid_nas_system.py | 16 |
| src/utils/ml_common/explainability/shap_lime_integration.py | 16 |
| src/utils/hmm/__init__.py | 16 |
| src/utils/core/math_utilities.py | 16 |
| exchanges/binance/klines_adapter.py | 16 |
| exchanges/mexc/klines_adapter.py | 16 |
| exchanges/bingx/klines_adapter.py | 16 |
| exchanges/phemex/klines_adapter.py | 16 |
| exchanges/shared/wallet/balance_manager_old.py | 16 |
| exchanges/okx/klines_adapter.py | 16 |
| exchanges/gateio/klines_adapter.py | 16 |
| src/models/causal_dilated_tcn.py | 15 |
| src/supervisor/optimizer.py | 15 |
| src/feature_generation/categories/negative_learning.py | 15 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/optimized_integration.py | 15 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_meta_learning.py | 15 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_config_manager.py | 15 |
| src/training/steps/market_analysis/regime_analysis/service.py | 15 |
| src/training/steps/market_analysis/nas_modeling/core/meta_learning.py | 15 |
| src/training/steps/market_analysis/nas_modeling/core/neural_state_space_nas.py | 15 |
| src/core/dependency_injection.py | 15 |
| src/core/decorators/errors.py | 15 |
| src/utils/ml_common/models/model_cache.py | 15 |
| src/utils/ml_common/optimization/bayesian_entry_timing_optimizer.py | 15 |
| src/end_to_end_roadmap.py | 14 |
| src/trading/cross_asset/cross_asset_trading_manager.py | 14 |
| src/sentinel/sentinel.py | 14 |
| src/feature_generation/utils/optimized_cross_timeframe_analysis_methods.py | 14 |
| src/feature_generation/utils/statistical_calculations_optimizer.py | 14 |
| src/feature_selection/core/framework.py | 14 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/random_seed_manager.py | 14 |
| src/training/steps/market_analysis/optimization_cache.py | 14 |
| src/training/steps/market_analysis/tas_regime/production/monitoring.py | 14 |
| src/training/steps/market_analysis/regime_data_splitting/config_utils.py | 14 |
| src/training/steps/market_analysis/nas_regime/core/neural_architectures.py | 14 |
| src/config/sr_comprehensive_config_loader.py | 14 |
| src/utils/data_loader.py | 14 |
| src/utils/unified_utility_registry.py | 14 |
| src/utils/ml_common/optimization/hpo_diagnostics_and_fixes.py | 14 |
| src/utils/ml_common/data_processing/regime_processing.py | 14 |
| src/utils/matrix_operations/vectorized_correlations.py | 14 |
| src/nas_tas/monitoring/performance_monitor.py | 14 |
| src/nas_tas/evaluation/financial_metrics.py | 14 |
| src/nas_tas/config/validation_config.py | 14 |
| src/nas_tas/config/base_config.py | 14 |
| src/feature_generation/core/rolling_operations_mixin.py | 13 |
| src/feature_generation/utils/consolidated_rolling_optimizer.py | 13 |
| src/research/profit_labeling/test_enhanced_integration.py | 13 |
| src/tactician/enhanced_execution_manager.py | 13 |
| src/feature_selection/vectorbt/vectorbt_utils.py | 13 |
| src/training/steps/market_analysis/regime_aware_triple_barrier_optimizer.py | 13 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/financial_architecture_primitives.py | 13 |
| src/training/steps/market_analysis/tas_regime/verify_migration.py | 13 |
| src/training/steps/market_analysis/tas_regime/backtesting/walk_forward_analysis.py | 13 |
| src/training/steps/market_analysis/nas_clustering/core/nas_search/search_space.py | 13 |
| src/training/steps/model_training/xgboost_custom.py | 13 |
| src/training/utils/feature_selection/enhanced_partial_information_decomposition.py | 13 |
| src/core/enhanced_factories.py | 13 |
| src/features_common/demo_vectorbt_default.py | 13 |
| src/features_common/config/vectorbt_config.py | 13 |
| src/utils/memory_management/streaming_data_processor.py | 13 |
| src/utils/data/quality/data_qualification_config.py | 13 |
| src/utils/config/loaders.py | 13 |
| src/analyst/ml_dynamic_target_predictor.py | 13 |
| research/feature_comparison/robust_scaling.py | 13 |
| exchanges/shared/auth/auth_manager.py | 13 |
| src/trading/cross_asset/consolidated_reporting.py | 12 |
| src/feature_generation/core/feature_registry.py | 12 |
| src/research/crypto_analysis/data_downloader.py | 12 |
| src/feature_selection/sparse/sparse_feature_selector.py | 12 |
| src/feature_selection/optimizations/vectorized_operations.py | 12 |
| src/training/steps/data_collection/data_quality_components/data_integrity_checker.py | 12 |
| src/training/steps/data_collection/data_quality_components/error_handler.py | 12 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/ml_common_integration.py | 12 |
| src/training/steps/market_analysis/components/clustering_config.py | 12 |
| src/training/steps/market_analysis/components/sr_detection.py | 12 |
| src/training/steps/market_analysis/clusters/validation_framework.py | 12 |
| src/training/config/data_locator.py | 12 |
| src/core/di_integration.py | 12 |
| src/features_common/test_optimization_demo.py | 12 |
| src/features_common/test_backward_compatibility.py | 12 |
| src/utils/structured_logging.py | 12 |
| src/utils/enhanced_error_handler.py | 12 |
| src/utils/sr_clustering/backtesting_enhanced_clustering.py | 12 |
| src/utils/ml_common/vectorbt_memory_manager.py | 12 |
| src/utils/ml_common/validation/integrated_validation_system.py | 12 |
| src/utils/matrix_operations/__init__.py | 12 |
| research/cluster_analysis/clustering/optimal_cluster_selection.py | 12 |
| research/crypto_analysis/data_downloader.py | 12 |
| research/clusters/data_driven_clustering_framework.py | 12 |
| src/launcher/ARES_LAUNCHER_VERIFICATION.py | 11 |
| src/trading/config/trading_config.py | 11 |
| src/trading/config/execution_config.py | 11 |
| src/supervisor/loss_functions/pnl_calculator.py | 11 |
| src/feature_generation/utils/migration_helper.py | 11 |
| src/feature_generation/utils/step06_labeling_components/fractional_triple_barrier_labeling.py | 11 |
| src/feature_selection/caching/intelligent_feature_cache.py | 11 |
| src/training/steps/data_collection/data_downloader.py | 11 |
| src/training/steps/data_collection/data_preparation/run_step1.py | 11 |
| src/training/steps/market_analysis/standalone_optimizer.py | 11 |
| src/training/steps/market_analysis/coverage_constrained_clustering/run.py | 11 |
| src/training/steps/market_analysis/tas_regime/uncertainty/uncertainty_estimation.py | 11 |
| src/training/steps/market_analysis/tas_regime/core/tas_result.py | 11 |
| src/training/steps/market_analysis/tas_regime/core/tas_config.py | 11 |
| src/training/simplified_architecture/standard_interfaces.py | 11 |
| src/core/errors/mapping.py | 11 |
| src/features_common/config/optimization_config.py | 11 |
| src/utils/step_validation_wrapper.py | 11 |
| src/utils/state_manager.py | 11 |
| src/utils/ml_common/matrix_cross_validation.py | 11 |
| src/utils/ml_common/explainability/model_explainability.py | 11 |
| src/utils/ml_common/validation/universal_ml_validation.py | 11 |
| src/utils/ml_common/data_processing/data_cleaning_utils.py | 11 |
| research/feature_comparison/feature_comparison_utils.py | 11 |
| exchanges/shared/auth/api_key_manager.py | 11 |
| exchanges/shared/pricing/price_manager.py | 11 |
| src/monitoring/auto_monitoring_demo.py | 10 |
| src/research/crypto_analysis/run_optimized_analysis.py | 10 |
| src/feature_selection/vectorbt/vectorbt_config.py | 10 |
| src/training/steps/feature_engineering/filters/advanced_filters_15m.py | 10 |
| src/training/steps/market_analysis/nas_clustering/core/nas_regime_optimizer.py | 10 |
| src/training/steps/market_analysis/nas_regime/optimization/multi_objective_optimizer.py | 10 |
| src/core/di_launcher.py | 10 |
| src/features_common/backward_compatibility.py | 10 |
| src/config/pipeline_modes.py | 10 |
| src/config/computational_optimization_config.py | 10 |
| src/utils/serialization_utils.py | 10 |
| src/utils/ml_common/ensembles/stacking_ensemble_manager.py | 10 |
| src/database/firestore_manager.py | 10 |
| research/cluster_analysis/price_patterns/pattern_validation.py | 10 |
| research/crypto_analysis/run_optimized_analysis.py | 10 |
| exchanges/shared/auth/time_sync.py | 10 |
| exchanges/shared/orders/order_manager_old.py | 10 |
| src/supervisor/coordinator/recovery_manager.py | 9 |
| src/feature_generation/core/auto_optimization_config.py | 9 |
| src/research/crypto_analysis/run_analysis.py | 9 |
| src/tactician/sr_levels/sr_modules/sr_probability_calculator.py | 9 |
| src/training/steps/models_training/enhanced_entry_quality_scorer.py | 9 |
| src/training/steps/models_training/negative_learning_training_integration.py | 9 |
| src/training/steps/market_analysis/__init__.py | 9 |
| src/training/steps/market_analysis/coverage_constrained_clustering/component.py | 9 |
| src/training/steps/market_analysis/components/clustering_algorithms.py | 9 |
| src/training/steps/market_analysis/nas_modeling/core/nas_trainer.py | 9 |
| src/training/steps/market_analysis/regime_data_splitting/validation_utils.py | 9 |
| src/core/unified_config_service.py | 9 |
| src/features_common/logging_config.py | 9 |
| src/config/fractional_implementations_config.py | 9 |
| src/config/regime_feature_thresholds.py | 9 |
| src/utils/decorators/__init__.py | 9 |
| src/utils/ml_common/validation/__init__.py | 9 |
| src/utils/hmm/optimization.py | 9 |
| src/analyst/market_health_analyzer.py | 9 |
| research/crypto_analysis/run_analysis.py | 9 |
| GUI/verify_gui_workflow.py | 9 |
| exchanges/shared/interfaces_typed.py | 9 |
| exchanges/shared/pricing/ohlcv_manager.py | 9 |
| exchanges/shared/market/precision_helper.py | 9 |
| src/monitoring/integration_manager.py | 8 |
| src/launcher/gui_manager.py | 8 |
| src/models/tcn_regressor.py | 8 |
| src/feature_generation/__init__.py | 8 |
| src/feature_engineering_roadmap/interactions.py | 8 |
| src/feature_selection/advanced/adaptive_weighting.py | 8 |
| src/training/steps/data_collection/data_quality_components/config_manager.py | 8 |
| src/training/steps/backtesting/abc_testing/paper_trading_engine.py | 8 |
| src/training/steps/feature_engineering/register_features.py | 8 |
| src/training/steps/market_analysis/nas_clustering/core/nas_feature_extractor.py | 8 |
| src/training/steps/market_analysis/nas_regime/integration/nas_unified_integration.py | 8 |
| src/training/utils/embedding_postprocessing.py | 8 |
| src/core/errors/handlers/http.py | 8 |
| src/features_common/factories/optimizer_factory.py | 8 |
| src/config/regime_specific_optimization_config.py | 8 |
| src/utils/pipeline_results_manager.py | 8 |
| src/utils/compat.py | 8 |
| src/utils/artifact_manager.py | 8 |
| src/utils/ml_common/feature_selection_backwards_compat.py | 8 |
| src/utils/data/ares_launcher_data_loader.py | 8 |
| src/utils/data/quality/statistical_distribution_validation.py | 8 |
| src/utils/hardware/demo_implementation.py | 8 |
| src/nas_tas/config/search_config.py | 8 |
| research/cluster_analysis/price_patterns/__init__.py | 8 |
| research/cluster_analysis/clustering/__init__.py | 8 |
| research/clusters/dynamic_targets.py | 8 |
| examples/tactician_t1_t4_models_usage.py | 8 |
| data_quality/mapping/dependency_graph.py | 8 |
| exchanges/shared/orders/idempotency_manager.py | 8 |
| exchanges/shared/risk/risk_calculator.py | 8 |
| src/monitoring/correlation_manager.py | 7 |
| src/supervisor/pnl_loss_functions.py | 7 |
| src/supervisor/coordinator/health_monitor.py | 7 |
| src/training/steps/feature_engineering/feature_selector.py | 7 |
| src/training/steps/market_analysis/automatic_timeframe_optimizer.py | 7 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_hardware_optimizer.py | 7 |
| src/training/steps/market_analysis/shared_utils/logging_utils.py | 7 |
| src/training/steps/market_analysis/shared_utils/calibration_registry.py | 7 |
| src/training/steps/market_analysis/tas_regime/integration/tas_unified_integration.py | 7 |
| src/training/steps/model_training/analyst_training_hardware.py | 7 |
| src/utils/purged_kfold.py | 7 |
| src/utils/artifact_pickup_utils.py | 7 |
| src/utils/ml_common/validation/thresholding.py | 7 |
| research/feature_comparison/method_settings.py | 7 |
| exchanges/shared/examples/high_level_usage.py | 7 |
| src/supervisor/global_portfolio_manager.py | 6 |
| src/supervisor/enhanced_model_monitor.py | 6 |
| src/supervisor/loss_functions/optimization_metrics.py | 6 |
| src/supervisor/coordinator/component_monitor.py | 6 |
| src/feature_engineering_roadmap/dynamic_feature_selector.py | 6 |
| src/training/steps/data_collection/data_preparation/step01_data_collection.py | 6 |
| src/training/steps/market_analysis/enhanced_multi_horizon_pipeline.py | 6 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/enhanced_clustering_integration.py | 6 |
| src/training/steps/market_analysis/tas_regime/regime_analysis/regime_reporting.py | 6 |
| src/training/steps/market_analysis/tas_regime/regime_analysis/regime_optimization.py | 6 |
| src/training/steps/market_analysis/tas_regime/utils/logging.py | 6 |
| src/core/domain.py | 6 |
| src/features_common/logging_enhancements.py | 6 |
| src/config/label_model_mapping.py | 6 |
| src/config/typed_config.py | 6 |
| src/utils/data_processing_utils.py | 6 |
| src/utils/caching.py | 6 |
| src/utils/ml_common/vectorbt_financial_metrics.py | 6 |
| src/utils/ml_common/config/universal_timeframe_config.py | 6 |
| src/utils/data/unified_data_utils.py | 6 |
| src/utils/data/real_data_loader.py | 6 |
| src/utils/data/__init__.py | 6 |
| src/database/influxdb_manager.py | 6 |
| research/clusters/__init__.py | 6 |
| examples/multi_stage_feature_selection_example.py | 6 |
| exchanges/shared/market/market_metadata.py | 6 |
| exchanges/base_exchange/exchange_interface.py | 6 |
| src/monitoring/ml_monitor.py | 5 |
| src/launcher/command_handlers.py | 5 |
| src/supervisor/loss_functions/loss_calculator.py | 5 |
| src/supervisor/loss_functions/performance_metrics.py | 5 |
| src/training/steps/data_collection/exchange_field_mappings.py | 5 |
| src/training/steps/market_analysis/gradient_flow_analysis.py | 5 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/regime_model_mapping/hybrid_integration.py | 5 |
| src/training/steps/market_analysis/tas_regime/regime_analysis/tree_regime_analyzer.py | 5 |
| src/training/steps/market_analysis/tas_regime/uncertainty/robustness_analysis.py | 5 |
| src/training/steps/market_analysis/nas_modeling/core/hardware_acceleration.py | 5 |
| src/training/steps/market_analysis/regime_model_mapping/nas_integration.py | 5 |
| src/training/steps/market_analysis/regime_model_mapping/tas_integration.py | 5 |
| src/training/steps/market_analysis/nas_regime/core/perfect_nas_config.py | 5 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_perfect_nas_config.py | 5 |
| src/utils/math_validation.py | 5 |
| src/utils/sr_clustering/__init__.py | 5 |
| src/utils/ml_common/vectorized_backtesting.py | 5 |
| src/utils/ml_common/evaluation/enhanced_learning_curve_analysis.py | 5 |
| research/cluster_analysis/market_factor_analysis/__init__.py | 5 |
| torch_stub/__init__.py | 4 |
| src/trading/ensemble_disagreement_features.py | 4 |
| src/trading/utils/ohlcv.py | 4 |
| src/supervisor/loss_functions/base.py | 4 |
| src/supervisor/loss_functions/risk_metrics.py | 4 |
| src/training/steps/data_collection/data_quality_components/result_builder.py | 4 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/financial_loss_functions.py | 4 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/config/multi_timeframe_config.py | 4 |
| src/training/steps/market_analysis/tas_regime/evaluation/tree_evaluator.py | 4 |
| src/core/generic_base.py | 4 |
| src/config/environment.py | 4 |
| src/config/config_manager.py | 4 |
| src/config/validation.py | 4 |
| src/utils/monitoring_utils.py | 4 |
| src/utils/version_manager.py | 4 |
| src/utils/numba_timestamps.py | 4 |
| src/utils/hmm/core_manager.py | 4 |
| src/nas_tas/config/tprint_config.py | 4 |
| exchanges/shared/tests/verify_improvements.py | 4 |
| src/monitoring/performance_monitor.py | 3 |
| src/trading/nas_tas_trading_main.py | 3 |
| src/supervisor/monitoring.py | 3 |
| src/supervisor/coordinator/circuit_breaker.py | 3 |
| src/training/steps/backtesting/vectorbt_optimization_example.py | 3 |
| src/training/steps/market_analysis/regime_analysis/reporting.py | 3 |
| src/training/steps/market_analysis/tas_regime/evaluation/multi_objective_evaluation.py | 3 |
| src/training/steps/market_analysis/nas_clustering/core/nas_config.py | 3 |
| src/training/steps/market_analysis/nas_modeling/core/rl_nas.py | 3 |
| src/training/steps/market_analysis/clustering/config/clustering_config.py | 3 |
| src/training/steps/market_analysis/nas_regime/meta_learning/adaptive_regime_learner.py | 3 |
| src/features_common/factories/registry_factory.py | 3 |
| src/features_common/config/unified_config.py | 3 |
| src/config/sr_optimization_config.py | 3 |
| src/config/m1_gpu_config.py | 3 |
| src/config/analytical_process_config.py | 3 |
| src/config/computational_optimization.py | 3 |
| src/config/enhanced_reporting_config.py | 3 |
| src/utils/performance.py | 3 |
| src/utils/tracing.py | 3 |
| src/utils/random_seeding.py | 3 |
| src/utils/validation_decorators.py | 3 |
| src/utils/signal_handler.py | 3 |
| src/utils/tprint_integration.py | 3 |
| src/utils/prometheus_metrics.py | 3 |
| src/utils/ml_common/__init__.py | 3 |
| src/utils/ml_common/validation/temporal_cross_validation.py | 3 |
| src/utils/data/quality/gap_collection_hook.py | 3 |
| src/monitoring/advanced_tracer.py | 2 |
| src/monitoring/gui/launch_dashboard.py | 2 |
| src/tactician/enhanced_prediction_integrator.py | 2 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/simplified_config.py | 2 |
| src/training/steps/market_analysis/tas_regime/evaluation/regime_evaluation.py | 2 |
| src/core/service_registry.py | 2 |
| src/features_common/factories/unified_factory.py | 2 |
| src/features_common/registry/base_registry.py | 2 |
| src/config/multi_output_config.py | 2 |
| src/utils/observability.py | 2 |
| src/utils/regime_transition_handler.py | 2 |
| src/utils/ml_common/config/enhanced_ml_config.py | 2 |
| src/utils/core/data_types.py | 2 |
| src/utils/hardware/__init__.py | 2 |
| src/nas_tas/logging.py | 2 |
| src/monitoring/performance_dashboard.py | 1 |
| src/trading/examples/cross_asset_trading_demo.py | 1 |
| src/trading/examples/full_monitoring_demo.py | 1 |
| src/feature_generation/example_usage.py | 1 |
| src/feature_generation/categories/cross_timeframe.py | 1 |
| src/feature_generation/categories/regime_features.py | 1 |
| src/feature_generation/categories/enhanced_vectorbt_volatility.py | 1 |
| src/feature_generation/categories/interaction.py | 1 |
| src/feature_generation/tests/test_cleanup_validation.py | 1 |
| src/feature_generation/core/__init__.py | 1 |
| src/feature_generation/utils/optimized_feature_orchestrator.py | 1 |
| src/feature_generation/utils/sr_feature_extractor.py | 1 |
| src/feature_generation/utils/contrastive_learning_guide.py | 1 |
| src/feature_generation/utils/enhanced_optimization_system.py | 1 |
| src/feature_generation/utils/step06_comprehensive_implementation.py | 1 |
| src/feature_generation/utils/temporal_feature_integration.py | 1 |
| src/feature_generation/utils/optimized_cross_timeframe_analysis_advanced.py | 1 |
| src/feature_generation/utils/memory_optimizer.py | 1 |
| src/feature_generation/utils/cross_timeframe_interaction_features.py | 1 |
| src/feature_generation/utils/feature_generators_compatibility.py | 1 |
| src/feature_generation/utils/cross_timeframe_talib_integration.py | 1 |
| src/feature_generation/utils/__init__.py | 1 |
| src/feature_generation/utils/feature_generation_optimization.py | 1 |
| src/feature_generation/utils/statsmodels_integration.py | 1 |
| src/feature_generation/utils/optimization/lookback_optimizer.py | 1 |
| src/feature_generation/utils/step06_labeling_components/optimized_triple_barrier_labeling.py | 1 |
| src/feature_generation/utils/step06_labeling_components/regime_specific_triple_barrier_optimizer.py | 1 |
| src/feature_generation/utils/step06_labeling_components/regime_aware_triple_barrier_labeling.py | 1 |
| src/feature_generation/utils/step06_labeling_components/profit_based_feature_engineering.py | 1 |
| src/research/crypto_analysis/config.py | 1 |
| src/feature_engineering_roadmap/lookback_selection.py | 1 |
| src/feature_engineering_roadmap/ensemble_meta_features.py | 1 |
| src/tactician/position_monitor.py | 1 |
| src/tactician/sr_detection_optimization.py | 1 |
| src/tactician/sr_levels/enhanced_sr_detection.py | 1 |
| src/tactician/sr_levels/sr_breakout_predictor_enhanced.py | 1 |
| src/feature_selection/specialized/directional_selector.py | 1 |
| src/training/steps/data_collection/enhanced_klines_processing_pipeline.py | 1 |
| src/training/steps/data_collection/data_preparation/step01_5_data_converter.py | 1 |
| src/training/steps/data_collection/data_preparation/sr_strength_optimizer.py | 1 |
| src/training/steps/models_training/tactician_pre_ml_orchestration.py | 1 |
| src/training/steps/models_training/enhanced_tactician_pre_ml_orchestration.py | 1 |
| src/training/steps/models_training/tactician_models_training.py | 1 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/vectorbt_enhancements.py | 1 |
| src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_feature_selection_step.py | 1 |
| src/training/steps/market_analysis/regime_analysis_script.py | 1 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/shared_optimization.py | 1 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/enhanced_regime_evaluator.py | 1 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/robust_scoring_models.py | 1 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/nas_financial_features.py | 1 |
| src/training/steps/market_analysis/components/deprecated_nas_tas_clustering.py | 1 |
| src/training/steps/market_analysis/components/standardized_features.py | 1 |
| src/training/steps/market_analysis/tas_regime/regime_analysis/regime_qualification.py | 1 |
| src/training/steps/market_analysis/tas_regime/shared_utils/analysis_components.py | 1 |
| src/training/steps/market_analysis/tas_regime/shared_utils/search_strategies.py | 1 |
| src/training/steps/market_analysis/tas_regime/shared_utils/position_aware_trading.py | 1 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/feature_engineering.py | 1 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/data_preprocessing.py | 1 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/regime_detection.py | 1 |
| src/training/steps/market_analysis/tas_regime/core/advanced_tas_search.py | 1 |
| src/training/steps/market_analysis/tas_regime/core/tas_regime_config.py | 1 |
| src/training/steps/market_analysis/tas_regime/core/tree_cvlSA_architecture.py | 1 |
| src/training/steps/market_analysis/regime_data_splitting/regime_data_splitting_main.py | 1 |
| src/training/steps/market_analysis/regime_data_splitting/regime_data_splitting_component.py | 1 |
| src/training/steps/market_analysis/regime_data_splitting/streamlined_regime_data_splitting.py | 1 |
| src/training/steps/market_analysis/nas_regime/evaluation/trading_viability_evaluator.py | 1 |
| src/training/steps/market_analysis/nas_regime/evaluation/economic_evaluator.py | 1 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_data_operations.py | 1 |
| src/training/steps/model_training/tactician_lookback_optimization.py | 1 |
| src/training/utils/feature_calculators.py | 1 |
| src/training/simplified_architecture/modular_components.py | 1 |
| src/training/simplified_architecture/migrated_components/data_components.py | 1 |
| src/core/injectable_base.py | 1 |
| src/features_common/vectorbt/optimization_engine.py | 1 |
| src/features_common/vectorbt/gpu_accelerator.py | 1 |
| src/config/training_modes.py | 1 |
| src/config/multi_timeframe_hmm_ensemble_config.py | 1 |
| src/config/enhanced_matrix_config.py | 1 |
| src/config/__init__.py | 1 |
| src/config/trading.py | 1 |
| src/utils/intensity_scaler.py | 1 |
| src/utils/graceful_module_handler.py | 1 |
| src/utils/feature_engineering_validation.py | 1 |
| src/utils/enhanced_data_operations.py | 1 |
| src/utils/ml_common/vectorbt_memory_optimizer.py | 1 |
| src/utils/ml_common/optimization/unsupervised_tree_nas.py | 1 |
| src/utils/ml_common/optimization/trading_tree_architecture_search.py | 1 |
| src/utils/ml_common/optimization/regime_trading_tree_nas.py | 1 |
| src/utils/ml_common/data_processing/data_quality.py | 1 |
| src/utils/ml_common/data_processing/multi_timeframe_training.py | 1 |
| src/utils/ml_common/ensembles/__init__.py | 1 |
| src/utils/core/time_utilities.py | 1 |
| src/utils/common_ml/backtesting/turnover.py | 1 |
| src/nas_tas/results/comparison_utils.py | 1 |
| src/analyst/advanced_feature_engineering.py | 1 |
| src/analyst/feature_engineering_orchestrator.py | 1 |
| src/analyst/location_classifier_optimization.py | 1 |
| src/analyst/autoencoder_feature_generator.py | 1 |
| src/analyst/unified_regime_classifier.py | 1 |
| src/analyst/unified_regime_classifier_sr_optimized.py | 1 |
| src/analyst/analyst.py | 1 |
| src/analyst/predictive_ensembles/directional_specialist_model.py | 1 |
| src/analyst/predictive_ensembles/regime_ensembles/base_ensemble.py | 1 |
| live_trading/error_handler.py | 1 |
| research/crypto_analysis/config.py | 1 |
| research/clusters/feature_selection.py | 1 |
| examples/enhanced_label_definitions_demo.py | 1 |
| exchanges/shared/market/risk_tier_manager.py | 1 |
