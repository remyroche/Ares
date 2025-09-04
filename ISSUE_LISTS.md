# Code Quality Issue Lists

## 1. Files with Syntax Errors (126 issues)

### Critical Syntax Errors (Invalid Syntax):
```
/workspace/src/analyst/unified_regime_classifier_fractal_enhanced.py:28
/workspace/src/analyst/unified_regime_classifier_fractal_simplified.py:26
/workspace/src/analyst/ml_confidence_predictor.py:2614
/workspace/src/training/steps/data_collection/enhanced_data_collection_demo.py:35
/workspace/src/training/steps/optimisation/step17_parameter_optimization_wrapper.py:41
/workspace/src/training/steps/model_training/step09_5_multi_timeframe_hmm_ensemble_validator.py:153
/workspace/src/training/steps/model_training/step13_analyst_ensemble_creation.py:163
/workspace/src/training/steps/market_analysis/step1/data_resampler.py:28
/workspace/src/training/steps/market_analysis/step1/data_gap_detector.py:38
/workspace/src/training/steps/data_collection/data_preparation/step01_5_data_converter_wrapper.py:37
/workspace/src/tactician/sr_levels/sr_weight_optimizer.py:26
/workspace/src/paper_trader.py:46
/workspace/src/launcher/enhanced_trading_launcher.py:165
/workspace/src/interfaces/enhanced_event_bus.py:24
/workspace/src/tactician/position_closing.py:18
/workspace/src/training/model_trainer.py:326
/workspace/src/training/progress_manager.py:18
/workspace/src/training/wavelet_caching_workflow.py:242
/workspace/src/exchange/binance.py:17
/workspace/src/utils/state_manager.py:19
/workspace/src/utils/data_access_protection.py:26
```

### Unexpected Indent Errors:
```
/workspace/src/analyst/meta_label_relevance.py:88
/workspace/src/training/steps/data_collection/step02_5_sr_optimization_validator.py:369
/workspace/src/training/steps/data_collection/integrated_data_quality_pipeline.py:484
/workspace/src/training/steps/backtesting/step18_walk_forward_validation_validator.py:428
/workspace/src/training/steps/backtesting/step19_monte_carlo_validation_validator.py:497
/workspace/src/training/steps/step06_labeling_components/optimized_triple_barrier_labeling.py:468
/workspace/src/training/steps/step06_labeling_components/regime_aware_triple_barrier_labeling.py:535
/workspace/src/training/steps/step06_labeling_components/profit_based_feature_engineering.py:708
/workspace/src/training/steps/market_analysis/step08_advanced_feature_selection.py:961
/workspace/src/training/steps/model_training/step12_analyst_enhancement.py:1687
/workspace/src/training/steps/market_analysis/step17_final_parameters_optimization/efficiency_optimizer.py:486
/workspace/src/training/steps/market_analysis/hmm_clustering/step03_parameter_optimization.py:620
/workspace/src/training/steps/market_analysis/hmm_clustering/step03_5_final_regime_clustering.py:882
/workspace/src/training/steps/data_collection/feature_engineering/step08_advanced_feature_selection_wrapper.py:46
/workspace/src/training/steps/data_collection/data_preparation/step02_5_sr_optimization.py:1780
/workspace/src/training/steps/data_collection/data_preparation/step01_data_collection.py:174
/workspace/src/tactician/sr_levels/sr_ensemble_predictor.py:55
/workspace/src/pipelines/improved_pipeline_executor.py:159
/workspace/src/tactician/enhanced_execution_manager.py:47
/workspace/src/training/enhanced_feature_engineering_optimizer.py:389
/workspace/src/training/enhanced_training_manager_optimized.py:292
/workspace/src/training/multi_objective_optimizer.py:67
/workspace/src/training/optimized_feature_selection_manager.py:304
/workspace/src/core/examples/decorator_usage.py:189
/workspace/src/utils/enhanced_mlflow_integration.py:1343
/workspace/src/utils/signal_handler.py:346
/workspace/src/utils/data_formatting_framework.py:345
/workspace/src/training/steps/model_training/validation_components/confidence_calibration_step.py:192
/workspace/src/training/steps/model_training/validation_components/ab_testing_step.py:35
```

## 2. Files with Import Issues (1,462 issues)

### Major Import Problem Areas:
```
/workspace/src/analyst/
/workspace/src/training/
/workspace/src/tactician/
/workspace/src/utils/
/workspace/src/database/
/workspace/src/strategist/
/workspace/src/supervisor/
/workspace/src/monitoring/
/workspace/src/launcher/
/workspace/src/interfaces/
/workspace/src/pipelines/
/workspace/src/integration/
/workspace/src/exchange/
```

## 3. Files with Complexity Issues (195 issues)

### Deep Nesting Issues (194 files):
```
/workspace/src/training/enhanced_training_manager.py:1695
/workspace/src/training/enhanced_matrix_operations.py:1526
/workspace/src/training/multi_output_model_trainer.py:1763
/workspace/src/training/steps/model_training/step09_hmm_based_training.py:1552
/workspace/src/training/steps/model_training/step10_unified_regime_intelligence.py:1912
/workspace/src/training/steps/market_analysis/step07_enhanced_matrix_operations.py:1118
/workspace/src/training/steps/market_analysis/step08_advanced_feature_selection.py:961
/workspace/src/training/steps/optimisation/step17_final_parameters_optimization_new.py:981
/workspace/src/training/steps/data_collection/data_preparation/step02_5_sr_optimization.py:1780
/workspace/src/training/steps/market_analysis/hmm_clustering/step03_5_final_regime_clustering.py:882
/workspace/src/training/steps/market_analysis/hmm_clustering/step03_parameter_optimization.py:620
/workspace/src/training/steps/market_analysis/step1/enhanced_data_quality_manager.py:464
/workspace/src/training/steps/market_analysis/step1/data_efficiency_optimizer.py:780
/workspace/src/training/steps/market_analysis/step1/wavelet_feature_selection_workflow.py:694
/workspace/src/training/steps/market_analysis/step1/fractional_feature_selector.py:685
/workspace/src/training/steps/market_analysis/step1/feature_selection_manager.py:886
/workspace/src/training/steps/market_analysis/step1/step07_enhanced_matrix_operations_per_regime.py:418
/workspace/src/training/steps/market_analysis/step1/step06_feature_engineering_per_regime.py:785
/workspace/src/training/steps/market_analysis/step1/enhanced_market_analysis_orchestrator.py:789
/workspace/src/training/steps/market_analysis/step1/step05_labeling.py:425
/workspace/src/training/steps/market_analysis/step1/combined_fractional_system.py:458
/workspace/src/training/steps/market_analysis/step1/step04_regime_data_splitting.py:188
/workspace/src/training/steps/market_analysis/step1/step04_5_triple_barrier_method_validator.py:45
/workspace/src/training/steps/market_analysis/step1/enhanced_step_validator.py:326
/workspace/src/training/steps/market_analysis/step1/step03_hmm_regime_discovery_1h.py:120
/workspace/src/training/steps/market_analysis/step1/step03_bayesian_parameter_optimization.py:59
/workspace/src/training/steps/market_analysis/step1/model_serializer.py:247
/workspace/src/training/steps/market_analysis/step1/metadata_tracker.py:137
/workspace/src/training/steps/market_analysis/step1/data_quality_monitor.py:135
/workspace/src/training/steps/market_analysis/step1/step02_5_sr_optimization_validator.py:369
/workspace/src/training/steps/market_analysis/step1/step02_data_reading_validator.py:49
/workspace/src/training/steps/market_analysis/step1/step02_data_reading.py:191
/workspace/src/training/steps/market_analysis/step1/unified_data_loader.py:25
/workspace/src/training/steps/market_analysis/step1/data_downloader.py:70
/workspace/src/training/steps/market_analysis/step1/enhanced_api_agnostic_data_collector.py:835
/workspace/src/training/steps/market_analysis/step1/step01_data_collection.py:174
/workspace/src/training/steps/market_analysis/step1/step01_5_data_converter_wrapper.py:37
/workspace/src/training/steps/market_analysis/step1/step08_advanced_feature_selection_wrapper.py:46
/workspace/src/training/steps/market_analysis/step1/step09_5_multi_timeframe_hmm_ensemble.py:338
/workspace/src/training/steps/market_analysis/step1/step09_hmm_based_training_per_regime.py:745
/workspace/src/training/steps/market_analysis/step1/step10_unified_regime_intelligence_validator.py:548
/workspace/src/training/steps/market_analysis/step1/step11_analyst_creation.py:153
/workspace/src/training/steps/market_analysis/step1/step12_analyst_enhancement.py:1687
/workspace/src/training/steps/market_analysis/step1/step15_tactician_specialist_training.py:1057
/workspace/src/training/steps/market_analysis/step1/step04_5_triple_barrier_method.py:169
/workspace/src/training/steps/market_analysis/step1/regime_specific_tpsl_optimizer.py:120
/workspace/src/training/steps/market_analysis/step1/confidence_calibration_step.py:192
/workspace/src/training/steps/market_analysis/step1/ab_testing_step.py:35
/workspace/src/training/steps/market_analysis/step1/efficiency_optimizer.py:486
/workspace/src/training/steps/market_analysis/step1/data_resampler.py:28
/workspace/src/training/steps/market_analysis/step1/data_gap_detector.py:38
/workspace/src/training/steps/market_analysis/step1/step03_parameter_optimization.py:620
/workspace/src/training/steps/market_analysis/step1/step03_5_final_regime_clustering.py:882
/workspace/src/training/steps/market_analysis/step1/step08_advanced_feature_selection_wrapper.py:46
/workspace/src/training/steps/market_analysis/step1/step02_5_sr_optimization.py:1780
/workspace/src/training/steps/market_analysis/step1/step01_data_collection.py:174
/workspace/src/training/steps/market_analysis/step1/step01_5_data_converter_wrapper.py:37
/workspace/src/training/steps/market_analysis/step1/sr_weight_optimizer.py:26
/workspace/src/training/steps/market_analysis/step1/sr_context_aware_calculator.py:424
/workspace/src/training/steps/market_analysis/step1/sr_ensemble_predictor.py:55
/workspace/src/training/steps/market_analysis/step1/sr_strength_optimizer.py:31
/workspace/src/training/steps/market_analysis/step1/sr_breakout_predictor.py:510
/workspace/src/training/steps/market_analysis/step1/sr_level_detector.py:14
/workspace/src/training/steps/market_analysis/step1/system_coordinator.py:157
```

### Long Function Issues (1 file):
```
/workspace/src/training/steps/model_training/step10_unified_regime_intelligence.py:1912
```

## 4. Files with Dead Code (2,306 issues)

### Unused Functions (1,740 issues):
```
/workspace/src/analyst/
/workspace/src/training/
/workspace/src/tactician/
/workspace/src/utils/
/workspace/src/database/
/workspace/src/strategist/
/workspace/src/supervisor/
/workspace/src/monitoring/
/workspace/src/launcher/
/workspace/src/interfaces/
/workspace/src/pipelines/
/workspace/src/integration/
/workspace/src/exchange/
```

### Unused Classes (566 issues):
```
/workspace/src/analyst/
/workspace/src/training/
/workspace/src/tactician/
/workspace/src/utils/
/workspace/src/database/
/workspace/src/strategist/
/workspace/src/supervisor/
/workspace/src/monitoring/
/workspace/src/launcher/
/workspace/src/interfaces/
/workspace/src/pipelines/
/workspace/src/integration/
/workspace/src/exchange/
```

## Priority Summary

**🔴 Critical (Fix First):**
- 126 Syntax Errors across 50+ files

**🟡 High Priority:**
- 195 Complexity Issues (deep nesting, long functions)

**🟢 Medium Priority:**
- 2,306 Dead Code Issues (unused functions/classes)

**🔵 Low Priority:**
- 1,462 Import Issues (unused imports)

**Total Files Affected:** ~400+ files
**Most Problematic Areas:** Training pipeline, market analysis, tactician modules