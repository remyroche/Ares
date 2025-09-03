# Complete List of Files with Syntax Errors

Total: **120 files** (6 already fixed, 114 remaining)

## src/analyst/ (9 files)
1. autoencoder_feature_generator.py
2. enhanced_prediction_integrator.py
3. liquidation_risk_model.py
4. meta_label_relevance.py
5. ml_confidence_predictor.py
6. regime_expert_orchestrator.py
7. unified_regime_classifier.py
8. **predictive_ensembles/**
   - multi_timeframe_ensemble.py

## src/database/ (2 files)
1. migration_utils.py
2. precomputed_features_manager.py

## src/exchange/ (1 file)
1. binance.py ✅ **FIXED**

## src/integration/ (1 file)
1. paper_trading_integration.py

## src/interfaces/ (1 file)
1. enhanced_event_bus.py

## src/launcher/ (1 file)
1. enhanced_trading_launcher.py

## src/pipelines/ (3 files)
1. improved_pipeline_executor.py
2. live_trading_pipeline.py
3. **components/**
   - data_manager.py
   - lifecycle_manager.py

## src/strategist/ (1 file)
1. strategist_backup.py

## src/supervisor/ (1 file)
1. system_coordinator_backup.py

## src/tactician/ (3 files)
1. enhanced_execution_manager.py
2. position_division_strategy.py
3. tactician.py

## src/training/ (29 files)
1. calibration_manager.py
2. data_efficiency_optimizer.py
3. data_manager.py
4. di_training_manager.py
5. enhanced_coarse_optimizer.py
6. enhanced_lm_optimizer.py
7. enhanced_matrix_operations.py
8. enhanced_training_manager.py
9. enhanced_training_manager_optimized.py
10. factory.py
11. feature_integration.py
12. model_trainer.py ✅ **FIXED**
13. multi_objective_optimizer.py
14. multi_output_model_trainer.py
15. multi_output_probability_trainer.py
16. optimized_feature_selection_manager.py
17. training_manager.py ✅ **FIXED**
18. training_orchestrator.py
19. wavelet_caching_workflow.py
20. **core/**
    - checkpoint_manager.py
    - pipeline_orchestrator.py
21. **optimization/**
    - computational_optimization_manager.py

## src/training/steps/ (60 files)

### Root level (33 files)
1. combined_fractional_system.py
2. enhanced_step1_5_data_converter.py
3. enhanced_step1_data_collection.py
4. fractional_differentiation.py
5. fractional_feature_selector.py
6. integrated_data_quality_pipeline.py
7. precompute_wavelet_features.py
8. raw_data_quality_checker.py
9. step02_5_sr_optimization.py ✅ **FIXED**
10. step02_5_sr_optimization_validator.py
11. step02_data_reading_validator.py
12. step03_5_final_regime_clustering.py
13. step03_5_final_regime_clustering_validator.py
14. step03_hmm_regime_discovery.py
15. step03_parameter_optimization.py
16. step03_parameter_optimization_validator.py
17. step04_5_triple_barrier_method.py
18. step04_5_triple_barrier_method_validator.py
19. step04_regime_data_splitting.py
20. step04_regime_data_splitting_validator.py
21. step05_labeling.py
22. step05_labeling_validator.py
23. step06_feature_engineering_validator.py
24. step07_enhanced_matrix_operations.py
25. step09_5_hmm_lm_generalist_training.py
26. step09_5_multi_timeframe_hmm_ensemble.py
27. step09_hmm_based_training.py
28. step09_hmm_based_training_validator.py
29. step10_unified_regime_intelligence.py
30. step10_unified_regime_intelligence_validator.py
31. step11_analyst_creation.py
32. step12_analyst_enhancement.py
33. step13_analyst_ensemble_creation.py
34. step14_tactician_labeling.py
35. step14_tactician_labeling_validator.py
36. step15_tactician_specialist_training.py
37. step16_confidence_calibration.py
38. step17_final_parameters_optimization.py
39. step18_walk_forward_validation.py
40. step18_walk_forward_validation_validator.py
41. step19_monte_carlo_validation.py
42. step19_monte_carlo_validation_validator.py
43. step21_saving.py
44. unified_data_loader.py

### analyst_training_components/ (1 file)
45. regime_specific_tpsl_optimizer.py

### data_preparation_components/ (1 file)
46. training_validation_config.py

### step1/ (7 files)
47. comprehensive_gap_filler.py
48. data_gap_detector.py
49. data_quality_dashboard.py
50. data_quality_monitor.py
51. data_resampler.py
52. enhanced_data_quality_manager.py
53. missing_data_downloader_and_gap_filler.py ✅ **FIXED**

### step4_analyst_labeling_feature_engineering_components/ (4 files)
54. fractional_triple_barrier_labeling.py
55. optimized_triple_barrier_labeling.py
56. profit_based_feature_engineering.py
57. regime_aware_triple_barrier_labeling.py

### step17_final_parameters_optimization/ (7 files)
58. advanced_optimization_engine.py
59. comprehensive_parameter_integration.py
60. efficiency_optimizer.py
61. optimized_optuna_optimization_enhanced.py
62. optimized_step17_implementation.py
63. regime_specific_triple_barrier_optimization.py
64. step17_probabilistic_bayesian_optimization.py

## src/utils/ (11 files)
1. base_validator.py
2. configuration_security.py
3. data_formatting_framework.py
4. data_optimizer.py
5. database_security.py
6. enhanced_memory_management.py
7. enhanced_missing_value_handler.py
8. model_manager.py ✅ **FIXED**
9. model_performance_monitor.py
10. quality_alert_system.py
11. vif_calculator.py

## Summary by Directory

| Directory | Files with Errors | Fixed | Remaining |
|-----------|------------------|-------|-----------|
| src/analyst/ | 9 | 0 | 9 |
| src/database/ | 2 | 0 | 2 |
| src/exchange/ | 1 | 1 | 0 |
| src/integration/ | 1 | 0 | 1 |
| src/interfaces/ | 1 | 0 | 1 |
| src/launcher/ | 1 | 0 | 1 |
| src/pipelines/ | 5 | 0 | 5 |
| src/strategist/ | 1 | 0 | 1 |
| src/supervisor/ | 1 | 0 | 1 |
| src/tactician/ | 3 | 0 | 3 |
| src/training/ | 29 | 2 | 27 |
| src/training/steps/ | 60 | 2 | 58 |
| src/utils/ | 11 | 1 | 10 |
| **TOTAL** | **120** | **6** | **114** |

## Files Already Fixed ✅
1. src/exchange/binance.py
2. src/training/training_manager.py
3. src/training/model_trainer.py
4. src/utils/model_manager.py
5. src/training/steps/step02_5_sr_optimization.py
6. src/training/steps/step1/missing_data_downloader_and_gap_filler.py