# Syntax Error Analysis Report

Total files with syntax errors: 120

## Error Categories:

### Indentation (62 files)
- src/analyst/enhanced_prediction_integrator.py
- src/analyst/meta_label_relevance.py
- src/analyst/regime_expert_orchestrator.py
- src/database/migration_utils.py
- src/exchange/binance.py
- src/launcher/enhanced_trading_launcher.py
- src/pipelines/components/data_manager.py
- src/pipelines/components/lifecycle_manager.py
- src/pipelines/improved_pipeline_executor.py
- src/supervisor/system_coordinator_backup.py
- ... and 52 more

### Indentation Mismatch (4 files)
- src/training/multi_output_probability_trainer.py
- src/training/steps/step1/data_resampler.py
- src/training/steps/step10_unified_regime_intelligence_validator.py
- src/training/steps/step11_analyst_creation.py

### Invalid Syntax (21 files)
- src/analyst/liquidation_risk_model.py
- src/analyst/predictive_ensembles/multi_timeframe_ensemble.py
- src/analyst/unified_regime_classifier.py
- src/database/precomputed_features_manager.py
- src/integration/paper_trading_integration.py
- src/interfaces/enhanced_event_bus.py
- src/strategist/strategist_backup.py
- src/training/core/checkpoint_manager.py
- src/training/core/pipeline_orchestrator.py
- src/training/optimized_feature_selection_manager.py
- ... and 11 more

### Missing Block (16 files)
- src/analyst/autoencoder_feature_generator.py
- src/training/enhanced_training_manager.py
- src/training/enhanced_training_manager_optimized.py
- src/training/feature_integration.py
- src/training/model_trainer.py
- src/training/multi_output_model_trainer.py
- src/training/steps/enhanced_step1_5_data_converter.py
- src/training/steps/enhanced_step1_data_collection.py
- src/training/steps/raw_data_quality_checker.py
- src/training/steps/step05_labeling.py
- ... and 6 more

### Other (3 files)
- src/pipelines/live_trading_pipeline.py
- src/training/steps/step1/missing_data_downloader_and_gap_filler.py
- src/training/steps/step17_final_parameters_optimization/optimized_optuna_optimization_enhanced.py

### Unmatched Brackets (12 files)
- src/analyst/ml_confidence_predictor.py
- src/tactician/position_division_strategy.py
- src/training/steps/fractional_differentiation.py
- src/training/steps/step04_5_triple_barrier_method.py
- src/training/steps/step1/data_quality_dashboard.py
- src/training/steps/step10_unified_regime_intelligence.py
- src/training/steps/step12_analyst_enhancement.py
- src/training/steps/step15_tactician_specialist_training.py
- src/training/steps/step18_walk_forward_validation.py
- src/training/steps/step19_monte_carlo_validation.py
- ... and 2 more

### Unterminated String (2 files)
- src/training/data_efficiency_optimizer.py
- src/training/data_manager.py

## Priority Files to Fix:

These are critical files that should be fixed first:

### src/exchange/binance.py
- Line 14: unexpected indent

### src/training/training_manager.py
- Line 205: unexpected indent

### src/training/model_trainer.py
- Line 322: expected an indented block after 'try' statement on line 321

### src/utils/model_manager.py
- Line 205: invalid syntax

### src/training/steps/step1/missing_data_downloader_and_gap_filler.py
- Line 0: Syntax error after auto-fixing