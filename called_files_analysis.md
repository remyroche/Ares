# Files Called by ares_launcher.py - Comprehensive Analysis

## Files Explicitly Called/Imported by ares_launcher.py

### Direct Imports in ares_launcher.py
1. `src.config.CONFIG` - Configuration module
2. `src.config.training_modes` - Training modes configuration
3. `src.utils.comprehensive_logger` - Comprehensive logging
4. `src.utils.error_handler` - Error handling utilities
5. `src.utils.logger` - Logger utilities
6. `src.utils.signal_handler` - Signal handling
7. `src.utils.observability` - Observability setup
8. `src.database.sqlite_manager` - Database manager
9. `src.training.enhanced_training_manager` - Enhanced training manager
10. `src.training.steps.precompute_wavelet_features` - Wavelet feature precomputation
11. `src.analyst.data_utils` - Data utilities
12. `src.analyst.unified_regime_classifier` - Unified regime classifier
13. `src.utils.validator_orchestrator` - Validator orchestrator
14. `src.utils.step_dependency_validator` - Step dependency validator
15. `src.training.step_orchestrator` - Step orchestrator

### Subprocess Executions in ares_launcher.py
1. `GUI/start.sh` - GUI startup script
2. `GUI/api_server.py` - GUI API server (fallback)
3. `src/supervisor/global_portfolio_manager.py` - Global portfolio manager
4. `src/ares_pipeline.py` - Main trading pipeline
5. `scripts/setup_challenger_model.py` - Challenger model setup
6. `src/training/steps/step1_data_collection.py` - Data collection step

### Missing Files Referenced in ares_launcher.py
1. `src/training/steps/backtesting_with_cached_features.py` - **MISSING**
2. `scripts/run_multi_timeframe_training.py` - **MISSING**
3. `scripts/blank_training_run.py` - **MISSING**
4. `backtesting/ares_data_downloader_optimized.py` - **MISSING**

## Files Called by Direct Imports (Recursive Analysis)

### From src/config.py
1. `src/config/modular_config.py`
2. `src/config/environment.py`
3. `src/config/system.py`
4. `src/config/trading.py`
5. `src/config/training.py`
6. `src/config/validation.py`

### From src/config/training_modes.py
- No additional imports found

### From src/utils/comprehensive_logger.py
1. `src.utils.structured_logging`
2. `src.utils.warning_symbols`

### From src/utils/error_handler.py
1. `src.utils.logger` (lazy import)
2. `src.utils.warning_symbols`

### From src/utils/logger.py
1. `src.utils.pipeline_standards`
2. `src.utils.structured_logging`
3. `src.utils.warning_symbols`

### From src/utils/signal_handler.py
1. `src.utils.error_handler`
2. `src.utils.logger`
3. `src.utils.warning_symbols`

### From src/utils/observability.py
1. `src.utils.warning_symbols`
2. `sentry_sdk` (external)
3. `opentelemetry` (external)

### From src/supervisor/global_portfolio_manager.py
1. `src.utils.logger`
2. `src.utils.error_handler`
3. `src.utils.warning_symbols`
4. `src.utils.supervisor_error_handler`

### From src/database/sqlite_manager.py
1. `src.utils.logger`
2. `src.config.constants`
3. `src.utils.error_handler`
4. `src.utils.warning_symbols`

### From src/training/enhanced_training_manager.py
1. `src.config.computational_optimization`
2. `src.training.enhanced_training_manager_optimized`
3. `src.training.optimization.computational_optimization_manager`
4. `src.training.steps.multi_timeframe_training.multi_timeframe_training_manager`
5. `src.utils.model_performance_monitor`
6. `src.utils.error_handler`
7. `src.utils.training_pipeline_decorators`
8. `src.utils.logger`
9. `src.utils.step_dependency_validator`
10. `src.utils.validator_orchestrator`

### From src/training/steps/precompute_wavelet_features.py
1. `src.training.steps.vectorized_advanced_feature_engineering`
2. `src.utils.data_optimizer`
3. `src.utils.centralized_decorators`
4. `src.utils.logger`
5. `src.utils.warning_symbols`

### From src/ares_pipeline.py
1. `src.analyst.analyst`
2. `src.config.environment`
3. `src.database.sqlite_manager`
4. `src.interfaces.event_bus`
5. `src.strategist.strategist`
6. `src.supervisor.supervisor`
7. `src.tactician.tactician`
8. `src.utils.state_manager`
9. `src.config`
10. `src.interfaces.base_interfaces`
11. `src.utils.observability`
12. `src.monitoring.performance_dashboard`
13. `src.monitoring.performance_monitor`
14. `src.utils.error_handler`
15. `src.utils.warning_symbols`
16. `src.utils.logger`
17. `src.core.dependency_injection`
18. `src.monitoring.dual_model_system`
19. `src.core.config_service`

### From src/analyst/data_utils.py
1. `src.utils.error_handler`
2. `src.utils.logger`
3. `src.utils.warning_symbols`

### From src/analyst/unified_regime_classifier.py
1. `src.config.CONFIG`
2. `src.tactician.sr_breakout_predictor`
3. `src.utils.logger`
4. `src.utils.error_handler`
5. `src.utils.warning_symbols`
6. `src.utils.centralized_decorators_simple`

### From src/utils/validator_orchestrator.py
1. `src.utils.logger`
2. `src.utils.pipeline_standards`
3. `src.utils.prometheus_metrics`
4. `src.utils.warning_symbols`

### From src/utils/step_dependency_validator.py
1. `src.utils.logger`
2. `src.utils.pipeline_standards`
3. `src.utils.warning_symbols`

### From src/training/step_orchestrator.py
1. `src.training.progress_manager`
2. `src.utils.logger`
3. `src.utils.warning_symbols`
4. `src.training.enhanced_training_manager`

## Additional Configuration Files Found
1. `src/config/constants.py`
2. `src/config/environment.py`
3. `src/config/system.py`
4. `src/config/trading.py`
5. `src/config/training.py`
6. `src/config/validation.py`
7. `src/config/computational_optimization.py`
8. `src/config/enhanced_reporting_config.py`
9. `src/config/typed_config.py`
10. `src/config/label_model_mapping.py`
11. `src/config/multi_timeframe_hmm_ensemble_config.py`
12. `src/config/computational_optimization_config.py`
13. `src/config/enhanced_feature_selection_config.py`
14. `src/config/enhanced_multi_timeframe_config.py`
15. `src/config/enhanced_matrix_config.py`
16. `src/config/enhanced_prediction_service_config.py`

## Summary of Called Files
This analysis identifies the files that are explicitly called or imported by `ares_launcher.py` and its direct dependencies. The list includes:

- **Direct imports**: 15 files
- **Subprocess executions**: 6 files (4 missing)
- **Recursive imports**: ~50+ files from direct dependencies
- **Configuration files**: 16 files

## Missing Files
The following files are referenced in `ares_launcher.py` but do not exist in the workspace:
1. `src/training/steps/backtesting_with_cached_features.py`
2. `scripts/run_multi_timeframe_training.py`
3. `scripts/blank_training_run.py`
4. `backtesting/ares_data_downloader_optimized.py`

## Next Steps
To complete the analysis, we need to:
1. Compare this list against all Python files in the workspace (593 total)
2. Identify files that are NOT in the called files list
3. Provide a comprehensive list of unused files