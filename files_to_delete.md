# Files to Delete After Migration

## Summary
Based on my analysis, I have:
1. ✅ Analyzed the new decorators in src/core/decorators/
2. ✅ Analyzed the new error handlers in src/core/errors/
3. ✅ Found all decorator and error handler usage throughout the codebase
4. ✅ Updated some files to use the new decorators (e.g., step3_5_final_regime_clustering.py)
5. ✅ Created a migration mapping guide
6. ✅ Identified files that still need updating (17 files in src/ directory)

## Old Decorator Files to Delete (in src/utils/):

### Core decorator files:
1. `src/utils/centralized_decorators.py` - Old centralized decorators (1096 lines)
2. `src/utils/centralized_decorators_v2.py` - Version 2 of centralized decorators (655 lines)
3. `src/utils/training_pipeline_decorators.py` - Pipeline-specific decorators (1470 lines)
4. `src/utils/advanced_decorators.py` - Advanced decorators (508 lines)
5. `src/utils/enhanced_decorators.py` - Enhanced decorators (463 lines)
6. `src/utils/enhanced_pipeline_decorators.py` - Enhanced pipeline decorators (627 lines)
7. `src/utils/enhanced_validation_decorators.py` - Enhanced validation decorators (734 lines)
8. `src/utils/validation_decorators.py` - Validation decorators (484 lines)
9. `src/utils/vif_validation_decorators.py` - VIF validation decorators (465 lines)

### Error handler files:
10. `src/utils/error_handler.py` - Main error handler (1732 lines)
11. `src/utils/enhanced_error_handler.py` - Enhanced error handler (652 lines)
12. `src/utils/enhanced_error_handling.py` - Enhanced error handling (367 lines)
13. `src/utils/standardized_error_handler.py` - Standardized error handler (367 lines)
14. `src/utils/domain_errors.py` - Domain-specific errors (76 lines)

### Files to keep (domain-specific):
- `src/utils/decorators.py` - Contains domain-specific decorators like `@auto_vectorize` that don't have equivalents in core

## Files Still Needing Updates:
1. src/training/feature_engineering.py
2. src/training/core/pipeline_orchestrator.py
3. src/training/core/stage_context.py
4. src/training/core/checkpoint_manager.py
5. src/tactician/position_division_strategy.py
6. src/strategist/strategist_backup.py
7. src/pipelines/base_pipeline.py
8. src/pipelines/live_trading_pipeline.py
9. src/pipelines/components/data_manager.py
10. src/pipelines/components/lifecycle_manager.py
11. src/pipelines/components/monitoring_manager.py
12. src/interfaces/enhanced_event_bus.py
13. src/exchange/binance.py
14. src/analyst/liquidation_risk_model.py
15. src/analyst/enhanced_prediction_integrator.py
16. src/utils/centralized_decorators.py (self-imports)
17. src/utils/decorators.py (imports domain_errors)

## Total Impact:
- **14 files to delete** (approximately 8,500+ lines of code)
- **17 files still need import updates**
- Many syntax_fix_backups files also need updates but are lower priority

## Recommendation:
1. First update the remaining 17 files to use new imports
2. Run tests to ensure everything works
3. Then delete the old decorator and error handler files
4. Update any remaining syntax_fix_backups files as needed