# Decorator Migration Report

## Summary

We have successfully migrated the codebase from old decorators to the new core decorator system from `src.core.decorators` and `src.core.errors`.

## Migration Statistics

- **Total files updated**: ~240 files
- **Directories updated**:
  - `src/training/steps/` - 102 files
  - `src/monitoring/` - 8 files  
  - `src/pipelines/` - 6 files
  - `src/tactician/` - 19 files
  - `src/training/` (excluding steps) - 57 files
  - `src/utils/` - 3 files

## Decorator Mappings Applied

### Direct Mappings
- `handle_errors` → `handles_errors`
- `handle_file_operations` → `handles_errors`
- `handle_specific_errors` → `handles_errors`
- `validate_step_prerequisites` → `validates`
- `validate_step_output` → `validates`
- `validate_dataframe_schema` → `validate_dataframe`
- `guard_dataframe_nulls` → `validates`
- `circuit_breaker_protection` → `circuit_breaker`
- `debug_training_step` → `log_call`
- `resource_monitor` → `log_execution_time`
- `intelligent_caching` → `cached`
- `with_tracing_span` → `traced`

## Custom Decorators Requiring Manual Migration

The following domain-specific decorators were preserved with TODO comments and still need manual migration:

### Validation Decorators
- `validate_data_quality`
- `validate_feature_engineering_with_lookahead_bias_detection`
- `validate_klines_data_quality`
- `validate_multi_timeframe_data_quality`
- `validate_ohlcv_data_quality`
- `validate_wavelet_data_quality`
- `validate_hmm_data_requirements`
- `validate_hmm_regime_discovery`
- Various step-specific validators (`validate_step2_operation`, etc.)

### Security & Processing Decorators
- `secure_data_processing`
- `prevent_data_leakage`
- `secure_step_execution`
- `ensure_data_integrity`

### Monitoring & Performance Decorators
- `monitor_step_execution`
- `monitor_feature_engineering`
- `monitor_pipeline_performance`
- `performance_monitor`
- `quality_gate`

### Other Domain-Specific Decorators
- `artifact_versioning`
- `deterministic_seed`
- `idempotent_step`
- `time_budget_watchdog`
- `smart_validation_cache`

## Next Steps

1. **Review Changes**: Carefully review all modified files to ensure decorators are correctly applied
2. **Test Thoroughly**: Run comprehensive tests to verify functionality hasn't been affected
3. **Migrate Custom Decorators**: Gradually migrate the domain-specific decorators to use the core system
4. **Clean Up**: Once migration is complete and tested, remove old decorator modules from `src/utils/`

## Files Still Using Old Decorators

All files now have a mix of new core decorators and old custom decorators marked with TODO comments. The old imports are preserved only for the custom decorators that don't have direct equivalents in the core system.

## Recommendations

1. Create wrapper decorators in the core system for common patterns like:
   - Data quality validation with specific parameters
   - Security and data leakage prevention
   - Step-specific monitoring and execution tracking

2. Consider adding these to core decorators:
   - A generic `@monitor` decorator for performance tracking
   - A `@secure` decorator for security-sensitive operations
   - Domain-specific validators that can be configured

3. Gradually phase out old decorators by:
   - Implementing equivalent functionality in core decorators
   - Creating composed decorators using the core `compose` utility
   - Updating code to use new patterns