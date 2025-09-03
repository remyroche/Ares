# Decorator and Error Handler Migration Mapping

## Most Common Decorators to Migrate

### From src/utils/centralized_decorators.py:
- `@handle_errors` → `@handles_errors` (from src.core.decorators)
- `@handle_specific_errors` → `@handles_errors` with specific error types
- `@comprehensive_data_validation` → `@validates` (from src.core.decorators)
- `@validate_data_quality` → `@validate_dataframe` (from src.core.decorators)
- `@guard_dataframe_nulls` → `@validate_dataframe` with null checks
- `@with_tracing_span` → `@traced` (from src.core.decorators)
- `@performance_monitor` → `@log_execution_time` (from src.core.decorators)
- `@secure_data_processing` → Combination of `@validates` and `@error_boundary`

### From src/utils/training_pipeline_decorators.py:
- `@validate_step_prerequisites` → `@validates` with prerequisite checks
- `@validate_pipeline_step` → `@validates` with pipeline validation
- `@ensure_data_integrity` → `@validate_dataframe` or `@validate_schema`
- `@artifact_versioning` → Custom implementation with `@traced`
- `@artifact_write_lock` → Custom implementation with locks

### From src/utils/decorators.py:
- `@auto_vectorize` → Keep as is (domain-specific)
- `@enforce_ndarray` → `@validates` with numpy array type checking
- `@guard_array_nan_inf` → `@validates` with NaN/Inf checks
- `@pa_check_input/output` → `@validate_schema` with pandera schemas

### From src/utils/advanced_decorators.py:
- `@performance_monitor` → `@log_execution_time`
- `@intelligent_caching` → `@cached` (from src.core.decorators)
- `@adaptive_resource_allocation` → Custom implementation with `@error_boundary`

## Error Handler Migration

### From src/utils/error_handler.py:
- `handle_errors()` function → Use `@handles_errors` decorator
- `handle_specific_errors()` function → Use `@handles_errors` with specific error types
- Domain-specific errors → Import from `src.core.errors` or create custom errors extending `AppError`

### From src/utils/domain_errors.py:
- `DataValidationError` → `ValidationError` (from src.core.errors)
- `DomainError` → `BusinessRuleError` (from src.core.errors)
- `ExternalServiceError` → `ServiceUnavailableError` (from src.core.errors)
- `NotFoundError` → `NotFoundError` (from src.core.errors)
- `OperationTimeoutError` → `TimeoutError` (from src.core.errors)

## Files to Delete After Migration

### Decorator Files:
- src/utils/centralized_decorators.py
- src/utils/centralized_decorators_v2.py
- src/utils/training_pipeline_decorators.py
- src/utils/advanced_decorators.py
- src/utils/enhanced_decorators.py
- src/utils/enhanced_pipeline_decorators.py
- src/utils/enhanced_validation_decorators.py
- src/utils/validation_decorators.py
- src/utils/vif_validation_decorators.py

### Error Handler Files:
- src/utils/error_handler.py
- src/utils/enhanced_error_handler.py
- src/utils/enhanced_error_handling.py
- src/utils/standardized_error_handler.py
- src/utils/domain_errors.py

### Keep These (Domain-Specific):
- src/utils/decorators.py (contains domain-specific decorators like auto_vectorize)