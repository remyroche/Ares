# Refactoring Agenda & Development Guidelines

## Core Refactoring Principles

### 1. Code Cleanup & Dead Code Removal
- **MANDATORY**: When refactoring, always delete no longer used code
- Remove unused imports, functions, classes, and variables
- Delete obsolete files and modules that are no longer referenced
- Clean up temporary files and debug code
- Maintain a clean, minimal codebase

### 2. Standardized Logging & Output
- **MANDATORY**: Use `tprint` from `src/utils/tprint.py` for all logging and print statements
- Replace all `print()` statements with appropriate `tprint` functions:
  - `tprint()` - General logging
  - `tprint_debug()` - Debug information
  - `tprint_info()` - Informational messages
  - `tprint_warning()` - Warnings
  - `tprint_error()` - Errors
  - `tprint_success()` - Success messages
  - `tprint_progress()` - Progress updates
  - `tprint_performance()` - Performance metrics
- **NEVER** swallow exceptions or allow silent failures — every caught error must emit a `tprint_error()` (or more specific severity) entry that captures full context.
- Configure `tprint` with appropriate log levels and file output
- Use structured logging with `tprint_structured()` for complex data

### 3. Hardware Optimization Tools
- **HEAVILY USE**: Tools in `src/utils/hardware/`
  - `unified_hardware_manager.py` - Central hardware management
  - `m1_optimizations.py` - M1-specific optimizations
  - `advanced_cpu_optimizer.py` - CPU optimization
  - `advanced_memory_optimizer.py` - Memory optimization
  - `enhanced_gpu_manager.py` - GPU management
- Leverage hardware-specific optimizations for better performance
- Use adaptive optimization engines for dynamic performance tuning

### 4. Matrix Operations & Computational Tools
- **HEAVILY USE**: Tools in `src/utils/matrix_operations/`
  - `unified_operations.py` - Core matrix operations
  - `vectorized_core.py` - Vectorized computations
  - `batch_operations.py` - Batch processing
  - `computation_toolbox.py` - Mathematical utilities
  - `hardware_integration.py` - Hardware-accelerated operations
- Prefer vectorized operations over loops
- Use batch processing for large datasets
- Leverage hardware acceleration when available

### 5. Data Access & Quality Tools
- **HEAVILY USE**: Data tools from:
  - `src/utils/serialization_utils.py` - Data serialization/deserialization
  - `src/utils/data/` - Comprehensive data utilities:
    - `unified_data_utils.py` - Core data operations
    - `quality/` - Data quality validation and cleaning
    - `processing/` - Data processing pipelines
    - `validation/` - Data validation utilities
  - `src/utils/data_loader.py` - Data loading utilities
  - `src/utils/parquet_utils.py` - Parquet file operations
- Implement data quality checks before processing
- Use proper serialization for data persistence
- Validate data integrity at each processing step

### 6. Machine Learning Tools
- **HEAVILY USE**: ML tools from:
  - `src/utils/ml_common/` - Core ML utilities:
    - `common_operations.py` - ML-specific operations
    - `confidence_metrics.py` - Confidence scoring
    - `data_drift_detector.py` - Data drift detection
    - `feature_selection.py` - Feature selection algorithms
    - `validation/` - ML validation utilities
    - `training/` - Training utilities
    - `evaluation/` - Model evaluation
  - `src/utils/nas_tas/` - Neural Architecture Search & Trading Algorithm Search:
    - `bayesian_tpe_optimizer.py` - Bayesian optimization
    - `evolutionary_search.py` - Evolutionary algorithms
    - `ensemble_optimizer.py` - Ensemble methods
    - `model_manager.py` - Model lifecycle management
    - `performance_tracker.py` - Performance monitoring
    - `risk_analysis/` - Risk analysis tools
- Use standardized ML pipelines and validation frameworks
- Implement proper model versioning and management
- Apply appropriate optimization algorithms for hyperparameter tuning

## Additional Useful Items to Add to Agents.md

### 7. Performance Monitoring & Optimization
- Use `src/utils/performance.py` and `src/utils/performance_utils.py` for performance tracking
- Implement `src/utils/parallel_processing_optimizer.py` for parallel execution
- Monitor memory usage with `src/utils/memory_management/`
- Use `src/utils/prometheus_metrics.py` for metrics collection

### 8. Error Handling & Recovery
- Implement robust error handling using `src/utils/error_handler.py`
- Use `src/utils/graceful_module_handler.py` for module-level error recovery
- Apply `src/utils/fallback_monitoring.py` for system resilience
- Leverage `src/utils/error_recovery/` for advanced error recovery patterns
- **NO SILENT FAILURES**: Always log detected issues with the appropriate `tprint_*` helper and propagate actionable context.
- **FAST-FAIL PREFERRED**: When operating interactive agent flows, prefer immediate failure over degraded fallbacks. Disable fallback/retry paths (for example, set `enable_fallback_mode=False` in `NASTASClusteringConfig`) unless the user explicitly requests resilience testing.

### 9. Validation & Quality Assurance
- Use `src/utils/validation/` for comprehensive validation frameworks
- Implement `src/utils/step_validation_system.py` for pipeline validation
- Apply `src/utils/math_validation.py` for mathematical validation
- Use `src/utils/lookahead_bias_detector.py` for bias detection

### 10. Configuration & Dependency Management
- Use `src/utils/dependency_injection.py` for dependency management
- Implement `src/utils/config/` for configuration management
- Apply `src/utils/service_discovery.py` for service discovery
- Use `src/utils/version_manager.py` for version control

### 11. Monitoring & Observability
- Implement `src/utils/observability.py` for system observability
- Use `src/utils/tracing.py` for distributed tracing
- Apply `src/utils/monitoring_utils.py` for monitoring utilities
- Use `src/utils/structured_logging.py` for structured logging

### 12. Caching & State Management (Hardware-Focused)
- **PREFER HARDWARE/**: Use `src/utils/hardware/` tools for caching and state management:
  - `unified_hardware_manager.py` - Hardware-aware caching strategies
  - `advanced_memory_optimizer.py` - Memory-based caching optimization
  - `m1_memory_optimizer.py` - M1-specific memory and caching optimizations
- Use `src/utils/caching.py` and `src/utils/unified_cache.py` as fallbacks
- Implement `src/utils/state_manager.py` for state management
- Use `src/utils/artifact_manager.py` for artifact management
- Apply hardware-aware caching strategies for optimal performance

### 13. Testing & Quality Assurance
- Implement comprehensive testing using `src/utils/testing/`
- Use `src/utils/purged_kfold.py` for time series cross-validation
- Apply proper test coverage and quality metrics
- Implement automated testing pipelines

### 14. Documentation & Reporting
- Use `src/utils/report_manager.py` and `src/utils/report_collector.py` for reporting
- Implement `src/utils/explainability/` for model explainability
- Generate comprehensive documentation for all modules
- Maintain up-to-date API documentation

## Implementation Checklist

When refactoring any module, ensure:

- [ ] All `print()` statements replaced with `tprint` functions
- [ ] Dead code and unused imports removed
- [ ] Hardware optimization tools integrated where applicable
- [ ] Matrix operations use vectorized implementations
- [ ] Data access uses proper serialization and quality tools
- [ ] ML operations use standardized ml_common and nas_tas tools
- [ ] Error handling is robust and comprehensive
- [ ] Performance monitoring is implemented
- [ ] Validation frameworks are applied
- [ ] Documentation is updated
- [ ] Tests are added or updated
- [ ] Configuration is properly managed
- [ ] Hardware-optimized caching strategies are implemented
- [ ] Observability is enhanced

## Code Quality Standards

- Follow PEP 8 style guidelines
- Use type hints consistently
- Implement proper docstrings for all functions and classes
- Maintain test coverage above 80%
- Use meaningful variable and function names
- Implement proper error messages and logging
- Follow the DRY (Don't Repeat Yourself) principle
- Use design patterns appropriately
- Implement proper resource cleanup

## Performance Targets

- Optimize for both CPU and memory efficiency
- Use hardware acceleration when available
- Implement parallel processing where beneficial
- Minimize I/O operations
- Use efficient data structures
- Implement proper caching strategies
- Monitor and optimize hot paths
- Use profiling tools to identify bottlenecks

This agenda ensures consistent, high-quality, and maintainable code across the entire codebase while leveraging the powerful utility infrastructure already in place.