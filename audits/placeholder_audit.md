# Placeholder and TODO Audit

This report summarizes pass statements, silent failures, and placeholder markers across the repository.

## Summary Totals

| Category | Count |
| --- | ---: |
| Pass | 235 |
| Silent Failure | 71 |
| Todo | 11 |
| Fixme | 8 |
| Placeholder | 296 |
| Mock | 232 |
| Stub | 107 |

## Top files by pass occurrences

| File | Count |
| --- | ---: |
| `src/utils/nas_tas/nas/neural_architecture_search.py` | 3 |
| `src/feature_generation/utils/cross_timeframe_analysis_pipeline.py` | 3 |
| `src/core/decorators/retry_timeout.py` | 3 |
| `src/core/errors/mapping.py` | 3 |
| `src/training/base_step.py` | 3 |
| `src/training/steps/data_collection/decorators/step_decorators.py` | 3 |
| `src/training/steps/data_collection/data_preparation_components/data_format_converter.py` | 3 |
| `src/training/steps/model_training/tactician_ensemble_training.py` | 3 |
| `code_quality/analyzers/enhanced_dead_code_analyzer.py` | 2 |
| `code_quality/analyzers/enhanced_security_analyzer.py` | 2 |
| `code_quality/analyzers/test_coverage_analyzer.py` | 2 |
| `code_quality/pipelines/function_import_analysis_pipeline.py` | 2 |
| `src/trading/utils/signal_enhancer_base.py` | 2 |
| `src/exchange/binance.py` | 2 |
| `src/analyst/analyst.py` | 2 |
| `src/utils/nas_testing.py` | 2 |
| `src/utils/regime_aware_financial_logging_decorator.py` | 2 |
| `src/utils/regime_data_access.py` | 2 |
| `src/utils/math_validation.py` | 2 |
| `src/utils/nas_tas/uncertainty_estimation.py` | 2 |
| `src/utils/nas_tas/evolutionary_search.py` | 2 |
| `src/utils/nas_tas/confidence_scoring.py` | 2 |
| `src/utils/nas_tas/tas/unsupervised_tree_nas.py` | 2 |
| `src/utils/data/processing/data_processing.py` | 2 |
| `src/utils/data/quality/data_qualification_imports.py` | 2 |

## Top files by silent failure occurrences

| File | Count |
| --- | ---: |
| `src/training/steps/data_collection/data_preparation_components/data_format_converter.py` | 3 |
| `code_quality/analyzers/test_coverage_analyzer.py` | 2 |
| `src/utils/nas_testing.py` | 2 |
| `src/utils/regime_data_access.py` | 2 |
| `src/utils/nas_tas/nas/neural_architecture_search.py` | 2 |
| `src/utils/data/processing/data_processing.py` | 2 |
| `src/utils/hardware/advanced_cpu_optimizer.py` | 2 |
| `src/research/cluster_analysis/economic_relevance/pattern_dimension_analysis.py` | 2 |
| `src/core/errors/mapping.py` | 2 |
| `src/training/steps/backtesting/abc_testing/multi_model_orchestrator.py` | 2 |
| `src/training/steps/backtesting/abc_testing/performance_monitoring.py` | 2 |
| `code_quality/analyzers/static_analysis_analyzer.py` | 1 |
| `code_quality/analyzers/enhanced_dead_code_analyzer.py` | 1 |
| `code_quality/scripts/safe_indentation_fixer.py` | 1 |
| `src/trading/cross_asset/cross_asset_trading_manager.py` | 1 |
| `src/analyst/analyst.py` | 1 |
| `src/analyst/autoencoder_feature_generator.py` | 1 |
| `src/utils/enhanced_financial_metrics_logger.py` | 1 |
| `src/utils/nas_logging.py` | 1 |
| `src/utils/validator_orchestrator.py` | 1 |
| `src/utils/feature_engineering_validation.py` | 1 |
| `src/utils/prometheus_metrics.py` | 1 |
| `src/utils/ml_common/feature_selection.py` | 1 |
| `src/utils/ml_common/utils/memory_optimization.py` | 1 |
| `src/utils/ml_common/validation/cv_utils.py` | 1 |

## Top files by todo occurrences

| File | Count |
| --- | ---: |
| `scripts/generate_placeholder_audit.py` | 4 |
| `code_quality/analyzers/stub_object_analyzer.py` | 2 |
| `scripts/generate_pass_silent_unimplemented_report.py` | 2 |
| `fix_tas_issues.py` | 1 |
| `code_quality/analyzers/configuration_analyzer.py` | 1 |
| `code_quality/analyzers/documentation_analyzer.py` | 1 |

## Top files by fixme occurrences

| File | Count |
| --- | ---: |
| `scripts/generate_placeholder_audit.py` | 3 |
| `scripts/generate_pass_silent_unimplemented_report.py` | 2 |
| `fix_tas_issues.py` | 1 |
| `code_quality/analyzers/stub_object_analyzer.py` | 1 |
| `code_quality/analyzers/documentation_analyzer.py` | 1 |

## Top files by placeholder occurrences

| File | Count |
| --- | ---: |
| `code_quality/analyzers/stub_object_analyzer.py` | 16 |
| `src/training/steps/market_analysis/regime_data_splitting/main.py` | 13 |
| `scripts/generate_placeholder_audit.py` | 10 |
| `src/training/steps/market_analysis/nas_clustering/core/essential_nas_clusterer.py` | 8 |
| `code_quality/analyzers/configuration_analyzer.py` | 7 |
| `code_quality/analyzers/secrets_analyzer.py` | 7 |
| `src/training/steps/market_analysis/feature_lookback_optimization/feature_lookback_optimization_modular.py` | 7 |
| `src/trading/execution/order_manager.py` | 6 |
| `src/training/steps/market_analysis/pid_based_feature_generation/pid_based_feature_orchestrator.py` | 6 |
| `src/analyst/ml_dynamic_target_predictor.py` | 5 |
| `src/utils/decorators.py` | 5 |
| `src/utils/ml_common/reporting/enhanced_reporting_system.py` | 5 |
| `src/research/cluster_analysis/market_factor_analysis/dimension_discovery.py` | 5 |
| `src/research/clusters/dimension_analyzer.py` | 5 |
| `src/training/steps/backtesting/abc_testing/multi_model_orchestrator.py` | 5 |
| `src/training/steps/market_analysis/hybrid_nas_tas_regime/hybrid_orchestrator.py` | 5 |
| `GUI/src/components/Backtesting.jsx` | 5 |
| `scripts/generate_pass_silent_unimplemented_report.py` | 5 |
| `src/utils/nas_tas/shared_serialization.py` | 4 |
| `src/training/steps/model_training/tactician_lookback_optimization.py` | 4 |
| `GUI/src/App.js` | 4 |
| `GUI/src/components/BotManagement.jsx` | 4 |
| `src/utils/nas_tas/uncertainty_estimation.py` | 3 |
| `src/utils/nas_tas/tas/realtime/realtime_optimization_engine.py` | 3 |
| `src/utils/hardware/m1_gpu_utils.py` | 3 |

## Top files by mock occurrences

| File | Count |
| --- | ---: |
| `GUI/api_server.py` | 22 |
| `src/monitoring/example_enhanced_monitoring_usage.py` | 16 |
| `src/utils/hardware/test_basic_functionality.py` | 13 |
| `nas_search/simple_test.py` | 12 |
| `src/supervisor/performance_reporter.py` | 11 |
| `src/training/steps/market_analysis/tas_regime/optimization/enhanced_hardware_optimization.py` | 11 |
| `GUI/api_server_simple.py` | 11 |
| `code_quality/tests/test_enhanced_pipelines.py` | 10 |
| `code_quality/analyzers/stub_object_analyzer.py` | 9 |
| `src/training/steps/model_training/tactician_pre_ml_orchestrator.py` | 8 |
| `src/training/steps/model_training/sub_pipeline.py` | 7 |
| `src/trading/examples/daily_recording_example.py` | 6 |
| `src/training/steps/backtesting/sub_pipeline.py` | 6 |
| `src/examples/sr_feature_integration_example.py` | 5 |
| `src/trading/examples/full_monitoring_demo.py` | 5 |
| `code_quality/analyzers/test_coverage_analyzer.py` | 4 |
| `src/utils/hardware/m1_gpu_utils.py` | 4 |
| `src/training/steps/market_analysis/test_ml_common_integration.py` | 4 |
| `code_quality/analyzers/enhanced_fallback_detector.py` | 3 |
| `src/trading/examples/comprehensive_monitoring_example.py` | 3 |
| `src/trading/execution/live_trading_scheduler.py` | 3 |
| `src/feature_generation/utils/fractional_differentiation_pipeline.py` | 3 |
| `src/training/steps/data_collection/sub_pipeline.py` | 3 |
| `src/training/steps/market_analysis/nas_modeling/core/neural_state_space_nas.py` | 3 |
| `scripts/generate_placeholder_audit.py` | 3 |

## Top files by stub occurrences

| File | Count |
| --- | ---: |
| `code_quality/analyzers/stub_object_analyzer.py` | 57 |
| `code_quality/analyzers/enhanced_fallback_detector.py` | 10 |
| `code_quality/analyzers/enhanced_function_analyzer.py` | 5 |
| `src/utils/ml_common/utils/memory_integration.py` | 4 |
| `src/training/steps/market_analysis/nas_clustering/core/nas_config.py` | 3 |
| `src/training/steps/market_analysis/nas_clustering/core/evaluation/multi_objective.py` | 3 |
| `src/training/steps/market_analysis/nas_clustering/core/nas_search/search_space.py` | 3 |
| `scripts/generate_placeholder_audit.py` | 3 |
| `src/utils/ml_common/models/clvsa_architecture.py` | 2 |
| `src/training/steps/market_analysis/nas_modeling/core/hardware_acceleration.py` | 2 |
| `src/training/steps/market_analysis/nas_modeling/core/rl_nas.py` | 2 |
| `src/training/steps/market_analysis/nas_clustering/core/micro_regime_detector.py` | 2 |
| `src/training/steps/market_analysis/nas_clustering/core/nas_search/evolutionary_search.py` | 2 |
| `src/paper_trader.py` | 1 |
| `src/trading/execution/paper_trader.py` | 1 |
| `src/utils/common_operations.pyi` | 1 |
| `src/core/generic_base.pyi` | 1 |
| `src/config/config.pyi` | 1 |
| `src/training/steps/market_analysis/nas_modeling/__init__.py` | 1 |
| `src/training/steps/market_analysis/nas_clustering/__init__.py` | 1 |
| `scripts/generate_pass_silent_unimplemented_report.py` | 1 |
| `tests/test_nas_tas_validations.py` | 1 |

---

_Only the top 25 files per category are shown. See the accompanying JSON file for full details, including example lines._
