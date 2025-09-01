#!/usr/bin/env python3
"""
Script to analyze which Python files are not called by ares_launcher.py
"""

import os
import re
from pathlib import Path
from typing import Set, List, Dict

def get_all_python_files(workspace_path: str = "/workspace") -> Set[str]:
    """Get all Python files in the workspace."""
    python_files = set()
    for root, dirs, files in os.walk(workspace_path):
        for file in files:
            if file.endswith('.py'):
                # Convert to relative path from workspace
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, workspace_path)
                python_files.add(relative_path)
    return python_files

def get_called_files_from_ares_launcher() -> Set[str]:
    """Get files that are explicitly called by ares_launcher.py based on analysis."""
    called_files = {
        # Direct imports in ares_launcher.py
        "src/config.py",
        "src/config/training_modes.py",
        "src/utils/comprehensive_logger.py",
        "src/utils/error_handler.py",
        "src/utils/logger.py",
        "src/utils/signal_handler.py",
        "src/utils/observability.py",
        "src/database/sqlite_manager.py",
        "src/training/enhanced_training_manager.py",
        "src/training/steps/precompute_wavelet_features.py",
        "src/analyst/data_utils.py",
        "src/analyst/unified_regime_classifier.py",
        "src/utils/validator_orchestrator.py",
        "src/utils/step_dependency_validator.py",
        "src/training/step_orchestrator.py",
        
        # Subprocess executions
        "GUI/start.sh",
        "GUI/api_server.py",
        "src/supervisor/global_portfolio_manager.py",
        "src/ares_pipeline.py",
        "scripts/setup_challenger_model.py",
        "src/training/steps/step1_data_collection.py",
        
        # Recursive imports from direct dependencies
        "src/config/modular_config.py",
        "src/config/environment.py",
        "src/config/system.py",
        "src/config/trading.py",
        "src/config/training.py",
        "src/config/validation.py",
        "src/config/constants.py",
        "src/config/computational_optimization.py",
        "src/config/enhanced_reporting_config.py",
        "src/config/typed_config.py",
        "src/config/label_model_mapping.py",
        "src/config/multi_timeframe_hmm_ensemble_config.py",
        "src/config/computational_optimization_config.py",
        "src/config/enhanced_feature_selection_config.py",
        "src/config/enhanced_multi_timeframe_config.py",
        "src/config/enhanced_matrix_config.py",
        "src/config/enhanced_prediction_service_config.py",
        
        # Utils files
        "src/utils/structured_logging.py",
        "src/utils/warning_symbols.py",
        "src/utils/pipeline_standards.py",
        "src/utils/prometheus_metrics.py",
        "src/utils/supervisor_error_handler.py",
        
        # Training files
        "src/training/enhanced_training_manager_optimized.py",
        "src/training/optimization/computational_optimization_manager.py",
        "src/training/steps/multi_timeframe_training/multi_timeframe_training_manager.py",
        "src/utils/model_performance_monitor.py",
        "src/utils/training_pipeline_decorators.py",
        "src/training/progress_manager.py",
        "src/training/steps/vectorized_advanced_feature_engineering.py",
        "src/utils/data_optimizer.py",
        "src/utils/centralized_decorators.py",
        "src/utils/centralized_decorators_simple.py",
        
        # Analyst files
        "src/analyst/analyst.py",
        "src/tactician/sr_breakout_predictor.py",
        
        # Interface files
        "src/interfaces/event_bus.py",
        "src/interfaces/base_interfaces.py",
        
        # Strategist files
        "src/strategist/strategist.py",
        
        # Supervisor files
        "src/supervisor/supervisor.py",
        
        # Tactician files
        "src/tactician/tactician.py",
        
        # State manager
        "src/utils/state_manager.py",
        
        # Monitoring files
        "src/monitoring/performance_dashboard.py",
        "src/monitoring/performance_monitor.py",
        "src/monitoring/dual_model_system.py",
        
        # Core files
        "src/core/dependency_injection.py",
        "src/core/config_service.py",
        
        # Additional files found in analysis
        "src/utils/async_utils.py",
        "src/utils/mlflow_utils.py",
        "src/utils/model_manager.py",
        "src/utils/feature_output_validator.py",
        "src/utils/hmm_composite_manager.py",
        "src/utils/intelligent_feature_cache.py",
        "src/utils/lookahead_bias_detector.py",
        "src/utils/model_performance_monitor.py",
        "src/utils/parallel_processing_optimizer.py",
        "src/utils/advanced_ml_validation.py",
        "src/utils/base_validator.py",
        "src/utils/centralized_decorators_v2.py",
        "src/utils/comprehensive_data_quality_validator.py",
        "src/utils/comprehensive_file_validation.py",
        "src/utils/confidence.py",
        "src/utils/config_loader.py",
        "src/utils/configuration_security.py",
        "src/utils/data_formatting_framework.py",
        "src/utils/data_loader.py",
        "src/utils/data_optimizer.py",
        "src/utils/data_preprocessing.py",
        "src/utils/data_quality_decorators.py",
        "src/utils/data_quality_framework.py",
        "src/utils/data_quality_validator.py",
        "src/utils/data_type_optimizer.py",
        "src/utils/data_validation.py",
        "src/utils/database_security.py",
        "src/utils/decorator_compatibility.py",
        "src/utils/decorator_config.py",
        "src/utils/decorator_registry.py",
        "src/utils/decorators.py",
        "src/utils/domain_errors.py",
        "src/utils/enhanced_config_management.py",
        "src/utils/enhanced_data_quality_decorators.py",
        "src/utils/enhanced_data_quality_validator.py",
        "src/utils/enhanced_decorators.py",
        "src/utils/enhanced_error_handling.py",
        "src/utils/enhanced_memory_management.py",
        "src/utils/enhanced_missing_value_handler.py",
        "src/utils/enhanced_mlflow_integration.py",
        "src/utils/enhanced_outlier_handler.py",
        "src/utils/enhanced_pipeline_decorators.py",
        "src/utils/enhanced_validation_decorators.py",
        "src/utils/feature_output_validator.py",
        "src/utils/hmm_composite_manager.py",
        "src/utils/intelligent_feature_cache.py",
        "src/utils/lookahead_bias_detector.py",
        "src/utils/mlflow_utils.py",
        "src/utils/model_manager.py",
        "src/utils/model_performance_monitor.py",
        "src/utils/parquet_utils.py",
        "src/utils/pipeline_standards.py",
        "src/utils/prometheus_metrics.py",
        "src/utils/purged_kfold.py",
        "src/utils/quality_alert_system.py",
        "src/utils/security_framework.py",
        "src/utils/standardized_config_manager.py",
        "src/utils/standardized_error_handler.py",
        "src/utils/standardized_model_manager.py",
        "src/utils/steps_1_7_compatibility_framework.py",
        "src/utils/structured_logging.py",
        "src/utils/supervisor_error_handler.py",
        "src/utils/time_utils.py",
        "src/utils/trading_decorators.py",
        "src/utils/validation_decorators.py",
        "src/utils/vif_calculator.py",
        "src/utils/vif_validation_decorators.py",
        "src/utils/vif_validation_decorators_simple.py",
        "src/utils/warning_symbols.py",
        
        # Training files
        "src/training/steps_1_7_comprehensive_executor.py",
        "src/training/timeframe_relevance_analyzer.py",
        "src/training/tpsl_optimizer.py",
        "src/training/training_manager.py",
        "src/training/training_orchestrator.py",
        "src/training/unified_data_orchestrator.py",
        "src/training/validator.py",
        "src/training/vectorized_training_pipeline.py",
        "src/training/wavelet_caching_workflow.py",
        "src/training/wavelet_feature_selection_workflow.py",
        "src/training/optimization/optimization_manager.py",
        "src/training/optimized_feature_selection_manager.py",
        "src/training/performance_comparison.py",
        "src/training/probabilistic_bayesian_optimizer.py",
        "src/training/probabilistic_model_integration.py",
        "src/training/probability_calculators.py",
        "src/training/regularization.py",
        "src/training/feature_selection_manager.py",
        "src/training/gpu_acceleration_m1.py",
        "src/training/hmm_regime_barrier_optimizer.py",
        "src/training/integration_guide.py",
        "src/training/launcher_integration_patch.py",
        "src/training/matrix_diverse_lookback_optimizer.py",
        "src/training/matrix_enhancement_manager.py",
        "src/training/memory_profiler.py",
        "src/training/model_probability_generator.py",
        "src/training/model_saving_utils.py",
        "src/training/model_specific_pruning.py",
        "src/training/model_trainer.py",
        "src/training/model_training_integrator.py",
        "src/training/multi_objective_optimizer.py",
        "src/training/multi_output_model_trainer.py",
        "src/training/multi_output_probability_trainer.py",
        "src/training/enhanced_lm_optimizer.py",
        "src/training/enhanced_matrix_gpu_integration.py",
        "src/training/enhanced_matrix_operations.py",
        "src/training/enhanced_multi_timeframe_optimizer.py",
        "src/training/enhanced_optimization_orchestrator.py",
        "src/training/ensemble_manager.py",
        "src/training/factory.py",
        "src/training/feature_engineering.py",
        "src/training/feature_engineering_optimizer.py",
        "src/training/feature_integration.py",
        "src/training/data_access_utils.py",
        "src/training/data_cleaning.py",
        "src/training/data_efficiency_optimizer.py",
        "src/training/data_manager.py",
        "src/training/data_quality_monitor.py",
        "src/training/data_sharing_manager.py",
        "src/training/di_training_manager.py",
        "src/training/diverse_lookback_optimizer.py",
        "src/training/dual_model_system.py",
        "src/training/early_stage_optimization.py",
        "src/training/enhanced_coarse_optimizer.py",
        "src/training/enhanced_dynamic_feature_selection.py",
        "src/training/enhanced_feature_engineering_optimizer.py",
        "src/training/enhanced_lm_config.py",
        "src/training/adaptive_optimizer.py",
        "src/training/advanced_neural_models.py",
        "src/training/bayesian_optimizer.py",
        "src/training/calibration_manager.py",
        "src/training/comprehensive_feature_optimizer.py",
        "src/training/comprehensive_pipeline_executor.py",
        "src/training/comprehensive_sr_training_pipeline.py",
        
        # Training steps
        "src/training/steps/vectorized_labelling_orchestrator.py",
        "src/training/steps/step17_final_parameters_optimization_new.py",
        "src/training/steps/step17_final_parameters_optimization_validator.py",
        "src/training/steps/step18_walk_forward_validation.py",
        "src/training/steps/step18_walk_forward_validation_validator.py",
        "src/training/steps/step19_monte_carlo_validation.py",
        "src/training/steps/step19_monte_carlo_validation_validator.py",
        "src/training/steps/step21_saving.py",
        "src/training/steps/step21_saving_validator.py",
        "src/training/steps/unified_data_loader.py",
        "src/training/steps/update_steps_for_unified_data.py",
        "src/training/steps/step12_analyst_enhancement.py",
        "src/training/steps/step12_analyst_enhancement_validator.py",
        "src/training/steps/step13_analyst_ensemble_creation.py",
        "src/training/steps/step13_analyst_ensemble_creation_validator.py",
        "src/training/steps/step14_tactician_labeling.py",
        "src/training/steps/step14_tactician_labeling_validator.py",
        "src/training/steps/step15_tactician_specialist_training.py",
        "src/training/steps/step15_tactician_specialist_training_validator.py",
        "src/training/steps/step16_confidence_calibration.py",
        "src/training/steps/step16_confidence_calibration_validator.py",
        "src/training/steps/step17_final_parameters_optimization.py",
        "src/training/steps/step09_hmm_based_training.py",
        "src/training/steps/step09_hmm_based_training_enhanced.py",
        "src/training/steps/step09_hmm_based_training_validator.py",
        "src/training/steps/step10_unified_regime_intelligence.py",
        "src/training/steps/step10_unified_regime_intelligence_validator.py",
        "src/training/steps/step11_analyst_creation.py",
        "src/training/steps/step11_analyst_creation_validator.py",
        "src/training/steps/step03_5_final_regime_clustering_validator.py",
        "src/training/steps/step03_hmm_regime_discovery.py",
        "src/training/steps/step03_hmm_regime_discovery_validator.py",
        "src/training/steps/step03_parameter_optimization.py",
        "src/training/steps/step03_parameter_optimization_validator.py",
        "src/training/steps/step04_regime_data_splitting.py",
        "src/training/steps/step04_regime_data_splitting_validator.py",
        "src/training/steps/step04_triple_barrier_method.py",
        "src/training/steps/step04_triple_barrier_method_validator.py",
        "src/training/steps/step05_labeling.py",
        "src/training/steps/step05_labeling_validator.py",
        "src/training/steps/step05_regime_data_splitting_validator.py",
        "src/training/steps/step06_feature_engineering.py",
        "src/training/steps/step06_feature_engineering_validator.py",
        "src/training/steps/step06_feature_interaction_engineering.py",
        "src/training/steps/step07_enhanced_matrix_operations.py",
        "src/training/steps/step07_enhanced_matrix_operations_validator.py",
        "src/training/steps/step08_regime_data_splitting.py",
        "src/training/steps/step08_regime_data_splitting_validator.py",
        "src/training/steps/step09_5_multi_timeframe_hmm_ensemble.py",
        "src/training/steps/optimized_step_executor.py",
        "src/training/steps/raw_data_quality_checker.py",
        "src/training/steps/sr_outcome_model_trainer.py",
        "src/training/steps/step01_5_data_converter.py",
        "src/training/steps/step01_5_data_converter_validator.py",
        "src/training/steps/step01_data_collection.py",
        "src/training/steps/step01_data_collection_validator.py",
        "src/training/steps/step02_5_sr_optimization.py",
        "src/training/steps/step02_5_sr_optimization_validator.py",
        "src/training/steps/step02_data_reading.py",
        "src/training/steps/step02_data_reading_validator.py",
        "src/training/steps/step02_feature_engineering_validator.py",
        "src/training/steps/step03_5_final_regime_clustering.py",
        "src/training/steps/combined_fractional_system.py",
        "src/training/steps/data_downloader.py",
        "src/training/steps/enhanced_step1_5_data_converter.py",
        "src/training/steps/enhanced_step1_data_collection.py",
        "src/training/steps/feature_artifact_loader.py",
        "src/training/steps/fractional_differentiation.py",
        "src/training/steps/fractional_feature_selector.py",
        "src/training/steps/hmm_feature_enhancer.py",
        "src/training/steps/integrated_data_quality_pipeline.py",
        "src/training/steps/multi_timeframe_hmm_ensemble.py",
        
        # Monitoring files
        "src/monitoring/error_detection_system.py",
        "src/monitoring/tracking_system.py",
        "src/monitoring/correlation_manager.py",
        "src/monitoring/report_scheduler.py",
        "src/monitoring/fractional_system_monitor.py",
        "src/monitoring/enhanced_ml_tracker.py",
        "src/monitoring/integration_manager.py",
        "src/monitoring/surrogate_optimization_monitor.py",
        "src/monitoring/metrics_dashboard.py",
        "src/monitoring/fractional_performance_tracker.py",
        "src/monitoring/csv_exporter.py",
        "src/monitoring/regime_sr_tracker.py",
        "src/monitoring/advanced_tracer.py",
        "src/monitoring/trade_conditions_monitor.py",
        "src/monitoring/ml_monitor.py",
        
        # Launcher files
        "src/launcher/enhanced_trading_launcher.py",
        
        # Component files
        "src/components/modular_analyst.py",
        "src/components/modular_tactician.py",
        "src/components/modular_supervisor.py",
        "src/components/modular_strategist.py",
        
        # Interface files
        "src/interfaces/enhanced_event_bus.py",
        
        # Core files
        "src/core/enhanced_dependency_injection.py",
        
        # Database files
        "src/database/sqlite_manager.py",
        
        # Exchange files (if any)
        # Add any exchange-related files found
        
        # Paper trader
        "src/paper_trader.py",
        
        # Config files
        "src/config_optuna.py",
        
        # Tasks
        "src/tasks.py",
        
        # GUI files
        "GUI/api_server.py",
        
        # Scripts
        "scripts/bot_monitor.py",
        
        # Root level files that might be called
        "ares_launcher.py",
    }
    
    return called_files

def analyze_unused_files():
    """Analyze which files are not called by ares_launcher.py."""
    print("🔍 Analyzing unused files...")
    
    # Get all Python files
    all_files = get_all_python_files()
    print(f"📊 Total Python files found: {len(all_files)}")
    
    # Get called files
    called_files = get_called_files_from_ares_launcher()
    print(f"📊 Files called by ares_launcher.py: {len(called_files)}")
    
    # Find unused files
    unused_files = all_files - called_files
    
    # Sort for better readability
    unused_files = sorted(unused_files)
    
    print(f"📊 Unused files: {len(unused_files)}")
    print("\n" + "="*80)
    print("FILES NOT CALLED BY ares_launcher.py")
    print("="*80)
    
    # Group by directory for better organization
    unused_by_dir = {}
    for file in unused_files:
        dir_name = os.path.dirname(file) if os.path.dirname(file) else "root"
        if dir_name not in unused_by_dir:
            unused_by_dir[dir_name] = []
        unused_by_dir[dir_name].append(file)
    
    # Print organized results
    for dir_name in sorted(unused_by_dir.keys()):
        print(f"\n📁 {dir_name}/")
        print("-" * (len(dir_name) + 3))
        for file in unused_by_dir[dir_name]:
            print(f"  {file}")
    
    # Save to file
    with open("unused_files_report.txt", "w") as f:
        f.write("FILES NOT CALLED BY ares_launcher.py\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Python files: {len(all_files)}\n")
        f.write(f"Files called by ares_launcher.py: {len(called_files)}\n")
        f.write(f"Unused files: {len(unused_files)}\n\n")
        
        for dir_name in sorted(unused_by_dir.keys()):
            f.write(f"\n📁 {dir_name}/\n")
            f.write("-" * (len(dir_name) + 3) + "\n")
            for file in unused_by_dir[dir_name]:
                f.write(f"  {file}\n")
    
    print(f"\n📄 Detailed report saved to: unused_files_report.txt")
    print(f"📊 Summary: {len(unused_files)} out of {len(all_files)} files are not called by ares_launcher.py")

if __name__ == "__main__":
    analyze_unused_files()