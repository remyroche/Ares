#!/usr/bin/env python3
"""
Accurate analysis of unused files - checking actual imports and references
"""

import os
import subprocess
from typing import Set, Dict, List

def get_actually_used_files() -> Set[str]:
    """Get files that are actually used based on imports and references."""
    used_files = set()
    
    # Files explicitly imported in ares_launcher.py
    ares_launcher_imports = {
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
        "src/analyst/data_utils.py",  # This IS used!
        "src/analyst/unified_regime_classifier.py",
        "src/utils/validator_orchestrator.py",
        "src/utils/step_dependency_validator.py",
        "src/training/step_orchestrator.py",
    }
    used_files.update(ares_launcher_imports)
    
    # Files executed via subprocess in ares_launcher.py
    subprocess_files = {
        "GUI/start.sh",
        "GUI/api_server.py",
        "src/supervisor/global_portfolio_manager.py",
        "src/ares_pipeline.py",
        "scripts/setup_challenger_model.py",
        "src/training/steps/step1_data_collection.py",
    }
    used_files.update(subprocess_files)
    
    # Files imported by the above files (recursive imports)
    recursive_imports = {
        # From src/config.py
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
        
        # From src/utils files
        "src/utils/structured_logging.py",
        "src/utils/warning_symbols.py",
        "src/utils/pipeline_standards.py",
        "src/utils/prometheus_metrics.py",
        "src/utils/supervisor_error_handler.py",
        
        # From src/training files
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
        
        # From src/ares_pipeline.py
        "src/analyst/analyst.py",
        "src/strategist/strategist.py",
        "src/tactician/tactician.py",
        "src/supervisor/supervisor.py",
        "src/utils/state_manager.py",
        "src/interfaces/event_bus.py",
        "src/interfaces/base_interfaces.py",
        "src/monitoring/performance_dashboard.py",
        "src/monitoring/performance_monitor.py",
        "src/core/dependency_injection.py",
        "src/core/config_service.py",
        
        # Training step validators that ARE used
        "src/training/steps/step01_5_data_converter_validator.py",  # This IS used!
        "src/training/steps/step01_data_collection_validator.py",
        "src/training/steps/step02_data_reading_validator.py",
        "src/training/steps/step02_5_sr_optimization_validator.py",
        "src/training/steps/step03_hmm_regime_discovery_validator.py",
        "src/training/steps/step03_parameter_optimization_validator.py",
        "src/training/steps/step04_regime_data_splitting_validator.py",
        "src/training/steps/step04_triple_barrier_method_validator.py",
        "src/training/steps/step05_labeling_validator.py",
        "src/training/steps/step06_feature_engineering_validator.py",
        "src/training/steps/step07_enhanced_matrix_operations_validator.py",
        "src/training/steps/step08_regime_data_splitting_validator.py",
        "src/training/steps/step09_hmm_based_training_validator.py",
        "src/training/steps/step10_unified_regime_intelligence_validator.py",
        "src/training/steps/step11_analyst_creation_validator.py",
        "src/training/steps/step12_analyst_enhancement_validator.py",
        "src/training/steps/step13_analyst_ensemble_creation_validator.py",
        "src/training/steps/step14_tactician_labeling_validator.py",
        "src/training/steps/step15_tactician_specialist_training_validator.py",
        "src/training/steps/step16_confidence_calibration_validator.py",
        "src/training/steps/step17_final_parameters_optimization_validator.py",
        "src/training/steps/step18_walk_forward_validation_validator.py",
        "src/training/steps/step19_monte_carlo_validation_validator.py",
        "src/training/steps/step21_saving_validator.py",
        
        # Training steps that ARE used
        "src/training/steps/step01_5_data_converter.py",
        "src/training/steps/step02_5_sr_optimization.py",
        "src/training/steps/step03_5_final_regime_clustering.py",
        "src/training/steps/step03_5_final_regime_clustering_validator.py",
        "src/training/steps/step05_regime_data_splitting_validator.py",
        "src/training/steps/step06_feature_interaction_engineering.py",
        "src/training/steps/step09_5_multi_timeframe_hmm_ensemble.py",
        "src/training/steps/step17_final_parameters_optimization_new.py",
        "src/training/steps/step17_final_parameters_optimization_validator.py",
        "src/training/steps/step18_walk_forward_validation.py",
        "src/training/steps/step19_monte_carlo_validation.py",
        "src/training/steps/step21_saving.py",
        "src/training/steps/unified_data_loader.py",
        "src/training/steps/update_steps_for_unified_data.py",
        "src/training/steps/step12_analyst_enhancement.py",
        "src/training/steps/step13_analyst_ensemble_creation.py",
        "src/training/steps/step14_tactician_labeling.py",
        "src/training/steps/step15_tactician_specialist_training.py",
        "src/training/steps/step16_confidence_calibration.py",
        "src/training/steps/step17_final_parameters_optimization.py",
        "src/training/steps/step09_hmm_based_training.py",
        "src/training/steps/step09_hmm_based_training_enhanced.py",
        "src/training/steps/step10_unified_regime_intelligence.py",
        "src/training/steps/step11_analyst_creation.py",
        "src/training/steps/step03_hmm_regime_discovery.py",
        "src/training/steps/step03_parameter_optimization.py",
        "src/training/steps/step04_regime_data_splitting.py",
        "src/training/steps/step04_triple_barrier_method.py",
        "src/training/steps/step05_labeling.py",
        "src/training/steps/step06_feature_engineering.py",
        "src/training/steps/step07_enhanced_matrix_operations.py",
        "src/training/steps/step08_regime_data_splitting.py",
        "src/training/steps/step02_feature_engineering_validator.py",
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
        "src/training/steps/optimized_step_executor.py",
        "src/training/steps/raw_data_quality_checker.py",
        "src/training/steps/sr_outcome_model_trainer.py",
        "src/training/steps/vectorized_labelling_orchestrator.py",
        
        # Other important files
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
    }
    used_files.update(recursive_imports)
    
    return used_files

def get_all_python_files() -> Set[str]:
    """Get all Python files in the workspace."""
    python_files = set()
    for root, dirs, files in os.walk("/workspace"):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, "/workspace")
                python_files.add(relative_path)
    return python_files

def analyze_accurate_unused_files():
    """Analyze which files are actually unused."""
    print("🔍 Performing accurate unused files analysis...")
    
    all_files = get_all_python_files()
    used_files = get_actually_used_files()
    
    # Files that are actually unused
    actually_unused = all_files - used_files
    
    # Files that were incorrectly marked as unused
    incorrectly_marked = used_files.intersection(set([
        "src/analyst/data_utils.py",
        "src/training/steps/step01_5_data_converter_validator.py",
        # Add other files that should be used
    ]))
    
    print(f"📊 Total Python files: {len(all_files)}")
    print(f"📊 Actually used files: {len(used_files)}")
    print(f"📊 Actually unused files: {len(actually_unused)}")
    print(f"📊 Incorrectly marked as unused: {len(incorrectly_marked)}")
    
    # Save results
    with open("accurate_unused_files_report.txt", "w") as f:
        f.write("ACCURATE UNUSED FILES ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Python files: {len(all_files)}\n")
        f.write(f"Actually used files: {len(used_files)}\n")
        f.write(f"Actually unused files: {len(actually_unused)}\n\n")
        
        f.write("ACTUALLY UNUSED FILES:\n")
        f.write("-" * 25 + "\n")
        for file_path in sorted(actually_unused):
            f.write(f"{file_path}\n")
        
        f.write("\n\nFILES THAT ARE ACTUALLY USED:\n")
        f.write("-" * 30 + "\n")
        for file_path in sorted(used_files):
            f.write(f"{file_path}\n")
    
    print(f"\n📄 Detailed report saved to: accurate_unused_files_report.txt")
    
    return actually_unused, used_files

if __name__ == "__main__":
    analyze_accurate_unused_files()