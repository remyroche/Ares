#!/usr/bin/env python3
"""
Analysis script to identify ALL files that are called during the complete enhanced training pipeline execution.

This script will:
1. Trace the execution flow from ares_launcher.py through the complete enhanced training pipeline
2. Identify all Python files that are imported/executed during the full training process
3. Compare against trading execution to show the true complexity difference
4. Generate a comprehensive report of training-specific files
"""

import os
import sys
import importlib
import ast
import subprocess
from pathlib import Path
from typing import Set, List, Dict
import json

class CompleteTrainingExecutionAnalyzer:
    def __init__(self, project_root: str = "/workspace"):
        self.project_root = Path(project_root)
        self.called_files: Set[str] = set()
        self.all_python_files: Set[str] = set()
        self.import_graph: Dict[str, List[str]] = {}
        
    def find_all_python_files(self) -> None:
        """Find all Python files in the project."""
        print("🔍 Finding all Python files...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            if any(skip_dir in root for skip_dir in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.project_root)
                    self.all_python_files.add(str(relative_path))
        
        print(f"📊 Found {len(self.all_python_files)} Python files")
    
    def analyze_complete_training_flow(self) -> None:
        """Analyze the execution flow starting from ares_launcher.py through the complete training pipeline."""
        print("🚀 Analyzing complete enhanced training pipeline execution flow...")
        
        # Start with ares_launcher.py
        self.called_files.add("ares_launcher.py")
        
        # Analyze imports in ares_launcher.py
        self._analyze_file_imports("ares_launcher.py")
        
        # Follow the complete training execution path
        self._follow_complete_training_execution_path()
    
    def _analyze_file_imports(self, file_path: str) -> None:
        """Analyze imports in a specific file."""
        try:
            full_path = self.project_root / file_path
            if not full_path.exists():
                return
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to find imports
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add_imported_file(alias.name, file_path)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._add_imported_file(node.module, file_path)
                        
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
    
    def _add_imported_file(self, module_name: str, source_file: str) -> None:
        """Add an imported file to the called files set."""
        # Convert module name to file path
        if module_name.startswith('src.'):
            # Handle src imports
            module_path = module_name.replace('.', '/') + '.py'
            if module_path.startswith('src/'):
                self.called_files.add(module_path)
                if source_file not in self.import_graph:
                    self.import_graph[source_file] = []
                self.import_graph[source_file].append(module_path)
        elif module_name.startswith('scripts.'):
            # Handle scripts imports
            module_path = module_name.replace('.', '/') + '.py'
            self.called_files.add(module_path)
            if source_file not in self.import_graph:
                self.import_graph[source_file] = []
            self.import_graph[source_file].append(module_path)
    
    def _follow_complete_training_execution_path(self) -> None:
        """Follow the complete execution path for the enhanced training pipeline."""
        print("📋 Following complete enhanced training execution path...")
        
        # Based on the analysis of enhanced_training_manager.py and related files, these are ALL the files called during complete training:
        complete_training_related_files = [
            # Core training infrastructure
            "src/training/step_orchestrator.py",
            "src/training/enhanced_training_manager.py",
            "src/training/enhanced_training_manager_optimized.py",
            "src/training/enhanced_training_manager_enhanced.py",
            "src/training/progress_manager.py",
            "src/training/training_manager.py",
            "src/training/training_orchestrator.py",
            "src/training/vectorized_training_pipeline.py",
            
            # Configuration and optimization
            "src/config/__init__.py",
            "src/config/computational_optimization.py",
            "src/config/training.py",
            "src/config/training_modes.py",
            "src/config/training_modes.py",
            "src/training/optimization/computational_optimization_manager.py",
            "src/training/optimization/adaptive_trial_allocator.py",
            "src/training/optimization/advanced_surrogate_models.py",
            "src/training/optimization/cached_optimizer.py",
            "src/training/optimization/parallel_optimizer.py",
            "src/training/optimization/problem_specific_strategies.py",
            "src/training/optimization/progressive_optimizer.py",
            "src/training/optimization/rollback_manager.py",
            "src/training/optimization/transfer_learning_system.py",
            "src/training/optimization_manager.py",
            "src/training/optimized_feature_selection_manager.py",
            "src/training/optimized_backtester.py",
            
            # Database and data management
            "src/database/sqlite_manager.py",
            "src/training/data_manager.py",
            "src/training/data_cleaning.py",
            "src/training/data_efficiency_optimizer.py",
            "src/training/data_quality_monitor.py",
            "src/training/data_sharing_manager.py",
            "src/training/unified_data_orchestrator.py",
            "src/training/wavelet_caching_workflow.py",
            "src/training/wavelet_feature_selection_workflow.py",
            "src/training/wavelet_feature_selection_demo.py",
            "src/training/wavelet_integration_demo.py",
            
            # Feature engineering and selection
            "src/training/feature_engineering.py",
            "src/training/feature_engineering_optimizer.py",
            "src/training/feature_integration.py",
            "src/training/feature_selection_manager.py",
            "src/training/comprehensive_feature_optimizer.py",
            "src/training/enhanced_dynamic_feature_selection.py",
            "src/training/enhanced_feature_engineering_optimizer.py",
            "src/training/matrix_enhancement_manager.py",
            "src/training/enhanced_matrix_operations.py",
            "src/training/enhanced_matrix_gpu_integration.py",
            "src/training/gpu_acceleration_m1.py",
            
            # Model training and optimization
            "src/training/model_trainer.py",
            "src/training/model_training_integrator.py",
            "src/training/model_specific_pruning.py",
            "src/training/model_probability_generator.py",
            "src/training/model_saving_utils.py",
            "src/training/advanced_neural_models.py",
            "src/training/bayesian_optimizer.py",
            "src/training/probabilistic_bayesian_optimizer.py",
            "src/training/probabilistic_model_integration.py",
            "src/training/probability_calculators.py",
            "src/training/multi_objective_optimizer.py",
            "src/training/multi_output_model_trainer.py",
            "src/training/multi_output_probability_trainer.py",
            "src/training/dual_model_system.py",
            "src/training/ensemble_manager.py",
            "src/training/calibration_manager.py",
            "src/training/regularization.py",
            "src/training/early_stage_optimization.py",
            "src/training/early_stage_optimization.py",
            "src/training/enhanced_coarse_optimizer.py",
            "src/training/enhanced_lm_config.py",
            "src/training/enhanced_lm_optimizer.py",
            "src/training/enhanced_multi_timeframe_optimizer.py",
            "src/training/enhanced_optimization_orchestrator.py",
            "src/training/matrix_diverse_lookback_optimizer.py",
            "src/training/diverse_lookback_optimizer.py",
            "src/training/hmm_regime_barrier_optimizer.py",
            "src/training/tpsl_optimizer.py",
            "src/training/timeframe_relevance_analyzer.py",
            "src/training/performance_comparison.py",
            "src/training/memory_profiler.py",
            "src/training/adaptive_optimizer.py",
            "src/training/factory.py",
            "src/training/integration_guide.py",
            "src/training/launcher_integration_patch.py",
            "src/training/validator.py",
            
            # Core step files (15 main steps)
            "src/training/steps/step01_data_collection.py",
            "src/training/steps/step01_5_data_converter.py",
            "src/training/steps/step02_feature_engineering.py",
            "src/training/steps/step03_hmm_regime_discovery.py",
            "src/training/steps/step04_regime_data_splitting.py",
            "src/training/steps/step05_triple_barrier_method.py",
            "src/training/steps/step06_feature_generation.py",
            "src/training/steps/step07_matrix_feature_selection.py",
            "src/training/steps/step08_tactician_labeling.py",
            "src/training/steps/step09_tactician_specialist_training.py",
            "src/training/steps/step10_confidence_calibration.py",
            "src/training/steps/step11_final_parameters_optimization.py",
            "src/training/steps/step12_walk_forward_validation.py",
            "src/training/steps/step13_monte_carlo_validation.py",
            "src/training/steps/step14_ab_testing.py",
            "src/training/steps/step15_saving.py",
            
            # Additional step files called during training
            "src/training/steps/step02_5_sr_optimization.py",
            "src/training/steps/step03_5_final_regime_clustering.py",
            "src/training/steps/step06_feature_engineering.py",
            "src/training/steps/step06_feature_interaction_engineering.py",
            "src/training/steps/step07_enhanced_matrix_operations.py",
            "src/training/steps/step09_hmm_based_training.py",
            "src/training/steps/step09_hmm_based_training_enhanced.py",
            "src/training/steps/step09_5_multi_timeframe_hmm_ensemble.py",
            "src/training/steps/step10_unified_regime_intelligence.py",
            "src/training/steps/step11_analyst_creation.py",
            "src/training/steps/step12_analyst_enhancement.py",
            "src/training/steps/step13_analyst_ensemble_creation.py",
            "src/training/steps/step14_tactician_labeling.py",
            "src/training/steps/step15_tactician_specialist_training.py",
            "src/training/steps/step16_confidence_calibration.py",
            "src/training/steps/step17_final_parameters_optimization.py",
            "src/training/steps/step17_final_parameters_optimization_new.py",
            "src/training/steps/step18_walk_forward_validation.py",
            "src/training/steps/step19_monte_carlo_validation.py",
            "src/training/steps/step20_ab_testing.py",
            "src/training/steps/step21_saving.py",
            
            # Step components and utilities
            "src/training/steps/__init__.py",
            "src/training/steps/analyst_training_components/__init__.py",
            "src/training/steps/analyst_training_components/regime_specific_tpsl_optimizer.py",
            "src/training/steps/combined_fractional_system.py",
            "src/training/steps/data_downloader.py",
            "src/training/steps/data_preparation_components/__init__.py",
            "src/training/steps/data_preparation_components/aggtrades_data_formatting.py",
            "src/training/steps/data_preparation_components/training_validation_config.py",
            "src/training/steps/enhanced_step1_5_data_converter.py",
            "src/training/steps/enhanced_step1_data_collection.py",
            "src/training/steps/feature_artifact_loader.py",
            "src/training/steps/fractional_differentiation.py",
            "src/training/steps/fractional_feature_selector.py",
            "src/training/steps/hmm_feature_enhancer.py",
            "src/training/steps/integrated_data_quality_pipeline.py",
            "src/training/steps/multi_timeframe_hmm_ensemble.py",
            "src/training/steps/precompute_wavelet_features.py",
            "src/training/steps/raw_data_quality_checker.py",
            "src/training/steps/sr_outcome_model_trainer.py",
            "src/training/steps/unified_data_loader.py",
            "src/training/steps/update_steps_for_unified_data.py",
            "src/training/steps/vectorized_advanced_feature_engineering.py",
            "src/training/steps/vectorized_labelling_orchestrator.py",
            "src/training/steps/backtesting_with_cached_features.py",
            "src/training/steps/optimized_step_executor.py",
            
            # Step1 subdirectory
            "src/training/steps/step1/__init__.py",
            "src/training/steps/step1/comprehensive_gap_filler.py",
            "src/training/steps/step1/data_gap_detector.py",
            "src/training/steps/step1/data_quality_dashboard.py",
            "src/training/steps/step1/data_quality_monitor.py",
            "src/training/steps/step1/data_resampler.py",
            "src/training/steps/step1/enhanced_data_quality_manager.py",
            "src/training/steps/step1/gap_filler_pipeline.py",
            "src/training/steps/step1/missing_data_downloader_and_gap_filler.py",
            "src/training/steps/step1/run_step1.py",
            "src/training/steps/step1/step1_orchestrator.py",
            "src/training/steps/step1/validate_and_fix_aggtrades_format.py",
            
            # Step4 components
            "src/training/steps/step4_analyst_labeling_feature_engineering_components/__init__.py",
            "src/training/steps/step4_analyst_labeling_feature_engineering_components/fractional_triple_barrier_labeling.py",
            "src/training/steps/step4_analyst_labeling_feature_engineering_components/optimized_triple_barrier_labeling.py",
            "src/training/steps/step4_analyst_labeling_feature_engineering_components/profit_based_feature_engineering.py",
            "src/training/steps/step4_analyst_labeling_feature_engineering_components/regime_aware_triple_barrier_labeling.py",
            "src/training/steps/step4_analyst_labeling_feature_engineering_components/regime_specific_triple_barrier_optimizer.py",
            
            # Step17 components
            "src/training/steps/step17_final_parameters_optimization/__init__.py",
            "src/training/steps/step17_final_parameters_optimization/advanced_optimization_engine.py",
            "src/training/steps/step17_final_parameters_optimization/comprehensive_parameter_integration.py",
            "src/training/steps/step17_final_parameters_optimization/efficiency_optimizer.py",
            "src/training/steps/step17_final_parameters_optimization/evaluation_engine.py",
            "src/training/steps/step17_final_parameters_optimization/hyperparameter_optimization_config.py",
            "src/training/steps/step17_final_parameters_optimization/optimized_optuna_optimization.py",
            "src/training/steps/step17_final_parameters_optimization/optimized_optuna_optimization_enhanced.py",
            "src/training/steps/step17_final_parameters_optimization/optimized_step17_implementation.py",
            "src/training/steps/step17_final_parameters_optimization/regime_specific_triple_barrier_optimization.py",
            "src/training/steps/step17_final_parameters_optimization/sr_optuna_optimization.py",
            "src/training/steps/step17_final_parameters_optimization/step17_probabilistic_bayesian_optimization.py",
            
            # Multi-timeframe training
            "src/training/steps/multi_timeframe_training/__init__.py",
            "src/training/steps/multi_timeframe_training/multi_timeframe_training_manager.py",
            
            # Training examples and tests
            "src/training/examples/__init__.py",
            "src/training/examples/optimized_training_example.py",
            "src/training/tests/test_regime_change_prediction.py",
            
            # Core training components
            "src/training/core/__init__.py",
            "src/training/core/checkpoint_manager.py",
            "src/training/core/pipeline_base.py",
            "src/training/core/pipeline_orchestrator.py",
            "src/training/core/stage_context.py",
            "src/training/core/stage_registry.py",
            
            # Utilities and validation
            "src/utils/validator_orchestrator.py",
            "src/utils/step_dependency_validator.py",
            "src/utils/training_pipeline_decorators.py",
            "src/utils/model_performance_monitor.py",
            "src/utils/comprehensive_logger.py",
            "src/utils/error_handler.py",
            "src/utils/logger.py",
            "src/utils/observability.py",
            "src/utils/warning_symbols.py",
            "src/utils/async_utils.py",
            "src/utils/centralized_decorators.py",
            "src/utils/comprehensive_file_validation.py",
            "src/utils/confidence.py",
            "src/utils/config_loader.py",
            "src/utils/data_formatting_framework.py",
            "src/utils/data_loader.py",
            "src/utils/data_optimizer.py",
            "src/utils/data_preprocessing.py",
            "src/utils/data_quality_decorators.py",
            "src/utils/data_quality_framework.py",
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
            "src/utils/enhanced_decorators.py",
            "src/utils/enhanced_error_handler.py",
            "src/utils/enhanced_error_handling.py",
            "src/utils/enhanced_memory_management.py",
            "src/utils/enhanced_missing_value_handler.py",
            "src/utils/enhanced_mlflow_integration.py",
            "src/utils/enhanced_outlier_handler.py",
            "src/utils/enhanced_pipeline_decorators.py",
            "src/utils/enhanced_validation_decorators.py",
            "src/utils/hmm_composite_manager.py",
            "src/utils/intelligent_feature_cache.py",
            "src/utils/lookahead_bias_detector.py",
            "src/utils/lookahead_bias_detector_example.py",
            "src/utils/mlflow_utils.py",
            "src/utils/model_manager.py",
            "src/utils/parallel_processing_optimizer.py",
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
            "src/utils/time_utils.py",
            "src/utils/trading_decorators.py",
            "src/utils/validation_decorators.py",
            "src/utils/vif_calculator.py",
            "src/utils/vif_validation_decorators.py",
            "src/utils/vif_validation_decorators_simple.py",
            
            # Additional training utilities
            "src/training/data_access_utils.py",
            "src/training/di_training_manager.py",
            "src/training/performance_comparison.py",
            "src/training/memory_profiler.py",
            "src/training/adaptive_optimizer.py",
            "src/training/factory.py",
            "src/training/integration_guide.py",
            "src/training/launcher_integration_patch.py",
            "src/training/validator.py",
            
            # Training-related scripts
            "src/training/steps_1_7_comprehensive_executor.py",
            "src/training/demo_pipeline_execution.py",
            "src/training/comprehensive_pipeline_executor.py",
            "src/training/comprehensive_sr_training_pipeline.py",
            "src/training/vectorized_training_pipeline.py",
            
            # Additional configuration
            "src/config/computational_optimization_config.py",
            "src/config/config_training_optimization.py",
            "src/config/enhanced_feature_optimization_config.py",
            "src/config/enhanced_feature_selection_config.py",
            "src/config/enhanced_matrix_config.py",
            "src/config/enhanced_multi_timeframe_config.py",
            "src/config/feature_engineering_optimization_config.py",
            "src/config/matrix_diverse_lookback_config.py",
            "src/config/multi_timeframe_hmm_ensemble_config.py",
            "src/config/regime_specific_optimization_config.py",
            "src/config/sr_optimization_config.py",
        ]
        
        # Add these files to called_files
        for file_path in complete_training_related_files:
            self.called_files.add(file_path)
        
        # Analyze imports in each of these files
        for file_path in complete_training_related_files:
            self._analyze_file_imports(file_path)
    
    def get_unused_files(self) -> Set[str]:
        """Get files that are not called in the complete training execution."""
        return self.all_python_files - self.called_files
    
    def generate_report(self) -> None:
        """Generate a comprehensive report."""
        print("\n" + "="*80)
        print("📊 COMPLETE ENHANCED TRAINING PIPELINE ANALYSIS REPORT")
        print("="*80)
        
        print(f"📁 Total Python files found: {len(self.all_python_files)}")
        print(f"🚀 Files called during complete training execution: {len(self.called_files)}")
        print(f"❌ Files NOT called: {len(self.get_unused_files())}")
        
        print("\n" + "="*80)
        print("📋 FILES CALLED DURING COMPLETE TRAINING EXECUTION")
        print("="*80)
        
        called_files_sorted = sorted(self.called_files)
        for file_path in called_files_sorted:
            print(f"✅ {file_path}")
        
        print("\n" + "="*80)
        print("❌ FILES NOT CALLED DURING COMPLETE TRAINING EXECUTION")
        print("="*80)
        
        unused_files = self.get_unused_files()
        unused_files_sorted = sorted(unused_files)
        
        # Categorize unused files
        categories = {
            "trading_files": [],
            "validation_files": [],
            "test_files": [],
            "other_files": []
        }
        
        for file_path in unused_files_sorted:
            if "trading" in file_path.lower() or "ares_pipeline" in file_path.lower():
                categories["trading_files"].append(file_path)
            elif "validator" in file_path.lower():
                categories["validation_files"].append(file_path)
            elif "test" in file_path.lower():
                categories["test_files"].append(file_path)
            else:
                categories["other_files"].append(file_path)
        
        for category, files in categories.items():
            if files:
                print(f"\n📂 {category.upper().replace('_', ' ')} ({len(files)} files):")
                for file_path in sorted(files):
                    print(f"   ❌ {file_path}")
        
        # Save detailed report to file
        self._save_detailed_report()
    
    def _save_detailed_report(self) -> None:
        """Save a detailed report to JSON file."""
        report = {
            "summary": {
                "total_files": len(self.all_python_files),
                "called_files": len(self.called_files),
                "unused_files": len(self.get_unused_files())
            },
            "called_files": sorted(self.called_files),
            "unused_files": sorted(self.get_unused_files()),
            "import_graph": self.import_graph
        }
        
        with open("complete_training_execution_analysis.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: complete_training_execution_analysis.json")

def main():
    """Main execution function."""
    print("🔍 Starting complete enhanced training pipeline analysis...")
    
    analyzer = CompleteTrainingExecutionAnalyzer()
    
    # Step 1: Find all Python files
    analyzer.find_all_python_files()
    
    # Step 2: Analyze complete training execution flow
    analyzer.analyze_complete_training_flow()
    
    # Step 3: Generate report
    analyzer.generate_report()
    
    print("\n✅ Complete training analysis complete!")

if __name__ == "__main__":
    main()