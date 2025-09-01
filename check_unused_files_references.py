#!/usr/bin/env python3
"""
Script to check if unused files are referenced anywhere in the codebase
"""

import os
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple

def get_unused_files() -> Set[str]:
    """Get the list of unused files from our previous analysis."""
    unused_files = {
        # Root level analysis scripts
        "analyze_complete_training_execution.py",
        "analyze_step1_execution.py",
        "analyze_strict_thresholds.py",
        "analyze_trading_execution.py",
        "analyze_unused_files.py",
        "analyze_validation_issues.py",
        "automated_syntax_fixer.py",
        "check_existing_data.py",
        "check_trade_data.py",
        "cleanup_actions.py",
        "cleanup_script.py",
        "complete_remaining_16_steps.py",
        "complete_remaining_steps.py",
        "complete_remaining_steps_integration.py",
        "comprehensive_code_quality_fixer.py",
        "comprehensive_fix.py",
        "comprehensive_gap_filler.py",
        "comprehensive_gap_filler_v2.py",
        "comprehensive_syntax_fixer.py",
        "comprehensive_training_fix.py",
        "conservative_syntax_fixer.py",
        "consolidate_aggtrades.py",
        "consolidate_data.py",
        "convert_csv_to_parquet.py",
        "create_30m_hmm_artifacts.py",
        "create_correct_mock_data.py",
        "create_regime_splits.py",
        "dead_code_remover.py",
        "debug_clustering.py",
        "debug_hmm_combinations.py",
        "debug_interaction_flow.py",
        "debug_low_variance_features.py",
        "debug_metadata_detection.py",
        "detect_and_fill_gaps_immediate.py",
        "diagnose_feature_pipeline.py",
        "diagnose_interaction_features.py",
        "diagnose_regime_data.py",
        "download_aggtrades_range.py",
        "download_futures_only.py",
        "download_missing_aggtrades_2023_2024.py",
        "download_missing_aggtrades_days.py",
        "download_missing_data.py",
        "download_missing_futures.py",
        "download_remaining_aggtrades.py",
        "download_specific_missing_data.py",
        "enhanced_syntax_fixer.py",
        "enhanced_validation_logging.py",
        "enhanced_validation_wrapper.py",
        "extract_feature_details.py",
        "feature_analysis_script.py",
        "feature_specific_validation.py",
        "final_fix.py",
        "final_fix_script.py",
        "final_targeted_fix.py",
        "final_targeted_fix_v2.py",
        "final_targeted_fix_v3.py",
        "final_utils_fix.py",
        "fix_syntax_errors.py",
        "fix_training_placeholders.py",
        "gap_filler_clean.py",
        "identify_deleted_aggtrades.py",
        "implement_feature_specific_validation.py",
        "kelly_criterion_formula.py",
        "multi_objective_hmm_optimizer.py",
        "optimize_hmm_regime_parameters.py",
        "optimize_hmm_regime_parameters_advanced.py",
        "optimize_hmm_regime_parameters_enhanced.py",
        "quick_error_scanner.py",
        "run_30m_hmm_step.py",
        "run_code_quality_tools.py",
        "run_fixed_hmm_regime_discovery.py",
        "run_pipeline_simple.py",
        "run_step2_direct.py",
        "run_syntax_fix.py",
        "simulate_regime_merging_from_existing_data.py",
        "simulate_regime_merging_optimization.py",
        "standardize_remaining_steps.py",
        "standardize_utility_modules.py",
        "syntax_error_scanner.py",
        "targeted_fix.py",
        "targeted_fix_training_placeholders.py",
        "targeted_syntax_fixer.py",
        "test_advanced_models_core.py",
        "universal_syntax_fixer.py",
        "update_aggtrades_gaps.py",
        "update_all_steps_mlflow_integration.py",
        "update_training_analysis.py",
        "verify_aggtrades_downloads.py",
        "verify_training_modes.py",
        
        # Analysis directory
        "analysis/data_collection_quality_analysis.py",
        "analysis/data_preparation_quality_analysis.py",
        "analysis/missing_values_analysis.py",
        "analysis/model_training_quality_analysis.py",
        
        # Code quality tools
        "code_quality/tools/batch_import_cleaner.py",
        "code_quality/tools/code_quality_analyzer.py",
        "code_quality/tools/dead_code_remover.py",
        "code_quality/tools/placeholder_finder.py",
        "code_quality/tools/syntax_fixer.py",
        
        # Crypto analysis
        "crypto_analysis/data_analyzer.py",
        "crypto_analysis/data_downloader.py",
        "crypto_analysis/run_analysis.py",
        
        # Docs
        "docs/enhanced_mlflow_step_integration_template.py",
        
        # Exchange files
        "exchange/__init__.py",
        "exchange/base_exchange.py",
        "exchange/binance.py",
        "exchange/factory.py",
        "exchange/gateio.py",
        "exchange/mexc.py",
        "exchange/mexc_optimized.py",
        "exchange/okx.py",
        
        # Enhanced analyst components
        "src/analyst/__init__.py",
        "src/analyst/advanced_feature_engineering.py",
        "src/analyst/autoencoder_feature_generator.py",
        "src/analyst/decision_aggregator.py",
        "src/analyst/di_analyst.py",
        "src/analyst/dynamic_regime_mapper.py",
        "src/analyst/enhanced_prediction_integrator.py",
        "src/analyst/enhanced_regime_predictor.py",
        "src/analyst/example_directional_analysis.py",
        "src/analyst/feature_engineering_orchestrator.py",
        "src/analyst/liquidation_risk_model.py",
        "src/analyst/meta_label_relevance.py",
        "src/analyst/meta_labeling_system.py",
        "src/analyst/ml_confidence_predictor.py",
        "src/analyst/multi_timeframe_feature_engineering.py",
        "src/analyst/multi_timeframe_regime_integration.py",
        "src/analyst/order_book_analyzer.py",
        "src/analyst/predictive_ensembles.py",
        "src/analyst/regime_expert_orchestrator.py",
        "src/analyst/regime_runtime.py",
        "src/analyst/unified_regime_intelligence_runtime.py",
        
        # Enhanced tactician components
        "src/tactician/__init__.py",
        "src/tactician/async_order_executor.py",
        "src/tactician/comprehensive_enhanced_scenario_predictor.py",
        "src/tactician/dynamic_barrier_calculator.py",
        "src/tactician/enhanced_execution_manager.py",
        "src/tactician/enhanced_order_manager.py",
        "src/tactician/enhanced_prediction_integrator.py",
        "src/tactician/enhanced_scenario_based_predictor.py",
        "src/tactician/leverage_sizer.py",
        "src/tactician/ml_target_updater.py",
        "src/tactician/ml_target_validator.py",
        "src/tactician/position_closing.py",
        "src/tactician/position_division_strategy.py",
        "src/tactician/position_monitor.py",
        "src/tactician/position_sizer.py",
        "src/tactician/scenario_based_predictor.py",
        "src/tactician/sr_backtesting_validator.py",
        "src/tactician/sr_data_integration.py",
        "src/tactician/sr_data_integration_simple.py",
        "src/tactician/sr_detection_optimization.py",
        "src/tactician/sr_levels_manager.py",
        "src/tactician/sr_weight_optimizer.py",
        "src/tactician/step17_optimized_tactician.py",
        "src/tactician/tactics_orchestrator.py",
        
        # Enhanced supervisor components
        "src/supervisor/__init__.py",
        "src/supervisor/ab_tester.py",
        "src/supervisor/dynamic_weighter.py",
        "src/supervisor/enhanced_model_monitor.py",
        "src/supervisor/enhanced_prediction_service.py",
        "src/supervisor/exchange_ab_tester.py",
        "src/supervisor/exchange_volume_adapter.py",
        "src/supervisor/main.py",
        "src/supervisor/model_behavior_tracker.py",
        "src/supervisor/monitoring.py",
        "src/supervisor/multi_exchange_ab_tester.py",
        "src/supervisor/optimizer.py",
        "src/supervisor/performance_monitor.py",
        "src/supervisor/performance_reporter.py",
        "src/supervisor/pnl_loss_functions.py",
        "src/supervisor/risk_allocator.py",
        
        # Training optimization components
        "src/training/optimization/__init__.py",
        "src/training/optimization/adaptive_trial_allocator.py",
        "src/training/optimization/advanced_surrogate_models.py",
        "src/training/optimization/cached_optimizer.py",
        "src/training/optimization/parallel_optimizer.py",
        "src/training/optimization/problem_specific_strategies.py",
        "src/training/optimization/progressive_optimizer.py",
        "src/training/optimization/rollback_manager.py",
        "src/training/optimization/transfer_learning_system.py",
        
        # Configuration files
        "src/config/__init__.py",
        "src/config/config.py",
        "src/config/config_confidence.py",
        "src/config/config_ensemble.py",
        "src/config/config_leverage.py",
        "src/config/config_manager.py",
        "src/config/config_position_sizing.py",
        "src/config/config_regime_transitions.py",
        "src/config/config_sr.py",
        "src/config/config_system_monitoring.py",
        "src/config/config_technical_indicators.py",
        "src/config/config_tpsl.py",
        "src/config/config_training_optimization.py",
        "src/config/config_two_tier.py",
        "src/config/diverse_lookback_config.py",
        "src/config/enhanced_feature_optimization_config.py",
        "src/config/feature_engineering_optimization_config.py",
        "src/config/fractional_implementations_config.py",
        "src/config/m1_gpu_config.py",
        "src/config/matrix_diverse_lookback_config.py",
        "src/config/multi_output_config.py",
        "src/config/regime_specific_optimization_config.py",
        "src/config/sr_optimization_config.py",
        
        # Utility modules
        "src/utils/__init__.py",
        "src/utils/advanced_decorators.py",
        "src/utils/enhanced_error_handler.py",
        "src/utils/supervisor_error_handler_example.py",
        
        # Core infrastructure
        "src/core/__init__.py",
        "src/core/di_integration.py",
        "src/core/di_launcher.py",
        "src/core/enhanced_factories.py",
        "src/core/generic_base.py",
        "src/core/injectable_base.py",
        "src/core/service_registry.py",
        
        # Custom types
        "src/custom_types/__init__.py",
        "src/custom_types/base_types.py",
        "src/custom_types/config_types.py",
        "src/custom_types/data_types.py",
        "src/custom_types/ml_types.py",
        "src/custom_types/protocol_types.py",
        "src/custom_types/trading_types.py",
        "src/custom_types/validation.py",
        
        # Database
        "src/database/efficient_features_database.py",
        "src/database/firestore_manager.py",
        "src/database/influxdb_manager.py",
        "src/database/migration_utils.py",
        "src/database/precomputed_features_manager.py",
        
        # Exchange
        "src/exchange/binance.py",
        
        # Integration
        "src/integration/paper_trading_integration.py",
        
        # Interfaces
        "src/interfaces/__init__.py",
        
        # Monitoring
        "src/monitoring/__init__.py",
        
        # Optimization
        "src/optimization/parameter_optimizer.py",
        
        # Pipelines
        "src/pipelines/__init__.py",
        "src/pipelines/base_pipeline.py",
        "src/pipelines/improved_pipeline_executor.py",
        "src/pipelines/live_trading_pipeline.py",
        
        # Protocols
        "src/protocols/trading_protocols.py",
        
        # Reports
        "src/reports/paper_trading_reporter.py",
        
        # Sentinel
        "src/sentinel/__init__.py",
        "src/sentinel/sentinel.py",
        
        # Strategist
        "src/strategist/__init__.py",
        
        # Tracking
        "src/tracking/trade_tracker.py",
        
        # Trading
        "src/trading/live_wavelet_analyzer.py",
        "src/trading/live_wavelet_integration.py",
        "src/trading/sr_trading_intelligence.py",
        
        # Training
        "src/training/__init__.py",
        "src/training/enhanced_training_manager_enhanced.py",
        "src/training/optimization_manager.py",
        
        # Training core
        "src/training/core/__init__.py",
        "src/training/core/checkpoint_manager.py",
        "src/training/core/pipeline_base.py",
        "src/training/core/pipeline_orchestrator.py",
        "src/training/core/stage_context.py",
        "src/training/core/stage_registry.py",
        
        # Training steps
        "src/training/steps/__init__.py",
        
        # Validation
        "src/validation/critical_path_validators.py",
        
        # Components
        "src/components/__init__.py",
        
        # And many more from the detailed list...
    }
    return unused_files

def search_for_references(file_path: str, search_term: str) -> List[Tuple[str, int, str]]:
    """Search for references to a file in the codebase."""
    references = []
    workspace_path = "/workspace"
    
    for root, dirs, files in os.walk(workspace_path):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if search_term in line:
                                relative_path = os.path.relpath(full_path, workspace_path)
                                references.append((relative_path, line_num, line.strip()))
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
    
    return references

def check_file_references():
    """Check if unused files are referenced anywhere in the codebase."""
    print("🔍 Checking for references to unused files...")
    
    unused_files = get_unused_files()
    referenced_files = {}
    safe_to_delete = []
    
    for file_path in unused_files:
        # Extract the module name from the file path
        if file_path.endswith('.py'):
            module_name = file_path.replace('/', '.').replace('.py', '')
            file_name = os.path.basename(file_path)
            
            # Search for various ways the file might be referenced
            search_terms = [
                file_name,
                module_name,
                file_path,
                os.path.splitext(file_name)[0],  # filename without extension
            ]
            
            references = []
            for search_term in search_terms:
                refs = search_for_references(file_path, search_term)
                references.extend(refs)
            
            # Remove self-references
            references = [ref for ref in references if ref[0] != file_path]
            
            if references:
                referenced_files[file_path] = references
                print(f"⚠️  {file_path} is referenced in:")
                for ref_file, line_num, line in references[:3]:  # Show first 3 references
                    print(f"    {ref_file}:{line_num} - {line[:100]}...")
                if len(references) > 3:
                    print(f"    ... and {len(references) - 3} more references")
            else:
                safe_to_delete.append(file_path)
                print(f"✅ {file_path} - SAFE TO DELETE")
    
    # Save results
    with open("safe_to_delete_files.txt", "w") as f:
        f.write("FILES SAFE TO DELETE\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total files checked: {len(unused_files)}\n")
        f.write(f"Safe to delete: {len(safe_to_delete)}\n")
        f.write(f"Referenced elsewhere: {len(referenced_files)}\n\n")
        
        f.write("SAFE TO DELETE:\n")
        f.write("-" * 20 + "\n")
        for file_path in sorted(safe_to_delete):
            f.write(f"{file_path}\n")
        
        f.write("\n\nREFERENCED ELSEWHERE:\n")
        f.write("-" * 25 + "\n")
        for file_path, references in sorted(referenced_files.items()):
            f.write(f"\n{file_path}:\n")
            for ref_file, line_num, line in references:
                f.write(f"  {ref_file}:{line_num} - {line}\n")
    
    print(f"\n📊 Summary:")
    print(f"  Total files checked: {len(unused_files)}")
    print(f"  Safe to delete: {len(safe_to_delete)}")
    print(f"  Referenced elsewhere: {len(referenced_files)}")
    print(f"\n📄 Detailed report saved to: safe_to_delete_files.txt")
    
    return safe_to_delete, referenced_files

if __name__ == "__main__":
    check_file_references()