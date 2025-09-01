#!/usr/bin/env python3
"""
Quick script to check if unused files are referenced using grep
"""

import os
import subprocess
from typing import Set, Dict, List

def get_unused_files() -> Set[str]:
    """Get a subset of unused files to check."""
    return {
        # Root level analysis scripts (most likely safe to delete)
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
    }

def check_references_with_grep():
    """Check for references using grep."""
    print("🔍 Checking for references to unused files using grep...")
    
    unused_files = get_unused_files()
    safe_to_delete = []
    referenced_files = {}
    
    for file_path in unused_files:
        file_name = os.path.basename(file_path)
        module_name = file_path.replace('/', '.').replace('.py', '')
        
        # Check for references using grep
        try:
            # Search for the filename
            result = subprocess.run(
                ['grep', '-r', '--include="*.py"', file_name, '/workspace'],
                capture_output=True, text=True, timeout=10
            )
            
            # Also check for module imports
            result2 = subprocess.run(
                ['grep', '-r', '--include="*.py"', f'import {module_name}', '/workspace'],
                capture_output=True, text=True, timeout=10
            )
            
            result3 = subprocess.run(
                ['grep', '-r', '--include="*.py"', f'from {module_name}', '/workspace'],
                capture_output=True, text=True, timeout=10
            )
            
            # Combine all results
            all_output = result.stdout + result2.stdout + result3.stdout
            
            # Filter out self-references
            lines = [line for line in all_output.split('\n') if line.strip() and file_path not in line]
            
            if lines:
                referenced_files[file_path] = lines
                print(f"⚠️  {file_path} - REFERENCED ({len(lines)} references)")
            else:
                safe_to_delete.append(file_path)
                print(f"✅ {file_path} - SAFE TO DELETE")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {file_path} - TIMEOUT (skipping)")
        except Exception as e:
            print(f"❌ {file_path} - ERROR: {e}")
    
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
            for ref in references[:5]:  # Show first 5 references
                f.write(f"  {ref}\n")
            if len(references) > 5:
                f.write(f"  ... and {len(references) - 5} more references\n")
    
    print(f"\n📊 Summary:")
    print(f"  Total files checked: {len(unused_files)}")
    print(f"  Safe to delete: {len(safe_to_delete)}")
    print(f"  Referenced elsewhere: {len(referenced_files)}")
    print(f"\n📄 Detailed report saved to: safe_to_delete_files.txt")
    
    return safe_to_delete, referenced_files

if __name__ == "__main__":
    check_references_with_grep()