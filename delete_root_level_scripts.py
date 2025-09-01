#!/usr/bin/env python3
"""
Script to safely delete root-level analysis/debugging scripts
"""

import os
import shutil
from pathlib import Path

def get_root_level_scripts_to_delete():
    """Get the list of root-level scripts that are safe to delete."""
    return {
        # Analysis scripts
        "analyze_complete_training_execution.py",
        "analyze_step1_execution.py",
        "analyze_strict_thresholds.py",
        "analyze_trading_execution.py",
        "analyze_unused_files.py",
        "analyze_validation_issues.py",
        
        # Automated fixers
        "automated_syntax_fixer.py",
        "comprehensive_code_quality_fixer.py",
        "comprehensive_fix.py",
        "comprehensive_gap_filler.py",
        "comprehensive_gap_filler_v2.py",
        "comprehensive_syntax_fixer.py",
        "comprehensive_training_fix.py",
        "conservative_syntax_fixer.py",
        "enhanced_syntax_fixer.py",
        "final_fix.py",
        "final_fix_script.py",
        "final_targeted_fix.py",
        "final_targeted_fix_v2.py",
        "final_targeted_fix_v3.py",
        "final_utils_fix.py",
        "fix_syntax_errors.py",
        "fix_training_placeholders.py",
        "targeted_fix.py",
        "targeted_fix_training_placeholders.py",
        "targeted_syntax_fixer.py",
        "universal_syntax_fixer.py",
        
        # Check and verification scripts
        "check_existing_data.py",
        "check_trade_data.py",
        "verify_aggtrades_downloads.py",
        "verify_training_modes.py",
        
        # Cleanup scripts
        "cleanup_actions.py",
        "cleanup_script.py",
        "dead_code_remover.py",
        
        # Completion scripts
        "complete_remaining_16_steps.py",
        "complete_remaining_steps.py",
        "complete_remaining_steps_integration.py",
        
        # Consolidation scripts
        "consolidate_aggtrades.py",
        "consolidate_data.py",
        "convert_csv_to_parquet.py",
        
        # Creation scripts
        "create_30m_hmm_artifacts.py",
        "create_correct_mock_data.py",
        "create_regime_splits.py",
        
        # Debug scripts
        "debug_clustering.py",
        "debug_hmm_combinations.py",
        "debug_interaction_flow.py",
        "debug_low_variance_features.py",
        "debug_metadata_detection.py",
        
        # Diagnosis scripts
        "detect_and_fill_gaps_immediate.py",
        "diagnose_feature_pipeline.py",
        "diagnose_interaction_features.py",
        "diagnose_regime_data.py",
        
        # Download scripts
        "download_aggtrades_range.py",
        "download_futures_only.py",
        "download_missing_aggtrades_2023_2024.py",
        "download_missing_aggtrades_days.py",
        "download_missing_data.py",
        "download_missing_futures.py",
        "download_remaining_aggtrades.py",
        "download_specific_missing_data.py",
        
        # Enhanced validation scripts
        "enhanced_validation_logging.py",
        "enhanced_validation_wrapper.py",
        
        # Extraction scripts
        "extract_feature_details.py",
        "feature_analysis_script.py",
        "feature_specific_validation.py",
        
        # Gap filling scripts
        "gap_filler_clean.py",
        "identify_deleted_aggtrades.py",
        
        # Implementation scripts
        "implement_feature_specific_validation.py",
        
        # Kelly criterion
        "kelly_criterion_formula.py",
        
        # Optimization scripts
        "multi_objective_hmm_optimizer.py",
        "optimize_hmm_regime_parameters.py",
        "optimize_hmm_regime_parameters_advanced.py",
        "optimize_hmm_regime_parameters_enhanced.py",
        
        # Quick scripts
        "quick_error_scanner.py",
        "quick_reference_check.py",
        
        # Run scripts
        "run_30m_hmm_step.py",
        "run_code_quality_tools.py",
        "run_fixed_hmm_regime_discovery.py",
        "run_pipeline_simple.py",
        "run_step2_direct.py",
        "run_syntax_fix.py",
        
        # Simulation scripts
        "simulate_regime_merging_from_existing_data.py",
        "simulate_regime_merging_optimization.py",
        
        # Standardization scripts
        "standardize_remaining_steps.py",
        "standardize_utility_modules.py",
        
        # Syntax scripts
        "syntax_error_scanner.py",
        
        # Test scripts
        "test_advanced_models_core.py",
        
        # Update scripts
        "update_aggtrades_gaps.py",
        "update_all_steps_mlflow_integration.py",
        "update_training_analysis.py",
        
        # Analysis scripts
        "accurate_unused_files_analysis.py",
        "check_unused_files_references.py",
    }

def delete_files_safely():
    """Delete the root-level scripts safely."""
    print("🗑️ Starting deletion of root-level analysis/debugging scripts...")
    
    files_to_delete = get_root_level_scripts_to_delete()
    deleted_files = []
    failed_deletions = []
    
    workspace_path = Path("/workspace")
    
    for filename in files_to_delete:
        file_path = workspace_path / filename
        
        if file_path.exists():
            try:
                # Delete the file
                file_path.unlink()
                deleted_files.append(filename)
                print(f"✅ Deleted: {filename}")
            except Exception as e:
                failed_deletions.append((filename, str(e)))
                print(f"❌ Failed to delete {filename}: {e}")
        else:
            print(f"⚠️ File not found: {filename}")
    
    # Save deletion report
    with open("deletion_report.txt", "w") as f:
        f.write("ROOT-LEVEL SCRIPTS DELETION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total files attempted: {len(files_to_delete)}\n")
        f.write(f"Successfully deleted: {len(deleted_files)}\n")
        f.write(f"Failed deletions: {len(failed_deletions)}\n\n")
        
        f.write("SUCCESSFULLY DELETED:\n")
        f.write("-" * 25 + "\n")
        for filename in sorted(deleted_files):
            f.write(f"{filename}\n")
        
        if failed_deletions:
            f.write("\n\nFAILED DELETIONS:\n")
            f.write("-" * 20 + "\n")
            for filename, error in failed_deletions:
                f.write(f"{filename}: {error}\n")
    
    print(f"\n📊 Deletion Summary:")
    print(f"  Total files attempted: {len(files_to_delete)}")
    print(f"  Successfully deleted: {len(deleted_files)}")
    print(f"  Failed deletions: {len(failed_deletions)}")
    print(f"\n📄 Detailed report saved to: deletion_report.txt")
    
    return deleted_files, failed_deletions

if __name__ == "__main__":
    delete_files_safely()