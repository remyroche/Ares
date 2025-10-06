"""
Verification Check for PID-Based Feature Generation Integration

This script verifies that the PID-based feature generation is properly integrated
into the market analysis sub-pipeline without requiring external dependencies.
"""

import os
import sys
from pathlib import Path


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists and report the result."""
    exists = os.path.exists(file_path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {file_path}")
    return exists


def check_import_in_file(file_path: str, import_statement: str, description: str) -> bool:
    """Check if an import statement exists in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            found = import_statement in content
            status = "✅" if found else "❌"
            print(f"{status} {description}: {import_statement}")
            return found
    except FileNotFoundError:
        print(f"❌ {description}: File not found - {file_path}")
        return False


def check_string_in_file(file_path: str, search_string: str, description: str) -> bool:
    """Check if a string exists in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            found = search_string in content
            status = "✅" if found else "❌"
            print(f"{status} {description}: {search_string}")
            return found
    except FileNotFoundError:
        print(f"❌ {description}: File not found - {file_path}")
        return False


def main():
    """Run verification checks."""
    
    print("🔍 PID-Based Feature Generation Integration Verification")
    print("=" * 60)
    
    # Set workspace root
    workspace_root = Path("/workspace")
    
    # Check 1: Verify PID-based feature generation directory exists
    print("\n📁 Directory Structure:")
    pid_dir = workspace_root / "src/training/steps/pre_training/pid_based_feature_generation"
    check_file_exists(str(pid_dir), "PID-based feature generation directory")
    
    # Check 2: Verify key files exist
    print("\n📄 Key Files:")
    key_files = [
        ("pid_based_feature_generation_component.py", "Main PID component"),
        ("feature_selection_mechanism.py", "Feature selection mechanism"),
        ("pid_based_feature_orchestrator.py", "Feature orchestrator"),
        ("interaction_feature_generator.py", "Interaction feature generator"),
        # ("polynomial_feature_generator.py", "Polynomial feature generator"),  # Removed due to empty except blocks
        ("cross_timeframe_feature_generator.py", "Cross-timeframe feature generator"),
        ("optimized_lookback_integration.py", "Lookback integration"),
        ("__init__.py", "Package initialization")
    ]
    
    for filename, description in key_files:
        file_path = pid_dir / filename
        check_file_exists(str(file_path), description)
    
    # Check 3: Verify sub-pipeline integration
    print("\n🔗 Sub-Pipeline Integration:")
    sub_pipeline_file = workspace_root / "src/training/steps/pre_training/sub_pipeline.py"
    
    sub_pipeline_checks = [
        ("pid_based_feature_generation", "PID-based feature generation stage name"),
        ("PID-Based Feature Generation", "PID-based feature generation stage description"),
        ("pid_based_feature_generation_result", "PID-based feature generation artifact"),
        ("combined_features", "Combined features extraction"),
        ("feature_importance_scores", "Feature importance scores extraction"),
        ("quality_metrics", "Quality metrics extraction"),
        ("optimization_metrics", "Optimization metrics extraction")
    ]
    
    for search_string, description in sub_pipeline_checks:
        check_string_in_file(str(sub_pipeline_file), search_string, description)
    
    # Check 4: Verify component factory integration
    print("\n🏭 Component Factory Integration:")
    factory_file = workspace_root / "src/training/steps/market_analysis/components/component_factory.py"
    
    factory_checks = [
        ("pid_based_feature_generation", "PID-based feature generation component registration"),
        ("PIDBasedFeatureGenerationComponent", "PID component import"),
        ("PID_COMPONENT_AVAILABLE", "PID component availability check")
    ]
    
    for search_string, description in factory_checks:
        check_string_in_file(str(factory_file), search_string, description)
    
    # Check 5: Verify backward compatibility
    print("\n🔄 Backward Compatibility:")
    adapter_file = workspace_root / "src/training/steps/market_analysis/components/cross_timeframe_analysis.py"
    
    compatibility_checks = [
        ("PIDBasedFeatureGenerationComponent", "Adapter imports PID component"),
        ("CrossTimeframeAnalysisComponent", "Adapter class definition"),
        ("backward compatibility", "Backward compatibility comment")
    ]
    
    for search_string, description in compatibility_checks:
        check_string_in_file(str(adapter_file), search_string, description)
    
    # Check 6: Verify artifact requirements
    print("\n📊 Artifact Requirements:")
    artifact_checks = [
        ("pid_based_feature_generation_result", "PID-based feature generation artifact requirement")
    ]
    
    for search_string, description in artifact_checks:
        check_string_in_file(str(sub_pipeline_file), search_string, description)
    
    # Check 7: Verify main module exports
    print("\n📤 Module Exports:")
    main_init_file = workspace_root / "src/training/steps/market_analysis/__init__.py"
    
    export_checks = [
        ("FeatureSelectionMechanism", "Feature selection mechanism export"),
        ("PIDBasedFeatureOrchestrator", "PID orchestrator export"),
        ("OptimizedLookbackIntegration", "Lookback integration export"),
        ("SelectionStrategy", "Selection strategy export")
    ]
    
    for search_string, description in export_checks:
        check_string_in_file(str(main_init_file), search_string, description)
    
    # Check 8: Verify PID package exports
    print("\n📦 PID Package Exports:")
    pid_init_file = pid_dir / "__init__.py"
    
    pid_export_checks = [
        ("FeatureSelectionMechanism", "Feature selection mechanism in PID package"),
        ("PIDBasedFeatureOrchestrator", "PID orchestrator in PID package"),
        ("OptimizedLookbackIntegration", "Lookback integration in PID package"),
        ("SelectionStrategy", "Selection strategy in PID package")
    ]
    
    for search_string, description in pid_export_checks:
        check_string_in_file(str(pid_init_file), search_string, description)
    
    # Check 9: Verify documentation
    print("\n📚 Documentation:")
    doc_files = [
        ("FEATURE_SELECTION_GUIDE.md", "Feature selection guide"),
        ("FEATURE_SELECTION_ANSWERS.md", "Feature selection answers"),
        ("DYNAMIC_THRESHOLD_EXAMPLE.md", "Dynamic threshold example"),
        ("UPGRADE_SUMMARY.md", "Upgrade summary"),
        ("README.md", "Main README")
    ]
    
    for filename, description in doc_files:
        file_path = pid_dir / filename
        check_file_exists(str(file_path), description)
    
    print("\n" + "=" * 60)
    print("🎯 Verification Complete!")
    print("\nIf all checks show ✅, the PID-based feature generation is properly integrated.")
    print("If any checks show ❌, those areas need attention.")


if __name__ == "__main__":
    main()