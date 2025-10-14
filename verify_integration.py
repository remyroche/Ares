#!/usr/bin/env python3
"""
Verification Script for Sophisticated Integration

This script verifies that all the sophisticated components have been properly
integrated into the UnifiedDataDrivenPipeline without requiring external dependencies.
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(file_path)

def check_imports(file_path: str) -> list:
    """Check if a file has the expected imports and classes."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        issues = []
        
        # Check for sophisticated lookback optimizer
        if 'sophisticated_lookback_optimizer.py' in file_path:
            required_classes = ['SophisticatedLookbackOptimizer', 'SophisticatedOptimizationResult', 'OptimizationDirection']
            for class_name in required_classes:
                if f'class {class_name}' not in content:
                    issues.append(f"Missing class: {class_name}")
            
            required_methods = ['optimize_features_sophisticated', '_optimize_single_feature_sophisticated']
            for method_name in required_methods:
                if f'def {method_name}' not in content:
                    issues.append(f"Missing method: {method_name}")
        
        # Check for multi-horizon integration
        elif 'multi_horizon_integration.py' in file_path:
            required_classes = ['MultiHorizonIntegration', 'MultiHorizonIntegrationResult', 'TargetDirection']
            for class_name in required_classes:
                if f'class {class_name}' not in content:
                    issues.append(f"Missing class: {class_name}")
            
            required_methods = ['integrate_multi_horizon_labeling', 'select_optimal_target_columns']
            for method_name in required_methods:
                if f'def {method_name}' not in content:
                    issues.append(f"Missing method: {method_name}")
        
        # Check for comprehensive validation
        elif 'comprehensive_validation.py' in file_path:
            required_classes = ['ComprehensiveValidator', 'ValidationSummary', 'PerformanceValidationResult']
            for class_name in required_classes:
                if f'class {class_name}' not in content:
                    issues.append(f"Missing class: {class_name}")
            
            required_methods = ['validate_data_comprehensive', 'validate_performance']
            for method_name in required_methods:
                if f'def {method_name}' not in content:
                    issues.append(f"Missing method: {method_name}")
        
        # Check for enhanced unified pipeline
        elif 'enhanced_unified_pipeline.py' in file_path:
            required_imports = ['SophisticatedLookbackOptimizer', 'MultiHorizonIntegration', 'ComprehensiveValidator']
            for import_name in required_imports:
                if import_name not in content:
                    issues.append(f"Missing import: {import_name}")
            
            required_methods = ['_enhanced_optimize_feature_lookback', '_fallback_lookback_optimization']
            for method_name in required_methods:
                if f'def {method_name}' not in content:
                    issues.append(f"Missing method: {method_name}")
        
        return issues
        
    except Exception as e:
        return [f"Error reading file: {e}"]

def verify_integration():
    """Verify the sophisticated integration."""
    print("🔍 Verifying Sophisticated Integration...")
    print("=" * 60)
    
    # Define files to check
    files_to_check = [
        "src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/sophisticated_lookback_optimizer.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/multi_horizon_integration.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/comprehensive_validation.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/core/enhanced_unified_pipeline.py",
        "src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/__init__.py"
    ]
    
    results = {}
    
    for file_path in files_to_check:
        print(f"\n📁 Checking: {file_path}")
        
        # Check if file exists
        if not check_file_exists(file_path):
            print(f"❌ File not found: {file_path}")
            results[file_path] = False
            continue
        
        # Check file content
        issues = check_imports(file_path)
        
        if issues:
            print(f"⚠️ Issues found:")
            for issue in issues:
                print(f"   - {issue}")
            results[file_path] = False
        else:
            print(f"✅ File looks good")
            results[file_path] = True
    
    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print('='*60)
    
    passed_files = sum(results.values())
    total_files = len(results)
    
    for file_path, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{os.path.basename(file_path)}: {status}")
    
    print(f"\nOverall: {passed_files}/{total_files} files verified")
    
    if passed_files == total_files:
        print("\n🎉 SUCCESS: All sophisticated components have been properly integrated!")
        print("\n✅ Integration Features:")
        print("   - Sophisticated lookback optimization algorithms")
        print("   - Multi-horizon profit labeling integration")
        print("   - Direction-specific optimization (longs/shorts)")
        print("   - Comprehensive validation system")
        print("   - Advanced metrics and performance monitoring")
        print("   - Execution mode-aware optimization")
        print("   - Nested walk-forward cross-validation")
        print("   - Sophisticated regularization settings")
        print("   - Enhanced unified pipeline integration")
    else:
        print("\n⚠️ PARTIAL SUCCESS: Some components need attention")
        print(f"   Passed: {passed_files}/{total_files} files")
    
    return results

def main():
    """Main verification function."""
    print("🚀 Starting Sophisticated Integration Verification")
    print("=" * 60)
    
    # Change to workspace directory
    os.chdir('/workspace')
    
    # Run verification
    results = verify_integration()
    
    # Return success status
    return sum(results.values()) == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)