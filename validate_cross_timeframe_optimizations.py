#!/usr/bin/env python3
"""
Validation script for Cross-Timeframe VectorBT Optimizations

This script validates that the cross_timeframe.py file has been properly
optimized with VectorBTRollingOptimizer and UnifiedVectorizationManager.
"""

import os
import re
from typing import List, Dict, Any

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(file_path)

def read_file_content(file_path: str) -> str:
    """Read file content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def validate_imports(content: str) -> Dict[str, bool]:
    """Validate that proper imports are present."""
    validations = {
        'vectorbt_import': 'import vectorbt as vbt' in content,
        'rolling_optimizer_import': 'from ..utils.vectorbt_rolling_optimizer import' in content,
        'vectorization_optimizer_import': 'from ..utils.vectorization_optimizer import' in content,
        'unified_vectorization_available': 'UNIFIED_VECTORIZATION_AVAILABLE' in content,
        'rolling_optimizer_available': 'ROLLING_OPTIMIZER_AVAILABLE' in content
    }
    return validations

def validate_class_optimizations(content: str) -> Dict[str, bool]:
    """Validate that classes are properly optimized."""
    validations = {
        'cross_timeframe_generator_optimized': 'self.rolling_optimizer = get_vectorbt_rolling_optimizer(' in content,
        'vectorization_manager_initialized': 'self.vectorization_manager = get_vectorization_optimizer(' in content,
        'performance_stats_enhanced': 'cross_timeframe_features_generated' in content,
        'enhanced_feature_generation': 'generate_enhanced_cross_timeframe_features' in content,
        'vectorbt_optimized_features': '_generate_vectorbt_optimized_features' in content,
        'unified_vectorization_features': '_generate_unified_vectorization_features' in content,
        'performance_report_method': 'get_performance_report' in content
    }
    return validations

def validate_method_optimizations(content: str) -> Dict[str, bool]:
    """Validate that methods use VectorBT optimizations."""
    validations = {
        'optimize_dataframe_processing_enhanced': 'self.vectorization_manager:' in content,
        'vectorized_rolling_operations_enhanced': 'self.vectorization_manager.vectorized_rolling_operations' in content,
        'momentum_generator_optimized': 'self.rolling_optimizer.rolling_apply' in content and 'CrossTimeframeMomentumGenerator' in content,
        'volatility_generator_optimized': 'self.rolling_optimizer.rolling_std' in content and 'CrossTimeframeVolatilityGenerator' in content,
        'volume_generator_optimized': 'self.rolling_optimizer.rolling_mean' in content and 'CrossTimeframeVolumeGenerator' in content,
        'performance_tracking_enhanced': 'performance_stats' in content and 'total_execution_time' in content
    }
    return validations

def validate_configuration_enhancements(content: str) -> Dict[str, bool]:
    """Validate configuration enhancements."""
    validations = {
        'gpu_acceleration_enabled': 'enable_gpu=True' in content,
        'parallel_processing_enabled': 'enable_parallel=True' in content,
        'memory_efficient_enabled': 'memory_efficient=True' in content,
        'chunked_processing_configured': 'chunk_size=' in content,
        'aggressive_vectorization': 'vectorization_strategy="aggressive"' in content,
        'enhanced_performance_tracking': 'performance_stats' in content
    }
    return validations

def validate_fallback_mechanisms(content: str) -> Dict[str, bool]:
    """Validate fallback mechanisms are in place."""
    validations = {
        'vectorbt_fallback': 'VECTORBT_AVAILABLE' in content,
        'rolling_optimizer_fallback': 'ROLLING_OPTIMIZER_AVAILABLE' in content,
        'vectorization_fallback': 'UNIFIED_VECTORIZATION_AVAILABLE' in content,
        'pandas_fallback': 'pandas fallback' in content.lower(),
        'error_handling': 'except Exception as e:' in content
    }
    return validations

def generate_report(validations: Dict[str, Dict[str, bool]]) -> str:
    """Generate a validation report."""
    report = []
    report.append("=" * 80)
    report.append("CROSS-TIMEFRAME VECTORBT OPTIMIZATION VALIDATION REPORT")
    report.append("=" * 80)
    
    total_checks = 0
    passed_checks = 0
    
    for category, checks in validations.items():
        report.append(f"\n{category.upper().replace('_', ' ')}:")
        report.append("-" * 50)
        
        for check_name, passed in checks.items():
            total_checks += 1
            if passed:
                passed_checks += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            report.append(f"  {status} {check_name.replace('_', ' ').title()}")
    
    report.append("\n" + "=" * 80)
    report.append(f"SUMMARY: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.1f}%)")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Main validation function."""
    print("🔍 Validating Cross-Timeframe VectorBT Optimizations...")
    
    # Check if the file exists
    file_path = "src/feature_generation/categories/cross_timeframe.py"
    if not check_file_exists(file_path):
        print(f"❌ File not found: {file_path}")
        return 1
    
    # Read file content
    content = read_file_content(file_path)
    if not content:
        print("❌ Could not read file content")
        return 1
    
    print(f"📄 File size: {len(content)} characters")
    print(f"📄 Lines: {len(content.splitlines())}")
    
    # Run validations
    validations = {
        'imports': validate_imports(content),
        'class_optimizations': validate_class_optimizations(content),
        'method_optimizations': validate_method_optimizations(content),
        'configuration_enhancements': validate_configuration_enhancements(content),
        'fallback_mechanisms': validate_fallback_mechanisms(content)
    }
    
    # Generate and print report
    report = generate_report(validations)
    print("\n" + report)
    
    # Check if all critical validations passed
    critical_checks = [
        'vectorbt_import',
        'rolling_optimizer_import',
        'vectorization_optimizer_import',
        'cross_timeframe_generator_optimized',
        'vectorization_manager_initialized',
        'enhanced_feature_generation',
        'gpu_acceleration_enabled',
        'parallel_processing_enabled',
        'memory_efficient_enabled'
    ]
    
    critical_passed = all(
        any(validations[category].get(check, False) for category in validations.keys())
        for check in critical_checks
    )
    
    if critical_passed:
        print("\n🎉 All critical optimizations are in place!")
        return 0
    else:
        print("\n⚠️ Some critical optimizations may be missing.")
        return 1

if __name__ == "__main__":
    exit(main())