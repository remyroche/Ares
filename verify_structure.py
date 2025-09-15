#!/usr/bin/env python3
"""
Simple verification script for the new Feature Lookback Optimization structure.

This script verifies that the reorganization was successful by checking imports.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        # Test main component import
        from src.training.steps.market_analysis.feature_lookback_optimization import (
            FeatureLookbackOptimizationComponent
        )
        print("✅ Main component import successful")
        
        # Test individual module imports
        from src.training.steps.market_analysis.feature_lookback_optimization.optimization_reporter import (
            OptimizationReporter
        )
        print("✅ Optimization reporter import successful")
        
        from src.training.steps.market_analysis.feature_lookback_optimization.validation_framework import (
            ValidationFramework, ValidationLevel, ValidationStatus
        )
        print("✅ Validation framework import successful")
        
        from src.training.steps.market_analysis.feature_lookback_optimization.dependency_manager import (
            DependencyManager, get_dependency, is_dependency_available
        )
        print("✅ Dependency manager import successful")
        
        from src.training.steps.market_analysis.feature_lookback_optimization.monitoring_metrics import (
            MonitoringMetrics, MetricType, MetricLevel
        )
        print("✅ Monitoring metrics import successful")
        
        # Test package-level imports
        from src.training.steps.market_analysis.feature_lookback_optimization import (
            FeatureLookbackOptimizationComponent,
            OptimizationReporter,
            ValidationFramework,
            ValidationLevel,
            ValidationStatus,
            DependencyManager,
            get_dependency,
            is_dependency_available,
            MonitoringMetrics,
            MetricType,
            MetricLevel
        )
        print("✅ Package-level imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that the file structure is correct."""
    print("🧪 Testing file structure...")
    
    base_path = Path("src/training/steps/market_analysis/feature_lookback_optimization")
    
    required_files = [
        "__init__.py",
        "feature_lookback_optimization.py",
        "optimization_reporter.py",
        "validation_framework.py",
        "dependency_manager.py",
        "monitoring_metrics.py"
    ]
    
    for file_name in required_files:
        file_path = base_path / file_name
        if file_path.exists():
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    return True

def test_old_files_removed():
    """Test that old files have been removed."""
    print("🧪 Testing old files removal...")
    
    old_path = Path("src/training/steps/market_analysis/components")
    
    old_files = [
        "feature_lookback_optimization.py",
        "optimization_reporter.py",
        "validation_framework.py",
        "dependency_manager.py",
        "monitoring_metrics.py"
    ]
    
    for file_name in old_files:
        file_path = old_path / file_name
        if not file_path.exists():
            print(f"✅ {file_name} removed from components directory")
        else:
            print(f"❌ {file_name} still exists in components directory")
            return False
    
    return True

def main():
    """Run all verification tests."""
    print("🚀 Starting Feature Lookback Optimization structure verification...")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Old Files Removal", test_old_files_removed),
        ("Import Functionality", test_imports),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test passed")
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All verification tests passed! Reorganization successful.")
        return True
    else:
        print("⚠️ Some verification tests failed. Review the structure.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)