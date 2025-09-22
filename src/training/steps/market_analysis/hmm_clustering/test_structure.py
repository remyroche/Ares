"""
Simple test to verify HMM clustering consolidation structure.
"""

import os
from pathlib import Path

def test_file_structure():
    """Test that all required files exist."""
    print("🧪 Testing HMM clustering file structure...")
    
    base_path = Path(__file__).parent
    
    required_files = [
        "__init__.py",
        "config.py",
        "README.md",
        "core/__init__.py",
        "core/base_clustering.py",
        "core/matrix_optimized.py",
        "core/enhanced_optimized.py",
        "metrics/__init__.py",
        "metrics/basic_metrics.py",
        "metrics/detailed_metrics.py",
        "metrics/evolution_report.py",
        "metrics/time_series_metrics.py",
        "integration/__init__.py",
        "integration/orchestrator.py",
        "integration/enhanced_integration.py",
        "integration/fast_fail.py",
        "utils/__init__.py",
        "utils/feature_engineering.py",
        "utils/data_processing.py",
        "utils/hardware_optimization.py",
        "utils/memory_management.py",
        "components/__init__.py",
        "components/clustering_component.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print(f"\n🎉 All {len(required_files)} required files exist!")
        return True

def test_import_structure():
    """Test that import statements are correctly structured."""
    print("\n🧪 Testing import structure...")
    
    base_path = Path(__file__).parent
    
    # Test main __init__.py
    init_file = base_path / "__init__.py"
    if init_file.exists():
        content = init_file.read_text()
        required_imports = [
            "MatrixOptimizedClusterer",
            "EnhancedMatrixOptimizedClusterer",
            "OptimalRegimeClusteringOrchestrator",
            "OptimalRegimeClusteringComponent",
            "HMMClusteringConfig"
        ]
        
        missing_imports = []
        for import_name in required_imports:
            if import_name not in content:
                missing_imports.append(import_name)
        
        if missing_imports:
            print(f"❌ Missing imports in __init__.py: {missing_imports}")
            return False
        else:
            print("  ✅ Main __init__.py has all required imports")
    else:
        print("❌ Main __init__.py not found")
        return False
    
    return True

def test_component_factory_integration():
    """Test that component factory integration is correct."""
    print("\n🧪 Testing component factory integration...")
    
    # Check if the import path in component factory is correct
    component_factory_path = Path(__file__).parent.parent / "components" / "component_factory.py"
    
    if component_factory_path.exists():
        content = component_factory_path.read_text()
        
        # Check for correct import path
        if "from ..hmm_clustering.components.clustering_component import OptimalRegimeClusteringComponent" in content:
            print("  ✅ Component factory has correct import path")
            return True
        else:
            print("❌ Component factory import path is incorrect")
            return False
    else:
        print("❌ Component factory file not found")
        return False

def main():
    """Run all structure tests."""
    print("🚀 Starting HMM clustering structure tests...\n")
    
    tests = [
        test_file_structure,
        test_import_structure,
        test_component_factory_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! HMM clustering consolidation is correctly implemented.")
        return True
    else:
        print("❌ Some structure tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)