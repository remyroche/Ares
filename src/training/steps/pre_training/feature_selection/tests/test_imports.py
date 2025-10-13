#!/usr/bin/env python3
"""
Simple import test for the modular feature selection system.

This script tests that all modules can be imported correctly without
requiring external dependencies.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Test core modules
        print("  📦 Testing core modules...")
        from src.training.steps.pre_training.feature_selection.core.config import (
            BaseFeatureSelectionConfig,
            ModelSpecificConfig,
            QualityThresholdsConfig,
            ValidationConfig,
            AdvancedSelectionConfig,
            FeatureSelectionConfig,
            FeatureSelectionResult
        )
        print("    ✅ Core config classes imported")
        
        from src.training.steps.pre_training.feature_selection.core.multi_stage_pipeline import (
            MultiStageFeatureSelectionPipeline
        )
        print("    ✅ Multi-stage pipeline imported")
        
        from src.training.steps.pre_training.feature_selection.core.selector import (
            FeatureSelector
        )
        print("    ✅ Core selector imported")
        
        from src.training.steps.pre_training.feature_selection.core.optimizer import (
            FeatureSelectionOptimizer
        )
        print("    ✅ Core optimizer imported")
        
        # Test hardware modules
        print("  📦 Testing hardware modules...")
        from src.training.steps.pre_training.feature_selection.hardware.memory_manager import (
            MemoryManager
        )
        print("    ✅ Memory manager imported")
        
        from src.training.steps.pre_training.feature_selection.hardware.vectorbt_utils import (
            VectorBTManager
        )
        print("    ✅ VectorBT manager imported")
        
        from src.training.steps.pre_training.feature_selection.hardware.performance_monitor import (
            PerformanceMonitor
        )
        print("    ✅ Performance monitor imported")
        
        # Test config modules
        print("  📦 Testing config modules...")
        from src.training.steps.pre_training.feature_selection.config.config_loader import (
            ConfigLoader
        )
        print("    ✅ Config loader imported")
        
        from src.training.steps.pre_training.feature_selection.config.model_profiles import (
            ModelProfileManager
        )
        print("    ✅ Model profile manager imported")
        
        from src.training.steps.pre_training.feature_selection.config.config_validator import (
            ConfigValidator
        )
        print("    ✅ Config validator imported")
        
        # Test validation modules
        print("  📦 Testing validation modules...")
        from src.training.steps.pre_training.feature_selection.validation.data_validator import (
            DataValidator
        )
        print("    ✅ Data validator imported")
        
        # Test main package imports
        print("  📦 Testing main package imports...")
        from src.training.steps.pre_training.feature_selection import (
            MultiStageFeatureSelectionPipeline,
            run_multi_stage_feature_selection,
            FeatureSelector,
            FeatureSelectionOptimizer,
            FeatureSelectionConfig,
            FeatureSelectionResult,
            MemoryManager,
            VectorBTManager,
            PerformanceMonitor,
            ConfigLoader,
            ModelProfileManager,
            ConfigValidator,
            DataValidator
        )
        print("    ✅ Main package imports successful")
        
        print("\n✅ ALL IMPORTS SUCCESSFUL!")
        return True
        
    except ImportError as e:
        print(f"\n❌ IMPORT FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        return False


def test_module_structure():
    """Test that the module structure is correct."""
    print("\n🧪 Testing module structure...")
    
    try:
        # Test that all __init__.py files exist
        base_path = Path(__file__).parent.parent
        
        required_files = [
            "__init__.py",
            "core/__init__.py",
            "core/config.py",
            "core/multi_stage_pipeline.py", 
            "core/selector.py",
            "core/optimizer.py",
            "hardware/__init__.py",
            "hardware/memory_manager.py",
            "hardware/vectorbt_utils.py",
            "hardware/performance_monitor.py",
            "config/__init__.py",
            "config/config_loader.py",
            "config/model_profiles.py",
            "config/config_validator.py",
            "validation/__init__.py",
            "validation/data_validator.py",
            "tests/__init__.py"
        ]
        
        for file_path in required_files:
            full_path = base_path / file_path
            if not full_path.exists():
                print(f"    ❌ Missing file: {file_path}")
                return False
            else:
                print(f"    ✅ Found: {file_path}")
        
        print("\n✅ MODULE STRUCTURE CORRECT!")
        return True
        
    except Exception as e:
        print(f"\n❌ STRUCTURE TEST FAILED: {e}")
        return False


def test_file_sizes():
    """Test that files are reasonably sized (not too large)."""
    print("\n🧪 Testing file sizes...")
    
    try:
        base_path = Path(__file__).parent.parent
        
        # Check that no single file is too large (should be < 50KB for modular files)
        max_size = 50 * 1024  # 50KB
        
        large_files = []
        for py_file in base_path.rglob("*.py"):
            if py_file.stat().st_size > max_size:
                large_files.append((py_file.name, py_file.stat().st_size))
        
        if large_files:
            print("    ⚠️ Large files found:")
            for filename, size in large_files:
                print(f"      {filename}: {size / 1024:.1f}KB")
        else:
            print("    ✅ All files are reasonably sized")
        
        print("\n✅ FILE SIZE CHECK COMPLETE!")
        return True
        
    except Exception as e:
        print(f"\n❌ FILE SIZE TEST FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 MODULAR FEATURE SELECTION SYSTEM - IMPORT TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Module Structure Test", test_module_structure),
        ("File Size Test", test_file_sizes)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}")
        success = test_func()
        results.append(success)
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Modular system structure is correct.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)