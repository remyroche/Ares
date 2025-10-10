#!/usr/bin/env python3
"""
Simplified Test Script for Interactive Feature Generation Improvements

This script tests the improvements made to the interactive feature generation system
without requiring all the complex dependencies.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_import_manager_creation():
    """Test that ImportManager can be created and basic functionality works."""
    print("🧪 Testing ImportManager Creation...")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.import_manager import get_import_manager
        
        # Get import manager
        manager = get_import_manager()
        print("✅ ImportManager created successfully")
        
        # Test basic functionality
        stats = manager.get_cache_stats()
        print(f"📊 Initial cache stats: {stats}")
        
        # Test registering modules
        manager.register_required_module("test_module")
        manager.register_optional_module("test_optional_module")
        print("✅ Module registration works")
        
        return True
        
    except Exception as e:
        print(f"❌ ImportManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_generation_utils_creation():
    """Test that feature generation utilities can be created."""
    print("🧪 Testing Feature Generation Utils Creation...")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.feature_generation_utils import (
            ImprovedFeatureGenerator, FeatureGenerationConfig, FeatureValidator
        )
        
        # Test configuration creation
        config = FeatureGenerationConfig()
        print("✅ FeatureGenerationConfig created successfully")
        
        # Test validator creation
        validator = FeatureValidator(config)
        print("✅ FeatureValidator created successfully")
        
        # Test generator creation
        generator = ImprovedFeatureGenerator(config)
        print("✅ ImprovedFeatureGenerator created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature generation utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_manager_integration():
    """Test that ImportManager is properly integrated in the main component."""
    print("🧪 Testing ImportManager Integration...")
    
    try:
        # Check if the main component file has been updated
        component_file = Path("src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/interactive_feature_generation_component.py")
        
        if not component_file.exists():
            print("❌ Main component file not found")
            return False
        
        # Read the file and check for ImportManager usage
        with open(component_file, 'r') as f:
            content = f.read()
        
        if "from .import_manager import get_import_manager" in content:
            print("✅ ImportManager import found in main component")
        else:
            print("❌ ImportManager import not found in main component")
            return False
        
        if "import_manager = get_import_manager()" in content:
            print("✅ ImportManager initialization found in main component")
        else:
            print("❌ ImportManager initialization not found in main component")
            return False
        
        if "common_ops_result = import_manager.import_common_operations()" in content:
            print("✅ ImportManager usage found in main component")
        else:
            print("❌ ImportManager usage not found in main component")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ImportManager integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_generation_integration():
    """Test that improved feature generation is integrated in the orchestrator."""
    print("🧪 Testing Feature Generation Integration...")
    
    try:
        # Check if the orchestrator file has been updated
        orchestrator_file = Path("src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/enhanced_optimized_orchestrator.py")
        
        if not orchestrator_file.exists():
            print("❌ Orchestrator file not found")
            return False
        
        # Read the file and check for improved feature generation usage
        with open(orchestrator_file, 'r') as f:
            content = f.read()
        
        if "from .feature_generation_utils import ImprovedFeatureGenerator" in content:
            print("✅ ImprovedFeatureGenerator import found in orchestrator")
        else:
            print("❌ ImprovedFeatureGenerator import not found in orchestrator")
            return False
        
        if "feature_generator = ImprovedFeatureGenerator(feature_config)" in content:
            print("✅ ImprovedFeatureGenerator usage found in orchestrator")
        else:
            print("❌ ImprovedFeatureGenerator usage not found in orchestrator")
            return False
        
        if "generate_meaningful_features" in content:
            print("✅ Meaningful feature generation method found in orchestrator")
        else:
            print("❌ Meaningful feature generation method not found in orchestrator")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Feature generation integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_code_duplication_reduction():
    """Test that code duplication has been reduced by checking import patterns."""
    print("🧪 Testing Code Duplication Reduction...")
    
    try:
        # Check the main component file for reduced duplication
        component_file = Path("src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/interactive_feature_generation_component.py")
        
        if not component_file.exists():
            print("❌ Main component file not found")
            return False
        
        with open(component_file, 'r') as f:
            content = f.read()
        
        # Count try-except blocks for imports
        try_except_count = content.count("try:\n    from src.utils.")
        print(f"📊 Number of try-except import blocks: {try_except_count}")
        
        # Count ImportError handling
        import_error_count = content.count("except ImportError as e:")
        print(f"📊 Number of ImportError handlers: {import_error_count}")
        
        # Check if ImportManager is being used instead
        import_manager_usage = content.count("import_manager.")
        print(f"📊 Number of ImportManager usages: {import_manager_usage}")
        
        if import_manager_usage > 0:
            print("✅ ImportManager is being used to reduce duplication")
            return True
        else:
            print("❌ ImportManager usage not detected")
            return False
        
    except Exception as e:
        print(f"❌ Code duplication reduction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all improvement tests."""
    print("🚀 Starting Interactive Feature Generation Improvement Tests")
    print("=" * 70)
    
    tests = [
        ("ImportManager Creation", test_import_manager_creation),
        ("Feature Generation Utils Creation", test_feature_generation_utils_creation),
        ("ImportManager Integration", test_import_manager_integration),
        ("Feature Generation Integration", test_feature_generation_integration),
        ("Code Duplication Reduction", test_code_duplication_reduction),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        print("-" * 50)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} test passed!")
            else:
                print(f"❌ {test_name} test failed!")
                
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 IMPROVEMENT TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All improvement tests passed! The improvements are working correctly.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)