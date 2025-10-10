#!/usr/bin/env python3
"""
Test Script for Fast-Fail Behavior

This script tests that the interactive feature generation system now uses
fast-fail patterns instead of graceful fallbacks.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_import_manager_fast_fail():
    """Test that ImportManager fails fast on missing dependencies."""
    print("🧪 Testing ImportManager Fast-Fail...")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.import_manager import get_import_manager
        
        manager = get_import_manager()
        
        # Test that it fails fast on missing module
        try:
            manager.safe_import("nonexistent.module", required=True)
            print("❌ Should have failed fast on missing module")
            return False
        except ImportError:
            print("✅ Correctly failed fast on missing module")
            return True
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ ImportManager test failed: {e}")
        return False

def test_feature_generation_fast_fail():
    """Test that feature generation fails fast on invalid input."""
    print("🧪 Testing Feature Generation Fast-Fail...")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.feature_generation_utils import (
            ImprovedFeatureGenerator, FeatureGenerationConfig
        )
        import pandas as pd
        
        # Test with empty data - should fail fast
        try:
            generator = ImprovedFeatureGenerator(FeatureGenerationConfig())
            generator.generate_meaningful_features(pd.DataFrame())
            print("❌ Should have failed fast on empty data")
            return False
        except ValueError as e:
            if "empty" in str(e).lower():
                print("✅ Correctly failed fast on empty data")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
        
        # Test with invalid data - should fail fast
        try:
            invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})  # Missing required columns
            generator.generate_meaningful_features(invalid_data)
            print("❌ Should have failed fast on invalid data")
            return False
        except ValueError as e:
            if "invalid" in str(e).lower() or "missing" in str(e).lower():
                print("✅ Correctly failed fast on invalid data")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Feature generation test failed: {e}")
        return False

def test_interaction_generation_fast_fail():
    """Test that interaction generation fails fast on invalid input."""
    print("🧪 Testing Interaction Generation Fast-Fail...")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.feature_generation_utils import (
            ImprovedFeatureGenerator, FeatureGenerationConfig
        )
        import pandas as pd
        
        generator = ImprovedFeatureGenerator(FeatureGenerationConfig())
        
        # Test with empty data - should fail fast
        try:
            generator.generate_interaction_features(pd.DataFrame())
            print("❌ Should have failed fast on empty data")
            return False
        except ValueError as e:
            if "empty" in str(e).lower():
                print("✅ Correctly failed fast on empty data")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
        
        # Test with insufficient columns - should fail fast
        try:
            single_col_data = pd.DataFrame({'col1': [1, 2, 3]})
            generator.generate_interaction_features(single_col_data)
            print("❌ Should have failed fast on insufficient columns")
            return False
        except ValueError as e:
            if "not enough" in str(e).lower():
                print("✅ Correctly failed fast on insufficient columns")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Interaction generation test failed: {e}")
        return False

def test_cross_timeframe_fast_fail():
    """Test that cross-timeframe generation fails fast on invalid input."""
    print("🧪 Testing Cross-Timeframe Generation Fast-Fail...")
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.feature_generation_utils import (
            ImprovedFeatureGenerator, FeatureGenerationConfig
        )
        import pandas as pd
        
        generator = ImprovedFeatureGenerator(FeatureGenerationConfig())
        
        # Test with empty data - should fail fast
        try:
            generator.generate_cross_timeframe_features(pd.DataFrame())
            print("❌ Should have failed fast on empty data")
            return False
        except ValueError as e:
            if "empty" in str(e).lower():
                print("✅ Correctly failed fast on empty data")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
        
        # Test with no numeric columns - should fail fast
        try:
            no_numeric_data = pd.DataFrame({'text_col': ['a', 'b', 'c']})
            generator.generate_cross_timeframe_features(no_numeric_data)
            print("❌ Should have failed fast on no numeric columns")
            return False
        except ValueError as e:
            if "numeric" in str(e).lower():
                print("✅ Correctly failed fast on no numeric columns")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-timeframe generation test failed: {e}")
        return False

def test_no_fallbacks_in_code():
    """Test that fallback patterns have been removed from the code."""
    print("🧪 Testing No Fallbacks in Code...")
    
    try:
        # Check the main component file
        component_file = Path("src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/interactive_feature_generation_component.py")
        
        if not component_file.exists():
            print("❌ Main component file not found")
            return False
        
        with open(component_file, 'r') as f:
            content = f.read()
        
        # Check for fallback patterns
        fallback_patterns = [
            "fallback",
            "fall back",
            "graceful",
            "try:\n    from",
            "except ImportError as e:\n    tprint_warning",
            "except Exception as e:\n    tprint_warning"
        ]
        
        found_fallbacks = []
        for pattern in fallback_patterns:
            if pattern in content.lower():
                found_fallbacks.append(pattern)
        
        if found_fallbacks:
            print(f"⚠️ Found fallback patterns: {found_fallbacks}")
            return False
        else:
            print("✅ No fallback patterns found in main component")
        
        # Check the orchestrator file
        orchestrator_file = Path("src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/enhanced_optimized_orchestrator.py")
        
        if not orchestrator_file.exists():
            print("❌ Orchestrator file not found")
            return False
        
        with open(orchestrator_file, 'r') as f:
            content = f.read()
        
        # Check for fallback patterns
        found_fallbacks = []
        for pattern in fallback_patterns:
            if pattern in content.lower():
                found_fallbacks.append(pattern)
        
        if found_fallbacks:
            print(f"⚠️ Found fallback patterns in orchestrator: {found_fallbacks}")
            return False
        else:
            print("✅ No fallback patterns found in orchestrator")
        
        return True
        
    except Exception as e:
        print(f"❌ No fallbacks test failed: {e}")
        return False

def main():
    """Run all fast-fail tests."""
    print("🚀 Starting Fast-Fail Behavior Tests")
    print("=" * 60)
    
    tests = [
        ("ImportManager Fast-Fail", test_import_manager_fast_fail),
        ("Feature Generation Fast-Fail", test_feature_generation_fast_fail),
        ("Interaction Generation Fast-Fail", test_interaction_generation_fast_fail),
        ("Cross-Timeframe Fast-Fail", test_cross_timeframe_fast_fail),
        ("No Fallbacks in Code", test_no_fallbacks_in_code),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        print("-" * 40)
        
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
    print("\n" + "=" * 60)
    print("📊 FAST-FAIL TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fast-fail tests passed! The system now uses fast-fail patterns.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)