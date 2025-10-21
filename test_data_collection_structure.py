#!/usr/bin/env python3
"""
Test script to verify KlinesParquetManager integration structure in data collection steps.
This test focuses on structural integration without requiring pandas/numpy.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_klines_downloading_structure():
    """Test KlinesDataProcessingPipeline structure."""
    try:
        print("🔍 Testing KlinesDataProcessingPipeline structure...")
        
        # Test import without instantiating
        from src.training.steps.data_collection.klines_downloading_processing import KlinesDataProcessingPipeline
        
        # Check if class inherits from BaseStep
        from src.training.steps.base_step import BaseStep
        if issubclass(KlinesDataProcessingPipeline, BaseStep):
            print("✅ KlinesDataProcessingPipeline inherits from BaseStep")
        else:
            print("❌ KlinesDataProcessingPipeline does not inherit from BaseStep")
            return False
        
        # Check if the class has the expected methods
        expected_methods = [
            '_is_klines_available',
            '_store_klines',
            '_load_klines',
            '_store_klines_with_context',
            '_load_klines_with_context',
            '_get_klines_context'
        ]
        
        for method in expected_methods:
            if hasattr(KlinesDataProcessingPipeline, method):
                print(f"✅ Has method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ KlinesDataProcessingPipeline structure test failed: {e}")
        return False

def test_enhanced_klines_structure():
    """Test EnhancedKlinesProcessingPipeline structure."""
    try:
        print("🔍 Testing EnhancedKlinesProcessingPipeline structure...")
        
        # Test import without instantiating
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
            EnhancedKlinesProcessingPipeline, 
            PipelineConfig
        )
        
        # Check if class inherits from BaseStep
        from src.training.steps.base_step import BaseStep
        if issubclass(EnhancedKlinesProcessingPipeline, BaseStep):
            print("✅ EnhancedKlinesProcessingPipeline inherits from BaseStep")
        else:
            print("❌ EnhancedKlinesProcessingPipeline does not inherit from BaseStep")
            return False
        
        # Check if the class has the expected methods
        expected_methods = [
            '_is_klines_available',
            '_store_klines',
            '_load_klines',
            'execute'  # Required by BaseStep
        ]
        
        for method in expected_methods:
            if hasattr(EnhancedKlinesProcessingPipeline, method):
                print(f"✅ Has method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ EnhancedKlinesProcessingPipeline structure test failed: {e}")
        return False

def test_import_structure():
    """Test that imports work correctly."""
    try:
        print("🔍 Testing import structure...")
        
        # Test BaseStep import
        from src.training.steps.base_step import BaseStep
        print("✅ BaseStep import successful")
        
        # Test that BaseStep has klines methods
        klines_methods = [
            '_is_klines_available',
            '_store_klines',
            '_load_klines',
            '_store_klines_with_context',
            '_load_klines_with_context',
            '_get_klines_context'
        ]
        
        for method in klines_methods:
            if hasattr(BaseStep, method):
                print(f"✅ BaseStep has method: {method}")
            else:
                print(f"❌ BaseStep missing method: {method}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        return False

def test_file_modifications():
    """Test that files have been properly modified."""
    try:
        print("🔍 Testing file modifications...")
        
        # Check klines_downloading_processing.py
        with open("src/training/steps/data_collection/klines_downloading_processing.py", "r") as f:
            content = f.read()
            
        if "_store_klines_with_context" in content:
            print("✅ klines_downloading_processing.py has context-aware methods")
        else:
            print("❌ klines_downloading_processing.py missing context-aware methods")
            return False
        
        if "_is_klines_available" in content:
            print("✅ klines_downloading_processing.py has availability check")
        else:
            print("❌ klines_downloading_processing.py missing availability check")
            return False
        
        # Check enhanced_klines_processing_pipeline.py
        with open("src/training/steps/data_collection/enhanced_klines_processing_pipeline.py", "r") as f:
            content = f.read()
            
        if "class EnhancedKlinesProcessingPipeline(BaseStep):" in content:
            print("✅ enhanced_klines_processing_pipeline.py inherits from BaseStep")
        else:
            print("❌ enhanced_klines_processing_pipeline.py does not inherit from BaseStep")
            return False
        
        if "async def execute" in content:
            print("✅ enhanced_klines_processing_pipeline.py has execute method")
        else:
            print("❌ enhanced_klines_processing_pipeline.py missing execute method")
            return False
        
        if "_is_klines_available" in content:
            print("✅ enhanced_klines_processing_pipeline.py has availability check")
        else:
            print("❌ enhanced_klines_processing_pipeline.py missing availability check")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ File modifications test failed: {e}")
        return False

def main():
    """Run all structure tests."""
    print("🚀 Testing KlinesParquetManager integration structure in data collection steps")
    print("=" * 80)
    
    tests = [
        ("Import Structure", test_import_structure),
        ("KlinesDataProcessingPipeline Structure", test_klines_downloading_structure),
        ("EnhancedKlinesProcessingPipeline Structure", test_enhanced_klines_structure),
        ("File Modifications", test_file_modifications)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
        print()
    
    print("=" * 80)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("🎉 ALL STRUCTURE TESTS PASSED!")
        print("✅ KlinesParquetManager integration structure is correct in data collection steps")
        print("⚠️ Note: Full functionality requires pandas/pyarrow dependencies")
    else:
        print("⚠️ Some structure tests failed - check the output above for details")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)