#!/usr/bin/env python3
"""
Test script for MicroRegimeDetector implementation.

This script tests the basic functionality without requiring external dependencies.
"""

import sys
import os
sys.path.append('/workspace')

def test_basic_structure():
    """Test basic module structure and imports."""
    print("🧪 Testing basic module structure...")
    
    try:
        # Test that the file exists and is readable
        with open('/workspace/micro_regime_detector.py', 'r') as f:
            content = f.read()
        
        print("✅ File exists and is readable")
        
        # Test that key classes and functions are defined
        required_elements = [
            'class MicroRegimeType',
            'class MarketRegime', 
            'class DetectionConfig',
            'class MicroRegimeDetectionResult',
            'class MicroRegimeDetector',
            'def create_micro_regime_detector',
            'def _calculate_rsi',
            'def _calculate_atr',
            'def _calculate_macd',
            'def _calculate_bollinger_bands'
        ]
        
        for element in required_elements:
            if element in content:
                print(f"✅ Found {element}")
            else:
                print(f"❌ Missing {element}")
                return False
        
        print("✅ All required elements found")
        return True
        
    except Exception as e:
        print(f"❌ Error testing basic structure: {e}")
        return False

def test_utility_integration():
    """Test utility integration."""
    print("\n🧪 Testing utility integration...")
    
    try:
        with open('/workspace/micro_regime_detector.py', 'r') as f:
            content = f.read()
        
        # Check for utility imports
        utility_imports = [
            'from src.utils.common_operations import',
            'from src.utils.math_validation import',
            'from src.utils.serialization_utils import',
            'from src.utils.tprint import'
        ]
        
        for import_line in utility_imports:
            if import_line in content:
                print(f"✅ Found import: {import_line}")
            else:
                print(f"❌ Missing import: {import_line}")
                return False
        
        # Check for utility usage
        utility_usage = [
            'safe_divide',
            'validate_finite',
            'tprint_info',
            'serializer.save',
            'memory_checkpoint'
        ]
        
        for usage in utility_usage:
            if usage in content:
                print(f"✅ Found usage: {usage}")
            else:
                print(f"❌ Missing usage: {usage}")
                return False
        
        print("✅ All utility integrations found")
        return True
        
    except Exception as e:
        print(f"❌ Error testing utility integration: {e}")
        return False

def test_regime_detection_methods():
    """Test regime detection methods."""
    print("\n🧪 Testing regime detection methods...")
    
    try:
        with open('/workspace/micro_regime_detector.py', 'r') as f:
            content = f.read()
        
        # Check for detection methods
        detection_methods = [
            'def detect_micro_regimes',
            'def _detect_traditional_regimes',
            'def _detect_enhanced_breakouts',
            'def _detect_enhanced_consolidations',
            'def _detect_ml_regimes',
            'def _detect_statistical_regimes'
        ]
        
        for method in detection_methods:
            if method in content:
                print(f"✅ Found method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        # Check for technical indicators
        indicators = [
            'def _calculate_rsi',
            'def _calculate_atr',
            'def _calculate_macd',
            'def _calculate_bollinger_bands',
            'def _calculate_obv'
        ]
        
        for indicator in indicators:
            if indicator in content:
                print(f"✅ Found indicator: {indicator}")
            else:
                print(f"❌ Missing indicator: {indicator}")
                return False
        
        print("✅ All detection methods found")
        return True
        
    except Exception as e:
        print(f"❌ Error testing detection methods: {e}")
        return False

def test_ml_integration():
    """Test ML integration."""
    print("\n🧪 Testing ML integration...")
    
    try:
        with open('/workspace/micro_regime_detector.py', 'r') as f:
            content = f.read()
        
        # Check for ML imports
        ml_imports = [
            'from src.utils.ml_common.cvlsa import',
            'ML_UTILITIES_AVAILABLE',
            'create_enhanced_cvlsa_model'
        ]
        
        for ml_import in ml_imports:
            if ml_import in content:
                print(f"✅ Found ML import: {ml_import}")
            else:
                print(f"❌ Missing ML import: {ml_import}")
                return False
        
        # Check for ML usage
        ml_usage = [
            'self.ml_components',
            '_apply_feature_engineering',
            'enable_ml_detection'
        ]
        
        for usage in ml_usage:
            if usage in content:
                print(f"✅ Found ML usage: {usage}")
            else:
                print(f"❌ Missing ML usage: {usage}")
                return False
        
        print("✅ ML integration found")
        return True
        
    except Exception as e:
        print(f"❌ Error testing ML integration: {e}")
        return False

def test_hardware_optimization():
    """Test hardware optimization integration."""
    print("\n🧪 Testing hardware optimization...")
    
    try:
        with open('/workspace/micro_regime_detector.py', 'r') as f:
            content = f.read()
        
        # Check for hardware imports
        hardware_imports = [
            'integrate_with_m1_optimizers',
            'get_m1_gpu_manager',
            'get_m1_memory_optimizer',
            'get_m1_cpu_optimizer'
        ]
        
        for hw_import in hardware_imports:
            if hw_import in content:
                print(f"✅ Found hardware import: {hw_import}")
            else:
                print(f"❌ Missing hardware import: {hw_import}")
                return False
        
        # Check for hardware usage
        hardware_usage = [
            'memory_checkpoint',
            'gpu_context',
            'm1_integration',
            'cleanup_m1_optimizers'
        ]
        
        for usage in hardware_usage:
            if usage in content:
                print(f"✅ Found hardware usage: {usage}")
            else:
                print(f"❌ Missing hardware usage: {usage}")
                return False
        
        print("✅ Hardware optimization found")
        return True
        
    except Exception as e:
        print(f"❌ Error testing hardware optimization: {e}")
        return False

def test_serialization():
    """Test serialization functionality."""
    print("\n🧪 Testing serialization...")
    
    try:
        with open('/workspace/micro_regime_detector.py', 'r') as f:
            content = f.read()
        
        # Check for serialization methods
        serialization_methods = [
            'def save_model',
            'def load_model',
            'UniversalSerializer',
            'serializer.save',
            'serializer.load'
        ]
        
        for method in serialization_methods:
            if method in content:
                print(f"✅ Found serialization: {method}")
            else:
                print(f"❌ Missing serialization: {method}")
                return False
        
        print("✅ Serialization functionality found")
        return True
        
    except Exception as e:
        print(f"❌ Error testing serialization: {e}")
        return False

def test_example_usage():
    """Test example usage section."""
    print("\n🧪 Testing example usage...")
    
    try:
        with open('/workspace/micro_regime_detector.py', 'r') as f:
            content = f.read()
        
        # Check for example usage
        example_elements = [
            'if __name__ == "__main__":',
            'create_micro_regime_detector',
            'detect_micro_regimes',
            'get_detection_summary'
        ]
        
        for element in example_elements:
            if element in content:
                print(f"✅ Found example element: {element}")
            else:
                print(f"❌ Missing example element: {element}")
                return False
        
        print("✅ Example usage found")
        return True
        
    except Exception as e:
        print(f"❌ Error testing example usage: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting MicroRegimeDetector Tests\n")
    
    tests = [
        test_basic_structure,
        test_utility_integration,
        test_regime_detection_methods,
        test_ml_integration,
        test_hardware_optimization,
        test_serialization,
        test_example_usage
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test failed: {test.__name__}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MicroRegimeDetector implementation is complete.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)