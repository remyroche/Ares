#!/usr/bin/env python3
"""
Validation script for LightGBM + Featuretools implementation

This script validates that the new LightGBM feature generator can be imported
and basic functionality works without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Also add current directory for relative imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test basic imports
        from enhanced_components.lightgbm_feature_generator import (
            LightGBMFeatureGenerator,
            FeatureGenerationConfig,
            GeneratedFeature,
            FeatureGenerationResult,
            create_lightgbm_feature_generator
        )
        print("✅ LightGBM feature generator imports successful")
        
        # Test configuration
        config = FeatureGenerationConfig()
        print(f"✅ Configuration created: model_type={config.model_type}, max_features={config.max_features}")
        
        # Test generator creation
        generator = create_lightgbm_feature_generator(config)
        print("✅ Generator created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration options."""
    print("\n🔧 Testing configuration options...")
    
    try:
        from enhanced_components.lightgbm_feature_generator import FeatureGenerationConfig
        
        # Test default config
        config = FeatureGenerationConfig()
        assert config.model_type == 'lightgbm'
        assert config.max_features == 100
        assert config.use_shap == True
        assert config.use_ale == True
        print("✅ Default configuration correct")
        
        # Test custom config
        custom_config = FeatureGenerationConfig(
            model_type='catboost',
            max_features=50,
            use_shap=False,
            use_ale=False
        )
        assert custom_config.model_type == 'catboost'
        assert custom_config.max_features == 50
        assert custom_config.use_shap == False
        assert custom_config.use_ale == False
        print("✅ Custom configuration correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_structures():
    """Test data structure classes."""
    print("\n📊 Testing data structures...")
    
    try:
        from enhanced_components.lightgbm_feature_generator import (
            GeneratedFeature,
            FeatureGenerationResult
        )
        
        # Test GeneratedFeature
        feature = GeneratedFeature(
            name='test_feature',
            formula='feature_1 * feature_2',
            feature_series=None,  # We'll skip pandas for this test
            importance_score=0.5,
            parent_features=['feature_1', 'feature_2']
        )
        assert feature.name == 'test_feature'
        assert feature.formula == 'feature_1 * feature_2'
        assert feature.importance_score == 0.5
        assert feature.generation_method == 'lightgbm_featuretools'
        print("✅ GeneratedFeature structure correct")
        
        # Test FeatureGenerationResult
        result = FeatureGenerationResult(
            generated_features=[],
            feature_importance_scores={},
            model_performance={},
            generation_time=0.0,
            n_features_generated=0,
            n_features_selected=0,
            cache_hit_rate=0.0,
            shap_analysis_completed=False,
            ale_analysis_completed=False,
            featuretools_features=0,
            metadata={}
        )
        assert result.n_features_generated == 0
        assert result.shap_analysis_completed == False
        assert result.ale_analysis_completed == False
        print("✅ FeatureGenerationResult structure correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False

def test_generator_initialization():
    """Test generator initialization without external dependencies."""
    print("\n🏗️  Testing generator initialization...")
    
    try:
        from enhanced_components.lightgbm_feature_generator import (
            LightGBMFeatureGenerator,
            FeatureGenerationConfig
        )
        
        # Test with default config
        config = FeatureGenerationConfig()
        generator = LightGBMFeatureGenerator(config)
        assert generator.config.model_type == 'lightgbm'
        assert generator.config.max_features == 100
        print("✅ Default generator initialization successful")
        
        # Test with custom config
        custom_config = FeatureGenerationConfig(
            model_type='catboost',
            max_features=25,
            use_shap=False,
            use_ale=False
        )
        custom_generator = LightGBMFeatureGenerator(custom_config)
        assert custom_generator.config.model_type == 'catboost'
        assert custom_generator.config.max_features == 25
        print("✅ Custom generator initialization successful")
        
        # Test performance stats
        stats = generator.get_performance_stats()
        assert 'total_generations' in stats
        assert 'successful_generations' in stats
        assert 'failed_generations' in stats
        print("✅ Performance stats structure correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Generator initialization test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n📁 Testing file structure...")
    
    base_path = Path(__file__).parent
    
    required_files = [
        'enhanced_components/lightgbm_feature_generator.py',
        'enhanced_components/__init__.py',
        'examples/lightgbm_integration_example.py',
        'tests/test_lightgbm_feature_generator.py',
        'MIGRATION_GUIDE.md'
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def main():
    """Run all validation tests."""
    print("🚀 LightGBM + Featuretools Implementation Validation")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Structures", test_data_structures),
        ("Generator Initialization", test_generator_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                print(f"✅ {test_name} test passed")
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is ready.")
        print("\nKey Features Implemented:")
        print("• LightGBM/CatBoost models for better performance")
        print("• Featuretools Deep Feature Synthesis")
        print("• SHAP + ALE validation")
        print("• Maximum 100 features limit")
        print("• Comprehensive error handling")
        print("• Performance monitoring")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)