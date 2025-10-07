#!/usr/bin/env python3
"""
Test script for Roadmap Integration

This script tests the integration of the roadmap feature generation component
into the training pipeline.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_component_factory():
    """Test component factory registration."""
    print("=== Testing Component Factory ===")
    
    try:
        from src.training.steps.pre_training.components.component_factory import ComponentFactory
        
        # Check if roadmap component is registered
        available_components = ComponentFactory.get_available_components()
        print(f"Available components: {available_components}")
        
        if 'roadmap_feature_generation' in available_components:
            print("✅ Roadmap component is registered")
            
            # Test component creation
            try:
                from src.training.steps.pre_training.components.base_component import ComponentConfig
                config = ComponentConfig(symbol="ETHUSDT", exchange="binance", timeframe="15m")
                component = ComponentFactory.create_component('roadmap_feature_generation', config)
                print("✅ Roadmap component created successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to create roadmap component: {e}")
                return False
        else:
            print("❌ Roadmap component not found in registry")
            return False
            
    except Exception as e:
        print(f"❌ Component factory test failed: {e}")
        return False

def test_sub_pipeline():
    """Test sub-pipeline integration."""
    print("\n=== Testing Sub-Pipeline Integration ===")
    
    try:
        from src.training.steps.pre_training.sub_pipeline import PreTrainingSubPipeline
        
        pipeline = PreTrainingSubPipeline()
        available_pipelines = pipeline.get_available_sub_pipelines()
        print(f"Available sub-pipelines: {available_pipelines}")
        
        if 'roadmap_feature_generation' in available_pipelines:
            print("✅ Roadmap feature generation is available in sub-pipeline")
            return True
        else:
            print("❌ Roadmap feature generation not found in sub-pipeline")
            return False
            
    except Exception as e:
        print(f"❌ Sub-pipeline test failed: {e}")
        return False

def test_imports():
    """Test all necessary imports."""
    print("\n=== Testing Imports ===")
    
    try:
        # Test roadmap component import
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.roadmap_feature_generation_component import RoadmapFeatureGenerationComponent
        print("✅ RoadmapFeatureGenerationComponent imported")
        
        # Test end-to-end roadmap import
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.end_to_end_roadmap import EndToEndRoadmapSystem
        print("✅ EndToEndRoadmapSystem imported")
        
        # Test feature engineering imports
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.feature_engineering.feature_registry import FeatureRegistry
        print("✅ FeatureRegistry imported")
        
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.feature_engineering.transforms import TransformRouter
        print("✅ TransformRouter imported")
        
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.feature_engineering.interactions import InteractionEngine
        print("✅ InteractionEngine imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n=== Testing Configuration ===")
    
    try:
        config_path = "src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/end_to_end_roadmap_config.yaml"
        
        if os.path.exists(config_path):
            print("✅ Configuration file found")
            
            # Try to read the config
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check for required sections
            required_sections = ['system', 'menus', 'interactions_locked', 'model']
            for section in required_sections:
                if section in config:
                    print(f"✅ Found section: {section}")
                else:
                    print(f"❌ Missing section: {section}")
                    return False
            
            return True
        else:
            print(f"❌ Configuration file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Roadmap Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Component Factory", test_component_factory),
        ("Sub-Pipeline", test_sub_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The roadmap integration is working correctly.")
        print("\nNext steps:")
        print("1. Run the full training pipeline with roadmap feature generation")
        print("2. Monitor the pipeline execution and feature generation")
        print("3. Verify the generated features meet the roadmap specifications")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())