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


def test_final_parameters_optimizer_extended_schema():
    """Validate FinalParametersOptimizer directional schema and JSON output."""
    print("\n=== Testing FinalParametersOptimizer Extended Schema ===")

    try:
        from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer

        config = {'n_trials': 1, 'timeout': 1}
        optimizer = FinalParametersOptimizer(config)

        required_categories = {
            'long_specific_parameters', 'short_specific_parameters',
            'directional_thresholds', 'asymmetric_risk_management',
            'tactician_analyst_integration', 'analyst_oof_weights',
            'merged_feature_importance'
        }

        missing_categories = sorted(required_categories - set(optimizer.categories))
        if missing_categories:
            print(f"❌ Missing optimizer categories: {missing_categories}")
            return False

        exit_space = optimizer.default_search_spaces.get('exit_strategy', {})
        trailing_keys = [
            'trailing_atr_multiplier', 'trailing_min_distance', 'trailing_confidence_activation'
        ]
        if not all(key in exit_space for key in trailing_keys):
            print("❌ Exit strategy search space missing trailing stop keys")
            return False

        sample_results = {
            'exit_strategy': {
                'confidence_very_low': 0.18,
                'confidence_low': 0.35,
                'confidence_medium': 0.58,
                'confidence_high': 0.82,
                'base_profit_target': 0.05,
                'min_confidence_for_profit': 0.62,
                'confidence_profit_multiplier': 0.4,
                'profit_tier_1': 0.25,
                'profit_tier_2': 0.5,
                'profit_tier_3': 0.75,
                'base_stop_loss': -0.04,
                'atr_multiplier': 1.9,
                'volatility_adjustment_factor': 1.2,
                'max_hold_time': 5400,
                'min_hold_time': 180,
                'confidence_time_scaling_factor': 1.1,
                'trailing_atr_multiplier': 2.0,
                'trailing_min_distance': 0.012,
                'trailing_confidence_activation': 0.7,
                'regime_transition_penalty': 0.15,
                'regime_specific_scaling': 1.05,
            },
            'long_specific_parameters': {'long_entry_patience': 1.1},
            'short_specific_parameters': {'short_entry_urgency': 1.2},
            'directional_thresholds': {'long_vs_short_bias_threshold': 0.2},
            'asymmetric_risk_management': {'long_max_position_duration': 30},
            'tactician_analyst_integration': {'integration_method': 'ensemble'},
            'analyst_oof_weights': {'adaptive_weighting': 'dynamic'},
            'merged_feature_importance': {'feature_selection_threshold': 0.05},
        }

        import asyncio
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                asyncio.run(
                    optimizer.save_optimization_results(
                        sample_results,
                        'ETHUSDT',
                        'BINANCE',
                        'generated'
                    )
                )

                json_path = Path('generated/backtesting/optimization_results/BINANCE_ETHUSDT_final_parameters.json')
                if not json_path.exists():
                    print("❌ Optimizer did not emit JSON output")
                    return False

                with open(json_path, 'r') as handle:
                    payload = json.load(handle)

                for category in required_categories:
                    if category not in payload:
                        print(f"❌ JSON output missing category: {category}")
                        return False

                exit_payload = payload.get('exit_strategy', {})
                if not all(key in exit_payload for key in trailing_keys):
                    print("❌ JSON output missing trailing stop keys")
                    return False

            finally:
                os.chdir(cwd)

        print("✅ FinalParametersOptimizer extended schema validated")
        return True

    except Exception as exc:  # pragma: no cover - integration diagnostic
        print(f"❌ FinalParametersOptimizer test failed: {exc}")
        return False

def main():
    """Run all tests."""
    print("Roadmap Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Component Factory", test_component_factory),
        ("Sub-Pipeline", test_sub_pipeline),
        ("Final Parameters Optimizer", test_final_parameters_optimizer_extended_schema)
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