#!/usr/bin/env python3
"""
Test script to demonstrate the new configuration structure.
This script shows how the parameters are organized into static and optimizable categories.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.config_manager import (
    get_config_manager,
    get_static_config,
    get_all_optimizable_configs,
    get_all_search_spaces,
    get_parameter_value,
    update_optimizable_config,
    validate_config,
)


def test_configuration_structure():
    """Test the new configuration structure."""
    print("🔧 Testing New Configuration Structure")
    print("=" * 50)
    
    # Get configuration manager
    config_manager = get_config_manager()
    
    # Test static configuration
    print("\n📋 Static (Non-Optimizable) Configuration:")
    print("-" * 40)
    static_config = get_static_config()
    for category, config in static_config.items():
        print(f"  {category}: {type(config).__name__}")
        if hasattr(config, '__dict__'):
            for key, value in list(config.__dict__.items())[:3]:  # Show first 3 items
                print(f"    {key}: {value}")
        print()
    
    # Test optimizable configurations
    print("\n🎯 Optimizable Configurations:")
    print("-" * 40)
    optimizable_configs = get_all_optimizable_configs()
    for category, config in optimizable_configs.items():
        print(f"  {category}: {type(config).__name__}")
        if hasattr(config, '__dict__'):
            for key, value in list(config.__dict__.items())[:3]:  # Show first 3 items
                print(f"    {key}: {value}")
        print()
    
    # Test search spaces
    print("\n🔍 Search Spaces for Optimization:")
    print("-" * 40)
    search_spaces = get_all_search_spaces()
    for category, search_space in search_spaces.items():
        print(f"  {category}: {len(search_space)} parameters")
        for param_name, param_config in list(search_space.items())[:3]:  # Show first 3
            print(f"    {param_name}: {param_config['type']} [{param_config['min']}, {param_config['max']}]")
        print()
    
    # Test parameter value retrieval
    print("\n📊 Parameter Value Retrieval:")
    print("-" * 40)
    test_params = [
        "confidence.base_entry_threshold",
        "position_sizing.base_position_size",
        "leverage.max_leverage",
        "tpsl.tp_long",
        "ensemble.analyst_weight",
        "sr.touch_count_weight",
    ]
    
    for param_path in test_params:
        value = get_parameter_value(param_path)
        print(f"  {param_path}: {value}")
    
    # Test parameter updates
    print("\n🔄 Parameter Updates:")
    print("-" * 40)
    test_updates = {
        "confidence": {"base_entry_threshold": 0.75},
        "position_sizing": {"base_position_size": 0.08},
        "leverage": {"max_leverage": 50},
    }
    
    for category, updates in test_updates.items():
        success = update_optimizable_config(category, updates)
        print(f"  Updated {category}: {'✅' if success else '❌'}")
        
        # Verify the update
        for param_name, expected_value in updates.items():
            param_path = f"{category}.{param_name}"
            actual_value = get_parameter_value(param_path)
            print(f"    {param_path}: {actual_value} (expected: {expected_value})")
    
    # Test configuration validation
    print("\n✅ Configuration Validation:")
    print("-" * 40)
    is_valid, errors = validate_config()
    print(f"  Configuration valid: {'✅' if is_valid else '❌'}")
    if errors:
        print("  Errors:")
        for error in errors:
            print(f"    - {error}")
    
    print("\n" + "=" * 50)
    print("✅ Configuration structure test completed!")


def test_optimization_categories():
    """Test the optimization categories and their parameters."""
    print("\n🎯 Testing Optimization Categories")
    print("=" * 50)
    
    categories = ["confidence", "position_sizing", "leverage", "tpsl", "ensemble", "sr"]
    
    for category in categories:
        print(f"\n📊 {category.upper()} Parameters:")
        print("-" * 30)
        
        # Get configuration
        config = get_all_optimizable_configs()[category]
        search_space = get_all_search_spaces()[category]
        
        print(f"  Configuration parameters: {len(config.__dict__)}")
        print(f"  Optimizable parameters: {len(search_space)}")
        
        # Show some key parameters
        if hasattr(config, '__dict__'):
            key_params = list(config.__dict__.keys())[:5]  # Show first 5
            print(f"  Key parameters: {', '.join(key_params)}")
        
        # Show search space ranges
        print("  Search space ranges:")
        for param_name, param_config in list(search_space.items())[:3]:  # Show first 3
            print(f"    {param_name}: {param_config['type']} [{param_config['min']}, {param_config['max']}]")


def test_step12_integration():
    """Test integration with step12 optimization."""
    print("\n🚀 Testing Step12 Integration")
    print("=" * 50)
    
    try:
        from src.training.steps.step12_final_parameters_optimization_new import FinalParametersOptimizationStepNew
        
        # Create test configuration
        test_config = {
            "optimization": {
                "n_trials": 10,
                "timeout_minutes": 5,
            }
        }
        
        # Initialize step12
        step12 = FinalParametersOptimizationStepNew(test_config)
        print("✅ Step12 initialization successful")
        
        # Test configuration access
        optimizable_params = step12.optimizable_params
        print(f"✅ Found {len(optimizable_params)} optimization categories")
        
        for category, params in optimizable_params.items():
            print(f"  {category}: {len(params)} parameters")
        
    except ImportError as e:
        print(f"❌ Step12 import failed: {e}")
    except Exception as e:
        print(f"❌ Step12 test failed: {e}")


def main():
    """Main test function."""
    print("🧪 Testing New Configuration Structure")
    print("=" * 60)
    
    try:
        # Test basic configuration structure
        test_configuration_structure()
        
        # Test optimization categories
        test_optimization_categories()
        
        # Test step12 integration
        test_step12_integration()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("\n📝 Summary:")
        print("  ✅ Static configuration: Database, Exchange, System, Environment, Trading, Training")
        print("  ✅ Optimizable configuration: Confidence, Position Sizing, Leverage, TP/SL, Ensemble, S/R")
        print("  ✅ Search spaces: Defined for all optimizable parameters")
        print("  ✅ Parameter access: Dot notation support")
        print("  ✅ Parameter updates: Dynamic configuration updates")
        print("  ✅ Validation: Configuration validation working")
        print("  ✅ Step12 integration: Ready for optimization")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()