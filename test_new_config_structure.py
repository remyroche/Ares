#!/usr/bin/env python3

"""
Test script for the new configuration structure.
This script validates that the new categorized configuration system works correctly.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

    get_config_manager,
    get_static_config_global,
    get_optimizable_config,
    get_all_optimizable_configs,
    get_search_space,
    get_all_search_spaces,
    get_complete_config,
    get_parameter_value,
    get_optimizable_parameters,
    update_optimizable_config,
    validate_config,
)


def test_config_loading():
    pass
    pass
    """Test that configurations can be loaded correctly."""
    print("🔧 Testing configuration loading...")

    try:
        # Test config manager initialization
    except Exception as e:
        pass
    except Exception as e:
        pass
        config_manager = get_config_manager()
        print("✅ Config manager initialized successfully")

        # Test static config loading
        static_config = get_static_config_global()
        print(f"✅ Static config loaded with {len(static_config)} sections")

        # Test optimizable configs loading
        optimizable_configs = get_all_optimizable_configs()
        expected_categories = [
            "confidence",
            "position_sizing",
            "leverage",
            "tpsl",
            "ensemble",
            "sr",
            "two_tier",
            "technical_indicators",
            "system_monitoring",
            "training_optimization",
            "regime_transitions"
        ]

        for category in expected_categories:
    pass
    pass
            if category in optimizable_configs:
    pass
    pass
                print(f"✅ {category} config loaded successfully")
            else:
                print(f"❌ {category} config missing")
                return False

        print("✅ All optimizable configurations loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Error loading configurations: {e}")
        return False


def test_parameter_access():
    pass
    pass
    """Test parameter access using dot notation."""
    print("\\\n🔍 Testing parameter access...")

    try:
        # Test static config parameter access
    except Exception as e:
        pass
    except Exception as e:
        pass
        db_host = get_parameter_value("database.host")
        if db_host == "localhost":
    pass
    pass
            print("✅ Database host parameter accessed correctly")
        else:
            print(f"❌ Database host parameter incorrect: {db_host}")
            return False

        # Test optimizable config parameter access
        base_entry_threshold = get_parameter_value("confidence.base_entry_threshold")
        if base_entry_threshold == 0.7:
    pass
    pass
            print("✅ Confidence base entry threshold accessed correctly")
        else:
            print(f"❌ Confidence base entry threshold incorrect: {base_entry_threshold}")
            return False

        # Test two-tier parameters
        two_tier_config = get_optimizable_config("two_tier")
        direction_threshold = two_tier_config.direction_threshold
        if direction_threshold == 0.6:  # Updated value after cleanup
            print("✅ Two-tier direction threshold accessed correctly")
        else:
            print(f"❌ Two-tier direction threshold incorrect: {direction_threshold}")
            return False

        print("✅ All parameter access tests passed")
        return True

    except Exception as e:
        print(f"❌ Error accessing parameters: {e}")
        return False


def test_search_spaces():
    pass
    pass
    """Test that search spaces are properly defined."""
    print("\\\n🎯 Testing search spaces...")

    try:
        # Test search spaces
    except Exception as e:
        pass
    except Exception as e:
        pass
        search_spaces = get_all_search_spaces()
        expected_categories = [
            "confidence",
            "position_sizing",
            "leverage",
            "tpsl",
            "ensemble",
            "sr",
            "two_tier",
            "technical_indicators",
            "system_monitoring"
        ]

        for category in expected_categories:
    pass
    pass
            if category in search_spaces:
    pass
    pass
                category_space = search_spaces[category]
                if category_space:
    pass
    pass
                    print(f"✅ {category} search space has {len(category_space)} parameters")
                else:
                    print(f"❌ {category} search space is empty")
                    return False
            else:
                print(f"❌ {category} search space missing")
                return False

        # Test specific search space parameters
        confidence_space = get_search_space("confidence")
        if "base_entry_threshold" in confidence_space:
    pass
    pass
            print("✅ Confidence search space contains expected parameters")
        else:
            print("❌ Confidence search space missing expected parameters")
            return False

        two_tier_space = get_search_space("two_tier")
        if "direction_threshold" in two_tier_space:
    pass
    pass
            print("✅ Two-tier search space contains expected parameters")
        else:
            print("❌ Two-tier search space missing expected parameters")
            return False

        technical_indicators_space = get_search_space("technical_indicators")
        if "rsi_period" in technical_indicators_space:
    pass
    pass
            print("✅ Technical indicators search space contains expected parameters")
        else:
            print("❌ Technical indicators search space missing expected parameters")
            return False

        system_monitoring_space = get_search_space("system_monitoring")
        if "analysis_interval" in system_monitoring_space:
    pass
    pass
            print("✅ System monitoring search space contains expected parameters")
        else:
            print("❌ System monitoring search space missing expected parameters")
            return False

        training_optimization_space = get_search_space("training_optimization")
        if "min_quality_score" in training_optimization_space:
    pass
    pass
            print("✅ Training optimization search space contains expected parameters")
        else:
            print("❌ Training optimization search space missing expected parameters")
            return False

        regime_transitions_space = get_search_space("regime_transitions")
        if "transition_intensity_threshold" in regime_transitions_space:
    pass
    pass
            print("✅ Regime transitions search space contains expected parameters")
        else:
            print("❌ Regime transitions search space missing expected parameters")
            return False

        print("✅ All search space tests passed")
        return True

    except Exception as e:
        print(f"❌ Error testing search spaces: {e}")
        return False


def test_config_updates():
    pass
    pass
    """Test that configurations can be updated."""
    print("\\\n🔄 Testing configuration updates...")

    try:
        # Test updating confidence config
    except Exception as e:
        pass
    except Exception as e:
        pass
        updates = {"base_entry_threshold": 0.75}
        success = update_optimizable_config("confidence", updates)
        if success:
    pass
    pass
            print("✅ Confidence config updated successfully")
        else:
            print("❌ Failed to update confidence config")
            return False

        # Verify the update
        new_threshold = get_parameter_value("confidence.base_entry_threshold")
        if new_threshold == 0.75:
    pass
    pass
            print("✅ Confidence config update verified")
        else:
            print(f"❌ Confidence config update not reflected: {new_threshold}")
            return False

        # Test updating two-tier config
        two_tier_updates = {"direction_threshold": 0.75}
        success = update_optimizable_config("two_tier", two_tier_updates)
        if success:
    pass
    pass
            print("✅ Two-tier config updated successfully")
        else:
            print("❌ Failed to update two-tier config")
            return False

        # Verify the update
        new_direction_threshold = get_parameter_value("two_tier.direction_threshold")
        if new_direction_threshold == 0.75:
    pass
    pass
            print("✅ Two-tier config update verified")
        else:
            print(f"❌ Two-tier config update not reflected: {new_direction_threshold}")
            return False

        print("✅ All configuration update tests passed")
        return True

    except Exception as e:
        print(f"❌ Error testing configuration updates: {e}")
        return False


def test_config_validation():
    pass
    pass
    """Test configuration validation."""
    print("\\\n✅ Testing configuration validation...")

    try:
        is_valid, errors = validate_config()
    except Exception as e:
        pass
    except Exception as e:
        pass
        if is_valid:
    pass
    pass
            print("✅ Configuration validation passed")
        else:
            print(f"❌ Configuration validation failed: {errors}")
            return False

        return True

    except Exception as e:
        print(f"❌ Error during configuration validation: {e}")
        return False


def test_complete_config():
    pass
    pass
    """Test complete configuration retrieval."""
    print("\\\n📋 Testing complete configuration...")

    try:
        # Test complete configuration
    except Exception as e:
        pass
    except Exception as e:
        pass
        complete_config = get_complete_config()

        # Test static sections
        static_sections = ["database", "exchange", "system", "environment", "trading", "training"]
        for section in static_sections:
    pass
    pass
            if section in complete_config:
    pass
    pass
                print(f"✅ Static section '{section}' found in complete config")
            else:
                print(f"❌ Static section '{section}' missing from complete config")
                return False

        # Test optimizable sections
        optimizable_sections = [
            "confidence",
            "position_sizing",
            "leverage",
            "tpsl",
            "ensemble",
            "sr",
            "two_tier",
            "technical_indicators",
            "system_monitoring",
            "training_optimization",
            "regime_transitions"
        ]
        for section in optimizable_sections:
    pass
    pass
            if section in complete_config:
    pass
    pass
                print(f"✅ Optimizable section '{section}' found in complete config")
            else:
                print(f"❌ Optimizable section '{section}' missing from complete config")
                return False

        print("✅ Complete configuration test passed")
        return True

    except Exception as e:
        print(f"❌ Error testing complete configuration: {e}")
        return False


def test_step12_integration():
    pass
    pass
    """Test basic step12 integration."""
    print("\\\n🚀 Testing step12 integration...")

    try:
        # Try to import required dependencies
    except Exception as e:
        pass
    except Exception as e:
        pass
        try:
            dependencies_available = True
    except Exception as e:
        pass
    except Exception as e:
        pass
        except ImportError as e:
            print(f"⚠️ Some dependencies not available: {e}")
            print("   This is expected in a minimal test environment")
            dependencies_available = False

        if not dependencies_available:
    pass
    pass
            print("✅ Step12 integration test skipped (dependencies not available)")
            return True

        # Import step12 class
        from training.steps.step17_final_parameters_optimization_new import FinalParametersOptimizationStepNew

        # Create step12 instance
import config = {"test": True}
        config = {"test": True}
        step12 = FinalParametersOptimizationStepNew(config)

        print("✅ Step12 class instantiated successfully")

        # Test that it can access the config manager
        config_manager = step12.config_manager
        if config_manager:
    pass
    pass
            print("✅ Step12 can access config manager")
        else:
            print("❌ Step12 cannot access config manager")
            return False

        # Test that it has access to optimizable parameters
        optimizable_params = step12.optimizable_params
        if optimizable_params and len(optimizable_params) > 0:
    pass
    pass
            print(f"✅ Step12 has access to {len(optimizable_params)} optimizable parameter categories")
        else:
            print("❌ Step12 has no access to optimizable parameters")
            return False

        print("✅ Step12 integration test passed")
        return True

    except Exception as e:
        print(f"❌ Error testing step12 integration: {e}")
        return False


def main():
    pass
    pass
    """Run all tests."""
    print("🧪 Testing New Configuration Structure")
    print("=" * 50)

    tests = [
        test_config_loading,
        test_parameter_access,
        test_search_spaces,
        test_config_updates,
        test_config_validation,
        test_complete_config,
        test_step12_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
    pass
    pass
        try:
            if test():
    pass
    except Exception as e:
        pass
    pass
                passed += 1
    except Exception as e:
        pass
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("\\\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
    pass
    pass
        print("🎉 All tests passed! The new configuration structure is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration structure.")
        return 1


if __name__ == "__main__":
    pass
    pass
    sys.exit(main())