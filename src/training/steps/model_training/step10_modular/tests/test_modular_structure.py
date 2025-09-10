"""Test Step 10 Modular Structure.

This module tests the basic functionality of the modular Step 10 implementation.
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

# Add the modular step10 to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .. import UnifiedRegimeIntelligenceOrchestrator
from ..config import Step10Config
from ..models import MultiTimeframeHMMEncoder
from ..base import validate_step10_imports


def test_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing imports...")

    try:
        # Test core imports
        from ..models import MultiTimeframeHMMEncoder
        from ..config import Step10Config, DEFAULT_CONFIG
        from ..base import setup_step10_logger, validate_step10_imports

        print("✅ Core imports successful")
        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("🧪 Testing configuration...")

    try:
        # Test configuration creation
        config = Step10Config()
        print(f"✅ Configuration created: {config.symbol}/{config.exchange}")

        # Test configuration validation
        errors = config.validate()
        if errors:
            print(f"⚠️ Configuration validation warnings: {errors}")
        else:
            print("✅ Configuration validation passed")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("🧪 Testing model creation...")

    try:
        # Create configuration
        config = Step10Config()

        # Create model
        model = MultiTimeframeHMMEncoder(config.get_model_config())
        print(f"✅ Model created: {model.__class__.__name__}")

        # Test model configuration
        print(f"   Timeframes: {model.timeframes}")
        print(f"   Model dimension: {model.d_model}")

        return True

    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False


def test_orchestrator_creation():
    """Test orchestrator creation."""
    print("🧪 Testing orchestrator creation...")

    try:
        # Create orchestrator
        orchestrator = UnifiedRegimeIntelligenceOrchestrator()
        print("✅ Orchestrator created")

        # Test status
        status = orchestrator.get_status()
        print(f"   Initialized: {status['initialized']}")
        print(f"   Components ready: {sum(status['components'].values())}/5")

        return True

    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False


async def test_orchestrator_initialization():
    """Test orchestrator initialization."""
    print("🧪 Testing orchestrator initialization...")

    try:
        # Create and initialize orchestrator
        orchestrator = UnifiedRegimeIntelligenceOrchestrator()

        success = await orchestrator.initialize()
        if success:
            print("✅ Orchestrator initialization successful")
            return True
        else:
            print("❌ Orchestrator initialization failed")
            return False

    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        return False


def run_tests():
    """Run all tests."""
    print("🚀 Running Step 10 Modular Structure Tests")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Model Creation Test", test_model_creation),
        ("Orchestrator Creation Test", test_orchestrator_creation),
    ]

    async_tests = [
        ("Orchestrator Initialization Test", test_orchestrator_initialization),
    ]

    results = []

    # Run synchronous tests
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append(result)

    # Run asynchronous tests
    for test_name, test_func in async_tests:
        print(f"\n{test_name}:")
        result = asyncio.run(test_func())
        results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("🧪 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"   Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Modular structure is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
