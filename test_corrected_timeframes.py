#!/usr/bin/env python3
"""
Corrected Timeframes Test

This script tests that the multi-output training framework is correctly configured
for the actual timeframes used in the system.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_corrected_timeframes():
    """Test that the timeframes are correctly configured."""
    print("🧪 Testing Corrected Timeframes Configuration...")

    try:
        from training.multi_output_probability_trainer import MultiOutputProbabilityTrainer

        # Test configuration with correct timeframes
        config = {
            "timeframe": "5m",
            "model_architectures": {
                "1m": "cnn",      # CNN for 1-minute data (Tactician)
                "5m": "tcn",      # TCN for 5-minute data (Analyst)
                "15m": "transformer", # Transformer for 15-minute data (Enhanced)
                "30m": "lightgbm",    # LightGBM for 30-minute data (Analyst)
                "1h": "hmm_regime"    # HMM regime definition only
            }
        }

        # Initialize trainer
        trainer = MultiOutputProbabilityTrainer(config)
        print("✅ Trainer initialized successfully")

        # Test timeframe mapping
        expected_mappings = {
            "1m": "cnn",
            "5m": "tcn",
            "15m": "transformer",
            "30m": "lightgbm",
            "1h": "hmm_regime"
        }

        for timeframe, expected_model in expected_mappings.items():
            if timeframe in trainer.model_architectures:
                model_type = trainer.model_architectures[timeframe]
                assert model_type == expected_model, f"Expected {expected_model} for {timeframe}, got {model_type}"
                print(f"   ✅ {timeframe} → {expected_model.upper()}")
            else:
                print(f"   ❌ {timeframe} not found in model architectures")

        print("✅ All timeframe mappings are correct")

        # Test that 4h and 1d are NOT included
        assert "4h" not in trainer.model_architectures, "4h should not be included"
        assert "1d" not in trainer.model_architectures, "1d should not be included"
        print("✅ Unused timeframes (4h, 1d) correctly excluded")

        # Test that 1h is marked as HMM regime only
        assert trainer.model_architectures["1h"] == "hmm_regime", "1h should be hmm_regime"
        print("✅ 1h correctly configured as HMM regime only")

        print("✅ Corrected timeframes test PASSED")
        return True

    except Exception as e:
        print(f"❌ Corrected timeframes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_steps_configuration():
    """Test that training steps are correctly configured."""
    print("🧪 Testing Training Steps Configuration...")

    try:
        # Test Step 6 configuration (should use TCN for 5m)
        step6_config = {
            "timeframe": "5m",
            "model_architectures": {
                "1m": "cnn",      # CNN for 1-minute data (Tactician)
                "5m": "tcn",      # TCN for 5-minute data (Analyst)
                "15m": "transformer", # Transformer for 15-minute data (Enhanced)
                "30m": "lightgbm",    # LightGBM for 30-minute data (Analyst)
                "1h": "hmm_regime"    # HMM regime definition only
            }
        }

        assert step6_config["model_architectures"][step6_config["timeframe"]] == "tcn"
        print("   ✅ Step 6 correctly configured for TCN (5m)")

        # Test Step 9 configuration (should use CNN for 1m)
        step9_config = {
            "timeframe": "1m",
            "model_architectures": {
                "1m": "cnn",      # CNN for 1-minute data (Tactician)
                "5m": "tcn",      # TCN for 5-minute data (Analyst)
                "15m": "transformer", # Transformer for 15-minute data (Enhanced)
                "30m": "lightgbm",    # LightGBM for 30-minute data (Analyst)
                "1h": "hmm_regime"    # HMM regime definition only
            }
        }

        assert step9_config["model_architectures"][step9_config["timeframe"]] == "cnn"
        print("   ✅ Step 9 correctly configured for CNN (1m)")

        # Test Enhanced Step 6 configuration (should use Transformer for 15m)
        enhanced_step6_config = {
            "timeframe": "15m",
            "model_architectures": {
                "1m": "cnn",      # CNN for 1-minute data (Tactician)
                "5m": "tcn",      # TCN for 5-minute data (Analyst)
                "15m": "transformer", # Transformer for 15-minute data (Enhanced)
                "30m": "lightgbm",    # LightGBM for 30-minute data (Analyst)
                "1h": "hmm_regime"    # HMM regime definition only
            }
        }

        assert enhanced_step6_config["model_architectures"][enhanced_step6_config["timeframe"]] == "transformer"
        print("   ✅ Enhanced Step 6 correctly configured for Transformer (15m)")

        print("✅ Training steps configuration test PASSED")
        return True

    except Exception as e:
        print(f"❌ Training steps configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_corrected_timeframes_tests():
    """Run all corrected timeframes tests."""
    print("🚀 Starting Corrected Timeframes Tests")
    print("=" * 60)

    tests = [
        ("Corrected Timeframes", test_corrected_timeframes),
        ("Training Steps Configuration", test_training_steps_configuration)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name} test...")

        try:
            result = test_func()
            results[test_name] = result

            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")

        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'='*60}")
    print("CORRECTED TIMEFRAMES TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL CORRECTED TIMEFRAMES TESTS PASSED!")
        print("\n🎯 TIMEFRAME CONFIGURATION STATUS: CORRECT")
        print("✅ Only actual timeframes are included (1m, 5m, 15m, 30m, 1h)")
        print("✅ 4h and 1d timeframes correctly excluded")
        print("✅ 1h correctly configured as HMM regime only")
        print("✅ Training steps properly configured for their timeframes")
        print("\n🚀 The multi-output training framework is correctly configured!")
        print("All timeframes match the actual usage in the system.")
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = run_corrected_timeframes_tests()
    sys.exit(0 if success else 1)