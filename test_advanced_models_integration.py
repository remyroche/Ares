#!/usr/bin/env python3
"""
Advanced Models Integration Test

This script tests the complete integration of all advanced model types
with the multi-output training framework.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import shutil

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_all_model_types():
    """Test all advanced model types with multi-output training."""
    print("🧪 Testing All Advanced Model Types Integration...")

    try:
        from training.multi_output_probability_trainer import MultiOutputProbabilityTrainer
        from training.model_saving_utils import save_multi_output_model_with_probabilities, load_model_with_probabilities

        # Test all model types
        model_types = [
            ("lightgbm", "30m"),
            ("randomforest", "1d"),
            ("tcn", "5m"),
            ("cnn", "1m"),
            ("transformer", "15m"),
            ("lstm", "1h"),
            ("gru", "4h")
        ]

        results = {}

        for model_type, timeframe in model_types:
            print(f"\n🔧 Testing {model_type.upper()} for {timeframe} timeframe...")

            try:
                # Create test data
                n_samples = 200
                n_features = 15

                X = np.random.randn(n_samples, n_features)
                y = np.random.choice([0, 1], size=n_samples)
                market_data = pd.DataFrame({
                    'close': np.random.randn(n_samples),
                    'volume': np.random.randn(n_samples)
                })

                # Configure multi-output training for specific model type
                config = {
                    "use_lightgbm": True,
                    "n_estimators": 50,  # Reduced for faster testing
                    "learning_rate": 0.1,
                    "max_depth": 4,
                    "profit_target": 0.02,
                    "stop_loss": 0.01,
                    "look_ahead_periods": 20,
                    "magnitude_threshold_factor": 0.8,
                    "adverse_threshold": 0.01,
                    "avoidance_look_ahead": 10,
                    # Advanced model configuration
                    "timeframe": timeframe,
                    "model_architectures": {
                        "1m": "cnn",
                        "5m": "tcn",
                        "15m": "transformer",
                        "30m": "lightgbm",
                        "1h": "lstm",
                        "4h": "gru",
                        "1d": "randomforest"
                    },
                    "neural_config": {
                        "tcn": {
                            "num_channels": [32, 64, 128],
                            "kernel_size": 2,
                            "dropout": 0.2,
                            "batch_size": 16,
                            "epochs": 10,
                            "learning_rate": 0.001
                        },
                        "cnn": {
                            "num_filters": [32, 64, 128],
                            "kernel_sizes": [3, 3, 3],
                            "dropout": 0.2,
                            "batch_size": 16,
                            "epochs": 10,
                            "learning_rate": 0.001
                        },
                        "transformer": {
                            "d_model": 64,
                            "nhead": 4,
                            "num_layers": 2,
                            "dropout": 0.1,
                            "batch_size": 16,
                            "epochs": 10,
                            "learning_rate": 0.001
                        },
                        "lstm": {
                            "hidden_size": 64,
                            "num_layers": 1,
                            "bidirectional": True,
                            "dropout": 0.2,
                            "batch_size": 16,
                            "epochs": 10,
                            "learning_rate": 0.001
                        },
                        "gru": {
                            "hidden_size": 64,
                            "num_layers": 1,
                            "bidirectional": True,
                            "dropout": 0.2,
                            "batch_size": 16,
                            "epochs": 10,
                            "learning_rate": 0.001
                        }
                    }
                }

                # Initialize trainer
                trainer = MultiOutputProbabilityTrainer(config)
                print(f"   ✅ {model_type.upper()} trainer initialized")

                # Generate multi-output targets
                y_multi = trainer.prepare_multi_output_targets(X, y, market_data)
                print(f"   ✅ Multi-output targets generated")

                # Verify targets are binary
                for target_name, target_values in y_multi.items():
                    unique_values = np.unique(target_values)
                    assert np.all(np.isin(unique_values, [0, 1])), f"Target {target_name} not binary"

                # Split data for training
                split_idx = int(0.8 * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train_multi = {k: v[:split_idx] for k, v in y_multi.items()}
                y_test_multi = {k: v[split_idx:] for k, v in y_multi.items()}

                # Train multi-output model
                trained_models = trainer.train_multi_output_model(
                    X_train, y_train_multi, X_test, y_test_multi
                )
                print(f"   ✅ {model_type.upper()} training completed")

                # Check training results
                assert len(trained_models) > 0, f"No models trained for {model_type}"
                print(f"   ✅ Trained {len(trained_models)} models")

                # Generate probability outputs
                price_action_probabilities = trainer.predict_probabilities(
                    X_test, market_data.iloc[split_idx:]
                )
                print(f"   ✅ Probability predictions generated")

                # Verify probability outputs
                expected_probabilities = [
                    "triple_barrier_probability",
                    "direction_probability",
                    "magnitude_probability",
                    "barrier_avoidance_probability"
                ]

                for prob_name in expected_probabilities:
                    assert prob_name in price_action_probabilities, f"Missing probability: {prob_name}"
                    prob_value = price_action_probabilities[prob_name]
                    assert 0.0 <= prob_value <= 1.0, f"Invalid probability value for {prob_name}: {prob_value}"
                    print(f"      {prob_name}: {prob_value:.4f}")

                # Test model saving and loading
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = os.path.join(temp_dir, f"test_{model_type}_model.pkl")

                    # Create model data for saving
                    model_data = {
                        "multi_output_trainer": trainer,
                        "trained_models": trained_models,
                        "model_type": "multi_output",
                        "architecture": model_type.upper(),
                        "timeframe": timeframe,
                        "training_date": datetime.now().isoformat(),
                        "hyperparameters": config,
                        "price_action_probabilities": price_action_probabilities
                    }

                    # Save model
                    saved_data = save_multi_output_model_with_probabilities(model_data, model_path)
                    print(f"   ✅ {model_type.upper()} model saved successfully")

                    # Load model
                    loaded_data = load_model_with_probabilities(model_path)
                    print(f"   ✅ {model_type.upper()} model loaded successfully")

                    # Test prediction with loaded model
                    loaded_trainer = loaded_data["multi_output_trainer"]
                    loaded_probabilities = loaded_trainer.predict_probabilities(X_test, market_data.iloc[split_idx:])

                    # Verify predictions are similar
                    for prob_name in expected_probabilities:
                        assert prob_name in loaded_probabilities
                        original_prob = price_action_probabilities[prob_name]
                        loaded_prob = loaded_probabilities[prob_name]
                        # Allow small differences due to floating point precision
                        assert abs(original_prob - loaded_prob) < 0.01, f"Probability mismatch for {prob_name}"

                    print(f"   ✅ {model_type.upper()} model saving/loading test passed")

                results[model_type] = True
                print(f"   ✅ {model_type.upper()} test PASSED")

            except Exception as e:
                print(f"   ❌ {model_type.upper()} test FAILED: {e}")
                results[model_type] = False
                import traceback
                traceback.print_exc()

        # Summary
        print(f"\n{'='*70}")
        print("ADVANCED MODELS INTEGRATION TEST SUMMARY")
        print(f"{'='*70}")

        passed = sum(1 for result in results.values() if result)
        total = len(results)

        for model_type, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{model_type.upper()}: {status}")

        print(f"\nOverall: {passed}/{total} model types passed")

        if passed == total:
            print("🎉 ALL ADVANCED MODELS INTEGRATION TESTS PASSED!")
            print("✅ Multi-output training framework supports all model types:")
            print("   - LightGBM (tree-based)")
            print("   - RandomForest (ensemble)")
            print("   - TCN (Temporal Convolutional Network)")
            print("   - CNN (Convolutional Neural Network)")
            print("   - Transformer (Attention-based)")
            print("   - LSTM (Long Short-Term Memory)")
            print("   - GRU (Gated Recurrent Unit)")
            print("\n🚀 The multi-output training framework is now COMPLETE!")
            print("All advanced models are integrated and working correctly.")
        else:
            print(f"⚠️ {total - passed} model types failed. Please check the implementation.")

        return passed == total

    except Exception as e:
        print(f"❌ Advanced models integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_configuration():
    """Test model configuration based on timeframe."""
    print("🧪 Testing Model Configuration by Timeframe...")

    try:
        from training.multi_output_probability_trainer import MultiOutputProbabilityTrainer

        # Test different timeframes
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        expected_models = ["cnn", "tcn", "transformer", "lightgbm", "lstm", "gru", "randomforest"]

        for timeframe, expected_model in zip(timeframes, expected_models):
            print(f"   Testing {timeframe} → {expected_model.upper()}...")

            config = {
                "timeframe": timeframe,
                "model_architectures": {
                    "1m": "cnn",
                    "5m": "tcn",
                    "15m": "transformer",
                    "30m": "lightgbm",
                    "1h": "lstm",
                    "4h": "gru",
                    "1d": "randomforest"
                }
            }

            trainer = MultiOutputProbabilityTrainer(config)

            # Check if the correct model type is configured
            for output_type in ['triple_barrier', 'direction', 'magnitude', 'barrier_avoidance']:
                model_type = trainer.config.get(f'{output_type}_model_type', 'lightgbm')
                assert model_type == expected_model, f"Expected {expected_model} for {timeframe}, got {model_type}"

            print(f"   ✅ {timeframe} correctly configured for {expected_model.upper()}")

        print("✅ Model configuration by timeframe test PASSED")
        return True

    except Exception as e:
        print(f"❌ Model configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neural_network_specific_features():
    """Test neural network specific features."""
    print("🧪 Testing Neural Network Specific Features...")

    try:
        from training.multi_output_probability_trainer import MultiOutputProbabilityTrainer
        from training.advanced_neural_models import NEURAL_MODEL_CONFIGS

        # Test TCN configuration
        config = {
            "timeframe": "5m",
            "model_architectures": {"5m": "tcn"},
            "neural_config": {
                "tcn": NEURAL_MODEL_CONFIGS["tcn"]
            }
        }

        trainer = MultiOutputProbabilityTrainer(config)

        # Check if neural config is properly set
        assert "tcn" in trainer.neural_config
        assert "num_channels" in trainer.neural_config["tcn"]
        assert "kernel_size" in trainer.neural_config["tcn"]

        print("   ✅ TCN configuration properly set")

        # Test Transformer configuration
        config = {
            "timeframe": "15m",
            "model_architectures": {"15m": "transformer"},
            "neural_config": {
                "transformer": NEURAL_MODEL_CONFIGS["transformer"]
            }
        }

        trainer = MultiOutputProbabilityTrainer(config)

        # Check if neural config is properly set
        assert "transformer" in trainer.neural_config
        assert "d_model" in trainer.neural_config["transformer"]
        assert "nhead" in trainer.neural_config["transformer"]

        print("   ✅ Transformer configuration properly set")

        print("✅ Neural network specific features test PASSED")
        return True

    except Exception as e:
        print(f"❌ Neural network specific features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_advanced_integration_tests():
    """Run all advanced integration tests."""
    print("🚀 Starting Advanced Models Integration Tests")
    print("=" * 70)

    tests = [
        ("All Model Types Integration", test_all_model_types),
        ("Model Configuration by Timeframe", test_model_configuration),
        ("Neural Network Specific Features", test_neural_network_specific_features)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*70}")
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
    print(f"\n{'='*70}")
    print("ADVANCED INTEGRATION TEST SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL ADVANCED INTEGRATION TESTS PASSED!")
        print("\n🎯 ADVANCED MODELS INTEGRATION STATUS: 100% COMPLETE")
        print("✅ All 7 model types are integrated and working")
        print("✅ Timeframe-based model configuration is working")
        print("✅ Neural network specific features are working")
        print("✅ Model saving and loading is working for all types")
        print("✅ Multi-output training framework is complete")
        print("\n🚀 The multi-output training framework now supports ALL advanced models!")
        print("The system is ready for production use with the full range of model architectures.")
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = run_advanced_integration_tests()
    sys.exit(0 if success else 1)