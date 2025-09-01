#!/usr/bin/env python3
"""
Test script for multi-output training implementation.

This script tests the complete multi-output training pipeline including:
1. Probability target generation
2. Multi-output model training
3. Model saving and loading
4. Probability output generation
"""

import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime

# Import the multi-output training components
from src.training.multi_output_probability_trainer import (
import MultiOutputProbabilityTrainer,
    MultiOutputProbabilityTrainer,
    ProbabilityTargetGenerator
)
    create_multi_output_trainer,
    MultiOutputModelConfig
)
from src.training.model_saving_utils import (
import save_multi_output_model_with_probabilities,
    save_multi_output_model_with_probabilities,
    load_multi_output_model_with_probabilities,
    validate_model_probabilities
)


def test_probability_target_generation():
    pass
    pass
    """Test probability target generation."""
    print("🧪 Testing probability target generation...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples)

    # Create market data
    market_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n_samples) * 0.01) + 100,  # Simulate price series
        'volume': np.random.randn(n_samples) * 1000 + 5000
    })

    # Initialize target generator
    config = {
        'profit_target': 0.02,
        'stop_loss': 0.01,
        'look_ahead_periods': 20,
        'magnitude_threshold_factor': 0.8,
        'adverse_threshold': 0.01,
        'avoidance_look_ahead': 10
    }

    target_generator = ProbabilityTargetGenerator(config)

    # Generate all targets
    targets = target_generator.generate_all_targets(X, y, market_data)

    # Validate targets
    assert len(targets) == 4, f"Expected 4 targets, got {len(targets)}"

    expected_keys = ['triple_barrier', 'direction', 'magnitude', 'barrier_avoidance']
    for key in expected_keys:
    pass
    pass
        assert key in targets, f"Missing target: {key}"
        assert len(targets[key]) == n_samples, f"Target {key} has wrong length"
        assert np.all((targets[key] >= 0) & (targets[key] <= 1)), f"Target {key} has values outside [0,1]"

    print("✅ Probability target generation test passed")
    return targets


def test_multi_output_probability_trainer():
    pass
    pass
    """Test the complete multi-output probability trainer."""
    print("🧪 Testing multi-output probability trainer...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples)

    # Create market data
    market_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
        'volume': np.random.randn(n_samples) * 1000 + 5000
    })

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    market_train = market_data.iloc[:split_idx]
    market_test = market_data.iloc[split_idx:]

    # Initialize trainer
    config = {
        'use_lightgbm': True,
        'n_estimators': 100,  # Reduced for testing
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    }

    trainer = MultiOutputProbabilityTrainer(config)

    # Generate targets
    y_train_multi = trainer.prepare_multi_output_targets(X_train, y_train, market_train)
    y_test_multi = trainer.prepare_multi_output_targets(X_test, y_test, market_test)

    # Train model
    trained_models = trainer.train_multi_output_model(X_train, y_train_multi, X_test, y_test_multi)

    # Generate predictions
    probabilities = trainer.predict_probabilities(X_test, market_test)

    # Validate outputs
    assert len(probabilities) >= 4, f"Expected at least 4 probabilities, got {len(probabilities)}"

    required_keys = [
        'triple_barrier_probability',
        'direction_probability',
        'magnitude_probability',
        'barrier_avoidance_probability'
    ]

    for key in required_keys:
    pass
    pass
        assert key in probabilities, f"Missing probability: {key}"
        assert 0.0 <= probabilities[key] <= 1.0, f"Invalid probability for {key}: {probabilities[key]}"

    print("✅ Multi-output probability trainer test passed")
    return trainer, probabilities


def test_multi_output_model_trainer():
    pass
    pass
    """Test the enhanced multi-output model trainer."""
    print("🧪 Testing enhanced multi-output model trainer...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples)

    # Create market data
    market_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
        'volume': np.random.randn(n_samples) * 1000 + 5000
    })

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    market_train = market_data.iloc[:split_idx]
    market_test = market_data.iloc[split_idx:]

    # Create multi-output trainer
    trainer = create_multi_output_trainer(
        model_type="LightGBM",
        enable_probability_outputs=True,
        use_profit_features=True,
        probability_config={
            "profit_target": 0.02,
            "stop_loss": 0.01,
            "look_ahead_periods": 20,
            "magnitude_threshold_factor": 0.8,
            "adverse_threshold": 0.01,
            "avoidance_look_ahead": 10
        }
    )

    # Train with probability targets
    training_result = trainer.train_with_probability_targets(
        X_train=X_train,
        X_val=X_test,
        y_train=y_train,
        y_val=y_test,
        market_data=market_data,
        feature_names=[f"feature_{i}" for i in range(n_features)]
    )

    # Validate results
    assert "trained_models" in training_result, "Missing trained_models in result"
    assert "probability_outputs" in training_result, "Missing probability_outputs in result"
    assert "probability_metrics" in training_result, "Missing probability_metrics in result"

    trained_models = training_result["trained_models"]
    probability_outputs = training_result["probability_outputs"]

    # Check that we have models for all probability types
    expected_prob_types = [
        "triple_barrier_probability",
        "direction_probability",
        "magnitude_probability",
        "barrier_avoidance_probability"
    ]

    for prob_type in expected_prob_types:
    pass
    pass
        assert prob_type in trained_models, f"Missing trained model for {prob_type}"
        assert prob_type in probability_outputs, f"Missing probability output for {prob_type}"
        assert 0.0 <= probability_outputs[prob_type] <= 1.0, f"Invalid probability for {prob_type}"

    print("✅ Enhanced multi-output model trainer test passed")
    return trainer, training_result


def test_model_saving_and_loading():
    pass
    pass
    """Test model saving and loading functionality."""
    print("🧪 Testing model saving and loading...")

    # Create a test trainer and train it
    trainer, training_result = test_multi_output_model_trainer()

    # Prepare model data for saving
    model_data = {
        "multi_output_trainer": trainer,
        "trained_models": training_result["trained_models"],
        "model_type": "multi_output",
        "training_date": datetime.now().isoformat(),
        "hyperparameters": {
            "model_type": "LightGBM",
            "enable_probability_outputs": True
        },
        "metrics": {},
        "probability_metrics": training_result["probability_metrics"]
    }

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_multi_output_model.pkl")

        # Save model
        try:
            saved_data = save_multi_output_model_with_probabilities(
                model_data, model_path, save_format="joblib"
    except Exception as e:
        pass
    except Exception as e:
        pass
            )
            print(f"✅ Model saved successfully to {model_path}")
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False

        # Load model
        try:
            loaded_data = load_multi_output_model_with_probabilities(model_path)
    except Exception as e:
        pass
    except Exception as e:
        pass
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

        # Validate loaded model
        is_valid = validate_model_probabilities(loaded_data)
        if is_valid:
    pass
    pass
            print("✅ Loaded model validation passed")
        else:
            print("❌ Loaded model validation failed")
            return False

        # Check that key components are preserved
        assert loaded_data["model_type"] == "multi_output", "Model type not preserved"
        assert "multi_output_trainer" in loaded_data, "Multi-output trainer not preserved"
        assert "trained_models" in loaded_data, "Trained models not preserved"
        assert "price_action_probabilities" in loaded_data, "Probability outputs not preserved"

    print("✅ Model saving and loading test passed")
    return True


def test_integration_with_existing_models():
    pass
    pass
    """Test integration with existing model architectures."""
    print("🧪 Testing integration with existing models...")

    # Test with different model types (using existing architectures)
    model_types = ["LightGBM", "RandomForest"]  # Core models with existing architectures

    # Add neural network models if available
    try:
        from src.training.multi_output_model_trainer import EXISTING_MODELS_AVAILABLE
    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import if EXISTING_MODELS_AVAILABLE:
        if EXISTING_MODELS_AVAILABLE:
    pass
    pass
            model_types.extend(["CNN", "TCN", "Transformer"])
            print("  Neural network models available for testing")
    except ImportError:
        print("  Neural network models not available for testing")

    for model_type in model_types:
    pass
    pass
        print(f"  Testing {model_type} (using existing architecture)...")

        # Generate synthetic data
        np.random.seed(42)
        n_samples = 300
        n_features = 8

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], size=n_samples)

        # Create market data
        market_data = pd.DataFrame({
            'close': np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
            'volume': np.random.randn(n_samples) * 1000 + 5000
        })

        # Split data
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Create trainer
        trainer = create_multi_output_trainer(
            model_type=model_type,
            enable_probability_outputs=True,
            use_profit_features=True
        )

        # Train with probability targets
        training_result = trainer.train_with_probability_targets(
            X_train=X_train,
            X_val=X_test,
            y_train=y_train,
            y_val=y_test,
            market_data=market_data,
            feature_names=[f"feature_{i}" for i in range(n_features)]
        )

        # Validate results
        assert training_result["model_type"] == f"MultiOutput_{model_type}", f"Wrong model type for {model_type}"
        assert "trained_models" in training_result, f"Missing trained models for {model_type}"
        assert "probability_outputs" in training_result, f"Missing probability outputs for {model_type}"

        print(f"    ✅ {model_type} integration test passed")

    print("✅ Integration with existing models test passed")
    return True


def run_comprehensive_test():
    pass
    pass
    """Run all tests."""
    print("🚀 Starting comprehensive multi-output training tests...")
    print("=" * 60)

    test_results = []

    # Test 1: Probability target generation
    try:
        targets = test_probability_target_generation()
    except Exception as e:
        pass
    except Exception as e:
        pass
        test_results.append(("Probability Target Generation", True))
    except Exception as e:
        print(f"❌ Probability target generation test failed: {e}")
        test_results.append(("Probability Target Generation", False))

    # Test 2: Multi-output probability trainer
    try:
        trainer, probabilities = test_multi_output_probability_trainer()
    except Exception as e:
        pass
    except Exception as e:
        pass
        test_results.append(("Multi-Output Probability Trainer", True))
    except Exception as e:
        print(f"❌ Multi-output probability trainer test failed: {e}")
        test_results.append(("Multi-Output Probability Trainer", False))

    # Test 3: Enhanced multi-output model trainer
    try:
        trainer, training_result = test_multi_output_model_trainer()
    except Exception as e:
        pass
    except Exception as e:
        pass
        test_results.append(("Enhanced Multi-Output Model Trainer", True))
    except Exception as e:
        print(f"❌ Enhanced multi-output model trainer test failed: {e}")
        test_results.append(("Enhanced Multi-Output Model Trainer", False))

    # Test 4: Model saving and loading
    try:
        success = test_model_saving_and_loading()
    except Exception as e:
        pass
    except Exception as e:
        pass
        test_results.append(("Model Saving and Loading", success))
    except Exception as e:
        print(f"❌ Model saving and loading test failed: {e}")
        test_results.append(("Model Saving and Loading", False))

    # Test 5: Integration with existing models
    try:
        success = test_integration_with_existing_models()
    except Exception as e:
        pass
    except Exception as e:
        pass
        test_results.append(("Integration with Existing Models", success))
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        test_results.append(("Integration with Existing Models", False))

    # Print summary
    print("\\\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
    pass
    pass
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<35} {status}")
        if result:
    pass
    pass
            passed += 1

    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed")

    if passed == total:
    pass
    pass
        print("🎉 All tests passed! Multi-output training implementation is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    pass
    pass
    success = run_comprehensive_test()
    exit(0 if success else 1)