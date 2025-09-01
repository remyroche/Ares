#!/usr/bin/env python3
"""
Test script for the probability generation framework.

This script tests the probability calculation framework and model probability generator
to ensure they work correctly with different model types.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_probability_calculators():
    """Test the probability calculators with different model types."""
    logger.info("🧪 Testing probability calculators...")

    try:
        from src.training.probability_calculators import (
            ClassificationProbabilityCalculator,
            RegressionProbabilityCalculator,
            get_probability_calculator
        )

        # Test classification calculator
        clf_calc = ClassificationProbabilityCalculator()
        logger.info("✅ ClassificationProbabilityCalculator imported successfully")

        # Test regression calculator
        reg_calc = RegressionProbabilityCalculator()
        logger.info("✅ RegressionProbabilityCalculator imported successfully")

        # Test factory function
        calc = get_probability_calculator("classification")
        assert isinstance(calc, ClassificationProbabilityCalculator)
        logger.info("✅ Factory function works correctly")

        return True

    except Exception as e:
        logger.error(f"❌ Error testing probability calculators: {e}")
        return False

def test_model_probability_generator():
    """Test the model probability generator."""
    logger.info("🧪 Testing model probability generator...")

    try:
        from src.training.model_probability_generator import ModelProbabilityGenerator

        generator = ModelProbabilityGenerator()
        logger.info("✅ ModelProbabilityGenerator imported successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Error testing model probability generator: {e}")
        return False

def test_model_saving_utils():
    """Test the model saving utilities."""
    logger.info("🧪 Testing model saving utilities...")

    try:
            save_model_with_probabilities,
            load_model_with_probabilities,
            validate_model_probabilities
        )

        logger.info("✅ Model saving utilities imported successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Error testing model saving utilities: {e}")
        return False

def test_end_to_end_probability_generation():
    """Test end-to-end probability generation with a real model."""
    logger.info("🧪 Testing end-to-end probability generation...")

    try:
        from src.training.model_probability_generator import ModelProbabilityGenerator
        from src.training.model_saving_utils import save_model_with_probabilities, load_model_with_probabilities

        # Generate synthetic data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create market data
        market_data = pd.DataFrame({
            'close': np.random.randn(len(X_test)),
            'volume': np.random.randn(len(X_test))
        })

        # Train a simple model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Test probability generation
        generator = ModelProbabilityGenerator()
        probabilities = generator.generate_price_action_probabilities(
            model, X_test, y_test, market_data, model_type="classification"
        )

        # Validate probabilities
        required_keys = [
            "triple_barrier_probability",
            "direction_probability",
            "magnitude_probability",
            "barrier_avoidance_probability"
        ]

        for key in required_keys:
            assert key in probabilities, f"Missing required key: {key}"
            assert 0.0 <= probabilities[key] <= 1.0, f"Invalid probability value for {key}: {probabilities[key]}"

        logger.info(f"✅ Generated probabilities: {probabilities}")

        # Test model saving and loading
        model_data = {
            "model": model,
            "model_type": "classification",
            "accuracy": accuracy_score(y_test, model.predict(X_test)),
            "training_date": "2024-01-01T00:00:00"
        }

        model_path = "test_model_with_probabilities.pkl"

        # Save model
        saved_data = save_model_with_probabilities(
            model_data, model_path, probabilities, save_format="joblib"
        )
        logger.info("✅ Model saved with probabilities successfully")

        # Load model
        loaded_data = load_model_with_probabilities(model_path)
        logger.info("✅ Model loaded with probabilities successfully")

        # Validate loaded model
        assert "price_action_probabilities" in loaded_data
        assert loaded_data["price_action_probabilities"] == probabilities

        logger.info("✅ End-to-end probability generation test passed!")

        # Clean up
        import os
        if os.path.exists(model_path):
            os.remove(model_path)

        return True

    except Exception as e:
        logger.error(f"❌ Error in end-to-end test: {e}")
        return False

def test_regression_model_probabilities():
    """Test probability generation with regression models."""
    logger.info("🧪 Testing regression model probability generation...")

    try:
        from src.training.model_probability_generator import ModelProbabilityGenerator

        # Generate synthetic regression data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)  # Continuous target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create market data
        market_data = pd.DataFrame({
            'close': np.random.randn(len(X_test)),
            'volume': np.random.randn(len(X_test))
        })

        # Train a regression model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Test probability generation
        generator = ModelProbabilityGenerator()
        probabilities = generator.generate_price_action_probabilities(
            model, X_test, y_test, market_data, model_type="regression"
        )

        # Validate probabilities
        required_keys = [
            "triple_barrier_probability",
            "direction_probability",
            "magnitude_probability",
            "barrier_avoidance_probability"
        ]

        for key in required_keys:
            assert key in probabilities, f"Missing required key: {key}"
            assert 0.0 <= probabilities[key] <= 1.0, f"Invalid probability value for {key}: {probabilities[key]}"

        logger.info(f"✅ Generated regression probabilities: {probabilities}")

        return True

    except Exception as e:
        logger.error(f"❌ Error in regression test: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting probability framework tests...")

    tests = [
        ("Probability Calculators", test_probability_calculators),
        ("Model Probability Generator", test_model_probability_generator),
        ("Model Saving Utils", test_model_saving_utils),
        ("End-to-End Classification", test_end_to_end_probability_generation),
        ("Regression Model Probabilities", test_regression_model_probabilities),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")

        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")

    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")

    if passed == total:
        logger.info("🎉 All tests passed! Probability framework is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)