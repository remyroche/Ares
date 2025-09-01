"""
Model Probability Generator

This module provides the main interface for generating probability outputs for trained models.
It coordinates the probability calculation framework to generate all 4 required probability
outputs for the Enhanced Prediction Service.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple
from datetime import datetime
import logging

from .probability_calculators import (
    get_probability_calculator,
    ClassificationProbabilityCalculator,
    RegressionProbabilityCalculator
)

logger = logging.getLogger(__name__)


class ModelProbabilityGenerator:
    """
    Main class for generating probability outputs for trained models.

    This class coordinates the probability calculation framework to generate
    all 4 required probability outputs:
    1. Triple barrier probability
    2. Direction probability
    3. Magnitude probability
    4. Barrier avoidance probability
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.classification_calculator = ClassificationProbabilityCalculator()
        self.regression_calculator = RegressionProbabilityCalculator()

    def generate_price_action_probabilities(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        market_data: pd.DataFrame,
        model_type: str = "classification",
        **kwargs
    ) -> Dict[str, float]:
        """
        Generate all 4 required probability outputs for a trained model.

        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test targets
            market_data: Market data with OHLCV information
            model_type: Type of model ('classification' or 'regression')
            **kwargs: Additional parameters for probability calculations

        Returns:
            Dict containing all 4 probability outputs
        """
        try:
            self.logger.info(f"Generating probability outputs for {model_type} model")

            # Get appropriate calculator
            calculator = get_probability_calculator(model_type)

            # Generate all 4 probability outputs
            probabilities = {
                "triple_barrier_probability": self._calculate_triple_barrier_probability(
                    calculator, model, X_test, market_data, **kwargs
                ),
                "direction_probability": self._calculate_direction_probability(
                    calculator, model, X_test, y_test, **kwargs
                ),
                "magnitude_probability": self._calculate_magnitude_probability(
                    calculator, model, X_test, market_data, **kwargs
                ),
                "barrier_avoidance_probability": self._calculate_barrier_avoidance_probability(
                    calculator, model, X_test, market_data, **kwargs
                )
            }

            # Add metadata
            probabilities["generation_timestamp"] = datetime.now().isoformat()
            probabilities["model_type"] = model_type

            self.logger.info(f"Generated probabilities: {probabilities}")
            return probabilities

        except Exception as e:
            self.logger.error(f"Error generating probability outputs: {e}")
            # Return default probabilities
            return self._get_default_probabilities(model_type)

    def _calculate_triple_barrier_probability(
        self,
        calculator: Union[ClassificationProbabilityCalculator, RegressionProbabilityCalculator],
        model: Any,
        X_test: np.ndarray,
        market_data: pd.DataFrame,
        **kwargs
    ) -> float:
        """Calculate triple barrier probability."""
        try:
            profit_target = kwargs.get('profit_target', 0.02)
            stop_loss = kwargs.get('stop_loss', 0.01)
            volatility_window = kwargs.get('volatility_window', 20)

            if isinstance(calculator, ClassificationProbabilityCalculator):
                return calculator.calculate_triple_barrier_probability(
                    model, X_test, market_data, profit_target, stop_loss, volatility_window
                )
            else:
                return calculator.calculate_triple_barrier_probability(
                    model, X_test, market_data, profit_target, stop_loss
                )
        except Exception as e:
            self.logger.error(f"Error calculating triple barrier probability: {e}")
            return 0.5

    def _calculate_direction_probability(
        self,
        calculator: Union[ClassificationProbabilityCalculator, RegressionProbabilityCalculator],
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs
    ) -> float:
        """Calculate direction probability."""
        try:
            return calculator.calculate_direction_probability(model, X_test, y_test)
        except Exception as e:
            self.logger.error(f"Error calculating direction probability: {e}")
            return 0.5

    def _calculate_magnitude_probability(
        self,
        calculator: Union[ClassificationProbabilityCalculator, RegressionProbabilityCalculator],
        model: Any,
        X_test: np.ndarray,
        market_data: pd.DataFrame,
        **kwargs
    ) -> float:
        """Calculate magnitude probability."""
        try:
            threshold_factor = kwargs.get('threshold_factor', 0.8)
            return calculator.calculate_magnitude_probability(
                model, X_test, market_data, threshold_factor
            )
        except Exception as e:
            self.logger.error(f"Error calculating magnitude probability: {e}")
            return 0.5

    def _calculate_barrier_avoidance_probability(
        self,
        calculator: Union[ClassificationProbabilityCalculator, RegressionProbabilityCalculator],
        model: Any,
        X_test: np.ndarray,
        market_data: pd.DataFrame,
        **kwargs
    ) -> float:
        """Calculate barrier avoidance probability."""
        try:
            adverse_threshold = kwargs.get('adverse_threshold', 0.01)
            return calculator.calculate_barrier_avoidance_probability(
                model, X_test, market_data, adverse_threshold
            )
        except Exception as e:
            self.logger.error(f"Error calculating barrier avoidance probability: {e}")
            return 0.5

    def validate_probabilities(self, probabilities: Dict[str, float]) -> bool:
        """
        Validate that all probability outputs are valid.

        Args:
            probabilities: Dictionary of probability outputs

        Returns:
            bool: True if all probabilities are valid
        """
        required_keys = [
            "triple_barrier_probability",
            "direction_probability",
            "magnitude_probability",
            "barrier_avoidance_probability"
        ]

        # Check all required keys exist
        for key in required_keys:
            if key not in probabilities:
                self.logger.error(f"Missing required probability key: {key}")
                return False

        # Check all probabilities are between 0 and 1
        for key in required_keys:
            prob = probabilities[key]
            if not isinstance(prob, (int, float)) or not 0.0 <= prob <= 1.0:
                self.logger.error(f"Invalid probability value for {key}: {prob}")
                return False

        return True


# Convenience function for easy access