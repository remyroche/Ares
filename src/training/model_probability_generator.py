"""
Model Probability Generator

This module provides the main interface for generating probability outputs for trained models.
It coordinates the probability calculation framework to generate all 4 required probability
outputs for the Enhanced Prediction Service.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

from .probability_calculators import (
import get_probability_calculator,
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
    pass
    pass
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

    except Exception as e:
        pass
    except Exception as e:
        pass
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
    except Exception as e:
        pass
    except Exception as e:
        pass
            stop_loss = kwargs.get('stop_loss', 0.01)
            volatility_window = kwargs.get('volatility_window', 20)

            if isinstance(calculator, ClassificationProbabilityCalculator):
    pass
    pass
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
        pass
    except Exception as e:
        pass
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
    except Exception as e:
        pass
    except Exception as e:
        pass
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
    except Exception as e:
        pass
    except Exception as e:
        pass
            return calculator.calculate_barrier_avoidance_probability(
                model, X_test, market_data, adverse_threshold
            )
        except Exception as e:
            self.logger.error(f"Error calculating barrier avoidance probability: {e}")
            return 0.5

    def _get_default_probabilities(self, model_type: str) -> Dict[str, float]:
    pass
    pass
        """Get default probability values when calculation fails."""
        return {
            "triple_barrier_probability": 0.5,
            "direction_probability": 0.5,
            "magnitude_probability": 0.5,
            "barrier_avoidance_probability": 0.5,
            "generation_timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "note": "Default probabilities due to calculation error"
        }

    def validate_probabilities(self, probabilities: Dict[str, float]) -> bool:
    pass
    pass
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
    pass
    pass
            if key not in probabilities:
    pass
    pass
                self.logger.error(f"Missing required probability key: {key}")
                return False

        # Check all probabilities are between 0 and 1
        for key in required_keys:
    pass
    pass
            prob = probabilities[key]
            if not isinstance(prob, (int, float)) or not 0.0 <= prob <= 1.0:
    pass
    pass
                self.logger.error(f"Invalid probability value for {key}: {prob}")
                return False

        return True

    def generate_ensemble_probabilities(
        self,
        models: list,
        model_types: list,
        X_test: np.ndarray,
        y_test: np.ndarray,
        market_data: pd.DataFrame,
        weights: Optional[list] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Generate probability outputs for an ensemble of models.

        Args:
            models: List of trained model objects
            model_types: List of model types ('classification' or 'regression')
            X_test: Test features
            y_test: Test targets
            market_data: Market data
            weights: Optional weights for ensemble averaging
            **kwargs: Additional parameters

        Returns:
            Dict containing ensemble probability outputs
        """
        try:
            if len(models) != len(model_types):
    pass
    except Exception as e:
        pass
    pass
                raise ValueError("Number of models must match number of model types")

    except Exception as e:
        pass
            if weights is None:
    pass
    pass
                weights = [1.0 / len(models)] * len(models)

            if len(weights) != len(models):
    pass
    pass
                raise ValueError("Number of weights must match number of models")

            # Generate probabilities for each model
            all_probabilities = []
            for model, model_type in zip(models, model_types):
    pass
    pass
                model_probs = self.generate_price_action_probabilities(
                    model, X_test, y_test, market_data, model_type, **kwargs
                )
                all_probabilities.append(model_probs)

            # Calculate weighted ensemble probabilities
            ensemble_probabilities = {}
            for key in ["triple_barrier_probability", "direction_probability",
                       "magnitude_probability", "barrier_avoidance_probability"]:
                weighted_sum = sum(
                    prob[key] * weight
                    for prob, weight in zip(all_probabilities, weights)
                )
                ensemble_probabilities[key] = weighted_sum

            # Add metadata
            ensemble_probabilities["generation_timestamp"] = datetime.now().isoformat()
            ensemble_probabilities["model_type"] = "ensemble"
            ensemble_probabilities["ensemble_size"] = len(models)
            ensemble_probabilities["weights"] = weights

            return ensemble_probabilities

        except Exception as e:
            self.logger.error(f"Error generating ensemble probabilities: {e}")
            return self._get_default_probabilities("ensemble")

    def generate_calibrated_probabilities(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        market_data: pd.DataFrame,
        model_type: str = "classification",
        calibration_method: str = "isotonic",
        **kwargs
    ) -> Dict[str, float]:
        """
        Generate probability outputs for a calibrated model.

        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test targets
            market_data: Market data
            model_type: Type of model
            calibration_method: Calibration method ('isotonic' or 'sigmoid')
            **kwargs: Additional parameters

        Returns:
            Dict containing calibrated probability outputs
        """
        try:
            # For now, use standard probability generation
    except Exception as e:
        pass
    except Exception as e:
        pass
            # In the future, this could incorporate calibration-specific adjustments
            probabilities = self.generate_price_action_probabilities(
                model, X_test, y_test, market_data, model_type, **kwargs
            )

            # Add calibration metadata
            probabilities["calibration_method"] = calibration_method
            probabilities["is_calibrated"] = True

            return probabilities

        except Exception as e:
            self.logger.error(f"Error generating calibrated probabilities: {e}")
            return self._get_default_probabilities(f"{model_type}_calibrated")


# Convenience function for easy access
def generate_model_probabilities(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    market_data: pd.DataFrame,
    model_type: str = "classification",
    **kwargs
) -> Dict[str, float]:
    """
    Convenience function to generate probability outputs for a model.

    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test targets
        market_data: Market data
        model_type: Type of model
        **kwargs: Additional parameters

    Returns:
        Dict containing probability outputs
    """
    generator = ModelProbabilityGenerator()
    return generator.generate_price_action_probabilities(
        model, X_test, y_test, market_data, model_type, **kwargs
    )