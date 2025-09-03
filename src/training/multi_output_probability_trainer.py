"""
Multi-Output Probability Trainer

This module implements multi-output training for probability outputs, replacing
the post-training calculation approach with direct training on probability targets.
"""

from src.core.decorators import (
    handles_errors,
    log_execution_time
)
from src.core.domain import (
    PerformanceLevel,
    ValidationLevel,
    comprehensive_validation
)
from typing import Any
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from src.core.decorators import validates, log_execution_time
from src.utils.logger import system_logger

# Import advanced neural network models
from .advanced_neural_models import (
    NEURAL_MODEL_CONFIGS,
    NeuralNetworkWrapper,
    create_neural_model,
)

from src.utils.logger import system_logger
from catboost import CatBoostClassifier
def fit(
        self,
        X_train: np.ndarray,
        y_train_multi: dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val_multi: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """
        Train all 4 probability models.

        Args:
            X_train: Training features
            y_train_multi: Training targets for all 4 probability types
            X_val: Validation features
            y_val_multi: Validation targets for all 4 probability types

        Returns:
            Dictionary containing trained models and metadata
        """
        self.logger.info("Starting multi-output model training")
        trained_models = {}

        for output_type in ["triple_barrier", "direction", "magnitude", "barrier_avoidance"]:
            self.logger.info(f"Training {output_type} model...")

            # Get model and targets
            model = self.models[output_type]
            y_train_target = y_train_multi[output_type]
            y_val_target = y_val_multi[output_type]

            # Update input size for neural networks if needed
            if hasattr(model, "model_class") and hasattr(model, "model_params"):
                # This is a neural network wrapper
                model.model_params["input_size"] = X_train.shape[1]

            # Handle class imbalance for certain targets
            sample_weights = None
            if output_type in ["triple_barrier", "barrier_avoidance"]:
                # These targets are often imbalanced
                try:
                    class_weights = compute_class_weight(
                        "balanced",
                        classes=np.unique(y_train_target),
                        y=y_train_target,
                    )
                    sample_weights = class_weights[y_train_target.astype(int)]
                except Exception as e:
                    self.logger.warning(f"Could not compute class weights for {output_type}: {e}")

            # Train model
            try:
                if hasattr(model, "fit"):
                    # Check if it's a neural network (NeuralNetworkWrapper)
                    if isinstance(model, NeuralNetworkWrapper):
                        # Neural networks handle their own training
                        model.fit(X_train, y_train_target)
                        trained_models[output_type] = model
                    else:
                        # Traditional ML models
                        if sample_weights is not None:
                            model.fit(X_train, y_train_target, sample_weight=sample_weights)
                        else:
                            model.fit(X_train, y_train_target)

                        # Calibrate probabilities for non-neural models
                        try:
                            calibrator = CalibratedClassifierCV(model, cv=5, method="isotonic")
                            calibrator.fit(X_val, y_val_target)
                            self.calibrators[output_type] = calibrator
                            trained_models[output_type] = calibrator
                        except Exception as e:
                            self.logger.warning(f"Calibration failed for {output_type}, using original model: {e}")
                            trained_models[output_type] = model
                else:
                    self.logger.error(f"Model {output_type} does not have fit method")
                    msg = f"Model {output_type} does not have fit method"
                    raise ValueError(msg)
            except Exception as e:
                self.logger.exception(f"Training failed for {output_type}: {e}")
                # Continue with other models instead of failing completely
                self.logger.warning(f"Skipping {output_type} model due to training failure")
                continue

        # Optimize ensemble weights
        self.ensemble_weights = self._optimize_ensemble_weights(
            trained_models, X_val, y_val_multi,
        )

        self.logger.info("Multi-output model training completed")
        self.logger.info(f"Successfully trained {len(trained_models)} out of 4 models")
        return trained_models

    @handles_errors(fallback=None)
    def _optimize_ensemble_weights(
        self,
        models: dict[str, Any],
        X_val: np.ndarray,
        y_val_multi: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """
        Optimize ensemble weights for better probability accuracy.

        Args:
            models: Trained models
            X_val: Validation features
            y_val_multi: Validation targets

        Returns:
            Dictionary of optimized weights
        """
        def objective(weights):
            """Objective function to minimize."""
            total_loss = 0

            for i, output_type in enumerate(["triple_barrier", "direction", "magnitude", "barrier_avoidance"]):
                model = models[output_type]
                y_true = y_val_multi[output_type]

                try:
                    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of positive class
                    # Calculate Brier score (lower is better)
                    brier_score = np.mean((y_pred_proba - y_true) ** 2)
                    total_loss += brier_score * weights[i]
                except Exception as e:
                    self.logger.warning(f"Error calculating loss for {output_type}: {e}")
                    total_loss += 1.0 * weights[i]  # Penalty for failed prediction

            return total_loss

        # Initial weights (equal)
        initial_weights = [0.25, 0.25, 0.25, 0.25]

        try:
            # Optimize weights
            result = minimize(
                objective,
                initial_weights,
                method="L-BFGS-B",
                bounds=[(0.1, 0.4) for _ in range(4)],  # Constrain weights
            )

            optimized_weights = dict(zip(
                ["triple_barrier", "direction", "magnitude", "barrier_avoidance"],
                result.x, strict=False,
            ))

            self.logger.info(f"Optimized ensemble weights: {optimized_weights}")
            return optimized_weights

        except Exception as e:
            self.logger.warning(f"Ensemble weight optimization failed: {e}")
            return dict(zip(
                ["triple_barrier", "direction", "magnitude", "barrier_avoidance"],
                initial_weights, strict=False,
            ))

    @handles_errors(fallback={})
    def predict_probabilities(
        self,
        X_test: np.ndarray,
        market_data: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Generate all 4 probability outputs.

        Args:
            X_test: Test features
            market_data: Market data (for compatibility)

        Returns:
            Dictionary containing all 4 probability outputs
        """
        probabilities = {}

        for output_type in ["triple_barrier", "direction", "magnitude", "barrier_avoidance"]:
            # Check if model exists
            if output_type not in self.models or self.models[output_type] is None:
                self.logger.warning(f"Model for {output_type} not available, using default probability")
                probabilities[f"{output_type}_probability"] = 0.5
                continue

            model = self.calibrators.get(output_type, self.models[output_type])

            try:
                # Get probability predictions
                if hasattr(model, "predict_proba"):
                    # Handle both traditional ML models and neural networks
                    if isinstance(model, NeuralNetworkWrapper):
                        # Neural networks return probabilities directly
                        proba = model.predict_proba(X_test)
                    else:
                        # Traditional ML models
                        proba = model.predict_proba(X_test)

                    if proba.shape[1] > 1:
                        # Binary classification, get positive class probability
                        prob_value = proba[:, 1].mean()
                    else:
                        # Single class, use the probability
                        prob_value = proba[:, 0].mean()
                else:
                    # Fallback to prediction
                    pred = model.predict(X_test)
                    prob_value = pred.mean()

                # Ensure probability is in [0, 1] range
                prob_value = np.clip(prob_value, 0.0, 1.0)

                probabilities[f"{output_type}_probability"] = float(prob_value)

            except Exception as e:
                self.logger.exception(f"Error predicting {output_type} probability: {e}")
                probabilities[f"{output_type}_probability"] = 0.5  # Default fallback

        # Add metadata
        probabilities["generation_timestamp"] = datetime.now().isoformat()
        probabilities["model_type"] = "multi_output"

        return probabilities

class MultiOutputProbabilityTrainer:
    """
    Main class for multi-output probability training.

    This class coordinates the entire multi-output training process,
    from target generation to model training and prediction.
from src.core.decorators import handles_errors
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger

        # Initialize components
        self.target_generator = ProbabilityTargetGenerator(config)
        self.multi_output_model = MultiOutputModel(config)

        # Training state
        self.is_trained = False
        self.trained_models = None
        self.ensemble_weights = None
        self.calibrators = None

        # Advanced model configuration
        self.model_architectures = self.config.get("model_architectures", {
            "1m": "cnn",      # CNN for 1-minute data (Tactician)
            "5m": "tcn",      # TCN for 5-minute data (Analyst)
            "15m": "transformer", # Transformer for 15-minute data (Enhanced)
            "30m": "lightgbm",    # LightGBM for 30-minute data (Analyst)
            "1h": "hmm_regime",    # HMM regime definition only
        })

        # Neural network configuration
        self.neural_config = self.config.get("neural_config", {})

        # Configure models based on timeframe if provided
        self.timeframe = self.config.get("timeframe", "30m")
        self._configure_models_for_timeframe()

    def _configure_models_for_timeframe(self):
        """Configure models based on the specified timeframe."""
        if self.timeframe in self.model_architectures:
            model_type = self.model_architectures[self.timeframe]
            self.logger.info(f"Configuring models for {self.timeframe} timeframe using {model_type}")

            # Update config for each output type
            for output_type in ["triple_barrier", "direction", "magnitude", "barrier_avoidance"]:
                self.config[f"{output_type}_model_type"] = model_type

            # Update neural config if it's a neural network
            if model_type in ["tcn", "cnn", "transformer", "lstm", "gru"]:
                self.neural_config[model_type] = NEURAL_MODEL_CONFIGS.get(model_type, {})
        else:
            self.logger.warning(f"No specific model configuration for timeframe {self.timeframe}, using defaults")

    @handles_errors(fallback={})
    @validates(strict=True)
    def prepare_multi_output_targets(
        self,
        X: np.ndarray,
        y: np.ndarray,
        market_data: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """
        Generate 4 probability targets for training.

        Args:
            X: Feature matrix
            y: Target values
            market_data: Market data with OHLCV information

        Returns:
            Dictionary containing all 4 probability targets
        """
        self.logger.info("Preparing multi-output targets for training")
        return self.target_generator.generate_all_targets(X, y, market_data)
    
    @log_execution_time()
    def train_multi_output_model(
        self,
        X_train: np.ndarray,
        y_train_multi: dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val_multi: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """
        Train model on all 4 probability targets.

        Args:
            X_train: Training features
            y_train_multi: Training targets for all 4 probability types
            X_val: Validation features
            y_val_multi: Validation targets for all 4 probability types

        Returns:
            Dictionary containing trained models and metadata
        """
        self.logger.info("Starting multi-output model training")

        # Train the multi-output model
        self.trained_models = self.multi_output_model.fit(
            X_train, y_train_multi, X_val, y_val_multi,
        )

        # Get ensemble weights and calibrators from the multi-output model
        self.ensemble_weights = self.multi_output_model.ensemble_weights
        self.calibrators = self.multi_output_model.calibrators

        self.is_trained = True
        self.logger.info("Multi-output model training completed")

        return self.trained_models

    @handles_errors(fallback={})
    def predict_probabilities(
        self,
        X_test: np.ndarray,
        market_data: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Generate all 4 probability outputs.

        Args:
            X_test: Test features
            market_data: Market data (for compatibility)

        Returns:
            Dictionary containing all 4 probability outputs
        """
        if not self.is_trained or self.trained_models is None:
            self.logger.error("Model not trained. Call train_multi_output_model first.")
            return self._get_default_probabilities()

        self.logger.info("Generating probability predictions")

        # Use the multi-output model's prediction method
        try:
            return self.multi_output_model.predict_probabilities(X_test, market_data)
        except Exception as e:
            self.logger.exception(f"Error in multi-output model prediction: {e}")
            return self._get_default_probabilities()

    def _get_default_probabilities(self) -> dict[str, float]:
        """Get default probabilities when training fails."""
        return {
            "triple_barrier_probability": 0.5,
            "direction_probability": 0.5,
            "magnitude_probability": 0.5,
            "barrier_avoidance_probability": 0.5,
            "generation_timestamp": datetime.now().isoformat(),
            "model_type": "multi_output",
        }

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the trained model."""
        if not self.is_trained:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "ensemble_weights": self.multi_output_model.ensemble_weights,
            "model_types": {name: type(model).__name__ for name, model in self.trained_models.items()},
            "calibrators": list(self.multi_output_model.calibrators.keys()),
        }
