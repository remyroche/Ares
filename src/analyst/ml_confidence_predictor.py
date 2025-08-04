import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
import numpy as np

from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class MLConfidencePredictor:
    """
    ML Confidence Predictor that generates predictions with confidence scores
    for price increases and expected price decreases in table format.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ML Confidence Predictor with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MLConfidencePredictor")

        # Model state
        self.model: Any | None = None
        self.is_trained: bool = False
        self.last_training_time: datetime | None = None
        self.model_performance: dict[str, float] = {}

        # Configuration
        self.predictor_config: dict[str, Any] = self.config.get(
            "ml_confidence_predictor",
            {},
        )
        self.model_path: str = self.predictor_config.get(
            "model_path",
            "models/confidence_predictor.joblib",
        )

        # Confidence score levels for price movements (direction-neutral)
        self.price_movement_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        
        # Adverse movement levels (opposite direction risk)
        self.adverse_movement_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Directional confidence analysis (0-2% range for high leverage trading)
        self.directional_confidence_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        self.liquidation_risk_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ML confidence predictor configuration"),
            AttributeError: (False, "Missing required predictor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="ML confidence predictor initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize ML Confidence Predictor with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing ML Confidence Predictor...")

            # Load predictor configuration
            await self._load_predictor_configuration()

            # Initialize model parameters
            await self._initialize_model_parameters()

            # Load existing model if available
            await self._load_existing_model()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(
                    "Invalid configuration for ML confidence predictor",
                )
                return False

            self.logger.info(
                "✅ ML Confidence Predictor initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(
                f"❌ ML Confidence Predictor initialization failed: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="predictor configuration loading",
    )
    async def _load_predictor_configuration(self) -> None:
        """Load predictor configuration."""
        # Set default predictor parameters
        self.predictor_config.setdefault("model_path", "models/confidence_predictor.joblib")
        self.predictor_config.setdefault("min_samples_for_training", 500)
        self.predictor_config.setdefault("confidence_threshold", 0.6)
        self.predictor_config.setdefault("max_prediction_horizon", 1)  # hours

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model parameters initialization",
    )
    async def _initialize_model_parameters(self) -> None:
        """Initialize model parameters."""
        # Ensure model directory exists
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # Initialize performance metrics
        self.model_performance = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    @handle_file_operations(
        default_return=None,
        context="model loading",
    )
    async def _load_existing_model(self) -> None:
        """Load existing model if available."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                self.logger.info("✅ Loaded existing confidence predictor model")
            except Exception as e:
                self.logger.warning(f"Failed to load existing model: {e}")
                self.model = None
                self.is_trained = False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate predictor configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate required parameters
            required_params = [
                "model_path",
                "retrain_interval_hours",
                "min_samples_for_training",
            ]
            
            for param in required_params:
                if param not in self.predictor_config:
                    self.logger.error(f"Missing required parameter: {param}")
                    return False

            # Validate parameter values
            if self.predictor_config["retrain_interval_hours"] <= 0:
                self.logger.error("Retrain interval must be positive")
                return False

            if self.predictor_config["min_samples_for_training"] < 100:
                self.logger.error("Minimum samples for training must be at least 100")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False


    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for prediction"),
            AttributeError: (None, "Model not properly trained"),
        },
        default_return=None,
        context="confidence prediction",
    )
    async def predict_confidence_table(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        Predict confidence scores for price movements (direction-neutral) and adverse movement risks.
        Always returns predictions, even if model is not fully trained.

        Args:
            market_data: Recent market data
            current_price: Current market price

        Returns:
            Optional[Dict[str, Any]]: Prediction table with direction-neutral analysis
        """
        try:
            if market_data.empty:
                self.logger.error("Empty market data provided")
                return None

            self.logger.info("Generating confidence prediction table...")

            # Prepare features
            features = await self._prepare_prediction_features(market_data)
            if features is None:
                self.logger.warning("Failed to prepare features, using fallback predictions")
                return self._generate_fallback_predictions(current_price)

            # Initialize predictions
            confidence_predictions = {}
            expected_decrease_predictions = {}

            # Check if model is trained and available
            if self.is_trained and self.model:
                self.logger.info("Using trained model for predictions")
                
                # Predict confidence scores for price increases
                for increase_level in self.price_movement_levels:
                    target_col = f"confidence_{increase_level}"
                    if target_col in self.model:
                        try:
                            confidence = self.model[target_col].predict(features.iloc[-1:])[0]
                            confidence_predictions[increase_level] = max(0.0, min(1.0, confidence))
                        except Exception as e:
                            self.logger.warning(f"Failed to predict for {target_col}: {e}")
                            confidence_predictions[increase_level] = self._get_fallback_confidence(increase_level)

                # Predict expected price decreases
                for decrease_level in self.adverse_movement_levels:
                    target_col = f"expected_decrease_{decrease_level}"
                    if target_col in self.model:
                        try:
                            decrease_prob = self.model[target_col].predict(features.iloc[-1:])[0]
                            expected_decrease_predictions[decrease_level] = max(0.0, min(1.0, decrease_prob))
                        except Exception as e:
                            self.logger.warning(f"Failed to predict for {target_col}: {e}")
                            expected_decrease_predictions[decrease_level] = self._get_fallback_decrease_probability(decrease_level)
            else:
                self.logger.warning("Model not trained, using fallback predictions")
                return self._generate_fallback_predictions(current_price)

            # Ensure all levels have predictions
            confidence_predictions = self._ensure_complete_predictions(
                confidence_predictions, self.price_movement_levels, "confidence"
            )
            expected_decrease_predictions = self._ensure_complete_predictions(
                expected_decrease_predictions, self.adverse_movement_levels, "decrease"
            )

            # Generate directional confidence analysis
            directional_analysis = self._generate_directional_confidence_analysis(
                confidence_predictions, expected_decrease_predictions, current_price
            )
            
            # Create direction-neutral prediction table
            prediction_table = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "movement_confidence_scores": confidence_predictions,
                "adverse_movement_risks": expected_decrease_predictions,
                "directional_confidence": directional_analysis,
                "model_performance": self.model_performance if hasattr(self, 'model_performance') else {},
                "prediction_source": "trained_model" if self.is_trained and self.model else "fallback",
            }

            self.logger.info("✅ Confidence prediction table generated successfully")
            return prediction_table

        except Exception as e:
            self.logger.error(f"Error generating confidence prediction table: {e}")
            # Return fallback predictions instead of None
            return self._generate_fallback_predictions(current_price)

    def _generate_fallback_predictions(self, current_price: float) -> dict[str, Any]:
        """
        Generate fallback predictions when model is not available or fails.
        
        Args:
            current_price: Current market price
            
        Returns:
            dict[str, Any]: Fallback prediction table
        """
        try:
            confidence_predictions = {}
            expected_decrease_predictions = {}
            
            # Generate fallback confidence scores
            for increase_level in self.price_movement_levels:
                confidence_predictions[increase_level] = self._get_fallback_confidence(increase_level)
            
            # Generate fallback decrease probabilities
            for decrease_level in self.adverse_movement_levels:
                expected_decrease_predictions[decrease_level] = self._get_fallback_decrease_probability(decrease_level)
            
            # Generate directional confidence analysis
            directional_analysis = self._generate_directional_confidence_analysis(
                confidence_predictions, expected_decrease_predictions, current_price
            )
            
            return {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "movement_confidence_scores": confidence_predictions,
                "adverse_movement_risks": expected_decrease_predictions,
                "directional_confidence": directional_analysis,
                "model_performance": {},
                "prediction_source": "fallback",
            }
            
        except Exception as e:
            self.logger.error(f"Error generating fallback predictions: {e}")
            return {}

    def _get_fallback_confidence(self, increase_level: float) -> float:
        """
        Get fallback confidence score for a given price increase level.
        
        Args:
            increase_level: Price increase level (0.3, 0.4, etc.)
            
        Returns:
            float: Fallback confidence score between 0 and 1
        """
        try:
            # Base confidence decreases with higher increase levels
            base_confidence = 0.5
            level_factor = 1.0 - (increase_level / 10.0)  # Higher levels = lower confidence
            confidence = base_confidence * level_factor
            
            # Add some randomness to avoid deterministic predictions
            import random
            confidence += random.uniform(-0.1, 0.1)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error getting fallback confidence: {e}")
            return 0.5

    def _get_fallback_decrease_probability(self, decrease_level: float) -> float:
        """
        Get fallback decrease probability for a given price decrease level.
        
        Args:
            decrease_level: Price decrease level (0.1, 0.2, etc.)
            
        Returns:
            float: Fallback decrease probability between 0 and 1
        """
        try:
            # Base probability increases with higher decrease levels
            base_probability = 0.3
            level_factor = decrease_level / 10.0  # Higher levels = higher probability
            probability = base_probability * level_factor
            
            # Add some randomness
            import random
            probability += random.uniform(-0.05, 0.05)
            
            return max(0.0, min(1.0, probability))
            
        except Exception as e:
            self.logger.error(f"Error getting fallback decrease probability: {e}")
            return 0.3

    def _ensure_complete_predictions(
        self, 
        predictions: dict[str, float], 
        levels: List[float], 
        prediction_type: str
    ) -> dict[str, float]:
        """
        Ensure all levels have predictions, filling missing ones with fallbacks.
        
        Args:
            predictions: Current predictions dictionary
            levels: List of levels that should have predictions
            prediction_type: Type of prediction ("confidence" or "decrease")
            
        Returns:
            dict[str, float]: Complete predictions dictionary
        """
        try:
            complete_predictions = predictions.copy()
            
            for level in levels:
                if level not in complete_predictions:
                    if prediction_type == "confidence":
                        complete_predictions[level] = self._get_fallback_confidence(level)
                    else:
                        complete_predictions[level] = self._get_fallback_decrease_probability(level)
            
            return complete_predictions
            
        except Exception as e:
            self.logger.error(f"Error ensuring complete predictions: {e}")
            return predictions

    def _generate_directional_confidence_analysis(
        self,
        confidence_scores: dict[str, float],
        expected_decreases: dict[str, float],
        current_price: float
    ) -> dict[str, Any]:
        """
        Generate directional confidence analysis to predict likelihood of reaching target
        before significant adverse movement that could cause liquidation.
        
        Args:
            confidence_scores: Confidence scores for price increases
            expected_decreases: Expected price decrease probabilities
            current_price: Current market price
            
        Returns:
            dict[str, Any]: Directional confidence analysis
        """
        try:
            directional_analysis = {
                "target_reach_confidence": {},
                "adverse_movement_risk": {},
                "liquidation_risk_assessment": {},
                "directional_safety_score": {},
                "recommended_leverage": {},
                "risk_adjusted_targets": {},
            }
            
            # Analyze each target level
            for target_level in self.directional_confidence_levels:
                target_price = current_price * (1 + target_level / 100)
                
                # Calculate confidence of reaching target
                target_confidence = self._calculate_target_reach_confidence(
                    target_level, confidence_scores
                )
                
                # Calculate risk of adverse movement before reaching target
                adverse_risk = self._calculate_adverse_movement_risk(
                    target_level, expected_decreases
                )
                
                # Calculate directional safety score
                safety_score = self._calculate_directional_safety_score(
                    target_confidence, adverse_risk
                )
                                
                # Calculate risk-adjusted target
                risk_adjusted_target = self._calculate_risk_adjusted_target(
                    target_level, safety_score
                )
                
                directional_analysis["target_reach_confidence"][target_level] = {
                    "confidence": target_confidence,
                    "target_price": target_price,
                    "price_movement_required": target_level,
                }
                
                directional_analysis["adverse_movement_risk"][target_level] = {
                    "risk": adverse_risk,
                    "risk_level": "high" if adverse_risk > 0.7 else "medium" if adverse_risk > 0.4 else "low",
                }
                
                
                directional_analysis["directional_safety_score"][target_level] = {
                    "safety_score": safety_score,
                    "safety_level": "high" if safety_score > 0.7 else "medium" if safety_score > 0.4 else "low",
                }
                                
                directional_analysis["risk_adjusted_targets"][target_level] = {
                    "original_target": target_level,
                    "risk_adjusted_target": risk_adjusted_target,
                    "adjustment_factor": safety_score,
                }
            
            return directional_analysis
            
        except Exception as e:
            self.logger.error(f"Error generating directional confidence analysis: {e}")
            return {}

    def _calculate_target_reach_confidence(
        self, target_level: float, confidence_scores: dict[str, float]
    ) -> float:
        """
        Calculate confidence of reaching a specific target level.
        
        Args:
            target_level: Target price movement percentage
            confidence_scores: Confidence scores for different levels
            
        Returns:
            float: Confidence of reaching target
        """
        try:
            # Find the closest confidence score for this target
            closest_level = min(confidence_scores.keys(), key=lambda x: abs(float(x) - target_level))
            base_confidence = confidence_scores[closest_level]
            
            # Adjust confidence based on target distance
            distance_factor = 1.0 - (target_level / 10.0)  # Higher targets = lower confidence
            adjusted_confidence = base_confidence * distance_factor
            
            return max(0.0, min(1.0, adjusted_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating target reach confidence: {e}")
            return 0.5

    def _calculate_adverse_movement_risk(
        self, target_level: float, expected_decreases: dict[str, float]
    ) -> float:
        """
        Calculate risk of adverse movement before reaching target.
        
        Args:
            target_level: Target price movement percentage
            expected_decreases: Expected decrease probabilities
            
        Returns:
            float: Risk of adverse movement
        """
        try:
            # Calculate weighted average of adverse movement risks
            total_risk = 0.0
            total_weight = 0.0
            
            for decrease_level, probability in expected_decreases.items():
                # Higher decrease levels have more impact on liquidation risk
                weight = float(decrease_level) / 10.0
                total_risk += probability * weight
                total_weight += weight
            
            if total_weight > 0:
                average_risk = total_risk / total_weight
            else:
                average_risk = 0.3  # Default risk
            
            # Adjust risk based on target distance (longer targets = higher risk)
            distance_factor = target_level / 10.0
            adjusted_risk = average_risk * (1.0 + distance_factor)
            
            return max(0.0, min(1.0, adjusted_risk))
            
        except Exception as e:
            self.logger.error(f"Error calculating adverse movement risk: {e}")
            return 0.3

    def _calculate_directional_safety_score(
        self, target_confidence: float, adverse_risk: float
    ) -> float:
        """
        Calculate directional safety score.
        
        Args:
            target_confidence: Confidence of reaching target
            adverse_risk: Risk of adverse movement
            
        Returns:
            float: Safety score (0.0 to 1.0)
        """
        try:
            # Safety score is high when:
            # 1. High target confidence
            # 2. Low adverse movement risk
            
            safety_score = (target_confidence * 0.7) + ((1.0 - adverse_risk) * 0.3)
            return max(0.0, min(1.0, safety_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating directional safety score: {e}")
            return 0.5

    def _calculate_risk_adjusted_target(
        self, target_level: float, safety_score: float
    ) -> float:
        """
        Calculate risk-adjusted target based on safety score.
        
        Args:
            target_level: Original target level
            safety_score: Directional safety score
            
        Returns:
            float: Risk-adjusted target level
        """
        try:
            # Adjust target based on safety score
            # Lower safety score = more conservative target
            adjustment_factor = 0.5 + (safety_score * 0.5)  # 0.5 to 1.0
            adjusted_target = target_level * adjustment_factor
            
            return adjusted_target
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted target: {e}")
            return target_level



    def get_model_performance(self) -> dict[str, float]:
        """Get model performance metrics."""
        return self.model_performance

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML confidence predictor cleanup",
    )
    async def stop(self) -> None:
        """Clean up resources."""
        try:
            self.logger.info("Stopping ML Confidence Predictor...")
            # Cleanup code here if needed
            self.logger.info("✅ ML Confidence Predictor stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping ML Confidence Predictor: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="ML confidence predictor setup",
)
async def setup_ml_confidence_predictor(
    config: dict[str, Any] | None = None,
) -> MLConfidencePredictor | None:
    """
    Setup ML Confidence Predictor.

    Args:
        config: Configuration dictionary

    Returns:
        Optional[MLConfidencePredictor]: Initialized predictor or None
    """
    try:
        if config is None:
            config = {}

        predictor = MLConfidencePredictor(config)
        if await predictor.initialize():
            return predictor
        else:
            return None

    except Exception as e:
        system_logger.error(f"Error setting up ML Confidence Predictor: {e}")
        return None
