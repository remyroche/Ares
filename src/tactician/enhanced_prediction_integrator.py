# src/tactician/enhanced_prediction_integrator.py

from src.core.decorators import (
    handles_errors,
    traced,
    validates
)
from pathlib import Path
from typing import Any
from datetime import datetime
import pandas as pd
import yaml

<<<<<<< HEAD
from src.core.decorators import handles_errors, traced
from src.tactician.dynamic_barrier_calculator import DynamicBarrierCalculator
from src.utils.logger import get_logger
from src.utils.warning_symbols import error, warning


class TacticianEnhancedPredictionIntegrator:
    """
    Enhanced Prediction Integrator for Tactician that delivers multi-outcome predictions
    similar to the Analyst but with smaller price deviations and higher confidence.

    Key Features:
    - Price deviation % without hitting the opposite barrier (smaller than Analyst)
    - Price direction prediction
    - Confidence that we will reach a certain price
    - Shorter timeframes (1m, 5m vs Analyst's longer timeframes)
    - Higher precision using dynamic barriers
    - Integration with Analyst predictions
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the Tactician enhanced prediction integrator."""
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize dynamic barrier calculator
        self.barrier_calculator = DynamicBarrierCalculator(config)

        # Load Tactician configuration
        self._load_tactician_config()

        # Initialize prediction models (placeholder for actual ML models)
        self.tactician_models = self._initialize_tactician_models()

        self.logger.info("🚀 Tactician Enhanced Prediction Integrator initialized")

    def _load_tactician_config(self) -> None:
        """Load Tactician-specific configuration."""
        tactician_config = self.config.get("tactician_triple_barrier", {})

        # Timeframe configuration
        self.timeframes = tactician_config.get("timeframes", ["1m", "5m"])
        self.primary_timeframe = tactician_config.get("primary_timeframe", "1m")
        self.secondary_timeframe = tactician_config.get("secondary_timeframe", "5m")

        # Precision configuration (simplified)
        self.precision_threshold = tactician_config.get("precision_threshold", 0.85)

        # Multi-outcome prediction configuration (similar to Analyst)
        self.prediction_types = [
            "price_deviation_prediction",    # Price deviation % for 2 barrier combinations
            "price_direction_prediction",    # Price direction (long/short)
            "price_target_confidence",        # Confidence to reach upper barrier before lower barrier
        ]

        # ML model confidence factors (to be determined by ML model, not hardcoded)
        step12_config = self.config.get("step12_confidence_optimization", {})
        ml_config = step12_config.get("ml_confidence_factors", {})

        self.ml_confidence_factors = {
            "price_deviation_prediction": ml_config.get("price_deviation_prediction"),  # Will be ML model output
            "price_direction_prediction": ml_config.get("price_direction_prediction"),  # Will be ML model output
            "price_target_confidence": ml_config.get("price_target_confidence"),         # Will be ML model output
        }

    def _initialize_tactician_models(self) -> dict[str, Any]:
        """Initialize Tactician prediction models."""
        # Placeholder for actual ML models
        # In production, these would be trained models for each prediction type
        models = {}

        for prediction_type in self.prediction_types:
            models[prediction_type] = {
                "model": None,  # Placeholder for actual model
                "confidence": 0.90,  # Higher base confidence than Analyst
                "timeframe": self.primary_timeframe,
                "ml_confidence_factor": self.ml_confidence_factors.get(prediction_type, 1.0),
            }

        return models

    @handles_errors
    @traced("Tactician.generateEnhancedPredictions")
    async def generate_tactician_predictions(
        self,
        market_data: pd.DataFrame,
        analyst_predictions: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str = None,
    ) -> dict[str, Any]:
        """
        Generate Tactician enhanced predictions based on Analyst predictions.

        Args:
            market_data: Market data for the timeframe
            analyst_predictions: Analyst's multi-outcome predictions
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Target timeframe (1m or 5m)

        Returns:
            dict: Tactician's enhanced multi-outcome predictions
        """
        try:
            # Determine timeframe
            if timeframe is None:
                timeframe = self._determine_optimal_timeframe(market_data)

            # Get dynamic barriers for this timeframe (smaller than Analyst)
            upper_barrier, lower_barrier = self.barrier_calculator.calculate_dynamic_barriers(timeframe)

            # Generate enhanced predictions for each type
            tactician_predictions = {}

            for prediction_type in self.prediction_types:
                enhanced_prediction = await self._generate_enhanced_prediction(
                    prediction_type=prediction_type,
                    market_data=market_data,
                    analyst_predictions=analyst_predictions,
                    upper_barrier=upper_barrier,
                    lower_barrier=lower_barrier,
                    timeframe=timeframe,
                    symbol=symbol,
                    exchange=exchange,
                )

                if enhanced_prediction:
                    tactician_predictions[prediction_type] = enhanced_prediction

            # Add metadata
            tactician_predictions["metadata"] = {
                "timeframe": timeframe,
                "symbol": symbol,
                "exchange": exchange,
                "upper_barrier": upper_barrier,
                "lower_barrier": lower_barrier,
                "precision_threshold": self.precision_threshold,
                "high_precision_mode": self.enable_high_precision_mode,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(f"🎯 Generated {len(tactician_predictions)} Tactician enhanced predictions for {timeframe}")

            return tactician_predictions

        except Exception as e:
            self.logger.exception(error(f"❌ Error generating Tactician predictions: {e}"))
            return {}

    async def _generate_enhanced_prediction(
        self,
        prediction_type: str,
        market_data: pd.DataFrame,
        analyst_predictions: dict[str, Any],
        upper_barrier: float,
        lower_barrier: float,
        timeframe: str,
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Generate enhanced prediction for a specific type."""
        try:
            # Get base prediction from Analyst
            base_prediction = self._extract_analyst_prediction(analyst_predictions, prediction_type)

            if base_prediction is None:
                return None

            # Apply Tactician-specific enhancement
            enhanced_prediction = self._apply_tactician_enhancement(
                prediction_type=prediction_type,
                base_prediction=base_prediction,
                market_data=market_data,
                upper_barrier=upper_barrier,
                lower_barrier=lower_barrier,
                timeframe=timeframe,
            )

            # Apply basic precision filtering
            if enhanced_prediction.get("confidence", 0.0) < self.precision_threshold:
                return None

            return enhanced_prediction

        except Exception as e:
            self.logger.exception(error(f"❌ Error generating {prediction_type} prediction: {e}"))
            return None

    def _extract_analyst_prediction(
        self,
        analyst_predictions: dict[str, Any],
        prediction_type: str,
    ) -> dict[str, Any] | None:
        """Extract base prediction from Analyst predictions."""
        try:
            # Map Tactician prediction types to Analyst prediction types
            analyst_type_mapping = {
                "price_deviation_prediction": "price_prediction",
                "price_direction_prediction": "direction_prediction",
                "price_target_confidence": "confidence_prediction",
            }

            analyst_type = analyst_type_mapping.get(prediction_type, prediction_type)

            # Look for prediction in various possible locations
            for key, value in analyst_predictions.items():
                if analyst_type in key.lower():
                    return value
                if isinstance(value, dict) and analyst_type in value:
                    return value[analyst_type]

            # Fallback: create synthetic prediction based on market data
            return self._create_synthetic_prediction(prediction_type, analyst_predictions)

        except Exception as e:
            self.logger.warning(warning(f"⚠️ Could not extract {prediction_type} from Analyst predictions: {e}"))
            return None

    def _create_synthetic_prediction(
        self,
        prediction_type: str,
        analyst_predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Create synthetic prediction when Analyst prediction is not available."""
        try:
            # Extract any available confidence or prediction values
            base_confidence = 0.5
            base_prediction = 0.0

            for value in analyst_predictions.values():
                if isinstance(value, dict):
                    if "confidence" in value:
                        base_confidence = value["confidence"]
                    if "prediction" in value:
                        base_prediction = value["prediction"]

            # Create synthetic prediction based on type
            if prediction_type == "price_deviation_prediction":
                # Price deviation for both 50% and 25% of Analyst barriers
                # This will be calculated by the ML model based on market conditions
                synthetic_value = base_prediction * 0.5  # Base 50% of Analyst deviation
            elif prediction_type == "price_direction_prediction":
                # Same direction as Analyst
                synthetic_value = base_prediction  # Keep same direction
            elif prediction_type == "price_target_confidence":
                # Confidence calculated by ML model (not just boosted)
                synthetic_value = base_confidence  # Base confidence, ML model will enhance
            else:
                synthetic_value = base_prediction

            return {
                "prediction": synthetic_value,
                "confidence": base_confidence,
                "model_type": "synthetic",
                "prediction_type": prediction_type,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.exception(error(f"❌ Error creating synthetic prediction: {e}"))
            return {
                "prediction": 0.0,
                "confidence": 0.5,
                "model_type": "synthetic",
                "prediction_type": prediction_type,
                "timestamp": datetime.now().isoformat(),
            }

    def _apply_tactician_enhancement(
        self,
        prediction_type: str,
        base_prediction: dict[str, Any],
        market_data: pd.DataFrame,
        upper_barrier: float,
        lower_barrier: float,
        timeframe: str,
    ) -> dict[str, Any]:
        """Apply Tactician-specific enhancement to base prediction."""
        try:
            # Get ML model confidence factor for this prediction type
            ml_confidence_factor = self.ml_confidence_factors.get(prediction_type, 1.0)
            if ml_confidence_factor is None:
                # Fallback to base confidence if ML model hasn't provided factor yet
                ml_confidence_factor = 1.0

            # Extract base values
            base_value = base_prediction.get("prediction", 0.0)
            base_confidence = base_prediction.get("confidence", 0.5)

            # Apply Tactician-specific enhancement based on prediction type
            enhanced_value = self._enhance_prediction_value(
                prediction_type=prediction_type,
                base_value=base_value,
                market_data=market_data,
                upper_barrier=upper_barrier,
                lower_barrier=lower_barrier,
                timeframe=timeframe,
            )

            # Calculate enhanced confidence using ML model factor
            enhanced_confidence = self._calculate_enhanced_confidence(
                base_confidence=base_confidence,
                ml_confidence_factor=ml_confidence_factor,
                market_data=market_data,
                timeframe=timeframe,
            )

            # Calculate precision score
            precision_score = self._calculate_precision_score(
                enhanced_confidence=enhanced_confidence,
                market_data=market_data,
                timeframe=timeframe,
            )

            return {
                "prediction": enhanced_value,
                "confidence": enhanced_confidence,
                "precision_score": precision_score,
                "model_type": "tactician_enhanced",
                "prediction_type": prediction_type,
                "timeframe": timeframe,
                "upper_barrier": upper_barrier,
                "lower_barrier": lower_barrier,
                "ml_confidence_factor": ml_confidence_factor,
                "base_prediction": base_prediction,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.exception(error(f"❌ Error applying Tactician enhancement: {e}"))
            return base_prediction

    def _enhance_prediction_value(
        self,
        prediction_type: str,
        base_value: float,
        market_data: pd.DataFrame,
        barrier_combinations: dict[str, tuple[float, float]],
        timeframe: str,
    ) -> dict[str, float]:
        """Enhance prediction value for 2 barrier combinations."""
        try:
            if prediction_type == "price_deviation_prediction":
                # Calculate price deviations for 2 barrier combinations
                current_price = market_data["close"].iloc[-1] if not market_data.empty else 100.0

                deviations = {}
                for barrier_name, (upper_barrier, lower_barrier) in barrier_combinations.items():
                    # Calculate deviations for this barrier combination
                    upper_deviation = (upper_barrier - current_price) / current_price
                    lower_deviation = (current_price - lower_barrier) / current_price

                    # Store both deviations for this combination
                    deviations[barrier_name] = {
                        "upper_deviation": upper_deviation,
                        "lower_deviation": lower_deviation,
                    }

                return deviations

            if prediction_type == "price_direction_prediction":
                # Same direction as Analyst for 2 barrier combinations
                directions = {}
                for barrier_name in barrier_combinations:
                    directions[barrier_name] = base_value  # Keep same direction

                return directions

            if prediction_type == "price_target_confidence":
                # Confidence to reach upper barrier before lower barrier for each combination
                # ML model calculates this based on market conditions, volatility, and barrier distances
                confidences = {}
                for barrier_name, (upper_barrier, lower_barrier) in barrier_combinations.items():
                    # ML model will calculate confidence based on:
                    # - Market volatility
                    # - Barrier distances
                    # - Recent price action
                    # - Support/resistance levels
                    # For now, use base confidence - ML model will enhance this
                    confidences[barrier_name] = base_value

                return confidences

            # Return base value for all combinations
            return dict.fromkeys(barrier_combinations.keys(), base_value)

        except Exception as e:
            self.logger.exception(f"❌ Error enhancing prediction value: {e}")
            # Return fallback values for all combinations
            return dict.fromkeys(barrier_combinations.keys(), base_value)

    def _calculate_enhanced_confidence(
        self,
        base_confidence: float,
        ml_confidence_factor: float,
        market_data: pd.DataFrame,
        timeframe: str,
    ) -> float:
        """Calculate enhanced confidence using ML model factor."""
        try:
            # Base enhancement from ML model factor
            enhanced_confidence = min(1.0, base_confidence * ml_confidence_factor)

            # Additional enhancement based on market data quality
            if not market_data.empty:
                # Check for recent volatility (lower volatility = higher confidence)
                recent_volatility = market_data["close"].pct_change().tail(10).std()
                volatility_factor = max(0.6, 1.0 - recent_volatility * 8)  # Less penalty for volatility

                # Check for data quality (more recent data = higher confidence)
                data_freshness = 1.0  # Assume fresh data

                # Apply factors
                enhanced_confidence *= volatility_factor * data_freshness

            return min(1.0, enhanced_confidence)

        except Exception as e:
            self.logger.exception(error(f"❌ Error calculating enhanced confidence: {e}"))
            return base_confidence

    def _calculate_precision_score(
        self,
        enhanced_confidence: float,
        market_data: pd.DataFrame,
        timeframe: str,
    ) -> float:
        """Calculate precision score for the enhanced prediction."""
        try:
            # Base precision from confidence
            precision_score = enhanced_confidence

            # Adjust based on timeframe (shorter timeframes get higher precision)
            timeframe_factor = 1.3 if timeframe == "1m" else 1.1  # Higher boost for 1m
            precision_score *= timeframe_factor

            # Adjust based on market data quality
            if not market_data.empty:
                # More data points = higher precision
                data_quality_factor = min(1.3, len(market_data) / 100)
                precision_score *= data_quality_factor

            return min(1.0, precision_score)

        except Exception as e:
            self.logger.exception(error(f"❌ Error calculating precision score: {e}"))
            return enhanced_confidence

    def _determine_optimal_timeframe(self, market_data: pd.DataFrame) -> str:
        """Determine optimal timeframe based on market data."""
        try:
            if market_data.empty:
                return self.primary_timeframe

            # Check data frequency
            if len(market_data) > 1:
                time_diff = market_data.index[1] - market_data.index[0]
                if time_diff <= timedelta(minutes=1):
                    return "1m"
                if time_diff <= timedelta(minutes=5):
                    return "5m"

            return self.primary_timeframe

        except Exception as e:
            self.logger.exception(error(f"❌ Error determining optimal timeframe: {e}"))
            return self.primary_timeframe

    @handles_errors
    async def validate_tactician_predictions(
        self,
        tactician_predictions: dict[str, Any],
        analyst_predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate Tactician predictions against Analyst predictions."""
        try:
            validation_results = {
                "is_valid": True,
                "validation_score": 0.0,
                "issues": [],
                "enhancements": [],
            }

            total_score = 0.0
            valid_predictions = 0

            for prediction_type in self.prediction_types:
                if prediction_type in tactician_predictions:
                    tactician_pred = tactician_predictions[prediction_type]
                    analyst_pred = self._extract_analyst_prediction(analyst_predictions, prediction_type)

                    if analyst_pred:
                        # Validate confidence enhancement using ML model factors
                        ml_confidence_factor = self.ml_confidence_factors.get(prediction_type, 1.0)
                        if ml_confidence_factor is None:
                            ml_confidence_factor = 1.0  # Fallback

                        tactician_confidence = tactician_pred.get("confidence", 0.0)
                        analyst_confidence = analyst_pred.get("confidence", 0.0)

                        # Check if Tactician confidence meets ML model expectations
                        if ml_confidence_factor > 1.0:
                            expected_confidence = analyst_confidence * ml_confidence_factor * 0.8
                            if tactician_confidence >= expected_confidence:
                                validation_results["enhancements"].append(f"{prediction_type}: ML model confidence enhancement")
                                total_score += 1.0
                            else:
                                validation_results["issues"].append(f"{prediction_type}: Insufficient ML model confidence enhancement")
                        else:
                            # ML model indicates no enhancement needed
                            validation_results["enhancements"].append(f"{prediction_type}: ML model baseline confidence")
                            total_score += 0.5

                        # Check prediction value enhancement
                        tactician_value = tactician_pred.get("prediction", 0.0)
                        analyst_value = analyst_pred.get("prediction", 0.0)

                        if prediction_type == "price_deviation_prediction":
                            # Tactician should have smaller deviation
                            if abs(tactician_value) <= abs(analyst_value):
                                total_score += 0.5
                            else:
                                validation_results["issues"].append(f"{prediction_type}: Deviation not smaller than Analyst")
                        else:
                            # Other predictions should be enhanced appropriately
                            total_score += 0.5

                        valid_predictions += 1
                    else:
                        validation_results["issues"].append(f"{prediction_type}: No Analyst prediction available")
                else:
                    validation_results["issues"].append(f"{prediction_type}: Missing Tactician prediction")

            # Calculate overall validation score
            if valid_predictions > 0:
                validation_results["validation_score"] = total_score / valid_predictions
                validation_results["is_valid"] = validation_results["validation_score"] >= 0.7

            return validation_results

        except Exception as e:
            self.logger.exception(error(f"❌ Error validating Tactician predictions: {e}"))
            return {"is_valid": False, "validation_score": 0.0, "issues": [str(e)], "enhancements": []}

    def update_ml_confidence_factors(self, new_factors: dict[str, float]) -> None:
        """Update ML confidence factors dynamically (called by ML model)."""
        try:
            for prediction_type, factor in new_factors.items():
                if prediction_type in self.ml_confidence_factors:
                    self.ml_confidence_factors[prediction_type] = factor
                    self.logger.info(f"Updated ML confidence factor for {prediction_type}: {factor}")
                else:
                    self.logger.warning(f"Unknown prediction type for ML confidence factor: {prediction_type}")
        except Exception as e:
            self.logger.exception(f"Error updating ML confidence factors: {e}")

    def load_step12_ml_confidence_factors(self, step12_results_path: str = None) -> bool:
        """
        Automatically load ML confidence factors from step12 results.
        This method is called automatically when step12 completes.

        Args:
            step12_results_path: Path to step12 results file (optional)

        Returns:
            bool: True if factors loaded successfully
        """
        try:
            # Try to load from step12 results
            if step12_results_path and Path(step12_results_path).exists():
                # Load from specific file
                with open(step12_results_path) as f:
                    step12_results = yaml.safe_load(f)
            else:
                # Try to load from default step12 results location
                default_paths = [
                    "step12_results.yaml",
                    "step12_ml_confidence_factors.yaml",
                    "src/config/step12_results.yaml",
                    "src/config/step12_ml_confidence_factors.yaml",
                ]

                step12_results = None
                for path in default_paths:
                    if Path(path).exists():
                        with open(path) as f:
                            step12_results = yaml.safe_load(f)
                            self.logger.info(f"Loaded step12 results from: {path}")
                            break

                if not step12_results:
                    self.logger.warning("No step12 results found, using default ML confidence factors")
                    return False

            # Extract ML confidence factors from step12 results
            if "ml_confidence_factors" in step12_results:
                ml_factors = step12_results["ml_confidence_factors"]

                # Update our ML confidence factors
                for prediction_type in self.ml_confidence_factors:
                    if prediction_type in ml_factors:
                        self.ml_confidence_factors[prediction_type] = ml_factors[prediction_type]
                        self.logger.info(f"Loaded ML confidence factor for {prediction_type}: {ml_factors[prediction_type]}")
                    else:
                        self.logger.warning(f"Missing ML confidence factor for {prediction_type} in step12 results")

                # Also update the models
                for prediction_type, model_data in self.tactician_models.items():
                    if prediction_type in self.ml_confidence_factors:
                        model_data["ml_confidence_factor"] = self.ml_confidence_factors[prediction_type]

                self.logger.info("✅ Successfully loaded ML confidence factors from step12 results")
                return True

            self.logger.warning("No ml_confidence_factors found in step12 results")
            return False

        except Exception as e:
            self.logger.exception(f"Error loading step12 ML confidence factors: {e}")
            return False

    def auto_refresh_from_step12(self) -> bool:
        """
        Automatically refresh ML confidence factors from step12 results.
        This method is called periodically to check for new step12 results.
        """
        try:
            # Check if step12 results have been updated
            step12_config = self.config.get("step12_confidence_optimization", {})
            auto_refresh = step12_config.get("auto_refresh", True)

            if not auto_refresh:
                return False

            # Try to load latest step12 results
            return self.load_step12_ml_confidence_factors()

        except Exception as e:
            self.logger.exception(f"Error in auto refresh from step12: {e}")
            return False

    def get_prediction_summary(self, tactician_predictions: dict[str, Any]) -> dict[str, Any]:
        """Get summary of Tactician predictions."""
        try:
            summary = {
                "total_predictions": 0,
                "high_precision_predictions": 0,
                "average_confidence": 0.0,
                "average_precision_score": 0.0,
                "prediction_types": {},
                "timeframe": tactician_predictions.get("metadata", {}).get("timeframe", "unknown"),
            }

            total_confidence = 0.0
            total_precision = 0.0
            valid_predictions = 0

            for prediction_type in self.prediction_types:
                if prediction_type in tactician_predictions:
                    pred = tactician_predictions[prediction_type]
                    summary["total_predictions"] += 1

                    confidence = pred.get("confidence", 0.0)
                    precision_score = pred.get("precision_score", 0.0)

                    total_confidence += confidence
                    total_precision += precision_score
                    valid_predictions += 1

                    if precision_score >= self.precision_threshold:
                        summary["high_precision_predictions"] += 1

                    summary["prediction_types"][prediction_type] = {
                        "prediction": pred.get("prediction", 0.0),
                        "confidence": confidence,
                        "precision_score": precision_score,
                    }

            if valid_predictions > 0:
                summary["average_confidence"] = total_confidence / valid_predictions
                summary["average_precision_score"] = total_precision / valid_predictions

            return summary

        except Exception as e:
            self.logger.exception(error(f"❌ Error getting prediction summary: {e}"))
            return {"error": str(e)}
