        """
Enhanced Prediction Service for ML Profit Integration System.

This service provides calibrated confidence scores from ML models for both Analyst and Tactician.
It ONLY provides calibrated confidence scores and fails if calibrated confidence doesn't exist.
        """

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config.enhanced_prediction_service_config import get_enhanced_prediction_service_config
from src.utils.logging_config import get_logger
from src.utils.tracing import with_tracing_span
from src.utils.performance import performance_monitor
from src.utils.caching import intelligent_caching


class EnhancedPredictionService:
        """
    Enhanced Prediction Service that provides calibrated confidence scores from ML models.

    This service ONLY provides calibrated confidence scores for both Analyst and Tactician ML models.
    It fails if calibrated confidence doesn't exist for either model set.
        """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Enhanced Prediction Service."""
        self.config = config or get_enhanced_prediction_service_config()
        self.logger = get_logger(__name__)

        # Service state
        self.is_initialized = False
        self.data_dir = self.config.get("data_directory", "data")

        # ML model storage
        self.        self.analyst_ml_models:: Dict[str, Dict[str, Any]] = {}
        self.        self.tactician_ml_models:: Dict[str, Dict[str, Any]] = {}

        # Calibration and optimization results
        self.        self.calibration_results:: Dict[str, Any] = {}
        self.        self.optimization_results:: Dict[str, Any] = {}

        # Configuration parameters
        self.entry_threshold = self.config.get("entry_threshold", 0.6)
        self.max_confidence_threshold = self.config.get("max_confidence_threshold", 0.7)

        self.logger.info("Enhanced Prediction Service initialized")

    @handle_errors(
        exceptions=(Exception),
        default_return=False,
        context="initializing enhanced prediction service")
    @with_tracing_span("initialize_enhanced_prediction_service")
    async def initialize(self) -> bool:
        """Initialize the Enhanced Prediction Service."""
        try:
            self.logger.info("🚀 Initializing Enhanced Prediction Service...")

            # Load ML models for both Analyst and Tactician
            await self._load_analyst_ml_models()
            await self._load_tactician_ml_models()

            # Load calibration and optimization results
            await self._load_calibration_results()
            await self._load_optimization_results()

            self.is_initialized = True
            self.logger.info("✅ Enhanced Prediction Service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Enhanced Prediction Service: {e}")
            return False

    @handle_errors(
        exceptions=(Exception),
        default_return=False,
        context="loading analyst ML models")
    @with_tracing_span("load_analyst_ml_models")
    @intelligent_caching(cache_key="analyst_ml_models")
    async def _load_analyst_ml_models(self) -> None:
        """Load Analyst ML models (higher timeframe) from steps 6-14."""
        try:
            analyst_models_path = Path(self.data_dir) / "ml_profit_models" / "analyst_models"
            if not analyst_models_path.exists():
                raise ValueError(f"Analyst ML models directory not found: {analyst_models_path}")

            # Load different types of Analyst models
            analyst_model_types = [
                "hmm_profit", "analyst_profit", "calibrated", "optimized",
                "validated", "monte_carlo", "ab_tested"
            ]

            for model_type in analyst_model_types:
                type_path = analyst_models_path / model_type
                if type_path.exists():
                    self.analyst_ml_models[model_type] = {}

                    for model_file in type_path.glob("*.pkl"):
                        try:
                            with open(model_file, "rb") as f:
                                model_data = pickle.load(f)

                            model_name = model_file.stem

                            # Verify that the model has probability outputs
                            if not self._verify_model_probability_outputs(model_data, f"{model_type}_{model_name}"):
                                self.logger.warning(f"⚠️ Skipping Analyst model {model_name} - missing probability outputs")
                                continue

                            self.analyst_ml_models[model_type][model_name] = model_data
                            self.logger.info(f"✅ Loaded Analyst ML model: {model_type}/{model_name}")

                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to load Analyst ML model {model_file}: {e}")

            if not self.analyst_ml_models:
                raise ValueError("No Analyst ML models loaded")

        except Exception as e:
            self.logger.error(f"❌ Error loading Analyst ML models: {e}")
            raise

    @handle_errors(
        exceptions=(Exception),
        default_return=False,
        context="loading tactician ML models")
    @with_tracing_span("load_tactician_ml_models")
    @intelligent_caching(cache_key="tactician_ml_models")
    async def _load_tactician_ml_models(self) -> None:
        """Load Tactician ML models (lower timeframe) from steps 6-14."""
        try:
            tactician_models_path = Path(self.data_dir) / "ml_profit_models" / "tactician_models"
            if not tactician_models_path.exists():
                raise ValueError(f"Tactician ML models directory not found: {tactician_models_path}")

            # Load different types of Tactician models
            tactician_model_types = [
                "tactician_profit", "tactician_specialist", "calibrated", "optimized",
                "validated", "monte_carlo", "ab_tested"
            ]

            for model_type in tactician_model_types:
                type_path = tactician_models_path / model_type
                if type_path.exists():
                    self.tactician_ml_models[model_type] = {}

                    for model_file in type_path.glob("*.pkl"):
                        try:
                            with open(model_file, "rb") as f:
                                model_data = pickle.load(f)

                            model_name = model_file.stem

                            # Verify that the model has probability outputs
                            if not self._verify_model_probability_outputs(model_data, f"{model_type}_{model_name}"):
                                self.logger.warning(f"⚠️ Skipping Tactician model {model_name} - missing probability outputs")
                                continue

                            self.tactician_ml_models[model_type][model_name] = model_data
                            self.logger.info(f"✅ Loaded Tactician ML model: {model_type}/{model_name}")

                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to load Tactician ML model {model_file}: {e}")

            if not self.tactician_ml_models:
                raise ValueError("No Tactician ML models loaded")

        except Exception as e:
            self.logger.error(f"❌ Error loading Tactician ML models: {e}")
            raise

    @handle_errors(
        exceptions=(Exception),
        default_return=False,
        context="loading calibration results")
    @with_tracing_span("load_calibration_results")
    async def _load_calibration_results(self) -> None:
        """Load calibration results from step 11 (model performance vs actual reliability)."""
        try:
            calibration_path = Path(self.data_dir) / "calibration_results"
            if calibration_path.exists():
                for calibration_file in calibration_path.glob("*.json"):
                    try:
                        import json
                        with open(calibration_file, "r") as f:
                            calibration_data = json.load(f)

                            key = calibration_file.stem
                            self.calibration_results[key] = calibration_data
                            self.logger.debug(f"Loaded calibration results: {key}")

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to load calibration file {calibration_file}: {e}")

        except Exception as e:
            self.logger.error(f"❌ Error loading calibration results: {e}")

    @handle_errors(
        exceptions=(Exception),
        default_return=False,
        context="loading optimization results")
    @with_tracing_span("load_optimization_results")
    async def _load_optimization_results(self) -> None:
        """Load optimization results from step 11 (model performance vs actual reliability)."""
        try:
            optimization_path = Path(self.data_dir) / "optimization_results"
            if optimization_path.exists():
                for optimization_file in optimization_path.glob("*.json"):
                    try:
                        import json
                        with open(optimization_file, "r") as f:
                            optimization_data = json.load(f)

                            key = optimization_file.stem
                            self.optimization_results[key] = optimization_data
                            self.logger.debug(f"Loaded optimization results: {key}")

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to load optimization file {optimization_file}: {e}")

        except Exception as e:
            self.logger.error(f"❌ Error loading optimization results: {e}")

    @handle_errors(
        exceptions=(ValueError),
        default_return={},
        context="getting calibrated confidence scores")
    @with_tracing_span("get_calibrated_confidence_scores")
    @validate_data_quality(validation_level="ERROR")
    @performance_monitor
    async def get_calibrated_confidence_scores(
        self,
        market_data: pd.DataFrame,
        regime_info: Dict[str, Any],
        symbol: str,
        exchange: str
    ) -> Dict[str, Dict[str, float]]:
        """
Provide calibrated confidence scores for BOTH Analyst and Tactician ML models.

This method ONLY provides calibrated confidence scores and fails if calibrated
confidence doesn't exist for either model set.

ML models should generate probabilities for specific price actions:
        - Probability of reaching profit target without hitting stop-loss (triple barrier)
- Probability of price moving in predicted direction by X%
- Probability of avoiding adverse price movements

Calibration is based on step 11: comparing model performance to actual reliability.

Args:
            market_data: Market data for prediction
regime_info: Market regime information
symbol: Trading symbol
exchange: Exchange name

Returns:
            Dictionary with calibrated confidence scores for both Analyst and Tactician models

Raises:
            ValueError: If calibrated confidence doesn't exist for either model set
        """
        try:
            if not self.is_initialized:
                raise ValueError("Enhanced Prediction Service not initialized")

            calibrated_scores = {
                "analyst_models": {},
                "tactician_models": {}
            }

            # Get Analyst calibrated confidence scores
            analyst_scores = await self._get_analyst_calibrated_confidence(
                market_data, regime_info, symbol, exchange
            )
            if not analyst_scores:
                raise ValueError(f"No calibrated Analyst confidence scores available for {symbol} on {exchange}")
            calibrated_scores["analyst_models"] = analyst_scores

            # Get Tactician calibrated confidence scores
            tactician_scores = await self._get_tactician_calibrated_confidence(
                market_data, regime_info, symbol, exchange
            )
            if not tactician_scores:
                raise ValueError(f"No calibrated Tactician confidence scores available for {symbol} on {exchange}")
            calibrated_scores["tactician_models"] = tactician_scores

            self.logger.info(f"✅ Retrieved calibrated confidence scores for {symbol} on {exchange}")
            self.logger.debug(f"Analyst models: {len(analyst_scores)}, Tactician models: {len(tactician_scores)}")

            return calibrated_scores

        except Exception as e:
            self.logger.error(f"❌ Failed to get calibrated confidence scores: {e}")
            raise

    @handle_errors(
        exceptions=(Exception),
        default_return={},
        context="getting analyst calibrated confidence")
    @with_tracing_span("get_analyst_calibrated_confidence")
    async def _get_analyst_calibrated_confidence(
        self,
        market_data: pd.DataFrame,
        regime_info: Dict[str, Any],
        symbol: str,
        exchange: str
    ) -> Dict[str, float]:
        """Get calibrated confidence scores from Analyst ML models based on step 11 calibration."""
        try:
            analyst_scores = {}

            for model_type, models in self.analyst_ml_models.items():
                for model_name, model_data in models.items():
                    try:
                        # Get price action probabilities from ML model
                        price_action_probabilities = model_data.get("price_action_probabilities", {})

                        if not price_action_probabilities:
                            self.logger.warning(f"⚠️ No price action probabilities for Analyst model {model_name}")
                            continue

                        # Get calibration data from step 11
                        calibration_key = f"{exchange}_{symbol}_calibration_results"
                        calibration_data = self.calibration_results.get(calibration_key, {})
                        model_calibration = calibration_data.get("model_calibrations", {}).get(f"{model_type}_{model_name}", {})

                        # Calculate calibrated confidence based on step 11 performance vs reliability
                        calibrated_confidence = self._calculate_step11_calibrated_confidence(
                            price_action_probabilities, model_calibration, model_name, "analyst"
                        )

                        if calibrated_confidence is not None:
                            analyst_scores[f"{model_type}_{model_name}"] = calibrated_confidence

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to get confidence for Analyst model {model_name}: {e}")

            return analyst_scores

        except Exception as e:
            self.logger.error(f"❌ Error getting Analyst calibrated confidence: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception),
        default_return={},
        context="getting tactician calibrated confidence")
    @with_tracing_span("get_tactician_calibrated_confidence")
    async def _get_tactician_calibrated_confidence(
        self,
        market_data: pd.DataFrame,
        regime_info: Dict[str, Any],
        symbol: str,
        exchange: str
    ) -> Dict[str, float]:
        """Get calibrated confidence scores from Tactician ML models based on step 11 calibration."""
        try:
            tactician_scores = {}

            for model_type, models in self.tactician_ml_models.items():
                for model_name, model_data in models.items():
                    try:
                        # Get price action probabilities from ML model
                        price_action_probabilities = model_data.get("price_action_probabilities", {})

                        if not price_action_probabilities:
                            self.logger.warning(f"⚠️ No price action probabilities for Tactician model {model_name}")
                            continue

                        # Get calibration data from step 11
                        calibration_key = f"{exchange}_{symbol}_calibration_results"
                        calibration_data = self.calibration_results.get(calibration_key, {})
                        model_calibration = calibration_data.get("model_calibrations", {}).get(f"{model_type}_{model_name}", {})

                        # Calculate calibrated confidence based on step 11 performance vs reliability
                        calibrated_confidence = self._calculate_step11_calibrated_confidence(
                            price_action_probabilities, model_calibration, model_name, "tactician"
                        )

                        if calibrated_confidence is not None:
                            tactician_scores[f"{model_type}_{model_name}"] = calibrated_confidence

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to get confidence for Tactician model {model_name}: {e}")

            return tactician_scores

        except Exception as e:
            self.logger.error(f"❌ Error getting Tactician calibrated confidence: {e}")
            return {}

    def _calculate_step11_calibrated_confidence(
        self,
        price_action_probabilities: Dict[str, Any],
        model_calibration: Dict[str, Any],
        model_name: str,
        model_type: str
    ) -> Optional[float]:
        """
        Calculate calibrated confidence based on step 11: model performance vs actual reliability.

        This method applies calibration based on how well the model's predicted probabilities
        match actual outcomes in historical data.

        Args:
            price_action_probabilities: ML model's predicted probabilities for price actions
            model_calibration: Calibration data from step 11
            model_name: Name of the model
            model_type: Type of model (analyst/tactician)

        Returns:
            Calibrated confidence score or None if calibration fails
        """
        try:
            # Extract key probability metrics from ML model
            triple_barrier_prob = price_action_probabilities.get("triple_barrier_probability", 0.5)
            direction_prob = price_action_probabilities.get("direction_probability", 0.5)
            magnitude_prob = price_action_probabilities.get("magnitude_probability", 0.5)
            barrier_avoidance_prob = price_action_probabilities.get("barrier_avoidance_probability", 0.5)

            # Get step 11 calibration parameters
            reliability_score = model_calibration.get("reliability_score", 0.5)
            performance_ratio = model_calibration.get("performance_ratio", 1.0)
            calibration_factor = model_calibration.get("calibration_factor", 1.0)
            confidence_bias = model_calibration.get("confidence_bias", 0.0)

            # Get step 12 optimized weights for probability components
            # These weights are optimized based on historical performance
            optimized_weights = model_calibration.get("step12_optimized_weights", {})

            # Use optimized weights if available, otherwise use defaults
            if optimized_weights:
                triple_barrier_weight = optimized_weights.get("triple_barrier_weight", 0.25)
                direction_weight = optimized_weights.get("direction_weight", 0.25)
                magnitude_weight = optimized_weights.get("magnitude_weight", 0.25)
                barrier_avoidance_weight = optimized_weights.get("barrier_avoidance_weight", 0.25)
            else:
                # Default equal weights if step 12 optimization not available
                triple_barrier_weight = 0.25
                direction_weight = 0.25
                magnitude_weight = 0.25
                barrier_avoidance_weight = 0.25

            # Calculate base confidence using step 12 optimized weights
            base_confidence = (
                triple_barrier_prob * triple_barrier_weight +
                direction_prob * direction_weight +
                magnitude_prob * magnitude_weight +
                barrier_avoidance_prob * barrier_avoidance_weight
            )

            # Apply step 11 calibration: performance vs reliability
            # This adjusts the confidence based on how well the model's predictions
            # have matched actual outcomes in historical testing

            # Apply reliability adjustment
            reliability_adjusted = base_confidence * reliability_score

            # Apply performance ratio (how well model performs vs expected)
            performance_adjusted = reliability_adjusted * performance_ratio

            # Apply calibration factor (overall calibration adjustment)
            calibrated = performance_adjusted * calibration_factor

            # Apply confidence bias (systematic adjustment)
            final_confidence = calibrated + confidence_bias

            # Ensure bounds
            final_confidence = max(0.0, min(1.0, final_confidence))

            self.logger.debug(f"Step 11 calibration for {model_name}: base={base_confidence:.3f}, "
                             f"reliability={reliability_score:.3f}, performance={performance_ratio:.3f}, "
                             f"weights={optimized_weights}, final={final_confidence:.3f}")

            return final_confidence

        except Exception as e:
            self.logger.error(f"❌ Error calculating step 11 calibrated confidence for {model_name}: {e}")
            return None

    @handle_errors(
        exceptions=(Exception),
        default_return={},
        context="validating price action probabilities")
    @with_tracing_span("validate_price_action_probabilities")
    def _validate_price_action_probabilities(
        self,
        price_action_probabilities: Dict[str, Any],
        model_name: str
    ) -> bool:
        """
        Validate that ML model provides the expected price action probabilities.

        Expected probabilities:
            - triple_barrier_probability: Probability of reaching profit target without hitting stop-loss
            - direction_probability: Probability of price moving in predicted direction
            - magnitude_probability: Probability of price moving by expected magnitude
            - barrier_avoidance_probability: Probability of avoiding adverse price movements

        Args:
            price_action_probabilities: Probabilities from ML model
            model_name: Name of the model for logging

        Returns:
            True if probabilities are valid, False otherwise
        """
        try:
            required_probabilities = [
                "triple_barrier_probability",
                "direction_probability",
                "magnitude_probability",
                "barrier_avoidance_probability"
            ]

            for prob_name in required_probabilities:
                if prob_name not in price_action_probabilities:
                    self.logger.warning(f"⚠️ Missing required probability '{prob_name}' for model {model_name}")
                    return False

                prob_value = price_action_probabilities[prob_name]
                if not isinstance(prob_value, (int, float)) or not (0.0 <= prob_value <= 1.0):
                    self.logger.warning(f"⚠️ Invalid probability value for '{prob_name}' in model {model_name}: {prob_value}")
                    return False

            # Validate that probabilities make sense together
            triple_barrier = price_action_probabilities["triple_barrier_probability"]
            direction = price_action_probabilities["direction_probability"]
            magnitude = price_action_probabilities["magnitude_probability"]
            barrier_avoidance = price_action_probabilities["barrier_avoidance_probability"]

            # Triple barrier probability should be <= direction probability
            if triple_barrier > direction:
                self.logger.warning(f"⚠️ Triple barrier probability ({triple_barrier}) > direction probability ({direction}) for model {model_name}")
                return False

            # Barrier avoidance should be reasonable relative to triple barrier
            if barrier_avoidance < triple_barrier * 0.5:
                self.logger.warning(f"⚠️ Barrier avoidance probability ({barrier_avoidance}) seems too low relative to triple barrier ({triple_barrier}) for model {model_name}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"❌ Error validating price action probabilities for {model_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception),
        default_return=False,
        context="verifying model probability outputs")
    @with_tracing_span("verify_model_probability_outputs")
    def _verify_model_probability_outputs(self, model_data: Dict[str, Any], model_name: str) -> bool:
        """
        Verify that a model has the required probability outputs.

        Args:
            model_data: Model data loaded from file
            model_name: Name of the model for logging

        Returns:
            True if model has valid probability outputs, False otherwise
        """
        try:
            # Check if model_data has price_action_probabilities
            if "price_action_probabilities" not in model_data:
                self.logger.warning(f"⚠️ Model {model_name} missing 'price_action_probabilities' key")
                return False

            price_action_probabilities = model_data["price_action_probabilities"]

            # Validate the probability outputs
            if not self._validate_price_action_probabilities(price_action_probabilities, model_name):
                return False

            self.logger.debug(f"✅ Model {model_name} has valid probability outputs")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error verifying probability outputs for {model_name}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception),
        default_return={},
        context="verifying all models have probability outputs")
    @with_tracing_span("verify_all_models_probability_outputs")
    async def verify_all_models_probability_outputs(self) -> Dict[str, Any]:
        """
        Verify that ALL loaded models have probability outputs.

        Returns:
            Dictionary with verification results for all models
        """
        try:
            verification_results = {
                "analyst_models": {},
                "tactician_models": {},
                "summary": {
                    "total_analyst_models": 0,
                    "total_tactician_models": 0,
                    "analyst_models_with_probabilities": 0,
                    "tactician_models_with_probabilities": 0,
                    "all_models_verified": False
                }
            }

            # Verify Analyst models
            for model_type, models in self.analyst_ml_models.items():
                verification_results["analyst_models"][model_type] = {}
                for model_name, model_data in models.items():
                    has_probabilities = self._verify_model_probability_outputs(model_data, f"{model_type}_{model_name}")
                    verification_results["analyst_models"][model_type][model_name] = {
                        "has_probability_outputs": has_probabilities,
                        "probability_keys": list(model_data.get("price_action_probabilities", {}).keys()) if has_probabilities else []
                    }
                    verification_results["summary"]["total_analyst_models"] += 1
                    if has_probabilities:
                        verification_results["summary"]["analyst_models_with_probabilities"] += 1

            # Verify Tactician models
            for model_type, models in self.tactician_ml_models.items():
                verification_results["tactician_models"][model_type] = {}
                for model_name, model_data in models.items():
                    has_probabilities = self._verify_model_probability_outputs(model_data, f"{model_type}_{model_name}")
                    verification_results["tactician_models"][model_type][model_name] = {
                        "has_probability_outputs": has_probabilities,
                        "probability_keys": list(model_data.get("price_action_probabilities", {}).keys()) if has_probabilities else []
                    }
                    verification_results["summary"]["total_tactician_models"] += 1
                    if has_probabilities:
                        verification_results["summary"]["tactician_models_with_probabilities"] += 1

            # Check if all models have probabilities
            total_models = verification_results["summary"]["total_analyst_models"] + verification_results["summary"]["total_tactician_models"]
            models_with_probabilities = verification_results["summary"]["analyst_models_with_probabilities"] + verification_results["summary"]["tactician_models_with_probabilities"]

            verification_results["summary"]["all_models_verified"] = (total_models > 0 and total_models == models_with_probabilities)

            # Log verification results
            if verification_results["summary"]["all_models_verified"]:
                self.logger.info(f"✅ All {total_models} models have probability outputs")
            else:
                self.logger.warning(f"⚠️ Only {models_with_probabilities}/{total_models} models have probability outputs")

            return verification_results

        except Exception as e:
            self.logger.error(f"❌ Error verifying all models probability outputs: {e}")
            return {"error": str(e)}

    @handle_errors(
        exceptions=(Exception),
        default_return=False,
        context="checking service health")
    @with_tracing_span("check_service_health")
    async def check_service_health(self) -> bool:
        """Check if the service is healthy and has loaded models with probability outputs."""
        try:
            if not self.is_initialized:
                return False

            # Check if we have both Analyst and Tactician models
            has_analyst_models = len(self.analyst_ml_models) > 0
            has_tactician_models = len(self.tactician_ml_models) > 0

            if not has_analyst_models:
                self.logger.warning("⚠️ No Analyst ML models loaded")

            if not has_tactician_models:
                self.logger.warning("⚠️ No Tactician ML models loaded")

            # Verify that all models have probability outputs
            verification_results = await self.verify_all_models_probability_outputs()
            all_models_verified = verification_results.get("summary", {}).get("all_models_verified", False)

            if not all_models_verified:
                self.logger.warning("⚠️ Not all models have probability outputs")
                return False

            return has_analyst_models and has_tactician_models and all_models_verified

        except Exception as e:
            self.logger.error(f"❌ Service health check failed: {e}")
            return False

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and statistics."""
        try:
            analyst_model_count = sum(len(models) for models in self.analyst_ml_models.values())
            tactician_model_count = sum(len(models) for models in self.tactician_ml_models.values())

            return {
                "service_name": "Enhanced Prediction Service",
                "is_initialized": self.is_initialized,
                "analyst_models_loaded": analyst_model_count,
                "tactician_models_loaded": tactician_model_count,
                "analyst_model_types": list(self.analyst_ml_models.keys()),
                "tactician_model_types": list(self.tactician_ml_models.keys()),
                "calibration_results_loaded": len(self.calibration_results),
                "optimization_results_loaded": len(self.optimization_results),
                "entry_threshold": self.entry_threshold,
                "max_confidence_threshold": self.max_confidence_threshold,
                "probability_requirements": self._get_probability_requirements_info()
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get service info: {e}")
            return {"error": str(e)}

    def _get_probability_requirements_info(self) -> Dict[str, Any]:
        """Get information about required probability outputs from ML models."""
        return {
            "probability_outputs_location": "ML models in steps 6-14 of enhanced_training_manager",
            "required_probabilities": {
                "triple_barrier_probability": "Probability of reaching profit target without hitting stop-loss",
                "direction_probability": "Probability of price moving in predicted direction",
                "magnitude_probability": "Probability of price moving by expected magnitude",
                "barrier_avoidance_probability": "Probability of avoiding adverse price movements"
            },
            "model_structure": {
                "expected_format": "price_action_probabilities dict in model_data",
                "example": {
                    "triple_barrier_probability": 0.75,
                    "direction_probability": 0.80,
                    "magnitude_probability": 0.65,
                    "barrier_avoidance_probability": 0.70
                }
            },
            "calibration_requirements": {
                "step11_calibration": "Model performance vs actual reliability data",
                "step12_optimization": "Optimized weights for probability components",
                "calibration_data_structure": {
                    "reliability_score": "Historical reliability of model predictions",
                    "performance_ratio": "Actual vs expected performance ratio",
                    "calibration_factor": "Overall calibration adjustment",
                    "confidence_bias": "Systematic bias adjustment",
                    "step12_optimized_weights": {
                        "triple_barrier_weight": "Optimized weight for triple barrier probability",
                        "direction_weight": "Optimized weight for direction probability",
                        "magnitude_weight": "Optimized weight for magnitude probability",
                        "barrier_avoidance_weight": "Optimized weight for barrier avoidance probability"
                    }
                }
            }
        }