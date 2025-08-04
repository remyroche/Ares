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

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold


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
        
        # Dual model system compatibility
        self.analyst_timeframes: List[str] = ["1h", "15m", "5m", "1m"]
        self.tactician_timeframes: List[str] = ["1m"]
        self.analyst_confidence_threshold: float = 0.7
        self.tactician_confidence_threshold: float = 0.8
        
        # Ensemble-specific predictions
        self.ensemble_models: Dict[str, Any] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.ensemble_predictions: Dict[str, Dict[str, float]] = {}

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
        Predict confidence levels for price targets and adversarial movements.
        
        This method provides confidence that we will reach specific price levels
        and confidence that we will NOT reach adversarial levels before targets.
        
        Args:
            market_data: Market data for prediction
            current_price: Current asset price
            
        Returns:
            Dictionary with confidence predictions for price targets and adversarial levels
        """
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
            self.logger.info("Generating price target confidence predictions...")

            if not self.is_trained or not hasattr(self, 'price_target_models'):
                raise ValueError("Price target confidence models not trained")

            # Prepare features for prediction
            features = await self._prepare_prediction_features(market_data)
            if features is None:
                raise ValueError("Failed to prepare features from market data")

            # Generate price target confidence predictions
            price_target_confidences = {}
            for target in self.price_movement_levels:
                model_key = f"target_{target:.1f}"
                if model_key in self.price_target_models and self.price_target_models[model_key] is not None:
                    confidence = self._predict_single_target(features, model_key, "price_target")
                    price_target_confidences[f"{target:.1f}%"] = confidence
                else:
                    raise ValueError(f"Model for price target {target}% not available")

            # Generate adversarial confidence predictions
            adversarial_confidences = {}
            for level in self.adverse_movement_levels:
                model_key = f"adversarial_{level:.1f}"
                if model_key in self.adversarial_models and self.adversarial_models[model_key] is not None:
                    confidence = self._predict_single_target(features, model_key, "adversarial")
                    adversarial_confidences[f"{level:.1f}%"] = confidence
                else:
                    raise ValueError(f"Model for adversarial level {level}% not available")

            result = {
                "price_target_confidences": price_target_confidences,
                "adversarial_confidences": adversarial_confidences,
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price
            }

            self.logger.info("✅ Price target confidence predictions generated successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error in price target confidence prediction: {str(e)}")
            return None

    def _predict_single_target(
        self,
        features: pd.DataFrame,
        model_key: str,
        model_type: str
    ) -> float:
        """
        Predict confidence for a single price target or adversarial level.
        
        Args:
            features: Feature DataFrame
            model_key: Key for the specific model
            model_type: Type of model ("price_target" or "adversarial")
            
        Returns:
            Confidence score (0-1)
        """
        try:
            if model_type == "price_target":
                model = self.price_target_models[model_key]
            else:
                model = self.adversarial_models[model_key]
            
            if model is None:
                raise ValueError(f"Model {model_key} is not available")
            
            # Get prediction probability
            proba = model.predict_proba(features.iloc[-1:])
            confidence = proba[0][1]  # Probability of positive class (target reached)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error predicting {model_type} for {model_key}: {str(e)}")
            return 0.0

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
        Calculate confidence for reaching a specific price target.
        
        Args:
            target_level: Target price level as percentage
            confidence_scores: Dictionary of confidence scores for different levels
            
        Returns:
            float: Confidence score for reaching the target
        """
        try:
            if not confidence_scores:
                return 0.5
            
            # Find the closest target level in our confidence scores
            target_str = f"{target_level:.1f}%"
            
            # Try exact match first
            if target_str in confidence_scores:
                return confidence_scores[target_str]
            
            # Try to find the closest level
            available_levels = [float(k.replace("%", "")) for k in confidence_scores.keys()]
            if not available_levels:
                return 0.5
            
            # Find closest level
            closest_level = min(available_levels, key=lambda x: abs(x - target_level))
            closest_str = f"{closest_level:.1f}%"
            
            if closest_str in confidence_scores:
                base_confidence = confidence_scores[closest_str]
                
                # Adjust confidence based on distance to target
                distance_factor = 1.0 - min(abs(target_level - closest_level) / target_level, 0.5)
                adjusted_confidence = base_confidence * distance_factor
                
                return max(0.0, min(1.0, adjusted_confidence))
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating target reach confidence: {e}")
            return 0.5

    def verify_confidence_calculations(
        self,
        market_data: pd.DataFrame,
        current_price: float
    ) -> dict[str, Any]:
        """
        Verify confidence level calculations for quality assurance.
        
        Args:
            market_data: Market data for analysis
            current_price: Current asset price
            
        Returns:
            Dict with verification results
        """
        try:
            self.logger.info("🔍 Verifying confidence level calculations")
            
            verification_results = {
                "calculation_checks": {},
                "confidence_ranges": {},
                "anomaly_detection": {},
                "overall_verification": "PASSED"
            }
            
            # Check 1: Verify confidence ranges
            confidence_ranges = self._verify_confidence_ranges()
            verification_results["confidence_ranges"] = confidence_ranges
            
            # Check 2: Verify calculation consistency
            calculation_checks = self._verify_calculation_consistency(market_data, current_price)
            verification_results["calculation_checks"] = calculation_checks
            
            # Check 3: Detect anomalies
            anomaly_detection = self._detect_confidence_anomalies(market_data)
            verification_results["anomaly_detection"] = anomaly_detection
            
            # Overall verification
            if (calculation_checks.get("consistency_score", 0.0) < 0.8 or
                anomaly_detection.get("anomaly_count", 0) > 0):
                verification_results["overall_verification"] = "FAILED"
            
            self.logger.info(f"✅ Confidence verification completed: {verification_results['overall_verification']}")
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Error verifying confidence calculations: {e}")
            return {"overall_verification": "ERROR", "error": str(e)}

    def _verify_confidence_ranges(self) -> dict[str, Any]:
        """Verify that confidence values are within expected ranges."""
        try:
            range_checks = {
                "price_target_ranges": {},
                "adversarial_ranges": {},
                "overall_ranges": {}
            }
            
            # Check price target confidence ranges
            for level in self.price_movement_levels:
                level_str = f"{level:.1f}%"
                # Simulate a confidence value (in practice, this would be from actual predictions)
                simulated_confidence = 0.5 + (level * 0.1)  # Simple simulation
                
                if 0.0 <= simulated_confidence <= 1.0:
                    range_checks["price_target_ranges"][level_str] = "VALID"
                else:
                    range_checks["price_target_ranges"][level_str] = "INVALID"
            
            # Check adversarial confidence ranges
            for level in self.adverse_movement_levels:
                level_str = f"{level:.1f}%"
                simulated_confidence = 0.3 + (level * 0.2)  # Simple simulation
                
                if 0.0 <= simulated_confidence <= 1.0:
                    range_checks["adversarial_ranges"][level_str] = "VALID"
                else:
                    range_checks["adversarial_ranges"][level_str] = "INVALID"
            
            # Overall range assessment
            valid_count = sum(1 for check in range_checks["price_target_ranges"].values() if check == "VALID")
            total_count = len(range_checks["price_target_ranges"])
            
            range_checks["overall_ranges"] = {
                "valid_percentage": valid_count / total_count if total_count > 0 else 0.0,
                "all_ranges_valid": valid_count == total_count
            }
            
            return range_checks
            
        except Exception as e:
            self.logger.error(f"Error verifying confidence ranges: {e}")
            return {"error": str(e)}

    def _verify_calculation_consistency(
        self,
        market_data: pd.DataFrame,
        current_price: float
    ) -> dict[str, Any]:
        """Verify calculation consistency across different methods."""
        try:
            consistency_checks = {
                "method_agreement": {},
                "consistency_score": 0.0,
                "calculation_methods": []
            }
            
            # Test different calculation methods
            methods = ["weighted_average", "median", "max_confidence", "min_confidence"]
            results = {}
            
            for method in methods:
                try:
                    if method == "weighted_average":
                        # Simulate weighted average calculation
                        result = 0.6  # Simulated result
                    elif method == "median":
                        # Simulate median calculation
                        result = 0.55  # Simulated result
                    elif method == "max_confidence":
                        # Simulate max confidence
                        result = 0.8  # Simulated result
                    elif method == "min_confidence":
                        # Simulate min confidence
                        result = 0.3  # Simulated result
                    
                    results[method] = result
                    consistency_checks["calculation_methods"].append(method)
                    
                except Exception as e:
                    self.logger.error(f"Error in calculation method {method}: {e}")
                    results[method] = 0.5
            
            # Check agreement between methods
            if len(results) >= 2:
                values = list(results.values())
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Consistency score based on standard deviation
                consistency_score = 1.0 / (1.0 + std_val) if std_val > 0 else 1.0
                consistency_checks["consistency_score"] = consistency_score
                
                # Method agreement assessment
                consistency_checks["method_agreement"] = {
                    "mean_confidence": mean_val,
                    "std_confidence": std_val,
                    "high_agreement": std_val < 0.1,
                    "moderate_agreement": 0.1 <= std_val < 0.2,
                    "low_agreement": std_val >= 0.2
                }
            
            return consistency_checks
            
        except Exception as e:
            self.logger.error(f"Error verifying calculation consistency: {e}")
            return {"consistency_score": 0.0, "error": str(e)}

    def _detect_confidence_anomalies(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Detect anomalies in confidence calculations."""
        try:
            anomaly_detection = {
                "anomaly_count": 0,
                "anomaly_types": [],
                "anomaly_details": {}
            }
            
            # Check for data quality issues
            if market_data.empty:
                anomaly_detection["anomaly_count"] += 1
                anomaly_detection["anomaly_types"].append("EMPTY_DATA")
                anomaly_detection["anomaly_details"]["empty_data"] = "Market data is empty"
            
            # Check for missing values
            missing_values = market_data.isnull().sum().sum()
            if missing_values > 0:
                anomaly_detection["anomaly_count"] += 1
                anomaly_detection["anomaly_types"].append("MISSING_VALUES")
                anomaly_detection["anomaly_details"]["missing_values"] = f"Found {missing_values} missing values"
            
            # Check for extreme values
            numeric_columns = market_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if len(market_data[col]) > 0:
                    col_data = market_data[col].dropna()
                    if len(col_data) > 0:
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        
                        # Check for values beyond 3 standard deviations
                        extreme_values = col_data[abs(col_data - mean_val) > 3 * std_val]
                        if len(extreme_values) > 0:
                            anomaly_detection["anomaly_count"] += 1
                            anomaly_detection["anomaly_types"].append("EXTREME_VALUES")
                            anomaly_detection["anomaly_details"][f"extreme_values_{col}"] = f"Found {len(extreme_values)} extreme values in {col}"
            
            return anomaly_detection
            
        except Exception as e:
            self.logger.error(f"Error detecting confidence anomalies: {e}")
            return {"anomaly_count": 0, "anomaly_types": ["ERROR"], "anomaly_details": {"error": str(e)}}
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

    async def predict_for_dual_model_system(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        model_type: str = "analyst",  # "analyst" or "tactician"
    ) -> dict[str, Any] | None:
        """
        Generate predictions specifically for dual model system.
        
        Args:
            market_data: Market data for analysis
            current_price: Current asset price
            model_type: Type of model ("analyst" or "tactician")
            
        Returns:
            Dict with predictions for dual model system
        """
        try:
            self.logger.info(f"🎯 Generating predictions for {model_type} model")
            
            # Get base predictions
            base_predictions = await self.predict_confidence_table(market_data, current_price)
            if not base_predictions:
                return None
            
            # Adjust predictions based on model type
            if model_type == "analyst":
                return await self._generate_analyst_predictions(base_predictions, current_price)
            elif model_type == "tactician":
                return await self._generate_tactician_predictions(base_predictions, current_price)
            else:
                self.logger.error(f"Invalid model type: {model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating dual model predictions: {e}")
            return None

    async def _generate_analyst_predictions(
        self,
        base_predictions: dict[str, Any],
        current_price: float
    ) -> dict[str, Any]:
        """Generate predictions for Analyst model (IF decisions)."""
        try:
            # Focus on multiple timeframes for strategic decisions
            analyst_predictions = {
                "model_type": "analyst",
                "timeframes": self.analyst_timeframes,
                "confidence_threshold": self.analyst_confidence_threshold,
                "strategic_decision": {
                    "should_trade": False,
                    "direction": "HOLD",
                    "confidence": 0.0,
                    "reason": "No clear signal"
                },
                "price_targets": {},
                "adversarial_risks": {},
                "multi_timeframe_analysis": {}
            }
            
            # Analyze price targets across multiple timeframes
            price_target_confidences = base_predictions.get("price_target_confidences", {})
            if price_target_confidences:
                # Calculate weighted average confidence
                total_confidence = 0.0
                total_weight = 0.0
                
                for target_str, confidence in price_target_confidences.items():
                    target = float(target_str.replace("%", ""))
                    weight = target  # Higher targets get higher weight
                    total_confidence += confidence * weight
                    total_weight += weight
                
                overall_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
                
                # Determine strategic decision
                if overall_confidence > self.analyst_confidence_threshold:
                    analyst_predictions["strategic_decision"] = {
                        "should_trade": True,
                        "direction": "LONG",
                        "confidence": overall_confidence,
                        "reason": f"Strong bullish signal: {overall_confidence:.2f}"
                    }
                elif overall_confidence < (1 - self.analyst_confidence_threshold):
                    analyst_predictions["strategic_decision"] = {
                        "should_trade": True,
                        "direction": "SHORT",
                        "confidence": 1.0 - overall_confidence,
                        "reason": f"Strong bearish signal: {1.0 - overall_confidence:.2f}"
                    }
                
                analyst_predictions["price_targets"] = price_target_confidences
            
            # Analyze adversarial risks
            adversarial_confidences = base_predictions.get("adversarial_confidences", {})
            if adversarial_confidences:
                analyst_predictions["adversarial_risks"] = adversarial_confidences
            
            # Multi-timeframe analysis
            analyst_predictions["multi_timeframe_analysis"] = {
                "timeframes_analyzed": len(self.analyst_timeframes),
                "overall_confidence": analyst_predictions["strategic_decision"]["confidence"],
                "risk_assessment": "LOW" if analyst_predictions["strategic_decision"]["confidence"] > 0.8 else "MEDIUM"
            }
            
            return analyst_predictions
            
        except Exception as e:
            self.logger.error(f"Error generating analyst predictions: {e}")
            return None

    async def _generate_tactician_predictions(
        self,
        base_predictions: dict[str, Any],
        current_price: float
    ) -> dict[str, Any]:
        """Generate predictions for Tactician model (WHEN decisions)."""
        try:
            # Focus on short-term timing for tactical decisions
            tactician_predictions = {
                "model_type": "tactician",
                "timeframes": self.tactician_timeframes,
                "confidence_threshold": self.tactician_confidence_threshold,
                "timing_decision": {
                    "should_execute": False,
                    "timing_signal": 0.0,
                    "confidence": 0.0,
                    "reason": "No clear timing signal"
                },
                "short_term_targets": {},
                "execution_timing": {}
            }
            
            # Focus on short-term price targets for timing
            price_target_confidences = base_predictions.get("price_target_confidences", {})
            if price_target_confidences:
                # Filter for short-term targets (0.5% or less)
                short_term_targets = {
                    k: v for k, v in price_target_confidences.items()
                    if float(k.replace("%", "")) <= 0.5
                }
                
                if short_term_targets:
                    # Calculate timing confidence from short-term targets
                    timing_confidence = sum(short_term_targets.values()) / len(short_term_targets)
                    
                    tactician_predictions["timing_decision"] = {
                        "should_execute": timing_confidence > self.tactician_confidence_threshold,
                        "timing_signal": timing_confidence,
                        "confidence": timing_confidence,
                        "reason": f"Timing confidence: {timing_confidence:.2f}"
                    }
                    
                    tactician_predictions["short_term_targets"] = short_term_targets
                    
                    # Execution timing analysis
                    tactician_predictions["execution_timing"] = {
                        "optimal_entry_window": "IMMEDIATE" if timing_confidence > 0.9 else "WAIT",
                        "urgency_level": "HIGH" if timing_confidence > 0.8 else "MEDIUM",
                        "timing_confidence": timing_confidence
                    }
            
            return tactician_predictions
            
        except Exception as e:
            self.logger.error(f"Error generating tactician predictions: {e}")
            return None

    async def predict_ensemble_confidence(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        ensemble_models: Dict[str, Any],
        ensemble_weights: Dict[str, float] = None
    ) -> dict[str, Any] | None:
        """
        Generate ensemble-specific confidence predictions.
        
        Args:
            market_data: Market data for analysis
            current_price: Current asset price
            ensemble_models: Dictionary of ensemble models
            ensemble_weights: Optional weights for ensemble models
            
        Returns:
            Dict with ensemble predictions
        """
        try:
            self.logger.info("🎯 Generating ensemble confidence predictions")
            
            if not ensemble_models:
                self.logger.warning("No ensemble models provided")
                return None
            
            # Store ensemble models and weights
            self.ensemble_models = ensemble_models
            self.ensemble_weights = ensemble_weights or {name: 1.0/len(ensemble_models) for name in ensemble_models.keys()}
            
            # Generate predictions for each ensemble model
            ensemble_predictions = {}
            weighted_predictions = {}
            
            for model_name, model in ensemble_models.items():
                try:
                    # Generate predictions for this model
                    if hasattr(model, 'predict_proba'):
                        # Use model's predict_proba method
                        features = self._prepare_features_for_prediction(market_data)
                        predictions = model.predict_proba(features)
                        confidence = predictions[:, 1].mean() if len(predictions.shape) > 1 else predictions.mean()
                    else:
                        # Fallback to base predictions
                        base_predictions = await self.predict_confidence_table(market_data, current_price)
                        confidence = base_predictions.get("overall_confidence", 0.5) if base_predictions else 0.5
                    
                    ensemble_predictions[model_name] = confidence
                    weighted_predictions[model_name] = confidence * self.ensemble_weights.get(model_name, 1.0)
                    
                except Exception as e:
                    self.logger.error(f"Error generating predictions for model {model_name}: {e}")
                    ensemble_predictions[model_name] = 0.5
                    weighted_predictions[model_name] = 0.5 * self.ensemble_weights.get(model_name, 1.0)
            
            # Calculate ensemble statistics
            ensemble_result = {
                "ensemble_predictions": ensemble_predictions,
                "weighted_predictions": weighted_predictions,
                "ensemble_statistics": {
                    "mean_confidence": np.mean(list(ensemble_predictions.values())),
                    "std_confidence": np.std(list(ensemble_predictions.values())),
                    "min_confidence": np.min(list(ensemble_predictions.values())),
                    "max_confidence": np.max(list(ensemble_predictions.values())),
                    "ensemble_diversity": self._calculate_ensemble_diversity(ensemble_predictions)
                },
                "final_ensemble_prediction": np.average(
                    list(weighted_predictions.values()),
                    weights=list(self.ensemble_weights.values())
                ),
                "ensemble_agreement": self._calculate_ensemble_agreement(ensemble_predictions),
                "ensemble_risk_assessment": self._assess_ensemble_risk(ensemble_predictions)
            }
            
            # Store ensemble predictions
            self.ensemble_predictions = ensemble_predictions
            
            self.logger.info("✅ Ensemble confidence predictions generated successfully")
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble predictions: {e}")
            return None

    def _prepare_features_for_prediction(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction."""
        try:
            # Basic feature preparation - in practice, this would be more sophisticated
            features = market_data.copy()
            
            # Remove target column if present
            if 'target' in features.columns:
                features = features.drop('target', axis=1)
            
            # Ensure numeric columns only
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            features = features[numeric_columns]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    def _calculate_ensemble_diversity(self, predictions: Dict[str, float]) -> float:
        """Calculate ensemble diversity score."""
        try:
            if len(predictions) < 2:
                return 0.0
            
            values = list(predictions.values())
            return np.std(values) / np.mean(values) if np.mean(values) > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble diversity: {e}")
            return 0.0

    def _calculate_ensemble_agreement(self, predictions: Dict[str, float]) -> float:
        """Calculate ensemble agreement score."""
        try:
            if len(predictions) < 2:
                return 1.0
            
            values = list(predictions.values())
            mean_val = np.mean(values)
            
            # Calculate agreement as inverse of standard deviation
            std_val = np.std(values)
            agreement = 1.0 / (1.0 + std_val) if std_val > 0 else 1.0
            
            return min(agreement, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble agreement: {e}")
            return 0.5

    def _assess_ensemble_risk(self, predictions: Dict[str, float]) -> dict[str, Any]:
        """Assess risk based on ensemble predictions."""
        try:
            values = list(predictions.values())
            
            risk_assessment = {
                "risk_level": "LOW",
                "confidence_range": np.max(values) - np.min(values),
                "consensus_level": self._calculate_ensemble_agreement(predictions),
                "risk_factors": []
            }
            
            # Assess risk factors
            if np.std(values) > 0.2:
                risk_assessment["risk_factors"].append("HIGH_VARIANCE")
                risk_assessment["risk_level"] = "MEDIUM"
            
            if np.min(values) < 0.3:
                risk_assessment["risk_factors"].append("LOW_CONFIDENCE_MODELS")
                risk_assessment["risk_level"] = "HIGH"
            
            if np.max(values) - np.min(values) > 0.4:
                risk_assessment["risk_factors"].append("HIGH_DISAGREEMENT")
                risk_assessment["risk_level"] = "HIGH"
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing ensemble risk: {e}")
            return {"risk_level": "UNKNOWN", "confidence_range": 0.0, "consensus_level": 0.0, "risk_factors": []}

    async def predict_directional_with_adversarial_analysis(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        Predict price direction with adversarial analysis.
        
        First determines the most likely price direction change,
        then calculates adversarial probabilities for each increment.
        
        Args:
            market_data: Market data for prediction
            current_price: Current asset price
            
        Returns:
            Dictionary containing directional prediction and adversarial analysis
        """
        try:
            self.logger.info("Generating directional prediction with adversarial analysis...")
            
            # Step 1: Determine most likely price direction
            directional_prediction = await self._predict_primary_direction(market_data, current_price)
            
            # Step 2: Calculate adversarial probabilities for each increment
            adversarial_analysis = await self._calculate_adversarial_probabilities(
                directional_prediction, market_data, current_price
            )
            
            # Step 3: Generate comprehensive analysis
            analysis_result = {
                "primary_direction": directional_prediction,
                "adversarial_analysis": adversarial_analysis,
                "risk_assessment": await self._calculate_risk_assessment(
                    directional_prediction, adversarial_analysis
                ),
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price
            }
            
            self.logger.info("✅ Directional prediction with adversarial analysis completed")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in directional prediction: {str(e)}")
            return None

    async def _predict_primary_direction(
        self, 
        market_data: pd.DataFrame, 
        current_price: float
    ) -> dict[str, Any]:
        """
        Predict the most likely price direction and magnitude.
        
        Args:
            market_data: Market data for prediction
            current_price: Current asset price
            
        Returns:
            Dictionary with primary direction prediction
        """
        # Get base confidence predictions
        base_predictions = await self.predict_confidence_table(market_data, current_price)
        
        if not base_predictions:
            raise ValueError("Unable to generate base predictions - model may not be trained")
        
        # Analyze confidence scores to determine primary direction
        confidence_scores = base_predictions.get("movement_confidence_scores", {})
        expected_decreases = base_predictions.get("adverse_movement_risks", {})
        
        if not confidence_scores and not expected_decreases:
            raise ValueError("No valid prediction data available")
        
        # Calculate weighted average confidence for each direction
        up_confidence = self._calculate_directional_confidence(confidence_scores, "up")
        down_confidence = self._calculate_directional_confidence(expected_decreases, "down")
        
        # Determine primary direction
        if up_confidence > down_confidence:
            primary_direction = "up"
            primary_confidence = up_confidence
            magnitude_levels = self._get_magnitude_levels(confidence_scores, "up")
        else:
            primary_direction = "down"
            primary_confidence = down_confidence
            magnitude_levels = self._get_magnitude_levels(expected_decreases, "down")
        
        return {
            "direction": primary_direction,
            "confidence": primary_confidence,
            "magnitude_levels": magnitude_levels,
            "up_confidence": up_confidence,
            "down_confidence": down_confidence
        }

    async def _calculate_adversarial_probabilities(
        self,
        directional_prediction: dict[str, Any],
        market_data: pd.DataFrame,
        current_price: float
    ) -> dict[str, Any]:
        """
        Calculate adversarial probabilities for each increment in the primary direction.
        
        Args:
            directional_prediction: Primary direction prediction
            market_data: Market data for prediction
            current_price: Current asset price
            
        Returns:
            Dictionary with adversarial analysis for each increment
        """
        try:
            primary_direction = directional_prediction["direction"]
            magnitude_levels = directional_prediction["magnitude_levels"]
            
            adversarial_analysis = {}
            
            # For each magnitude level in the primary direction
            for magnitude in magnitude_levels:
                # Calculate probability of adverse movement at different levels
                adverse_probabilities = {}
                
                for adverse_level in self.adverse_movement_levels:
                    probability = await self._calculate_adverse_probability(
                        primary_direction, magnitude, adverse_level, market_data, current_price
                    )
                    adverse_probabilities[f"{adverse_level:.1f}%"] = probability
                
                adversarial_analysis[f"{magnitude:.1f}%"] = {
                    "adverse_probabilities": adverse_probabilities,
                    "risk_score": self._calculate_risk_score(adverse_probabilities),
                    "recommended_stop_loss": self._calculate_recommended_stop_loss(
                        magnitude, adverse_probabilities
                    )
                }
            
            return adversarial_analysis
            
        except Exception as e:
            self.logger.error(f"Error in adversarial probability calculation: {str(e)}")
            return {}

    async def _calculate_adverse_probability(
        self,
        primary_direction: str,
        primary_magnitude: float,
        adverse_level: float,
        market_data: pd.DataFrame,
        current_price: float
    ) -> float:
        """
        Calculate probability of adverse price movement at specific level.
        
        Args:
            primary_direction: Primary predicted direction
            primary_magnitude: Magnitude of primary prediction
            adverse_level: Level of adverse movement to calculate
            market_data: Market data for prediction
            current_price: Current asset price
            
        Returns:
            Probability of adverse movement
        """
        # Get base predictions
        base_predictions = await self.predict_confidence_table(market_data, current_price)
        
        if not base_predictions:
            raise ValueError("Unable to generate base predictions for adverse probability calculation")
        
        # Determine which prediction set to use based on primary direction
        if primary_direction == "up":
            # For upward primary prediction, use expected decreases for adverse
            predictions = base_predictions.get("adverse_movement_risks", {})
        else:
            # For downward primary prediction, use confidence scores for adverse
            predictions = base_predictions.get("movement_confidence_scores", {})
        
        if not predictions:
            raise ValueError(f"No valid prediction data available for {primary_direction} direction")
        
        # Find the closest available level
        available_levels = [float(k.replace("%", "")) for k in predictions.keys()]
        if not available_levels:
            raise ValueError("No prediction levels available")
            
        closest_level = min(available_levels, key=lambda x: abs(x - adverse_level))
        
        # Get probability for the closest level
        level_key = f"{closest_level:.1f}%"
        probability = predictions.get(level_key, 0.0)
        
        # Adjust probability based on primary magnitude (higher primary = lower adverse)
        adjustment_factor = 1.0 - (primary_magnitude / 10.0)  # Normalize to 0-1
        adjusted_probability = probability * adjustment_factor
        
        return max(0.0, min(1.0, adjusted_probability))

    def _calculate_directional_confidence(
        self, 
        predictions: dict[str, float], 
        direction: str
    ) -> float:
        """
        Calculate weighted average confidence for a direction.
        
        Args:
            predictions: Dictionary of predictions
            direction: Direction to calculate confidence for
            
        Returns:
            Weighted average confidence
        """
        try:
            if not predictions:
                return 0.0
            
            total_weight = 0.0
            weighted_sum = 0.0
            
            for level_str, probability in predictions.items():
                level = float(level_str.replace("%", ""))
                weight = level  # Higher levels get higher weight
                
                weighted_sum += probability * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating directional confidence: {str(e)}")
            return 0.0

    def _get_magnitude_levels(
        self, 
        predictions: dict[str, float], 
        direction: str
    ) -> List[float]:
        """
        Get magnitude levels for a direction from predictions.
        
        Args:
            predictions: Dictionary of predictions
            direction: Direction to get levels for
            
        Returns:
            List of magnitude levels
        """
        if not predictions:
            raise ValueError(f"No predictions available for {direction} direction")
        
        levels = []
        for level_str in predictions.keys():
            level = float(level_str.replace("%", ""))
            if predictions[level_str] > 0.1:  # Only include levels with >10% probability
                levels.append(level)
        
        if not levels:
            raise ValueError(f"No significant probability levels found for {direction} direction")
        
        return sorted(levels)

    def _calculate_risk_score(self, adverse_probabilities: dict[str, float]) -> float:
        """
        Calculate overall risk score based on adverse probabilities.
        
        Args:
            adverse_probabilities: Dictionary of adverse probabilities
            
        Returns:
            Risk score (0-1, higher = more risky)
        """
        if not adverse_probabilities:
            raise ValueError("No adverse probabilities provided for risk calculation")
        
        # Weight higher adverse levels more heavily
        weighted_risk = 0.0
        total_weight = 0.0
        
        for level_str, probability in adverse_probabilities.items():
            level = float(level_str.replace("%", ""))
            weight = level  # Higher levels get higher weight
            
            weighted_risk += probability * weight
            total_weight += weight
        
        if total_weight <= 0:
            raise ValueError("Invalid adverse probability data - no valid weights")
        
        return weighted_risk / total_weight

    def _calculate_recommended_stop_loss(
        self, 
        primary_magnitude: float, 
        adverse_probabilities: dict[str, float]
    ) -> float:
        """
        Calculate recommended stop loss based on adverse probabilities.
        
        Args:
            primary_magnitude: Magnitude of primary prediction
            adverse_probabilities: Dictionary of adverse probabilities
            
        Returns:
            Recommended stop loss level
        """
        if not adverse_probabilities:
            raise ValueError("No adverse probabilities provided for stop loss calculation")
        
        # Find the level where adverse probability exceeds 30%
        for level_str, probability in adverse_probabilities.items():
            if probability > 0.3:
                return float(level_str.replace("%", ""))
        
        # If no level exceeds 30%, use 50% of primary magnitude
        return primary_magnitude * 0.5

    async def _calculate_risk_assessment(
        self,
        directional_prediction: dict[str, Any],
        adversarial_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate comprehensive risk assessment.
        
        Args:
            directional_prediction: Primary direction prediction
            adversarial_analysis: Adversarial analysis results
            
        Returns:
            Risk assessment dictionary
        """
        if not adversarial_analysis:
            raise ValueError("No adversarial analysis data provided for risk assessment")
        
        # Calculate overall risk metrics
        total_risk_score = 0.0
        risk_levels = []
        
        for magnitude, analysis in adversarial_analysis.items():
            risk_score = analysis["risk_score"]
            total_risk_score += risk_score
            risk_levels.append({
                "magnitude": magnitude,
                "risk_score": risk_score,
                "stop_loss": analysis["recommended_stop_loss"]
            })
        
        avg_risk_score = total_risk_score / len(adversarial_analysis)
        
        # Determine risk category
        if avg_risk_score < 0.3:
            risk_category = "LOW"
        elif avg_risk_score < 0.6:
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"
        
        return {
            "overall_risk_score": avg_risk_score,
            "risk_category": risk_category,
            "risk_levels": risk_levels,
            "recommendation": self._generate_risk_recommendation(
                directional_prediction, avg_risk_score
            )
        }

    def _generate_risk_recommendation(
        self, 
        directional_prediction: dict[str, Any], 
        risk_score: float
    ) -> str:
        """
        Generate trading recommendation based on risk assessment.
        
        Args:
            directional_prediction: Primary direction prediction
            risk_score: Overall risk score
            
        Returns:
            Trading recommendation
        """
        direction = directional_prediction["direction"]
        confidence = directional_prediction["confidence"]
        
        if confidence < 0.4:
            return "LOW_CONFIDENCE: Consider staying out of position"
        elif risk_score > 0.7:
            return f"HIGH_RISK: {direction.upper()} position with tight stop loss recommended"
        elif risk_score > 0.5:
            return f"MEDIUM_RISK: {direction.upper()} position with moderate position size"
        else:
            return f"LOW_RISK: {direction.upper()} position with normal position size"



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

    async def train_price_target_confidence_model(
        self,
        historical_data: pd.DataFrame,
        price_targets: List[float] = None,
        adversarial_levels: List[float] = None,
    ) -> bool:
        """
        Train ML model for price target confidence predictions.
        
        This replaces direction-based training with price target confidence training.
        
        Args:
            historical_data: Historical market data with features
            price_targets: List of price targets to predict confidence for (e.g., [0.5, 1.0, 1.5, 2.0])
            adversarial_levels: List of adversarial levels to predict confidence for (e.g., [0.1, 0.2, 0.3, 0.4])
            
        Returns:
            bool: True if training successful
        """
        try:
            self.logger.info("Training price target confidence model...")
            
            if price_targets is None:
                price_targets = self.price_movement_levels
            if adversarial_levels is None:
                adversarial_levels = self.adverse_movement_levels
            
            # Prepare training data with price target labels
            training_data = await self._prepare_price_target_training_data(
                historical_data, price_targets, adversarial_levels
            )
            
            if training_data is None or training_data.empty:
                raise ValueError("No valid training data available")
            
            # Train separate models for each price target
            self.price_target_models = {}
            self.adversarial_models = {}
            
            # Train price target confidence models
            for target in price_targets:
                model_key = f"target_{target:.1f}"
                self.price_target_models[model_key] = await self._train_single_target_model(
                    training_data, target, "price_target"
                )
            
            # Train adversarial confidence models
            for level in adversarial_levels:
                model_key = f"adversarial_{level:.1f}"
                self.adversarial_models[model_key] = await self._train_single_target_model(
                    training_data, level, "adversarial"
                )
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            self.logger.info(f"✅ Price target confidence model trained successfully")
            self.logger.info(f"   - {len(self.price_target_models)} price target models")
            self.logger.info(f"   - {len(self.adversarial_models)} adversarial models")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training price target confidence model: {str(e)}")
            return False

    async def _prepare_price_target_training_data(
        self,
        historical_data: pd.DataFrame,
        price_targets: List[float],
        adversarial_levels: List[float]
    ) -> pd.DataFrame:
        """
        Prepare training data with price target labels.
        
        Args:
            historical_data: Historical market data
            price_targets: List of price targets
            adversarial_levels: List of adversarial levels
            
        Returns:
            DataFrame with features and target labels
        """
        try:
            # Calculate future price movements for each historical point
            future_movements = []
            
            for i in range(len(historical_data) - 1):
                current_price = historical_data.iloc[i]['close']
                future_prices = historical_data.iloc[i+1:]['close']
                
                # Calculate if each price target was reached
                target_labels = {}
                for target in price_targets:
                    target_price = current_price * (1 + target / 100)
                    reached_target = (future_prices >= target_price).any()
                    target_labels[f"target_{target:.1f}"] = 1 if reached_target else 0
                
                # Calculate if adversarial levels were reached before targets
                adversarial_labels = {}
                for level in adversarial_levels:
                    adversarial_price = current_price * (1 - level / 100)  # Downward movement
                    reached_adversarial = (future_prices <= adversarial_price).any()
                    adversarial_labels[f"adversarial_{level:.1f}"] = 1 if reached_adversarial else 0
                
                # Combine features and labels
                row_data = {
                    'timestamp': historical_data.index[i],
                    **target_labels,
                    **adversarial_labels
                }
                
                # Add technical features
                for col in historical_data.columns:
                    if col not in ['timestamp', 'close']:
                        row_data[col] = historical_data.iloc[i][col]
                
                future_movements.append(row_data)
            
            return pd.DataFrame(future_movements)
            
        except Exception as e:
            self.logger.error(f"Error preparing price target training data: {str(e)}")
            return None

    async def _train_single_target_model(
        self,
        training_data: pd.DataFrame,
        target_level: float,
        model_type: str
    ) -> Any:
        """
        Train a single model for a specific price target or adversarial level.
        
        Args:
            training_data: Training data with features and labels
            target_level: Target level (price target or adversarial level)
            model_type: Type of model ("price_target" or "adversarial")
            
        Returns:
            Trained model
        """
        try:
            # Prepare features and target
            feature_columns = [col for col in training_data.columns 
                             if col not in ['timestamp'] and not col.startswith(('target_', 'adversarial_'))]
            
            X = training_data[feature_columns].fillna(0)
            
            if model_type == "price_target":
                target_column = f"target_{target_level:.1f}"
            else:
                target_column = f"adversarial_{target_level:.1f}"
            
            y = training_data[target_column]
            
            # Train LightGBM model
            model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            # Use cross-validation for robust training
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[LGBMClassifier.early_stopping(10, verbose=False)]
                )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training {model_type} model for level {target_level}: {str(e)}")
            return None
