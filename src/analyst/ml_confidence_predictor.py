import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold

# Import enhanced order manager for tactician order management
from src.tactician.enhanced_order_manager import (
    OrderSide,
)
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
    Integrates with enhanced training manager to use properly trained models.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ML Confidence Predictor with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MLConfidencePredictor")

        # Model state - Initialize properly
        self.model: Any | None = None
        self.is_trained: bool = False
        self.last_training_time: datetime | None = None
        self.model_performance: dict[str, float] = {}

        # Initialize price target and adversarial models
        self.price_target_models: dict[str, Any] = {}
        self.adversarial_models: dict[str, Any] = {}

        # Configuration
        from src.config_optuna import get_parameter_value

        self.predictor_config: dict[str, Any] = self.config.get(
            "ml_confidence_predictor",
            {},
        )
        self.model_path: str = get_parameter_value(
            "ml_confidence_predictor_parameters.model_path",
            "models/confidence_predictor.joblib",
        )

        # Confidence score levels for price movements (direction-neutral)
        self.price_movement_levels: list[float] = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
        ]

        # Adverse movement levels (opposite direction risk)
        self.adversarial_movement_levels: list[float] = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]

        # Directional confidence analysis (0-2% range for high leverage trading)
        self.directional_confidence_levels: list[float] = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
        ]

        # Dual model system compatibility
        self.analyst_timeframes: list[str] = ["30m", "15m", "5m"]
        self.tactician_timeframes: list[str] = ["1m"]
        from src.config_optuna import get_parameter_value

        self.analyst_confidence_threshold: float = get_parameter_value(
            "confidence_thresholds.analyst_confidence_threshold",
            0.7,
        )
        self.tactician_confidence_threshold: float = get_parameter_value(
            "confidence_thresholds.tactician_confidence_threshold",
            0.8,
        )

        # Ensemble-specific predictions
        self.ensemble_models: dict[str, Any] = {}
        self.ensemble_weights: dict[str, float] = {}
        self.ensemble_predictions: dict[str, dict[str, float]] = {}

        # Enhanced training manager integration
        self.enhanced_training_manager: Any = None
        self.trained_models: dict[str, Any] = {}
        self.calibrated_models: dict[str, Any] = {}
        self.regime_models: dict[str, Any] = {}
        self.multi_timeframe_models: dict[str, Any] = {}

        # Meta-labeling system integration
        self.meta_labeling_system: Any = None
        self.analyst_labels: list[str] = [
            "STRONG_TREND_CONTINUATION",
            "EXHAUSTION_REVERSAL",
            "RANGE_MEAN_REVERSION",
            "BREAKOUT_FAILURE",
            "BREAKOUT_SUCCESS",
            "NO_SETUP",
            "VOLATILITY_COMPRESSION",
            "ABSORPTION_AT_LEVEL",
            "FLAG_FORMATION",
            "TRENDING_RANGE",
            "MOVING_AVERAGE_BOUNCE",
            "HEAD_AND_SHOULDERS",
            "DOUBLE_TOP_BOTTOM",
            "CLIMACTIC_REVERSAL",
            "VOLATILITY_EXPANSION",
            "MOMENTUM_IGNITION",
            "GRADUAL_MOMENTUM_FADE",
            "TRIANGLE_FORMATION",
            "RECTANGLE_FORMATION",
            "LIQUIDITY_GRAB",
        ]
        self.tactician_labels: list[str] = [
            "LOWEST_PRICE_NEXT_1m",
            "HIGHEST_PRICE_NEXT_1m",
            "LIMIT_ORDER_RETURN",
            "VWAP_REVERSION_ENTRY",
            "MARKET_ORDER_NOW",
            "CHASE_MICRO_BREAKOUT",
            "MAX_ADVERSE_EXCURSION_RETURN",
            "ORDERBOOK_IMBALANCE_FLIP",
            "AGGRESSIVE_TAKER_SPIKE",
            "ABORT_ENTRY_SIGNAL",
        ]

        # Enhanced order manager for tactician
        self.enhanced_order_manager = None
        self.order_manager_config = config.get("enhanced_order_manager", {})

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
            self.logger.info("Generating price target confidence predictions...")

            # Check model availability and prepare for prediction
            if not await self._prepare_for_prediction():
                return self._generate_fallback_predictions(current_price)

            # Prepare features for prediction
            features = await self._prepare_prediction_features(market_data)
            if features is None or features.empty:
                self.logger.warning(
                    "Could not prepare features for prediction, using fallback",
                )
                return self._generate_fallback_predictions(current_price)

            # Generate predictions
            price_target_confidences = await self._generate_price_target_predictions(features)
            adversarial_confidences = await self._generate_adversarial_predictions(features)
            directional_analysis = self._generate_directional_confidence_analysis(
                price_target_confidences,
                adversarial_confidences,
                current_price,
            )
            ensemble_predictions = await self._generate_ensemble_predictions_if_available(features)

            # Build and return result
            result = self._build_prediction_result(
                price_target_confidences,
                adversarial_confidences,
                directional_analysis,
                ensemble_predictions,
                current_price,
            )

            self.logger.info(
                f"✅ Generated predictions with {len(price_target_confidences)} price targets and {len(adversarial_confidences)} adversarial levels",
            )
            return result

        except Exception as e:
            self.logger.error(f"Error in price target confidence prediction: {str(e)}")
            return self._generate_fallback_predictions(current_price)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="prediction preparation",
    )
    async def _prepare_for_prediction(self) -> bool:
        """
        Prepare the predictor for making predictions.

        Returns:
            bool: True if ready for prediction, False otherwise
        """
        # Check if enhanced training manager is available and has models
        if not self.is_enhanced_training_available():
            self.logger.warning(
                "Enhanced training manager not available or no models loaded - using fallback predictions",
            )
            return False

        # Try to refresh models from enhanced training manager if not trained
        if not self.is_trained:
            self.logger.info(
                "Attempting to refresh models from enhanced training manager...",
            )
            await self.refresh_models_from_enhanced_training()

        # Check if we have trained models from enhanced training manager
        if not self._has_trained_models():
            self.logger.warning(
                "No trained models available, using fallback predictions",
            )
            return False

        return True

    def _has_trained_models(self) -> bool:
        """
        Check if trained models are available.

        Returns:
            bool: True if trained models are available
        """
        return (
            self.is_trained
            and (
                self.price_target_models
                or self.adversarial_models
                or self.ensemble_models
                or self.regime_models
                or self.multi_timeframe_models
            )
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="price target predictions generation",
    )
    async def _generate_price_target_predictions(
        self,
        features: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Generate price target confidence predictions.

        Args:
            features: Prepared features for prediction

        Returns:
            dict: Price target confidence predictions
        """
        price_target_confidences = {}
        
        for target in self.price_movement_levels:
            model_key = f"price_target_{target:.1f}"
            confidence = self._get_prediction_for_target(
                features, model_key, "price_target", target
            )
            price_target_confidences[f"{target:.1f}%"] = confidence

        return price_target_confidences

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="adversarial predictions generation",
    )
    async def _generate_adversarial_predictions(
        self,
        features: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Generate adversarial confidence predictions.

        Args:
            features: Prepared features for prediction

        Returns:
            dict: Adversarial confidence predictions
        """
        adversarial_confidences = {}
        
        for level in self.adversarial_movement_levels:
            model_key = f"adversarial_{level:.1f}"
            confidence = self._get_prediction_for_target(
                features, model_key, "adversarial", level
            )
            adversarial_confidences[f"{level:.1f}%"] = confidence

        return adversarial_confidences

    def _get_prediction_for_target(
        self,
        features: pd.DataFrame,
        model_key: str,
        model_type: str,
        target_level: float,
    ) -> float:
        """
        Get prediction for a specific target level.

        Args:
            features: Prepared features
            model_key: Model key
            model_type: Type of model
            target_level: Target level

        Returns:
            float: Prediction confidence
        """
        if model_type == "price_target":
            models = self.price_target_models
            fallback_func = self._get_fallback_confidence
        else:  # adversarial
            models = self.adversarial_models
            fallback_func = self._get_fallback_decrease_probability

        if model_key in models and models[model_key] is not None:
            return self._predict_single_target(features, model_key, model_type)
        else:
            return fallback_func(target_level)

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="ensemble predictions generation",
    )
    async def _generate_ensemble_predictions_if_available(
        self,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate ensemble predictions if available.

        Args:
            features: Prepared features

        Returns:
            dict: Ensemble predictions
        """
        if self.ensemble_models:
            return await self._generate_ensemble_predictions(features)
        return {}

    def _build_prediction_result(
        self,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
        directional_analysis: dict[str, Any],
        ensemble_predictions: dict[str, Any],
        current_price: float,
    ) -> dict[str, Any]:
        """
        Build the final prediction result.

        Args:
            price_target_confidences: Price target confidence predictions
            adversarial_confidences: Adversarial confidence predictions
            directional_analysis: Directional confidence analysis
            ensemble_predictions: Ensemble predictions
            current_price: Current price

        Returns:
            dict: Complete prediction result
        """
        return {
            "price_target_confidences": price_target_confidences,
            "adversarial_confidences": adversarial_confidences,
            "directional_analysis": directional_analysis,
            "ensemble_predictions": ensemble_predictions,
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "model_status": "enhanced_training" if self.is_trained else "fallback",
            "model_info": self.get_enhanced_training_model_info(),
            "availability_status": self.get_model_availability_status(),
        }

    async def predict_with_meta_labeling(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        model_type: str = "analyst",  # "analyst" or "tactician"
    ) -> dict[str, Any] | None:
        """
        Generate predictions with meta-labeling integration.

        Args:
            market_data: Market data for prediction
            current_price: Current price
            model_type: Type of model ("analyst" or "tactician")

        Returns:
            Dictionary containing predictions with meta-labels
        """
        try:
            if not self.meta_labeling_system:
                self.logger.warning("Meta-labeling system not available")
                return None

            # Generate base confidence predictions
            base_predictions = await self.predict_confidence_table(
                market_data,
                current_price,
            )
            if not base_predictions:
                return None

            # Generate meta-labels
            if model_type == "analyst":
                meta_labels = await self._generate_analyst_meta_labels(market_data)
            else:
                meta_labels = await self._generate_tactician_meta_labels(market_data)

            # Combine predictions with meta-labels
            combined_predictions = {
                **base_predictions,
                "meta_labels": meta_labels,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(
                f"Generated predictions with {len(meta_labels)} meta-labels for {model_type}",
            )
            return combined_predictions

        except Exception as e:
            self.logger.error(f"Error generating predictions with meta-labeling: {e}")
            return None

    async def _generate_ensemble_predictions(
        self,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate predictions using ensemble models from enhanced training manager."""
        try:
            if not self.ensemble_models:
                return {}

            ensemble_predictions = {}

            for ensemble_name, ensemble_model in self.ensemble_models.items():
                try:
                    # Use the ensemble model to make predictions
                    if hasattr(ensemble_model, "predict"):
                        prediction = ensemble_model.predict(features)
                        ensemble_predictions[ensemble_name] = prediction
                    elif hasattr(ensemble_model, "predict_proba"):
                        prediction = ensemble_model.predict_proba(features)
                        ensemble_predictions[ensemble_name] = prediction
                    else:
                        self.logger.warning(
                            f"Ensemble model {ensemble_name} has no predict method",
                        )

                except Exception as e:
                    self.logger.error(
                        f"Error predicting with ensemble {ensemble_name}: {e}",
                    )
                    continue

            return ensemble_predictions

        except Exception as e:
            self.logger.error(f"Error generating ensemble predictions: {e}")
            return {}

    async def _generate_analyst_meta_labels(
        self,
        market_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate analyst meta-labels for setup identification."""
        try:
            if not self.meta_labeling_system:
                return {}

            # Create volume data (assuming volume column exists)
            volume_data = (
                market_data[["volume"]]
                if "volume" in market_data.columns
                else pd.DataFrame({"volume": [1.0] * len(market_data)})
            )

            # Generate analyst labels
            analyst_labels = await self.meta_labeling_system._generate_analyst_labels(
                market_data,
                volume_data,
                None,
            )

            return analyst_labels

        except Exception as e:
            self.logger.error(f"Error generating analyst meta-labels: {e}")
            return {}

    async def refresh_models_from_enhanced_training(self) -> bool:
        """Refresh models from enhanced training manager."""
        try:
            if not self.enhanced_training_manager:
                self.logger.warning("Enhanced training manager not available")
                return False

            self.logger.info("Refreshing models from enhanced training manager...")

            # Clear existing models
            self.price_target_models.clear()
            self.adversarial_models.clear()
            self.ensemble_models.clear()
            self.calibrated_models.clear()
            self.regime_models.clear()
            self.multi_timeframe_models.clear()

            # Reload models from enhanced training manager
            await self._load_trained_models_from_enhanced_training()

            # Update training status
            if self.is_trained:
                self.logger.info(
                    "✅ Models refreshed successfully from enhanced training manager",
                )
                return True
            self.logger.warning(
                "No models found when refreshing from enhanced training manager",
            )
            return False

        except Exception as e:
            self.logger.error(
                f"Error refreshing models from enhanced training manager: {e}",
            )
            return False

    def get_enhanced_training_model_info(self) -> dict[str, Any]:
        """Get information about models from enhanced training manager."""
        try:
            if not self.enhanced_training_manager:
                return {"error": "Enhanced training manager not available"}

            # Get training results from enhanced training manager
            try:
                training_results = (
                    self.enhanced_training_manager.get_enhanced_training_results()
                )
            except AttributeError:
                training_results = {}

            # Get training status
            try:
                training_status = (
                    self.enhanced_training_manager.get_enhanced_training_status()
                )
            except AttributeError:
                training_status = {"status": "unknown"}

            # Get analyst models info
            analyst_models_info = {}
            if hasattr(self.enhanced_training_manager, "analyst_models"):
                analyst_models = self.enhanced_training_manager.analyst_models
                analyst_models_info = {
                    "count": len(analyst_models),
                    "models": list(analyst_models.keys()),
                }

            # Get tactician models info
            tactician_models_info = {}
            if hasattr(self.enhanced_training_manager, "tactician_models"):
                tactician_models = self.enhanced_training_manager.tactician_models
                tactician_models_info = {
                    "count": len(tactician_models),
                    "models": list(tactician_models.keys()),
                }

            model_info = {
                "price_target_models": len(self.price_target_models),
                "adversarial_models": len(self.adversarial_models),
                "ensemble_models": len(self.ensemble_models),
                "regime_models": len(self.regime_models),
                "multi_timeframe_models": len(self.multi_timeframe_models),
                "calibrated_models": len(self.calibrated_models),
                "is_trained": self.is_trained,
                "last_training_time": self.last_training_time.isoformat()
                if self.last_training_time
                else None,
                "training_status": training_status,
                "available_training_results": list(training_results.keys())
                if training_results
                else [],
                "analyst_models": analyst_models_info,
                "tactician_models": tactician_models_info,
            }

            return model_info

        except Exception as e:
            self.logger.error(f"Error getting enhanced training model info: {e}")
            return {"error": str(e)}

    async def _generate_tactician_meta_labels(
        self,
        market_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate tactician meta-labels for entry optimization."""
        try:
            if not self.meta_labeling_system:
                return {}

            # Create volume data (assuming volume column exists)
            volume_data = (
                market_data[["volume"]]
                if "volume" in market_data.columns
                else pd.DataFrame({"volume": [1.0] * len(market_data)})
            )

            # Generate tactician labels
            tactician_labels = (
                await self.meta_labeling_system._generate_tactician_labels(
                    market_data,
                    volume_data,
                    None,
                )
            )

            return tactician_labels

        except Exception as e:
            self.logger.error(f"Error generating tactician meta-labels: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training integration initialization",
    )
    async def _initialize_enhanced_training_integration(self) -> None:
        """Initialize integration with enhanced training manager."""
        try:
            # Import enhanced training manager
            from src.training.enhanced_training_manager import EnhancedTrainingManager

            # Initialize enhanced training manager
            self.enhanced_training_manager = EnhancedTrainingManager(self.config)
            await self.enhanced_training_manager.initialize()

            # Load trained models from enhanced training manager
            await self._load_trained_models_from_enhanced_training()

            # Initialize model training capabilities
            await self._initialize_model_training_capabilities()

            self.logger.info("Enhanced training integration initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing enhanced training integration: {e}")
            # Continue without enhanced training manager if not available
            self.enhanced_training_manager = None

    async def _initialize_model_training_capabilities(self) -> None:
        """Initialize model training capabilities."""
        try:
            # Set up training configuration
            self.training_config = self.config.get(
                "model_training",
                {
                    "enable_continuous_training": True,
                    "enable_adaptive_training": True,
                    "enable_incremental_training": True,
                    "training_interval_hours": 24,
                    "min_samples_for_retraining": 1000,
                    "performance_degradation_threshold": 0.1,
                    "enable_model_calibration": True,
                    "enable_ensemble_training": True,
                    "enable_regime_specific_training": True,
                    "enable_multi_timeframe_training": True,
                    "enable_dual_model_training": True,
                    "enable_confidence_calibration": True,
                },
            )

            # Initialize training state
            self.last_training_time = None
            self.training_history = []
            self.model_performance_history = []

            self.logger.info("✅ Model training capabilities initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing model training capabilities: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature engineering integration initialization",
    )
    async def _initialize_feature_engineering_integration(self) -> None:
        """Initialize feature engineering integration."""
        try:
            # Import feature engineering components
            from src.analyst.advanced_feature_engineering import (
                AdvancedFeatureEngineering,
            )
            from src.analyst.feature_engineering_orchestrator import (
                FeatureEngineeringOrchestrator,
            )
            from src.analyst.multi_timeframe_feature_engineering import (
                MultiTimeframeFeatureEngineering,
            )

            # Get configuration for feature engineering
            feature_config = self.config.get(
                "feature_engineering",
                {
                    "enable_advanced_features": True,
                    "enable_multi_timeframe_features": True,
                    "enable_autoencoder_features": True,
                    "enable_legacy_features": True,
                    "feature_cache_duration": 300,  # 5 minutes
                    "enable_feature_selection": True,
                    "max_features": 500,
                    "multi_timeframe_feature_engineering": {
                        "enable_mtf_features": True,
                        "enable_timeframe_adaptation": True,
                    },
                },
            )

            # Initialize feature engineering components
            self.advanced_feature_engineering = AdvancedFeatureEngineering(
                feature_config,
            )
            await self.advanced_feature_engineering.initialize()

            self.multi_timeframe_feature_engineering = MultiTimeframeFeatureEngineering(
                feature_config,
            )

            self.feature_engineering_orchestrator = FeatureEngineeringOrchestrator(
                feature_config,
            )

            self.logger.info(
                "✅ Feature engineering integration initialized successfully",
            )

        except Exception as e:
            self.logger.error(
                f"Error initializing feature engineering integration: {e}",
            )
            self.advanced_feature_engineering = None
            self.multi_timeframe_feature_engineering = None
            self.feature_engineering_orchestrator = None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="meta labeling system initialization",
    )
    async def _initialize_meta_labeling_system(self) -> None:
        """Initialize meta-labeling system integration."""
        try:
            # Import meta-labeling system
            from src.analyst.meta_labeling_system import MetaLabelingSystem

            # Get configuration for meta-labeling
            meta_config = self.config.get(
                "meta_labeling",
                {
                    "enable_analyst_labels": True,
                    "enable_tactician_labels": True,
                    "pattern_detection": {
                        "volatility_threshold": 0.02,
                        "momentum_threshold": 0.01,
                        "volume_threshold": 1.5,
                    },
                    "entry_prediction": {
                        "prediction_horizon": 5,
                        "max_adverse_excursion": 0.02,
                    },
                },
            )

            # Initialize meta-labeling system
            self.meta_labeling_system = MetaLabelingSystem(meta_config)
            await self.meta_labeling_system.initialize()

            self.logger.info("✅ Meta-labeling system initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing meta-labeling system: {e}")
            # Continue without meta-labeling system if not available
            self.meta_labeling_system = None

    async def predict_with_meta_labeling(
        self,
        market_data: pd.DataFrame,
        timeframe: str = "1m",
    ) -> dict[str, Any]:
        """
        Generate predictions combined with meta-labels.

        Args:
            market_data: OHLCV market data
            timeframe: Timeframe for analysis

        Returns:
            Dict containing predictions and meta-labels
        """
        try:
            if not self.meta_labeling_system:
                self.logger.warning("Meta-labeling system not available")
                return await self.predict_confidence_table(
                    market_data,
                    market_data["close"].iloc[-1],
                )

            # Generate meta-labels
            volume_data = (
                market_data[["volume"]]
                if "volume" in market_data.columns
                else pd.DataFrame({"volume": [1000] * len(market_data)})
            )

            if timeframe in ["30m", "15m", "5m"]:
                # Analyst labels for multi-timeframe
                meta_labels = await self.meta_labeling_system.generate_analyst_labels(
                    market_data,
                    volume_data,
                    timeframe,
                )
            else:
                # Tactician labels for 1m timeframe
                meta_labels = await self.meta_labeling_system.generate_tactician_labels(
                    market_data,
                    volume_data,
                    None,
                    timeframe,
                )

            # Generate base predictions
            base_predictions = await self.predict_confidence_table(
                market_data,
                market_data["close"].iloc[-1],
            )

            # Combine predictions with meta-labels
            combined_result = {
                **base_predictions,
                "meta_labels": meta_labels,
                "prediction_source": "ml_confidence_with_meta_labeling",
                "timeframe": timeframe,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            self.logger.info(
                f"Generated predictions with {len(meta_labels)} meta-labels for {timeframe}",
            )
            return combined_result

        except Exception as e:
            self.logger.error(f"Error generating predictions with meta-labeling: {e}")
            return await self.predict_confidence_table(
                market_data,
                market_data["close"].iloc[-1],
            )

    async def _generate_analyst_meta_labels(
        self,
        market_data: pd.DataFrame,
        timeframes: list[str] = None,
    ) -> dict[str, Any]:
        """
        Generate analyst meta-labels for multiple timeframes.

        Args:
            market_data: OHLCV market data
            timeframes: List of timeframes to analyze

        Returns:
            Dict containing analyst meta-labels
        """
        try:
            if not self.meta_labeling_system:
                return {}

            if timeframes is None:
                timeframes = ["30m", "15m", "5m"]

            analyst_labels = {}
            volume_data = (
                market_data[["volume"]]
                if "volume" in market_data.columns
                else pd.DataFrame({"volume": [1000] * len(market_data)})
            )

            for timeframe in timeframes:
                labels = await self.meta_labeling_system.generate_analyst_labels(
                    market_data,
                    volume_data,
                    timeframe,
                )
                analyst_labels[timeframe] = labels

            return analyst_labels

        except Exception as e:
            self.logger.error(f"Error generating analyst meta-labels: {e}")
            return {}

    async def _generate_tactician_meta_labels(
        self,
        market_data: pd.DataFrame,
        timeframe: str = "1m",
    ) -> dict[str, Any]:
        """
        Generate tactician meta-labels for entry optimization.

        Args:
            market_data: OHLCV market data
            timeframe: Timeframe for analysis (typically 1m)

        Returns:
            Dict containing tactician meta-labels
        """
        try:
            if not self.meta_labeling_system:
                return {}

            volume_data = (
                market_data[["volume"]]
                if "volume" in market_data.columns
                else pd.DataFrame({"volume": [1000] * len(market_data)})
            )

            tactician_labels = (
                await self.meta_labeling_system.generate_tactician_labels(
                    market_data,
                    volume_data,
                    None,
                    timeframe,
                )
            )

            return tactician_labels

        except Exception as e:
            self.logger.error(f"Error generating tactician meta-labels: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trained models loading from enhanced training",
    )
    async def _load_trained_models_from_enhanced_training(self) -> None:
        """Load trained models from enhanced training manager."""
        try:
            if not self.enhanced_training_manager:
                self.logger.warning("Enhanced training manager not available")
                return

            # Get trained models from enhanced training manager
            try:
                training_results = self.enhanced_training_manager.get_enhanced_training_results()
            except AttributeError:
                training_results = {}

            # Load different types of models
            self._load_analyst_models()
            self._load_tactician_models()
            self._load_ensemble_models()
            self._load_calibrated_models()
            self._load_regime_models()
            self._load_multi_timeframe_models()

            # Log summary of loaded models
            self._log_model_loading_summary()

        except Exception as e:
            self.logger.error(f"Error loading trained models: {e}")
            raise

    def _load_analyst_models(self) -> None:
        """Load analyst models (multi-timeframe models) from enhanced training manager."""
        if not hasattr(self.enhanced_training_manager, "analyst_models"):
            return
        
        analyst_models = self.enhanced_training_manager.analyst_models
        if not analyst_models:
            self.logger.warning("No analyst models available in enhanced training manager")
            return
        
        for timeframe in self.analyst_timeframes:
            for model_name in ["tcn", "tabnet", "transformer"]:
                model_key = f"{timeframe}_{model_name}"
                if model_key in analyst_models:
                    # Create price target models for different confidence levels
                    for level in self.price_movement_levels:
                        target_key = f"price_target_{level:.1f}"
                        self.price_target_models[target_key] = analyst_models[model_key]
                    self.logger.info(f"Loaded analyst model: {model_key}")
                else:
                    self.logger.debug(f"Analyst model not found: {model_key}")


    def _load_tactician_models(self) -> None:
        """Load tactician models (1m timeframe models) from enhanced training manager."""
        if not hasattr(self.enhanced_training_manager, "tactician_models"):
            return
        
        tactician_models = self.enhanced_training_manager.tactician_models
        if not tactician_models:
            self.logger.warning("No tactician models available in enhanced training manager")
            return
        
        for model_name in ["lstm", "gru", "transformer"]:
            model_key = f"1m_{model_name}"
            if model_key in tactician_models:
                # Create adversarial models for different risk levels
                for level in self.adversarial_movement_levels:
                    adversarial_key = f"adversarial_{level:.1f}"
                    self.adversarial_models[adversarial_key] = tactician_models[model_key]
                self.logger.info(f"Loaded tactician model: {model_key}")
            else:
                self.logger.debug(f"Tactician model not found: {model_key}")


    def _load_ensemble_models(self) -> None:
        """Load ensemble models from enhanced training manager."""
        if not (hasattr(self.enhanced_training_manager, "ensemble_creator") and 
                self.enhanced_training_manager.ensemble_creator):
            return
        
        try:
            ensemble_models = self.enhanced_training_manager.ensemble_creator.get_ensembles()
            if ensemble_models:
                self.ensemble_models = ensemble_models
                self.logger.info(f"Loaded {len(ensemble_models)} ensemble models")
            else:
                self.logger.debug("No ensemble models available")
        except Exception as e:
            self.logger.warning(f"Could not load ensemble models: {e}")


    def _load_calibrated_models(self) -> None:
        """Load calibrated models from enhanced training manager."""
        if not hasattr(self.enhanced_training_manager, "calibration_systems"):
            return
        
        calibration_systems = self.enhanced_training_manager.calibration_systems
        if calibration_systems:
            self.calibrated_models = calibration_systems
            self.logger.info(f"Loaded {len(calibration_systems)} calibrated models")
        else:
            self.logger.debug("No calibrated models available")


    def _load_regime_models(self) -> None:
        """Load regime-specific models from enhanced training manager."""
        if not (hasattr(self.enhanced_training_manager, "regime_training_manager") and 
                self.enhanced_training_manager.regime_training_manager):
            return
        
        try:
            regime_models = self.enhanced_training_manager.regime_training_manager.get_regime_models()
            if regime_models:
                self.regime_models = regime_models
                self.logger.info(f"Loaded {len(regime_models)} regime models")
            else:
                self.logger.debug("No regime models available")
        except Exception as e:
            self.logger.warning(f"Could not load regime models: {e}")


    def _load_multi_timeframe_models(self) -> None:
        """Load multi-timeframe models from enhanced training manager."""
        if not (hasattr(self.enhanced_training_manager, "multi_timeframe_manager") and 
                self.enhanced_training_manager.multi_timeframe_manager):
            return
        
        try:
            multi_timeframe_models = self.enhanced_training_manager.multi_timeframe_manager.get_timeframe_models()
            if multi_timeframe_models:
                self.multi_timeframe_models = multi_timeframe_models
                self.logger.info(f"Loaded {len(multi_timeframe_models)} multi-timeframe models")
            else:
                self.logger.debug("No multi-timeframe models available")
        except Exception as e:
            self.logger.warning(f"Could not load multi-timeframe models: {e}")


    def _log_model_loading_summary(self) -> None:
        """Log a summary of all loaded models."""
        self.logger.info("Model loading summary:")
        self.logger.info(f"  - Price target models: {len(self.price_target_models)}")
        self.logger.info(f"  - Adversarial models: {len(self.adversarial_models)}")
        self.logger.info(f"  - Ensemble models: {len(self.ensemble_models)}")
        self.logger.info(f"  - Calibrated models: {len(self.calibrated_models)}")
        self.logger.info(f"  - Regime models: {len(self.regime_models)}")
        self.logger.info(f"  - Multi-timeframe models: {len(self.multi_timeframe_models)}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="predictor configuration loading",
    )
    async def _load_predictor_configuration(self) -> None:
        """Load predictor configuration."""
        # Set default predictor parameters
        self.predictor_config.setdefault(
            "model_path",
            "models/confidence_predictor.joblib",
        )
        self.predictor_config.setdefault("min_samples_for_training", 500)
        self.predictor_config.setdefault(
            "confidence_threshold",
            get_parameter_value("confidence_thresholds.base_entry_threshold", 0.6),
        )
        self.predictor_config.setdefault("max_prediction_horizon", 1)  # hours
        self.predictor_config.setdefault("enhanced_training_integration", True)

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
                "min_samples_for_training",
            ]

            for param in required_params:
                if param not in self.predictor_config:
                    self.logger.error(f"Missing required parameter: {param}")
                    return False

            # Validate parameter values
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
            self.logger.info("Generating price target confidence predictions...")

            # Check model availability and prepare for prediction
            if not await self._prepare_for_prediction():
                return self._generate_fallback_predictions(current_price)

            # Prepare features for prediction
            features = await self._prepare_prediction_features(market_data)
            if features is None or features.empty:
                self.logger.warning(
                    "Could not prepare features for prediction, using fallback",
                )
                return self._generate_fallback_predictions(current_price)

            # Generate predictions
            price_target_confidences = await self._generate_price_target_predictions(features)
            adversarial_confidences = await self._generate_adversarial_predictions(features)
            directional_analysis = self._generate_directional_confidence_analysis(
                price_target_confidences,
                adversarial_confidences,
                current_price,
            )
            ensemble_predictions = await self._generate_ensemble_predictions_if_available(features)

            # Build and return result
            result = self._build_prediction_result(
                price_target_confidences,
                adversarial_confidences,
                directional_analysis,
                ensemble_predictions,
                current_price,
            )

            self.logger.info(
                f"✅ Generated predictions with {len(price_target_confidences)} price targets and {len(adversarial_confidences)} adversarial levels",
            )
            return result

        except Exception as e:
            self.logger.error(f"Error in price target confidence prediction: {str(e)}")
            return self._generate_fallback_predictions(current_price)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="prediction preparation",
    )
    async def _prepare_for_prediction(self) -> bool:
        """
        Prepare the predictor for making predictions.

        Returns:
            bool: True if ready for prediction, False otherwise
        """
        # Check if enhanced training manager is available and has models
        if not self.is_enhanced_training_available():
            self.logger.warning(
                "Enhanced training manager not available or no models loaded - using fallback predictions",
            )
            return False

        # Try to refresh models from enhanced training manager if not trained
        if not self.is_trained:
            self.logger.info(
                "Attempting to refresh models from enhanced training manager...",
            )
            await self.refresh_models_from_enhanced_training()

        # Check if we have trained models from enhanced training manager
        if not self._has_trained_models():
            self.logger.warning(
                "No trained models available, using fallback predictions",
            )
            return False

        return True

    def _has_trained_models(self) -> bool:
        """
        Check if trained models are available.

        Returns:
            bool: True if trained models are available
        """
        return (
            self.is_trained
            and (
                self.price_target_models
                or self.adversarial_models
                or self.ensemble_models
                or self.regime_models
                or self.multi_timeframe_models
            )
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="price target predictions generation",
    )
    async def _generate_price_target_predictions(
        self,
        features: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Generate price target confidence predictions.

        Args:
            features: Prepared features for prediction

        Returns:
            dict: Price target confidence predictions
        """
        price_target_confidences = {}
        
        for target in self.price_movement_levels:
            model_key = f"price_target_{target:.1f}"
            confidence = self._get_prediction_for_target(
                features, model_key, "price_target", target
            )
            price_target_confidences[f"{target:.1f}%"] = confidence

        return price_target_confidences

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="adversarial predictions generation",
    )
    async def _generate_adversarial_predictions(
        self,
        features: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Generate adversarial confidence predictions.

        Args:
            features: Prepared features for prediction

        Returns:
            dict: Adversarial confidence predictions
        """
        adversarial_confidences = {}
        
        for level in self.adversarial_movement_levels:
            model_key = f"adversarial_{level:.1f}"
            confidence = self._get_prediction_for_target(
                features, model_key, "adversarial", level
            )
            adversarial_confidences[f"{level:.1f}%"] = confidence

        return adversarial_confidences

    def _get_prediction_for_target(
        self,
        features: pd.DataFrame,
        model_key: str,
        model_type: str,
        target_level: float,
    ) -> float:
        """
        Get prediction for a specific target level.

        Args:
            features: Prepared features
            model_key: Model key
            model_type: Type of model
            target_level: Target level

        Returns:
            float: Prediction confidence
        """
        if model_type == "price_target":
            models = self.price_target_models
            fallback_func = self._get_fallback_confidence
        else:  # adversarial
            models = self.adversarial_models
            fallback_func = self._get_fallback_decrease_probability

        if model_key in models and models[model_key] is not None:
            return self._predict_single_target(features, model_key, model_type)
        else:
            return fallback_func(target_level)

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="ensemble predictions generation",
    )
    async def _generate_ensemble_predictions_if_available(
        self,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate ensemble predictions if available.

        Args:
            features: Prepared features

        Returns:
            dict: Ensemble predictions
        """
        if self.ensemble_models:
            return await self._generate_ensemble_predictions(features)
        return {}

    def _build_prediction_result(
        self,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
        directional_analysis: dict[str, Any],
        ensemble_predictions: dict[str, Any],
        current_price: float,
    ) -> dict[str, Any]:
        """
        Build the final prediction result.

        Args:
            price_target_confidences: Price target confidence predictions
            adversarial_confidences: Adversarial confidence predictions
            directional_analysis: Directional confidence analysis
            ensemble_predictions: Ensemble predictions
            current_price: Current price

        Returns:
            dict: Complete prediction result
        """
        return {
            "price_target_confidences": price_target_confidences,
            "adversarial_confidences": adversarial_confidences,
            "directional_analysis": directional_analysis,
            "ensemble_predictions": ensemble_predictions,
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "model_status": "enhanced_training" if self.is_trained else "fallback",
            "model_info": self.get_enhanced_training_model_info(),
            "availability_status": self.get_model_availability_status(),
        }

    async def predict_ensemble_confidence(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        ensemble_models: dict[str, Any],
        ensemble_weights: dict[str, float] = None,
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
            self.ensemble_weights = ensemble_weights or {
                name: 1.0 / len(ensemble_models) for name in ensemble_models
            }

            # Generate predictions for each ensemble model
            ensemble_predictions = {}
            weighted_predictions = {}

            for model_name, model in ensemble_models.items():
                try:
                    # Generate predictions for this model
                    if hasattr(model, "predict_proba"):
                        # Use model's predict_proba method
                        features = self._prepare_features_for_prediction(market_data)
                        predictions = model.predict_proba(features)
                        confidence = (
                            predictions[:, 1].mean()
                            if len(predictions.shape) > 1
                            else predictions.mean()
                        )
                    else:
                        # Fallback to base predictions
                        base_predictions = await self.predict_confidence_table(
                            market_data,
                            current_price,
                        )
                        confidence = (
                            base_predictions.get("overall_confidence", 0.5)
                            if base_predictions
                            else 0.5
                        )

                    ensemble_predictions[model_name] = confidence
                    weighted_predictions[model_name] = (
                        confidence * self.ensemble_weights.get(model_name, 1.0)
                    )

                except Exception as e:
                    self.logger.error(
                        f"Error generating predictions for model {model_name}: {e}",
                    )
                    ensemble_predictions[model_name] = 0.5
                    weighted_predictions[model_name] = 0.5 * self.ensemble_weights.get(
                        model_name,
                        1.0,
                    )

            # Calculate ensemble statistics
            ensemble_result = {
                "ensemble_predictions": ensemble_predictions,
                "weighted_predictions": weighted_predictions,
                "ensemble_statistics": {
                    "mean_confidence": np.mean(list(ensemble_predictions.values())),
                    "std_confidence": np.std(list(ensemble_predictions.values())),
                    "min_confidence": np.min(list(ensemble_predictions.values())),
                    "max_confidence": np.max(list(ensemble_predictions.values())),
                    "ensemble_diversity": self._calculate_ensemble_diversity(
                        ensemble_predictions,
                    ),
                },
                "final_ensemble_prediction": np.average(
                    list(weighted_predictions.values()),
                    weights=list(self.ensemble_weights.values()),
                ),
                "ensemble_agreement": self._calculate_ensemble_agreement(
                    ensemble_predictions,
                ),
                "ensemble_risk_assessment": self._assess_ensemble_risk(
                    ensemble_predictions,
                ),
            }

            # Store ensemble predictions
            self.ensemble_predictions = ensemble_predictions

            self.logger.info(
                "✅ Ensemble confidence predictions generated successfully",
            )
            return ensemble_result

        except Exception as e:
            self.logger.error(f"Error generating ensemble predictions: {e}")
            return None

    def _prepare_features_for_prediction(
        self,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare features for model prediction."""
        try:
            # Basic feature preparation - in practice, this would be more sophisticated
            features = market_data.copy()

            # Remove target column if present
            if "target" in features.columns:
                features = features.drop("target", axis=1)

            # Ensure numeric columns only
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            features = features[numeric_columns]

            return features

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    def _calculate_ensemble_diversity(self, predictions: dict[str, float]) -> float:
        """Calculate ensemble diversity score."""
        try:
            if len(predictions) < 2:
                return 0.0

            values = list(predictions.values())
            return np.std(values) / np.mean(values) if np.mean(values) > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating ensemble diversity: {e}")
            return 0.0

    def _calculate_ensemble_agreement(self, predictions: dict[str, float]) -> float:
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

    def _assess_ensemble_risk(self, predictions: dict[str, float]) -> dict[str, Any]:
        """Assess risk based on ensemble predictions."""
        try:
            values = list(predictions.values())

            risk_assessment = {
                "risk_level": "LOW",
                "confidence_range": np.max(values) - np.min(values),
                "consensus_level": self._calculate_ensemble_agreement(predictions),
                "risk_factors": [],
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
            return {
                "risk_level": "UNKNOWN",
                "confidence_range": 0.0,
                "consensus_level": 0.0,
                "risk_factors": [],
            }

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
            self.logger.info(
                "Generating directional prediction with adversarial analysis...",
            )

            # Step 1: Determine most likely price direction
            directional_prediction = await self._predict_primary_direction(
                market_data,
                current_price,
            )

            # Step 2: Calculate adversarial probabilities for each increment
            adversarial_analysis = await self._calculate_adversarial_probabilities(
                directional_prediction,
                market_data,
                current_price,
            )

            # Step 3: Generate comprehensive analysis
            analysis_result = {
                "primary_direction": directional_prediction,
                "adversarial_analysis": adversarial_analysis,
                "risk_assessment": await self._calculate_risk_assessment(
                    directional_prediction,
                    adversarial_analysis,
                ),
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price,
            }

            self.logger.info(
                "✅ Directional prediction with adversarial analysis completed",
            )
            return analysis_result

        except Exception as e:
            self.logger.error(f"Error in directional prediction: {str(e)}")
            return None

    async def _predict_primary_direction(
        self,
        market_data: pd.DataFrame,
        current_price: float,
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
        base_predictions = await self.predict_confidence_table(
            market_data,
            current_price,
        )

        if not base_predictions:
            raise ValueError(
                "Unable to generate base predictions - model may not be trained",
            )

        # Analyze confidence scores to determine primary direction
        price_target_confidences = base_predictions.get("price_target_confidences", {})
        adversarial_confidences = base_predictions.get("adversarial_confidences", {})

        if not price_target_confidences and not adversarial_confidences:
            raise ValueError("No valid prediction data available")

        # Calculate weighted average confidence for each direction
        up_confidence = self._calculate_directional_confidence(
            price_target_confidences,
            "up",
        )
        down_confidence = self._calculate_directional_confidence(
            adversarial_confidences,
            "down",
        )

        # Determine primary direction
        if up_confidence > down_confidence:
            primary_direction = "up"
            primary_confidence = up_confidence
            magnitude_levels = self._get_magnitude_levels(
                price_target_confidences,
                "up",
            )
        else:
            primary_direction = "down"
            primary_confidence = down_confidence
            magnitude_levels = self._get_magnitude_levels(
                adversarial_confidences,
                "down",
            )

        return {
            "direction": primary_direction,
            "confidence": primary_confidence,
            "magnitude_levels": magnitude_levels,
            "up_confidence": up_confidence,
            "down_confidence": down_confidence,
        }

    async def _calculate_adversarial_probabilities(
        self,
        directional_prediction: dict[str, Any],
        market_data: pd.DataFrame,
        current_price: float,
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

                for adverse_level in self.adversarial_movement_levels:
                    probability = await self._calculate_adverse_probability(
                        primary_direction,
                        magnitude,
                        adverse_level,
                        market_data,
                        current_price,
                    )
                    adverse_probabilities[f"{adverse_level:.1f}%"] = probability

                adversarial_analysis[f"{magnitude:.1f}%"] = {
                    "adverse_probabilities": adverse_probabilities,
                    "risk_score": self._calculate_risk_score(adverse_probabilities),
                    "recommended_stop_loss": self._calculate_recommended_stop_loss(
                        magnitude,
                        adverse_probabilities,
                    ),
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
        current_price: float,
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
        base_predictions = await self.predict_confidence_table(
            market_data,
            current_price,
        )

        if not base_predictions:
            raise ValueError(
                "Unable to generate base predictions for adverse probability calculation",
            )

        # Determine which prediction set to use based on primary direction
        if primary_direction == "up":
            # For upward primary prediction, use expected decreases for adverse
            predictions = base_predictions.get("adversarial_confidences", {})
        else:
            # For downward primary prediction, use confidence scores for adverse
            predictions = base_predictions.get("price_target_confidences", {})

        if not predictions:
            raise ValueError(
                f"No valid prediction data available for {primary_direction} direction",
            )

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
        direction: str,
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
        direction: str,
    ) -> list[float]:
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
        for level_str in predictions:
            level = float(level_str.replace("%", ""))
            if (
                predictions[level_str] > 0.1
            ):  # Only include levels with >10% probability
                levels.append(level)

        if not levels:
            raise ValueError(
                f"No significant probability levels found for {direction} direction",
            )

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
        adverse_probabilities: dict[str, float],
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
            raise ValueError(
                "No adverse probabilities provided for stop loss calculation",
            )

        # Find the level where adverse probability exceeds 30%
        for level_str, probability in adverse_probabilities.items():
            if probability > 0.3:
                return float(level_str.replace("%", ""))

        # If no level exceeds 30%, use 50% of primary magnitude
        return primary_magnitude * 0.5

    async def _calculate_risk_assessment(
        self,
        directional_prediction: dict[str, Any],
        adversarial_analysis: dict[str, Any],
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
            raise ValueError(
                "No adversarial analysis data provided for risk assessment",
            )

        # Calculate overall risk metrics
        total_risk_score = 0.0
        risk_levels = []

        for magnitude, analysis in adversarial_analysis.items():
            risk_score = analysis["risk_score"]
            total_risk_score += risk_score
            risk_levels.append(
                {
                    "magnitude": magnitude,
                    "risk_score": risk_score,
                    "stop_loss": analysis["recommended_stop_loss"],
                },
            )

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
                directional_prediction,
                avg_risk_score,
            ),
        }

    def _generate_risk_recommendation(
        self,
        directional_prediction: dict[str, Any],
        risk_score: float,
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
        if risk_score > 0.7:
            return f"HIGH_RISK: {direction.upper()} position with tight stop loss recommended"
        if risk_score > 0.5:
            return (
                f"MEDIUM_RISK: {direction.upper()} position with moderate position size"
            )
        return f"LOW_RISK: {direction.upper()} position with normal position size"

    async def _initialize_enhanced_order_manager(self) -> None:
        """Initialize enhanced order manager and async order executor for tactician order management."""
        try:
            # Import order management components
            from src.tactician.async_order_executor import setup_async_order_executor
            from src.tactician.enhanced_order_manager import (
                setup_enhanced_order_manager,
            )

            # Get configuration for order management
            order_config = self.config.get(
                "enhanced_order_manager",
                {
                    "enable_enhanced_order_manager": True,
                    "enable_async_order_executor": True,
                    "enable_chase_micro_breakout": True,
                    "enable_limit_order_return": True,
                    "enable_partial_fill_management": True,
                    "max_order_retries": 3,
                    "order_timeout_seconds": 30,
                    "slippage_tolerance": 0.001,
                    "volume_threshold": 1.5,
                    "momentum_threshold": 0.02,
                    "execution_strategies": {
                        "immediate": {"max_slippage": 0.001, "timeout_seconds": 30},
                        "batch": {"batch_size": 0.1, "batch_interval": 5},
                        "twap": {"duration_minutes": 10, "intervals": 20},
                        "vwap": {"volume_threshold": 1.5, "price_deviation": 0.002},
                        "iceberg": {"iceberg_qty": 0.1, "display_qty": 0.01},
                        "adaptive": {
                            "dynamic_slippage": True,
                            "market_impact_aware": True,
                        },
                    },
                },
            )

            # Initialize enhanced order manager
            self.enhanced_order_manager = await setup_enhanced_order_manager(
                order_config,
            )
            if self.enhanced_order_manager:
                self.logger.info("✅ Enhanced order manager initialized successfully")
            else:
                self.logger.warning("Failed to initialize enhanced order manager")

            # Initialize async order executor
            self.async_order_executor = await setup_async_order_executor(order_config)
            if self.async_order_executor:
                self.logger.info("✅ Async order executor initialized successfully")
            else:
                self.logger.warning("Failed to initialize async order executor")

        except Exception as e:
            self.logger.error(f"Error initializing enhanced order manager: {e}")
            self.enhanced_order_manager = None
            self.async_order_executor = None

    async def execute_chase_micro_breakout(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        breakout_price: float,
        strategy_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute CHASE_MICRO_BREAKOUT strategy with stop-limit order placement.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            quantity: Order quantity
            current_price: Current market price
            breakout_price: Expected breakout price
            strategy_id: Strategy identifier
            **kwargs: Additional parameters

        Returns:
            Dictionary containing execution results
        """
        try:
            if not self.enhanced_order_manager:
                return {
                    "success": False,
                    "error": "Enhanced order manager not initialized",
                    "order_id": None,
                }

            # Convert side string to OrderSide enum
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            # Place the chase micro breakout order
            order_state = (
                await self.enhanced_order_manager.place_chase_micro_breakout_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=quantity,
                    current_price=current_price,
                    breakout_price=breakout_price,
                    strategy_id=strategy_id,
                    **kwargs,
                )
            )

            if order_state:
                return {
                    "success": True,
                    "order_id": order_state.order_id,
                    "order_type": "CHASE_MICRO_BREAKOUT",
                    "stop_price": order_state.stop_price,
                    "limit_price": order_state.price,
                    "quantity": order_state.original_quantity,
                    "status": order_state.status.value,
                    "strategy_id": strategy_id,
                }
            return {
                "success": False,
                "error": "Failed to place chase micro breakout order",
                "order_id": None,
            }

        except Exception as e:
            self.logger.error(f"Error executing CHASE_MICRO_BREAKOUT: {e}")
            return {"success": False, "error": str(e), "order_id": None}

    async def execute_limit_order_return(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        leverage: float | None = None,
        strategy_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute LIMIT_ORDER_RETURN strategy with leveraged limit order placement.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            quantity: Order quantity
            price: Limit price
            leverage: Leverage to use (optional)
            strategy_id: Strategy identifier
            **kwargs: Additional parameters

        Returns:
            Dictionary containing execution results
        """
        try:
            if not self.enhanced_order_manager:
                return {
                    "success": False,
                    "error": "Enhanced order manager not initialized",
                    "order_id": None,
                }

            # Convert side string to OrderSide enum
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            # Place the limit order return
            order_state = await self.enhanced_order_manager.place_limit_order_return(
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                price=price,
                leverage=leverage,
                strategy_id=strategy_id,
                **kwargs,
            )

            if order_state:
                return {
                    "success": True,
                    "order_id": order_state.order_id,
                    "order_type": "LIMIT_ORDER_RETURN",
                    "price": order_state.price,
                    "quantity": order_state.original_quantity,
                    "leverage": order_state.leverage,
                    "status": order_state.status.value,
                    "strategy_id": strategy_id,
                }
            return {
                "success": False,
                "error": "Failed to place limit order return",
                "order_id": None,
            }

        except Exception as e:
            self.logger.error(f"Error executing LIMIT_ORDER_RETURN: {e}")
            return {"success": False, "error": str(e), "order_id": None}

    def get_order_status(self, order_id: str) -> dict[str, Any] | None:
        """Get the status of an order."""
        try:
            if not self.enhanced_order_manager:
                return None

            order_state = self.enhanced_order_manager.get_order_status(order_id)
            if order_state:
                return {
                    "order_id": order_state.order_id,
                    "symbol": order_state.symbol,
                    "side": order_state.side.value,
                    "order_type": order_state.order_type.value,
                    "status": order_state.status.value,
                    "original_quantity": order_state.original_quantity,
                    "executed_quantity": order_state.executed_quantity,
                    "remaining_quantity": order_state.remaining_quantity,
                    "average_price": order_state.average_price,
                    "price": order_state.price,
                    "leverage": order_state.leverage,
                    "strategy_type": order_state.strategy_type,
                    "created_time": order_state.created_time.isoformat(),
                    "updated_time": order_state.updated_time.isoformat(),
                }
            return None

        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None

    def get_strategy_orders(self, strategy_id: str) -> list[dict[str, Any]]:
        """Get all orders for a specific strategy."""
        try:
            if not self.enhanced_order_manager:
                return []

            order_states = self.enhanced_order_manager.get_strategy_orders(strategy_id)
            return [
                {
                    "order_id": order_state.order_id,
                    "symbol": order_state.symbol,
                    "side": order_state.side.value,
                    "order_type": order_state.order_type.value,
                    "status": order_state.status.value,
                    "original_quantity": order_state.original_quantity,
                    "executed_quantity": order_state.executed_quantity,
                    "remaining_quantity": order_state.remaining_quantity,
                    "average_price": order_state.average_price,
                    "price": order_state.price,
                    "leverage": order_state.leverage,
                    "strategy_type": order_state.strategy_type,
                    "created_time": order_state.created_time.isoformat(),
                    "updated_time": order_state.updated_time.isoformat(),
                }
                for order_state in order_states
            ]

        except Exception as e:
            self.logger.error(f"Error getting strategy orders: {e}")
            return []

    def get_order_manager_performance(self) -> dict[str, Any]:
        """Get enhanced order manager performance metrics."""
        try:
            if not self.enhanced_order_manager:
                return {}

            return self.enhanced_order_manager.get_performance_metrics()

        except Exception as e:
            self.logger.error(f"Error getting order manager performance: {e}")
            return {}

    async def execute_order_with_strategy(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        strategy_type: str = "immediate",
        leverage: float | None = None,
        strategy_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute order with specified strategy using async order executor.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            quantity: Order quantity
            price: Order price (optional for market orders)
            strategy_type: Execution strategy ("immediate", "batch", "twap", "vwap", "iceberg", "adaptive")
            leverage: Leverage (optional)
            strategy_id: Strategy identifier
            **kwargs: Additional parameters

        Returns:
            Dictionary containing execution results
        """
        try:
            if not self.async_order_executor:
                return {
                    "success": False,
                    "error": "Async order executor not available",
                    "execution_id": None,
                }

            # Import required components
            from src.tactician.async_order_executor import (
                ExecutionRequest,
                ExecutionStrategy,
                OrderSide,
                OrderType,
            )

            # Convert side string to OrderSide enum
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            # Determine order type
            order_type = OrderType.LIMIT if price else OrderType.MARKET

            # Convert strategy type to ExecutionStrategy enum
            strategy_map = {
                "immediate": ExecutionStrategy.IMMEDIATE,
                "batch": ExecutionStrategy.BATCH,
                "twap": ExecutionStrategy.TWAP,
                "vwap": ExecutionStrategy.VWAP,
                "iceberg": ExecutionStrategy.ICEBERG,
                "adaptive": ExecutionStrategy.ADAPTIVE,
            }
            execution_strategy = strategy_map.get(
                strategy_type,
                ExecutionStrategy.IMMEDIATE,
            )

            # Create execution request
            execution_request = ExecutionRequest(
                symbol=symbol,
                side=order_side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                leverage=leverage,
                strategy_type=strategy_type,
                execution_strategy=execution_strategy,
                strategy_id=strategy_id,
                metadata=kwargs,
            )

            # Execute order
            execution_id = await self.async_order_executor.execute_order_async(
                execution_request,
            )

            return {
                "success": True,
                "execution_id": execution_id,
                "strategy_type": strategy_type,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "leverage": leverage,
            }

        except Exception as e:
            self.logger.error(f"Error executing order with strategy: {e}")
            return {"success": False, "error": str(e), "execution_id": None}

    def get_execution_status(self, execution_id: str) -> dict[str, Any] | None:
        """Get execution status for a specific execution ID."""
        try:
            if not self.async_order_executor:
                return {"error": "Async order executor not available"}

            execution_result = self.async_order_executor.get_execution_status(
                execution_id,
            )
            if execution_result:
                return {
                    "execution_id": execution_result.execution_id,
                    "status": execution_result.status.value,
                    "executed_quantity": execution_result.executed_quantity,
                    "average_price": execution_result.average_price,
                    "slippage": execution_result.slippage,
                    "execution_time": execution_result.execution_time,
                    "performance_metrics": execution_result.performance_metrics,
                }
            return {"error": "Execution not found"}

        except Exception as e:
            self.logger.error(f"Error getting execution status: {e}")
            return {"error": str(e)}

    def get_execution_performance(self) -> dict[str, Any]:
        """Get overall execution performance metrics."""
        try:
            if not self.async_order_executor:
                return {"error": "Async order executor not available"}

            return self.async_order_executor.get_performance_metrics()

        except Exception as e:
            self.logger.error(f"Error getting execution performance: {e}")
            return {"error": str(e)}

    async def trigger_model_training(
        self,
        training_data: pd.DataFrame,
        training_type: str = "continuous",
        force_training: bool = False,
    ) -> dict[str, Any]:
        """
        Trigger model training based on conditions or force.

        Args:
            training_data: Historical data for training
            training_type: Type of training ("continuous", "adaptive", "incremental", "full")
            force_training: Force training regardless of conditions

        Returns:
            Dictionary containing training results
        """
        try:
            if not self.enhanced_training_manager:
                return {
                    "success": False,
                    "error": "Enhanced training manager not available",
                }

            # Check if training is needed
            if not force_training and not self._should_trigger_training():
                return {
                    "success": False,
                    "reason": "Training conditions not met",
                    "last_training": self.last_training_time,
                    "performance_degradation": self._calculate_performance_degradation(),
                }

            # Prepare training input
            training_input = {
                "symbol": "ETHUSDT",  # Default symbol
                "exchange": "binance",  # Default exchange
                "timeframes": self.analyst_timeframes + self.tactician_timeframes,
                "training_data": training_data,
                "training_type": training_type,
                "model_types": {
                    "analyst": ["tcn", "tabnet", "transformer"],
                    "tactician": ["lstm", "gru", "transformer"],
                },
                "enable_ensemble_training": self.training_config.get(
                    "enable_ensemble_training",
                    True,
                ),
                "enable_regime_specific_training": self.training_config.get(
                    "enable_regime_specific_training",
                    True,
                ),
                "enable_multi_timeframe_training": self.training_config.get(
                    "enable_multi_timeframe_training",
                    True,
                ),
                "enable_dual_model_training": self.training_config.get(
                    "enable_dual_model_training",
                    True,
                ),
                "enable_confidence_calibration": self.training_config.get(
                    "enable_confidence_calibration",
                    True,
                ),
            }

            # Execute training
            training_success = (
                await self.enhanced_training_manager.execute_enhanced_training(
                    training_input,
                )
            )

            if training_success:
                # Update training state
                self.last_training_time = datetime.now()
                self.training_history.append(
                    {
                        "timestamp": self.last_training_time,
                        "training_type": training_type,
                        "success": True,
                    },
                )

                # Refresh models
                await self.refresh_models_from_enhanced_training()

                return {
                    "success": True,
                    "training_type": training_type,
                    "timestamp": self.last_training_time.isoformat(),
                    "models_updated": True,
                }
            return {"success": False, "error": "Training execution failed"}

        except Exception as e:
            self.logger.error(f"Error triggering model training: {e}")
            return {"success": False, "error": str(e)}

    def _should_trigger_training(self) -> bool:
        """Determine if training should be triggered based on conditions."""
        try:
            # Check time-based conditions
            if self.last_training_time is None:
                return True  # First training

            hours_since_training = (
                datetime.now() - self.last_training_time
            ).total_seconds() / 3600
            if hours_since_training >= self.training_config.get(
                "training_interval_hours",
                24,
            ):
                return True

            # Check performance degradation
            performance_degradation = self._calculate_performance_degradation()
            if performance_degradation > self.training_config.get(
                "performance_degradation_threshold",
                0.1,
            ):
                return True

            # Check data availability
            if len(self.model_performance_history) >= self.training_config.get(
                "min_samples_for_retraining",
                1000,
            ):
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking training conditions: {e}")
            return False

    def _calculate_performance_degradation(self) -> float:
        """Calculate model performance degradation."""
        try:
            if len(self.model_performance_history) < 2:
                return 0.0

            # Calculate average performance over last 10 samples
            recent_performance = self.model_performance_history[-10:]
            if not recent_performance:
                return 0.0

            avg_recent = sum(p.get("accuracy", 0.0) for p in recent_performance) / len(
                recent_performance,
            )

            # Compare with baseline performance
            baseline_performance = 0.7  # Expected baseline accuracy
            degradation = max(0.0, baseline_performance - avg_recent)

            return degradation

        except Exception as e:
            self.logger.error(f"Error calculating performance degradation: {e}")
            return 0.0

    async def update_model_performance(
        self,
        performance_metrics: dict[str, Any],
    ) -> None:
        """Update model performance history."""
        try:
            self.model_performance_history.append(
                {"timestamp": datetime.now(), "metrics": performance_metrics},
            )

            # Keep only last 100 performance records
            if len(self.model_performance_history) > 100:
                self.model_performance_history = self.model_performance_history[-100:]

        except Exception as e:
            self.logger.error(f"Error updating model performance: {e}")

    def get_training_status(self) -> dict[str, Any]:
        """Get current training status and history."""
        try:
            return {
                "last_training_time": self.last_training_time.isoformat()
                if self.last_training_time
                else None,
                "training_history": self.training_history[
                    -10:
                ],  # Last 10 training events
                "model_performance_history": self.model_performance_history[
                    -10:
                ],  # Last 10 performance records
                "training_config": self.training_config,
                "should_trigger_training": self._should_trigger_training(),
                "performance_degradation": self._calculate_performance_degradation(),
            }

        except Exception as e:
            self.logger.error(f"Error getting training status: {e}")
            return {"error": str(e)}

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

    def update_ensemble_weights(self, performance_history: dict[str, float] = None, regime: str = None):
        """
        Dynamically update ensemble weights based on recent performance, regime, or meta-model.
        If a meta-model is available, use it for weighting; otherwise, use recent accuracy.
        """
        if performance_history:
            total = sum(performance_history.values())
            if total > 0:
                self.ensemble_weights = {k: v / total for k, v in performance_history.items()}
            else:
                self.ensemble_weights = {k: 1.0 / len(performance_history) for k in performance_history}
        # Placeholder: regime-specific or meta-model weighting can be added here
        # Example: if regime and regime in self.regime_models: ...
        self.logger.info(f"Updated ensemble weights: {self.ensemble_weights}")

    def ablation_study(self, features: pd.DataFrame, y_true: np.ndarray) -> dict:
        """
        Perform ablation study: remove each ensemble member in turn and measure performance drop.
        Returns a dict of member: performance_with_removal.
        """
        results = {}
        for member in self.ensemble_models:
            others = {k: v for k, v in self.ensemble_models.items() if k != member}
            if not others:
                continue
            preds = np.mean([m.predict(features) for m in others.values()], axis=0)
            acc = np.mean((preds > 0.5) == y_true)
            results[member] = acc
        self.logger.info(f"Ablation study results: {results}")
        return results


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
        return None

    except Exception as e:
        system_logger.error(f"Error setting up ML Confidence Predictor: {e}")
        return None

    async def train_price_target_confidence_model(
        self,
        historical_data: pd.DataFrame,
        price_targets: list[float] = None,
        adversarial_levels: list[float] = None,
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
                adversarial_levels = self.adversarial_movement_levels

            # Prepare training data with price target labels
            training_data = await self._prepare_price_target_training_data(
                historical_data,
                price_targets,
                adversarial_levels,
            )

            if training_data is None or training_data.empty:
                raise ValueError("No valid training data available")

            # Train separate models for each price target
            self.price_target_models = {}
            self.adversarial_models = {}

            # Train price target confidence models
            for target in price_targets:
                model_key = f"price_target_{target:.1f}"
                self.price_target_models[
                    model_key
                ] = await self._train_single_target_model(
                    training_data,
                    target,
                    "price_target",
                )

            # Train adversarial confidence models
            for level in adversarial_levels:
                model_key = f"adversarial_{level:.1f}"
                self.adversarial_models[
                    model_key
                ] = await self._train_single_target_model(
                    training_data,
                    level,
                    "adversarial",
                )

            self.is_trained = True
            self.last_training_time = datetime.now()

            self.logger.info("✅ Price target confidence model trained successfully")
            self.logger.info(
                f"   - {len(self.price_target_models)} price target models",
            )
            self.logger.info(f"   - {len(self.adversarial_models)} adversarial models")

            return True

        except Exception as e:
            self.logger.error(f"Error training price target confidence model: {str(e)}")
            return False

    async def _prepare_price_target_training_data(
        self,
        historical_data: pd.DataFrame,
        price_targets: list[float],
        adversarial_levels: list[float],
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
                current_price = historical_data.iloc[i]["close"]
                future_prices = historical_data.iloc[i + 1 :]["close"]

                # Calculate if each price target was reached
                target_labels = {}
                for target in price_targets:
                    target_price = current_price * (1 + target / 100)
                    reached_target = (future_prices >= target_price).any()
                    target_labels[f"target_{target:.1f}"] = 1 if reached_target else 0

                # Calculate if adversarial levels were reached before targets
                adversarial_labels = {}
                for level in adversarial_levels:
                    adversarial_price = current_price * (
                        1 - level / 100
                    )  # Downward movement
                    reached_adversarial = (future_prices <= adversarial_price).any()
                    adversarial_labels[f"adversarial_{level:.1f}"] = (
                        1 if reached_adversarial else 0
                    )

                # Combine features and labels
                row_data = {
                    "timestamp": historical_data.index[i],
                    **target_labels,
                    **adversarial_labels,
                }

                # Add technical features
                for col in historical_data.columns:
                    if col not in ["timestamp", "close"]:
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
        model_type: str,
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
            feature_columns = [
                col
                for col in training_data.columns
                if col not in ["timestamp"]
                and not col.startswith(("target_", "adversarial_"))
            ]

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
                verbose=-1,
            )

            # Use cross-validation for robust training
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(10, verbose=False), log_evaluation(0)],
                )

            return model

        except Exception as e:
            self.logger.error(
                f"Error training {model_type} model for level {target_level}: {str(e)}",
            )
            return None

    def _create_mock_models_for_testing(self) -> None:
        """Create mock models for testing when no real models are available."""
        try:
            self.logger.info("Creating mock models for testing...")

            # Create mock price target models
            for level in self.price_movement_levels:
                target_key = f"price_target_{level:.1f}"
                self.price_target_models[target_key] = self._create_mock_model(
                    f"price_target_{level}",
                )

            # Create mock adversarial models
            for level in self.adversarial_movement_levels:
                adversarial_key = f"adversarial_{level:.1f}"
                self.adversarial_models[adversarial_key] = self._create_mock_model(
                    f"adversarial_{level}",
                )

            # Create mock ensemble models
            ensemble_names = [
                "analyst_ensemble",
                "tactician_ensemble",
                "combined_ensemble",
            ]
            for name in ensemble_names:
                self.ensemble_models[name] = self._create_mock_ensemble(name)

            # Create mock regime models
            regime_names = ["trending", "ranging", "volatile", "stable"]
            for name in regime_names:
                self.regime_models[name] = self._create_mock_model(f"regime_{name}")

            # Create mock multi-timeframe models
            timeframe_names = ["30m", "15m", "5m", "1m"]
            for timeframe in timeframe_names:
                self.multi_timeframe_models[timeframe] = self._create_mock_model(
                    f"timeframe_{timeframe}",
                )

            self.is_trained = True
            self.last_training_time = datetime.now()
            self.logger.info("✅ Mock models created successfully for testing")

        except Exception as e:
            self.logger.error(f"Error creating mock models: {e}")

    def _create_mock_model(self, model_name: str) -> Any:
        """Create a mock model for testing."""

        class MockModel:
            def __init__(self, name: str):
                self.name = name
                self.is_trained = True

            def predict(self, X):
                # Return random predictions between 0 and 1
                import numpy as np

                return np.random.random(len(X))

            def predict_proba(self, X):
                # Return random probabilities
                import numpy as np

                proba = np.random.random((len(X), 2))
                # Normalize to sum to 1
                proba = proba / proba.sum(axis=1, keepdims=True)
                return proba

        return MockModel(model_name)

    def _create_mock_ensemble(self, ensemble_name: str) -> Any:
        """Create a mock ensemble model for testing."""

        class MockEnsemble:
            def __init__(self, name: str):
                self.name = name
                self.is_trained = True
                self.models = [
                    self._create_mock_model(f"{name}_model_{i}") for i in range(3)
                ]

            def predict(self, X):
                # Average predictions from all models
                import numpy as np

                predictions = [model.predict(X) for model in self.models]
                return np.mean(predictions, axis=0)

            def predict_proba(self, X):
                # Average probabilities from all models
                import numpy as np

                probas = [model.predict_proba(X) for model in self.models]
                return np.mean(probas, axis=0)

            def _create_mock_model(self, name: str):
                return self._create_mock_model(name)

        return MockEnsemble(ensemble_name)

    def is_enhanced_training_available(self) -> bool:
        """Check if enhanced training manager is available and has models."""
        try:
            if not self.enhanced_training_manager:
                return False

            # Check if enhanced training manager is initialized
            if not hasattr(
                self.enhanced_training_manager,
                "analyst_models",
            ) or not hasattr(self.enhanced_training_manager, "tactician_models"):
                return False

            # Check if we have any models loaded
            analyst_models = self.enhanced_training_manager.analyst_models
            tactician_models = self.enhanced_training_manager.tactician_models

            return len(analyst_models) > 0 or len(tactician_models) > 0

        except Exception as e:
            self.logger.error(f"Error checking enhanced training availability: {e}")
            return False

    def get_model_availability_status(self) -> dict[str, Any]:
        """Get detailed status of model availability."""
        try:
            status = {
                "enhanced_training_manager_available": self.enhanced_training_manager
                is not None,
                "enhanced_training_initialized": False,
                "analyst_models_available": False,
                "tactician_models_available": False,
                "ensemble_models_available": False,
                "calibrated_models_available": False,
                "regime_models_available": False,
                "multi_timeframe_models_available": False,
                "total_models_loaded": 0,
                "is_trained": self.is_trained,
                "last_training_time": self.last_training_time.isoformat()
                if self.last_training_time
                else None,
            }

            if self.enhanced_training_manager:
                status["enhanced_training_initialized"] = True

                # Check analyst models
                if hasattr(self.enhanced_training_manager, "analyst_models"):
                    analyst_models = self.enhanced_training_manager.analyst_models
                    status["analyst_models_available"] = len(analyst_models) > 0
                    status["total_models_loaded"] += len(analyst_models)

                # Check tactician models
                if hasattr(self.enhanced_training_manager, "tactician_models"):
                    tactician_models = self.enhanced_training_manager.tactician_models
                    status["tactician_models_available"] = len(tactician_models) > 0
                    status["total_models_loaded"] += len(tactician_models)

                # Check ensemble models
                if (
                    hasattr(self.enhanced_training_manager, "ensemble_creator")
                    and self.enhanced_training_manager.ensemble_creator
                ):
                    try:
                        ensemble_models = self.enhanced_training_manager.ensemble_creator.get_ensembles()
                        status["ensemble_models_available"] = (
                            len(ensemble_models) > 0 if ensemble_models else False
                        )
                        status["total_models_loaded"] += (
                            len(ensemble_models) if ensemble_models else 0
                        )
                    except Exception:
                        status["ensemble_models_available"] = False

                # Check calibrated models
                if hasattr(self.enhanced_training_manager, "calibration_systems"):
                    calibration_systems = (
                        self.enhanced_training_manager.calibration_systems
                    )
                    status["calibrated_models_available"] = len(calibration_systems) > 0
                    status["total_models_loaded"] += len(calibration_systems)

                # Check regime models
                if (
                    hasattr(self.enhanced_training_manager, "regime_training_manager")
                    and self.enhanced_training_manager.regime_training_manager
                ):
                    try:
                        regime_models = self.enhanced_training_manager.regime_training_manager.get_regime_models()
                        status["regime_models_available"] = (
                            len(regime_models) > 0 if regime_models else False
                        )
                        status["total_models_loaded"] += (
                            len(regime_models) if regime_models else 0
                        )
                    except Exception:
                        status["regime_models_available"] = False

                # Check multi-timeframe models
                if (
                    hasattr(self.enhanced_training_manager, "multi_timeframe_manager")
                    and self.enhanced_training_manager.multi_timeframe_manager
                ):
                    try:
                        multi_timeframe_models = self.enhanced_training_manager.multi_timeframe_manager.get_timeframe_models()
                        status["multi_timeframe_models_available"] = (
                            len(multi_timeframe_models) > 0
                            if multi_timeframe_models
                            else False
                        )
                        status["total_models_loaded"] += (
                            len(multi_timeframe_models) if multi_timeframe_models else 0
                        )
                    except Exception:
                        status["multi_timeframe_models_available"] = False

            return status

        except Exception as e:
            self.logger.error(f"Error getting model availability status: {e}")
            return {
                "error": str(e),
                "enhanced_training_manager_available": False,
                "is_trained": False,
            }
