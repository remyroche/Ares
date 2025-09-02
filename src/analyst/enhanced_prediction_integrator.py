# src/analyst/enhanced_prediction_integrator.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error, warning, failed, missing
from src.utils.centralized_decorators import (
import copy

    validate_data_quality,
    with_tracing_span,
    comprehensive_validation,
    intelligent_caching,
    performance_monitor,
    ValidationLevel,
    PerformanceLevel,
)


class EnhancedPredictionIntegrator:
    """
    Enhanced Prediction Integrator for Analyst that integrates price and confidence predictions
    from the enhanced training manager steps 6-14.
    
    This component loads and integrates:
    - HMM-based model predictions (step 6-8)
    - Analyst enhancement predictions (step 9)
    - Confidence calibration results (step 11)
    - Final parameter optimization results (step 12-14)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the enhanced prediction integrator.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("EnhancedPredictionIntegrator")

        # Model state
        self.is_initialized: bool = False
        self.models_loaded: bool = False
        self.calibration_loaded: bool = False

        # Loaded models and predictions
        self.hmm_models: dict[str, Any] = {}
        self.analyst_enhanced_models: dict[str, Any] = {}
        self.calibration_results: dict[str, Any] = {}
        self.optimization_results: dict[str, Any] = {}

        # Configuration
        self.integrator_config: dict[str, Any] = self.config.get("enhanced_prediction_integrator", {})
        self.data_dir: str = self.integrator_config.get("data_dir", "data/training")
        self.models_dir: str = self.integrator_config.get("models_dir", "models")
        
        # Prediction thresholds
        self.confidence_threshold: float = self.integrator_config.get("confidence_threshold", 0.7)
        self.price_prediction_threshold: float = self.integrator_config.get("price_prediction_threshold", 0.6)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhanced prediction integrator initialization",
    )
    @comprehensive_validation(validation_level=ValidationLevel.STRICT)
    @performance_monitor(performance_level=PerformanceLevel.HIGH)
    async def initialize(self) -> bool:
        """
        Initialize the enhanced prediction integrator.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🚀 Initializing Enhanced Prediction Integrator...")

            # Load HMM-based models (step 6-8)
            await self._load_hmm_models()

            # Load analyst enhanced models (step 9)
            await self._load_analyst_enhanced_models()

            # Load confidence calibration results (step 11)
            await self._load_calibration_results()

            # Load optimization results (step 12-14)
            await self._load_optimization_results()

            # Apply optimized parameters if available
            await self._apply_optimized_parameters()

            self.is_initialized = True
            self.logger.info("✅ Enhanced Prediction Integrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Enhanced Prediction Integrator initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="loading HMM models",
    )
    @with_tracing_span("load_hmm_models")
    @intelligent_caching(cache_key="hmm_models")
    async def _load_hmm_models(self) -> None:
        """Load HMM-based models from step 6-8."""
        try:
            hmm_models_path = Path(self.data_dir) / "hmm_models"
            if not hmm_models_path.exists():
                self.logger.warning(warning(f"⚠️ HMM models directory not found: {hmm_models_path}"))
                return

            for model_file in hmm_models_path.glob("*.pkl"):
                try:
                    with open(model_file, "rb") as f:
                        model_data = pickle.load(f)
                    
                    model_name = model_file.stem
                    self.hmm_models[model_name] = model_data
                    self.logger.info(f"✅ Loaded HMM model: {model_name}")
                
                except Exception as e:
                    self.logger.warning(warning(f"⚠️ Failed to load HMM model {model_file}: {e}"))

        except Exception as e:
            self.logger.error(error(f"❌ Error loading HMM models: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="loading analyst enhanced models",
    )
    async def _load_analyst_enhanced_models(self) -> None:
        """Load analyst enhanced models from step 9."""
        try:
            analyst_models_path = Path(self.data_dir) / "enhanced_analyst_models"
            if not analyst_models_path.exists():
                self.logger.warning(warning(f"⚠️ Analyst enhanced models directory not found: {analyst_models_path}"))
                return

            for regime_dir in analyst_models_path.iterdir():
                if regime_dir.is_dir():
                    regime_name = regime_dir.name
                    regime_models = {}
                    
                    for model_file in regime_dir.glob("*.pkl"):
                        try:
                            with open(model_file, "rb") as f:
                                model_data = pickle.load(f)
                            
                            model_name = model_file.stem
                            regime_models[model_name] = model_data
                            self.logger.info(f"✅ Loaded analyst model: {regime_name}/{model_name}")
                        
                        except Exception as e:
                            self.logger.warning(warning(f"⚠️ Failed to load analyst model {model_file}: {e}"))
                    
                    self.analyst_enhanced_models[regime_name] = regime_models

        except Exception as e:
            self.logger.error(error(f"❌ Error loading analyst enhanced models: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="loading calibration results",
    )
    async def _load_calibration_results(self) -> None:
        """Load confidence calibration results from step 11."""
        try:
            calibration_path = Path(self.data_dir) / "calibration_results"
            if not calibration_path.exists():
                self.logger.warning(warning(f"⚠️ Calibration results directory not found: {calibration_path}"))
                return

            for calibration_file in calibration_path.glob("*.pkl"):
                try:
                    with open(calibration_file, "rb") as f:
                        calibration_data = pickle.load(f)
                    
                    calibration_name = calibration_file.stem
                    self.calibration_results[calibration_name] = calibration_data
                    self.logger.info(f"✅ Loaded calibration results: {calibration_name}")
                
                except Exception as e:
                    self.logger.warning(warning(f"⚠️ Failed to load calibration results {calibration_file}: {e}"))

        except Exception as e:
            self.logger.error(error(f"❌ Error loading calibration results: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="loading optimization results",
    )
    async def _load_optimization_results(self) -> None:
        """Load optimization results from step 12-14."""
        try:
            optimization_path = Path(self.data_dir) / "optimization_results"
            if not optimization_path.exists():
                self.logger.warning(warning(f"⚠️ Optimization results directory not found: {optimization_path}"))
                return

            for optimization_file in optimization_path.glob("*.json"):
                try:
                    with open(optimization_file, "r") as f:
                        optimization_data = json.load(f)
                    
                    optimization_name = optimization_file.stem
                    self.optimization_results[optimization_name] = optimization_data
                    self.logger.info(f"✅ Loaded optimization results: {optimization_name}")
                
                except Exception as e:
                    self.logger.warning(warning(f"⚠️ Failed to load optimization results {optimization_file}: {e}"))

        except Exception as e:
            self.logger.error(error(f"❌ Error loading optimization results: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="applying optimized parameters",
    )
    @with_tracing_span("apply_optimized_parameters")
    async def _apply_optimized_parameters(self) -> bool:
        """Apply optimized parameters from step 12 optimization."""
        try:
            if not self.optimization_results:
                self.logger.info("ℹ️ No optimization results available, using default parameters")
                return True

            # Get confidence thresholds from optimization
            confidence_thresholds = self.optimization_results.get("confidence_thresholds", {})
            optimized_params = confidence_thresholds.get("optimized_parameters", {})

            # Apply enhanced prediction integrator parameters
            if "enhanced_prediction_confidence_threshold" in optimized_params:
                self.confidence_threshold = optimized_params["enhanced_prediction_confidence_threshold"]
                self.logger.info(f"✅ Applied optimized confidence threshold: {self.confidence_threshold}")

            if "enhanced_prediction_price_threshold" in optimized_params:
                self.price_prediction_threshold = optimized_params["enhanced_prediction_price_threshold"]
                self.logger.info(f"✅ Applied optimized price threshold: {self.price_prediction_threshold}")

            return True

        except Exception as e:
            self.logger.error(error(f"❌ Error applying optimized parameters: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating enhanced predictions",
    )
    @validate_data_quality(validation_level="WARNING")
    @with_tracing_span("generate_enhanced_predictions")
    @performance_monitor(performance_level=PerformanceLevel.HIGH)
    async def generate_enhanced_predictions(
        self, 
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """
        Generate enhanced predictions using all loaded models and calibration.

        Args:
            market_data: Market data for prediction
            regime_info: Current regime information
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            dict: Enhanced predictions with confidence scores
        """
        try:
            if not self.is_initialized:
                self.logger.error(error("❌ Enhanced Prediction Integrator not initialized"))
                return {}

            predictions = {
                "price_predictions": {},
                "confidence_scores": {},
                "regime_predictions": {},
                "calibrated_predictions": {},
                "optimization_weights": {},
                "timestamp": datetime.now().isoformat()
            }

            # Generate HMM-based predictions
            hmm_predictions = await self._generate_hmm_predictions(
                market_data, regime_info, symbol, exchange, timeframe
            )
            predictions["price_predictions"].update(hmm_predictions)

            # Generate analyst enhanced predictions
            analyst_predictions = await self._generate_analyst_predictions(
                market_data, regime_info, symbol, exchange, timeframe
            )
            predictions["price_predictions"].update(analyst_predictions)

            # Apply confidence calibration
            calibrated_predictions = await self._apply_confidence_calibration(
                predictions["price_predictions"], symbol, exchange
            )
            predictions["calibrated_predictions"] = calibrated_predictions

            # Apply optimization weights
            optimized_predictions = await self._apply_optimization_weights(
                predictions["calibrated_predictions"], symbol, exchange
            )
            predictions["optimization_weights"] = optimized_predictions

            # Generate final confidence scores
            final_confidence = await self._generate_final_confidence_scores(
                predictions["calibrated_predictions"], predictions["optimization_weights"]
            )
            predictions["confidence_scores"] = final_confidence

            return predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating enhanced predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating HMM predictions",
    )
    async def _generate_hmm_predictions(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """Generate predictions using HMM-based models."""
        try:
            predictions = {}
            
            for model_name, model_data in self.hmm_models.items():
                if "model" in model_data and hasattr(model_data["model"], "predict"):
                    try:
                        # Prepare features for prediction
                        features = self._prepare_features_for_prediction(market_data, regime_info)
                        
                        # Generate prediction
                        raw_prediction = model_data["model"].predict(features)
                        
                        # Apply model-specific post-processing
                        processed_prediction = self._process_hmm_prediction(
                            raw_prediction, model_data, model_name
                        )
                        
                        predictions[f"hmm_{model_name}"] = processed_prediction
                        
                    except Exception as e:
                        self.logger.warning(warning(f"⚠️ Failed to generate HMM prediction for {model_name}: {e}"))

            return predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating HMM predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating analyst predictions",
    )
    async def _generate_analyst_predictions(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """Generate predictions using analyst enhanced models."""
        try:
            predictions = {}
            
            current_regime = regime_info.get("regime", "default")
            regime_models = self.analyst_enhanced_models.get(current_regime, {})
            
            for model_name, model_data in regime_models.items():
                if "model" in model_data and hasattr(model_data["model"], "predict"):
                    try:
                        # Prepare features for prediction
                        features = self._prepare_features_for_prediction(market_data, regime_info)
                        
                        # Generate prediction
                        raw_prediction = model_data["model"].predict(features)
                        
                        # Apply model-specific post-processing
                        processed_prediction = self._process_analyst_prediction(
                            raw_prediction, model_data, model_name
                        )
                        
                        predictions[f"analyst_{current_regime}_{model_name}"] = processed_prediction
                        
                    except Exception as e:
                        self.logger.warning(warning(f"⚠️ Failed to generate analyst prediction for {model_name}: {e}"))

            return predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating analyst predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="applying confidence calibration",
    )
    async def _apply_confidence_calibration(
        self,
        predictions: dict[str, Any],
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """Apply confidence calibration to predictions."""
        try:
            calibrated_predictions = {}
            
            for prediction_name, prediction_data in predictions.items():
                # Find relevant calibration data
                calibration_key = f"{exchange}_{symbol}_calibration_results"
                calibration_data = self.calibration_results.get(calibration_key, {})
                
                if calibration_data:
                    # Apply calibration if available
                    calibrated_prediction = self._calibrate_prediction(
                        prediction_data, calibration_data, prediction_name
                    )
                    calibrated_predictions[prediction_name] = calibrated_prediction
                else:
                    # Use original prediction if no calibration available
                    calibrated_predictions[prediction_name] = prediction_data

            return calibrated_predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error applying confidence calibration: {e}"))
            return predictions

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="applying optimization weights",
    )
    async def _apply_optimization_weights(
        self,
        calibrated_predictions: dict[str, Any],
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """Apply optimization weights to calibrated predictions."""
        try:
            optimization_weights = {}
            
            # Find relevant optimization results
            optimization_key = f"{exchange}_{symbol}_optimization_results"
            optimization_data = self.optimization_results.get(optimization_key, {})
            
            if optimization_data:
                # Apply optimization weights if available
                for prediction_name in calibrated_predictions.keys():
                    weight = optimization_data.get("model_weights", {}).get(prediction_name, 1.0)
                    optimization_weights[prediction_name] = weight
            else:
                # Use equal weights if no optimization data available
                for prediction_name in calibrated_predictions.keys():
                    optimization_weights[prediction_name] = 1.0

            return optimization_weights

        except Exception as e:
            self.logger.error(error(f"❌ Error applying optimization weights: {e}"))
            return {name: 1.0 for name in calibrated_predictions.keys()}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating final confidence scores",
    )
    async def _generate_final_confidence_scores(
        self,
        calibrated_predictions: dict[str, Any],
        optimization_weights: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate final confidence scores for all predictions."""
        try:
            confidence_scores = {}
            
            for prediction_name, prediction_data in calibrated_predictions.items():
                # Extract confidence from prediction data
                base_confidence = prediction_data.get("confidence", 0.5)
                
                # Apply optimization weight
                weight = optimization_weights.get(prediction_name, 1.0)
                
                # Calculate weighted confidence
                weighted_confidence = base_confidence * weight
                
                # Normalize confidence score
                normalized_confidence = min(max(weighted_confidence, 0.0), 1.0)
                
                confidence_scores[prediction_name] = {
                    "base_confidence": base_confidence,
                    "weight": weight,
                    "weighted_confidence": weighted_confidence,
                    "normalized_confidence": normalized_confidence,
                    "confidence_level": self._get_confidence_level(normalized_confidence)
                }

            return confidence_scores

        except Exception as e:
            self.logger.error(error(f"❌ Error generating final confidence scores: {e}"))
            return {}

    def _prepare_features_for_prediction(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any]
    ) -> pd.DataFrame:
        """Prepare features for model prediction."""
        try:
            # Create a copy of market data
            features = market_data.copy()
            
            # Add regime information (contextual, not technical indicators)
            features["regime"] = regime_info.get("regime", "unknown")
            features["regime_confidence"] = regime_info.get("confidence", 0.5)
            
            # IMPORTANT: Do NOT add technical indicators here
            # The ML models in steps 6-14 already have comprehensive feature engineering
            # Adding RSI/MACD here would be redundant and potentially inconsistent
            # The models were trained with specific feature sets - we should respect that
            
            # Select relevant features for prediction (basic market data + regime info only)
            feature_columns = [
                "open", "high", "low", "close", "volume",
                "regime", "regime_confidence"
            ]
            
            available_features = [col for col in feature_columns if col in features.columns]
            return features[available_features].iloc[-1:].fillna(0)

        except Exception as e:
            self.logger.error(error(f"❌ Error preparing features: {e}"))
            return pd.DataFrame()

    def _process_hmm_prediction(
        self,
        raw_prediction: Any,
        model_data: dict[str, Any],
        model_name: str
    ) -> dict[str, Any]:
        """Process HMM model prediction."""
        try:
            if isinstance(raw_prediction, np.ndarray):
                prediction_value = float(raw_prediction[0]) if raw_prediction.size > 0 else 0.0
            else:
                prediction_value = float(raw_prediction)

            return {
                "prediction": prediction_value,
                "confidence": model_data.get("confidence", 0.5),
                "model_type": "hmm",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error processing HMM prediction: {e}"))
            return {"prediction": 0.0, "confidence": 0.0, "model_type": "hmm", "model_name": model_name}

    def _process_analyst_prediction(
        self,
        raw_prediction: Any,
        model_data: dict[str, Any],
        model_name: str
    ) -> dict[str, Any]:
        """Process analyst model prediction."""
        try:
            if isinstance(raw_prediction, np.ndarray):
                prediction_value = float(raw_prediction[0]) if raw_prediction.size > 0 else 0.0
            else:
                prediction_value = float(raw_prediction)

            return {
                "prediction": prediction_value,
                "confidence": model_data.get("confidence", 0.5),
                "model_type": "analyst",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error processing analyst prediction: {e}"))
            return {"prediction": 0.0, "confidence": 0.0, "model_type": "analyst", "model_name": model_name}

    def _calibrate_prediction(
        self,
        prediction_data: dict[str, Any],
        calibration_data: dict[str, Any],
        prediction_name: str
    ) -> dict[str, Any]:
        """Apply calibration to prediction."""
        try:
            calibrated_prediction = prediction_data.copy()
            
            # Find relevant calibration for this prediction
            model_calibration = calibration_data.get("model_calibrations", {}).get(prediction_name, {})
            
            if model_calibration:
                # Apply calibration transformation
                original_confidence = prediction_data.get("confidence", 0.5)
                calibrated_confidence = model_calibration.get("calibrated_confidence", original_confidence)
                
                calibrated_prediction["confidence"] = calibrated_confidence
                calibrated_prediction["calibration_applied"] = True
            else:
                calibrated_prediction["calibration_applied"] = False

            return calibrated_prediction

        except Exception as e:
            self.logger.error(error(f"❌ Error calibrating prediction: {e}"))
            return prediction_data

    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description."""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.8:
            return "high"
        elif confidence >= 0.7:
            return "medium_high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.5:
            return "medium_low"
        else:
            return "low"

    # REMOVED: RSI and MACD calculation methods
    # These technical indicators should be handled by the ML models in steps 6-14
    # The integrator should focus on integrating predictions, not generating features