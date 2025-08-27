# src/tactician/enhanced_prediction_integrator.py

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
    validate_data_quality,
    with_tracing_span,
    comprehensive_validation,
    intelligent_caching,
    performance_monitor,
    ValidationLevel,
    PerformanceLevel,
)


class TacticianEnhancedPredictionIntegrator:
    """
    Enhanced Prediction Integrator for Tactician that enhances existing position and leverage sizers
    with predictions from the enhanced training manager steps 6-14.
    
    This component enhances existing tactician components:
    - Enhances PositionSizer with ML-based confidence predictions
    - Enhances LeverageSizer with calibrated risk predictions
    - Provides additional context for existing tactician decision-making
    - Integrates HMM-based model predictions (step 6-8)
    - Integrates confidence calibration results (step 11)
    - Integrates optimization results (steps 12-14)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the tactician enhanced prediction integrator.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("TacticianEnhancedPredictionIntegrator")

        # Model state
        self.is_initialized: bool = False
        self.models_loaded: bool = False
        self.calibration_loaded: bool = False

        # Loaded models and predictions
        self.hmm_models: dict[str, Any] = {}
        self.calibration_results: dict[str, Any] = {}
        self.optimization_results: dict[str, Any] = {}
        
        # References to existing tactician components (will be set during integration)
        self.position_sizer = None
        self.leverage_sizer = None

        # Configuration
        self.integrator_config: dict[str, Any] = self.config.get("tactician_enhanced_prediction_integrator", {})
        self.data_dir: str = self.integrator_config.get("data_dir", "data/training")
        self.models_dir: str = self.integrator_config.get("models_dir", "models")
        
        # Prediction thresholds
        self.confidence_threshold: float = self.integrator_config.get("confidence_threshold", 0.7)
        self.price_prediction_threshold: float = self.integrator_config.get("price_prediction_threshold", 0.6)
        self.entry_threshold: float = self.integrator_config.get("entry_threshold", 0.65)
        self.exit_threshold: float = self.integrator_config.get("exit_threshold", 0.55)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tactician enhanced prediction integrator initialization",
    )
    @comprehensive_validation(validation_level=ValidationLevel.STRICT)
    @performance_monitor(performance_level=PerformanceLevel.HIGH)
    async def initialize(self) -> bool:
        """
        Initialize the tactician enhanced prediction integrator.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🚀 Initializing Tactician Enhanced Prediction Integrator...")

            # Load HMM-based models (step 6-8)
            await self._load_hmm_models()

            # Load confidence calibration results (step 11)
            await self._load_calibration_results()

                    # Load optimization results (step 12-14)
        await self._load_optimization_results()

        # Apply optimized parameters if available
        await self._apply_optimized_parameters()

        self.is_initialized = True
        self.logger.info("✅ Tactician Enhanced Prediction Integrator initialized successfully")
        return True

        except Exception as e:
            self.logger.error(failed(f"❌ Tactician Enhanced Prediction Integrator initialization failed: {e}"))
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
        context="loading tactician specialist models",
    )
    async def _load_tactician_specialist_models(self) -> None:
        """Load tactician specialist models from step 9."""
        try:
            tactician_models_path = Path(self.data_dir) / "tactician_specialist_models"
            if not tactician_models_path.exists():
                self.logger.warning(warning(f"⚠️ Tactician specialist models directory not found: {tactician_models_path}"))
                return

            for regime_dir in tactician_models_path.iterdir():
                if regime_dir.is_dir():
                    regime_name = regime_dir.name
                    regime_models = {}
                    
                    for model_file in regime_dir.glob("*.pkl"):
                        try:
                            with open(model_file, "rb") as f:
                                model_data = pickle.load(f)
                            
                            model_name = model_file.stem
                            regime_models[model_name] = model_data
                            self.logger.info(f"✅ Loaded tactician specialist model: {regime_name}/{model_name}")
                        
                        except Exception as e:
                            self.logger.warning(warning(f"⚠️ Failed to load tactician specialist model {model_file}: {e}"))
                    
                    self.tactician_specialist_models[regime_name] = regime_models

        except Exception as e:
            self.logger.error(error(f"❌ Error loading tactician specialist models: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="loading tactician labeling models",
    )
    async def _load_tactician_labeling_models(self) -> None:
        """Load tactician labeling models from step 10."""
        try:
            tactician_labeling_path = Path(self.data_dir) / "tactician_labeling"
            if not tactician_labeling_path.exists():
                self.logger.warning(warning(f"⚠️ Tactician labeling directory not found: {tactician_labeling_path}"))
                return

            for labeling_file in tactician_labeling_path.glob("*.pkl"):
                try:
                    with open(labeling_file, "rb") as f:
                        labeling_data = pickle.load(f)
                    
                    labeling_name = labeling_file.stem
                    self.tactician_labeling_models[labeling_name] = labeling_data
                    self.logger.info(f"✅ Loaded tactician labeling model: {labeling_name}")
                
                except Exception as e:
                    self.logger.warning(warning(f"⚠️ Failed to load tactician labeling model {labeling_file}: {e}"))

        except Exception as e:
            self.logger.error(error(f"❌ Error loading tactician labeling models: {e}"))

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

            # Store optimized parameters for use in enhancement methods
            self.optimized_params = optimized_params

            return True

        except Exception as e:
            self.logger.error(error(f"❌ Error applying optimized parameters: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating tactician enhanced predictions",
    )
    @validate_data_quality(validation_level="WARNING")
    @with_tracing_span("generate_tactician_enhanced_predictions")
    @performance_monitor(performance_level=PerformanceLevel.HIGH)
    async def generate_tactician_enhanced_predictions(
        self, 
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        analyst_signals: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """
        Generate enhanced predictions to augment existing tactician components.

        Args:
            market_data: Market data for prediction
            regime_info: Current regime information
            analyst_signals: Analyst signals and predictions
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            dict: Enhanced predictions to augment existing tactician components
        """
        try:
            if not self.is_initialized:
                self.logger.error(error("❌ Tactician Enhanced Prediction Integrator not initialized"))
                return {}

            enhanced_predictions = {
                "ml_confidence_predictions": {},
                "calibrated_confidence_scores": {},
                "optimization_weights": {},
                "hmm_predictions": {},
                "timestamp": datetime.now().isoformat()
            }

            # Generate HMM-based predictions for additional context
            hmm_predictions = await self._generate_hmm_predictions(
                market_data, regime_info, symbol, exchange, timeframe
            )
            enhanced_predictions["hmm_predictions"] = hmm_predictions

            # Generate ML confidence predictions to enhance existing components
            ml_confidence = await self._generate_ml_confidence_predictions(
                hmm_predictions, analyst_signals, symbol, exchange
            )
            enhanced_predictions["ml_confidence_predictions"] = ml_confidence

            # Apply confidence calibration
            calibrated_confidence = await self._apply_confidence_calibration(
                ml_confidence, symbol, exchange
            )
            enhanced_predictions["calibrated_confidence_scores"] = calibrated_confidence

            # Apply optimization weights
            optimization_weights = await self._apply_optimization_weights(
                calibrated_confidence, symbol, exchange
            )
            enhanced_predictions["optimization_weights"] = optimization_weights

            return enhanced_predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating tactician enhanced predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating HMM predictions",
    )
    @with_tracing_span("generate_hmm_predictions")
    @validate_data_quality(validation_level="WARNING")
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
        context="generating ML confidence predictions",
    )
    @with_tracing_span("generate_ml_confidence_predictions")
    @validate_data_quality(validation_level="WARNING")
    async def _generate_ml_confidence_predictions(
        self,
        hmm_predictions: dict[str, Any],
        analyst_signals: dict[str, Any],
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """Generate ML confidence predictions to enhance existing tactician components."""
        try:
            ml_confidence = {}
            
            # Calculate aggregate confidence from HMM predictions
            total_confidence = 0.0
            valid_predictions = 0
            
            for prediction_name, prediction_data in hmm_predictions.items():
                confidence = prediction_data.get("confidence", 0.0)
                if confidence > 0:
                    total_confidence += confidence
                    valid_predictions += 1
            
            avg_confidence = total_confidence / max(valid_predictions, 1)
            
            # Combine with analyst signals
            analyst_confidence = analyst_signals.get("confidence", 0.5)
            
            # Calculate weighted ML confidence using optimized parameters if available
            ml_weight = getattr(self, 'optimized_params', {}).get("ml_weight", 0.7)
            analyst_weight = getattr(self, 'optimized_params', {}).get("analyst_weight", 0.3)
            
            weighted_ml_confidence = (avg_confidence * ml_weight) + (analyst_confidence * analyst_weight)
            
            ml_confidence["aggregate_ml_confidence"] = {
                "hmm_avg_confidence": avg_confidence,
                "analyst_confidence": analyst_confidence,
                "weighted_ml_confidence": weighted_ml_confidence,
                "ml_weight": ml_weight,
                "analyst_weight": analyst_weight,
                "prediction_count": valid_predictions
            }
            
            return ml_confidence

        except Exception as e:
            self.logger.error(error(f"❌ Error generating ML confidence predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="applying confidence calibration",
    )
    @with_tracing_span("apply_confidence_calibration")
    @validate_data_quality(validation_level="WARNING")
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
    @with_tracing_span("apply_optimization_weights")
    @validate_data_quality(validation_level="WARNING")
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
        default_return=False,
        context="integrating with existing tactician components",
    )
    @with_tracing_span("integrate_with_tactician_components")
    async def integrate_with_tactician_components(
        self,
        position_sizer,
        leverage_sizer
    ) -> bool:
        """Integrate with existing tactician components to enhance their functionality."""
        try:
            self.position_sizer = position_sizer
            self.leverage_sizer = leverage_sizer
            
            self.logger.info("✅ Integrated with existing tactician components")
            return True
            
        except Exception as e:
            self.logger.error(error(f"❌ Error integrating with tactician components: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="enhancing position sizer with ML predictions",
    )
    @with_tracing_span("enhance_position_sizer")
    @validate_data_quality(validation_level="WARNING")
    async def enhance_position_sizer(
        self,
        base_position_size: float,
        analyst_confidence: float,
        enhanced_predictions: dict[str, Any]
    ) -> dict[str, Any]:
        """Enhance position sizer with ML-based confidence predictions."""
        try:
            if not self.position_sizer:
                return {"enhanced_position_size": base_position_size}
            
            # Get ML confidence from enhanced predictions
            ml_confidence = enhanced_predictions.get("ml_confidence_predictions", {})
            aggregate_ml_confidence = ml_confidence.get("aggregate_ml_confidence", {})
            weighted_ml_confidence = aggregate_ml_confidence.get("weighted_ml_confidence", 0.5)
            
            # Enhance position size calculation with ML confidence using optimized parameters
            confidence_multiplier = getattr(self, 'optimized_params', {}).get("position_sizing_confidence_multiplier", 1.5)
            ml_confidence_multiplier = min(weighted_ml_confidence * confidence_multiplier, 2.0)  # Max 2x multiplier
            enhanced_position_size = base_position_size * ml_confidence_multiplier
            
            return {
                "enhanced_position_size": enhanced_position_size,
                "ml_confidence_multiplier": ml_confidence_multiplier,
                "weighted_ml_confidence": weighted_ml_confidence,
                "original_position_size": base_position_size
            }
            
        except Exception as e:
            self.logger.error(error(f"❌ Error enhancing position sizer: {e}"))
            return {"enhanced_position_size": base_position_size}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="enhancing leverage sizer with ML predictions",
    )
    @with_tracing_span("enhance_leverage_sizer")
    @validate_data_quality(validation_level="WARNING")
    async def enhance_leverage_sizer(
        self,
        base_leverage: float,
        risk_score: float,
        enhanced_predictions: dict[str, Any]
    ) -> dict[str, Any]:
        """Enhance leverage sizer with ML-based risk predictions."""
        try:
            if not self.leverage_sizer:
                return {"enhanced_leverage": base_leverage}
            
            # Get calibrated confidence from enhanced predictions
            calibrated_confidence = enhanced_predictions.get("calibrated_confidence_scores", {})
            
            # Calculate ML-based risk adjustment using optimized parameters
            risk_multiplier = getattr(self, 'optimized_params', {}).get("leverage_sizing_risk_multiplier", 1.0)
            ml_risk_multiplier = 1.0
            if calibrated_confidence:
                # Higher confidence = lower risk = higher leverage
                avg_confidence = sum(cal.values() for cal in calibrated_confidence.values()) / max(len(calibrated_confidence), 1)
                ml_risk_multiplier = 0.5 + (avg_confidence * risk_multiplier)  # Use optimized risk multiplier
            
            enhanced_leverage = base_leverage * ml_risk_multiplier
            
            return {
                "enhanced_leverage": enhanced_leverage,
                "ml_risk_multiplier": ml_risk_multiplier,
                "original_leverage": base_leverage
            }
            
        except Exception as e:
            self.logger.error(error(f"❌ Error enhancing leverage sizer: {e}"))
            return {"enhanced_leverage": base_leverage}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating final confidence scores",
    )
    @with_tracing_span("generate_final_confidence_scores")
    @validate_data_quality(validation_level="WARNING")
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

    @validate_data_quality(validation_level="WARNING")
    def _prepare_features_for_prediction(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any]
    ) -> pd.DataFrame:
        """Prepare features for model prediction."""
        try:
            # Create a copy of market data
            features = market_data.copy()
            
            # Add regime information
            features["regime"] = regime_info.get("regime", "unknown")
            features["regime_confidence"] = regime_info.get("confidence", 0.5)
            
            # Add technical indicators if not present
            if "rsi" not in features.columns:
                features["rsi"] = self._calculate_rsi(features["close"])
            
            if "macd" not in features.columns:
                features["macd"] = self._calculate_macd(features["close"])
            
            # Select relevant features for prediction
            feature_columns = [
                "open", "high", "low", "close", "volume",
                "rsi", "macd", "regime", "regime_confidence"
            ]
            
            available_features = [col for col in feature_columns if col in features.columns]
            return features[available_features].iloc[-1:].fillna(0)

        except Exception as e:
            self.logger.error(error(f"❌ Error preparing features: {e}"))
            return pd.DataFrame()

    @validate_data_quality(validation_level="WARNING")
    def _prepare_features_with_analyst_signals(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        analyst_signals: dict[str, Any]
    ) -> pd.DataFrame:
        """Prepare features for prediction with analyst signals."""
        try:
            features = self._prepare_features_for_prediction(market_data, regime_info)
            
            # Add analyst signal information
            features["analyst_signal"] = analyst_signals.get("signal", 0)
            features["analyst_confidence"] = analyst_signals.get("confidence", 0.5)
            features["analyst_prediction"] = analyst_signals.get("prediction", 0.0)
            
            return features

        except Exception as e:
            self.logger.error(error(f"❌ Error preparing features with analyst signals: {e}"))
            return pd.DataFrame()

    @validate_data_quality(validation_level="WARNING")
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

    def _process_tactician_specialist_prediction(
        self,
        raw_prediction: Any,
        model_data: dict[str, Any],
        model_name: str
    ) -> dict[str, Any]:
        """Process tactician specialist model prediction."""
        try:
            if isinstance(raw_prediction, np.ndarray):
                prediction_value = float(raw_prediction[0]) if raw_prediction.size > 0 else 0.0
            else:
                prediction_value = float(raw_prediction)

            return {
                "prediction": prediction_value,
                "confidence": model_data.get("confidence", 0.5),
                "model_type": "tactician_specialist",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error processing tactician specialist prediction: {e}"))
            return {"prediction": 0.0, "confidence": 0.0, "model_type": "tactician_specialist", "model_name": model_name}

    def _process_tactician_labeling_prediction(
        self,
        raw_prediction: Any,
        labeling_data: dict[str, Any],
        labeling_name: str
    ) -> dict[str, Any]:
        """Process tactician labeling prediction."""
        try:
            if isinstance(raw_prediction, np.ndarray):
                prediction_value = float(raw_prediction[0]) if raw_prediction.size > 0 else 0.0
            else:
                prediction_value = float(raw_prediction)

            return {
                "prediction": prediction_value,
                "confidence": labeling_data.get("confidence", 0.5),
                "model_type": "tactician_labeling",
                "model_name": labeling_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error processing tactician labeling prediction: {e}"))
            return {"prediction": 0.0, "confidence": 0.0, "model_type": "tactician_labeling", "model_name": labeling_name}

    @validate_data_quality(validation_level="WARNING")
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

    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze current market conditions for exit decisions."""
        try:
            if market_data.empty:
                return {"volatility": "unknown", "trend": "unknown", "volume": "unknown"}
            
            # Calculate volatility
            returns = market_data["close"].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate trend
            sma_short = market_data["close"].rolling(5).mean().iloc[-1]
            sma_long = market_data["close"].rolling(20).mean().iloc[-1]
            trend = "bullish" if sma_short > sma_long else "bearish"
            
            # Calculate volume trend
            avg_volume = market_data["volume"].rolling(10).mean().iloc[-1]
            current_volume = market_data["volume"].iloc[-1]
            volume_trend = "high" if current_volume > avg_volume * 1.2 else "normal"
            
            return {
                "volatility": "high" if volatility > 0.02 else "low",
                "trend": trend,
                "volume": volume_trend,
                "volatility_value": float(volatility)
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error analyzing market conditions: {e}"))
            return {"volatility": "unknown", "trend": "unknown", "volume": "unknown"}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50.0] * len(prices), index=prices.index)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except Exception:
            return pd.Series([0.0] * len(prices), index=prices.index)