# src/supervisor/enhanced_prediction_service.py

"""
Centralized Enhanced Prediction Service
This service integrates price and confidence predictions from the enhanced training manager steps 6-14
and provides them to both Analyst and Tactician components.
"""

import asyncio
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

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


class EnhancedPredictionService:
    """
    Centralized Enhanced Prediction Service that provides ML-powered predictions
    to both Analyst and Tactician components.
    
    This service loads and integrates:
    - HMM-based model predictions (step 6-8)
    - Analyst enhanced models (step 9)
    - Tactician specialist models (step 9)
    - Confidence calibration results (step 11)
    - Final parameter optimization results (step 12-14)
    - ML Profit Integration System (Universal ML Profit Integration)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("EnhancedPredictionService")
        
        # Service state
        self.is_initialized: bool = False
        
        # Loaded models and predictions
        self.hmm_models: dict[str, Any] = {}
        self.analyst_enhanced_models: dict[str, Any] = {}
        self.tactician_specialist_models: dict[str, Any] = {}
        self.calibration_results: dict[str, Any] = {}
        self.optimization_results: dict[str, Any] = {}
        
        # ML Profit Integration System
        self.ml_profit_models: dict[str, Any] = {}
        self.profit_prediction_results: dict[str, Any] = {}
        self.barrier_analysis_results: dict[str, Any] = {}
        
        # Configuration
        self.service_config: dict[str, Any] = self.config.get("enhanced_prediction_service", {})
        self.data_dir: str = self.service_config.get("data_dir", "data/training")
        self.models_dir: str = self.service_config.get("models_dir", "models")
        
        # Prediction thresholds
        self.confidence_threshold: float = self.service_config.get("confidence_threshold", 0.7)
        self.price_prediction_threshold: float = self.service_config.get("price_prediction_threshold", 0.6)
        
        # ML Profit Integration thresholds
        self.profit_threshold: float = self.service_config.get("profit_threshold", 0.02)  # 2% default
        self.barrier_threshold: float = self.service_config.get("barrier_threshold", 0.01)  # 1% default
        self.direction_confidence_threshold: float = self.service_config.get("direction_confidence_threshold", 0.65)
        
        # Timeframe configuration
        self.tactician_timeframes: list[str] = self.service_config.get("timeframes", ["1m", "5m"])
        self.primary_timeframe: str = self.service_config.get("primary_timeframe", "1m")
        self.secondary_timeframe: str = self.service_config.get("secondary_timeframe", "5m")
        
        # Optimized parameters (will be loaded during initialization)
        self.optimized_params: dict[str, Any] = {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhanced prediction service initialization",
    )
    @comprehensive_validation(validation_level=ValidationLevel.STRICT)
    @performance_monitor(performance_level=PerformanceLevel.HIGH)
    async def initialize(self) -> bool:
        """
        Initialize the enhanced prediction service.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🚀 Initializing Enhanced Prediction Service...")

            # Load HMM-based models (step 6-8)
            await self._load_hmm_models()

            # Load analyst enhanced models (step 9)
            await self._load_analyst_enhanced_models()

            # Load tactician specialist models (step 9)
            await self._load_tactician_specialist_models()

            # Load confidence calibration results (step 11)
            await self._load_calibration_results()

            # Load optimization results (step 12-14)
            await self._load_optimization_results()

            # Load ML Profit Integration models (steps 6-14)
            await self._load_ml_profit_models()

            # Apply optimized parameters if available
            await self._apply_optimized_parameters()

            self.is_initialized = True
            self.logger.info("✅ Enhanced Prediction Service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Enhanced Prediction Service initialization failed: {e}"))
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
                            self.logger.info(f"✅ Loaded tactician model: {regime_name}/{model_name}")
                        
                        except Exception as e:
                            self.logger.warning(warning(f"⚠️ Failed to load tactician model {model_file}: {e}"))
                    
                    self.tactician_specialist_models[regime_name] = regime_models

        except Exception as e:
            self.logger.error(error(f"❌ Error loading tactician specialist models: {e}"))

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

            for calibration_file in calibration_path.glob("*.json"):
                try:
                    with open(calibration_file, "r") as f:
                        calibration_data = json.load(f)
                    
                    symbol_exchange = calibration_file.stem
                    self.calibration_results[symbol_exchange] = calibration_data
                    self.logger.info(f"✅ Loaded calibration results: {symbol_exchange}")
                
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
                    
                    symbol_exchange = optimization_file.stem
                    self.optimization_results[symbol_exchange] = optimization_data
                    self.logger.info(f"✅ Loaded optimization results: {symbol_exchange}")
                
                except Exception as e:
                    self.logger.warning(warning(f"⚠️ Failed to load optimization results {optimization_file}: {e}"))

        except Exception as e:
            self.logger.error(error(f"❌ Error loading optimization results: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="loading ML profit models",
    )
    @with_tracing_span("load_ml_profit_models")
    @intelligent_caching(cache_key="ml_profit_models")
    async def _load_ml_profit_models(self) -> None:
        """Load ML profit models from steps 6-14."""
        try:
            ml_profit_path = Path(self.data_dir) / "ml_profit_models"
            if not ml_profit_path.exists():
                self.logger.warning(warning(f"⚠️ ML profit models directory not found: {ml_profit_path}"))
                return

            # Load different types of ML profit models
            model_types = ["hmm_profit", "analyst_profit", "tactician_profit", "ensemble_profit"]
            
            for model_type in model_types:
                type_path = ml_profit_path / model_type
                if type_path.exists():
                    self.ml_profit_models[model_type] = {}
                    
                    for model_file in type_path.glob("*.pkl"):
                        try:
                            with open(model_file, "rb") as f:
                                model_data = pickle.load(f)
                            
                            model_name = model_file.stem
                            self.ml_profit_models[model_type][model_name] = model_data
                            self.logger.info(f"✅ Loaded ML profit model: {model_type}/{model_name}")
                        
                        except Exception as e:
                            self.logger.warning(warning(f"⚠️ Failed to load ML profit model {model_file}: {e}"))

        except Exception as e:
            self.logger.error(error(f"❌ Error loading ML profit models: {e}"))

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

            # Apply enhanced prediction service parameters
            if "enhanced_prediction_confidence_threshold" in optimized_params:
                self.confidence_threshold = optimized_params["enhanced_prediction_confidence_threshold"]
                self.logger.info(f"✅ Applied optimized confidence threshold: {self.confidence_threshold}")

            if "enhanced_prediction_price_threshold" in optimized_params:
                self.price_prediction_threshold = optimized_params["enhanced_prediction_price_threshold"]
                self.logger.info(f"✅ Applied optimized price threshold: {self.price_prediction_threshold}")

            # Store optimized parameters for use in prediction methods
            self.optimized_params = optimized_params

            return True

        except Exception as e:
            self.logger.error(error(f"❌ Error applying optimized parameters: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating analyst predictions",
    )
    @with_tracing_span("generate_analyst_predictions")
    @validate_data_quality(validation_level="WARNING")
    async def generate_analyst_predictions(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str = "1m"
    ) -> dict[str, Any]:
        """
        Generate enhanced predictions for the Analyst component.

        Args:
            market_data: Market data for prediction
            regime_info: Current regime information
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            dict: Enhanced predictions for Analyst
        """
        try:
            if not self.is_initialized:
                self.logger.error(error("❌ Enhanced Prediction Service not initialized"))
                return {}

            predictions = {
                "ml_profit_predictions": {},
                "enhanced_confidence_scores": {},
                "barrier_analysis": {},
                "regime_predictions": {},
                "timeframe_used": timeframe,
                "timestamp": datetime.now().isoformat()
            }

            # Generate ML profit predictions (Universal ML Profit Integration)
            ml_profit_predictions = await self._generate_ml_profit_predictions(
                market_data, regime_info, symbol, exchange, timeframe
            )
            predictions["ml_profit_predictions"] = ml_profit_predictions

            # Generate enhanced confidence scores with barrier analysis
            enhanced_confidence = await self._generate_enhanced_confidence_scores(
                ml_profit_predictions, market_data, symbol, exchange
            )
            predictions["enhanced_confidence_scores"] = enhanced_confidence

            # Generate barrier analysis
            barrier_analysis = await self._generate_barrier_analysis(
                ml_profit_predictions, market_data, symbol, exchange
            )
            predictions["barrier_analysis"] = barrier_analysis

            # Generate regime predictions
            regime_predictions = await self._generate_regime_predictions(
                market_data, regime_info, symbol, exchange
            )
            predictions["regime_predictions"] = regime_predictions

            return predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating analyst predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating tactician predictions",
    )
    @validate_data_quality(validation_level="WARNING")
    @with_tracing_span("generate_tactician_predictions")
    @performance_monitor(performance_level=PerformanceLevel.HIGH)
    async def generate_tactician_predictions(
        self, 
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        analyst_signals: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str = "1m"  # Default to 1m for Tactician
    ) -> dict[str, Any]:
        """
        Generate enhanced predictions for the Tactician component.

        Args:
            market_data: Market data for prediction
            regime_info: Current regime information
            analyst_signals: Analyst signals and predictions
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            dict: Enhanced predictions for Tactician
        """
        try:
            if not self.is_initialized:
                self.logger.error(error("❌ Enhanced Prediction Service not initialized"))
                return {}

            # Validate timeframe for Tactician (should be 1m or 5m)
            if timeframe not in self.tactician_timeframes:
                self.logger.warning(warning(f"⚠️ Invalid timeframe for Tactician: {timeframe}. Using {self.primary_timeframe}"))
                timeframe = self.primary_timeframe

            predictions = {
                "ml_confidence_predictions": {},
                "calibrated_confidence_scores": {},
                "optimization_weights": {},
                "hmm_predictions": {},
                "timeframe_used": timeframe,
                "timestamp": datetime.now().isoformat()
            }

            # Generate HMM-based predictions for additional context
            hmm_predictions = await self._generate_hmm_predictions(
                market_data, regime_info, symbol, exchange, timeframe
            )
            predictions["hmm_predictions"] = hmm_predictions

            # Generate ML confidence predictions to enhance existing components
            ml_confidence = await self._generate_ml_confidence_predictions(
                hmm_predictions, analyst_signals, symbol, exchange
            )
            predictions["ml_confidence_predictions"] = ml_confidence

            # Apply confidence calibration
            calibrated_confidence = await self._apply_confidence_calibration(
                ml_confidence, symbol, exchange
            )
            predictions["calibrated_confidence_scores"] = calibrated_confidence

            # Apply optimization weights
            optimization_weights = await self._apply_optimization_weights(
                calibrated_confidence, symbol, exchange
            )
            predictions["optimization_weights"] = optimization_weights

            return predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating tactician predictions: {e}"))
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
        """Generate HMM-based predictions."""
        try:
            predictions = {}
            
            for model_name, model_data in self.hmm_models.items():
                if "model" in model_data and hasattr(model_data["model"], "predict"):
                    try:
                        # Prepare features for prediction
                        features = self._prepare_features_for_prediction(
                            market_data, regime_info
                        )
                        
                        # Generate prediction
                        raw_prediction = model_data["model"].predict(features)
                        
                        # Process prediction
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
        context="generating analyst enhanced predictions",
    )
    async def _generate_analyst_enhanced_predictions(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """Generate analyst enhanced predictions."""
        try:
            predictions = {}
            
            current_regime = regime_info.get("regime", "default")
            regime_models = self.analyst_enhanced_models.get(current_regime, {})
            
            for model_name, model_data in regime_models.items():
                if "model" in model_data and hasattr(model_data["model"], "predict"):
                    try:
                        # Prepare features for prediction
                        features = self._prepare_features_for_prediction(
                            market_data, regime_info
                        )
                        
                        # Generate prediction
                        raw_prediction = model_data["model"].predict(features)
                        
                        # Process prediction
                        processed_prediction = self._process_analyst_prediction(
                            raw_prediction, model_data, model_name
                        )
                        
                        predictions[f"analyst_{current_regime}_{model_name}"] = processed_prediction
                        
                    except Exception as e:
                        self.logger.warning(warning(f"⚠️ Failed to generate analyst prediction for {model_name}: {e}"))

            return predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating analyst enhanced predictions: {e}"))
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
            # This balances different ML model types (Analyst vs Tactician ML models)
            analyst_ml_weight = self.optimized_params.get("analyst_ml_weight", 0.6)
            tactician_ml_weight = self.optimized_params.get("tactician_ml_weight", 0.4)
            
            # Analyst ML models focus on market analysis and regime detection
            # Tactician ML models focus on execution and position sizing
            weighted_ml_confidence = (avg_confidence * tactician_ml_weight) + (analyst_confidence * analyst_ml_weight)
            
            ml_confidence["aggregate_ml_confidence"] = {
                "hmm_avg_confidence": avg_confidence,
                "analyst_confidence": analyst_confidence,
                "weighted_ml_confidence": weighted_ml_confidence,
                "analyst_ml_weight": analyst_ml_weight,
                "tactician_ml_weight": tactician_ml_weight,
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
            
            # Find relevant calibration data
            calibration_key = f"{exchange}_{symbol}_calibration_results"
            calibration_data = self.calibration_results.get(calibration_key, {})
            
            for prediction_name, prediction_data in predictions.items():
                calibrated_prediction = self._calibrate_prediction(
                    prediction_data, calibration_data, prediction_name
                )
                calibrated_predictions[prediction_name] = calibrated_prediction

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
        """Generate final confidence scores."""
        try:
            confidence_scores = {}
            
            for prediction_name, prediction_data in calibrated_predictions.items():
                base_confidence = prediction_data.get("confidence", 0.5)
                weight = optimization_weights.get(prediction_name, 1.0)
                
                # Apply weight to confidence
                weighted_confidence = base_confidence * weight
                normalized_confidence = min(1.0, max(0.0, weighted_confidence))
                
                confidence_scores[prediction_name] = {
                    "normalized_confidence": normalized_confidence,
                    "confidence_level": self._get_confidence_level(normalized_confidence),
                    "base_confidence": base_confidence,
                    "applied_weight": weight
                }

            return confidence_scores

        except Exception as e:
            self.logger.error(error(f"❌ Error generating final confidence scores: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating ML profit predictions",
    )
    @with_tracing_span("generate_ml_profit_predictions")
    @validate_data_quality(validation_level="WARNING")
    async def _generate_ml_profit_predictions(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """Generate ML profit predictions from steps 6-14 models."""
        try:
            predictions = {}
            
            # Generate predictions from different ML model types
            for model_type, models in self.ml_profit_models.items():
                for model_name, model_data in models.items():
                    if "model" in model_data and hasattr(model_data["model"], "predict"):
                        try:
                            # Prepare features for prediction
                            features = self._prepare_features_for_prediction(
                                market_data, regime_info
                            )
                            
                            # Generate prediction
                            raw_prediction = model_data["model"].predict(features)
                            
                            # Process prediction based on model type
                            if model_type == "hmm_profit":
                                processed_prediction = self._process_hmm_profit_prediction(
                                    raw_prediction, model_data, model_name
                                )
                            elif model_type == "analyst_profit":
                                processed_prediction = self._process_analyst_profit_prediction(
                                    raw_prediction, model_data, model_name
                                )
                            elif model_type == "tactician_profit":
                                processed_prediction = self._process_tactician_profit_prediction(
                                    raw_prediction, model_data, model_name
                                )
                            else:  # ensemble_profit
                                processed_prediction = self._process_ensemble_profit_prediction(
                                    raw_prediction, model_data, model_name
                                )
                            
                            predictions[f"{model_type}_{model_name}"] = processed_prediction
                            
                        except Exception as e:
                            self.logger.warning(warning(f"⚠️ Failed to generate ML profit prediction for {model_type}/{model_name}: {e}"))

            return predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating ML profit predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating enhanced confidence scores",
    )
    @with_tracing_span("generate_enhanced_confidence_scores")
    @validate_data_quality(validation_level="WARNING")
    async def _generate_enhanced_confidence_scores(
        self,
        ml_profit_predictions: dict[str, Any],
        market_data: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """
        Generate enhanced confidence scores with barrier analysis.
        
        This function calculates the confidence that price will move AT LEAST by x% 
        in a direction without hitting the barrier in the other direction first.
        """
        try:
            enhanced_confidence = {}
            
            # Extract current price and volatility
            current_price = market_data['close'].iloc[-1]
            price_volatility = market_data['close'].pct_change().std()
            
            # Calculate price movement thresholds
            profit_threshold_pct = self.profit_threshold
            barrier_threshold_pct = self.barrier_threshold
            
            profit_threshold_price = current_price * (1 + profit_threshold_pct)
            barrier_threshold_price = current_price * (1 - barrier_threshold_pct)
            
            for prediction_name, prediction_data in ml_profit_predictions.items():
                try:
                    # Extract prediction components
                    predicted_direction = prediction_data.get("direction", 0)  # -1, 0, 1
                    predicted_magnitude = prediction_data.get("magnitude", 0.0)
                    base_confidence = prediction_data.get("confidence", 0.5)
                    
                    # Calculate enhanced confidence with barrier analysis
                    enhanced_confidence_score = await self._calculate_directional_confidence_with_barriers(
                        predicted_direction=predicted_direction,
                        predicted_magnitude=predicted_magnitude,
                        base_confidence=base_confidence,
                        current_price=current_price,
                        profit_threshold_price=profit_threshold_price,
                        barrier_threshold_price=barrier_threshold_price,
                        price_volatility=price_volatility,
                        prediction_name=prediction_name
                    )
                    
                    enhanced_confidence[prediction_name] = {
                        "enhanced_confidence": enhanced_confidence_score,
                        "base_confidence": base_confidence,
                        "direction": predicted_direction,
                        "magnitude": predicted_magnitude,
                        "profit_threshold": profit_threshold_pct,
                        "barrier_threshold": barrier_threshold_pct,
                        "current_price": current_price,
                        "profit_target": profit_threshold_price,
                        "barrier_price": barrier_threshold_price,
                        "volatility": price_volatility,
                        "calculation_method": "directional_with_barriers"
                    }
                    
                except Exception as e:
                    self.logger.warning(warning(f"⚠️ Failed to calculate enhanced confidence for {prediction_name}: {e}"))
                    enhanced_confidence[prediction_name] = {
                        "enhanced_confidence": 0.5,
                        "base_confidence": prediction_data.get("confidence", 0.5),
                        "error": str(e)
                    }

            return enhanced_confidence

        except Exception as e:
            self.logger.error(error(f"❌ Error generating enhanced confidence scores: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=0.5,
        context="calculating directional confidence with barriers",
    )
    @with_tracing_span("calculate_directional_confidence_with_barriers")
    async def _calculate_directional_confidence_with_barriers(
        self,
        predicted_direction: int,
        predicted_magnitude: float,
        base_confidence: float,
        current_price: float,
        profit_threshold_price: float,
        barrier_threshold_price: float,
        price_volatility: float,
        prediction_name: str
    ) -> float:
        """
        Calculate confidence that price will move AT LEAST by x% in a direction 
        without hitting the barrier in the other direction first.
        
        This is the enhanced confidence calculation function that considers:
        1. Directional probability
        2. Magnitude probability
        3. Barrier avoidance probability
        4. Volatility-adjusted confidence
        """
        try:
            if predicted_direction == 0:
                return 0.5  # Neutral direction
            
            # Calculate directional probability
            directional_prob = self._calculate_directional_probability(
                predicted_direction, base_confidence, price_volatility
            )
            
            # Calculate magnitude probability (probability of reaching profit target)
            magnitude_prob = self._calculate_magnitude_probability(
                predicted_magnitude, profit_threshold_price, current_price, price_volatility
            )
            
            # Calculate barrier avoidance probability
            barrier_avoidance_prob = self._calculate_barrier_avoidance_probability(
                predicted_direction, barrier_threshold_price, current_price, price_volatility
            )
            
            # Combine probabilities using Bayesian approach
            # P(success) = P(direction) * P(magnitude) * P(no_barrier)
            combined_probability = directional_prob * magnitude_prob * barrier_avoidance_prob
            
            # Apply volatility adjustment
            volatility_adjustment = self._calculate_volatility_adjustment(price_volatility)
            adjusted_confidence = combined_probability * volatility_adjustment
            
            # Ensure confidence is within bounds
            final_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            self.logger.debug(f"Enhanced confidence calculation for {prediction_name}:")
            self.logger.debug(f"  Directional prob: {directional_prob:.4f}")
            self.logger.debug(f"  Magnitude prob: {magnitude_prob:.4f}")
            self.logger.debug(f"  Barrier avoidance prob: {barrier_avoidance_prob:.4f}")
            self.logger.debug(f"  Volatility adjustment: {volatility_adjustment:.4f}")
            self.logger.debug(f"  Final confidence: {final_confidence:.4f}")
            
            return final_confidence

        except Exception as e:
            self.logger.error(error(f"❌ Error calculating directional confidence with barriers: {e}"))
            return base_confidence

    def _calculate_directional_probability(
        self,
        predicted_direction: int,
        base_confidence: float,
        price_volatility: float
    ) -> float:
        """Calculate probability of correct direction prediction."""
        try:
            # Base directional probability from model confidence
            base_directional_prob = base_confidence
            
            # Adjust for volatility (higher volatility = lower directional confidence)
            volatility_factor = 1.0 / (1.0 + price_volatility * 10)  # Scale volatility impact
            
            # Adjust for direction strength
            direction_strength = abs(predicted_direction)
            direction_factor = min(1.0, direction_strength)
            
            # Combine factors
            directional_probability = base_directional_prob * volatility_factor * direction_factor
            
            return max(0.1, min(0.95, directional_probability))  # Bounded between 0.1 and 0.95
            
        except Exception as e:
            self.logger.error(error(f"❌ Error calculating directional probability: {e}"))
            return 0.5

    def _calculate_magnitude_probability(
        self,
        predicted_magnitude: float,
        profit_threshold_price: float,
        current_price: float,
        price_volatility: float
    ) -> float:
        """Calculate probability of reaching the profit target."""
        try:
            # Calculate required price movement
            required_movement = abs(profit_threshold_price - current_price) / current_price
            
            # Use predicted magnitude as base probability
            if predicted_magnitude > 0:
                # Normalize predicted magnitude to probability
                magnitude_prob = min(1.0, predicted_magnitude / required_movement)
            else:
                magnitude_prob = 0.1  # Low probability if no magnitude prediction
            
            # Adjust for volatility (higher volatility = higher chance of large moves)
            volatility_boost = min(0.3, price_volatility * 5)  # Cap volatility boost at 30%
            adjusted_prob = magnitude_prob + volatility_boost
            
            return max(0.05, min(0.9, adjusted_prob))  # Bounded between 0.05 and 0.9
            
        except Exception as e:
            self.logger.error(error(f"❌ Error calculating magnitude probability: {e}"))
            return 0.5

    def _calculate_barrier_avoidance_probability(
        self,
        predicted_direction: int,
        barrier_threshold_price: float,
        current_price: float,
        price_volatility: float
    ) -> float:
        """Calculate probability of avoiding the barrier price."""
        try:
            # Calculate distance to barrier
            barrier_distance = abs(barrier_threshold_price - current_price) / current_price
            
            # Base probability of avoiding barrier (further barrier = higher probability)
            base_avoidance_prob = min(0.95, barrier_distance * 10)  # Scale distance to probability
            
            # Adjust for volatility (higher volatility = lower barrier avoidance probability)
            volatility_penalty = min(0.4, price_volatility * 8)  # Cap volatility penalty at 40%
            adjusted_prob = base_avoidance_prob - volatility_penalty
            
            # Direction-specific adjustment
            if predicted_direction > 0:  # Bullish prediction
                # More likely to avoid downside barrier if bullish
                direction_boost = 0.1
            elif predicted_direction < 0:  # Bearish prediction
                # More likely to avoid upside barrier if bearish
                direction_boost = 0.1
            else:
                direction_boost = 0.0
            
            final_prob = adjusted_prob + direction_boost
            
            return max(0.1, min(0.95, final_prob))  # Bounded between 0.1 and 0.95
            
        except Exception as e:
            self.logger.error(error(f"❌ Error calculating barrier avoidance probability: {e}"))
            return 0.7

    def _calculate_volatility_adjustment(self, price_volatility: float) -> float:
        """Calculate volatility adjustment factor for confidence."""
        try:
            # Higher volatility generally reduces confidence in predictions
            # Use a sigmoid-like function to smooth the adjustment
            volatility_factor = 1.0 / (1.0 + np.exp(price_volatility * 20 - 5))
            
            # Ensure adjustment is reasonable
            return max(0.5, min(1.2, volatility_factor))
            
        except Exception as e:
            self.logger.error(error(f"❌ Error calculating volatility adjustment: {e}"))
            return 1.0

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating barrier analysis",
    )
    @with_tracing_span("generate_barrier_analysis")
    async def _generate_barrier_analysis(
        self,
        ml_profit_predictions: dict[str, Any],
        market_data: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """Generate barrier analysis for risk management."""
        try:
            barrier_analysis = {}
            
            current_price = market_data['close'].iloc[-1]
            price_volatility = market_data['close'].pct_change().std()
            
            for prediction_name, prediction_data in ml_profit_predictions.items():
                try:
                    # Calculate barrier metrics
                    barrier_metrics = self._calculate_barrier_metrics(
                        prediction_data, current_price, price_volatility
                    )
                    
                    barrier_analysis[prediction_name] = barrier_metrics
                    
                except Exception as e:
                    self.logger.warning(warning(f"⚠️ Failed to calculate barrier metrics for {prediction_name}: {e}"))

            return barrier_analysis

        except Exception as e:
            self.logger.error(error(f"❌ Error generating barrier analysis: {e}"))
            return {}

    def _calculate_barrier_metrics(
        self,
        prediction_data: dict[str, Any],
        current_price: float,
        price_volatility: float
    ) -> dict[str, Any]:
        """Calculate barrier-related metrics for risk management."""
        try:
            predicted_direction = prediction_data.get("direction", 0)
            predicted_magnitude = prediction_data.get("magnitude", 0.0)
            
            # Calculate profit and barrier levels
            profit_threshold = self.profit_threshold
            barrier_threshold = self.barrier_threshold
            
            if predicted_direction > 0:  # Bullish
                profit_target = current_price * (1 + profit_threshold)
                barrier_level = current_price * (1 - barrier_threshold)
            elif predicted_direction < 0:  # Bearish
                profit_target = current_price * (1 - profit_threshold)
                barrier_level = current_price * (1 + barrier_threshold)
            else:  # Neutral
                profit_target = current_price
                barrier_level = current_price
            
            # Calculate distances
            profit_distance = abs(profit_target - current_price) / current_price
            barrier_distance = abs(barrier_level - current_price) / current_price
            
            # Calculate risk-reward ratio
            risk_reward_ratio = profit_distance / barrier_distance if barrier_distance > 0 else 0
            
            # Calculate probability-weighted expected value
            confidence = prediction_data.get("confidence", 0.5)
            expected_value = (profit_distance * confidence) - (barrier_distance * (1 - confidence))
            
            return {
                "profit_target": profit_target,
                "barrier_level": barrier_level,
                "profit_distance": profit_distance,
                "barrier_distance": barrier_distance,
                "risk_reward_ratio": risk_reward_ratio,
                "expected_value": expected_value,
                "direction": predicted_direction,
                "confidence": confidence,
                "volatility": price_volatility
            }
            
        except Exception as e:
            self.logger.error(error(f"❌ Error calculating barrier metrics: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating regime predictions",
    )
    @with_tracing_span("generate_regime_predictions")
    async def _generate_regime_predictions(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """Generate regime-based predictions."""
        try:
            regime_predictions = {}
            
            current_regime = regime_info.get("regime", "unknown")
            regime_confidence = regime_info.get("confidence", 0.5)
            
            # Get regime-specific models
            regime_models = self.analyst_enhanced_models.get(current_regime, {})
            
            for model_name, model_data in regime_models.items():
                try:
                    features = self._prepare_features_for_prediction(market_data, regime_info)
                    raw_prediction = model_data["model"].predict(features)
                    
                    processed_prediction = self._process_regime_prediction(
                        raw_prediction, model_data, model_name, current_regime
                    )
                    
                    regime_predictions[f"regime_{current_regime}_{model_name}"] = processed_prediction
                    
                except Exception as e:
                    self.logger.warning(warning(f"⚠️ Failed to generate regime prediction for {model_name}: {e}"))

            return regime_predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating regime predictions: {e}"))
            return {}

    def _process_hmm_profit_prediction(
        self,
        raw_prediction: Any,
        model_data: dict[str, Any],
        model_name: str
    ) -> dict[str, Any]:
        """Process HMM profit prediction."""
        try:
            if isinstance(raw_prediction, np.ndarray):
                prediction_value = float(raw_prediction[0]) if raw_prediction.size > 0 else 0.0
            else:
                prediction_value = float(raw_prediction)

            # Extract direction and magnitude from prediction
            direction = 1 if prediction_value > 0 else (-1 if prediction_value < 0 else 0)
            magnitude = abs(prediction_value)

            return {
                "prediction": prediction_value,
                "direction": direction,
                "magnitude": magnitude,
                "confidence": model_data.get("confidence", 0.5),
                "model_type": "hmm_profit",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error processing HMM profit prediction: {e}"))
            return {"prediction": 0.0, "direction": 0, "magnitude": 0.0, "confidence": 0.0, "model_type": "hmm_profit", "model_name": model_name}

    def _process_analyst_profit_prediction(
        self,
        raw_prediction: Any,
        model_data: dict[str, Any],
        model_name: str
    ) -> dict[str, Any]:
        """Process analyst profit prediction."""
        try:
            if isinstance(raw_prediction, np.ndarray):
                prediction_value = float(raw_prediction[0]) if raw_prediction.size > 0 else 0.0
            else:
                prediction_value = float(raw_prediction)

            # Extract direction and magnitude from prediction
            direction = 1 if prediction_value > 0 else (-1 if prediction_value < 0 else 0)
            magnitude = abs(prediction_value)

            return {
                "prediction": prediction_value,
                "direction": direction,
                "magnitude": magnitude,
                "confidence": model_data.get("confidence", 0.5),
                "model_type": "analyst_profit",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error processing analyst profit prediction: {e}"))
            return {"prediction": 0.0, "direction": 0, "magnitude": 0.0, "confidence": 0.0, "model_type": "analyst_profit", "model_name": model_name}

    def _process_tactician_profit_prediction(
        self,
        raw_prediction: Any,
        model_data: dict[str, Any],
        model_name: str
    ) -> dict[str, Any]:
        """Process tactician profit prediction."""
        try:
            if isinstance(raw_prediction, np.ndarray):
                prediction_value = float(raw_prediction[0]) if raw_prediction.size > 0 else 0.0
            else:
                prediction_value = float(raw_prediction)

            # Extract direction and magnitude from prediction
            direction = 1 if prediction_value > 0 else (-1 if prediction_value < 0 else 0)
            magnitude = abs(prediction_value)

            return {
                "prediction": prediction_value,
                "direction": direction,
                "magnitude": magnitude,
                "confidence": model_data.get("confidence", 0.5),
                "model_type": "tactician_profit",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error processing tactician profit prediction: {e}"))
            return {"prediction": 0.0, "direction": 0, "magnitude": 0.0, "confidence": 0.0, "model_type": "tactician_profit", "model_name": model_name}

    def _process_ensemble_profit_prediction(
        self,
        raw_prediction: Any,
        model_data: dict[str, Any],
        model_name: str
    ) -> dict[str, Any]:
        """Process ensemble profit prediction."""
        try:
            if isinstance(raw_prediction, np.ndarray):
                prediction_value = float(raw_prediction[0]) if raw_prediction.size > 0 else 0.0
            else:
                prediction_value = float(raw_prediction)

            # Extract direction and magnitude from prediction
            direction = 1 if prediction_value > 0 else (-1 if prediction_value < 0 else 0)
            magnitude = abs(prediction_value)

            return {
                "prediction": prediction_value,
                "direction": direction,
                "magnitude": magnitude,
                "confidence": model_data.get("confidence", 0.5),
                "model_type": "ensemble_profit",
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error processing ensemble profit prediction: {e}"))
            return {"prediction": 0.0, "direction": 0, "magnitude": 0.0, "confidence": 0.0, "model_type": "ensemble_profit", "model_name": model_name}

    def _process_regime_prediction(
        self,
        raw_prediction: Any,
        model_data: dict[str, Any],
        model_name: str,
        regime: str
    ) -> dict[str, Any]:
        """Process regime prediction."""
        try:
            if isinstance(raw_prediction, np.ndarray):
                prediction_value = float(raw_prediction[0]) if raw_prediction.size > 0 else 0.0
            else:
                prediction_value = float(raw_prediction)

            return {
                "prediction": prediction_value,
                "confidence": model_data.get("confidence", 0.5),
                "model_type": "regime",
                "model_name": model_name,
                "regime": regime,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error processing regime prediction: {e}"))
            return {"prediction": 0.0, "confidence": 0.0, "model_type": "regime", "model_name": model_name, "regime": regime}

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

    # REMOVED: RSI and MACD calculation methods
    # These technical indicators should be handled by the ML models in steps 6-14
    # The integrator should focus on integrating predictions, not generating features