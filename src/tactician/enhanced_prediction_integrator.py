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


class TacticianEnhancedPredictionIntegrator:
    """
    Enhanced Prediction Integrator for Tactician that integrates price and confidence predictions
    from the enhanced training manager steps 6-14, with focus on tactician-specific predictions.
    
    This component loads and integrates:
    - HMM-based model predictions (step 6-8)
    - Tactician specialist training predictions (step 9)
    - Tactician labeling predictions (step 10)
    - Confidence calibration results (step 11)
    - Final parameter optimization results (step 12-14)
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
        self.tactician_specialist_models: dict[str, Any] = {}
        self.tactician_labeling_models: dict[str, Any] = {}
        self.calibration_results: dict[str, Any] = {}
        self.optimization_results: dict[str, Any] = {}

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

            # Load tactician specialist models (step 9)
            await self._load_tactician_specialist_models()

            # Load tactician labeling models (step 10)
            await self._load_tactician_labeling_models()

            # Load confidence calibration results (step 11)
            await self._load_calibration_results()

            # Load optimization results (step 12-14)
            await self._load_optimization_results()

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
        default_return={},
        context="generating tactician enhanced predictions",
    )
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
        Generate enhanced tactician predictions using all loaded models and calibration.

        Args:
            market_data: Market data for prediction
            regime_info: Current regime information
            analyst_signals: Analyst signals and predictions
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            dict: Enhanced tactician predictions with confidence scores
        """
        try:
            if not self.is_initialized:
                self.logger.error(error("❌ Tactician Enhanced Prediction Integrator not initialized"))
                return {}

            predictions = {
                "entry_predictions": {},
                "exit_predictions": {},
                "position_sizing_predictions": {},
                "risk_management_predictions": {},
                "confidence_scores": {},
                "tactician_labels": {},
                "calibrated_predictions": {},
                "optimization_weights": {},
                "timestamp": datetime.now().isoformat()
            }

            # Generate HMM-based predictions
            hmm_predictions = await self._generate_hmm_predictions(
                market_data, regime_info, symbol, exchange, timeframe
            )
            predictions["entry_predictions"].update(hmm_predictions)

            # Generate tactician specialist predictions
            specialist_predictions = await self._generate_tactician_specialist_predictions(
                market_data, regime_info, analyst_signals, symbol, exchange, timeframe
            )
            predictions["entry_predictions"].update(specialist_predictions)

            # Generate tactician labeling predictions
            labeling_predictions = await self._generate_tactician_labeling_predictions(
                market_data, regime_info, analyst_signals, symbol, exchange, timeframe
            )
            predictions["tactician_labels"] = labeling_predictions

            # Generate position sizing predictions
            sizing_predictions = await self._generate_position_sizing_predictions(
                predictions["entry_predictions"], analyst_signals, symbol, exchange
            )
            predictions["position_sizing_predictions"] = sizing_predictions

            # Generate risk management predictions
            risk_predictions = await self._generate_risk_management_predictions(
                predictions["entry_predictions"], predictions["tactician_labels"], symbol, exchange
            )
            predictions["risk_management_predictions"] = risk_predictions

            # Apply confidence calibration
            calibrated_predictions = await self._apply_confidence_calibration(
                predictions["entry_predictions"], symbol, exchange
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

            # Generate exit predictions based on entry predictions and market conditions
            exit_predictions = await self._generate_exit_predictions(
                predictions["entry_predictions"], predictions["confidence_scores"], market_data
            )
            predictions["exit_predictions"] = exit_predictions

            return predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating tactician enhanced predictions: {e}"))
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
        context="generating tactician specialist predictions",
    )
    async def _generate_tactician_specialist_predictions(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        analyst_signals: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """Generate predictions using tactician specialist models."""
        try:
            predictions = {}
            
            current_regime = regime_info.get("regime", "default")
            regime_models = self.tactician_specialist_models.get(current_regime, {})
            
            for model_name, model_data in regime_models.items():
                if "model" in model_data and hasattr(model_data["model"], "predict"):
                    try:
                        # Prepare features for prediction with analyst signals
                        features = self._prepare_features_with_analyst_signals(
                            market_data, regime_info, analyst_signals
                        )
                        
                        # Generate prediction
                        raw_prediction = model_data["model"].predict(features)
                        
                        # Apply model-specific post-processing
                        processed_prediction = self._process_tactician_specialist_prediction(
                            raw_prediction, model_data, model_name
                        )
                        
                        predictions[f"tactician_specialist_{current_regime}_{model_name}"] = processed_prediction
                        
                    except Exception as e:
                        self.logger.warning(warning(f"⚠️ Failed to generate tactician specialist prediction for {model_name}: {e}"))

            return predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating tactician specialist predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating tactician labeling predictions",
    )
    async def _generate_tactician_labeling_predictions(
        self,
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        analyst_signals: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """Generate tactician labeling predictions."""
        try:
            predictions = {}
            
            for labeling_name, labeling_data in self.tactician_labeling_models.items():
                try:
                    # Prepare features for labeling
                    features = self._prepare_features_with_analyst_signals(
                        market_data, regime_info, analyst_signals
                    )
                    
                    # Generate labeling prediction
                    if "model" in labeling_data and hasattr(labeling_data["model"], "predict"):
                        raw_prediction = labeling_data["model"].predict(features)
                        
                        processed_prediction = self._process_tactician_labeling_prediction(
                            raw_prediction, labeling_data, labeling_name
                        )
                        
                        predictions[f"tactician_labeling_{labeling_name}"] = processed_prediction
                    
                except Exception as e:
                    self.logger.warning(warning(f"⚠️ Failed to generate tactician labeling prediction for {labeling_name}: {e}"))

            return predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating tactician labeling predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating position sizing predictions",
    )
    async def _generate_position_sizing_predictions(
        self,
        entry_predictions: dict[str, Any],
        analyst_signals: dict[str, Any],
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """Generate position sizing predictions based on entry predictions and analyst signals."""
        try:
            sizing_predictions = {}
            
            # Calculate aggregate confidence from entry predictions
            total_confidence = 0.0
            valid_predictions = 0
            
            for prediction_name, prediction_data in entry_predictions.items():
                confidence = prediction_data.get("confidence", 0.0)
                if confidence > 0:
                    total_confidence += confidence
                    valid_predictions += 1
            
            avg_confidence = total_confidence / max(valid_predictions, 1)
            
            # Calculate position size based on confidence
            base_position_size = 0.1  # 10% base position size
            confidence_multiplier = min(avg_confidence * 2, 2.0)  # Max 2x multiplier
            position_size = base_position_size * confidence_multiplier
            
            # Adjust based on analyst signals
            analyst_confidence = analyst_signals.get("confidence", 0.5)
            analyst_multiplier = 0.5 + (analyst_confidence * 0.5)  # 0.5x to 1.0x multiplier
            
            final_position_size = position_size * analyst_multiplier
            
            sizing_predictions["position_size"] = {
                "base_size": base_position_size,
                "confidence_multiplier": confidence_multiplier,
                "analyst_multiplier": analyst_multiplier,
                "final_size": final_position_size,
                "avg_confidence": avg_confidence,
                "analyst_confidence": analyst_confidence
            }
            
            return sizing_predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating position sizing predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating risk management predictions",
    )
    async def _generate_risk_management_predictions(
        self,
        entry_predictions: dict[str, Any],
        tactician_labels: dict[str, Any],
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """Generate risk management predictions."""
        try:
            risk_predictions = {}
            
            # Calculate risk metrics based on predictions
            total_risk_score = 0.0
            prediction_count = 0
            
            for prediction_name, prediction_data in entry_predictions.items():
                confidence = prediction_data.get("confidence", 0.0)
                prediction_value = prediction_data.get("prediction", 0.0)
                
                # Risk score is inverse of confidence
                risk_score = 1.0 - confidence
                total_risk_score += risk_score
                prediction_count += 1
            
            avg_risk_score = total_risk_score / max(prediction_count, 1)
            
            # Calculate stop loss and take profit levels
            base_stop_loss = 0.02  # 2% base stop loss
            base_take_profit = 0.04  # 4% base take profit
            
            # Adjust based on risk score
            risk_multiplier = 1.0 + avg_risk_score  # 1.0x to 2.0x multiplier
            
            stop_loss = base_stop_loss * risk_multiplier
            take_profit = base_take_profit * risk_multiplier
            
            risk_predictions["risk_management"] = {
                "avg_risk_score": avg_risk_score,
                "stop_loss_pct": stop_loss,
                "take_profit_pct": take_profit,
                "risk_multiplier": risk_multiplier,
                "max_position_size": max(0.05, 0.2 - (avg_risk_score * 0.15))  # 5% to 20% based on risk
            }
            
            return risk_predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating risk management predictions: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating exit predictions",
    )
    async def _generate_exit_predictions(
        self,
        entry_predictions: dict[str, Any],
        confidence_scores: dict[str, Any],
        market_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Generate exit predictions based on entry predictions and market conditions."""
        try:
            exit_predictions = {}
            
            # Calculate aggregate exit signal
            total_exit_signal = 0.0
            valid_signals = 0
            
            for prediction_name, prediction_data in entry_predictions.items():
                confidence = confidence_scores.get(prediction_name, {}).get("normalized_confidence", 0.0)
                prediction_value = prediction_data.get("prediction", 0.0)
                
                # Exit signal is inverse of entry signal
                exit_signal = -prediction_value * confidence
                total_exit_signal += exit_signal
                valid_signals += 1
            
            avg_exit_signal = total_exit_signal / max(valid_signals, 1)
            
            # Determine exit conditions
            exit_conditions = {
                "exit_signal": avg_exit_signal,
                "should_exit": abs(avg_exit_signal) > self.exit_threshold,
                "exit_direction": "sell" if avg_exit_signal > 0 else "buy",
                "exit_confidence": abs(avg_exit_signal),
                "market_conditions": self._analyze_market_conditions(market_data)
            }
            
            exit_predictions["exit_conditions"] = exit_conditions
            
            return exit_predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating exit predictions: {e}"))
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