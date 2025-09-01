"""
Unified Regime Intelligence Runtime

Integrates S/R level monitoring with regime analysis for comprehensive market intelligence.
"""

import os
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.training.steps.step05_5_unified_regime_intelligence import (
    UnifiedRegimeIntelligenceStep,
)
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor

logger = system_logger.getChild("UnifiedRegimeIntelligenceRuntime")


class UnifiedRegimeIntelligenceRuntime:
    """Runtime for unified regime intelligence with S/R level monitoring."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logger

        # Initialize SRBreakoutPredictor for S/R monitoring with optimized parameters
        sr_config = config.copy()
        sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
        sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
        self.sr_predictor = SRBreakoutPredictor(sr_config)
        self.sr_outcome_model = None

        # S/R monitoring configuration
        self.sr_monitoring_config = config.get("sr_monitoring", {})
        self.enable_sr_monitoring = self.sr_monitoring_config.get(
            "enable_sr_monitoring", True
        )
        self.sr_alert_threshold = self.sr_monitoring_config.get(
            "sr_alert_threshold", 0.7
        )

        # Initialize unified step for integrated predictions
        self.unified_step = UnifiedRegimeIntelligenceStep(config)

        # Model components
        self.model = None
        self.label_encoders = {}
        self.scaler = None

        # Configuration
        self.timeframes = config.get(
            "timeframes", ["5m", "15m", "30m"]
        )  # Less noisy for regime detection
        self.sequence_length = config.get("sequence_length", 20)
        self.artifacts_dir = config.get(
            "artifacts_dir", "checkpoints/unified_regime_intelligence"
        )

        # Runtime state
        self.is_initialized = False
        self.current_regime = None
        self.regime_history = []
        self.transition_probability = 0.0
        self.tpsl_direction = None

        # Expert activation thresholds
        self.regime_confidence_threshold = config.get(
            "regime_confidence_threshold", 0.7
        )
        self.transition_threshold = config.get("transition_threshold", 0.6)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="unified regime intelligence initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the unified regime intelligence runtime."""
        try:
            self.logger.info("Initializing Unified Regime Intelligence Runtime...")

            # Initialize SR predictor
            sr_init_success = await self.sr_predictor.initialize()
            if not sr_init_success:
                self.logger.warning("Failed to initialize SRBreakoutPredictor")

            # Load S/R outcome model if available
            await self._load_sr_outcome_model()

            # Initialize unified step (includes SRBreakoutPredictor)
            if not await self.unified_step.initialize():
                self.logger.error("Failed to initialize unified step")
                return False

            # Load model
            if not await self._load_model():
                self.logger.error("Failed to load unified regime intelligence model")
                return False

            # Load label encoders
            if not await self._load_label_encoders():
                self.logger.error("Failed to load label encoders")
                return False

            # Load configuration
            if not await self._load_configuration():
                self.logger.error("Failed to load configuration")
                return False

            self.is_initialized = True
            self.logger.info(
                "✅ Unified Regime Intelligence Runtime initialized successfully"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to initialize Unified Regime Intelligence Runtime: {e}"
            )
            return False

    async def _load_model(self) -> bool:
        """Load the trained unified regime intelligence model."""
        try:
            model_path = os.path.join(self.artifacts_dir, "final_model.pth")
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False

            # Load model configuration
            config_path = os.path.join(self.artifacts_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    model_config = json.load(f)
            else:
                model_config = self.config

            # Initialize model
            self.model = MultiTimeframeHMMEncoder(model_config)

            # Load trained weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.model.eval()

            self.logger.info("Unified regime intelligence model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    async def _load_label_encoders(self) -> bool:
        """Load label encoders for regime, transition, and TPSL predictions."""
        try:
            encoder_names = ["regime", "intensity", "transition", "tpsl"]

            for name in encoder_names:
                encoder_path = os.path.join(self.artifacts_dir, f"{name}_encoder.pkl")
                if os.path.exists(encoder_path):
                    with open(encoder_path, "rb") as f:
                        self.label_encoders[name] = pickle.load(f)
                    self.logger.info(f"Loaded {name} label encoder")
                else:
                    self.logger.warning(f"Label encoder not found: {encoder_path}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading label encoders: {e}")
            return False

    async def _load_configuration(self) -> bool:
        """Load runtime configuration."""
        try:
            config_path = os.path.join(self.artifacts_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    saved_config = json.load(f)

                # Update configuration with saved values
                self.timeframes = saved_config.get("timeframes", self.timeframes)
                self.sequence_length = saved_config.get(
                    "sequence_length", self.sequence_length
                )

                self.logger.info("Runtime configuration loaded successfully")

            return True

        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False

    async def _load_sr_outcome_model(self) -> None:
        """Load the trained S/R outcome model."""
        try:
            model_path = self.config.get(
                "sr_outcome_model_path", "models/sr_outcome/ensemble_model.pkl"
            )

            import os
            import pickle

            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.sr_outcome_model = pickle.load(f)

                # Load supporting artifacts
                scaler_path = model_path.replace(
                    "ensemble_model.pkl", "sr_outcome_scaler.pkl"
                )
                encoder_path = model_path.replace(
                    "ensemble_model.pkl", "sr_outcome_encoder.pkl"
                )

                if os.path.exists(scaler_path):
                    with open(scaler_path, "rb") as f:
                        self.sr_scaler = pickle.load(f)

                if os.path.exists(encoder_path):
                    with open(encoder_path, "rb") as f:
                        self.sr_encoder = pickle.load(f)

                self.logger.info("✅ S/R outcome model loaded successfully")
            else:
                self.logger.warning(f"S/R outcome model not found at {model_path}")

        except Exception as e:
            self.logger.error(f"Error loading S/R outcome model: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="market analysis with S/R monitoring",
    )
    async def analyze_market_with_sr_monitoring(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        regime_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform market analysis with S/R level monitoring.

        Args:
            market_data: Recent market data
            current_price: Current market price
            regime_analysis: Existing regime analysis results

        Returns:
            dict: Enhanced analysis with S/R monitoring results
        """
        try:
            # Perform base regime analysis
            analysis_result = {
                **regime_analysis,
                "sr_monitoring": {},
                "sr_opportunity_detected": False,
            }

            if not self.enable_sr_monitoring:
                return analysis_result

            # Check S/R proximity using centralized logic
            sr_context = await self.sr_predictor.get_sr_context(
                market_data=market_data, current_price=current_price
            )
            is_near_sr = self.sr_predictor.is_near_sr_level(
                current_price=current_price, sr_context=sr_context
            )

            if is_near_sr:
                # Get detailed S/R proximity information
                sr_proximity_details = self.sr_predictor.get_sr_proximity_details(
                    current_price=current_price, sr_context=sr_context
                )

                # Predict S/R outcome using centralized logic
                sr_outcome = await self.sr_predictor.predict_sr_outcome(
                    market_data=market_data, current_price=current_price, sr_context=sr_context
                )

                # Check if opportunity meets confidence threshold
                opportunity_detected = (
                    sr_outcome.get("confidence", 0) >= self.sr_alert_threshold
                )

                analysis_result["sr_monitoring"] = {
                    "is_near_sr_level": True,
                    "sr_proximity_details": sr_proximity_details,
                    "sr_outcome": sr_outcome,
                    "opportunity_detected": opportunity_detected,
                    "recommendation": self._generate_sr_recommendation(
                        sr_outcome, sr_proximity_details
                    ),
                }

                analysis_result["sr_opportunity_detected"] = opportunity_detected

                if opportunity_detected:
                    self.logger.info(
                        f"🚨 S/R Opportunity Detected: {sr_outcome.get('outcome', 'unknown')} "
                        f"(confidence: {sr_outcome.get('confidence', 0):.2f})"
                    )
            else:
                analysis_result["sr_monitoring"] = {
                    "is_near_sr_level": False,
                    "sr_proximity_details": {},
                    "sr_outcome": None,
                    "opportunity_detected": False,
                    "recommendation": "No S/R opportunity - not near significant levels",
                }

            return analysis_result

        except Exception as e:
            self.logger.error(f"Error in market analysis with S/R monitoring: {e}")
            return {
                **regime_analysis,
                "sr_monitoring": {
                    "error": f"S/R monitoring failed: {e}",
                    "is_near_sr_level": False,
                    "opportunity_detected": False,
                },
                "sr_opportunity_detected": False,
            }

    def _generate_sr_recommendation(
        self, sr_outcome: dict[str, Any], sr_proximity_details: dict[str, Any]
    ) -> str:
        """Generate S/R recommendation for the Tactician."""
        try:
            outcome = sr_outcome.get("outcome", "consolidation")
            confidence = sr_outcome.get("confidence", 0)

            # Get S/R context to determine position direction
            sr_context = sr_outcome.get("sr_context", {})
            current_price = sr_context.get("current_price", 0)
            nearest_resistance = sr_context.get("nearest_resistance", current_price)
            nearest_support = sr_context.get("nearest_support", current_price)

            distance_to_resistance = (
                abs(current_price - nearest_resistance) / current_price
            )
            distance_to_support = abs(current_price - nearest_support) / current_price

            if outcome == "breakout" and confidence >= 0.8:
                if distance_to_resistance < distance_to_support:
                    return "STRONG_BREAKOUT_SIGNAL - Breaking out from RESISTANCE -> SHORT position with tight stops"
                else:
                    return "STRONG_BREAKOUT_SIGNAL - Breaking out from SUPPORT -> LONG position with tight stops"
            elif outcome == "breakout" and confidence >= 0.6:
                if distance_to_resistance < distance_to_support:
                    return "BREAKOUT_LIKELY - Breaking out from RESISTANCE -> Monitor for SHORT entry confirmation"
                else:
                    return "BREAKOUT_LIKELY - Breaking out from SUPPORT -> Monitor for LONG entry confirmation"
            elif outcome == "rebounce" and confidence >= 0.8:
                if distance_to_resistance < distance_to_support:
                    return "STRONG_REBOUNCE_SIGNAL - Rebouncing from RESISTANCE -> LONG position with tight stops"
                else:
                    return "STRONG_REBOUNCE_SIGNAL - Rebouncing from SUPPORT -> SHORT position with tight stops"
            elif outcome == "rebounce" and confidence >= 0.6:
                if distance_to_resistance < distance_to_support:
                    return "REBOUNCE_LIKELY - Rebouncing from RESISTANCE -> Monitor for LONG entry confirmation"
                else:
                    return "REBOUNCE_LIKELY - Rebouncing from SUPPORT -> Monitor for SHORT entry confirmation"
            elif outcome == "consolidation" and confidence >= 0.7:
                return "CONSOLIDATION_EXPECTED - Avoid directional trades, consider range strategies"
            else:
                return "UNCERTAIN_OUTCOME - Low confidence, wait for clearer signals"

        except Exception as e:
            self.logger.error(f"Error generating S/R recommendation: {e}")
            return "ERROR_GENERATING_RECOMMENDATION"

    @handle_errors(
        exceptions=(Exception,), default_return={}, context="S/R opportunity alert"
    )
    async def get_sr_opportunity_alert(
        self, market_data: pd.DataFrame, current_price: float
    ) -> dict[str, Any]:
        """
        Get S/R opportunity alert for the Tactician.

        Args:
            market_data: Recent market data
            current_price: Current market price

        Returns:
            dict: S/R opportunity alert with actionable information
        """
        try:
            if not self.enable_sr_monitoring:
                return {"opportunity_detected": False}

            # Get S/R context and outcome
            sr_context = await self.sr_predictor.get_sr_context(
                market_data=market_data, current_price=current_price
            )
            is_near_sr = self.sr_predictor.is_near_sr_level(
                current_price=current_price, sr_context=sr_context
            )

            if not is_near_sr:
                return {"opportunity_detected": False}

            # Predict S/R outcome using centralized logic
            sr_outcome = await self.sr_predictor.predict_sr_outcome(
                market_data=market_data, current_price=current_price, sr_context=sr_context
            )

            # Check if opportunity meets confidence threshold
            opportunity_detected = (
                sr_outcome.get("confidence", 0) >= self.sr_alert_threshold
            )

            if not opportunity_detected:
                return {"opportunity_detected": False}

            # Generate detailed alert for Tactician
            alert = {
                "opportunity_detected": True,
                "outcome": sr_outcome.get("outcome", "consolidation"),
                "confidence": sr_outcome.get("confidence", 0),
                "probabilities": sr_outcome.get("probabilities", {}),
                "sr_context": sr_context,
                "current_price": current_price,
                "timestamp": pd.Timestamp.now().isoformat(),
                "tactician_recommendations": self._generate_tactician_recommendations(
                    sr_outcome, sr_context
                ),
            }

            return alert

        except Exception as e:
            self.logger.error(f"Error getting S/R opportunity alert: {e}")
            return {"opportunity_detected": False, "error": str(e)}

    def _generate_tactician_recommendations(
        self, sr_outcome: dict[str, Any], sr_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate specific recommendations for the Tactician."""
        try:
            outcome = sr_outcome.get("outcome", "consolidation")
            confidence = sr_outcome.get("confidence", 0)
            current_price = sr_context.get("current_price", 0)

            # Determine position direction based on S/R context
            nearest_resistance = sr_context.get("nearest_resistance", current_price)
            nearest_support = sr_context.get("nearest_support", current_price)

            distance_to_resistance = (
                abs(current_price - nearest_resistance) / current_price
            )
            distance_to_support = abs(current_price - nearest_support) / current_price

            recommendations = {
                "action": "MONITOR",  # MONITOR, PREPARE, EXECUTE
                "position_direction": "NONE",  # LONG, SHORT, NONE
                "position_size": 0.0,
                "leverage": 1.0,
                "entry_strategy": "WAIT_FOR_CONFIRMATION",
                "stop_loss_strategy": "TIGHT_STOPS",
                "take_profit_strategy": "CONSERVATIVE_TP",
            }

            if outcome == "breakout" and confidence >= 0.8:
                # Determine position direction based on which level we're breaking out from
                if distance_to_resistance < distance_to_support:
                    # Breaking out from resistance = SHORT position
                    position_direction = "SHORT"
                else:
                    # Breaking out from support = LONG position
                    position_direction = "LONG"

                recommendations.update(
                    {
                        "action": "PREPARE",
                        "position_direction": position_direction,
                        "position_size": min(
                            confidence * 0.8, 0.6
                        ),  # Conservative sizing
                        "leverage": 1.0,
                        "entry_strategy": "BREAKOUT_CONFIRMATION",
                        "stop_loss_strategy": "TIGHT_STOPS",
                        "take_profit_strategy": "MOMENTUM_BASED_TP",
                    }
                )
            elif outcome == "breakout" and confidence >= 0.6:
                # Determine position direction based on which level we're breaking out from
                if distance_to_resistance < distance_to_support:
                    position_direction = "SHORT"
                else:
                    position_direction = "LONG"

                recommendations.update(
                    {
                        "action": "MONITOR",
                        "position_direction": position_direction,
                        "position_size": min(confidence * 0.6, 0.4),
                        "entry_strategy": "WAIT_FOR_CONFIRMATION",
                        "stop_loss_strategy": "TIGHT_STOPS",
                    }
                )
            elif outcome == "rebounce" and confidence >= 0.8:
                # Determine position direction based on which level we're rebouncing from
                if distance_to_resistance < distance_to_support:
                    # Rebouncing from resistance = LONG position (price bounces down from resistance)
                    position_direction = "LONG"
                else:
                    # Rebouncing from support = SHORT position (price bounces up from support)
                    position_direction = "SHORT"

                recommendations.update(
                    {
                        "action": "PREPARE",
                        "position_direction": position_direction,
                        "position_size": min(confidence * 0.8, 0.6),
                        "leverage": 1.0,
                        "entry_strategy": "REBOUNCE_CONFIRMATION",
                        "stop_loss_strategy": "TIGHT_STOPS",
                        "take_profit_strategy": "MOMENTUM_BASED_TP",
                    }
                )
            elif outcome == "rebounce" and confidence >= 0.6:
                # Determine position direction based on which level we're rebouncing from
                if distance_to_resistance < distance_to_support:
                    position_direction = "LONG"
                else:
                    position_direction = "SHORT"

                recommendations.update(
                    {
                        "action": "MONITOR",
                        "position_direction": position_direction,
                        "position_size": min(confidence * 0.6, 0.4),
                        "entry_strategy": "WAIT_FOR_CONFIRMATION",
                        "stop_loss_strategy": "TIGHT_STOPS",
                    }
                )
            elif outcome == "consolidation" and confidence >= 0.7:
                recommendations.update(
                    {
                        "action": "MONITOR",
                        "position_direction": "NONE",
                        "entry_strategy": "AVOID_DIRECTIONAL_TRADES",
                        "stop_loss_strategy": "NONE",
                    }
                )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating Tactician recommendations: {e}")
            return {"action": "MONITOR", "position_direction": "NONE"}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="unified regime intelligence prediction",
    )
    async def predict(
        self,
        hmm_states: Dict[str, np.ndarray],
        market_features: np.ndarray,
        market_data: pd.DataFrame,
        current_price: float,
        timestamp: datetime = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Make unified regime intelligence predictions with integrated S/R analysis.

        Args:
            hmm_states: HMM states for each timeframe
            market_features: Market features
            market_data: Raw market data for S/R analysis
            current_price: Current market price
            timestamp: Current timestamp (optional)

        Returns:
            dict: Unified predictions with regime, transition, TPSL, and S/R information
        """
        try:
            if not self.is_initialized:
                self.logger.error("Runtime not initialized")
                return None

            # Use unified step for integrated predictions (includes S/R analysis)
            integrated_prediction = await self.unified_step.predict_with_sr_integration(
                hmm_states, market_features, market_data, current_price
            )

            if not integrated_prediction:
                self.logger.error("Failed to get integrated prediction")
                return None

            # Enhance predictions with intensity analysis
            enhanced_prediction = await self._enhance_predictions(
                integrated_prediction, current_price, timestamp
            )

            # Update runtime state
            await self._update_runtime_state(enhanced_prediction)

            return enhanced_prediction

        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return None

    async def _prepare_inputs(
        self, hmm_states: Dict[str, np.ndarray], market_features: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Prepare inputs for the unified model."""
        try:
            # Ensure we have the required sequence length
            if market_features.shape[0] < self.sequence_length:
                self.logger.warning(
                    f"Insufficient features: {market_features.shape[0]} < {self.sequence_length}"
                )
                return None

            # Take the most recent sequence
            recent_features = market_features[-self.sequence_length :]

            # Prepare HMM states for each timeframe
            prepared_hmm_states = {}
            for tf in self.timeframes:
                if tf in hmm_states:
                    tf_states = hmm_states[tf]
                    if len(tf_states) >= self.sequence_length:
                        prepared_hmm_states[tf] = tf_states[-self.sequence_length :]
                    else:
                        self.logger.warning(f"Insufficient HMM states for {tf}")
                        return None

            return {"hmm_states": prepared_hmm_states, "features": recent_features}

        except Exception as e:
            self.logger.error(f"Error preparing inputs: {e}")
            return None

    async def _enhance_predictions(
        self, prediction: Dict[str, Any], current_price: float, timestamp: datetime
    ) -> Dict[str, Any]:
        """Enhance raw predictions with additional context and analysis."""
        try:
            enhanced = prediction.copy()

            # Decode regime prediction
            if "regime" in prediction and "regime" in self.label_encoders:
                regime_pred = prediction["regime"]["prediction"]
                try:
                    regime_name = self.label_encoders["regime"].inverse_transform(
                        [regime_pred]
                    )[0]
                    enhanced["regime"]["name"] = regime_name
                except:
                    enhanced["regime"]["name"] = f"REGIME_{regime_pred}"

            # Decode intensity scores
            if "intensity" in prediction:
                enhanced["intensity"] = {
                    "scores": prediction["intensity"].get("scores", []),
                    "top_regimes": prediction["intensity"].get("top_regimes", []),
                    "transition_analysis": prediction["intensity"].get(
                        "transition_analysis", {}
                    ),
                }

            # Decode TPSL prediction
            if "tpsl" in prediction and "tpsl" in self.label_encoders:
                tpsl_pred = prediction["tpsl"]["prediction"]
                try:
                    tpsl_name = self.label_encoders["tpsl"].inverse_transform(
                        [tpsl_pred]
                    )[0]
                    enhanced["tpsl"]["name"] = tpsl_name
                except:
                    enhanced["tpsl"]["name"] = prediction["tpsl"].get(
                        "direction", "hold"
                    )

            # Add expert activation logic based on step01_7 regimes and step5-10 models
            enhanced["expert_activation"] = await self._determine_expert_activation(
                enhanced
            )

            # Add timestamp
            enhanced["timestamp"] = timestamp.isoformat()

            return enhanced

        except Exception as e:
            self.logger.error(f"Error enhancing predictions: {e}")
            return prediction

    async def _determine_expert_activation(
        self, prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine which expert models should be activated based on intensity-based regime analysis."""
        try:
            activation = {
                "primary_expert": None,
                "secondary_experts": [],
                "activation_reason": "",
                "confidence": 0.0,
                "intensity_analysis": {},
            }

            regime_name = prediction.get("regime", {}).get("name", "")
            regime_conf = prediction.get("regime", {}).get("confidence", 0.0)
            tpsl_direction = prediction.get("tpsl", {}).get("direction", "hold")
            intensity_data = prediction.get("intensity", {})

            # Analyze intensity scores for "in the middle" scenarios
            intensity_scores = intensity_data.get("scores", [])
            top_regimes = intensity_data.get("top_regimes", [])

            if len(top_regimes) >= 2:
                # We're "in the middle" of several regimes
                activation["intensity_analysis"] = {
                    "scenario": "multiple_regimes",
                    "top_regimes": top_regimes,
                    "intensity_scores": intensity_scores,
                }

                # Activate multiple experts based on intensity weights
                for regime_info in top_regimes[:3]:  # Top 3 regimes
                    regime_id = regime_info.get("regime_id")
                    intensity = regime_info.get("intensity", 0.0)

                    if intensity > 0.3:  # Significant intensity threshold
                        expert_name = f"REGIME_{regime_id}_EXPERT"
                        activation["secondary_experts"].append(expert_name)
                        activation["activation_reason"] += (
                            f" + {expert_name} (intensity={intensity:.2f})"
                        )

                # Set primary expert to highest intensity regime
                if top_regimes:
                    primary_regime = top_regimes[0]
                    activation["primary_expert"] = (
                        f"REGIME_{primary_regime['regime_id']}_EXPERT"
                    )
                    activation["confidence"] = primary_regime.get("intensity", 0.0)
                    activation["activation_reason"] = (
                        f"Primary: {activation['primary_expert']} (intensity={activation['confidence']:.2f})"
                    )

            else:
                # Single dominant regime
                if regime_conf >= self.regime_confidence_threshold:
                    # Map regime to expert based on step01_7 archetype descriptions
                    if "BULL" in regime_name or "TREND" in regime_name:
                        activation["primary_expert"] = "BULL_TREND_EXPERT"
                        activation["activation_reason"] = (
                            f"Bullish regime detected: {regime_name}"
                        )
                    elif "BEAR" in regime_name:
                        activation["primary_expert"] = "BEAR_TREND_EXPERT"
                        activation["activation_reason"] = (
                            f"Bearish regime detected: {regime_name}"
                        )
                    elif "SIDEWAYS" in regime_name or "RANGE" in regime_name:
                        activation["primary_expert"] = "SIDEWAYS_EXPERT"
                        activation["activation_reason"] = (
                            f"Sideways regime detected: {regime_name}"
                        )
                    elif "VOLATILITY" in regime_name:
                        activation["primary_expert"] = "VOLATILITY_EXPERT"
                        activation["activation_reason"] = (
                            f"Volatile regime detected: {regime_name}"
                        )
                    else:
                        activation["primary_expert"] = "GENERAL_EXPERT"
                        activation["activation_reason"] = (
                            f"General regime: {regime_name}"
                        )

                    activation["confidence"] = regime_conf

            # Add transition experts when intensity changes are detected
            transition_prob = prediction.get("transition", {}).get("probability", 0.0)
            if transition_prob >= self.transition_threshold:
                activation["secondary_experts"].append("TRANSITION_EXPERT")
                activation["activation_reason"] += (
                    f" + Transition expert (P={transition_prob:.2f})"
                )

                # Add momentum experts during transitions
                activation["secondary_experts"].append("MOMENTUM_EXPERT")
                activation["activation_reason"] += (
                    " + Momentum expert (transition detected)"
                )

            # Add TPSL-based direction information
            if tpsl_direction != "hold":
                activation["activation_reason"] += (
                    f" + TPSL direction: {tpsl_direction}"
                )

            return activation

        except Exception as e:
            self.logger.error(f"Error determining expert activation: {e}")
            return {
                "primary_expert": None,
                "secondary_experts": [],
                "activation_reason": "Error",
                "confidence": 0.0,
            }

    async def _update_runtime_state(self, prediction: Dict[str, Any]) -> None:
        """Update runtime state with current prediction."""
        try:
            # Update current regime
            regime_name = prediction.get("regime", {}).get("name", "UNKNOWN")
            if regime_name != self.current_regime:
                self.logger.info(
                    f"Regime change detected: {self.current_regime} -> {regime_name}"
                )
                self.current_regime = regime_name

            # Update regime history
            self.regime_history.append(
                {
                    "regime": regime_name,
                    "confidence": prediction.get("regime", {}).get("confidence", 0.0),
                    "timestamp": prediction.get(
                        "timestamp", datetime.now().isoformat()
                    ),
                }
            )

            # Keep only recent history
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]

            # Update transition probability
            self.transition_probability = prediction.get("transition", {}).get(
                "probability", 0.0
            )

            # Update TPSL direction
            self.tpsl_direction = prediction.get("tpsl", {}).get("direction", "hold")

        except Exception as e:
            self.logger.error(f"Error updating runtime state: {e}")

    def get_current_state(self) -> Dict[str, Any]:
        """Get current runtime state."""
        return {
            "current_regime": self.current_regime,
            "transition_probability": self.transition_probability,
            "tpsl_direction": self.tpsl_direction,
            "regime_history": self.regime_history[-10:],  # Last 10 entries
            "is_initialized": self.is_initialized,
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the unified model."""
        try:
            if not self.regime_history:
                return {"error": "No regime history available"}

            # Calculate regime stability
            recent_regimes = [entry["regime"] for entry in self.regime_history[-20:]]
            regime_changes = sum(
                1
                for i in range(1, len(recent_regimes))
                if recent_regimes[i] != recent_regimes[i - 1]
            )
            stability_score = 1.0 - (regime_changes / max(1, len(recent_regimes) - 1))

            # Calculate average confidence
            avg_confidence = np.mean(
                [entry["confidence"] for entry in self.regime_history[-20:]]
            )

            return {
                "regime_stability": stability_score,
                "average_confidence": avg_confidence,
                "total_predictions": len(self.regime_history),
                "current_regime_duration": self._calculate_regime_duration(),
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}

    def _calculate_regime_duration(self) -> int:
        """Calculate how long the current regime has been active."""
        if not self.regime_history or not self.current_regime:
            return 0

        duration = 0
        for entry in reversed(self.regime_history):
            if entry["regime"] == self.current_regime:
                duration += 1
            else:
                break

        return duration
