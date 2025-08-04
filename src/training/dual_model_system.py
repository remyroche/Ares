# src/training/dual_model_system.py

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger

# Import ML Confidence Predictor
from src.analyst.ml_confidence_predictor import MLConfidencePredictor


class DualModelSystem:
    """
    Dual Model System for trading decisions.
    
    Analyst Model: Decides IF we enter/exit a trade (multiple timeframes)
    Tactician Model: Decides WHEN we enter/exit a trade (1m timeframe)
    
    Both models use ml_confidence_predictor.py for predictions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize Dual Model System.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("DualModelSystem")

        # Model state
        self.analyst_model: Optional[Any] = None
        self.tactician_model: Optional[Any] = None
        self.ml_confidence_predictor: Optional[MLConfidencePredictor] = None
        self.is_initialized: bool = False

        # Configuration
        self.dual_model_config: dict[str, Any] = self.config.get(
            "dual_model_system",
            {},
        )
        
        # Analyst model configuration (IF decisions)
        self.analyst_timeframes: List[str] = self.dual_model_config.get(
            "analyst_timeframes",
            ["1h", "15m", "5m", "1m"]
        )
        self.analyst_confidence_threshold: float = self.dual_model_config.get(
            "analyst_confidence_threshold",
            0.7
        )
        
        # Tactician model configuration (WHEN decisions)
        self.tactician_timeframes: List[str] = self.dual_model_config.get(
            "tactician_timeframes",
            ["1m"]
        )
        self.tactician_confidence_threshold: float = self.dual_model_config.get(
            "tactician_confidence_threshold",
            0.8
        )
        
        # Ensemble configuration
        self.enable_ensemble_analysis: bool = self.dual_model_config.get(
            "enable_ensemble_analysis",
            True
        ) 

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid dual model system configuration"),
            AttributeError: (False, "Missing required dual model parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="dual model system initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize Dual Model System with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Dual Model System...")

            # Load dual model configuration
            await self._load_dual_model_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for dual model system")
                return False

            # Initialize ML Confidence Predictor
            await self._initialize_ml_confidence_predictor()

            # Initialize Analyst Model
            await self._initialize_analyst_model()

            # Initialize Tactician Model
            await self._initialize_tactician_model()

            self.is_initialized = True
            self.logger.info("✅ Dual Model System initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Dual Model System initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="dual model configuration loading",
    )
    async def _load_dual_model_configuration(self) -> None:
        """Load dual model configuration."""
        try:
            # Set default dual model parameters
            self.dual_model_config.setdefault("analyst_timeframes", ["1h", "15m", "5m", "1m"])
            self.dual_model_config.setdefault("tactician_timeframes", ["1m"])
            self.dual_model_config.setdefault("analyst_confidence_threshold", 0.7)
            self.dual_model_config.setdefault("tactician_confidence_threshold", 0.8)
            self.dual_model_config.setdefault("enable_ensemble_analysis", True)

            # Update configuration
            self.analyst_timeframes = self.dual_model_config["analyst_timeframes"]
            self.tactician_timeframes = self.dual_model_config["tactician_timeframes"]
            self.analyst_confidence_threshold = self.dual_model_config["analyst_confidence_threshold"]
            self.tactician_confidence_threshold = self.dual_model_config["tactician_confidence_threshold"]
            self.enable_ensemble_analysis = self.dual_model_config["enable_ensemble_analysis"]

            self.logger.info("Dual model configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading dual model configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate dual model configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate analyst timeframes
            if not self.analyst_timeframes:
                self.logger.error("Analyst timeframes cannot be empty")
                return False

            # Validate tactician timeframes
            if not self.tactician_timeframes:
                self.logger.error("Tactician timeframes cannot be empty")
                return False

            # Validate confidence thresholds
            if not (0.0 <= self.analyst_confidence_threshold <= 1.0):
                self.logger.error("Analyst confidence threshold must be between 0 and 1")
                return False

            if not (0.0 <= self.tactician_confidence_threshold <= 1.0):
                self.logger.error("Tactician confidence threshold must be between 0 and 1")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML confidence predictor initialization",
    )
    async def _initialize_ml_confidence_predictor(self) -> None:
        """Initialize ML Confidence Predictor."""
        try:
            self.ml_confidence_predictor = MLConfidencePredictor(self.config)
            await self.ml_confidence_predictor.initialize()
            
            self.logger.info("ML Confidence Predictor initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing ML Confidence Predictor: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst model initialization",
    )
    async def _initialize_analyst_model(self) -> None:
        """Initialize Analyst Model for IF decisions."""
        try:
            # Load analyst model from training results
            analyst_model_path = "models/analyst_model.pkl"
            
            if os.path.exists(analyst_model_path):
                import pickle
                with open(analyst_model_path, 'rb') as f:
                    self.analyst_model = pickle.load(f)
                self.logger.info("Analyst model loaded successfully")
            else:
                self.logger.warning("Analyst model not found, will use ML Confidence Predictor")
                self.analyst_model = None

        except Exception as e:
            self.logger.error(f"Error initializing Analyst model: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician model initialization",
    )
    async def _initialize_tactician_model(self) -> None:
        """Initialize Tactician Model for WHEN decisions."""
        try:
            # Load tactician model from training results
            tactician_model_path = "models/tactician_model.pkl"
            
            if os.path.exists(tactician_model_path):
                import pickle
                with open(tactician_model_path, 'rb') as f:
                    self.tactician_model = pickle.load(f)
                self.logger.info("Tactician model loaded successfully")
            else:
                self.logger.warning("Tactician model not found, will use ML Confidence Predictor")
                self.tactician_model = None

        except Exception as e:
            self.logger.error(f"Error initializing Tactician model: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid market data for decision making"),
            AttributeError: (None, "Models not properly initialized"),
        },
        default_return=None,
        context="dual model decision making",
    )
    async def make_trading_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        current_position: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Make trading decision using dual model system.

        Args:
            market_data: Market data for analysis
            current_price: Current asset price
            current_position: Current position information (if any)

        Returns:
            Dictionary with trading decision
        """
        try:
            if not self.is_initialized:
                raise ValueError("Dual Model System not initialized")

            self.logger.info("🎯 Making Dual Model Trading Decision")

            # Check if we have an open position for exit logic
            if current_position:
                return await self._make_exit_decision(
                    market_data, current_price, current_position
                )
            
            return await self._make_entry_decision(market_data, current_price)

        except Exception as e:
            self.logger.error(f"Error making trading decision: {e}")
            return self._get_fallback_decision()

    async def _make_entry_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """Make entry decision using dual model system."""
        try:
            # Step 1: Analyst Model - IF decision
            analyst_decision = await self._get_analyst_decision(market_data, current_price)
            
            if not analyst_decision["should_trade"]:
                return {
                    "action": "HOLD",
                    "reason": "Analyst model: No clear trading opportunity",
                    "analyst_confidence": analyst_decision["confidence"],
                    "timestamp": datetime.now().isoformat(),
                }

            # Step 2: Tactician Model - WHEN decision
            tactician_decision = await self._get_tactician_decision(
                market_data, current_price, analyst_decision
            )

            # Combine decisions
            final_decision = {
                "action": "ENTRY" if tactician_decision["should_execute"] else "HOLD",
                "direction": analyst_decision["direction"],
                "strategy": analyst_decision["strategy"],
                "analyst_confidence": analyst_decision["confidence"],
                "tactician_confidence": tactician_decision["confidence"],
                "entry_timing": tactician_decision["timing_signal"],
                "position_size": self._calculate_position_size(
                    analyst_decision, tactician_decision
                ),
                "reason": tactician_decision["reason"],
                "analyst_timeframes": self.analyst_timeframes,
                "tactician_timeframes": self.tactician_timeframes,
                "timestamp": datetime.now().isoformat(),
            }

            return final_decision

        except Exception as e:
            self.logger.error(f"Error making entry decision: {e}")
            return self._get_fallback_decision()

    async def _make_exit_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        current_position: dict[str, Any],
    ) -> dict[str, Any]:
        """Make exit decision using dual model system."""
        try:
            # Step 1: Analyst Model - IF exit decision
            analyst_exit_decision = await self._get_analyst_exit_decision(
                market_data, current_price, current_position
            )
            
            if not analyst_exit_decision["should_exit"]:
                return {
                    "action": "HOLD_POSITION",
                    "reason": "Analyst model: No clear exit signal",
                    "analyst_confidence": analyst_exit_decision["confidence"],
                    "timestamp": datetime.now().isoformat(),
                }

            # Step 2: Tactician Model - WHEN exit decision
            tactician_exit_decision = await self._get_tactician_exit_decision(
                market_data, current_price, analyst_exit_decision
            )

            # Combine decisions
            final_decision = {
                "action": "EXIT" if tactician_exit_decision["should_execute"] else "HOLD_POSITION",
                "exit_type": analyst_exit_decision["exit_type"],
                "strategy": analyst_exit_decision["strategy"],
                "analyst_confidence": analyst_exit_decision["confidence"],
                "tactician_confidence": tactician_exit_decision["confidence"],
                "exit_timing": tactician_exit_decision["timing_signal"],
                "exit_reason": tactician_exit_decision["reason"],
                "analyst_timeframes": self.analyst_timeframes,
                "tactician_timeframes": self.tactician_timeframes,
                "timestamp": datetime.now().isoformat(),
            }

            return final_decision

        except Exception as e:
            self.logger.error(f"Error making exit decision: {e}")
            return self._get_fallback_decision()

    async def _get_analyst_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """Get Analyst model decision for IF we should trade."""
        try:
            # Use ML Confidence Predictor for analyst decision
            if self.ml_confidence_predictor:
                # Use the new dual model system prediction method
                analyst_predictions = await self.ml_confidence_predictor.predict_for_dual_model_system(
                    market_data=market_data,
                    current_price=current_price,
                    model_type="analyst"
                )
                
                if analyst_predictions:
                    # Verify confidence calculations
                    verification_results = self.ml_confidence_predictor.verify_confidence_calculations(
                        market_data, current_price
                    )
                    
                    # Add verification results to analyst predictions
                    analyst_predictions["verification_results"] = verification_results
                    
                    return analyst_predictions
                else:
                    # Fallback to original method
                    confidence_predictions = await self.ml_confidence_predictor.predict_confidence_table(
                        market_data, current_price
                    )
                    
                    if confidence_predictions:
                        return self._analyze_analyst_confidence(confidence_predictions, current_price)
            
            # Fallback to model-based decision
            if self.analyst_model:
                return await self._get_model_based_analyst_decision(market_data, current_price)
            
            # Final fallback
            return {
                "should_trade": False,
                "direction": "HOLD",
                "strategy": "UNKNOWN",
                "confidence": 0.5,
                "reason": "No analyst model available"
            }

        except Exception as e:
            self.logger.error(f"Error getting analyst decision: {e}")
            return {
                "should_trade": False,
                "direction": "HOLD",
                "strategy": "ERROR",
                "confidence": 0.0,
                "reason": f"Analyst decision error: {e}"
            }

    def _analyze_analyst_confidence(
        self,
        confidence_predictions: dict[str, Any],
        current_price: float,
    ) -> dict[str, Any]:
        """Analyze confidence predictions for analyst decision."""
        try:
            price_target_confidences = confidence_predictions.get("price_target_confidences", {})
            adversarial_confidences = confidence_predictions.get("adversarial_confidences", {})
            
            # Calculate overall confidence
            if price_target_confidences:
                # Calculate weighted average confidence for price targets
                total_confidence = 0.0
                total_weight = 0.0
                
                for target_str, confidence in price_target_confidences.items():
                    target = float(target_str.replace("%", ""))
                    weight = target  # Higher targets get higher weight
                    total_confidence += confidence * weight
                    total_weight += weight
                
                overall_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
            else:
                overall_confidence = 0.5
            
            # Determine direction and strategy
            if overall_confidence > self.analyst_confidence_threshold:
                direction = "LONG"
                should_trade = True
                strategy = "BULLISH"
            elif overall_confidence < (1 - self.analyst_confidence_threshold):
                direction = "SHORT"
                should_trade = True
                strategy = "BEARISH"
            else:
                direction = "HOLD"
                should_trade = False
                strategy = "NEUTRAL"
            
            return {
                "should_trade": should_trade,
                "direction": direction,
                "strategy": strategy,
                "confidence": overall_confidence,
                "price_target_confidences": price_target_confidences,
                "adversarial_confidences": adversarial_confidences,
                "reason": f"Analyst confidence: {overall_confidence:.2f}"
            }

        except Exception as e:
            self.logger.error(f"Error analyzing analyst confidence: {e}")
            return {
                "should_trade": False,
                "direction": "HOLD",
                "strategy": "ERROR",
                "confidence": 0.0,
                "reason": f"Confidence analysis error: {e}"
            }

    async def _get_model_based_analyst_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """Get analyst decision using trained model."""
        try:
            # Simulate model-based decision
            # In practice, this would use the trained analyst model
            return {
                "should_trade": True,
                "direction": "LONG",
                "strategy": "BULLISH",
                "confidence": 0.75,
                "reason": "Model-based analyst decision"
            }

        except Exception as e:
            self.logger.error(f"Error getting model-based analyst decision: {e}")
            return {
                "should_trade": False,
                "direction": "HOLD",
                "strategy": "ERROR",
                "confidence": 0.0,
                "reason": f"Model-based decision error: {e}"
            }

    async def _get_tactician_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        analyst_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Get Tactician model decision for WHEN we should trade."""
        try:
            # Use ML Confidence Predictor for tactician decision
            if self.ml_confidence_predictor:
                # Use the new dual model system prediction method
                tactician_predictions = await self.ml_confidence_predictor.predict_for_dual_model_system(
                    market_data=market_data,
                    current_price=current_price,
                    model_type="tactician"
                )
                
                if tactician_predictions:
                    # Verify confidence calculations
                    verification_results = self.ml_confidence_predictor.verify_confidence_calculations(
                        market_data, current_price
                    )
                    
                    # Add verification results to tactician predictions
                    tactician_predictions["verification_results"] = verification_results
                    
                    return tactician_predictions
                else:
                    # Fallback to original method
                    confidence_predictions = await self.ml_confidence_predictor.predict_confidence_table(
                        market_data, current_price
                    )
                    
                    if confidence_predictions:
                        return self._analyze_tactician_confidence(
                            confidence_predictions, current_price, analyst_decision
                        )
            
            # Fallback to model-based decision
            if self.tactician_model:
                return await self._get_model_based_tactician_decision(
                    market_data, current_price, analyst_decision
                )
            
            # Final fallback
            return {
                "should_execute": False,
                "timing_signal": 0.5,
                "confidence": 0.5,
                "reason": "No tactician model available"
            }

        except Exception as e:
            self.logger.error(f"Error getting tactician decision: {e}")
            return {
                "should_execute": False,
                "timing_signal": 0.0,
                "confidence": 0.0,
                "reason": f"Tactician decision error: {e}"
            }

    def _analyze_tactician_confidence(
        self,
        confidence_predictions: dict[str, Any],
        current_price: float,
        analyst_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze confidence predictions for tactician decision."""
        try:
            price_target_confidences = confidence_predictions.get("price_target_confidences", {})
            
            # Focus on short-term targets for timing
            short_term_targets = {k: v for k, v in price_target_confidences.items() 
                                if float(k.replace("%", "")) <= 1.0}
            
            if short_term_targets:
                # Calculate timing confidence from short-term targets
                timing_confidence = sum(short_term_targets.values()) / len(short_term_targets)
            else:
                timing_confidence = 0.5
            
            # Determine if we should execute based on timing confidence
            should_execute = timing_confidence > self.tactician_confidence_threshold
            
            # Adjust based on analyst direction
            direction = analyst_decision.get("direction", "HOLD")
            if direction == "SHORT":
                # For short positions, invert the confidence
                timing_confidence = 1.0 - timing_confidence
                should_execute = timing_confidence > self.tactician_confidence_threshold
            
            return {
                "should_execute": should_execute,
                "timing_signal": timing_confidence,
                "confidence": timing_confidence,
                "short_term_targets": short_term_targets,
                "reason": f"Tactician timing confidence: {timing_confidence:.2f}"
            }

        except Exception as e:
            self.logger.error(f"Error analyzing tactician confidence: {e}")
            return {
                "should_execute": False,
                "timing_signal": 0.0,
                "confidence": 0.0,
                "reason": f"Tactician confidence analysis error: {e}"
            }

    async def _get_model_based_tactician_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        analyst_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Get tactician decision using trained model."""
        try:
            # Simulate model-based tactician decision
            # In practice, this would use the trained tactician model
            return {
                "should_execute": True,
                "timing_signal": 0.85,
                "confidence": 0.85,
                "reason": "Model-based tactician decision"
            }

        except Exception as e:
            self.logger.error(f"Error getting model-based tactician decision: {e}")
            return {
                "should_execute": False,
                "timing_signal": 0.0,
                "confidence": 0.0,
                "reason": f"Model-based tactician decision error: {e}"
            }

    async def _get_analyst_exit_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        current_position: dict[str, Any],
    ) -> dict[str, Any]:
        """Get Analyst model exit decision."""
        try:
            # Use ML Confidence Predictor for exit analysis
            if self.ml_confidence_predictor:
                confidence_predictions = await self.ml_confidence_predictor.predict_confidence_table(
                    market_data, current_price
                )
                
                if confidence_predictions:
                    return self._analyze_analyst_exit_confidence(
                        confidence_predictions, current_price, current_position
                    )
            
            # Fallback
            return {
                "should_exit": False,
                "exit_type": "HOLD",
                "strategy": "HOLD",
                "confidence": 0.5,
                "reason": "No clear exit signal"
            }

        except Exception as e:
            self.logger.error(f"Error getting analyst exit decision: {e}")
            return {
                "should_exit": False,
                "exit_type": "HOLD",
                "strategy": "ERROR",
                "confidence": 0.0,
                "reason": f"Analyst exit decision error: {e}"
            }

    def _analyze_analyst_exit_confidence(
        self,
        confidence_predictions: dict[str, Any],
        current_price: float,
        current_position: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze confidence predictions for analyst exit decision."""
        try:
            price_target_confidences = confidence_predictions.get("price_target_confidences", {})
            adversarial_confidences = confidence_predictions.get("adversarial_confidences", {})
            
            position_type = current_position.get("type", "LONG")
            
            # Analyze exit conditions based on position type
            if position_type == "LONG":
                # For long positions, check if we should take profit or stop loss
                if price_target_confidences:
                    # Check if we've reached profit targets
                    profit_targets = {k: v for k, v in price_target_confidences.items() 
                                    if float(k.replace("%", "")) > 0}
                    if profit_targets:
                        max_profit_confidence = max(profit_targets.values())
                        if max_profit_confidence > self.analyst_confidence_threshold:
                            return {
                                "should_exit": True,
                                "exit_type": "TAKE_PROFIT",
                                "strategy": "PROFIT_TAKING",
                                "confidence": max_profit_confidence,
                                "reason": f"Profit target reached: {max_profit_confidence:.2f}"
                            }
                
                # Check stop loss conditions
                if adversarial_confidences:
                    stop_loss_confidence = max(adversarial_confidences.values())
                    if stop_loss_confidence > 0.7:  # High confidence for stop loss
                        return {
                            "should_exit": True,
                            "exit_type": "STOP_LOSS",
                            "strategy": "RISK_MANAGEMENT",
                            "confidence": stop_loss_confidence,
                            "reason": f"Stop loss triggered: {stop_loss_confidence:.2f}"
                        }
            
            elif position_type == "SHORT":
                # For short positions, check if we should take profit or stop loss
                if adversarial_confidences:
                    # Check if we've reached profit targets (price went down)
                    profit_targets = {k: v for k, v in adversarial_confidences.items() 
                                    if float(k.replace("%", "")) > 0}
                    if profit_targets:
                        max_profit_confidence = max(profit_targets.values())
                        if max_profit_confidence > self.analyst_confidence_threshold:
                            return {
                                "should_exit": True,
                                "exit_type": "TAKE_PROFIT",
                                "strategy": "PROFIT_TAKING",
                                "confidence": max_profit_confidence,
                                "reason": f"Profit target reached: {max_profit_confidence:.2f}"
                            }
                
                # Check stop loss conditions (price went up)
                if price_target_confidences:
                    stop_loss_confidence = max(price_target_confidences.values())
                    if stop_loss_confidence > 0.7:  # High confidence for stop loss
                        return {
                            "should_exit": True,
                            "exit_type": "STOP_LOSS",
                            "strategy": "RISK_MANAGEMENT",
                            "confidence": stop_loss_confidence,
                            "reason": f"Stop loss triggered: {stop_loss_confidence:.2f}"
                        }
            
            return {
                "should_exit": False,
                "exit_type": "HOLD",
                "strategy": "HOLD",
                "confidence": 0.5,
                "reason": "No clear exit signal"
            }

        except Exception as e:
            self.logger.error(f"Error analyzing analyst exit confidence: {e}")
            return {
                "should_exit": False,
                "exit_type": "HOLD",
                "strategy": "ERROR",
                "confidence": 0.0,
                "reason": f"Exit confidence analysis error: {e}"
            }

    async def _get_tactician_exit_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        analyst_exit_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Get Tactician model exit decision."""
        try:
            # Use ML Confidence Predictor for exit timing
            if self.ml_confidence_predictor:
                confidence_predictions = await self.ml_confidence_predictor.predict_confidence_table(
                    market_data, current_price
                )
                
                if confidence_predictions:
                    return self._analyze_tactician_exit_confidence(
                        confidence_predictions, current_price, analyst_exit_decision
                    )
            
            # Fallback
            return {
                "should_execute": False,
                "timing_signal": 0.5,
                "confidence": 0.5,
                "reason": "No clear exit timing"
            }

        except Exception as e:
            self.logger.error(f"Error getting tactician exit decision: {e}")
            return {
                "should_execute": False,
                "timing_signal": 0.0,
                "confidence": 0.0,
                "reason": f"Tactician exit decision error: {e}"
            }

    def _analyze_tactician_exit_confidence(
        self,
        confidence_predictions: dict[str, Any],
        current_price: float,
        analyst_exit_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze confidence predictions for tactician exit decision."""
        try:
            exit_type = analyst_exit_decision.get("exit_type", "HOLD")
            
            if exit_type == "TAKE_PROFIT":
                # For take profit, look for high confidence in short-term targets
                price_target_confidences = confidence_predictions.get("price_target_confidences", {})
                short_term_targets = {k: v for k, v in price_target_confidences.items() 
                                    if float(k.replace("%", "")) <= 0.5}
                
                if short_term_targets:
                    timing_confidence = max(short_term_targets.values())
                else:
                    timing_confidence = 0.5
                    
            elif exit_type == "STOP_LOSS":
                # For stop loss, look for high confidence in adverse movements
                adversarial_confidences = confidence_predictions.get("adversarial_confidences", {})
                short_term_adversarial = {k: v for k, v in adversarial_confidences.items() 
                                        if float(k.replace("%", "")) <= 0.5}
                
                if short_term_adversarial:
                    timing_confidence = max(short_term_adversarial.values())
                else:
                    timing_confidence = 0.5
            else:
                timing_confidence = 0.5
            
            should_execute = timing_confidence > self.tactician_confidence_threshold
            
            return {
                "should_execute": should_execute,
                "timing_signal": timing_confidence,
                "confidence": timing_confidence,
                "reason": f"Exit timing confidence: {timing_confidence:.2f}"
            }

        except Exception as e:
            self.logger.error(f"Error analyzing tactician exit confidence: {e}")
            return {
                "should_execute": False,
                "timing_signal": 0.0,
                "confidence": 0.0,
                "reason": f"Exit confidence analysis error: {e}"
            }

    def _calculate_position_size(
        self,
        analyst_decision: dict[str, Any],
        tactician_decision: dict[str, Any],
    ) -> float:
        """Calculate position size based on both analyst and tactician decisions."""
        try:
            base_size = 1.0

            # Adjust based on analyst confidence
            analyst_confidence = analyst_decision.get("confidence", 0.5)
            if analyst_confidence > 0.8:
                analyst_multiplier = 1.2
            elif analyst_confidence > 0.6:
                analyst_multiplier = 1.0
            else:
                analyst_multiplier = 0.8

            # Adjust based on tactician confidence
            tactician_confidence = tactician_decision.get("confidence", 0.5)
            if tactician_confidence > 0.9:
                tactician_multiplier = 1.3
            elif tactician_confidence > 0.8:
                tactician_multiplier = 1.1
            else:
                tactician_multiplier = 0.9

            # Calculate final position size
            final_size = base_size * analyst_multiplier * tactician_multiplier

            return min(final_size, 1.5)  # Cap at 150% position size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1.0

    def _get_fallback_decision(self) -> dict[str, Any]:
        """Get fallback decision when models fail."""
        return {
            "action": "HOLD",
            "reason": "Fallback decision - models unavailable",
            "analyst_confidence": 0.0,
            "tactician_confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

    def get_system_info(self) -> dict[str, Any]:
        """Get information about the dual model system."""
        return {
            "analyst_timeframes": self.analyst_timeframes,
            "tactician_timeframes": self.tactician_timeframes,
            "analyst_confidence_threshold": self.analyst_confidence_threshold,
            "tactician_confidence_threshold": self.tactician_confidence_threshold,
            "enable_ensemble_analysis": self.enable_ensemble_analysis,
            "is_initialized": self.is_initialized,
            "analyst_model_loaded": self.analyst_model is not None,
            "tactician_model_loaded": self.tactician_model is not None,
            "ml_confidence_predictor_loaded": self.ml_confidence_predictor is not None,
            "description": "Dual model system for trading decisions",
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="dual model system cleanup",
    )
    async def stop(self) -> None:
        """Stop the dual model system."""
        self.logger.info("🛑 Stopping Dual Model System...")

        try:
            # Stop ML Confidence Predictor
            if self.ml_confidence_predictor:
                await self.ml_confidence_predictor.stop()

            # Clear models
            self.analyst_model = None
            self.tactician_model = None
            self.ml_confidence_predictor = None
            self.is_initialized = False

            self.logger.info("✅ Dual Model System stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping dual model system: {e}")


# Global dual model system instance
dual_model_system: DualModelSystem | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="dual model system setup",
)
async def setup_dual_model_system(
    config: dict[str, Any] | None = None,
) -> DualModelSystem | None:
    """
    Setup global dual model system.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[DualModelSystem]: Global dual model system instance
    """
    try:
        global dual_model_system

        if config is None:
            config = {
                "dual_model_system": {
                    "analyst_timeframes": ["1h", "15m", "5m", "1m"],
                    "tactician_timeframes": ["1m"],
                    "analyst_confidence_threshold": 0.7,
                    "tactician_confidence_threshold": 0.8,
                    "enable_ensemble_analysis": True,
                },
            }

        # Create dual model system
        dual_model_system = DualModelSystem(config)

        # Initialize dual model system
        success = await dual_model_system.initialize()
        if success:
            return dual_model_system
        return None

    except Exception as e:
        print(f"Error setting up dual model system: {e}")
        return None 