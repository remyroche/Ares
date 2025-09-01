"""
Step17 Optimized Tactician

This module implements a complete step17-optimized Tactician where ALL decision logic,
position sizing, leverage, and confidence calculations are configurable by step17.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

from .comprehensive_enhanced_scenario_predictor import ComprehensiveEnhancedScenarioPredictor

# Simple logger setup
logger = logging.getLogger(__name__)

# Simple error handling decorator
def handle_errors(func):
    """Simple error handling decorator."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper


class Step17OptimizedTactician:
    """
    Step17 Optimized Tactician with ALL decision logic configurable.

    This replaces the old multi-output system entirely with:
    - Fractal scenario analysis (17 scenarios: 8 profit, 8 risk, 1 neutral)
    - ALL technical indicators (50+ indicators, 350+ features)
    - 15-minute look-ahead period
    - COMPLETE step17 optimization for ALL parameters including decision logic
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize step17 optimized Tactician.

        Args:
            config: Configuration dictionary with step17 optimization parameters
        """
        self.config = config
        self.logger = logger

        # Load step17 optimization parameters
        step17_config = config.get("step17_optimization", {})
        tactician_config = step17_config.get("step17_optimized_tactician", {})

        # Comprehensive scenario predictor
        self.scenario_predictor = None

        # COMPREHENSIVE decision thresholds (ALL configurable by step17)
        self.decision_thresholds = {
            # Entry decision thresholds (ALL configurable)
            "entry_profit_threshold": tactician_config.get("entry_profit_threshold", 0.6),
            "entry_risk_threshold": tactician_config.get("entry_risk_threshold", 0.2),
            "entry_confidence_threshold": tactician_config.get("entry_confidence_threshold", 0.7),
            "entry_profit_risk_ratio": tactician_config.get("entry_profit_risk_ratio", 2.0),
            "entry_scenario_dominance": tactician_config.get("entry_scenario_dominance", 0.4),
            "entry_analyst_confidence_min": tactician_config.get("entry_analyst_confidence_min", 0.5),
            "entry_neutral_threshold": tactician_config.get("entry_neutral_threshold", 0.3),
            "entry_volatility_threshold": tactician_config.get("entry_volatility_threshold", 0.02),
            "entry_volume_threshold": tactician_config.get("entry_volume_threshold", 1.2),

            # Exit decision thresholds (ALL configurable)
            "exit_risk_threshold": tactician_config.get("exit_risk_threshold", 0.5),
            "exit_confidence_drop": tactician_config.get("exit_confidence_drop", 0.2),
            "exit_profit_threshold": tactician_config.get("exit_profit_threshold", 0.8),
            "exit_time_threshold": tactician_config.get("exit_time_threshold", 3600),  # seconds
            "exit_drawdown_threshold": tactician_config.get("exit_drawdown_threshold", 0.05),
            "exit_volatility_spike": tactician_config.get("exit_volatility_spike", 0.05),

            # Direction decision thresholds (ALL configurable)
            "direction_profit_bias": tactician_config.get("direction_profit_bias", 0.1),
            "direction_risk_bias": tactician_config.get("direction_risk_bias", 0.1),
            "direction_neutral_bias": tactician_config.get("direction_neutral_bias", 0.05),
            "direction_confidence_bias": tactician_config.get("direction_confidence_bias", 0.15),

            # Confidence calculation weights (ALL configurable)
            "confidence_base_weight": tactician_config.get("confidence_base_weight", 0.4),
            "confidence_scenario_dominance_weight": tactician_config.get("confidence_scenario_dominance_weight", 0.2),
            "confidence_risk_reward_weight": tactician_config.get("confidence_risk_reward_weight", 0.1),
            "confidence_analyst_weight": tactician_config.get("confidence_analyst_weight", 0.1),
            "confidence_volatility_weight": tactician_config.get("confidence_volatility_weight", 0.1),
            "confidence_volume_weight": tactician_config.get("confidence_volume_weight", 0.1),

            # Position sizing parameters (ALL configurable)
            "position_size_base_multiplier": tactician_config.get("position_size_base_multiplier", 1.0),
            "position_size_confidence_multiplier": tactician_config.get("position_size_confidence_multiplier", 1.5),
            "position_size_scenario_dominance_multiplier": tactician_config.get("position_size_scenario_dominance_multiplier", 1.2),
            "position_size_risk_reward_multiplier": tactician_config.get("position_size_risk_reward_multiplier", 1.3),
            "position_size_analyst_confidence_multiplier": tactician_config.get("position_size_analyst_confidence_multiplier", 1.1),
            "position_size_volatility_multiplier": tactician_config.get("position_size_volatility_multiplier", 0.8),
            "position_size_volume_multiplier": tactician_config.get("position_size_volume_multiplier", 1.1),

            # Leverage calculation parameters (ALL configurable)
            "leverage_base_multiplier": tactician_config.get("leverage_base_multiplier", 1.0),
            "leverage_confidence_multiplier": tactician_config.get("leverage_confidence_multiplier", 2.0),
            "leverage_scenario_dominance_multiplier": tactician_config.get("leverage_scenario_dominance_multiplier", 1.5),
            "leverage_risk_reward_multiplier": tactician_config.get("leverage_risk_reward_multiplier", 1.8),
            "leverage_analyst_confidence_multiplier": tactician_config.get("leverage_analyst_confidence_multiplier", 1.2),
            "leverage_volatility_multiplier": tactician_config.get("leverage_volatility_multiplier", 0.7),
            "leverage_volume_multiplier": tactician_config.get("leverage_volume_multiplier", 1.3),

            # Stop loss and take profit multipliers (ALL configurable)
            "stop_loss_base_multiplier": tactician_config.get("stop_loss_base_multiplier", 1.0),
            "stop_loss_confidence_multiplier": tactician_config.get("stop_loss_confidence_multiplier", 0.8),
            "stop_loss_volatility_multiplier": tactician_config.get("stop_loss_volatility_multiplier", 1.2),
            "stop_loss_risk_multiplier": tactician_config.get("stop_loss_risk_multiplier", 1.1),

            "take_profit_base_multiplier": tactician_config.get("take_profit_base_multiplier", 1.0),
            "take_profit_confidence_multiplier": tactician_config.get("take_profit_confidence_multiplier", 1.2),
            "take_profit_volatility_multiplier": tactician_config.get("take_profit_volatility_multiplier", 0.8),
            "take_profit_profit_multiplier": tactician_config.get("take_profit_profit_multiplier", 1.3),
        }

        # Risk management parameters (ALL configurable by step17)
        self.risk_management = {
            "max_position_size": tactician_config.get("max_position_size", 0.1),
            "max_leverage": tactician_config.get("max_leverage", 3.0),
            "max_drawdown": tactician_config.get("max_drawdown", 0.05),
            "correlation_threshold": tactician_config.get("correlation_threshold", 0.8),
            "volatility_cap": tactician_config.get("volatility_cap", 0.1),
            "volume_cap": tactician_config.get("volume_cap", 5.0),
            "confidence_cap": tactician_config.get("confidence_cap", 0.95),
            "scenario_dominance_cap": tactician_config.get("scenario_dominance_cap", 0.9),
            "risk_reward_cap": tactician_config.get("risk_reward_cap", 5.0),
        }

        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_trade_duration": 0.0,
            "avg_profit_per_trade": 0.0,
            "avg_loss_per_trade": 0.0,
        }

        # State management
        self.is_initialized = False
        self.current_position = None
        self.position_history = []

    async def initialize(self) -> bool:
        """
        Initialize step17 optimized Tactician.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Step17 Optimized Tactician...")

            # Initialize comprehensive scenario predictor
            self.scenario_predictor = ComprehensiveEnhancedScenarioPredictor(self.config)
            success = await self.scenario_predictor.initialize()

            if not success:
                self.logger.error("Failed to initialize comprehensive scenario predictor")
                return False

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for step17 optimized Tactician")
                return False

            self.is_initialized = True
            self.logger.info("✅ Step17 Optimized Tactician initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Step17 Optimized Tactician initialization failed: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """
        Validate step17 optimized Tactician configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate decision thresholds
            for threshold_name, threshold in self.decision_thresholds.items():
                if threshold < 0:
                    self.logger.error(f"Invalid threshold for {threshold_name}")
                    return False

            # Validate risk management parameters
            if self.risk_management["max_position_size"] <= 0:
                self.logger.error("Invalid max_position_size")
                return False

            if self.risk_management["max_leverage"] <= 0:
                self.logger.error("Invalid max_leverage")
                return False

            return True

        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False

    @handle_errors
    async def generate_predictions(
        self,
        market_data: pd.DataFrame,
        analyst_barriers: Dict[str, float],
        symbol: str,
        timeframe: str,
        analyst_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate predictions using comprehensive scenario analysis with step17 optimization.

        Args:
            market_data: Market data with OHLCV
            analyst_barriers: Analyst's barrier values (for reference)
            symbol: Trading symbol
            timeframe: Current timeframe
            analyst_confidence: Analyst's confidence score

        Returns:
            dict: Comprehensive predictions and decisions
        """
        try:
            if not self.is_initialized:
                self.logger.error("Tactician not initialized")
                return self._generate_error_predictions(symbol, timeframe)

            # Extract comprehensive features
            features = self.scenario_predictor.extract_comprehensive_features(market_data)
            features = features.reshape(1, -1)  # Reshape for single prediction

            # Generate scenario predictions
            scenario_predictions = await self.scenario_predictor.predict_scenarios(
                features, market_data
            )

            # Make step17-optimized trading decisions
            trading_decisions = self._make_step17_optimized_decisions(
                scenario_predictions, analyst_confidence, market_data
            )

            # Calculate step17-optimized position sizing and leverage
            position_management = self._calculate_step17_optimized_position_management(
                scenario_predictions, trading_decisions, analyst_barriers, analyst_confidence, market_data
            )

            result = {
                "scenario_predictions": scenario_predictions,
                "trading_decisions": trading_decisions,
                "position_management": position_management,
                "metadata": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_type": "step17_optimized_tactician",
                    "analyst_confidence": analyst_confidence,
                    "n_scenarios": len(self.scenario_predictor.scenarios),
                    "n_features": 350
                }
            }

            self.logger.info(f"Generated step17 optimized predictions for {symbol}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Prediction generation failed: {e}")
            return self._generate_error_predictions(symbol, timeframe)

    def _make_step17_optimized_decisions(
        self,
        scenario_predictions: Dict[str, Any],
        analyst_confidence: float,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Make step17-optimized trading decisions with ALL logic configurable.

        Args:
            scenario_predictions: Scenario predictions
            analyst_confidence: Analyst's confidence score
            market_data: Market data

        Returns:
            dict: Step17-optimized trading decisions
        """
        try:
            scenario_analysis = scenario_predictions.get("scenario_analysis", {})
            confidence = scenario_predictions.get("confidence", 0.0)

            # Extract key metrics
            profit_zone_prob = scenario_analysis.get("profit_zone_probability", 0.0)
            risk_zone_prob = scenario_analysis.get("risk_zone_probability", 0.0)
            neutral_prob = scenario_analysis.get("neutral_probability", 0.0)
            risk_reward_ratio = scenario_analysis.get("risk_reward_ratio", 0.0)
            scenario_dominance = scenario_analysis.get("scenario_dominance", 0.0)
            dominant_zone = scenario_analysis.get("dominant_zone", "neutral")

            # Calculate volatility and volume metrics
            volatility = self._calculate_volatility(market_data)
            volume_ratio = self._calculate_volume_ratio(market_data)

            # Step17-optimized entry decision logic (ALL configurable)
            entry_conditions = [
                profit_zone_prob > self.decision_thresholds["entry_profit_threshold"],
                risk_zone_prob < self.decision_thresholds["entry_risk_threshold"],
                confidence > self.decision_thresholds["entry_confidence_threshold"],
                risk_reward_ratio > self.decision_thresholds["entry_profit_risk_ratio"],
                scenario_dominance > self.decision_thresholds["entry_scenario_dominance"],
                analyst_confidence > self.decision_thresholds["entry_analyst_confidence_min"],
                neutral_prob < self.decision_thresholds["entry_neutral_threshold"],
                volatility < self.decision_thresholds["entry_volatility_threshold"],
                volume_ratio > self.decision_thresholds["entry_volume_threshold"],
                dominant_zone == "profit"
            ]

            entry_signal = all(entry_conditions)

            # Step17-optimized exit decision logic (ALL configurable)
            exit_signal = False
            if self.current_position:
                exit_conditions = [
                    risk_zone_prob > self.decision_thresholds["exit_risk_threshold"],
                    confidence < (self.current_position.get("entry_confidence", 0.0) - self.decision_thresholds["exit_confidence_drop"]),
                    profit_zone_prob > self.decision_thresholds["exit_profit_threshold"],
                    self._check_exit_time_threshold(),
                    self._check_exit_drawdown_threshold(),
                    volatility > self.decision_thresholds["exit_volatility_spike"],
                    dominant_zone == "risk"
                ]
                exit_signal = any(exit_conditions)

            # Step17-optimized direction decision (ALL configurable)
            direction = self._calculate_step17_optimized_direction(
                profit_zone_prob, risk_zone_prob, neutral_prob, confidence, dominant_zone
            )

            # Step17-optimized confidence calculation
            decision_confidence = self._calculate_step17_optimized_confidence(
                scenario_analysis, confidence, analyst_confidence, volatility, volume_ratio
            )

            # Step17-optimized reasoning
            reasoning = self._generate_step17_optimized_reasoning(
                entry_signal, exit_signal, scenario_analysis, confidence, analyst_confidence,
                volatility, volume_ratio
            )

            return {
                "entry_signal": entry_signal,
                "exit_signal": exit_signal,
                "direction": direction,
                "confidence": decision_confidence,
                "reasoning": reasoning,
                "scenario_metrics": {
                    "profit_zone_probability": profit_zone_prob,
                    "risk_zone_probability": risk_zone_prob,
                    "neutral_probability": neutral_prob,
                    "risk_reward_ratio": risk_reward_ratio,
                    "scenario_dominance": scenario_dominance,
                    "dominant_zone": dominant_zone,
                    "volatility": volatility,
                    "volume_ratio": volume_ratio,
                    "predicted_scenario": scenario_predictions.get("predicted_scenario", 16),
                    "scenario_name": scenario_predictions.get("scenario_name", "Neutral")
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Step17 optimized decision making failed: {e}")
            return {
                "entry_signal": False,
                "exit_signal": False,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": f"Error in step17 optimized decision making: {e}",
                "scenario_metrics": {}
            }

    def _calculate_step17_optimized_position_management(
        self,
        scenario_predictions: Dict[str, Any],
        trading_decisions: Dict[str, Any],
        analyst_barriers: Dict[str, float],
        analyst_confidence: float,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate step17-optimized position sizing and leverage with ALL parameters configurable.

        Args:
            scenario_predictions: Scenario predictions
            trading_decisions: Trading decisions
            analyst_barriers: Analyst's barrier values
            analyst_confidence: Analyst's confidence score
            market_data: Market data

        Returns:
            dict: Step17-optimized position management parameters
        """
        try:
            scenario_analysis = scenario_predictions.get("scenario_analysis", {})
            confidence = scenario_predictions.get("confidence", 0.0)

            # Calculate volatility and volume metrics
            volatility = self._calculate_volatility(market_data)
            volume_ratio = self._calculate_volume_ratio(market_data)

            # Step17-optimized position size calculation (ALL configurable)
            base_position_size = self.risk_management["max_position_size"]

            # Apply step17-optimized multipliers
            position_size = base_position_size * self.decision_thresholds["position_size_base_multiplier"]
            position_size *= (1 + confidence * self.decision_thresholds["position_size_confidence_multiplier"])
            position_size *= (1 + scenario_analysis.get("scenario_dominance", 0.0) * self.decision_thresholds["position_size_scenario_dominance_multiplier"])
            position_size *= (1 + scenario_analysis.get("risk_reward_ratio", 1.0) * self.decision_thresholds["position_size_risk_reward_multiplier"])
            position_size *= (1 + analyst_confidence * self.decision_thresholds["position_size_analyst_confidence_multiplier"])
            position_size *= (1 - volatility * self.decision_thresholds["position_size_volatility_multiplier"])
            position_size *= (1 + volume_ratio * self.decision_thresholds["position_size_volume_multiplier"])

            # Apply caps
            position_size = min(position_size, self.risk_management["max_position_size"])
            position_size = max(position_size, 0.0)

            # Step17-optimized leverage calculation (ALL configurable)
            base_leverage = 1.0

            # Apply step17-optimized multipliers
            leverage = base_leverage * self.decision_thresholds["leverage_base_multiplier"]
            leverage *= (1 + confidence * self.decision_thresholds["leverage_confidence_multiplier"])
            leverage *= (1 + scenario_analysis.get("scenario_dominance", 0.0) * self.decision_thresholds["leverage_scenario_dominance_multiplier"])
            leverage *= (1 + scenario_analysis.get("risk_reward_ratio", 1.0) * self.decision_thresholds["leverage_risk_reward_multiplier"])
            leverage *= (1 + analyst_confidence * self.decision_thresholds["leverage_analyst_confidence_multiplier"])
            leverage *= (1 - volatility * self.decision_thresholds["leverage_volatility_multiplier"])
            leverage *= (1 + volume_ratio * self.decision_thresholds["leverage_volume_multiplier"])

            # Apply caps
            leverage = min(leverage, self.risk_management["max_leverage"])
            leverage = max(leverage, 1.0)

            # Step17-optimized stop loss and take profit (ALL configurable)
            analyst_upper = analyst_barriers.get("upper_barrier", 0.02)
            analyst_lower = analyst_barriers.get("lower_barrier", -0.01)

            # Apply step17-optimized multipliers
            stop_loss = analyst_lower * self.decision_thresholds["stop_loss_base_multiplier"]
            stop_loss *= (1 - confidence * self.decision_thresholds["stop_loss_confidence_multiplier"])
            stop_loss *= (1 + volatility * self.decision_thresholds["stop_loss_volatility_multiplier"])
            stop_loss *= (1 + risk_zone_prob * self.decision_thresholds["stop_loss_risk_multiplier"])

            take_profit = analyst_upper * self.decision_thresholds["take_profit_base_multiplier"]
            take_profit *= (1 + confidence * self.decision_thresholds["take_profit_confidence_multiplier"])
            take_profit *= (1 - volatility * self.decision_thresholds["take_profit_volatility_multiplier"])
            take_profit *= (1 + profit_zone_prob * self.decision_thresholds["take_profit_profit_multiplier"])

            return {
                "position_size": position_size,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_metrics": {
                    "max_drawdown": self.risk_management["max_drawdown"],
                    "correlation_threshold": self.risk_management["correlation_threshold"],
                    "volatility_cap": self.risk_management["volatility_cap"],
                    "volume_cap": self.risk_management["volume_cap"],
                    "confidence_cap": self.risk_management["confidence_cap"],
                    "scenario_dominance_cap": self.risk_management["scenario_dominance_cap"],
                    "risk_reward_cap": self.risk_management["risk_reward_cap"],
                    "calculated_volatility": volatility,
                    "calculated_volume_ratio": volume_ratio
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Step17 optimized position management calculation failed: {e}")
            return {
                "position_size": 0.0,
                "leverage": 1.0,
                "stop_loss": -0.01,
                "take_profit": 0.02,
                "risk_metrics": {}
            }

    def _calculate_step17_optimized_direction(
        self,
        profit_zone_prob: float,
        risk_zone_prob: float,
        neutral_prob: float,
        confidence: float,
        dominant_zone: str
    ) -> str:
        """
        Calculate step17-optimized direction with ALL logic configurable.

        Args:
            profit_zone_prob: Profit zone probability
            risk_zone_prob: Risk zone probability
            neutral_prob: Neutral probability
            confidence: Model confidence
            dominant_zone: Dominant zone

        Returns:
            str: Step17-optimized direction
        """
        try:
            # Step17-optimized direction calculation (ALL configurable)
            if dominant_zone == "profit" and profit_zone_prob > self.decision_thresholds["direction_profit_bias"]:
                return "LONG"
            elif dominant_zone == "risk" and risk_zone_prob > self.decision_thresholds["direction_risk_bias"]:
                return "SHORT"
            elif neutral_prob > self.decision_thresholds["direction_neutral_bias"]:
                return "NEUTRAL"
            elif confidence > self.decision_thresholds["direction_confidence_bias"]:
                return "LONG" if profit_zone_prob > risk_zone_prob else "SHORT"
            else:
                return "NEUTRAL"

        except Exception as e:
            self.logger.error(f"❌ Step17 optimized direction calculation failed: {e}")
            return "NEUTRAL"

    def _calculate_step17_optimized_confidence(
        self,
        scenario_analysis: Dict[str, Any],
        model_confidence: float,
        analyst_confidence: float,
        volatility: float,
        volume_ratio: float
    ) -> float:
        """
        Calculate step17-optimized confidence with ALL weights configurable.

        Args:
            scenario_analysis: Scenario analysis results
            model_confidence: Model confidence
            analyst_confidence: Analyst confidence
            volatility: Volatility metric
            volume_ratio: Volume ratio metric

        Returns:
            float: Step17-optimized confidence
        """
        try:
            # Step17-optimized confidence calculation (ALL weights configurable)
            confidence = 0.0

            # Base confidence
            confidence += model_confidence * self.decision_thresholds["confidence_base_weight"]

            # Scenario dominance boost
            scenario_dominance = scenario_analysis.get("scenario_dominance", 0.0)
            confidence += scenario_dominance * self.decision_thresholds["confidence_scenario_dominance_weight"]

            # Risk-reward ratio boost
            risk_reward_ratio = scenario_analysis.get("risk_reward_ratio", 1.0)
            confidence += min(risk_reward_ratio / 2.0, 1.0) * self.decision_thresholds["confidence_risk_reward_weight"]

            # Analyst confidence boost
            confidence += analyst_confidence * self.decision_thresholds["confidence_analyst_weight"]

            # Volatility adjustment
            volatility_factor = 1.0 - min(volatility / 0.05, 1.0)
            confidence += volatility_factor * self.decision_thresholds["confidence_volatility_weight"]

            # Volume adjustment
            volume_factor = min(volume_ratio / 2.0, 1.0)
            confidence += volume_factor * self.decision_thresholds["confidence_volume_weight"]

            # Apply caps
            confidence = min(confidence, self.risk_management["confidence_cap"])
            confidence = max(confidence, 0.0)

            return confidence

        except Exception as e:
            self.logger.error(f"❌ Step17 optimized confidence calculation failed: {e}")
            return 0.5

    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility metric for step17 optimization."""
        try:
            returns = market_data['close'].pct_change().dropna()
            return returns.std()
        except Exception as e:
            self.logger.error(f"❌ Volatility calculation failed: {e}")
            return 0.02

    def _calculate_volume_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate volume ratio metric for step17 optimization."""
        try:
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            return current_volume / avg_volume if avg_volume > 0 else 1.0
        except Exception as e:
            self.logger.error(f"❌ Volume ratio calculation failed: {e}")
            return 1.0

    def _check_exit_time_threshold(self) -> bool:
        """Check if exit time threshold is met."""
        try:
            if self.current_position:
                entry_time = self.current_position.get("entry_time")
                if entry_time:
                    elapsed_time = (datetime.now() - entry_time).total_seconds()
                    return elapsed_time > self.decision_thresholds["exit_time_threshold"]
            return False
        except Exception as e:
            self.logger.error(f"❌ Exit time threshold check failed: {e}")
            return False

    def _check_exit_drawdown_threshold(self) -> bool:
        """Check if exit drawdown threshold is met."""
        try:
            if self.current_position:
                entry_price = self.current_position.get("entry_price", 0)
                current_price = self.current_position.get("current_price", 0)
                if entry_price > 0 and current_price > 0:
                    drawdown = (entry_price - current_price) / entry_price
                    return drawdown > self.decision_thresholds["exit_drawdown_threshold"]
            return False
        except Exception as e:
            self.logger.error(f"❌ Exit drawdown threshold check failed: {e}")
            return False

    def _generate_step17_optimized_reasoning(
        self,
        entry_signal: bool,
        exit_signal: bool,
        scenario_analysis: Dict[str, Any],
        model_confidence: float,
        analyst_confidence: float,
        volatility: float,
        volume_ratio: float
    ) -> str:
        """
        Generate step17-optimized reasoning with ALL logic configurable.

        Args:
            entry_signal: Entry signal
            exit_signal: Exit signal
            scenario_analysis: Scenario analysis results
            model_confidence: Model confidence
            analyst_confidence: Analyst confidence
            volatility: Volatility metric
            volume_ratio: Volume ratio metric

        Returns:
            str: Step17-optimized reasoning
        """
        try:
            reasoning_parts = []

            if entry_signal:
                reasoning_parts.append("ENTRY SIGNAL: Step17-optimized conditions met")

                profit_prob = scenario_analysis.get("profit_zone_probability", 0.0)
                risk_prob = scenario_analysis.get("risk_zone_probability", 0.0)
                risk_reward = scenario_analysis.get("risk_reward_ratio", 0.0)
                dominance = scenario_analysis.get("scenario_dominance", 0.0)

                reasoning_parts.append(f"Profit probability: {profit_prob:.1%}")
                reasoning_parts.append(f"Risk probability: {risk_prob:.1%}")
                reasoning_parts.append(f"Risk-reward ratio: {risk_reward:.2f}")
                reasoning_parts.append(f"Scenario dominance: {dominance:.1%}")
                reasoning_parts.append(f"Model confidence: {model_confidence:.1%}")
                reasoning_parts.append(f"Analyst confidence: {analyst_confidence:.1%}")
                reasoning_parts.append(f"Volatility: {volatility:.3f}")
                reasoning_parts.append(f"Volume ratio: {volume_ratio:.2f}")

            elif exit_signal:
                reasoning_parts.append("EXIT SIGNAL: Step17-optimized risk conditions detected")
                risk_prob = scenario_analysis.get("risk_zone_probability", 0.0)
                reasoning_parts.append(f"Risk probability: {risk_prob:.1%}")

            else:
                reasoning_parts.append("NO SIGNAL: Step17-optimized conditions not favorable")
                dominant_zone = scenario_analysis.get("dominant_zone", "neutral")
                reasoning_parts.append(f"Dominant zone: {dominant_zone}")

            return " | ".join(reasoning_parts)

        except Exception as e:
            self.logger.error(f"❌ Step17 optimized reasoning generation failed: {e}")
            return f"Error generating step17 optimized reasoning: {e}"

    def _generate_error_predictions(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Generate error predictions when something goes wrong."""
        return {
            "scenario_predictions": {
                "probabilities": {i: 1.0/17 for i in range(17)},
                "predicted_scenario": 16,
                "scenario_name": "Neutral",
                "confidence": 0.0,
                "scenario_analysis": {
                    "profit_zone_probability": 0.0,
                    "risk_zone_probability": 0.0,
                    "neutral_probability": 1.0,
                    "dominant_zone": "neutral",
                    "risk_reward_ratio": 0.0,
                    "scenario_dominance": 0.0
                },
                "metadata": {
                    "model_type": "step17_optimized_tactician_error",
                    "generation_timestamp": datetime.now().isoformat(),
                    "is_trained": False
                }
            },
            "trading_decisions": {
                "entry_signal": False,
                "exit_signal": False,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": "Error in step17 optimized prediction generation",
                "scenario_metrics": {}
            },
            "position_management": {
                "position_size": 0.0,
                "leverage": 1.0,
                "stop_loss": -0.01,
                "take_profit": 0.02,
                "risk_metrics": {}
            },
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "generation_timestamp": datetime.now().isoformat(),
                "model_type": "step17_optimized_tactician_error",
                "analyst_confidence": 0.0,
                "n_scenarios": 17,
                "n_features": 350
            }
        }

    def update_position(self, position_data: Dict[str, Any]) -> None:
        """Update current position information."""
        try:
            self.current_position = position_data
            self.position_history.append({
                **position_data,
                "timestamp": datetime.now().isoformat()
            })

            # Keep only last 100 positions
            if len(self.position_history) > 100:
                self.position_history = self.position_history[-100:]

        except Exception as e:
            self.logger.error(f"❌ Position update failed: {e}")

    def update_performance_metrics(self, trade_result: Dict[str, Any]) -> None:
        """Update performance metrics with trade result."""
        try:
            self.performance_metrics["total_trades"] += 1

            if trade_result.get("profit", 0) > 0:
                self.performance_metrics["winning_trades"] += 1
                self.performance_metrics["total_profit"] += trade_result["profit"]
            else:
                self.performance_metrics["losing_trades"] += 1
                self.performance_metrics["total_loss"] += abs(trade_result.get("profit", 0))

            # Calculate derived metrics
            total_trades = self.performance_metrics["total_trades"]
            if total_trades > 0:
                self.performance_metrics["win_rate"] = self.performance_metrics["winning_trades"] / total_trades
                self.performance_metrics["avg_profit_per_trade"] = self.performance_metrics["total_profit"] / self.performance_metrics["winning_trades"] if self.performance_metrics["winning_trades"] > 0 else 0.0
                self.performance_metrics["avg_loss_per_trade"] = self.performance_metrics["total_loss"] / self.performance_metrics["losing_trades"] if self.performance_metrics["losing_trades"] > 0 else 0.0
                self.performance_metrics["profit_factor"] = self.performance_metrics["total_profit"] / max(self.performance_metrics["total_loss"], 0.001)

        except Exception as e:
            self.logger.error(f"❌ Performance metrics update failed: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "performance_metrics": self.performance_metrics,
            "current_position": self.current_position,
            "position_history_count": len(self.position_history),
            "is_initialized": self.is_initialized,
            "scenario_predictor_status": {
                "is_trained": self.scenario_predictor.is_trained if self.scenario_predictor else False,
                "n_scenarios": len(self.scenario_predictor.scenarios) if self.scenario_predictor else 0,
                "last_training_time": self.scenario_predictor.last_training_time.isoformat() if self.scenario_predictor and self.scenario_predictor.last_training_time else None
            }
        }

    def get_step17_configuration_summary(self) -> Dict[str, Any]:
        """
        Get step17 configuration summary with ALL parameters.

        Returns:
            dict: Complete step17 configuration summary
        """
        return {
            "decision_thresholds": self.decision_thresholds,
            "risk_management": self.risk_management,
            "scenario_predictor_config": self.scenario_predictor.get_comprehensive_configuration_summary() if self.scenario_predictor else {},
            "is_initialized": self.is_initialized,
            "total_configurable_parameters": len(self.decision_thresholds) + len(self.risk_management) + 50  # Approximate count
        }