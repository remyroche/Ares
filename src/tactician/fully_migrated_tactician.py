"""Fully Migrated Tactician.

This module implements a complete migration to the enhanced scenario-based prediction
system, replacing the old multi-output system entirely. All decision logic is now based
on fractal scenario analysis with comprehensive technical indicators.
"""

import asyncio
import logging
import os.path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors

from .enhanced_scenario_based_predictor import EnhancedScenarioBasedPredictor

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


class FullyMigratedTactician:
    """Fully migrated Tactician using only enhanced scenario-based predictions.

    This replaces the old multi-output system entirely with:
    - Fractal scenario analysis (17 scenarios: 8 profit, 8 risk, 1 neutral)
    - All step07 technical indicators
    - 15-minute look-ahead period
    - Complete step17 optimization
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize fully migrated Tactician.

        Args:
            config: Configuration dictionary with step17 optimization parameters
        """
        self.config = config
        self.logger = logger

        # Load step17 optimization parameters
        step17_config = config.get("step17_optimization", {})
        tactician_config = step17_config.get("fully_migrated_tactician", {})

        # Enhanced scenario predictor
        self.scenario_predictor = None

        # Decision thresholds (configurable for step17)
        self.decision_thresholds = {
            "entry_profit_threshold": tactician_config.get(
                "entry_profit_threshold", 0.6
            ),
            "entry_risk_threshold": tactician_config.get("entry_risk_threshold", 0.2),
            "entry_confidence_threshold": tactician_config.get(
                "entry_confidence_threshold", 0.7
            ),
            "entry_profit_risk_ratio": tactician_config.get(
                "entry_profit_risk_ratio", 2.0
            ),
            "entry_scenario_dominance": tactician_config.get(
                "entry_scenario_dominance", 0.4
            ),
            "exit_risk_threshold": tactician_config.get("exit_risk_threshold", 0.5),
            "exit_confidence_drop": tactician_config.get("exit_confidence_drop", 0.2),
            "position_size_multiplier": tactician_config.get(
                "position_size_multiplier", 1.0
            ),
            "leverage_multiplier": tactician_config.get("leverage_multiplier", 1.0),
        }

        # Risk management parameters (configurable for step17)
        self.risk_management = {
            "max_position_size": tactician_config.get("max_position_size", 0.1),
            "max_leverage": tactician_config.get("max_leverage", 3.0),
            "stop_loss_multiplier": tactician_config.get("stop_loss_multiplier", 1.0),
            "take_profit_multiplier": tactician_config.get(
                "take_profit_multiplier", 1.0
            ),
            "max_drawdown": tactician_config.get("max_drawdown", 0.05),
            "correlation_threshold": tactician_config.get("correlation_threshold", 0.8),
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
        }

        # State management
        self.is_initialized = False
        self.current_position = None
        self.position_history = []

    async def initialize(self) -> bool:
        """Initialize fully migrated Tactician.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Fully Migrated Tactician...")

            # Initialize enhanced scenario predictor
            self.scenario_predictor = EnhancedScenarioBasedPredictor(self.config)
            success = await self.scenario_predictor.initialize()

            if not success:
                self.logger.error("Failed to initialize enhanced scenario predictor")
                return False

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for fully migrated Tactician")
                return False

            self.is_initialized = True
            self.logger.info("✅ Fully Migrated Tactician initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Fully Migrated Tactician initialization failed: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate fully migrated Tactician configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate decision thresholds
            for threshold_name, threshold in self.decision_thresholds.items():
                if threshold < 0 or threshold > 1:
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

    @handles_errors
    async def generate_predictions(
        self,
        market_data: pd.DataFrame,
        analyst_barriers: Dict[str, float],
        symbol: str,
        timeframe: str,
        analyst_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Generate predictions using only enhanced scenario analysis.

        Args:
            market_data: Market data with OHLCV
            analyst_barriers: Analyst's barrier values (for reference)
            symbol: Trading symbol
            timeframe: Current timeframe
            analyst_confidence: Analyst's confidence score

        Returns:
            dict: Enhanced predictions and decisions
        """
        try:
            if not self.is_initialized:
                self.logger.error("Tactician not initialized")
                return self._generate_error_predictions(symbol, timeframe)

            # Extract comprehensive features
            features = self.scenario_predictor.extract_comprehensive_features(
                market_data
            )
            features = features.reshape(1, -1)  # Reshape for single prediction

            # Generate scenario predictions
            scenario_predictions = await self.scenario_predictor.predict_scenarios(
                features, market_data
            )

            # Make trading decisions
            trading_decisions = self._make_trading_decisions(
                scenario_predictions, analyst_confidence, market_data
            )

            # Calculate position sizing and leverage
            position_management = self._calculate_position_management(
                scenario_predictions, trading_decisions, analyst_barriers
            )

            result = {
                "scenario_predictions": scenario_predictions,
                "trading_decisions": trading_decisions,
                "position_management": position_management,
                "metadata": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_type": "fully_migrated_tactician",
                    "analyst_confidence": analyst_confidence,
                    "n_scenarios": len(self.scenario_predictor.scenarios),
                },
            }

            self.logger.info(f"Generated fully migrated predictions for {symbol}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Prediction generation failed: {e}")
            return self._generate_error_predictions(symbol, timeframe)

    def _make_trading_decisions(
        self,
        scenario_predictions: Dict[str, Any],
        analyst_confidence: float,
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Make trading decisions based on scenario analysis.

        Args:
            scenario_predictions: Scenario predictions
            analyst_confidence: Analyst's confidence score
            market_data: Market data

        Returns:
            dict: Trading decisions
        """
        try:
            scenario_analysis = scenario_predictions.get("scenario_analysis", {})
            confidence = scenario_predictions.get("confidence", 0.0)

            # Extract key metrics
            profit_zone_prob = scenario_analysis.get("profit_zone_probability", 0.0)
            risk_zone_prob = scenario_analysis.get("risk_zone_probability", 0.0)
            risk_reward_ratio = scenario_analysis.get("risk_reward_ratio", 0.0)
            scenario_dominance = scenario_analysis.get("scenario_dominance", 0.0)
            dominant_zone = scenario_analysis.get("dominant_zone", "neutral")

            # Entry decision logic
            entry_conditions = [
                profit_zone_prob > self.decision_thresholds["entry_profit_threshold"],
                risk_zone_prob < self.decision_thresholds["entry_risk_threshold"],
                confidence > self.decision_thresholds["entry_confidence_threshold"],
                risk_reward_ratio > self.decision_thresholds["entry_profit_risk_ratio"],
                scenario_dominance
                > self.decision_thresholds["entry_scenario_dominance"],
                dominant_zone == "profit",
                analyst_confidence > 0.5,  # Require some analyst confidence
            ]

            entry_signal = all(entry_conditions)

            # Exit decision logic (for existing positions)
            exit_signal = False
            if self.current_position:
                exit_conditions = [
                    risk_zone_prob > self.decision_thresholds["exit_risk_threshold"],
                    confidence
                    < (
                        self.current_position.get("entry_confidence", 0.0)
                        - self.decision_thresholds["exit_confidence_drop"]
                    ),
                    dominant_zone == "risk",
                ]
                exit_signal = any(exit_conditions)

            # Direction decision
            direction = (
                "LONG" if entry_signal and dominant_zone == "profit" else "NEUTRAL"
            )
            if exit_signal:
                direction = "EXIT"

            # Confidence scoring
            decision_confidence = self._calculate_decision_confidence(
                scenario_analysis, confidence, analyst_confidence
            )

            # Reasoning
            reasoning = self._generate_decision_reasoning(
                entry_signal,
                exit_signal,
                scenario_analysis,
                confidence,
                analyst_confidence,
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
                    "risk_reward_ratio": risk_reward_ratio,
                    "scenario_dominance": scenario_dominance,
                    "dominant_zone": dominant_zone,
                    "predicted_scenario": scenario_predictions.get(
                        "predicted_scenario", 16
                    ),
                    "scenario_name": scenario_predictions.get(
                        "scenario_name", "Neutral"
                    ),
                },
            }

        except Exception as e:
            self.logger.error(f"❌ Trading decision making failed: {e}")
            return {
                "entry_signal": False,
                "exit_signal": False,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": f"Error in decision making: {e}",
                "scenario_metrics": {},
            }

    def _calculate_position_management(
        self,
        scenario_predictions: Dict[str, Any],
        trading_decisions: Dict[str, Any],
        analyst_barriers: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calculate position sizing and leverage based on scenario analysis.

        Args:
            scenario_predictions: Scenario predictions
            trading_decisions: Trading decisions
            analyst_barriers: Analyst's barrier values

        Returns:
            dict: Position management parameters
        """
        try:
            scenario_analysis = scenario_predictions.get("scenario_analysis", {})
            confidence = scenario_predictions.get("confidence", 0.0)

            # Base position size from confidence
            base_position_size = confidence * self.risk_management["max_position_size"]

            # Adjust based on scenario dominance
            scenario_dominance = scenario_analysis.get("scenario_dominance", 0.0)
            dominance_multiplier = 1.0 + (scenario_dominance - 0.5) * 0.5

            # Adjust based on risk-reward ratio
            risk_reward_ratio = scenario_analysis.get("risk_reward_ratio", 1.0)
            ratio_multiplier = min(risk_reward_ratio / 2.0, 1.5)

            # Final position size
            position_size = base_position_size * dominance_multiplier * ratio_multiplier
            position_size = min(
                position_size, self.risk_management["max_position_size"]
            )

            # Leverage calculation
            base_leverage = 1.0 + (confidence - 0.5) * 2.0
            leverage = min(base_leverage, self.risk_management["max_leverage"])

            # Stop loss and take profit
            analyst_upper = analyst_barriers.get("upper_barrier", 0.02)
            analyst_lower = analyst_barriers.get("lower_barrier", -0.01)

            stop_loss = analyst_lower * self.risk_management["stop_loss_multiplier"]
            take_profit = analyst_upper * self.risk_management["take_profit_multiplier"]

            return {
                "position_size": position_size,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_metrics": {
                    "max_drawdown": self.risk_management["max_drawdown"],
                    "correlation_threshold": self.risk_management[
                        "correlation_threshold"
                    ],
                    "dominance_multiplier": dominance_multiplier,
                    "ratio_multiplier": ratio_multiplier,
                },
            }

        except Exception as e:
            self.logger.error(f"❌ Position management calculation failed: {e}")
            return {
                "position_size": 0.0,
                "leverage": 1.0,
                "stop_loss": -0.01,
                "take_profit": 0.02,
                "risk_metrics": {},
            }

    def _calculate_decision_confidence(
        self,
        scenario_analysis: Dict[str, Any],
        model_confidence: float,
        analyst_confidence: float,
    ) -> float:
        """Calculate decision confidence combining scenario analysis and analyst
        confidence.

        Args:
            scenario_analysis: Scenario analysis results
            model_confidence: Model confidence
            analyst_confidence: Analyst confidence

        Returns:
            float: Combined decision confidence
        """
        try:
            # Base confidence from model
            base_confidence = model_confidence

            # Boost from scenario dominance
            scenario_dominance = scenario_analysis.get("scenario_dominance", 0.0)
            dominance_boost = scenario_dominance * 0.2

            # Boost from risk-reward ratio
            risk_reward_ratio = scenario_analysis.get("risk_reward_ratio", 1.0)
            ratio_boost = min((risk_reward_ratio - 1.0) * 0.1, 0.2)

            # Analyst confidence boost
            analyst_boost = analyst_confidence * 0.1

            # Final confidence
            final_confidence = (
                base_confidence + dominance_boost + ratio_boost + analyst_boost
            )

            return np.clip(final_confidence, 0.0, 1.0)

        except Exception as e:
            self.logger.error(f"❌ Decision confidence calculation failed: {e}")
            return 0.5

    def _generate_decision_reasoning(
        self,
        entry_signal: bool,
        exit_signal: bool,
        scenario_analysis: Dict[str, Any],
        model_confidence: float,
        analyst_confidence: float,
    ) -> str:
        """Generate human-readable reasoning for decisions.

        Args:
            entry_signal: Entry signal
            exit_signal: Exit signal
            scenario_analysis: Scenario analysis results
            model_confidence: Model confidence
            analyst_confidence: Analyst confidence

        Returns:
            str: Decision reasoning
        """
        try:
            reasoning_parts = []

            if entry_signal:
                reasoning_parts.append(
                    "ENTRY SIGNAL: Strong scenario analysis indicates favorable conditions"
                )

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

            elif exit_signal:
                reasoning_parts.append("EXIT SIGNAL: Risk conditions detected")
                risk_prob = scenario_analysis.get("risk_zone_probability", 0.0)
                reasoning_parts.append(f"Risk probability: {risk_prob:.1%}")

            else:
                reasoning_parts.append("NO SIGNAL: Conditions not favorable for entry")
                dominant_zone = scenario_analysis.get("dominant_zone", "neutral")
                reasoning_parts.append(f"Dominant zone: {dominant_zone}")

            return " | ".join(reasoning_parts)

        except Exception as e:
            self.logger.error(f"❌ Decision reasoning generation failed: {e}")
            return f"Error generating reasoning: {e}"

    def _generate_error_predictions(
        self, symbol: str, timeframe: str
    ) -> Dict[str, Any]:
        """Generate error predictions when something goes wrong.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            dict: Error predictions
        """
        return {
            "scenario_predictions": {
                "probabilities": {i: 1.0 / 17 for i in range(17)},
                "predicted_scenario": 16,
                "scenario_name": "Neutral",
                "confidence": 0.0,
                "scenario_analysis": {
                    "profit_zone_probability": 0.0,
                    "risk_zone_probability": 0.0,
                    "neutral_probability": 1.0,
                    "dominant_zone": "neutral",
                    "risk_reward_ratio": 0.0,
                    "scenario_dominance": 0.0,
                },
                "metadata": {
                    "model_type": "fully_migrated_tactician_error",
                    "generation_timestamp": datetime.now().isoformat(),
                    "is_trained": False,
                },
            },
            "trading_decisions": {
                "entry_signal": False,
                "exit_signal": False,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": "Error in prediction generation",
                "scenario_metrics": {},
            },
            "position_management": {
                "position_size": 0.0,
                "leverage": 1.0,
                "stop_loss": -0.01,
                "take_profit": 0.02,
                "risk_metrics": {},
            },
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "generation_timestamp": datetime.now().isoformat(),
                "model_type": "fully_migrated_tactician_error",
                "analyst_confidence": 0.0,
                "n_scenarios": 17,
            },
        }

    def update_position(self, position_data: Dict[str, Any]) -> None:
        """Update current position information.

        Args:
            position_data: Position data
        """
        try:
            self.current_position = position_data
            self.position_history.append(
                {**position_data, "timestamp": datetime.now().isoformat()}
            )

            # Keep only last 100 positions
            if len(self.position_history) > 100:
                self.position_history = self.position_history[-100:]

        except Exception as e:
            self.logger.error(f"❌ Position update failed: {e}")

    def update_performance_metrics(self, trade_result: Dict[str, Any]) -> None:
        """Update performance metrics with trade result.

        Args:
            trade_result: Trade result data
        """
        try:
            self.performance_metrics["total_trades"] += 1

            if trade_result.get("profit", 0) > 0:
                self.performance_metrics["winning_trades"] += 1
                self.performance_metrics["total_profit"] += trade_result["profit"]
            else:
                self.performance_metrics["losing_trades"] += 1
                self.performance_metrics["total_loss"] += abs(
                    trade_result.get("profit", 0)
                )

            # Calculate derived metrics
            win_rate = self.performance_metrics["winning_trades"] / max(
                self.performance_metrics["total_trades"], 1
            )
            profit_factor = self.performance_metrics["total_profit"] / max(
                self.performance_metrics["total_loss"], 0.001
            )

            self.performance_metrics["win_rate"] = win_rate
            self.performance_metrics["profit_factor"] = profit_factor

        except Exception as e:
            self.logger.error(f"❌ Performance metrics update failed: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary.

        Returns:
            dict: Performance summary
        """
        return {
            "performance_metrics": self.performance_metrics,
            "current_position": self.current_position,
            "position_history_count": len(self.position_history),
            "is_initialized": self.is_initialized,
            "scenario_predictor_status": {
                "is_trained": (
                    self.scenario_predictor.is_trained
                    if self.scenario_predictor
                    else False
                ),
                "n_scenarios": (
                    len(self.scenario_predictor.scenarios)
                    if self.scenario_predictor
                    else 0
                ),
                "last_training_time": (
                    self.scenario_predictor.last_training_time.isoformat()
                    if self.scenario_predictor
                    and self.scenario_predictor.last_training_time
                    else None
                ),
            },
        }

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary for step17 optimization.

        Returns:
            dict: Configuration summary
        """
        return {
            "decision_thresholds": self.decision_thresholds,
            "risk_management": self.risk_management,
            "scenario_predictor_config": (
                self.scenario_predictor.get_enhanced_configuration_summary()
                if self.scenario_predictor
                else {}
            ),
            "is_initialized": self.is_initialized,
        }
