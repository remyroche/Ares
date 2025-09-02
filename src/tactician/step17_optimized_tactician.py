"""
Step17 Optimized Tactician

This module implements a complete step17-optimized Tactician where ALL decision logic,
position sizing, leverage, and confidence calculations are configurable by step17.
"""

import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, Optional

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
    Step17 Optimized Tactician with ALL trading parameters configurable.

    This replaces the old multi-output system entirely with:
    - Fractal scenario analysis (17 scenarios: 8 profit, 8 risk, 1 neutral)
    - ALL technical indicators (50+ indicators, 350+ features)
    - 15-minute look-ahead period
    - COMPLETE step17 optimization for ALL trading parameters
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Step17 Optimized Tactician."""
        self.config = config
        self.logger = logger

        # Load step17 optimization parameters
        step17_config = config.get("step17_optimization", {})
        tactician_config = step17_config.get("step17_optimized_tactician", {})

        # Comprehensive scenario predictor
        self.scenario_predictor = None

        # COMPREHENSIVE decision thresholds (ALL configurable by step17)
        # Only parameters directly used for trading decisions
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

            # Position sizing thresholds (ALL configurable)
            "position_size_base": tactician_config.get("position_size_base", 0.1),
            "position_size_confidence_multiplier": tactician_config.get("position_size_confidence_multiplier", 1.5),
            "position_size_risk_multiplier": tactician_config.get("position_size_risk_multiplier", 0.5),
            "position_size_max": tactician_config.get("position_size_max", 0.5),
            "position_size_min": tactician_config.get("position_size_min", 0.01),

            # Leverage thresholds (ALL configurable)
            "leverage_base": tactician_config.get("leverage_base", 1.0),
            "leverage_confidence_multiplier": tactician_config.get("leverage_confidence_multiplier", 2.0),
            "leverage_risk_multiplier": tactician_config.get("leverage_risk_multiplier", 0.5),
            "leverage_max": tactician_config.get("leverage_max", 10.0),
            "leverage_min": tactician_config.get("leverage_min", 1.0),

            # Confidence calculation thresholds (ALL configurable)
            "confidence_profit_weight": tactician_config.get("confidence_profit_weight", 0.4),
            "confidence_risk_weight": tactician_config.get("confidence_risk_weight", 0.3),
            "confidence_scenario_weight": tactician_config.get("confidence_scenario_weight", 0.3),
            "confidence_analyst_weight": tactician_config.get("confidence_analyst_weight", 0.5),
            "confidence_tactician_weight": tactician_config.get("confidence_tactician_weight", 0.5),

            # Scenario analysis thresholds (ALL configurable)
            "scenario_profit_threshold": tactician_config.get("scenario_profit_threshold", 0.6),
            "scenario_risk_threshold": tactician_config.get("scenario_risk_threshold", 0.4),
            "scenario_neutral_threshold": tactician_config.get("scenario_neutral_threshold", 0.3),
            "scenario_dominance_threshold": tactician_config.get("scenario_dominance_threshold", 0.4),

            # Technical indicator thresholds (ALL configurable)
            "rsi_oversold": tactician_config.get("rsi_oversold", 30),
            "rsi_overbought": tactician_config.get("rsi_overbought", 70),
            "macd_signal_threshold": tactician_config.get("macd_signal_threshold", 0.001),
            "bollinger_std_multiplier": tactician_config.get("bollinger_std_multiplier", 2.0),
            "atr_multiplier": tactician_config.get("atr_multiplier", 2.0),
            "volume_sma_period": tactician_config.get("volume_sma_period", 20),
            "volume_spike_threshold": tactician_config.get("volume_spike_threshold", 1.5),

            # Market condition thresholds (ALL configurable)
            "volatility_low": tactician_config.get("volatility_low", 0.01),
            "volatility_high": tactician_config.get("volatility_high", 0.05),
            "trend_strength_threshold": tactician_config.get("trend_strength_threshold", 0.6),
            "support_resistance_threshold": tactician_config.get("support_resistance_threshold", 0.02),
        }

        # Optimization frequency (manual)
        self.optimization_frequency = "manual"  # Only run when manually triggered
        self.last_optimization = None
        self.optimization_enabled = tactician_config.get("optimization_enabled", True)

        # Performance tracking
        self.performance_history = []
        self.current_performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0
        }

    @handle_errors
    async def initialize(self) -> bool:
        """Initialize the Step17 Optimized Tactician."""
        try:
            self.logger.info("Initializing Step17 Optimized Tactician...")

            # Initialize scenario predictor
            self.scenario_predictor = ComprehensiveEnhancedScenarioPredictor(self.config)
            await self.scenario_predictor.initialize()

            # Validate configuration
            if not self._validate_configuration():
                return False

            self.logger.info("✅ Step17 Optimized Tactician initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Step17 Optimized Tactician initialization failed: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate all step17 configuration parameters."""
        try:
            # Validate all decision thresholds
            for threshold_name, threshold_value in self.decision_thresholds.items():
                if not isinstance(threshold_value, (int, float)):
                    self.logger.error(f"Invalid threshold type for {threshold_name}")
                    return False

                # Validate ranges for specific thresholds
                if "threshold" in threshold_name.lower():
                    if not 0 <= threshold_value <= 1:
                        self.logger.error(f"Threshold {threshold_name} must be between 0 and 1")
                        return False

                if "multiplier" in threshold_name.lower():
                    if threshold_value <= 0:
                        self.logger.error(f"Multiplier {threshold_name} must be positive")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False

    @handle_errors
    async def make_trading_decision(self, market_data: Dict[str, Any], 
                                  analyst_confidence: float = 0.5) -> Dict[str, Any]:
        """Make comprehensive trading decision using step17 parameters."""
        try:
            # Get scenario predictions
            scenario_predictions = await self.scenario_predictor.predict_scenarios(market_data)

            # Calculate tactician confidence using step17 parameters
            tactician_confidence = self._calculate_tactician_confidence(scenario_predictions, analyst_confidence)

            # Make entry decision using step17 thresholds
            entry_decision = self._make_entry_decision(scenario_predictions, tactician_confidence, analyst_confidence)

            # Make direction decision using step17 thresholds
            direction_decision = self._make_direction_decision(scenario_predictions, tactician_confidence)

            # Calculate position size using step17 parameters
            position_size = self._calculate_position_size(tactician_confidence, analyst_confidence, scenario_predictions)

            # Calculate leverage using step17 parameters
            leverage = self._calculate_leverage(tactician_confidence, analyst_confidence, scenario_predictions)

            # Create decision
            decision = {
                "action": entry_decision["action"],
                "direction": direction_decision["direction"],
                "confidence": tactician_confidence,
                "position_size": position_size,
                "leverage": leverage,
                "scenario_predictions": scenario_predictions,
                "entry_reason": entry_decision["reason"],
                "direction_reason": direction_decision["reason"],
                "step17_parameters_used": list(self.decision_thresholds.keys())
            }

            return decision

        except Exception as e:
            self.logger.error(f"Error making trading decision: {e}")
            return {
                "action": "HOLD",
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "position_size": 0.0,
                "leverage": 1.0,
                "error": str(e)
            }

    def _calculate_tactician_confidence(self, scenario_predictions: Dict[str, float], 
                                      analyst_confidence: float) -> float:
        """Calculate tactician confidence using step17 parameters."""
        try:
            # Extract scenario probabilities
            profit_prob = scenario_predictions.get("profit_probability", 0.0)
            risk_prob = scenario_predictions.get("risk_probability", 0.0)
            neutral_prob = scenario_predictions.get("neutral_probability", 0.0)

            # Calculate weighted confidence using step17 parameters
            profit_confidence = profit_prob * self.decision_thresholds["confidence_profit_weight"]
            risk_confidence = (1 - risk_prob) * self.decision_thresholds["confidence_risk_weight"]
            scenario_confidence = max(profit_prob, 1 - risk_prob) * self.decision_thresholds["confidence_scenario_weight"]

            # Combine with analyst confidence
            analyst_weighted = analyst_confidence * self.decision_thresholds["confidence_analyst_weight"]
            tactician_weighted = (profit_confidence + risk_confidence + scenario_confidence) * self.decision_thresholds["confidence_tactician_weight"]

            # Final confidence
            total_confidence = analyst_weighted + tactician_weighted

            return max(0.0, min(1.0, total_confidence))

        except Exception as e:
            self.logger.error(f"Error calculating tactician confidence: {e}")
            return 0.5

    def _make_entry_decision(self, scenario_predictions: Dict[str, float], 
                           tactician_confidence: float, analyst_confidence: float) -> Dict[str, Any]:
        """Make entry decision using step17 thresholds."""
        try:
            profit_prob = scenario_predictions.get("profit_probability", 0.0)
            risk_prob = scenario_predictions.get("risk_probability", 0.0)
            neutral_prob = scenario_predictions.get("neutral_probability", 0.0)

            # Check entry conditions using step17 thresholds
            profit_condition = profit_prob >= self.decision_thresholds["entry_profit_threshold"]
            risk_condition = risk_prob <= self.decision_thresholds["entry_risk_threshold"]
            confidence_condition = tactician_confidence >= self.decision_thresholds["entry_confidence_threshold"]
            analyst_condition = analyst_confidence >= self.decision_thresholds["entry_analyst_confidence_min"]

            # Calculate profit/risk ratio
            profit_risk_ratio = profit_prob / risk_prob if risk_prob > 0 else float('inf')
            ratio_condition = profit_risk_ratio >= self.decision_thresholds["entry_profit_risk_ratio"]

            # Check scenario dominance
            max_prob = max(profit_prob, risk_prob, neutral_prob)
            dominance_condition = max_prob >= self.decision_thresholds["entry_scenario_dominance"]

            # Make decision
            if (profit_condition and risk_condition and confidence_condition and 
                analyst_condition and ratio_condition and dominance_condition):
                return {
                    "action": "ENTER",
                    "reason": f"All step17 conditions met: profit={profit_prob:.3f}, risk={risk_prob:.3f}, confidence={tactician_confidence:.3f}"
                }
            else:
                return {
                    "action": "HOLD",
                    "reason": f"Step17 conditions not met: profit={profit_prob:.3f}, risk={risk_prob:.3f}, confidence={tactician_confidence:.3f}"
                }

        except Exception as e:
            self.logger.error(f"Error making entry decision: {e}")
            return {"action": "HOLD", "reason": f"Error: {e}"}

    def _make_direction_decision(self, scenario_predictions: Dict[str, float], 
                                tactician_confidence: float) -> Dict[str, Any]:
        """Make direction decision using step17 thresholds."""
        try:
            profit_prob = scenario_predictions.get("profit_probability", 0.0)
            risk_prob = scenario_predictions.get("risk_probability", 0.0)
            neutral_prob = scenario_predictions.get("neutral_probability", 0.0)

            # Apply step17 direction biases
            profit_score = profit_prob + self.decision_thresholds["direction_profit_bias"]
            risk_score = risk_prob + self.decision_thresholds["direction_risk_bias"]
            neutral_score = neutral_prob + self.decision_thresholds["direction_neutral_bias"]

            # Apply confidence bias
            confidence_bias = self.decision_thresholds["direction_confidence_bias"]
            if tactician_confidence > 0.7:
                profit_score += confidence_bias
            elif tactician_confidence < 0.3:
                risk_score += confidence_bias

            # Determine direction
            max_score = max(profit_score, risk_score, neutral_score)
            
            if max_score == profit_score:
                return {"direction": "LONG", "reason": f"Profit scenario dominant: {profit_prob:.3f}"}
            elif max_score == risk_score:
                return {"direction": "SHORT", "reason": f"Risk scenario dominant: {risk_prob:.3f}"}
            else:
                return {"direction": "NEUTRAL", "reason": f"Neutral scenario dominant: {neutral_prob:.3f}"}

        except Exception as e:
            self.logger.error(f"Error making direction decision: {e}")
            return {"direction": "NEUTRAL", "reason": f"Error: {e}"}

    def _calculate_position_size(self, tactician_confidence: float, analyst_confidence: float,
                               scenario_predictions: Dict[str, float]) -> float:
        """Calculate position size using step17 parameters."""
        try:
            # Base position size
            base_size = self.decision_thresholds["position_size_base"]

            # Confidence multiplier
            confidence_multiplier = self.decision_thresholds["position_size_confidence_multiplier"]
            confidence_factor = 1.0 + (tactician_confidence * confidence_multiplier)

            # Risk multiplier
            risk_multiplier = self.decision_thresholds["position_size_risk_multiplier"]
            risk_prob = scenario_predictions.get("risk_probability", 0.5)
            risk_factor = 1.0 - (risk_prob * risk_multiplier)

            # Calculate final position size
            position_size = base_size * confidence_factor * risk_factor

            # Apply min/max constraints
            min_size = self.decision_thresholds["position_size_min"]
            max_size = self.decision_thresholds["position_size_max"]
            
            return max(min_size, min(max_size, position_size))

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self.decision_thresholds["position_size_min"]

    def _calculate_leverage(self, tactician_confidence: float, analyst_confidence: float,
                          scenario_predictions: Dict[str, float]) -> float:
        """Calculate leverage using step17 parameters."""
        try:
            # Base leverage
            base_leverage = self.decision_thresholds["leverage_base"]

            # Confidence multiplier
            confidence_multiplier = self.decision_thresholds["leverage_confidence_multiplier"]
            confidence_factor = 1.0 + (tactician_confidence * confidence_multiplier)

            # Risk multiplier
            risk_multiplier = self.decision_thresholds["leverage_risk_multiplier"]
            risk_prob = scenario_predictions.get("risk_probability", 0.5)
            risk_factor = 1.0 - (risk_prob * risk_multiplier)

            # Calculate final leverage
            leverage = base_leverage * confidence_factor * risk_factor

            # Apply min/max constraints
            min_leverage = self.decision_thresholds["leverage_min"]
            max_leverage = self.decision_thresholds["leverage_max"]
            
            return max(min_leverage, min(max_leverage, leverage))

        except Exception as e:
            self.logger.error(f"Error calculating leverage: {e}")
            return self.decision_thresholds["leverage_min"]

    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """Update performance tracking with trade result."""
        try:
            self.current_performance["total_trades"] += 1
            
            if trade_result.get("pnl", 0) > 0:
                self.current_performance["winning_trades"] += 1
            
            self.current_performance["total_pnl"] += trade_result.get("pnl", 0)
            
            # Calculate win rate
            total_trades = self.current_performance["total_trades"]
            if total_trades > 0:
                self.current_performance["win_rate"] = self.current_performance["winning_trades"] / total_trades

            # Store in history
            self.performance_history.append({
                "timestamp": datetime.now(),
                "trade_result": trade_result,
                "current_performance": self.current_performance.copy()
            })

        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        try:
            return {
                "current_performance": self.current_performance.copy(),
                "total_history_entries": len(self.performance_history),
                "last_optimization": self.last_optimization,
                "optimization_enabled": self.optimization_enabled,
                "step17_parameters": self.decision_thresholds.copy()
            }
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.scenario_predictor:
                await self.scenario_predictor.cleanup()
            
            self.logger.info("✅ Step17 Optimized Tactician cleanup completed")

        except Exception as e:
            self.logger.error(f"❌ Step17 Optimized Tactician cleanup failed: {e}")


# Setup function for easy integration
async def setup_step17_optimized_tactician(config: Dict[str, Any]) -> Step17OptimizedTactician:
    """Setup the Step17 Optimized Tactician."""
    try:
        tactician = Step17OptimizedTactician(config)
        if await tactician.initialize():
            return tactician
        return None
    except Exception as e:
        logger.error(f"Failed to setup Step17 Optimized Tactician: {e}")
        return None