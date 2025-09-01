# src/tactician/enhanced_execution_manager.py

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.centralized_decorators import (
    guard_dataframe_nulls,
    handle_errors,
    with_tracing_span,
)
from src.utils.logger import get_logger


class EnhancedExecutionManager:
    """Enhanced execution manager for Tactician with high precision triple barrier completion.

    This manager ensures the Tactician completes the Analyst nicely by:
    1. Using smaller triple barriers (50% and 25% of Analyst barriers)
    2. Implementing high precision execution filters
    3. Requiring Analyst signal agreement
    4. Providing adaptive risk management
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the enhanced execution manager."""
        self.config = config.get("tactician_triple_barrier", {})
        self.logger = get_logger("EnhancedExecutionManager")

        # Load configuration
        self._load_config()

        # Performance tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.precision_metrics: Dict[str, float] = {}

    def _load_config(self) -> None:
        """Load configuration for high precision execution."""
        # Import dynamic barrier calculator
        from src.tactician.dynamic_barrier_calculator import DynamicBarrierCalculator

        # Initialize dynamic barrier calculator
        self.barrier_calculator = DynamicBarrierCalculator(self.config)

        # Get dynamic barriers for primary timeframe (1m)
        self.upper_barrier_pct, self.lower_barrier_pct = self.barrier_calculator.calculate_dynamic_barriers(
            timeframe="1m"
        )

        # Precision Settings
        self.precision_threshold = self.config.get("precision_threshold", 0.85)
        self.min_signal_strength = self.config.get("min_signal_strength", 0.8)

        # Risk Management
        self.max_risk_per_trade = self.config.get("max_risk_per_trade", 0.001)
        self.position_size_multiplier = self.config.get("position_size_multiplier", 0.5)
        self.leverage_multiplier = self.config.get("leverage_multiplier", 0.75)

        # Integration Settings
        self.analyst_signal_requirement = self.config.get("analyst_signal_requirement", True)
        self.direction_agreement_required = self.config.get("direction_agreement_required", True)
        self.confidence_boost_threshold = self.config.get("confidence_boost_threshold", 0.9)

        # Execution Timing
        self.entry_delay_seconds = self.config.get("entry_delay_seconds", 5)
        self.max_execution_time = self.config.get("max_execution_time", 30)

        # Timeframe settings
        self.timeframes = self.config.get("timeframes", ["1m", "5m"])
        self.primary_timeframe = self.config.get("primary_timeframe", "1m")
        self.secondary_timeframe = self.config.get("secondary_timeframe", "5m")

        self.logger.info(f"🔧 Enhanced Execution Manager Configuration (Dynamic):")
        self.logger.info(f"   Timeframes: {self.timeframes}")
        self.logger.info(f"   Primary: {self.primary_timeframe}, Secondary: {self.secondary_timeframe}")
        self.logger.info(f"   Dynamic Upper Barrier: {self.upper_barrier_pct:.4f} ({self.upper_barrier_pct*100:.3f}%)")
        self.logger.info(f"   Dynamic Lower Barrier: {self.lower_barrier_pct:.4f} ({self.lower_barrier_pct*100:.3f}%)")
        self.logger.info(f"   Precision Threshold: {self.precision_threshold}")
        self.logger.info(f"   Position Size Multiplier: {self.position_size_multiplier}")

    @handle_errors(
        exceptions=(Exception,),
        default_return={"should_execute": False, "reason": "error"},
        context="enhanced_execution_manager.validate_analyst_signal"
    )
    @with_tracing_span("EnhancedExecution.validateAnalystSignal")
    def validate_analyst_predictions(
        self,
        analyst_predictions: Dict[str, Any],
        tactician_predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate Analyst predictions and Tactician predictions for execution."""
        try:
            if not analyst_predictions or not tactician_predictions:
                return {
                    "valid": False,
                    "reason": "missing_predictions",
                    "should_execute": False
                }

            # Extract key predictions from Analyst
            analyst_price_pred = analyst_predictions.get("price_prediction", {}).get("prediction", 0.0)
            analyst_confidence = analyst_predictions.get("confidence_prediction", {}).get("prediction", 0.5)
            analyst_regime = analyst_predictions.get("regime_prediction", {}).get("prediction", "unknown")

            # Extract key predictions from Tactician
            tactician_price_pred = tactician_predictions.get("price_prediction", {}).get("prediction", 0.0)
            tactician_confidence = tactician_predictions.get("confidence_prediction", {}).get("prediction", 0.5)
            tactician_regime = tactician_predictions.get("regime_prediction", {}).get("prediction", "unknown")

            # Check if predictions are valid
            if analyst_confidence < 0.5 or tactician_confidence < 0.5:
                return {
                    "valid": False,
                    "should_execute": False,
                    "reason": "insufficient_confidence",
                    "analyst_confidence": analyst_confidence,
                    "tactician_confidence": tactician_confidence
                }

            # Determine trade direction based on price predictions
            if analyst_price_pred > 0 and tactician_price_pred > 0:
                trade_direction = "long"
            elif analyst_price_pred < 0 and tactician_price_pred < 0:
                trade_direction = "short"
            else:
                return {
                    "valid": False,
                    "should_execute": False,
                    "reason": "conflicting_price_predictions",
                    "analyst_price": analyst_price_pred,
                    "tactician_price": tactician_price_pred
                }

            # Calculate combined confidence
            combined_confidence = (analyst_confidence + tactician_confidence) / 2

            if combined_confidence < self.precision_threshold:
                return {
                    "valid": False,
                    "should_execute": False,
                    "reason": "insufficient_combined_confidence",
                    "analyst_confidence": analyst_confidence,
                    "tactician_confidence": tactician_confidence,
                    "combined_confidence": combined_confidence
                }

            return {
                "valid": True,
                "should_execute": True,
                "trade_direction": trade_direction,
                "analyst_confidence": analyst_confidence,
                "tactician_confidence": tactician_confidence,
                "combined_confidence": combined_confidence,
                "analyst_regime": analyst_regime,
                "tactician_regime": tactician_regime
            }

        except Exception as e:
            self.logger.error(f"❌ Error validating predictions: {e}")
            return {
                "valid": False,
                "reason": "validation_error",
                "error": str(e),
                "should_execute": False
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return={"should_execute": False, "reason": "error"},
        context="enhanced_execution_manager.calculate_execution_parameters"
    )
    @with_tracing_span("EnhancedExecution.calculateParameters")
    # Removed dynamic adaptation methods - ML model handles market conditions

    def _calculate_risk_adjusted_size(
        self,
        base_size: float,
        stop_loss_price: float,
        current_price: float
    ) -> float:
        """Calculate risk-adjusted position size."""
        try:
            # Calculate risk per unit
            risk_per_unit = abs(current_price - stop_loss_price) / current_price

            # Calculate maximum position size based on risk limit
            max_size = self.max_risk_per_trade / risk_per_unit

            # Use the smaller of base size and max size
            return min(base_size, max_size)

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating risk-adjusted size: {e}")
            return base_size

    def _calculate_entry_timing(self, market_data: pd.DataFrame, confidence: float) -> Dict[str, Any]:
        """Calculate optimal entry timing."""
        try:
            # Simple timing based on confidence and market conditions
            if confidence > self.confidence_boost_threshold:
                delay = 0  # Immediate execution for high confidence
            else:
                delay = self.entry_delay_seconds

            return {
                "entry_delay_seconds": delay,
                "max_execution_time": self.max_execution_time,
                "confidence_based_timing": True
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating entry timing: {e}")
            return {
                "entry_delay_seconds": self.entry_delay_seconds,
                "max_execution_time": self.max_execution_time,
                "confidence_based_timing": False
            }

    def _calculate_precision_score(
        self,
        combined_confidence: float,
        volatility: float,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate precision score for execution quality."""
        try:
            # Base precision from confidence
            base_precision = combined_confidence

            # Volatility penalty (higher volatility = lower precision)
            volatility_penalty = min(0.2, volatility * 10)

            # Market condition bonus (if recent price action is favorable)
            market_bonus = 0.0
            if len(market_data) >= 5:
                recent_returns = market_data["close"].pct_change().tail(5).mean()
                if abs(recent_returns) < 0.001:  # Low volatility period
                    market_bonus = 0.05

            precision_score = base_precision - volatility_penalty + market_bonus
            return max(0.0, min(1.0, precision_score))

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating precision score: {e}")
            return combined_confidence

    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "reason": "error"},
        context="enhanced_execution_manager.execute_trade"
    )
    @with_tracing_span("EnhancedExecution.executeTrade")
    async def execute_trade(
        self,
        execution_params: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute trade with high precision parameters."""
        try:
            if not execution_params.get("should_execute", False):
                return {
                    "success": False,
                    "reason": execution_params.get("reason", "no_execution_params")
                }

            # Simulate trade execution (replace with actual execution logic)
            execution_time = datetime.now()

            # Record execution in history
            execution_record = {
                "timestamp": execution_time,
                "execution_params": execution_params,
                "market_conditions": {
                    "volatility": execution_params.get("volatility", 0.0),
                    "price": execution_params.get("entry_price", 0.0),
                    "volume": market_data["volume"].iloc[-1] if "volume" in market_data.columns else 0.0
                }
            }
            self.execution_history.append(execution_record)

            # Update precision metrics
            self._update_precision_metrics(execution_params)

            self.logger.info(f"✅ High Precision Trade Executed:")
            self.logger.info(f"   Direction: {execution_params['trade_direction']}")
            self.logger.info(f"   Entry Price: {execution_params['entry_price']:.4f}")
            self.logger.info(f"   Position Size: {execution_params['position_size']:.4f}")
            self.logger.info(f"   Precision Score: {execution_params['precision_score']:.3f}")

            return {
                "success": True,
                "execution_time": execution_time.isoformat(),
                "execution_params": execution_params,
                "reason": "high_precision_completion"
            }

        except Exception as e:
            self.logger.error(f"❌ Error executing trade: {e}")
            return {
                "success": False,
                "reason": "execution_error",
                "error": str(e)
            }

    def _update_precision_metrics(self, execution_params: Dict[str, Any]) -> None:
        """Update precision metrics for performance tracking."""
        try:
            precision_score = execution_params.get("precision_score", 0.0)

            # Update running averages
            if "avg_precision" not in self.precision_metrics:
                self.precision_metrics["avg_precision"] = precision_score
                self.precision_metrics["execution_count"] = 1
            else:
                current_avg = self.precision_metrics["avg_precision"]
                count = self.precision_metrics["execution_count"]
                new_avg = (current_avg * count + precision_score) / (count + 1)
                self.precision_metrics["avg_precision"] = new_avg
                self.precision_metrics["execution_count"] = count + 1

            # Update max precision
            if "max_precision" not in self.precision_metrics:
                self.precision_metrics["max_precision"] = precision_score
            else:
                self.precision_metrics["max_precision"] = max(
                    self.precision_metrics["max_precision"], precision_score
                )

        except Exception as e:
            self.logger.warning(f"⚠️ Error updating precision metrics: {e}")
