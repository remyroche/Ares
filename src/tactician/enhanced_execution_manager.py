# src/tactician/enhanced_execution_manager.py

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

    def _determine_tactician_direction(self, confidence: float) -> str:
        """Determine Tactician direction based on confidence."""
        # This would be based on the specific Tactician model outputs
        # For now, use a simple threshold-based approach
        if confidence > 0.7:
            return "long"
        elif confidence < 0.3:
            return "short"
        else:
            return "neutral"

    def _directions_agree(self, analyst_direction: str, tactician_direction: str) -> bool:
        """Check if Analyst and Tactician agree on trade direction."""
        if analyst_direction == "neutral" or tactician_direction == "neutral":
            return False
        return analyst_direction == tactician_direction

    @handle_errors(
        exceptions=(Exception,),
        default_return={"should_execute": False, "reason": "error"},
        context="enhanced_execution_manager.calculate_execution_parameters"
    )
    @with_tracing_span("EnhancedExecution.calculateParameters")
    def calculate_execution_parameters(
        self,
        market_data: pd.DataFrame,
        analyst_predictions: Dict[str, Any],
        tactician_predictions: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """Calculate execution parameters with high precision triple barrier strategy based on multi-outcome predictions."""
        try:
            # Validate predictions first
            validation = self.validate_analyst_predictions(analyst_predictions, tactician_predictions)
            if not validation["should_execute"]:
                return {
                    "should_execute": False,
                    "reason": validation["reason"],
                    "validation_details": validation
                }
            
            # Calculate dynamic barriers for the appropriate timeframe
            # Determine timeframe based on market data frequency or use primary timeframe
            timeframe = self._determine_timeframe(market_data)
            
            # Get dynamic barriers for this timeframe
            dynamic_upper, dynamic_lower = self.barrier_calculator.calculate_dynamic_barriers(
                timeframe=timeframe
            )
            
            # Calculate adaptive barriers based on market conditions
            volatility = self._calculate_volatility(market_data)
            adaptive_upper, adaptive_lower = self._calculate_adaptive_barriers(
                current_price, volatility, validation["trade_direction"], dynamic_upper, dynamic_lower
            )
            
            # Calculate position sizing with precision multiplier
            base_position_size = analyst_signal.get("position_size", 0.1)
            precision_position_size = base_position_size * self.position_size_multiplier
            
            # Calculate leverage with precision multiplier
            base_leverage = analyst_signal.get("leverage", 1.0)
            precision_leverage = base_leverage * self.leverage_multiplier
            
            # Calculate risk-adjusted parameters
            risk_adjusted_size = self._calculate_risk_adjusted_size(
                precision_position_size, adaptive_lower, current_price
            )
            
            # Calculate execution timing
            entry_timing = self._calculate_entry_timing(market_data, tactician_confidence)
            
            # Calculate precision score
            precision_score = self._calculate_precision_score(
                validation["combined_confidence"], volatility, market_data
            )
            
            execution_params = {
                "should_execute": True,
                "trade_direction": validation["trade_direction"],
                "entry_price": current_price,
                "upper_barrier_price": adaptive_upper,
                "lower_barrier_price": adaptive_lower,
                "position_size": risk_adjusted_size,
                "leverage": precision_leverage,
                "entry_timing": entry_timing,
                "precision_score": precision_score,
                "volatility": volatility,
                "analyst_confidence": validation["analyst_confidence"],
                "tactician_confidence": validation["tactician_confidence"],
                "combined_confidence": validation["combined_confidence"],
                "execution_reason": "high_precision_completion"
            }
            
            # Log execution parameters
            self.logger.info(f"🎯 High Precision Execution Parameters:")
            self.logger.info(f"   Direction: {execution_params['trade_direction']}")
            self.logger.info(f"   Entry Price: {execution_params['entry_price']:.4f}")
            self.logger.info(f"   Upper Barrier: {execution_params['upper_barrier_price']:.4f}")
            self.logger.info(f"   Lower Barrier: {execution_params['lower_barrier_price']:.4f}")
            self.logger.info(f"   Position Size: {execution_params['position_size']:.4f}")
            self.logger.info(f"   Precision Score: {execution_params['precision_score']:.3f}")
            
            return execution_params
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating execution parameters: {e}")
            return {
                "should_execute": False,
                "reason": "calculation_error",
                "error": str(e)
            }

    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate current market volatility."""
        try:
            if len(market_data) >= 20:
                returns = market_data["close"].pct_change().dropna().tail(20)
                return returns.std()
            return 0.01  # Default volatility
        except Exception:
            return 0.01

    def _determine_timeframe(self, market_data: pd.DataFrame) -> str:
        """Determine the timeframe based on market data frequency."""
        try:
            if market_data is None or len(market_data) < 2:
                return self.primary_timeframe
            
            # Calculate time difference between consecutive rows
            time_diff = market_data.index[1] - market_data.index[0]
            
            # Convert to minutes
            if hasattr(time_diff, 'total_seconds'):
                minutes_diff = time_diff.total_seconds() / 60
            else:
                # If not datetime index, assume 1m
                minutes_diff = 1
            
            # Determine timeframe based on frequency
            if minutes_diff <= 1.5:
                return "1m"
            elif minutes_diff <= 7.5:  # Allow some tolerance for 5m
                return "5m"
            else:
                return self.primary_timeframe
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error determining timeframe: {e}")
            return self.primary_timeframe

    def _calculate_adaptive_barriers(
        self, 
        current_price: float, 
        volatility: float, 
        direction: str,
        base_upper_pct: float,
        base_lower_pct: float
    ) -> Tuple[float, float]:
        """Calculate adaptive barriers based on volatility and direction using dynamic base values."""
        try:
            # Volatility adjustment
            volatility_multiplier = min(2.0, max(0.5, 1.0 / (volatility * 100)))
            
            # Direction adjustment
            if direction == "short":
                # For short positions, invert the barriers
                adaptive_upper = current_price * (1 - base_upper_pct * volatility_multiplier)
                adaptive_lower = current_price * (1 + base_lower_pct * volatility_multiplier)
            else:
                # For long positions, use standard barriers
                adaptive_upper = current_price * (1 + base_upper_pct * volatility_multiplier)
                adaptive_lower = current_price * (1 - base_lower_pct * volatility_multiplier)
            
            return adaptive_upper, adaptive_lower
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating adaptive barriers: {e}")
            # Fallback to base barriers
            base_upper = current_price * (1 + base_upper_pct)
            base_lower = current_price * (1 - base_lower_pct)
            return base_upper, base_lower

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

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the enhanced execution manager."""
        try:
            if not self.execution_history:
                return {
                    "total_executions": 0,
                    "success_rate": 0.0,
                    "avg_precision": 0.0,
                    "max_precision": 0.0
                }
            
            total_executions = len(self.execution_history)
            successful_executions = sum(
                1 for record in self.execution_history 
                if record.get("execution_params", {}).get("should_execute", False)
            )
            
            return {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": successful_executions / total_executions if total_executions > 0 else 0.0,
                "avg_precision": self.precision_metrics.get("avg_precision", 0.0),
                "max_precision": self.precision_metrics.get("max_precision", 0.0),
                "execution_count": self.precision_metrics.get("execution_count", 0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting performance summary: {e}")
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_precision": 0.0,
                "max_precision": 0.0,
                "error": str(e)
            }