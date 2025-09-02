# src/tactician/sr_weight_optimizer.py

"""
SR Weight Optimizer for optimizing support/resistance breakout prediction weights.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor, ensure_optimized_sr_config
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import (
    failed,
    invalid,
    warning,
)


@dataclass
class WeightOptimizationResult:
    """Result of weight optimization backtesting."""

    weights: Dict[str, float]
    performance_metrics: Dict[str, float]
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    optimization_score: float
    backtest_periods: int
    confidence_level: float


class SRWeightOptimizer:
    """
    SR Weight Optimizer for optimizing support/resistance breakout prediction weights.

    Features:
    - Online learning with incremental updates
    - Multi-objective optimization (total PnL, Sharpe ratio, win rate)
    - Performance metrics calculation
    - Result validation and ranking
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SR Weight Optimizer."""
        self.config = config
        self.logger = system_logger.getChild("SRWeightOptimizer")

        # Configuration
        self.optimizer_config = config.get("sr_weight_optimizer", {})
        self.max_iterations = self.optimizer_config.get("max_iterations", 100)
        self.min_confidence = self.optimizer_config.get("min_confidence", 0.6)
        self.backtest_periods = self.optimizer_config.get("backtest_periods", 30)

        # Online learning configuration
        self.online_config = self.optimizer_config.get("online_learning", {})
        self.learning_rate = self.online_config.get("learning_rate", 0.01)
        self.update_frequency = self.online_config.get("update_frequency", 10)  # updates every 10 trades
        self.momentum = self.online_config.get("momentum", 0.9)
        self.decay_rate = self.online_config.get("decay_rate", 0.95)

        # Multi-objective optimization weights
        self.objective_weights = self.optimizer_config.get("objective_weights", {
            "total_pnl": 0.4,
            "sharpe_ratio": 0.3,
            "win_rate": 0.3
        })

        # Component managers
        self.sr_predictor = None

        # Optimization state
        self.optimization_results: List[WeightOptimizationResult] = []
        self.best_weights: Optional[Dict[str, float]] = None
        self.optimization_history: List[Dict[str, Any]] = []
        self.current_weights: Dict[str, float] = {}
        self.weight_gradients: Dict[str, float] = {}
        self.update_count = 0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="SR weight optimizer initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the SR Weight Optimizer."""
        try:
            self.logger.info("Initializing SR Weight Optimizer...")

            # Initialize SR predictor
            self.sr_predictor = await setup_sr_breakout_predictor(self.config)
            if not self.sr_predictor:
                self.logger.error("Failed to initialize SR predictor")
                return False

            # Initialize default weights
            self.current_weights = self._initialize_default_weights()

            # Validate configuration
            if not self._validate_configuration():
                return False

            self.logger.info("✅ SR Weight Optimizer initialized successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ SR Weight Optimizer initialization failed: {e}"))
            return False

    def _initialize_default_weights(self) -> Dict[str, float]:
        """Initialize default weights for SR components."""
        return {
            "fractal_weight": 0.25,
            "volume_weight": 0.25,
            "pivot_weight": 0.25,
            "atr_weight": 0.25,
            "touch_count_weight": 0.2,
            "total_volume_weight": 0.2,
            "level_age_weight": 0.2,
            "bounce_rate_weight": 0.2,
            "isolation_score_weight": 0.2
        }

    def _validate_configuration(self) -> bool:
        """Validate optimizer configuration."""
        try:
            # Validate learning parameters
            if not 0 < self.learning_rate < 1:
                self.logger.error(invalid("Learning rate must be between 0 and 1"))
                return False

            if not 0 < self.momentum < 1:
                self.logger.error(invalid("Momentum must be between 0 and 1"))
                return False

            if not 0 < self.decay_rate < 1:
                self.logger.error(invalid("Decay rate must be between 0 and 1"))
                return False

            # Validate objective weights
            total_weight = sum(self.objective_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                self.logger.error(invalid("Objective weights must sum to 1.0"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Configuration validation failed: {e}"))
            return False

    async def update_weights_online(self, trade_result: Dict[str, Any]) -> None:
        """Update weights using online learning from trade results."""
        try:
            self.update_count += 1

            # Calculate performance metrics for this trade
            performance_metrics = self._calculate_trade_performance(trade_result)

            # Calculate multi-objective score
            objective_score = self._calculate_multi_objective_score(performance_metrics)

            # Calculate gradients for each weight
            gradients = self._calculate_weight_gradients(trade_result, objective_score)

            # Update weights using online learning
            self._update_weights_with_gradients(gradients)

            # Apply momentum and decay
            self._apply_momentum_and_decay()

            # Store optimization history
            self._store_optimization_step(trade_result, performance_metrics, objective_score)

            # Update SR predictor with new weights
            await self._update_sr_predictor_weights()

            if self.update_count % self.update_frequency == 0:
                self.logger.info(f"✅ Online weight update #{self.update_count} completed")

        except Exception as e:
            self.logger.error(failed(f"❌ Error in online weight update: {e}"))

    def _calculate_trade_performance(self, trade_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for a trade."""
        try:
            pnl = trade_result.get("pnl", 0.0)
            duration = trade_result.get("duration", 1.0)
            initial_capital = trade_result.get("initial_capital", 1.0)

            # Calculate metrics
            total_pnl = pnl
            sharpe_ratio = pnl / (duration ** 0.5) if duration > 0 else 0.0
            win_rate = 1.0 if pnl > 0 else 0.0

            return {
                "total_pnl": total_pnl,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Error calculating trade performance: {e}"))
            return {"total_pnl": 0.0, "sharpe_ratio": 0.0, "win_rate": 0.0}

    def _calculate_multi_objective_score(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate multi-objective optimization score."""
        try:
            # Normalize metrics to [0, 1] range
            normalized_pnl = max(0, min(1, performance_metrics["total_pnl"] / 1000))  # Normalize to 1000 max
            normalized_sharpe = max(0, min(1, performance_metrics["sharpe_ratio"] / 2))  # Normalize to 2 max
            normalized_win_rate = performance_metrics["win_rate"]  # Already 0-1

            # Calculate weighted score
            score = (
                self.objective_weights["total_pnl"] * normalized_pnl +
                self.objective_weights["sharpe_ratio"] * normalized_sharpe +
                self.objective_weights["win_rate"] * normalized_win_rate
            )

            return score

        except Exception as e:
            self.logger.error(failed(f"❌ Error calculating multi-objective score: {e}"))
            return 0.0

    def _calculate_weight_gradients(self, trade_result: Dict[str, Any], objective_score: float) -> Dict[str, float]:
        """Calculate gradients for weight updates."""
        try:
            gradients = {}

            # Calculate gradients for each weight
            for weight_name in self.current_weights.keys():
                # Simple gradient calculation based on performance
                # In a more sophisticated implementation, this would use backpropagation
                if objective_score > 0.5:  # Good performance
                    # Reinforce current weights
                    gradients[weight_name] = self.learning_rate * (1 - objective_score)
                else:  # Poor performance
                    # Adjust weights away from current values
                    gradients[weight_name] = -self.learning_rate * objective_score

            return gradients

        except Exception as e:
            self.logger.error(failed(f"❌ Error calculating weight gradients: {e}"))
            return {}

    def _update_weights_with_gradients(self, gradients: Dict[str, float]) -> None:
        """Update weights using calculated gradients."""
        try:
            for weight_name, gradient in gradients.items():
                if weight_name in self.current_weights:
                    # Apply gradient update
                    self.current_weights[weight_name] += gradient

                    # Ensure weights stay in valid range [0, 1]
                    self.current_weights[weight_name] = max(0, min(1, self.current_weights[weight_name]))

            # Normalize weights to sum to 1
            self._normalize_weights()

        except Exception as e:
            self.logger.error(failed(f"❌ Error updating weights: {e}"))

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        try:
            total_weight = sum(self.current_weights.values())
            if total_weight > 0:
                for weight_name in self.current_weights:
                    self.current_weights[weight_name] /= total_weight

        except Exception as e:
            self.logger.error(failed(f"❌ Error normalizing weights: {e}"))

    def _apply_momentum_and_decay(self) -> None:
        """Apply momentum and decay to weight updates."""
        try:
            # Apply momentum to gradients
            for weight_name in self.weight_gradients:
                if weight_name in self.current_weights:
                    self.current_weights[weight_name] = (
                        self.momentum * self.current_weights[weight_name] +
                        (1 - self.momentum) * self.weight_gradients[weight_name]
                    )

            # Apply decay to learning rate
            self.learning_rate *= self.decay_rate

        except Exception as e:
            self.logger.error(failed(f"❌ Error applying momentum and decay: {e}"))

    def _store_optimization_step(self, trade_result: Dict[str, Any], 
                                performance_metrics: Dict[str, float], 
                                objective_score: float) -> None:
        """Store optimization step in history."""
        try:
            step_data = {
                "timestamp": trade_result.get("timestamp", ""),
                "trade_id": trade_result.get("trade_id", ""),
                "performance_metrics": performance_metrics,
                "objective_score": objective_score,
                "current_weights": self.current_weights.copy(),
                "learning_rate": self.learning_rate
            }
            self.optimization_history.append(step_data)

        except Exception as e:
            self.logger.error(failed(f"❌ Error storing optimization step: {e}"))

    async def _update_sr_predictor_weights(self) -> None:
        """Update SR predictor with new optimized weights."""
        try:
            if self.sr_predictor and hasattr(self.sr_predictor, 'update_weights'):
                await self.sr_predictor.update_weights(self.current_weights)
                self.logger.info("✅ SR predictor weights updated")

        except Exception as e:
            self.logger.error(failed(f"❌ Error updating SR predictor weights: {e}"))

    async def run_backtest_optimization(self, historical_data: pd.DataFrame) -> WeightOptimizationResult:
        """Run backtest optimization to find optimal weights."""
        try:
            self.logger.info("Running backtest optimization...")

            # Initialize optimization
            best_score = -float('inf')
            best_weights = None
            best_metrics = {}

            # Run multiple optimization iterations
            for iteration in range(self.max_iterations):
                # Generate candidate weights
                candidate_weights = self._generate_candidate_weights()
                
                # Run backtest with candidate weights
                performance_metrics = await self._run_backtest_with_weights(
                    historical_data, candidate_weights
                )
                
                # Calculate optimization score
                score = self._calculate_multi_objective_score(performance_metrics)
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_weights = candidate_weights.copy()
                    best_metrics = performance_metrics

            # Create optimization result
            result = WeightOptimizationResult(
                weights=best_weights or self.current_weights,
                performance_metrics=best_metrics,
                sharpe_ratio=best_metrics.get("sharpe_ratio", 0.0),
                max_drawdown=best_metrics.get("max_drawdown", 0.0),
                win_rate=best_metrics.get("win_rate", 0.0),
                profit_factor=best_metrics.get("profit_factor", 0.0),
                total_return=best_metrics.get("total_return", 0.0),
                optimization_score=best_score,
                backtest_periods=self.backtest_periods,
                confidence_level=min(best_score, 1.0)
            )

            # Store result
            self.optimization_results.append(result)
            
            # Update current weights if result is better
            if best_score > self.min_confidence:
                self.current_weights = best_weights.copy()
                self.best_weights = best_weights.copy()

            self.logger.info(f"✅ Backtest optimization completed with score: {best_score:.4f}")
            return result

        except Exception as e:
            self.logger.error(failed(f"❌ Error in backtest optimization: {e}"))
            return WeightOptimizationResult(
                weights=self.current_weights,
                performance_metrics={},
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_return=0.0,
                optimization_score=0.0,
                backtest_periods=0,
                confidence_level=0.0
            )

    def _generate_candidate_weights(self) -> Dict[str, float]:
        """Generate candidate weights for optimization."""
        try:
            # Generate random weights around current values
            candidate_weights = {}
            for weight_name, current_weight in self.current_weights.items():
                # Add random perturbation
                perturbation = np.random.normal(0, 0.1)
                candidate_weights[weight_name] = max(0, min(1, current_weight + perturbation))
            
            # Normalize to sum to 1
            total_weight = sum(candidate_weights.values())
            if total_weight > 0:
                for weight_name in candidate_weights:
                    candidate_weights[weight_name] /= total_weight
            
            return candidate_weights

        except Exception as e:
            self.logger.error(failed(f"❌ Error generating candidate weights: {e}"))
            return self.current_weights.copy()

    async def _run_backtest_with_weights(self, historical_data: pd.DataFrame, 
                                       weights: Dict[str, float]) -> Dict[str, float]:
        """Run backtest simulation with given weights."""
        try:
            # This is a simplified backtest - in practice, you'd implement full backtesting logic
            # For now, return mock performance metrics
            return {
                "total_return": np.random.normal(0.05, 0.1),
                "sharpe_ratio": np.random.normal(0.5, 0.3),
                "win_rate": np.random.uniform(0.4, 0.7),
                "max_drawdown": np.random.uniform(0.05, 0.15),
                "profit_factor": np.random.uniform(0.8, 1.5)
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Error running backtest: {e}"))
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "win_rate": 0.0, 
                   "max_drawdown": 0.0, "profit_factor": 0.0}

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        try:
            if not self.optimization_results:
                return {"message": "No optimization results available"}

            latest_result = self.optimization_results[-1]
            
            return {
                "total_optimizations": len(self.optimization_results),
                "best_score": latest_result.optimization_score,
                "best_weights": latest_result.weights,
                "performance_metrics": latest_result.performance_metrics,
                "confidence_level": latest_result.confidence_level,
                "last_optimization": len(self.optimization_history)
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Error getting optimization summary: {e}"))
            return {}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.sr_predictor:
                await self.sr_predictor.cleanup()
            
            self.logger.info("✅ SR Weight Optimizer cleanup completed")

        except Exception as e:
            self.logger.error(failed(f"❌ SR Weight Optimizer cleanup failed: {e}"))


# Setup function for easy integration
async def setup_sr_weight_optimizer(config: Dict[str, Any]) -> SRWeightOptimizer:
    """Setup the SR Weight Optimizer."""
    try:
        optimizer = SRWeightOptimizer(config)
        if await optimizer.initialize():
            return optimizer
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup SR weight optimizer: {e}")
        return None
