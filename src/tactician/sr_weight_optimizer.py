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
        """
        Initialize the SR weight optimizer.

        Args:
            config: Configuration dictionary
        """
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
        """
        Initialize the SR weight optimizer.

        Returns:
            bool: True if initialization successful
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
        """Initialize default weights for SR prediction."""
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
        """
        Validate optimizer configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
        """
        Update weights using online learning with incremental updates.

        Args:
            trade_result: Result of a completed trade
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
        """
        Calculate performance metrics for a single trade.

        Args:
            trade_result: Trade result data

        Returns:
            Dict: Performance metrics
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
        """
        Calculate multi-objective optimization score.

        Args:
            performance_metrics: Performance metrics

        Returns:
            float: Multi-objective score
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
        """
        Calculate gradients for each weight based on trade result.

        Args:
            trade_result: Trade result data
            objective_score: Multi-objective score

        Returns:
            Dict: Weight gradients
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
        """
        Update weights using calculated gradients.

        Args:
            gradients: Weight gradients
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
        """Normalize weights to sum to 1."""
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
                self.weight_gradients[weight_name] *= self.momentum

            # Apply decay to learning rate
            self.learning_rate *= self.decay_rate

        except Exception as e:
            self.logger.error(failed(f"❌ Error applying momentum and decay: {e}"))

    def _store_optimization_step(self, trade_result: Dict[str, Any], performance_metrics: Dict[str, float], objective_score: float) -> None:
        """
        Store optimization step in history.

        Args:
            trade_result: Trade result data
            performance_metrics: Performance metrics
            objective_score: Multi-objective score
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            step_data = {
                "update_count": self.update_count,
                "timestamp": pd.Timestamp.now().isoformat(),
                "weights": self.current_weights.copy(),
                "trade_result": trade_result,
                "performance_metrics": performance_metrics,
                "objective_score": objective_score,
                "learning_rate": self.learning_rate
            }

            self.optimization_history.append(step_data)

            # Keep history size manageable
            if len(self.optimization_history) > 1000:
                self.optimization_history = self.optimization_history[-1000:]

        except Exception as e:
            self.logger.error(failed(f"❌ Error storing optimization step: {e}"))

    async def _update_sr_predictor_weights(self) -> None:
        """Update SR predictor with new weights."""
        try:
            if self.sr_predictor:
                # Update SR predictor weights
                self.sr_predictor.model_weights = self.current_weights.copy()
                self.logger.debug("Updated SR predictor weights")

        except Exception as e:
            self.logger.error(failed(f"❌ Error updating SR predictor weights: {e}"))

    async def run_batch_optimization(self, historical_data: pd.DataFrame) -> WeightOptimizationResult:
        """
        Run batch optimization on historical data.

        Args:
            historical_data: Historical market data

        Returns:
            WeightOptimizationResult: Optimization result
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            self.logger.info("Running batch optimization...")

            best_score = -np.inf
            best_weights = self.current_weights.copy()

            # Grid search over weight combinations
            for iteration in range(self.max_iterations):
                # Generate weight combination
                test_weights = self._generate_weight_combination()

                # Test weights on historical data
                performance_metrics = await self._test_weights_on_data(test_weights, historical_data)

                # Calculate multi-objective score
                objective_score = self._calculate_multi_objective_score(performance_metrics)

                # Update best weights if better
                if objective_score > best_score:
                    best_score = objective_score
                    best_weights = test_weights.copy()

                if iteration % 10 == 0:
                    self.logger.info(f"Batch optimization progress: {iteration}/{self.max_iterations}")

            # Create optimization result
            final_performance = await self._test_weights_on_data(best_weights, historical_data)
            result = WeightOptimizationResult(
                weights=best_weights,
                performance_metrics=final_performance,
                sharpe_ratio=final_performance.get("sharpe_ratio", 0.0),
                max_drawdown=final_performance.get("max_drawdown", 0.0),
                win_rate=final_performance.get("win_rate", 0.0),
                profit_factor=final_performance.get("profit_factor", 0.0),
                total_return=final_performance.get("total_pnl", 0.0),
                optimization_score=best_score,
                backtest_periods=self.backtest_periods,
                confidence_level=self.min_confidence
            )

            # Update current weights
            self.current_weights = best_weights
            self.best_weights = best_weights

            # Store result
            self.optimization_results.append(result)

            self.logger.info(f"✅ Batch optimization completed. Best score: {best_score:.4f}")
            return result

        except Exception as e:
            self.logger.error(failed(f"❌ Error in batch optimization: {e}"))
            return None

    def _generate_weight_combination(self) -> Dict[str, float]:
        """Generate a random weight combination for testing."""
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            # Generate random weights
            weights = {}
            for weight_name in self.current_weights.keys():
                weights[weight_name] = np.random.random()

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for weight_name in weights:
                    weights[weight_name] /= total_weight

            return weights

        except Exception as e:
            self.logger.error(failed(f"❌ Error generating weight combination: {e}"))
            return self.current_weights.copy()

    async def _test_weights_on_data(self, weights: Dict[str, float], data: pd.DataFrame) -> Dict[str, float]:
        """
        Test weights on historical data.

        Args:
            weights: Weights to test
            data: Historical data

        Returns:
            Dict: Performance metrics
        """
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            # Update SR predictor with test weights
            if self.sr_predictor:
                original_weights = self.sr_predictor.model_weights.copy()
                self.sr_predictor.model_weights = weights

                # Run backtest
                backtest_result = await self.sr_predictor.run_backtest(data)

                # Restore original weights
                self.sr_predictor.model_weights = original_weights

                return backtest_result

            return {"total_pnl": 0.0, "sharpe_ratio": 0.0, "win_rate": 0.0}

        except Exception as e:
            self.logger.error(failed(f"❌ Error testing weights: {e}"))
            return {"total_pnl": 0.0, "sharpe_ratio": 0.0, "win_rate": 0.0}

    def get_current_weights(self) -> Dict[str, float]:
        """
        Get current optimized weights.

        Returns:
            Dict: Current weights
        """
        return self.current_weights.copy()

    def get_optimization_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get optimization history.

        Args:
            limit: Maximum number of records to return

        Returns:
            List: Optimization history
        """
        try:
            if limit:
                return self.optimization_history[-limit:]
            return self.optimization_history.copy()

        except Exception as e:
            self.logger.error(failed(f"❌ Error getting optimization history: {e}"))
            return []

    def get_best_weights(self) -> Optional[Dict[str, float]]:
        """
        Get best weights from optimization.

        Returns:
            Dict: Best weights or None
        """
        return self.best_weights.copy() if self.best_weights else None

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.sr_predictor:
                await self.sr_predictor.cleanup()

            self.logger.info("✅ SR Weight Optimizer cleanup completed")

        except Exception as e:
            self.logger.error(failed(f"❌ SR Weight Optimizer cleanup failed: {e}"))


# Setup function for easy integration
async def setup_sr_weight_optimizer(config: Dict[str, Any]) -> Optional[SRWeightOptimizer]:
    """Setup SR weight optimizer."""
    try:
        optimizer = SRWeightOptimizer(config)
        if await optimizer.initialize():
            return optimizer
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup SR weight optimizer: {e}")
        return None
