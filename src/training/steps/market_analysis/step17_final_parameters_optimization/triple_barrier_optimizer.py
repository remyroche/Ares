"""Triple Barrier Parameter Optimizer for Step 17.

This module optimizes triple barrier parameters during the training process,
ensuring that barrier values are tuned for optimal performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import optuna
from dataclasses import dataclass
import logging

from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger
from src.core.decorators.errors import handles_errors


@dataclass
class BarrierOptimizationResult:
    """Results from barrier optimization."""
    optimal_profit_multiplier: float
    optimal_stop_multiplier: float
    optimal_time_barrier: int
    optimization_score: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    n_trials: int
    best_trial: int


class TripleBarrierOptimizer:
    """Optimizes triple barrier parameters for different market regimes."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the triple barrier optimizer."""
        self.config = config
        self.logger = system_logger.getChild("TripleBarrierOptimizer")
        
        # Optimization configuration
        self.optim_config = config.get("triple_barrier_optimization", {})
        self.n_trials = self.optim_config.get("n_trials", 100)
        self.n_jobs = self.optim_config.get("n_jobs", -1)
        
        # Parameter ranges for optimization
        self.param_ranges = {
            "profit_multiplier": self.optim_config.get("profit_range", (0.001, 0.005)),
            "stop_multiplier": self.optim_config.get("stop_range", (0.0005, 0.0025)),
            "time_barrier": self.optim_config.get("time_range", (10, 60))
        }
        
        # Regime-specific optimization settings
        self.regime_specific = self.optim_config.get("regime_specific", True)
        self.min_samples_per_regime = self.optim_config.get("min_samples_per_regime", 1000)
        
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="optimize triple barriers"
    )
    @traced(span_name="TripleBarrier.optimize")
    async def optimize_barriers(
        self,
        market_data: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None
    ) -> Dict[str, BarrierOptimizationResult]:
        """
        Optimize triple barrier parameters.
        
        Args:
            market_data: Historical market data with OHLCV
            regime_labels: Optional regime labels for regime-specific optimization
            
        Returns:
            Dictionary of optimization results (by regime if applicable)
        """
        try:
            self.logger.info("🎯 Starting triple barrier optimization...")
            
            if self.regime_specific and regime_labels is not None:
                # Optimize for each regime
                results = await self._optimize_regime_specific_barriers(
                    market_data, regime_labels
                )
            else:
                # Global optimization
                result = await self._optimize_global_barriers(market_data)
                results = {"global": result}
            
            # Log optimization results
            self._log_optimization_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error optimizing barriers: {e}")
            return {}
    
    async def _optimize_regime_specific_barriers(
        self,
        market_data: pd.DataFrame,
        regime_labels: pd.Series
    ) -> Dict[str, BarrierOptimizationResult]:
        """Optimize barriers for each regime separately."""
        results = {}
        unique_regimes = regime_labels.unique()
        
        self.logger.info(f"Optimizing barriers for {len(unique_regimes)} regimes...")
        
        for regime in unique_regimes:
            # Get regime-specific data
            regime_mask = regime_labels == regime
            regime_data = market_data[regime_mask]
            
            if len(regime_data) < self.min_samples_per_regime:
                self.logger.warning(
                    f"Regime {regime} has insufficient samples ({len(regime_data)}), skipping"
                )
                continue
            
            self.logger.info(f"Optimizing regime {regime} with {len(regime_data)} samples...")
            
            # Create Optuna study for this regime
            study = optuna.create_study(
                direction="maximize",
                study_name=f"triple_barrier_regime_{regime}"
            )
            
            # Define objective function for this regime
            def objective(trial):
                return self._barrier_objective(trial, regime_data, regime)
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=self.n_trials,
                n_jobs=1  # Sequential for now to avoid issues
            )
            
            # Extract best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Calculate detailed metrics for best parameters
            metrics = self._calculate_barrier_metrics(
                regime_data,
                best_params["profit_multiplier"],
                best_params["stop_multiplier"],
                best_params["time_barrier"]
            )
            
            results[f"regime_{regime}"] = BarrierOptimizationResult(
                optimal_profit_multiplier=best_params["profit_multiplier"],
                optimal_stop_multiplier=best_params["stop_multiplier"],
                optimal_time_barrier=best_params["time_barrier"],
                optimization_score=best_value,
                sharpe_ratio=metrics["sharpe_ratio"],
                win_rate=metrics["win_rate"],
                profit_factor=metrics["profit_factor"],
                max_drawdown=metrics["max_drawdown"],
                n_trials=len(study.trials),
                best_trial=study.best_trial.number
            )
        
        return results
    
    async def _optimize_global_barriers(
        self,
        market_data: pd.DataFrame
    ) -> BarrierOptimizationResult:
        """Optimize barriers globally across all data."""
        
        self.logger.info(f"Optimizing global barriers with {len(market_data)} samples...")
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name="triple_barrier_global"
        )
        
        # Define objective function
        def objective(trial):
            return self._barrier_objective(trial, market_data, "global")
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=1
        )
        
        # Extract best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Calculate detailed metrics
        metrics = self._calculate_barrier_metrics(
            market_data,
            best_params["profit_multiplier"],
            best_params["stop_multiplier"],
            best_params["time_barrier"]
        )
        
        return BarrierOptimizationResult(
            optimal_profit_multiplier=best_params["profit_multiplier"],
            optimal_stop_multiplier=best_params["stop_multiplier"],
            optimal_time_barrier=best_params["time_barrier"],
            optimization_score=best_value,
            sharpe_ratio=metrics["sharpe_ratio"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            max_drawdown=metrics["max_drawdown"],
            n_trials=len(study.trials),
            best_trial=study.best_trial.number
        )
    
    def _barrier_objective(
        self,
        trial: optuna.Trial,
        data: pd.DataFrame,
        regime: str
    ) -> float:
        """Objective function for barrier optimization."""
        
        # Sample parameters
        profit_mult = trial.suggest_float(
            "profit_multiplier",
            self.param_ranges["profit_multiplier"][0],
            self.param_ranges["profit_multiplier"][1]
        )
        stop_mult = trial.suggest_float(
            "stop_multiplier",
            self.param_ranges["stop_multiplier"][0],
            self.param_ranges["stop_multiplier"][1]
        )
        time_barrier = trial.suggest_int(
            "time_barrier",
            self.param_ranges["time_barrier"][0],
            self.param_ranges["time_barrier"][1]
        )
        
        # Apply triple barrier labeling
        labels = self._apply_triple_barrier(
            data, profit_mult, stop_mult, time_barrier
        )
        
        # Calculate objective score
        score = self._calculate_objective_score(data, labels)
        
        return score
    
    def _apply_triple_barrier(
        self,
        data: pd.DataFrame,
        profit_mult: float,
        stop_mult: float,
        time_barrier: int
    ) -> pd.Series:
        """Apply triple barrier labeling with given parameters."""
        
        labels = pd.Series(index=data.index, dtype=int)
        labels[:] = 0  # Initialize with zeros
        
        close_prices = data["close"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        
        n = len(data)
        
        for i in range(n - 1):
            entry_price = close_prices[i]
            profit_barrier = entry_price * (1 + profit_mult)
            stop_barrier = entry_price * (1 - stop_mult)
            
            # Time barrier
            end_idx = min(i + time_barrier, n)
            
            # Check barriers
            for j in range(i + 1, end_idx):
                # Profit barrier hit
                if high_prices[j] >= profit_barrier:
                    labels.iloc[i] = 1
                    break
                # Stop barrier hit
                elif low_prices[j] <= stop_barrier:
                    labels.iloc[i] = -1
                    break
        
        return labels
    
    def _calculate_objective_score(
        self,
        data: pd.DataFrame,
        labels: pd.Series
    ) -> float:
        """Calculate objective score for optimization."""
        
        # Filter out zero labels for binary classification
        mask = labels != 0
        if mask.sum() < 100:  # Need minimum signals
            return -1.0
        
        filtered_labels = labels[mask]
        
        # Calculate metrics
        win_rate = (filtered_labels == 1).sum() / len(filtered_labels)
        
        # Balance between win rate and number of signals
        signal_rate = len(filtered_labels) / len(labels)
        
        # Penalize extreme imbalance
        balance_penalty = abs(win_rate - 0.5)
        
        # Objective: maximize signals while maintaining good win rate
        score = signal_rate * (1 - balance_penalty) * win_rate
        
        return score
    
    def _calculate_barrier_metrics(
        self,
        data: pd.DataFrame,
        profit_mult: float,
        stop_mult: float,
        time_barrier: int
    ) -> Dict[str, float]:
        """Calculate detailed metrics for barrier configuration."""
        
        # Apply barriers
        labels = self._apply_triple_barrier(
            data, profit_mult, stop_mult, time_barrier
        )
        
        # Calculate returns
        returns = data["close"].pct_change()
        
        # Simulate trading
        positions = labels.shift(1).fillna(0)
        trading_returns = positions * returns
        
        # Filter out zero positions
        active_returns = trading_returns[positions != 0]
        
        if len(active_returns) == 0:
            return {
                "sharpe_ratio": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0
            }
        
        # Calculate metrics
        sharpe_ratio = np.sqrt(252) * active_returns.mean() / active_returns.std() if active_returns.std() > 0 else 0
        
        wins = active_returns[active_returns > 0]
        losses = active_returns[active_returns < 0]
        
        win_rate = len(wins) / len(active_returns) if len(active_returns) > 0 else 0
        
        profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + active_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown
        }
    
    def _log_optimization_results(
        self,
        results: Dict[str, BarrierOptimizationResult]
    ) -> None:
        """Log optimization results."""
        
        self.logger.info("📊 Triple Barrier Optimization Results:")
        
        for regime, result in results.items():
            self.logger.info(f"\n{regime.upper()} Results:")
            self.logger.info(f"  Optimal Profit Multiplier: {result.optimal_profit_multiplier:.4f}")
            self.logger.info(f"  Optimal Stop Multiplier: {result.optimal_stop_multiplier:.4f}")
            self.logger.info(f"  Optimal Time Barrier: {result.optimal_time_barrier} bars")
            self.logger.info(f"  Optimization Score: {result.optimization_score:.4f}")
            self.logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            self.logger.info(f"  Win Rate: {result.win_rate:.2%}")
            self.logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
            self.logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
            self.logger.info(f"  Best Trial: {result.best_trial}/{result.n_trials}")
    
    def get_optimized_config(
        self,
        results: Dict[str, BarrierOptimizationResult],
        current_regime: Optional[str] = None
    ) -> Dict[str, float]:
        """Get optimized configuration for current regime."""
        
        if current_regime and f"regime_{current_regime}" in results:
            result = results[f"regime_{current_regime}"]
        elif "global" in results:
            result = results["global"]
        else:
            # Fallback to first available result
            result = list(results.values())[0]
        
        return {
            "profit_take_multiplier": result.optimal_profit_multiplier,
            "stop_loss_multiplier": result.optimal_stop_multiplier,
            "time_barrier_minutes": result.optimal_time_barrier,
            "binary_classification": True
        }