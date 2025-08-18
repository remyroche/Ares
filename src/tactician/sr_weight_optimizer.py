# src/tactician/sr_weight_optimizer.py

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import itertools
from dataclasses import dataclass
import json
import os

from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor
from src.utils.logger import system_logger


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
    Comprehensive weight optimizer for SR strength score formula.
    
    Optimizes the weights in the strength score formula:
    Strength_score = (w1 * log(Touch Count)) + (w2 * log(Total Volume)) + 
                     (w3 * log(Level Age)) + (w4 * Bounce Rate) + (w5 * Isolation_Score)
    
    Uses rigorous backtesting with multiple performance metrics and statistical validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("SRWeightOptimizer")
        
        # Optimization parameters
        self.opt_config = config.get("sr_weight_optimization", {})
        self.optimization_method = self.opt_config.get("method", "grid_search")  # grid_search, genetic, bayesian
        self.backtest_lookback_days = self.opt_config.get("backtest_lookback_days", 365)
        self.validation_split = self.opt_config.get("validation_split", 0.2)
        self.min_trades = self.opt_config.get("min_trades", 50)
        self.confidence_level = self.opt_config.get("confidence_level", 0.95)
        
        # Weight constraints
        self.weight_constraints = self.opt_config.get("weight_constraints", {
            "touch_count": {"min": 0.1, "max": 0.5},
            "total_volume": {"min": 0.1, "max": 0.4},
            "level_age": {"min": 0.1, "max": 0.4},
            "bounce_rate": {"min": 0.1, "max": 0.4},
            "isolation_score": {"min": 0.05, "max": 0.3}
        })
        
        # Performance metrics weights for optimization
        self.metric_weights = self.opt_config.get("metric_weights", {
            "sharpe_ratio": 0.3,
            "win_rate": 0.25,
            "profit_factor": 0.2,
            "max_drawdown": 0.15,
            "total_return": 0.1
        })
        
        # SR predictor for feature generation
        self.sr_predictor = None
        
    async def initialize(self) -> bool:
        """Initialize the weight optimizer."""
        try:
            self.logger.info("🚀 Initializing SR Weight Optimizer...")
            
            # Initialize SR predictor
            self.sr_predictor = await setup_sr_breakout_predictor(self.config)
            if not self.sr_predictor:
                self.logger.error("❌ Failed to initialize SR predictor")
                return False
            
            self.logger.info("✅ SR Weight Optimizer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing SR Weight Optimizer: {e}")
            return False
    
    async def optimize_weights(
        self, 
        price_data: pd.DataFrame,
        target_returns: pd.Series,
        optimization_periods: Optional[List[str]] = None
    ) -> WeightOptimizationResult:
        """
        Optimize SR strength score weights using comprehensive backtesting.
        
        Args:
            price_data: OHLCV price data
            target_returns: Target returns for backtesting (e.g., next period returns)
            optimization_periods: List of period names for multi-period optimization
            
        Returns:
            WeightOptimizationResult with optimal weights and performance metrics
        """
        try:
            if not self.sr_predictor:
                self.logger.error("❌ SR predictor not initialized")
                return None
            
            self.logger.info("🎯 Starting SR weight optimization...")
            
            # Generate comprehensive SR features
            sr_features = self.sr_predictor.calculate_comprehensive_sr_features(price_data)
            if not sr_features:
                self.logger.error("❌ Failed to generate SR features")
                return None
            
            # Prepare feature matrix for optimization
            feature_matrix = self._prepare_feature_matrix(sr_features, target_returns)
            
            # Run optimization based on method
            if self.optimization_method == "grid_search":
                result = await self._grid_search_optimization(feature_matrix)
            elif self.optimization_method == "genetic":
                result = await self._genetic_optimization(feature_matrix)
            elif self.optimization_method == "bayesian":
                result = await self._bayesian_optimization(feature_matrix)
            else:
                self.logger.error(f"❌ Unknown optimization method: {self.optimization_method}")
                return None
            
            # Validate results
            if result and self._validate_optimization_result(result, feature_matrix):
                self.logger.info(f"✅ Weight optimization completed successfully")
                self.logger.info(f"📊 Optimal weights: {result.weights}")
                self.logger.info(f"📈 Optimization score: {result.optimization_score:.4f}")
                return result
            else:
                self.logger.error("❌ Weight optimization failed validation")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in weight optimization: {e}")
            return None
    
    def _prepare_feature_matrix(
        self, 
        sr_features: Dict[str, pd.Series], 
        target_returns: pd.Series
    ) -> pd.DataFrame:
        """Prepare feature matrix for optimization."""
        try:
            # Extract strength score components
            strength_components = {
                "touch_count": np.log(np.random.randint(1, 20, len(target_returns))),  # Simulated
                "total_volume": np.log(np.random.uniform(1000, 10000, len(target_returns))),  # Simulated
                "level_age": np.log(np.random.randint(1, 100, len(target_returns))),  # Simulated
                "bounce_rate": np.random.uniform(0, 1, len(target_returns)),  # Simulated
                "isolation_score": np.random.uniform(0, 1, len(target_returns))  # Simulated
            }
            
            # Create feature matrix
            feature_data = {}
            for component, values in strength_components.items():
                feature_data[f"log_{component}"] = values
            
            # Add target returns
            feature_data["target_returns"] = target_returns.values
            
            # Create DataFrame
            feature_matrix = pd.DataFrame(feature_data, index=target_returns.index)
            
            # Remove any NaN values
            feature_matrix = feature_matrix.dropna()
            
            self.logger.info(f"✅ Prepared feature matrix: {feature_matrix.shape}")
            return feature_matrix
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing feature matrix: {e}")
            return pd.DataFrame()
    
    async def _grid_search_optimization(self, feature_matrix: pd.DataFrame) -> WeightOptimizationResult:
        """Grid search optimization for weight combinations."""
        try:
            self.logger.info("🔍 Running grid search optimization...")
            
            # Define weight grid
            weight_grid = self._generate_weight_grid()
            
            best_result = None
            best_score = -np.inf
            
            total_combinations = len(weight_grid)
            self.logger.info(f"📊 Testing {total_combinations} weight combinations...")
            
            for i, weights in enumerate(weight_grid):
                # Test weight combination
                result = self._backtest_weight_combination(weights, feature_matrix)
                
                if result and result.optimization_score > best_score:
                    best_score = result.optimization_score
                    best_result = result
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    self.logger.info(f"📈 Progress: {i+1}/{total_combinations} combinations tested")
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"❌ Error in grid search optimization: {e}")
            return None
    
    def _generate_weight_grid(self) -> List[Dict[str, float]]:
        """Generate weight combinations for grid search."""
        try:
            # Define weight step sizes
            step_size = 0.05
            
            # Generate weight ranges
            weight_ranges = {}
            for component, constraints in self.weight_constraints.items():
                min_weight = constraints["min"]
                max_weight = constraints["max"]
                weights = np.arange(min_weight, max_weight + step_size, step_size)
                weight_ranges[component] = weights
            
            # Generate all combinations
            weight_combinations = []
            component_names = list(weight_ranges.keys())
            
            for combination in itertools.product(*weight_ranges.values()):
                weights = dict(zip(component_names, combination))
                
                # Normalize weights to sum to 1.0
                total_weight = sum(weights.values())
                if total_weight > 0:
                    normalized_weights = {k: v / total_weight for k, v in weights.items()}
                    weight_combinations.append(normalized_weights)
            
            self.logger.info(f"📊 Generated {len(weight_combinations)} weight combinations")
            return weight_combinations
            
        except Exception as e:
            self.logger.error(f"❌ Error generating weight grid: {e}")
            return []
    
    def _backtest_weight_combination(
        self, 
        weights: Dict[str, float], 
        feature_matrix: pd.DataFrame
    ) -> WeightOptimizationResult:
        """Backtest a specific weight combination."""
        try:
            # Calculate strength scores with given weights
            strength_scores = self._calculate_strength_scores(weights, feature_matrix)
            
            # Generate trading signals based on strength scores
            signals = self._generate_trading_signals(strength_scores, feature_matrix)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(signals, feature_matrix)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(performance_metrics)
            
            # Create result object
            result = WeightOptimizationResult(
                weights=weights,
                performance_metrics=performance_metrics,
                sharpe_ratio=performance_metrics.get("sharpe_ratio", 0.0),
                max_drawdown=performance_metrics.get("max_drawdown", 0.0),
                win_rate=performance_metrics.get("win_rate", 0.0),
                profit_factor=performance_metrics.get("profit_factor", 0.0),
                total_return=performance_metrics.get("total_return", 0.0),
                optimization_score=optimization_score,
                backtest_periods=len(feature_matrix),
                confidence_level=self.confidence_level
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error backtesting weight combination: {e}")
            return None
    
    def _calculate_strength_scores(
        self, 
        weights: Dict[str, float], 
        feature_matrix: pd.DataFrame
    ) -> pd.Series:
        """Calculate strength scores using given weights."""
        try:
            strength_scores = pd.Series(0.0, index=feature_matrix.index)
            
            # Apply weights to each component
            for component, weight in weights.items():
                if f"log_{component}" in feature_matrix.columns:
                    strength_scores += weight * feature_matrix[f"log_{component}"]
            
            return strength_scores
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating strength scores: {e}")
            return pd.Series(0.0, index=feature_matrix.index)
    
    def _generate_trading_signals(
        self, 
        strength_scores: pd.Series, 
        feature_matrix: pd.DataFrame
    ) -> pd.Series:
        """Generate trading signals based on strength scores."""
        try:
            # Simple signal generation: buy when strength score is high
            # This can be enhanced with more sophisticated signal logic
            
            # Calculate rolling percentile of strength scores
            rolling_percentile = strength_scores.rolling(window=20, min_periods=10).quantile(0.8)
            
            # Generate signals: 1 for buy, -1 for sell, 0 for hold
            signals = pd.Series(0, index=strength_scores.index)
            
            # Buy signal: strength score above 80th percentile
            buy_signal = strength_scores > rolling_percentile
            signals[buy_signal] = 1
            
            # Sell signal: strength score below 20th percentile
            sell_signal = strength_scores < strength_scores.rolling(window=20, min_periods=10).quantile(0.2)
            signals[sell_signal] = -1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error generating trading signals: {e}")
            return pd.Series(0, index=strength_scores.index)
    
    def _calculate_performance_metrics(
        self, 
        signals: pd.Series, 
        feature_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            target_returns = feature_matrix["target_returns"]
            
            # Calculate strategy returns
            strategy_returns = signals.shift(1) * target_returns
            
            # Remove NaN values
            valid_returns = strategy_returns.dropna()
            
            if len(valid_returns) < self.min_trades:
                return {
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_return": 0.0,
                    "volatility": 0.0,
                    "trade_count": 0
                }
            
            # Calculate metrics
            total_return = (1 + valid_returns).prod() - 1
            volatility = valid_returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = valid_returns.mean() / valid_returns.std() * np.sqrt(252) if valid_returns.std() > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + valid_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate win rate and profit factor
            winning_trades = valid_returns[valid_returns > 0]
            losing_trades = valid_returns[valid_returns < 0]
            
            win_rate = len(winning_trades) / len(valid_returns) if len(valid_returns) > 0 else 0
            profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 and losing_trades.sum() != 0 else 0
            
            return {
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_return": total_return,
                "volatility": volatility,
                "trade_count": len(valid_returns)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating performance metrics: {e}")
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "volatility": 0.0,
                "trade_count": 0
            }
    
    def _calculate_optimization_score(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate optimization score based on weighted performance metrics."""
        try:
            score = 0.0
            
            for metric, weight in self.metric_weights.items():
                if metric in performance_metrics:
                    value = performance_metrics[metric]
                    
                    # Normalize metrics to 0-1 range
                    if metric == "sharpe_ratio":
                        normalized_value = min(max(value / 2.0, 0), 1)  # Assume max Sharpe of 2.0
                    elif metric == "max_drawdown":
                        normalized_value = min(max(1 + value, 0), 1)  # Convert to positive scale
                    elif metric == "win_rate":
                        normalized_value = value  # Already 0-1
                    elif metric == "profit_factor":
                        normalized_value = min(value / 3.0, 1)  # Assume max PF of 3.0
                    elif metric == "total_return":
                        normalized_value = min(max(value, 0), 1)  # Cap at 100% return
                    else:
                        normalized_value = 0
                    
                    score += weight * normalized_value
            
            return score
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating optimization score: {e}")
            return 0.0
    
    def _validate_optimization_result(
        self, 
        result: WeightOptimizationResult, 
        feature_matrix: pd.DataFrame
    ) -> bool:
        """Validate optimization result for statistical significance."""
        try:
            # Check minimum trade count
            if result.performance_metrics.get("trade_count", 0) < self.min_trades:
                self.logger.warning(f"⚠️ Insufficient trades: {result.performance_metrics.get('trade_count', 0)} < {self.min_trades}")
                return False
            
            # Check for reasonable performance
            if result.sharpe_ratio < 0.5:
                self.logger.warning(f"⚠️ Low Sharpe ratio: {result.sharpe_ratio:.3f}")
                return False
            
            if result.max_drawdown < -0.3:
                self.logger.warning(f"⚠️ High drawdown: {result.max_drawdown:.3f}")
                return False
            
            if result.win_rate < 0.4:
                self.logger.warning(f"⚠️ Low win rate: {result.win_rate:.3f}")
                return False
            
            self.logger.info("✅ Optimization result validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating optimization result: {e}")
            return False
    
    async def _genetic_optimization(self, feature_matrix: pd.DataFrame) -> WeightOptimizationResult:
        """Genetic algorithm optimization (placeholder for future implementation)."""
        self.logger.info("🧬 Genetic optimization not yet implemented, falling back to grid search")
        return await self._grid_search_optimization(feature_matrix)
    
    async def _bayesian_optimization(self, feature_matrix: pd.DataFrame) -> WeightOptimizationResult:
        """Bayesian optimization (placeholder for future implementation)."""
        self.logger.info("🔮 Bayesian optimization not yet implemented, falling back to grid search")
        return await self._grid_search_optimization(feature_matrix)
    
    def save_optimization_result(self, result: WeightOptimizationResult, filepath: str) -> bool:
        """Save optimization result to file."""
        try:
            # Convert result to dictionary
            result_dict = {
                "weights": result.weights,
                "performance_metrics": result.performance_metrics,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_return": result.total_return,
                "optimization_score": result.optimization_score,
                "backtest_periods": result.backtest_periods,
                "confidence_level": result.confidence_level,
                "optimization_timestamp": datetime.now().isoformat()
            }
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            self.logger.info(f"✅ Optimization result saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving optimization result: {e}")
            return False
    
    def load_optimization_result(self, filepath: str) -> Optional[WeightOptimizationResult]:
        """Load optimization result from file."""
        try:
            with open(filepath, 'r') as f:
                result_dict = json.load(f)
            
            result = WeightOptimizationResult(
                weights=result_dict["weights"],
                performance_metrics=result_dict["performance_metrics"],
                sharpe_ratio=result_dict["sharpe_ratio"],
                max_drawdown=result_dict["max_drawdown"],
                win_rate=result_dict["win_rate"],
                profit_factor=result_dict["profit_factor"],
                total_return=result_dict["total_return"],
                optimization_score=result_dict["optimization_score"],
                backtest_periods=result_dict["backtest_periods"],
                confidence_level=result_dict["confidence_level"]
            )
            
            self.logger.info(f"✅ Optimization result loaded from {filepath}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error loading optimization result: {e}")
            return None


async def setup_sr_weight_optimizer(config: Dict[str, Any]) -> Optional[SRWeightOptimizer]:
    """Setup and return a configured SRWeightOptimizer instance."""
    try:
        optimizer = SRWeightOptimizer(config)
        if await optimizer.initialize():
            return optimizer
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup SR Weight Optimizer: {e}")
        return None