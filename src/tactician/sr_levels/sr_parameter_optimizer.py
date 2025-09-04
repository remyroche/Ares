"""S/R Parameter Optimizer Module.

This module optimizes S/R probability calculation parameters through comprehensive backtesting,
finding the optimal weights and thresholds for price action, volatility, volume, and other factors.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import optuna
import json

from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger
from src.core.decorators.errors import handles_errors


@dataclass
class SRParameterSet:
    """Container for S/R calculation parameters."""
    # Price action parameters
    price_action_weight: float = 0.3
    momentum_weight: float = 0.2
    trend_strength_weight: float = 0.2
    
    # Volume parameters
    volume_weight: float = 0.2
    volume_surge_multiplier: float = 2.0
    volume_confirmation_threshold: float = 1.5
    
    # Volatility parameters
    volatility_weight: float = 0.1
    high_volatility_breakout_boost: float = 0.15
    low_volatility_consolidation_boost: float = 0.1
    
    # S/R strength parameters
    level_strength_weight: float = 0.2
    touch_count_weight: float = 0.3
    age_decay_factor: float = 0.95
    
    # Proximity parameters
    proximity_threshold: float = 0.002
    proximity_decay_rate: float = 2.0
    
    # Probability thresholds
    min_breakout_probability: float = 0.2
    max_breakout_probability: float = 0.8
    default_probability: float = 0.33


@dataclass
class OptimizationResult:
    """Results from S/R parameter optimization."""
    best_parameters: SRParameterSet
    optimization_score: float
    backtest_metrics: Dict[str, float]
    n_trials: int
    best_trial: int
    optimization_time: float


class SRParameterOptimizer:
    """Optimizes S/R probability calculation parameters through backtesting."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the S/R parameter optimizer."""
        self.config = config
        self.logger = system_logger.getChild("SRParameterOptimizer")
        
        # Optimization configuration
        self.optim_config = config.get("sr_parameter_optimization", {})
        self.n_trials = self.optim_config.get("n_trials", 200)
        self.lookback_periods = self.optim_config.get("lookback_periods", 500)
        self.min_test_points = self.optim_config.get("min_test_points", 20)
        
        # Parameter ranges for optimization
        self.param_ranges = self._get_parameter_ranges()
        
        # Current best parameters
        self.best_parameters = SRParameterSet()
        self.optimization_history = []
        
    def _get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges for optimization."""
        return {
            # Price action parameters
            "price_action_weight": (0.1, 0.4),
            "momentum_weight": (0.1, 0.3),
            "trend_strength_weight": (0.1, 0.3),
            
            # Volume parameters
            "volume_weight": (0.1, 0.3),
            "volume_surge_multiplier": (1.5, 3.0),
            "volume_confirmation_threshold": (1.2, 2.0),
            
            # Volatility parameters
            "volatility_weight": (0.05, 0.2),
            "high_volatility_breakout_boost": (0.05, 0.25),
            "low_volatility_consolidation_boost": (0.05, 0.2),
            
            # S/R strength parameters
            "level_strength_weight": (0.1, 0.3),
            "touch_count_weight": (0.2, 0.4),
            "age_decay_factor": (0.9, 0.99),
            
            # Proximity parameters
            "proximity_threshold": (0.001, 0.005),
            "proximity_decay_rate": (1.5, 3.0),
            
            # Probability thresholds
            "min_breakout_probability": (0.1, 0.3),
            "max_breakout_probability": (0.7, 0.9),
        }
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="optimize SR parameters"
    )
    @traced(span_name="SROptimizer.optimize")
    async def optimize_parameters(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        validation_data: Optional[pd.DataFrame] = None
    ) -> OptimizationResult:
        """
        Optimize S/R probability calculation parameters.
        
        Args:
            market_data: Historical market data for optimization
            sr_levels: Detected S/R levels
            validation_data: Optional out-of-sample validation data
            
        Returns:
            OptimizationResult with optimized parameters
        """
        try:
            self.logger.info("🎯 Starting S/R parameter optimization...")
            start_time = datetime.now()
            
            # Create Optuna study
            study = optuna.create_study(
                direction="maximize",
                study_name="sr_parameter_optimization"
            )
            
            # Define objective function
            def objective(trial):
                return self._optimization_objective(
                    trial, market_data, sr_levels
                )
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            
            # Extract best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Create parameter set from best params
            optimized_params = self._create_parameter_set(best_params)
            
            # Validate on out-of-sample data if provided
            if validation_data is not None:
                validation_score = await self._validate_parameters(
                    optimized_params, validation_data, sr_levels
                )
                self.logger.info(f"Validation score: {validation_score:.4f}")
            
            # Calculate detailed metrics
            backtest_metrics = await self._calculate_backtest_metrics(
                optimized_params, market_data, sr_levels
            )
            
            # Create result
            result = OptimizationResult(
                best_parameters=optimized_params,
                optimization_score=best_value,
                backtest_metrics=backtest_metrics,
                n_trials=len(study.trials),
                best_trial=study.best_trial.number,
                optimization_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Store best parameters
            self.best_parameters = optimized_params
            self.optimization_history.append(result)
            
            # Log results
            self._log_optimization_results(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing S/R parameters: {e}")
            return None
    
    def _optimization_objective(
        self,
        trial: optuna.Trial,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]]
    ) -> float:
        """Objective function for parameter optimization."""
        
        # Sample parameters
        params = {}
        for param_name, (min_val, max_val) in self.param_ranges.items():
            params[param_name] = trial.suggest_float(
                param_name, min_val, max_val
            )
        
        # Ensure weights sum to approximately 1
        weight_params = [
            "price_action_weight", "momentum_weight", "trend_strength_weight",
            "volume_weight", "volatility_weight"
        ]
        weight_sum = sum(params[p] for p in weight_params)
        for p in weight_params:
            params[p] /= weight_sum
        
        # Create parameter set
        param_set = self._create_parameter_set(params)
        
        # Calculate S/R probabilities with these parameters
        probabilities = self._calculate_sr_probabilities(
            market_data, sr_levels, param_set
        )
        
        # Evaluate performance
        score = self._evaluate_probability_accuracy(
            market_data, sr_levels, probabilities
        )
        
        return score
    
    def _calculate_sr_probabilities(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        params: SRParameterSet
    ) -> Dict[str, Dict[str, float]]:
        """Calculate S/R probabilities with given parameters."""
        
        probabilities = {}
        
        for i in range(len(market_data)):
            timestamp = market_data.index[i]
            current_price = market_data["close"].iloc[i]
            
            # Find nearest S/R levels
            nearest_support = None
            nearest_resistance = None
            
            for level in sr_levels:
                if level["type"] == "support" and level["price"] < current_price:
                    if nearest_support is None or level["price"] > nearest_support["price"]:
                        nearest_support = level
                elif level["type"] == "resistance" and level["price"] > current_price:
                    if nearest_resistance is None or level["price"] < nearest_resistance["price"]:
                        nearest_resistance = level
            
            # Calculate probabilities for this point
            probs = self._calculate_point_probabilities(
                market_data, i, nearest_support, nearest_resistance, params
            )
            
            probabilities[timestamp] = probs
        
        return probabilities
    
    def _calculate_point_probabilities(
        self,
        market_data: pd.DataFrame,
        idx: int,
        nearest_support: Optional[Dict[str, Any]],
        nearest_resistance: Optional[Dict[str, Any]],
        params: SRParameterSet
    ) -> Dict[str, float]:
        """Calculate probabilities for a single point in time."""
        
        # Initialize base probabilities
        breakout_prob = params.default_probability
        rebounce_prob = params.default_probability
        consolidation_prob = params.default_probability
        
        # Get market data
        current_price = market_data["close"].iloc[idx]
        volume = market_data["volume"].iloc[idx]
        
        # Calculate components
        components = {}
        
        # 1. Price Action Component
        price_action_score = self._calculate_price_action_score(
            market_data, idx, params
        )
        components["price_action"] = price_action_score * params.price_action_weight
        
        # 2. Momentum Component
        momentum_score = self._calculate_momentum_score(
            market_data, idx, params
        )
        components["momentum"] = momentum_score * params.momentum_weight
        
        # 3. Trend Strength Component
        trend_score = self._calculate_trend_strength_score(
            market_data, idx, params
        )
        components["trend"] = trend_score * params.trend_strength_weight
        
        # 4. Volume Component
        volume_score = self._calculate_volume_score(
            market_data, idx, params
        )
        components["volume"] = volume_score * params.volume_weight
        
        # 5. Volatility Component
        volatility_score = self._calculate_volatility_score(
            market_data, idx, params
        )
        components["volatility"] = volatility_score * params.volatility_weight
        
        # 6. S/R Proximity Component
        if nearest_support or nearest_resistance:
            proximity_score = self._calculate_proximity_score(
                current_price, nearest_support, nearest_resistance, params
            )
            components["proximity"] = proximity_score
        
        # Combine components to get final probabilities
        combined_score = sum(components.values())
        
        # Adjust probabilities based on combined score
        if combined_score > 0.6:
            # Strong breakout signal
            breakout_prob = min(params.max_breakout_probability, 
                              params.default_probability + combined_score * 0.5)
            rebounce_prob = params.default_probability * 0.7
            consolidation_prob = 1 - breakout_prob - rebounce_prob
        elif combined_score < 0.4:
            # Strong rebounce signal
            rebounce_prob = min(params.max_breakout_probability,
                              params.default_probability + (1 - combined_score) * 0.5)
            breakout_prob = params.default_probability * 0.7
            consolidation_prob = 1 - breakout_prob - rebounce_prob
        else:
            # Consolidation likely
            consolidation_prob = 0.5
            breakout_prob = 0.25
            rebounce_prob = 0.25
        
        # Normalize probabilities
        total = breakout_prob + rebounce_prob + consolidation_prob
        
        return {
            "breakout": breakout_prob / total,
            "rebounce": rebounce_prob / total,
            "consolidation": consolidation_prob / total,
            "combined_score": combined_score
        }
    
    def _calculate_price_action_score(
        self,
        market_data: pd.DataFrame,
        idx: int,
        params: SRParameterSet
    ) -> float:
        """Calculate price action component score."""
        
        if idx < 20:
            return 0.5
        
        # Recent price changes
        recent_returns = market_data["close"].iloc[idx-10:idx].pct_change()
        
        # Bullish/bearish candles
        bullish_candles = (market_data["close"].iloc[idx-10:idx] > 
                          market_data["open"].iloc[idx-10:idx]).sum()
        
        # Price acceleration
        acceleration = recent_returns.diff().mean()
        
        # Combine factors
        score = 0.5
        score += 0.2 * (bullish_candles / 10 - 0.5)  # Normalize around 0.5
        score += 0.3 * np.tanh(acceleration * 100)   # Bounded [-0.3, 0.3]
        
        return np.clip(score, 0, 1)
    
    def _calculate_momentum_score(
        self,
        market_data: pd.DataFrame,
        idx: int,
        params: SRParameterSet
    ) -> float:
        """Calculate momentum component score."""
        
        if idx < 20:
            return 0.5
        
        # RSI calculation (simplified)
        returns = market_data["close"].iloc[idx-14:idx].pct_change()
        gains = returns[returns > 0].mean()
        losses = -returns[returns < 0].mean()
        
        if losses == 0:
            rsi = 100
        else:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
        
        # Convert RSI to score (high RSI = breakout, low RSI = rebounce)
        if rsi > 70:
            score = 0.7 + (rsi - 70) / 100
        elif rsi < 30:
            score = 0.3 - (30 - rsi) / 100
        else:
            score = 0.5
        
        return np.clip(score, 0, 1)
    
    def _calculate_trend_strength_score(
        self,
        market_data: pd.DataFrame,
        idx: int,
        params: SRParameterSet
    ) -> float:
        """Calculate trend strength component score."""
        
        if idx < 50:
            return 0.5
        
        # Moving averages
        sma_20 = market_data["close"].iloc[idx-20:idx].mean()
        sma_50 = market_data["close"].iloc[idx-50:idx].mean()
        current_price = market_data["close"].iloc[idx]
        
        # Trend direction and strength
        trend_20 = (current_price - sma_20) / sma_20
        trend_50 = (current_price - sma_50) / sma_50
        
        # Average trend strength
        avg_trend = (trend_20 + trend_50) / 2
        
        # Convert to score (strong uptrend = high score)
        score = 0.5 + np.tanh(avg_trend * 50)
        
        return np.clip(score, 0, 1)
    
    def _calculate_volume_score(
        self,
        market_data: pd.DataFrame,
        idx: int,
        params: SRParameterSet
    ) -> float:
        """Calculate volume component score."""
        
        if idx < 20:
            return 0.5
        
        # Current and average volume
        current_volume = market_data["volume"].iloc[idx]
        avg_volume = market_data["volume"].iloc[idx-20:idx].mean()
        
        if avg_volume == 0:
            return 0.5
        
        # Volume ratio
        volume_ratio = current_volume / avg_volume
        
        # High volume suggests breakout potential
        if volume_ratio > params.volume_surge_multiplier:
            score = 0.8
        elif volume_ratio > params.volume_confirmation_threshold:
            score = 0.6 + 0.2 * (volume_ratio - params.volume_confirmation_threshold)
        else:
            score = 0.5 * volume_ratio
        
        return np.clip(score, 0, 1)
    
    def _calculate_volatility_score(
        self,
        market_data: pd.DataFrame,
        idx: int,
        params: SRParameterSet
    ) -> float:
        """Calculate volatility component score."""
        
        if idx < 20:
            return 0.5
        
        # Calculate ATR-based volatility
        high_low = market_data["high"].iloc[idx-14:idx] - market_data["low"].iloc[idx-14:idx]
        high_close = np.abs(market_data["high"].iloc[idx-14:idx] - 
                           market_data["close"].iloc[idx-15:idx-1].values)
        low_close = np.abs(market_data["low"].iloc[idx-14:idx] - 
                          market_data["close"].iloc[idx-15:idx-1].values)
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.mean()
        
        # Normalize by price
        current_price = market_data["close"].iloc[idx]
        volatility = atr / current_price
        
        # High volatility increases breakout probability
        if volatility > 0.02:
            score = 0.5 + params.high_volatility_breakout_boost
        elif volatility < 0.005:
            score = 0.5 - params.low_volatility_consolidation_boost
        else:
            score = 0.5
        
        return np.clip(score, 0, 1)
    
    def _calculate_proximity_score(
        self,
        current_price: float,
        nearest_support: Optional[Dict[str, Any]],
        nearest_resistance: Optional[Dict[str, Any]],
        params: SRParameterSet
    ) -> float:
        """Calculate S/R proximity component score."""
        
        score = 0.5
        
        # Check proximity to support
        if nearest_support:
            support_distance = (current_price - nearest_support["price"]) / current_price
            if support_distance < params.proximity_threshold:
                # Very close to support - high rebounce probability
                proximity_factor = 1 - (support_distance / params.proximity_threshold)
                score -= 0.3 * (proximity_factor ** params.proximity_decay_rate)
        
        # Check proximity to resistance
        if nearest_resistance:
            resistance_distance = (nearest_resistance["price"] - current_price) / current_price
            if resistance_distance < params.proximity_threshold:
                # Very close to resistance - high rebounce probability
                proximity_factor = 1 - (resistance_distance / params.proximity_threshold)
                score -= 0.3 * (proximity_factor ** params.proximity_decay_rate)
        
        return np.clip(score, 0, 1)
    
    def _evaluate_probability_accuracy(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        probabilities: Dict[str, Dict[str, float]]
    ) -> float:
        """Evaluate accuracy of probability predictions."""
        
        correct_predictions = 0
        total_predictions = 0
        profit_factor = 0
        
        timestamps = list(probabilities.keys())
        
        for i in range(len(timestamps) - 10):  # Need lookahead
            current_time = timestamps[i]
            current_idx = market_data.index.get_loc(current_time)
            
            if current_idx >= len(market_data) - 10:
                continue
            
            # Get predictions
            probs = probabilities[current_time]
            predicted_outcome = max(probs, key=probs.get)
            
            # Check actual outcome
            current_price = market_data["close"].iloc[current_idx]
            future_prices = market_data["close"].iloc[current_idx+1:current_idx+11]
            
            max_price = future_prices.max()
            min_price = future_prices.min()
            
            actual_outcome = "consolidation"
            
            # Breakout if price moves up significantly
            if (max_price - current_price) / current_price > 0.002:
                actual_outcome = "breakout"
            # Rebounce if price moves down significantly
            elif (current_price - min_price) / current_price > 0.002:
                actual_outcome = "rebounce"
            
            # Check if prediction was correct
            if predicted_outcome == actual_outcome:
                correct_predictions += 1
                
                # Calculate profit based on confidence
                if actual_outcome != "consolidation":
                    profit_factor += probs[predicted_outcome]
            
            total_predictions += 1
        
        # Calculate accuracy
        accuracy = correct_predictions / max(total_predictions, 1)
        
        # Calculate score combining accuracy and profit factor
        normalized_profit = profit_factor / max(total_predictions, 1)
        
        # Weighted score
        score = 0.6 * accuracy + 0.4 * normalized_profit
        
        return score
    
    async def _validate_parameters(
        self,
        params: SRParameterSet,
        validation_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]]
    ) -> float:
        """Validate parameters on out-of-sample data."""
        
        # Calculate probabilities with optimized parameters
        probabilities = self._calculate_sr_probabilities(
            validation_data, sr_levels, params
        )
        
        # Evaluate performance
        score = self._evaluate_probability_accuracy(
            validation_data, sr_levels, probabilities
        )
        
        return score
    
    async def _calculate_backtest_metrics(
        self,
        params: SRParameterSet,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate detailed backtest metrics."""
        
        # Calculate probabilities
        probabilities = self._calculate_sr_probabilities(
            market_data, sr_levels, params
        )
        
        # Initialize metrics
        metrics = {
            "accuracy": 0,
            "precision_breakout": 0,
            "precision_rebounce": 0,
            "recall_breakout": 0,
            "recall_rebounce": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "signal_count": 0
        }
        
        # Calculate confusion matrix elements
        true_positives = {"breakout": 0, "rebounce": 0}
        false_positives = {"breakout": 0, "rebounce": 0}
        true_negatives = {"breakout": 0, "rebounce": 0}
        false_negatives = {"breakout": 0, "rebounce": 0}
        
        # Track returns for financial metrics
        returns = []
        
        timestamps = list(probabilities.keys())
        
        for i in range(len(timestamps) - 10):
            current_time = timestamps[i]
            current_idx = market_data.index.get_loc(current_time)
            
            if current_idx >= len(market_data) - 10:
                continue
            
            probs = probabilities[current_time]
            predicted = max(probs, key=probs.get)
            
            # Determine actual outcome
            current_price = market_data["close"].iloc[current_idx]
            future_prices = market_data["close"].iloc[current_idx+1:current_idx+11]
            
            max_move_up = (future_prices.max() - current_price) / current_price
            max_move_down = (current_price - future_prices.min()) / current_price
            
            actual = "consolidation"
            if max_move_up > 0.002:
                actual = "breakout"
            elif max_move_down > 0.002:
                actual = "rebounce"
            
            # Update confusion matrix
            if predicted == actual:
                if predicted != "consolidation":
                    true_positives[predicted] += 1
                    # Calculate return
                    if predicted == "breakout":
                        returns.append(max_move_up)
                    else:
                        returns.append(max_move_down)
            else:
                if predicted != "consolidation":
                    false_positives[predicted] += 1
                    # Calculate loss
                    returns.append(-0.001)  # Assume fixed loss
                if actual != "consolidation":
                    false_negatives[actual] += 1
        
        # Calculate metrics
        total_predictions = sum(true_positives.values()) + sum(false_positives.values())
        
        if total_predictions > 0:
            metrics["accuracy"] = sum(true_positives.values()) / total_predictions
            metrics["signal_count"] = total_predictions
            
            # Precision and recall for each class
            for class_name in ["breakout", "rebounce"]:
                tp = true_positives[class_name]
                fp = false_positives[class_name]
                fn = false_negatives[class_name]
                
                if tp + fp > 0:
                    metrics[f"precision_{class_name}"] = tp / (tp + fp)
                if tp + fn > 0:
                    metrics[f"recall_{class_name}"] = tp / (tp + fn)
        
        # Financial metrics
        if returns:
            returns_series = pd.Series(returns)
            
            # Sharpe ratio
            if returns_series.std() > 0:
                metrics["sharpe_ratio"] = np.sqrt(252) * returns_series.mean() / returns_series.std()
            
            # Profit factor
            profits = returns_series[returns_series > 0].sum()
            losses = abs(returns_series[returns_series < 0].sum())
            if losses > 0:
                metrics["profit_factor"] = profits / losses
            
            # Max drawdown
            cumulative = (1 + returns_series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics["max_drawdown"] = abs(drawdown.min())
        
        return metrics
    
    def _create_parameter_set(self, params: Dict[str, float]) -> SRParameterSet:
        """Create parameter set from dictionary."""
        return SRParameterSet(**{
            k: v for k, v in params.items() 
            if k in SRParameterSet.__dataclass_fields__
        })
    
    def _log_optimization_results(self, result: OptimizationResult) -> None:
        """Log optimization results."""
        
        self.logger.info("📊 S/R Parameter Optimization Results:")
        self.logger.info(f"  Optimization Score: {result.optimization_score:.4f}")
        self.logger.info(f"  Best Trial: {result.best_trial}/{result.n_trials}")
        self.logger.info(f"  Optimization Time: {result.optimization_time:.1f}s")
        
        self.logger.info("\n🎯 Optimized Parameters:")
        params = asdict(result.best_parameters)
        for param, value in params.items():
            self.logger.info(f"  {param}: {value:.4f}")
        
        self.logger.info("\n📈 Backtest Metrics:")
        for metric, value in result.backtest_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric}: {value:.4f}")
            else:
                self.logger.info(f"  {metric}: {value}")
    
    def save_optimized_parameters(self, filepath: str) -> None:
        """Save optimized parameters to file."""
        params_dict = asdict(self.best_parameters)
        
        with open(filepath, 'w') as f:
            json.dump({
                "parameters": params_dict,
                "optimization_history": [
                    {
                        "score": r.optimization_score,
                        "metrics": r.backtest_metrics,
                        "timestamp": datetime.now().isoformat()
                    }
                    for r in self.optimization_history[-10:]  # Last 10 results
                ]
            }, f, indent=2)
        
        self.logger.info(f"💾 Saved optimized parameters to {filepath}")
    
    def load_parameters(self, filepath: str) -> SRParameterSet:
        """Load parameters from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        params = SRParameterSet(**data["parameters"])
        self.best_parameters = params
        
        self.logger.info(f"📂 Loaded parameters from {filepath}")
        return params


async def setup_sr_parameter_optimizer(config: Dict[str, Any]) -> Optional[SRParameterOptimizer]:
    """Factory function to create and initialize SR parameter optimizer."""
    try:
        optimizer = SRParameterOptimizer(config)
        return optimizer
    except Exception as e:
        system_logger.error(f"Failed to setup SR parameter optimizer: {e}")
        return None