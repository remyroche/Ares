# src/training/steps/step12_final_parameters_optimization/evaluation_engine.py

"""
Advanced Evaluation Engine for Hyperparameter Optimization

This module provides comprehensive evaluation capabilities for assessing the performance
of different parameter combinations during hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from src.utils.logger import system_logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Basic metrics
    win_rate: float = 0.0
    profit_factor: float = 1.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    volatility: float = 0.0
    value_at_risk: float = 0.0
    conditional_value_at_risk: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Time metrics
    average_trade_duration: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Additional metrics
    recovery_factor: float = 0.0
    profit_factor_ratio: float = 0.0
    risk_reward_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "value_at_risk": self.value_at_risk,
            "conditional_value_at_risk": self.conditional_value_at_risk,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "average_trade_duration": self.average_trade_duration,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "recovery_factor": self.recovery_factor,
            "profit_factor_ratio": self.profit_factor_ratio,
            "risk_reward_ratio": self.risk_reward_ratio,
        }


class AdvancedEvaluationEngine:
    """Advanced evaluation engine for hyperparameter optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EvaluationEngine")
        
        # Evaluation settings
        self.risk_free_rate = config.get("risk_free_rate", 0.02)
        self.confidence_level = config.get("confidence_level", 0.95)
        self.min_trades_for_evaluation = config.get("min_trades_for_evaluation", 10)
        
        # Performance thresholds
        self.performance_thresholds = config.get("performance_thresholds", {
            "min_win_rate": 0.4,
            "min_profit_factor": 1.2,
            "max_drawdown": 0.25,
            "min_sharpe_ratio": 0.5,
        })
    
    def evaluate_parameters(
        self,
        parameters: Dict[str, Any],
        calibration_results: Dict[str, Any],
        backtest_data: Optional[pd.DataFrame] = None,
    ) -> PerformanceMetrics:
        """
        Evaluate a set of parameters using comprehensive metrics.
        
        Args:
            parameters: Parameters to evaluate
            calibration_results: Results from confidence calibration
            backtest_data: Optional backtest data for evaluation
            
        Returns:
            PerformanceMetrics object with evaluation results
        """
        try:
            self.logger.info(f"Evaluating parameters: {list(parameters.keys())}")
            
            # Simulate trading performance based on parameters
            # In real implementation, this would use actual backtesting
            performance_data = self._simulate_trading_performance(parameters, calibration_results)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_performance_metrics(performance_data)
            
            # Validate metrics against thresholds
            self._validate_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {e}")
            return PerformanceMetrics()
    
    def _simulate_trading_performance(
        self,
        parameters: Dict[str, Any],
        calibration_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simulate trading performance based on parameters."""
        try:
            # Extract key parameters
            analyst_threshold = parameters.get("analyst_confidence_threshold", 0.7)
            tactician_threshold = parameters.get("tactician_confidence_threshold", 0.65)
            ensemble_threshold = parameters.get("ensemble_confidence_threshold", 0.75)
            base_position_size = parameters.get("base_position_size", 0.05)
            stop_loss_multiplier = parameters.get("stop_loss_atr_multiplier", 2.0)
            
            # Simulate trade signals and outcomes
            n_trades = 100  # Simulate 100 trades
            np.random.seed(42)  # For reproducible results
            
            # Generate simulated trade data
            trades = []
            cumulative_return = 0.0
            returns = []
            
            for i in range(n_trades):
                # Simulate trade outcome based on confidence thresholds
                analyst_confidence = np.random.uniform(0.3, 0.9)
                tactician_confidence = np.random.uniform(0.3, 0.9)
                ensemble_confidence = np.random.uniform(0.3, 0.9)
                
                # Determine if trade should be taken
                take_trade = (
                    analyst_confidence >= analyst_threshold and
                    tactician_confidence >= tactician_threshold and
                    ensemble_confidence >= ensemble_threshold
                )
                
                if take_trade:
                    # Simulate trade outcome
                    win_probability = min(0.8, (analyst_confidence + tactician_confidence + ensemble_confidence) / 3)
                    is_win = np.random.random() < win_probability
                    
                    # Calculate position size based on confidence
                    position_size = base_position_size * (ensemble_confidence / 0.75)
                    
                    # Simulate return
                    if is_win:
                        # Winning trade
                        win_multiplier = np.random.uniform(1.5, 3.0)
                        trade_return = position_size * win_multiplier
                    else:
                        # Losing trade
                        loss_multiplier = np.random.uniform(0.5, 1.0)
                        trade_return = -position_size * loss_multiplier
                    
                    # Apply stop loss logic
                    if trade_return < -position_size * stop_loss_multiplier:
                        trade_return = -position_size * stop_loss_multiplier
                    
                    trades.append({
                        "trade_id": i,
                        "analyst_confidence": analyst_confidence,
                        "tactician_confidence": tactician_confidence,
                        "ensemble_confidence": ensemble_confidence,
                        "position_size": position_size,
                        "return": trade_return,
                        "is_win": is_win,
                        "timestamp": datetime.now() + timedelta(hours=i)
                    })
                    
                    cumulative_return += trade_return
                    returns.append(trade_return)
            
            return {
                "trades": trades,
                "returns": returns,
                "cumulative_return": cumulative_return,
                "n_trades": len(trades)
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating trading performance: {e}")
            return {"trades": [], "returns": [], "cumulative_return": 0.0, "n_trades": 0}
    
    def _calculate_performance_metrics(self, performance_data: Dict[str, Any]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            trades = performance_data.get("trades", [])
            returns = performance_data.get("returns", [])
            
            if not trades:
                return PerformanceMetrics()
            
            # Convert to DataFrame for easier calculations
            df = pd.DataFrame(trades)
            
            # Basic metrics
            total_trades = len(trades)
            winning_trades = len(df[df["is_win"] == True])
            losing_trades = len(df[df["is_win"] == False])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Return metrics
            total_return = performance_data.get("cumulative_return", 0.0)
            returns_series = pd.Series(returns)
            
            # Profit factor
            gross_profit = df[df["return"] > 0]["return"].sum()
            gross_loss = abs(df[df["return"] < 0]["return"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average win/loss
            average_win = df[df["is_win"] == True]["return"].mean() if winning_trades > 0 else 0.0
            average_loss = df[df["is_win"] == False]["return"].mean() if losing_trades > 0 else 0.0
            
            # Largest win/loss
            largest_win = df["return"].max() if len(df) > 0 else 0.0
            largest_loss = df["return"].min() if len(df) > 0 else 0.0
            
            # Risk metrics
            volatility = returns_series.std() if len(returns_series) > 0 else 0.0
            
            # Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(returns_series)
            
            # Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio(returns_series)
            
            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(returns_series)
            
            # Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0.0
            
            # Value at Risk
            var_95 = self._calculate_value_at_risk(returns_series, 0.95)
            cvar_95 = self._calculate_conditional_value_at_risk(returns_series, 0.95)
            
            # Consecutive wins/losses
            max_consecutive_wins = self._calculate_max_consecutive_wins(df)
            max_consecutive_losses = self._calculate_max_consecutive_losses(df)
            
            # Additional metrics
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0.0
            profit_factor_ratio = profit_factor if profit_factor != float('inf') else 10.0
            risk_reward_ratio = average_win / abs(average_loss) if average_loss != 0 else 0.0
            
            return PerformanceMetrics(
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                value_at_risk=var_95,
                conditional_value_at_risk=cvar_95,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                recovery_factor=recovery_factor,
                profit_factor_ratio=profit_factor_ratio,
                risk_reward_ratio=risk_reward_ratio,
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
            if excess_returns.std() == 0:
                return 0.0
            
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = returns - self.risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0
            
            return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(returns) == 0:
                return 0.0
            
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            return abs(drawdown.min())
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_value_at_risk(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        try:
            if len(returns) == 0:
                return 0.0
            
            return np.percentile(returns, (1 - confidence_level) * 100)
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_conditional_value_at_risk(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        try:
            if len(returns) == 0:
                return 0.0
            
            var = self._calculate_value_at_risk(returns, confidence_level)
            return returns[returns <= var].mean()
        except Exception as e:
            self.logger.error(f"Error calculating CVaR: {e}")
            return 0.0
    
    def _calculate_max_consecutive_wins(self, df: pd.DataFrame) -> int:
        """Calculate maximum consecutive wins."""
        try:
            if len(df) == 0:
                return 0
            
            consecutive_wins = 0
            max_consecutive_wins = 0
            
            for is_win in df["is_win"]:
                if is_win:
                    consecutive_wins += 1
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_wins = 0
            
            return max_consecutive_wins
        except Exception as e:
            self.logger.error(f"Error calculating max consecutive wins: {e}")
            return 0
    
    def _calculate_max_consecutive_losses(self, df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losses."""
        try:
            if len(df) == 0:
                return 0
            
            consecutive_losses = 0
            max_consecutive_losses = 0
            
            for is_win in df["is_win"]:
                if not is_win:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            return max_consecutive_losses
        except Exception as e:
            self.logger.error(f"Error calculating max consecutive losses: {e}")
            return 0
    
    def _validate_metrics(self, metrics: PerformanceMetrics) -> bool:
        """Validate metrics against performance thresholds."""
        try:
            thresholds = self.performance_thresholds
            
            # Check minimum requirements
            if metrics.total_trades < self.min_trades_for_evaluation:
                self.logger.warning(f"Insufficient trades for evaluation: {metrics.total_trades}")
                return False
            
            # Check win rate threshold
            if metrics.win_rate < thresholds.get("min_win_rate", 0.4):
                self.logger.warning(f"Win rate below threshold: {metrics.win_rate:.3f}")
            
            # Check profit factor threshold
            if metrics.profit_factor < thresholds.get("min_profit_factor", 1.2):
                self.logger.warning(f"Profit factor below threshold: {metrics.profit_factor:.3f}")
            
            # Check max drawdown threshold
            if metrics.max_drawdown > thresholds.get("max_drawdown", 0.25):
                self.logger.warning(f"Max drawdown above threshold: {metrics.max_drawdown:.3f}")
            
            # Check Sharpe ratio threshold
            if metrics.sharpe_ratio < thresholds.get("min_sharpe_ratio", 0.5):
                self.logger.warning(f"Sharpe ratio below threshold: {metrics.sharpe_ratio:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating metrics: {e}")
            return False
    
    def calculate_composite_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate a composite score for optimization."""
        try:
            # Weighted combination of key metrics
            weights = {
                "sharpe_ratio": 0.3,
                "profit_factor": 0.25,
                "win_rate": 0.2,
                "max_drawdown": 0.15,
                "total_return": 0.1
            }
            
            # Normalize metrics to 0-1 range
            normalized_metrics = {
                "sharpe_ratio": min(metrics.sharpe_ratio / 2.0, 1.0),  # Cap at 2.0
                "profit_factor": min(metrics.profit_factor / 3.0, 1.0),  # Cap at 3.0
                "win_rate": metrics.win_rate,
                "max_drawdown": max(0, 1 - metrics.max_drawdown / 0.5),  # Invert and cap at 50%
                "total_return": min(max(metrics.total_return / 0.5, 0), 1.0)  # Cap at 50%
            }
            
            # Calculate weighted score
            composite_score = sum(
                weights[metric] * normalized_metrics[metric]
                for metric in weights.keys()
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"Error calculating composite score: {e}")
            return 0.0
    
    def generate_evaluation_report(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        try:
            report = {
                "evaluation_summary": {
                    "total_trades": metrics.total_trades,
                    "win_rate": f"{metrics.win_rate:.3f}",
                    "profit_factor": f"{metrics.profit_factor:.3f}",
                    "total_return": f"{metrics.total_return:.3f}",
                    "sharpe_ratio": f"{metrics.sharpe_ratio:.3f}",
                    "max_drawdown": f"{metrics.max_drawdown:.3f}",
                },
                "risk_metrics": {
                    "volatility": f"{metrics.volatility:.3f}",
                    "value_at_risk_95": f"{metrics.value_at_risk:.3f}",
                    "conditional_value_at_risk_95": f"{metrics.conditional_value_at_risk:.3f}",
                    "sortino_ratio": f"{metrics.sortino_ratio:.3f}",
                    "calmar_ratio": f"{metrics.calmar_ratio:.3f}",
                },
                "trade_analysis": {
                    "winning_trades": metrics.winning_trades,
                    "losing_trades": metrics.losing_trades,
                    "average_win": f"{metrics.average_win:.3f}",
                    "average_loss": f"{metrics.average_loss:.3f}",
                    "largest_win": f"{metrics.largest_win:.3f}",
                    "largest_loss": f"{metrics.largest_loss:.3f}",
                    "max_consecutive_wins": metrics.max_consecutive_wins,
                    "max_consecutive_losses": metrics.max_consecutive_losses,
                },
                "composite_score": f"{self.calculate_composite_score(metrics):.3f}",
                "evaluation_timestamp": datetime.now().isoformat(),
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {e}")
            return {"error": str(e)}


def create_evaluation_engine(config: Dict[str, Any]) -> AdvancedEvaluationEngine:
    """Create an evaluation engine instance."""
    return AdvancedEvaluationEngine(config)


if __name__ == "__main__":
    # Test the evaluation engine
    config = {
        "risk_free_rate": 0.02,
        "confidence_level": 0.95,
        "min_trades_for_evaluation": 10,
        "performance_thresholds": {
            "min_win_rate": 0.4,
            "min_profit_factor": 1.2,
            "max_drawdown": 0.25,
            "min_sharpe_ratio": 0.5,
        }
    }
    
    engine = create_evaluation_engine(config)
    
    # Test parameters
    test_parameters = {
        "analyst_confidence_threshold": 0.7,
        "tactician_confidence_threshold": 0.65,
        "ensemble_confidence_threshold": 0.75,
        "base_position_size": 0.05,
        "stop_loss_atr_multiplier": 2.0,
    }
    
    calibration_results = {"calibration_data": "test"}
    
    # Evaluate parameters
    metrics = engine.evaluate_parameters(test_parameters, calibration_results)
    
    # Generate report
    report = engine.generate_evaluation_report(metrics)
    
    print("Evaluation Results:")
    print(f"  Win Rate: {metrics.win_rate:.3f}")
    print(f"  Profit Factor: {metrics.profit_factor:.3f}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.3f}")
    print(f"  Composite Score: {engine.calculate_composite_score(metrics):.3f}")
    
    print("\nDetailed Report:")
    for section, data in report.items():
        if isinstance(data, dict):
            print(f"\n{section}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {data}")