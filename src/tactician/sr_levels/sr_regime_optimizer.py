#!/usr/bin/env python3
"""S/R Market Regime Optimizer.

This module optimizes market regime detection and adaptation through backtesting,
ensuring that regime-specific logic is validated and optimized for performance.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.core.sr_error_handlers import sr_error_handler, SROptimizationError, SRDataError


@dataclass
class RegimePerformance:
    """Performance metrics for a specific market regime."""
    regime_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    sample_count: int
    confidence_interval: Tuple[float, float]


@dataclass
class RegimeOptimizationResult:
    """Result of regime optimization."""
    regime_weights: Dict[str, float]
    regime_performance: Dict[str, RegimePerformance]
    optimization_score: float
    backtest_periods: int
    adaptation_rate: float
    memory_decay: float
    total_samples: int


class SRRegimeOptimizer:
    """Optimizes market regime detection and adaptation through backtesting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize regime optimizer."""
        self.config = config
        self.logger = system_logger.getChild("SRRegimeOptimizer")
        self.regime_config = config.get("market_regime_detection", {})
        self.optimization_config = self.regime_config.get("regime_optimization", {})
        
        # Regime performance tracking
        self.regime_performance_history: Dict[str, List[float]] = {
            "trending": [],
            "ranging": [],
            "transitional": []
        }
        
        # Current regime weights (will be optimized)
        self.regime_weights = {
            "trending": 1.0,
            "ranging": 1.0,
            "transitional": 1.0
        }
        
        # Optimization parameters
        self.enable_optimization = self.optimization_config.get("enable_regime_optimization", True)
        self.backtest_periods = self.optimization_config.get("regime_backtest_periods", 50)
        self.performance_threshold = self.optimization_config.get("regime_performance_threshold", 0.6)
        self.adaptation_rate = self.optimization_config.get("regime_adaptation_rate", 0.1)
        self.memory_decay = self.optimization_config.get("regime_memory_decay", 0.95)
        self.min_samples = self.optimization_config.get("min_regime_samples", 10)
    
    @sr_error_handler(
        exceptions=(SROptimizationError, SRDataError),
        default_return=None,
        context="regime optimization",
        max_retries=2
    )
    async def optimize_regime_weights(
        self, 
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        target_timeframe: str = "15m"
    ) -> Optional[RegimeOptimizationResult]:
        """Optimize regime weights through backtesting."""
        try:
            if not self.enable_optimization:
                self.logger.info("Regime optimization disabled, using default weights")
                return self._create_default_result()
            
            self.logger.info(f"🎯 Starting regime optimization for {target_timeframe}")
            
            # Validate input data
            if len(market_data) < self.backtest_periods * 2:
                raise SRDataError(f"Insufficient data for regime optimization: {len(market_data)} < {self.backtest_periods * 2}")
            
            # Detect regimes in historical data
            regime_sequence = await self._detect_regime_sequence(market_data)
            
            # Backtest each regime
            regime_performance = await self._backtest_regime_performance(
                market_data, sr_levels, regime_sequence
            )
            
            # Optimize weights based on performance
            optimized_weights = await self._optimize_weights_from_performance(regime_performance)
            
            # Update regime weights
            self.regime_weights.update(optimized_weights)
            
            # Create result
            result = RegimeOptimizationResult(
                regime_weights=self.regime_weights.copy(),
                regime_performance=regime_performance,
                optimization_score=self._calculate_optimization_score(regime_performance),
                backtest_periods=self.backtest_periods,
                adaptation_rate=self.adaptation_rate,
                memory_decay=self.memory_decay,
                total_samples=len(market_data)
            )
            
            self.logger.info(f"✅ Regime optimization completed. Score: {result.optimization_score:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Regime optimization failed: {e}")
            return None
    
    async def _detect_regime_sequence(self, market_data: pd.DataFrame) -> List[str]:
        """Detect market regime sequence in historical data."""
        try:
            regimes = []
            
            # Calculate technical indicators
            sma_20 = market_data['close'].rolling(window=20).mean()
            sma_50 = market_data['close'].rolling(window=50).mean()
            rsi = self._calculate_rsi(market_data['close'], 14)
            
            # Detect regime for each period
            for i in range(len(market_data)):
                if i < 50:  # Need enough data for indicators
                    regimes.append("transitional")
                    continue
                
                # Get current values
                current_sma_20 = sma_20.iloc[i]
                current_sma_50 = sma_50.iloc[i]
                current_rsi = rsi.iloc[i]
                
                # Determine regime
                regime = self._classify_regime(current_sma_20, current_sma_50, current_rsi)
                regimes.append(regime)
            
            return regimes
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return ["transitional"] * len(market_data)
    
    def _classify_regime(self, sma_20: float, sma_50: float, rsi: float) -> str:
        """Classify market regime based on technical indicators."""
        try:
            # Calculate trend
            trend_ratio = sma_20 / sma_50
            trend_threshold = self.regime_config.get("trend_threshold", 0.02)
            
            # Determine trend
            if trend_ratio > 1 + trend_threshold:
                trend = "uptrend"
            elif trend_ratio < 1 - trend_threshold:
                trend = "downtrend"
            else:
                trend = "sideways"
            
            # Determine strength
            rsi_overbought = self.regime_config.get("rsi_overbought", 70)
            rsi_oversold = self.regime_config.get("rsi_oversold", 30)
            
            if rsi > rsi_overbought:
                strength = "overbought"
            elif rsi < rsi_oversold:
                strength = "oversold"
            else:
                strength = "neutral"
            
            # Classify regime
            if trend in ["uptrend", "downtrend"] and strength == "neutral":
                return "trending"
            elif trend == "sideways":
                return "ranging"
            else:
                return "transitional"
                
        except Exception as e:
            self.logger.warning(f"Regime classification failed: {e}")
            return "transitional"
    
    async def _backtest_regime_performance(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        regime_sequence: List[str]
    ) -> Dict[str, RegimePerformance]:
        """Backtest performance for each regime."""
        try:
            regime_performance = {}
            
            for regime_type in ["trending", "ranging", "transitional"]:
                # Find periods with this regime
                regime_indices = [i for i, regime in enumerate(regime_sequence) if regime == regime_type]
                
                if len(regime_indices) < self.min_samples:
                    # Not enough samples, use default performance
                    regime_performance[regime_type] = self._create_default_regime_performance(regime_type)
                    continue
                
                # Extract regime-specific data
                regime_data = market_data.iloc[regime_indices]
                
                # Calculate performance metrics
                performance = await self._calculate_regime_performance(
                    regime_data, sr_levels, regime_type
                )
                
                regime_performance[regime_type] = performance
                
                # Update performance history
                self.regime_performance_history[regime_type].append(performance.accuracy)
                
                # Apply memory decay
                if len(self.regime_performance_history[regime_type]) > 100:
                    self.regime_performance_history[regime_type] = self.regime_performance_history[regime_type][-50:]
            
            return regime_performance
            
        except Exception as e:
            self.logger.error(f"Regime backtesting failed: {e}")
            return self._create_default_regime_performance_dict()
    
    async def _calculate_regime_performance(
        self,
        regime_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        regime_type: str
    ) -> RegimePerformance:
        """Calculate performance metrics for a specific regime."""
        try:
            if len(regime_data) < 10:
                return self._create_default_regime_performance(regime_type)
            
            # Simulate S/R level interactions for this regime
            interactions = await self._simulate_sr_interactions(regime_data, sr_levels)
            
            if not interactions:
                return self._create_default_regime_performance(regime_type)
            
            # Calculate metrics
            accuracy = self._calculate_accuracy(interactions)
            precision = self._calculate_precision(interactions)
            recall = self._calculate_recall(interactions)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate trading metrics
            profit_factor = self._calculate_profit_factor(interactions)
            sharpe_ratio = self._calculate_sharpe_ratio(interactions)
            max_drawdown = self._calculate_max_drawdown(interactions)
            win_rate = self._calculate_win_rate(interactions)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(interactions)
            
            return RegimePerformance(
                regime_type=regime_type,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                sample_count=len(interactions),
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            self.logger.error(f"Performance calculation failed for {regime_type}: {e}")
            return self._create_default_regime_performance(regime_type)
    
    async def _simulate_sr_interactions(
        self,
        regime_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simulate S/R level interactions for backtesting."""
        try:
            interactions = []
            
            for i in range(len(regime_data) - 1):
                current_price = regime_data['close'].iloc[i]
                next_price = regime_data['close'].iloc[i + 1]
                
                # Check for S/R level interactions
                for level in sr_levels:
                    level_price = level.get('price', 0)
                    level_type = level.get('type', 'unknown')
                    
                    if level_price <= 0:
                        continue
                    
                    # Check if price is near level
                    proximity = abs(current_price - level_price) / level_price
                    if proximity < 0.02:  # 2% proximity threshold
                        
                        # Simulate interaction
                        interaction = {
                            'level_price': level_price,
                            'level_type': level_type,
                            'current_price': current_price,
                            'next_price': next_price,
                            'proximity': proximity,
                            'volume': regime_data['volume'].iloc[i],
                            'timestamp': regime_data.index[i]
                        }
                        
                        # Determine if level held or broke
                        if level_type == 'support':
                            interaction['level_held'] = next_price >= level_price * 0.99
                        else:  # resistance
                            interaction['level_held'] = next_price <= level_price * 1.01
                        
                        interactions.append(interaction)
            
            return interactions
            
        except Exception as e:
            self.logger.error(f"SR interaction simulation failed: {e}")
            return []
    
    async def _optimize_weights_from_performance(
        self,
        regime_performance: Dict[str, RegimePerformance]
    ) -> Dict[str, float]:
        """Optimize regime weights based on performance."""
        try:
            optimized_weights = {}
            
            for regime_type, performance in regime_performance.items():
                # Calculate performance score (weighted combination of metrics)
                performance_score = (
                    performance.accuracy * 0.3 +
                    performance.f1_score * 0.3 +
                    performance.profit_factor * 0.2 +
                    performance.sharpe_ratio * 0.2
                )
                
                # Apply performance threshold
                if performance_score < self.performance_threshold:
                    # Reduce weight for poor performance
                    new_weight = self.regime_weights[regime_type] * 0.8
                else:
                    # Increase weight for good performance
                    new_weight = self.regime_weights[regime_type] * (1 + self.adaptation_rate * (performance_score - self.performance_threshold))
                
                # Apply memory decay to historical performance
                if regime_type in self.regime_performance_history:
                    historical_avg = np.mean(self.regime_performance_history[regime_type][-10:]) if self.regime_performance_history[regime_type] else 0.5
                    new_weight = new_weight * (1 - self.memory_decay) + historical_avg * self.memory_decay
                
                # Ensure weight is within reasonable bounds
                optimized_weights[regime_type] = max(0.1, min(2.0, new_weight))
            
            # Normalize weights
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {k: v / total_weight * 3.0 for k, v in optimized_weights.items()}
            
            return optimized_weights
            
        except Exception as e:
            self.logger.error(f"Weight optimization failed: {e}")
            return self.regime_weights.copy()
    
    def _calculate_optimization_score(self, regime_performance: Dict[str, RegimePerformance]) -> float:
        """Calculate overall optimization score."""
        try:
            if not regime_performance:
                return 0.0
            
            # Weighted average of regime performance
            total_score = 0.0
            total_weight = 0.0
            
            for regime_type, performance in regime_performance.items():
                weight = self.regime_weights.get(regime_type, 1.0)
                regime_score = (
                    performance.accuracy * 0.4 +
                    performance.f1_score * 0.3 +
                    performance.profit_factor * 0.2 +
                    performance.sharpe_ratio * 0.1
                )
                
                total_score += regime_score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Optimization score calculation failed: {e}")
            return 0.0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral RSI
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_accuracy(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate accuracy of S/R level predictions."""
        if not interactions:
            return 0.5
        
        correct_predictions = sum(1 for i in interactions if i.get('level_held', False))
        return correct_predictions / len(interactions)
    
    def _calculate_precision(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate precision of S/R level predictions."""
        if not interactions:
            return 0.5
        
        true_positives = sum(1 for i in interactions if i.get('level_held', False))
        false_positives = len(interactions) - true_positives
        
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.5
    
    def _calculate_recall(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate recall of S/R level predictions."""
        # For S/R levels, recall is similar to accuracy
        return self._calculate_accuracy(interactions)
    
    def _calculate_profit_factor(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate profit factor from interactions."""
        if not interactions:
            return 1.0
        
        profits = []
        losses = []
        
        for interaction in interactions:
            if interaction.get('level_held', False):
                # Level held, assume small profit
                profits.append(0.01)
            else:
                # Level broke, assume small loss
                losses.append(0.01)
        
        total_profit = sum(profits) if profits else 0.01
        total_loss = sum(losses) if losses else 0.01
        
        return total_profit / total_loss if total_loss > 0 else 1.0
    
    def _calculate_sharpe_ratio(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate Sharpe ratio from interactions."""
        if len(interactions) < 2:
            return 0.0
        
        returns = [0.01 if i.get('level_held', False) else -0.01 for i in interactions]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return mean_return / std_return if std_return > 0 else 0.0
    
    def _calculate_max_drawdown(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown from interactions."""
        if not interactions:
            return 0.0
        
        cumulative_returns = []
        cumulative = 0.0
        
        for interaction in interactions:
            if interaction.get('level_held', False):
                cumulative += 0.01
            else:
                cumulative -= 0.01
            cumulative_returns.append(cumulative)
        
        peak = cumulative_returns[0]
        max_dd = 0.0
        
        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            dd = peak - ret
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_win_rate(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate win rate from interactions."""
        if not interactions:
            return 0.5
        
        wins = sum(1 for i in interactions if i.get('level_held', False))
        return wins / len(interactions)
    
    def _calculate_confidence_interval(self, interactions: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate confidence interval for performance."""
        if len(interactions) < 10:
            return (0.0, 1.0)
        
        accuracies = [1.0 if i.get('level_held', False) else 0.0 for i in interactions]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        # 95% confidence interval
        margin = 1.96 * std_acc / np.sqrt(len(interactions))
        return (max(0.0, mean_acc - margin), min(1.0, mean_acc + margin))
    
    def _create_default_regime_performance(self, regime_type: str) -> RegimePerformance:
        """Create default performance for a regime."""
        return RegimePerformance(
            regime_type=regime_type,
            accuracy=0.5,
            precision=0.5,
            recall=0.5,
            f1_score=0.5,
            profit_factor=1.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.5,
            sample_count=0,
            confidence_interval=(0.0, 1.0)
        )
    
    def _create_default_regime_performance_dict(self) -> Dict[str, RegimePerformance]:
        """Create default performance dictionary."""
        return {
            regime_type: self._create_default_regime_performance(regime_type)
            for regime_type in ["trending", "ranging", "transitional"]
        }
    
    def _create_default_result(self) -> RegimeOptimizationResult:
        """Create default optimization result."""
        return RegimeOptimizationResult(
            regime_weights=self.regime_weights.copy(),
            regime_performance=self._create_default_regime_performance_dict(),
            optimization_score=0.5,
            backtest_periods=0,
            adaptation_rate=self.adaptation_rate,
            memory_decay=self.memory_decay,
            total_samples=0
        )
    
    def get_optimized_regime_weights(self) -> Dict[str, float]:
        """Get current optimized regime weights."""
        return self.regime_weights.copy()
    
    def update_regime_weights(self, new_weights: Dict[str, float]) -> None:
        """Update regime weights."""
        self.regime_weights.update(new_weights)
        self.logger.info(f"Updated regime weights: {self.regime_weights}")