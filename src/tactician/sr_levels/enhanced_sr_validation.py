#!/usr/bin/env python3
"""Enhanced S/R Validation and Backtesting Module.

This module provides comprehensive validation and backtesting capabilities
for S/R level detection with advanced performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger


@dataclass
class ValidationResult:
    """Result of S/R level validation."""
    level_id: str
    level_price: float
    level_type: str
    validation_score: float
    bounce_rate: float
    false_breakout_rate: float
    volume_confirmation_rate: float
    time_to_breakout: Optional[float]
    max_bounce_ratio: float
    avg_bounce_ratio: float
    touch_count: int
    failure_count: int
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    metadata: Dict[str, Any]


@dataclass
class BacktestResult:
    """Result of S/R backtesting."""
    total_levels: int
    validated_levels: int
    avg_validation_score: float
    avg_bounce_rate: float
    avg_false_breakout_rate: float
    avg_volume_confirmation: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    successful_trades: int
    failed_trades: int
    performance_metrics: Dict[str, float]
    level_performance: Dict[str, ValidationResult]


class EnhancedSRValidator:
    """Enhanced S/R validator with comprehensive backtesting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced S/R validator."""
        self.config = config
        self.logger = system_logger.getChild("EnhancedSRValidator")
        
        # Validation parameters
        self.bounce_threshold = config.get("bounce_threshold", 0.001)
        self.breakout_threshold = config.get("breakout_threshold", 0.005)
        self.volume_spike_threshold = config.get("volume_spike_threshold", 1.5)
        self.min_validation_period = config.get("min_validation_period", 100)
        self.max_validation_period = config.get("max_validation_period", 1000)
        
        # Statistical parameters
        self.confidence_level = config.get("confidence_level", 0.95)
        self.min_sample_size = config.get("min_sample_size", 10)
        
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="validate SR levels"
    )
    @traced(span_name="EnhancedSR.validate_levels")
    def validate_sr_levels(
        self, 
        levels: List[Any], 
        market_data: pd.DataFrame
    ) -> List[ValidationResult]:
        """
        Validate S/R levels with comprehensive backtesting.
        
        Args:
            levels: List of S/R levels to validate
            market_data: Historical market data for validation
            
        Returns:
            List of validation results
        """
        try:
            self.logger.info(f"🔍 Validating {len(levels)} S/R levels...")
            
            validation_results = []
            
            for i, level in enumerate(levels):
                try:
                    result = self._validate_single_level(level, market_data, i)
                    if result:
                        validation_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Validation failed for level {i}: {e}")
                    continue
            
            self.logger.info(f"✅ Validated {len(validation_results)} S/R levels")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"S/R validation failed: {e}")
            return []
    
    def _validate_single_level(
        self, 
        level: Any, 
        market_data: pd.DataFrame, 
        level_id: int
    ) -> Optional[ValidationResult]:
        """Validate a single S/R level."""
        try:
            # Extract level information
            level_price = getattr(level, 'price', 0.0)
            level_type = getattr(level, 'type', 'unknown')
            
            if level_price <= 0:
                return None
            
            # Find validation period
            validation_data = self._get_validation_period(level, market_data)
            if validation_data is None or len(validation_data) < self.min_sample_size:
                return None
            
            # Calculate validation metrics
            bounce_rate = self._calculate_bounce_rate(level, validation_data)
            false_breakout_rate = self._calculate_false_breakout_rate(level, validation_data)
            volume_confirmation_rate = self._calculate_volume_confirmation_rate(level, validation_data)
            time_to_breakout = self._calculate_time_to_breakout(level, validation_data)
            bounce_ratios = self._calculate_bounce_ratios(level, validation_data)
            touch_count = self._count_touches(level, validation_data)
            failure_count = self._count_failures(level, validation_data)
            
            # Calculate statistical metrics
            confidence_interval = self._calculate_confidence_interval(bounce_ratios)
            statistical_significance = self._calculate_statistical_significance(level, validation_data)
            
            # Calculate overall validation score
            validation_score = self._calculate_validation_score(
                bounce_rate, false_breakout_rate, volume_confirmation_rate,
                touch_count, failure_count, statistical_significance
            )
            
            # Create validation result
            result = ValidationResult(
                level_id=f"level_{level_id}",
                level_price=level_price,
                level_type=level_type,
                validation_score=validation_score,
                bounce_rate=bounce_rate,
                false_breakout_rate=false_breakout_rate,
                volume_confirmation_rate=volume_confirmation_rate,
                time_to_breakout=time_to_breakout,
                max_bounce_ratio=bounce_ratios['max'] if bounce_ratios else 0.0,
                avg_bounce_ratio=bounce_ratios['avg'] if bounce_ratios else 0.0,
                touch_count=touch_count,
                failure_count=failure_count,
                confidence_interval=confidence_interval,
                statistical_significance=statistical_significance,
                metadata={
                    'validation_period_bars': len(validation_data),
                    'level_strength': getattr(level, 'strength', 0.0),
                    'level_confidence': getattr(level, 'confidence_score', 0.0)
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Single level validation failed: {e}")
            return None
    
    def _get_validation_period(
        self, 
        level: Any, 
        market_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Get validation period for S/R level."""
        try:
            # Get level creation time
            first_touch_time = getattr(level, 'first_touch_time', None)
            if first_touch_time is None:
                # Use middle of data for validation
                start_idx = len(market_data) // 2
            else:
                # Find index of first touch time
                start_idx = market_data.index.get_loc(first_touch_time) if first_touch_time in market_data.index else len(market_data) // 2
            
            # Determine validation period length
            validation_length = min(
                max(self.min_validation_period, len(market_data) - start_idx),
                self.max_validation_period
            )
            
            # Extract validation data
            end_idx = min(start_idx + validation_length, len(market_data))
            validation_data = market_data.iloc[start_idx:end_idx]
            
            return validation_data if len(validation_data) >= self.min_sample_size else None
            
        except Exception as e:
            self.logger.warning(f"Validation period extraction failed: {e}")
            return None
    
    def _calculate_bounce_rate(self, level: Any, data: pd.DataFrame) -> float:
        """Calculate bounce rate for S/R level."""
        try:
            level_price = getattr(level, 'price', 0.0)
            level_type = getattr(level, 'type', 'unknown')
            threshold = level_price * self.bounce_threshold
            
            bounces = 0
            total_touches = 0
            
            for i in range(len(data) - 1):
                current_row = data.iloc[i]
                next_row = data.iloc[i + 1]
                
                if level_type == 'support':
                    # Check for support touch
                    if abs(current_row['low'] - level_price) <= threshold:
                        total_touches += 1
                        # Check for bounce (next high above level)
                        if next_row['high'] > level_price + threshold:
                            bounces += 1
                else:  # resistance
                    # Check for resistance touch
                    if abs(current_row['high'] - level_price) <= threshold:
                        total_touches += 1
                        # Check for bounce (next low below level)
                        if next_row['low'] < level_price - threshold:
                            bounces += 1
            
            return bounces / total_touches if total_touches > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Bounce rate calculation failed: {e}")
            return 0.0
    
    def _calculate_false_breakout_rate(self, level: Any, data: pd.DataFrame) -> float:
        """Calculate false breakout rate for S/R level."""
        try:
            level_price = getattr(level, 'price', 0.0)
            level_type = getattr(level, 'type', 'unknown')
            threshold = level_price * self.breakout_threshold
            
            false_breakouts = 0
            total_breakouts = 0
            
            for i in range(len(data) - 5):  # Need 5 bars to check for false breakout
                current_row = data.iloc[i]
                
                if level_type == 'support':
                    # Check for support breakout
                    if current_row['close'] < level_price - threshold:
                        total_breakouts += 1
                        # Check if price returns above level within 5 bars
                        for j in range(i + 1, min(i + 6, len(data))):
                            if data.iloc[j]['close'] > level_price + threshold:
                                false_breakouts += 1
                                break
                else:  # resistance
                    # Check for resistance breakout
                    if current_row['close'] > level_price + threshold:
                        total_breakouts += 1
                        # Check if price returns below level within 5 bars
                        for j in range(i + 1, min(i + 6, len(data))):
                            if data.iloc[j]['close'] < level_price - threshold:
                                false_breakouts += 1
                                break
            
            return false_breakouts / total_breakouts if total_breakouts > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"False breakout rate calculation failed: {e}")
            return 0.0
    
    def _calculate_volume_confirmation_rate(self, level: Any, data: pd.DataFrame) -> float:
        """Calculate volume confirmation rate for S/R level."""
        try:
            if 'volume' not in data.columns:
                return 0.0
            
            level_price = getattr(level, 'price', 0.0)
            level_type = getattr(level, 'type', 'unknown')
            threshold = level_price * self.bounce_threshold
            
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_confirmations = 0
            total_touches = 0
            
            for i, row in data.iterrows():
                if level_type == 'support':
                    if abs(row['low'] - level_price) <= threshold:
                        total_touches += 1
                        if row['volume'] > volume_ma.iloc[i] * self.volume_spike_threshold:
                            volume_confirmations += 1
                else:  # resistance
                    if abs(row['high'] - level_price) <= threshold:
                        total_touches += 1
                        if row['volume'] > volume_ma.iloc[i] * self.volume_spike_threshold:
                            volume_confirmations += 1
            
            return volume_confirmations / total_touches if total_touches > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Volume confirmation rate calculation failed: {e}")
            return 0.0
    
    def _calculate_time_to_breakout(self, level: Any, data: pd.DataFrame) -> Optional[float]:
        """Calculate average time to breakout for S/R level."""
        try:
            level_price = getattr(level, 'price', 0.0)
            level_type = getattr(level, 'type', 'unknown')
            threshold = level_price * self.breakout_threshold
            
            breakout_times = []
            
            for i in range(len(data) - 1):
                current_row = data.iloc[i]
                
                if level_type == 'support':
                    if current_row['close'] < level_price - threshold:
                        # Find when level was first touched
                        for j in range(max(0, i - 50), i):
                            if abs(data.iloc[j]['low'] - level_price) <= threshold:
                                breakout_times.append(i - j)
                                break
                else:  # resistance
                    if current_row['close'] > level_price + threshold:
                        # Find when level was first touched
                        for j in range(max(0, i - 50), i):
                            if abs(data.iloc[j]['high'] - level_price) <= threshold:
                                breakout_times.append(i - j)
                                break
            
            return np.mean(breakout_times) if breakout_times else None
            
        except Exception as e:
            self.logger.warning(f"Time to breakout calculation failed: {e}")
            return None
    
    def _calculate_bounce_ratios(self, level: Any, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate bounce ratios for S/R level."""
        try:
            level_price = getattr(level, 'price', 0.0)
            level_type = getattr(level, 'type', 'unknown')
            threshold = level_price * self.bounce_threshold
            
            bounce_ratios = []
            
            for i in range(len(data) - 1):
                current_row = data.iloc[i]
                next_row = data.iloc[i + 1]
                
                if level_type == 'support':
                    if abs(current_row['low'] - level_price) <= threshold:
                        bounce_ratio = (next_row['high'] - level_price) / level_price
                        if bounce_ratio > 0:
                            bounce_ratios.append(bounce_ratio)
                else:  # resistance
                    if abs(current_row['high'] - level_price) <= threshold:
                        bounce_ratio = (level_price - next_row['low']) / level_price
                        if bounce_ratio > 0:
                            bounce_ratios.append(bounce_ratio)
            
            if bounce_ratios:
                return {
                    'avg': np.mean(bounce_ratios),
                    'max': np.max(bounce_ratios),
                    'min': np.min(bounce_ratios),
                    'std': np.std(bounce_ratios)
                }
            else:
                return {'avg': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}
                
        except Exception as e:
            self.logger.warning(f"Bounce ratios calculation failed: {e}")
            return {'avg': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}
    
    def _count_touches(self, level: Any, data: pd.DataFrame) -> int:
        """Count touches of S/R level."""
        try:
            level_price = getattr(level, 'price', 0.0)
            level_type = getattr(level, 'type', 'unknown')
            threshold = level_price * self.bounce_threshold
            
            touches = 0
            
            for i, row in data.iterrows():
                if level_type == 'support':
                    if abs(row['low'] - level_price) <= threshold:
                        touches += 1
                else:  # resistance
                    if abs(row['high'] - level_price) <= threshold:
                        touches += 1
            
            return touches
            
        except Exception as e:
            self.logger.warning(f"Touch counting failed: {e}")
            return 0
    
    def _count_failures(self, level: Any, data: pd.DataFrame) -> int:
        """Count failures (breakouts) of S/R level."""
        try:
            level_price = getattr(level, 'price', 0.0)
            level_type = getattr(level, 'type', 'unknown')
            threshold = level_price * self.breakout_threshold
            
            failures = 0
            
            for i, row in data.iterrows():
                if level_type == 'support':
                    if row['close'] < level_price - threshold:
                        failures += 1
                else:  # resistance
                    if row['close'] > level_price + threshold:
                        failures += 1
            
            return failures
            
        except Exception as e:
            self.logger.warning(f"Failure counting failed: {e}")
            return 0
    
    def _calculate_confidence_interval(self, bounce_ratios: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval for bounce ratios."""
        try:
            if not bounce_ratios or bounce_ratios['std'] == 0:
                return (0.0, 0.0)
            
            # Simple confidence interval calculation
            mean = bounce_ratios['avg']
            std = bounce_ratios['std']
            n = 10  # Approximate sample size
            
            # 95% confidence interval
            margin_of_error = 1.96 * (std / np.sqrt(n))
            
            return (mean - margin_of_error, mean + margin_of_error)
            
        except Exception as e:
            self.logger.warning(f"Confidence interval calculation failed: {e}")
            return (0.0, 0.0)
    
    def _calculate_statistical_significance(self, level: Any, data: pd.DataFrame) -> float:
        """Calculate statistical significance of S/R level."""
        try:
            # Simple statistical significance based on touch count and bounce rate
            touch_count = self._count_touches(level, data)
            bounce_rate = self._calculate_bounce_rate(level, data)
            
            # Higher significance for more touches and higher bounce rate
            significance = min((touch_count / 10.0) * bounce_rate, 1.0)
            
            return significance
            
        except Exception as e:
            self.logger.warning(f"Statistical significance calculation failed: {e}")
            return 0.0
    
    def _calculate_validation_score(
        self,
        bounce_rate: float,
        false_breakout_rate: float,
        volume_confirmation_rate: float,
        touch_count: int,
        failure_count: int,
        statistical_significance: float
    ) -> float:
        """Calculate overall validation score for S/R level."""
        try:
            # Weighted scoring system
            weights = {
                'bounce_rate': 0.3,
                'false_breakout_rate': 0.2,
                'volume_confirmation': 0.2,
                'touch_count': 0.15,
                'statistical_significance': 0.15
            }
            
            # Normalize touch count (0-1 scale)
            normalized_touches = min(touch_count / 10.0, 1.0)
            
            # Penalty for failures
            failure_penalty = min(failure_count * 0.1, 0.3)
            
            # Calculate weighted score
            score = (
                weights['bounce_rate'] * bounce_rate +
                weights['false_breakout_rate'] * (1.0 - false_breakout_rate) +  # Invert false breakout rate
                weights['volume_confirmation'] * volume_confirmation_rate +
                weights['touch_count'] * normalized_touches +
                weights['statistical_significance'] * statistical_significance
            ) - failure_penalty
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f"Validation score calculation failed: {e}")
            return 0.0
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="run comprehensive backtest"
    )
    @traced(span_name="EnhancedSR.backtest")
    def run_comprehensive_backtest(
        self,
        levels: List[Any],
        market_data: pd.DataFrame
    ) -> Optional[BacktestResult]:
        """
        Run comprehensive backtest on S/R levels.
        
        Args:
            levels: List of S/R levels to backtest
            market_data: Historical market data
            
        Returns:
            Comprehensive backtest result
        """
        try:
            self.logger.info(f"🚀 Running comprehensive backtest on {len(levels)} S/R levels...")
            
            # Validate all levels
            validation_results = self.validate_sr_levels(levels, market_data)
            
            if not validation_results:
                self.logger.warning("No validation results available for backtest")
                return None
            
            # Calculate aggregate metrics
            total_levels = len(levels)
            validated_levels = len(validation_results)
            
            avg_validation_score = np.mean([r.validation_score for r in validation_results])
            avg_bounce_rate = np.mean([r.bounce_rate for r in validation_results])
            avg_false_breakout_rate = np.mean([r.false_breakout_rate for r in validation_results])
            avg_volume_confirmation = np.mean([r.volume_confirmation_rate for r in validation_results])
            
            # Calculate trading performance metrics
            performance_metrics = self._calculate_trading_performance(validation_results, market_data)
            
            # Create backtest result
            result = BacktestResult(
                total_levels=total_levels,
                validated_levels=validated_levels,
                avg_validation_score=avg_validation_score,
                avg_bounce_rate=avg_bounce_rate,
                avg_false_breakout_rate=avg_false_breakout_rate,
                avg_volume_confirmation=avg_volume_confirmation,
                sharpe_ratio=performance_metrics.get('sharpe_ratio', 0.0),
                max_drawdown=performance_metrics.get('max_drawdown', 0.0),
                win_rate=performance_metrics.get('win_rate', 0.0),
                profit_factor=performance_metrics.get('profit_factor', 0.0),
                total_trades=performance_metrics.get('total_trades', 0),
                successful_trades=performance_metrics.get('successful_trades', 0),
                failed_trades=performance_metrics.get('failed_trades', 0),
                performance_metrics=performance_metrics,
                level_performance={r.level_id: r for r in validation_results}
            )
            
            self.logger.info(f"✅ Backtest completed: {validated_levels}/{total_levels} levels validated")
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive backtest failed: {e}")
            return None
    
    def _calculate_trading_performance(
        self,
        validation_results: List[ValidationResult],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate trading performance metrics."""
        try:
            # Simulate trades based on validation results
            trades = []
            
            for result in validation_results:
                if result.validation_score > 0.6:  # Only trade high-quality levels
                    # Simulate entry and exit
                    entry_price = result.level_price
                    
                    # Find actual price movements
                    level_data = market_data[
                        (market_data.index >= market_data.index[0]) &
                        (market_data.index <= market_data.index[-1])
                    ]
                    
                    if len(level_data) > 0:
                        # Simple trade simulation
                        if result.level_type == 'support':
                            # Buy at support, sell at resistance or stop loss
                            exit_price = level_data['high'].max() * 0.99  # 1% below high
                            stop_loss = entry_price * 0.98  # 2% stop loss
                            
                            if exit_price > stop_loss:
                                profit = (exit_price - entry_price) / entry_price
                                trades.append(profit)
                            else:
                                trades.append(-0.02)  # Stop loss hit
                        else:  # resistance
                            # Sell at resistance, buy at support or stop loss
                            exit_price = level_data['low'].min() * 1.01  # 1% above low
                            stop_loss = entry_price * 1.02  # 2% stop loss
                            
                            if exit_price < stop_loss:
                                profit = (entry_price - exit_price) / entry_price
                                trades.append(profit)
                            else:
                                trades.append(-0.02)  # Stop loss hit
            
            if not trades:
                return {
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'total_trades': 0,
                    'successful_trades': 0,
                    'failed_trades': 0
                }
            
            # Calculate performance metrics
            trades_array = np.array(trades)
            
            win_rate = np.mean(trades_array > 0)
            successful_trades = np.sum(trades_array > 0)
            failed_trades = np.sum(trades_array <= 0)
            
            # Calculate Sharpe ratio
            if np.std(trades_array) > 0:
                sharpe_ratio = np.mean(trades_array) / np.std(trades_array) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            cumulative_returns = np.cumprod(1 + trades_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Calculate profit factor
            gross_profit = np.sum(trades_array[trades_array > 0])
            gross_loss = abs(np.sum(trades_array[trades_array < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(trades),
                'successful_trades': successful_trades,
                'failed_trades': failed_trades,
                'avg_return': np.mean(trades_array),
                'volatility': np.std(trades_array)
            }
            
        except Exception as e:
            self.logger.warning(f"Trading performance calculation failed: {e}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'successful_trades': 0,
                'failed_trades': 0
            }