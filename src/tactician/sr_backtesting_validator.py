# src/tactician/sr_backtesting_validator.py

"""
S/R Backtesting Validator

This module implements comprehensive backtesting to validate whether detected
S/R levels are actually effective. It simulates price interactions with S/R
levels and measures their success in predicting support/resistance behavior.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


@dataclass
class SRLevelTest:
    """Individual S/R level test result."""
    
    level_price: float
    level_type: str  # "support" or "resistance"
    test_start_time: datetime
    test_end_time: datetime
    touches: int = 0
    bounces: int = 0
    breakouts: int = 0
    false_breakouts: int = 0
    bounce_rate: float = 0.0
    breakout_rate: float = 0.0
    false_breakout_rate: float = 0.0
    avg_bounce_strength: float = 0.0
    avg_breakout_strength: float = 0.0
    level_strength: float = 0.0
    confidence_score: float = 0.0


@dataclass
class BacktestResult:
    """Result of S/R backtesting."""
    
    # Overall metrics
    total_levels_tested: int = 0
    successful_levels: int = 0
    overall_bounce_rate: float = 0.0
    overall_breakout_rate: float = 0.0
    overall_false_breakout_rate: float = 0.0
    
    # Support vs Resistance metrics
    support_bounce_rate: float = 0.0
    resistance_bounce_rate: float = 0.0
    support_breakout_rate: float = 0.0
    resistance_breakout_rate: float = 0.0
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    
    # Level quality metrics
    avg_level_strength: float = 0.0
    avg_confidence_score: float = 0.0
    level_detection_accuracy: float = 0.0
    
    # Detailed results
    level_tests: List[SRLevelTest] = None
    trade_signals: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.level_tests is None:
            self.level_tests = []
        if self.trade_signals is None:
            self.trade_signals = []


class SRBacktestingValidator:
    """
    Comprehensive S/R backtesting validator.
    
    This class implements proper backtesting to validate S/R levels by:
    1. Detecting when price touches S/R levels
    2. Measuring bounce vs breakout behavior
    3. Calculating success metrics and confidence scores
    4. Simulating trading strategies based on S/R levels
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the S/R backtesting validator."""
        self.config = config
        self.logger = system_logger.getChild("SRBacktestingValidator")
        
        # Backtesting configuration
        self.backtest_config = config.get("sr_backtesting", {})
        self.touch_threshold = self.backtest_config.get("touch_threshold", 0.001)  # 0.1% touch threshold
        self.bounce_threshold = self.backtest_config.get("bounce_threshold", 0.005)  # 0.5% bounce threshold
        self.breakout_threshold = self.backtest_config.get("breakout_threshold", 0.01)  # 1% breakout threshold
        self.false_breakout_threshold = self.backtest_config.get("false_breakout_threshold", 0.02)  # 2% false breakout
        self.confirmation_periods = self.backtest_config.get("confirmation_periods", 3)
        self.min_touches = self.backtest_config.get("min_touches", 2)
        
        # Trading simulation configuration
        self.enable_trading_simulation = self.backtest_config.get("enable_trading_simulation", True)
        self.position_size = self.backtest_config.get("position_size", 0.1)  # 10% of capital
        self.stop_loss = self.backtest_config.get("stop_loss", 0.02)  # 2% stop loss
        self.take_profit = self.backtest_config.get("take_profit", 0.04)  # 4% take profit
        
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid data for S/R backtesting"),
            AttributeError: (None, "Backtesting validator not properly initialized"),
        },
        default_return=None,
        context="S/R backtesting validation"
    )
    async def validate_sr_levels(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        current_price: float
    ) -> Optional[BacktestResult]:
        """
        Validate S/R levels through comprehensive backtesting.
        
        Args:
            market_data: Historical market data for backtesting
            sr_levels: List of detected S/R levels
            current_price: Current market price
            
        Returns:
            BacktestResult: Comprehensive backtesting results
        """
        try:
            self.logger.info(f"🔍 Starting S/R level validation with {len(sr_levels)} levels")
            
            # Initialize results
            result = BacktestResult()
            result.total_levels_tested = len(sr_levels)
            
            # Test each S/R level
            for level in sr_levels:
                level_test = await self._test_single_level(market_data, level, current_price)
                if level_test:
                    result.level_tests.append(level_test)
                    
                    # Update overall metrics
                    if level_test.bounce_rate > 0.6:  # Consider successful if >60% bounce rate
                        result.successful_levels += 1
            
            # Calculate overall metrics
            await self._calculate_overall_metrics(result)
            
            # Simulate trading if enabled
            if self.enable_trading_simulation:
                await self._simulate_trading_strategy(market_data, sr_levels, result)
            
            # Calculate final performance metrics
            await self._calculate_performance_metrics(result)
            
            self.logger.info(f"✅ S/R validation completed. Success rate: {result.win_rate:.2%}")
            return result
            
        except Exception as e:
            self.logger.error(f"S/R validation failed: {e}")
            return None
    
    async def _test_single_level(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any],
        current_price: float
    ) -> Optional[SRLevelTest]:
        """Test a single S/R level."""
        try:
            level_price = level.get("price", 0.0)
            level_type = level.get("type", "support")
            level_strength = level.get("enhanced_strength", level.get("strength", 0.5))
            
            # Create test result
            test = SRLevelTest(
                level_price=level_price,
                level_type=level_type,
                test_start_time=market_data.index[0],
                test_end_time=market_data.index[-1],
                level_strength=level_strength
            )
            
            # Analyze price interactions with this level
            touches, bounces, breakouts, false_breakouts = await self._analyze_level_interactions(
                market_data, level_price, level_type
            )
            
            # Update test results
            test.touches = touches
            test.bounces = bounces
            test.breakouts = breakouts
            test.false_breakouts = false_breakouts
            
            # Calculate rates
            if touches > 0:
                test.bounce_rate = bounces / touches
                test.breakout_rate = breakouts / touches
                test.false_breakout_rate = false_breakouts / touches
            
            # Calculate confidence score
            test.confidence_score = self._calculate_level_confidence(test)
            
            return test
            
        except Exception as e:
            self.logger.error(f"Failed to test level {level.get('price', 0)}: {e}")
            return None
    
    async def _analyze_level_interactions(
        self,
        market_data: pd.DataFrame,
        level_price: float,
        level_type: str
    ) -> Tuple[int, int, int, int]:
        """
        Analyze how price interacts with a specific S/R level.
        
        Returns:
            Tuple of (touches, bounces, breakouts, false_breakouts)
        """
        try:
            touches = 0
            bounces = 0
            breakouts = 0
            false_breakouts = 0
            
            # Define touch zone around the level
            touch_zone_upper = level_price * (1 + self.touch_threshold)
            touch_zone_lower = level_price * (1 - self.touch_threshold)
            
            i = 0
            while i < len(market_data) - self.confirmation_periods:
                # Check if price touches the level
                high = market_data['high'].iloc[i]
                low = market_data['low'].iloc[i]
                
                if low <= touch_zone_upper and high >= touch_zone_lower:
                    touches += 1
                    
                    # Analyze what happens after the touch
                    touch_result = await self._analyze_touch_outcome(
                        market_data, i, level_price, level_type
                    )
                    
                    if touch_result == "bounce":
                        bounces += 1
                    elif touch_result == "breakout":
                        breakouts += 1
                    elif touch_result == "false_breakout":
                        false_breakouts += 1
                
                i += 1
            
            return touches, bounces, breakouts, false_breakouts
            
        except Exception as e:
            self.logger.error(f"Failed to analyze level interactions: {e}")
            return 0, 0, 0, 0
    
    async def _analyze_touch_outcome(
        self,
        market_data: pd.DataFrame,
        touch_index: int,
        level_price: float,
        level_type: str
    ) -> str:
        """
        Analyze what happens after price touches an S/R level.
        
        Returns:
            "bounce", "breakout", "false_breakout", or "inconclusive"
        """
        try:
            # Look ahead for confirmation
            end_index = min(touch_index + self.confirmation_periods, len(market_data))
            future_data = market_data.iloc[touch_index:end_index]
            
            if level_type == "support":
                # For support levels, check if price bounces up or breaks down
                min_price = future_data['low'].min()
                max_price = future_data['high'].max()
                
                # Check for bounce (price moves up significantly)
                if max_price > level_price * (1 + self.bounce_threshold):
                    return "bounce"
                
                # Check for breakout (price moves down significantly)
                elif min_price < level_price * (1 - self.breakout_threshold):
                    # Check if it's a false breakout (price comes back)
                    if max_price > level_price * (1 + self.false_breakout_threshold):
                        return "false_breakout"
                    else:
                        return "breakout"
            
            elif level_type == "resistance":
                # For resistance levels, check if price bounces down or breaks up
                min_price = future_data['low'].min()
                max_price = future_data['high'].max()
                
                # Check for bounce (price moves down significantly)
                if min_price < level_price * (1 - self.bounce_threshold):
                    return "bounce"
                
                # Check for breakout (price moves up significantly)
                elif max_price > level_price * (1 + self.breakout_threshold):
                    # Check if it's a false breakout (price comes back)
                    if min_price < level_price * (1 - self.false_breakout_threshold):
                        return "false_breakout"
                    else:
                        return "breakout"
            
            return "inconclusive"
            
        except Exception as e:
            self.logger.error(f"Failed to analyze touch outcome: {e}")
            return "inconclusive"
    
    def _calculate_level_confidence(self, test: SRLevelTest) -> float:
        """Calculate confidence score for a level based on its performance."""
        try:
            confidence = 0.0
            
            # Base confidence from bounce rate
            if test.bounce_rate > 0.8:
                confidence += 0.4
            elif test.bounce_rate > 0.6:
                confidence += 0.3
            elif test.bounce_rate > 0.4:
                confidence += 0.2
            elif test.bounce_rate > 0.2:
                confidence += 0.1
            
            # Penalty for false breakouts
            if test.false_breakout_rate > 0.3:
                confidence -= 0.2
            elif test.false_breakout_rate > 0.2:
                confidence -= 0.1
            
            # Bonus for number of touches (more touches = more reliable)
            if test.touches >= 5:
                confidence += 0.2
            elif test.touches >= 3:
                confidence += 0.1
            
            # Bonus for level strength
            confidence += test.level_strength * 0.2
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate level confidence: {e}")
            return 0.5
    
    async def _calculate_overall_metrics(self, result: BacktestResult) -> None:
        """Calculate overall metrics from individual level tests."""
        try:
            if not result.level_tests:
                return
            
            # Calculate overall rates
            total_touches = sum(test.touches for test in result.level_tests)
            total_bounces = sum(test.bounces for test in result.level_tests)
            total_breakouts = sum(test.breakouts for test in result.level_tests)
            total_false_breakouts = sum(test.false_breakouts for test in result.level_tests)
            
            if total_touches > 0:
                result.overall_bounce_rate = total_bounces / total_touches
                result.overall_breakout_rate = total_breakouts / total_touches
                result.overall_false_breakout_rate = total_false_breakouts / total_touches
            
            # Calculate support vs resistance rates
            support_tests = [test for test in result.level_tests if test.level_type == "support"]
            resistance_tests = [test for test in result.level_tests if test.level_type == "resistance"]
            
            if support_tests:
                support_touches = sum(test.touches for test in support_tests)
                support_bounces = sum(test.bounces for test in support_tests)
                support_breakouts = sum(test.breakouts for test in support_tests)
                
                if support_touches > 0:
                    result.support_bounce_rate = support_bounces / support_touches
                    result.support_breakout_rate = support_breakouts / support_touches
            
            if resistance_tests:
                resistance_touches = sum(test.touches for test in resistance_tests)
                resistance_bounces = sum(test.bounces for test in resistance_tests)
                resistance_breakouts = sum(test.breakouts for test in resistance_tests)
                
                if resistance_touches > 0:
                    result.resistance_bounce_rate = resistance_bounces / resistance_touches
                    result.resistance_breakout_rate = resistance_breakouts / resistance_touches
            
            # Calculate average metrics
            result.avg_level_strength = np.mean([test.level_strength for test in result.level_tests])
            result.avg_confidence_score = np.mean([test.confidence_score for test in result.level_tests])
            
            # Calculate level detection accuracy
            if result.total_levels_tested > 0:
                result.level_detection_accuracy = result.successful_levels / result.total_levels_tested
            
        except Exception as e:
            self.logger.error(f"Failed to calculate overall metrics: {e}")
    
    async def _simulate_trading_strategy(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        result: BacktestResult
    ) -> None:
        """Simulate trading strategy based on S/R levels."""
        try:
            if not sr_levels:
                return
            
            # Initialize trading variables
            capital = 10000  # Starting capital
            position = 0  # Current position (0 = no position, 1 = long, -1 = short)
            entry_price = 0
            trades = []
            
            for i in range(len(market_data) - 1):
                current_price = market_data['close'].iloc[i]
                next_price = market_data['close'].iloc[i + 1]
                
                # Check for entry signals
                if position == 0:  # No position
                    signal = await self._check_entry_signal(
                        current_price, next_price, sr_levels, result.level_tests
                    )
                    
                    if signal == "long":
                        position = 1
                        entry_price = next_price
                        trades.append({
                            "type": "entry",
                            "side": "long",
                            "price": entry_price,
                            "timestamp": market_data.index[i + 1],
                            "capital": capital
                        })
                    
                    elif signal == "short":
                        position = -1
                        entry_price = next_price
                        trades.append({
                            "type": "entry",
                            "side": "short",
                            "price": entry_price,
                            "timestamp": market_data.index[i + 1],
                            "capital": capital
                        })
                
                # Check for exit signals
                elif position != 0:
                    exit_signal = await self._check_exit_signal(
                        position, entry_price, next_price
                    )
                    
                    if exit_signal:
                        # Calculate P&L
                        if position == 1:  # Long position
                            pnl = (next_price - entry_price) / entry_price
                        else:  # Short position
                            pnl = (entry_price - next_price) / entry_price
                        
                        # Update capital
                        capital *= (1 + pnl * self.position_size)
                        
                        trades.append({
                            "type": "exit",
                            "side": "long" if position == 1 else "short",
                            "entry_price": entry_price,
                            "exit_price": next_price,
                            "pnl": pnl,
                            "capital": capital,
                            "timestamp": market_data.index[i + 1]
                        })
                        
                        position = 0
                        entry_price = 0
            
            # Calculate trading metrics
            result.trade_signals = trades
            await self._calculate_trading_metrics(result, capital)
            
        except Exception as e:
            self.logger.error(f"Failed to simulate trading strategy: {e}")
    
    async def _check_entry_signal(
        self,
        current_price: float,
        next_price: float,
        sr_levels: List[Dict[str, Any]],
        level_tests: List[SRLevelTest]
    ) -> Optional[str]:
        """Check for entry signals based on S/R levels."""
        try:
            for level, test in zip(sr_levels, level_tests):
                level_price = level.get("price", 0)
                level_type = level.get("type", "support")
                
                # Only trade on high-confidence levels
                if test.confidence_score < 0.6:
                    continue
                
                # Check for support bounce (long signal)
                if level_type == "support":
                    if (current_price <= level_price * (1 + self.touch_threshold) and
                        next_price > current_price and
                        test.bounce_rate > 0.6):
                        return "long"
                
                # Check for resistance bounce (short signal)
                elif level_type == "resistance":
                    if (current_price >= level_price * (1 - self.touch_threshold) and
                        next_price < current_price and
                        test.bounce_rate > 0.6):
                        return "short"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to check entry signal: {e}")
            return None
    
    async def _check_exit_signal(
        self,
        position: int,
        entry_price: float,
        current_price: float
    ) -> bool:
        """Check for exit signals based on stop loss and take profit."""
        try:
            if position == 1:  # Long position
                # Check stop loss
                if current_price <= entry_price * (1 - self.stop_loss):
                    return True
                # Check take profit
                elif current_price >= entry_price * (1 + self.take_profit):
                    return True
            
            elif position == -1:  # Short position
                # Check stop loss
                if current_price >= entry_price * (1 + self.stop_loss):
                    return True
                # Check take profit
                elif current_price <= entry_price * (1 - self.take_profit):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check exit signal: {e}")
            return False
    
    async def _calculate_trading_metrics(self, result: BacktestResult, final_capital: float) -> None:
        """Calculate trading performance metrics."""
        try:
            if not result.trade_signals:
                return
            
            # Calculate returns
            initial_capital = 10000
            result.total_return = (final_capital - initial_capital) / initial_capital
            
            # Calculate win rate
            winning_trades = [t for t in result.trade_signals if t.get("pnl", 0) > 0]
            result.win_rate = len(winning_trades) / len(result.trade_signals) if result.trade_signals else 0
            
            # Calculate profit factor
            gross_profit = sum(t.get("pnl", 0) for t in winning_trades)
            losing_trades = [t for t in result.trade_signals if t.get("pnl", 0) < 0]
            gross_loss = abs(sum(t.get("pnl", 0) for t in losing_trades))
            
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate Sharpe ratio (simplified)
            returns = [t.get("pnl", 0) for t in result.trade_signals]
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                result.sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            # Calculate max drawdown
            capital_curve = [10000]  # Starting capital
            for trade in result.trade_signals:
                if trade["type"] == "exit":
                    capital_curve.append(trade["capital"])
            
            if len(capital_curve) > 1:
                peak = capital_curve[0]
                max_dd = 0
                for capital in capital_curve[1:]:
                    if capital > peak:
                        peak = capital
                    dd = (peak - capital) / peak
                    max_dd = max(max_dd, dd)
                result.max_drawdown = -max_dd
            
        except Exception as e:
            self.logger.error(f"Failed to calculate trading metrics: {e}")
    
    async def _calculate_performance_metrics(self, result: BacktestResult) -> None:
        """Calculate final performance metrics."""
        try:
            # Combine S/R validation metrics with trading metrics
            # This gives us a comprehensive view of S/R level effectiveness
            
            # S/R validation score (0-1)
            sr_validation_score = (
                result.overall_bounce_rate * 0.4 +
                (1 - result.overall_false_breakout_rate) * 0.3 +
                result.level_detection_accuracy * 0.3
            )
            
            # Trading performance score (0-1)
            trading_score = (
                max(0, result.win_rate) * 0.3 +
                min(1, result.profit_factor / 2) * 0.3 +
                max(0, result.sharpe_ratio / 2) * 0.2 +
                max(0, 1 + result.total_return) * 0.2
            )
            
            # Overall performance score
            overall_score = (sr_validation_score * 0.6 + trading_score * 0.4)
            
            # Store the overall score for optimization
            result.overall_performance_score = overall_score
            
            self.logger.info(f"📊 Performance Metrics:")
            self.logger.info(f"  S/R Validation Score: {sr_validation_score:.3f}")
            self.logger.info(f"  Trading Score: {trading_score:.3f}")
            self.logger.info(f"  Overall Score: {overall_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")


# Setup function for easy integration
async def setup_sr_backtesting_validator(config: Dict[str, Any]) -> Optional[SRBacktestingValidator]:
    """Setup S/R backtesting validator."""
    try:
        validator = SRBacktestingValidator(config)
        return validator
    except Exception as e:
        system_logger.error(f"Failed to setup S/R backtesting validator: {e}")
        return None