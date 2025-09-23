"""
Trading Signal Generation for NAS Models

This module provides actual trading capabilities:
- Buy/sell signal generation from regime detection
- Position sizing based on confidence and risk
- Entry/exit timing optimization
- Portfolio construction and rebalancing
- Risk management integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = "strong_buy"      # High confidence bullish
    BUY = "buy"                    # Moderate bullish
    NEUTRAL = "neutral"            # No clear direction
    SELL = "sell"                  # Moderate bearish
    STRONG_SELL = "strong_sell"    # High confidence bearish
    HOLD = "hold"                  # Maintain position

class PositionSize(Enum):
    """Position sizing categories."""
    NONE = 0.0      # No position
    SMALL = 0.25    # 25% of capital
    MEDIUM = 0.5    # 50% of capital
    LARGE = 0.75    # 75% of capital
    FULL = 1.0      # 100% of capital

@dataclass
class TradingConfig:
    """Configuration for trading signal generation."""
    regime_confidence_threshold: float = 0.7
    signal_strength_threshold: float = 0.8
    max_position_size: float = 1.0
    risk_per_trade: float = 0.02  # 2% of portfolio
    volatility_adjustment: bool = True
    market_condition_filter: bool = True
    stop_loss_multiplier: float = 2.0
    take_profit_multiplier: float = 3.0
    use_trailing_stops: bool = True
    min_holding_period: int = 1  # Minimum days to hold
    max_holding_period: int = 30  # Maximum days to hold

class TradingSignalGenerator:
    """
    Generates actual trading signals from regime detection.

    Converts regime predictions into actionable trading decisions.
    """

    def __init__(self, config: TradingConfig):
        """Initialize trading signal generator.

        Args:
            config: Trading configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Signal history for analysis
        self.signal_history = []
        self.position_history = []
        self.performance_metrics = {}

    def generate_signal(self, regime_prediction: np.ndarray,
                       market_data: pd.DataFrame,
                       current_position: float = 0.0) -> Dict[str, Any]:
        """
        Generate trading signal from regime prediction.

        Args:
            regime_prediction: Regime classification probabilities
            market_data: Current market data
            current_position: Current position size (-1 to 1)

        Returns:
            Trading signal and metadata
        """
        # Get regime confidence and type
        regime_confidence, regime_type = self._analyze_regime_prediction(regime_prediction)

        # Get market conditions
        market_condition = self._assess_market_condition(market_data)

        # Calculate signal strength
        signal_strength = self._calculate_signal_strength(
            regime_confidence, market_condition, current_position
        )

        # Determine signal type
        signal_type = self._determine_signal_type(
            regime_type, signal_strength, market_condition
        )

        # Calculate position size
        position_size = self._calculate_position_size(
            signal_type, signal_strength, current_position
        )

        # Generate stop loss and take profit levels
        stop_loss, take_profit = self._calculate_risk_levels(
            signal_type, market_data, position_size
        )

        signal = {
            'timestamp': pd.Timestamp.now(),
            'regime_prediction': regime_prediction,
            'regime_confidence': regime_confidence,
            'regime_type': regime_type,
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'position_size': position_size,
            'current_position': current_position,
            'target_position': position_size,
            'position_change': position_size - current_position,
            'market_condition': market_condition,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': abs(position_size - current_position) * self.config.risk_per_trade
        }

        # Log signal
        self.signal_history.append(signal)
        self.logger.info(f"📈 Generated {signal_type.value} signal with strength {signal_strength:.3f}")

        return signal

    def _analyze_regime_prediction(self, prediction: np.ndarray) -> Tuple[float, str]:
        """Analyze regime prediction to get confidence and type."""
        # Regime mapping (customize based on your regime classes)
        regime_classes = {
            0: "bullish_strong",
            1: "bullish_moderate",
            2: "bearish_strong",
            3: "bearish_moderate",
            4: "volatile",
            5: "sideways"
        }

        # Get highest probability regime
        max_prob = np.max(prediction)
        regime_idx = np.argmax(prediction)
        regime_type = regime_classes.get(regime_idx, "unknown")

        return max_prob, regime_type

    def _assess_market_condition(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Assess current market conditions."""
        try:
            close_prices = market_data['close'].values
            volume = market_data['volume'].values

            # Calculate recent metrics
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns[-20:])  # 20-period volatility
            volume_avg = np.mean(volume[-20:])  # Average volume
            trend = np.mean(returns[-10:])      # Recent trend

            # Normalize metrics
            volatility_score = min(volatility * 100, 1.0)  # Scale to 0-1
            volume_score = min(volume_avg / np.mean(volume), 2.0)  # Relative volume
            trend_score = np.tanh(trend * 100)  # Scale trend to -1 to 1

            return {
                'volatility': volatility_score,
                'volume': volume_score,
                'trend': trend_score,
                'overall': (volatility_score + volume_score + abs(trend_score)) / 3
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Market condition assessment failed: {e}")
            return {'volatility': 0.5, 'volume': 1.0, 'trend': 0.0, 'overall': 0.5}

    def _calculate_signal_strength(self, regime_confidence: float,
                                 market_condition: Dict[str, float],
                                 current_position: float) -> float:
        """Calculate overall signal strength."""
        # Base strength from regime confidence
        strength = regime_confidence

        # Adjust for market conditions
        if self.config.volatility_adjustment:
            # Reduce strength in high volatility
            volatility_factor = 1.0 - market_condition['volatility'] * 0.3
            strength *= volatility_factor

        # Adjust for market filter
        if self.config.market_condition_filter:
            # Boost strength in favorable conditions
            condition_factor = market_condition['overall']
            strength *= (0.7 + 0.3 * condition_factor)

        # Adjust based on current position
        if abs(current_position) > 0.5:
            # Reduce strength if heavily positioned
            position_factor = 1.0 - abs(current_position) * 0.2
            strength *= position_factor

        return np.clip(strength, 0.0, 1.0)

    def _determine_signal_type(self, regime_type: str,
                             signal_strength: float,
                             market_condition: Dict[str, float]) -> SignalType:
        """Determine the type of trading signal."""
        # Map regime to signal
        regime_to_signal = {
            "bullish_strong": SignalType.STRONG_BUY,
            "bullish_moderate": SignalType.BUY,
            "bearish_strong": SignalType.STRONG_SELL,
            "bearish_moderate": SignalType.SELL,
            "volatile": SignalType.NEUTRAL,
            "sideways": SignalType.HOLD
        }

        base_signal = regime_to_signal.get(regime_type, SignalType.NEUTRAL)

        # Adjust based on signal strength
        if signal_strength >= self.config.signal_strength_threshold:
            # High confidence - keep strong signal
            return base_signal
        elif signal_strength >= self.config.regime_confidence_threshold:
            # Moderate confidence - downgrade strong signals
            if base_signal == SignalType.STRONG_BUY:
                return SignalType.BUY
            elif base_signal == SignalType.STRONG_SELL:
                return SignalType.SELL
            else:
                return base_signal
        else:
            # Low confidence - neutral or hold
            return SignalType.NEUTRAL

    def _calculate_position_size(self, signal_type: SignalType,
                               signal_strength: float,
                               current_position: float) -> float:
        """Calculate position size based on signal and risk."""
        if signal_type == SignalType.STRONG_BUY:
            base_size = PositionSize.FULL.value
        elif signal_type == SignalType.BUY:
            base_size = PositionSize.LARGE.value
        elif signal_type == SignalType.SELL:
            base_size = PositionSize.LARGE.value * -1
        elif signal_type == SignalType.STRONG_SELL:
            base_size = PositionSize.FULL.value * -1
        elif signal_type == SignalType.NEUTRAL:
            base_size = 0.0
        elif signal_type == SignalType.HOLD:
            base_size = current_position  # Maintain current position
        else:
            base_size = 0.0

        # Scale by signal strength
        position_size = base_size * signal_strength

        # Apply risk limits
        position_size = np.clip(position_size, -self.config.max_position_size, self.config.max_position_size)

        return position_size

    def _calculate_risk_levels(self, signal_type: SignalType,
                             market_data: pd.DataFrame,
                             position_size: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        try:
            current_price = market_data['close'].iloc[-1]
            volatility = market_data['close'].pct_change().std()

            # ATR-based levels (simplified)
            atr = volatility * current_price

            if position_size > 0:  # Long position
                stop_loss = current_price - (atr * self.config.stop_loss_multiplier)
                take_profit = current_price + (atr * self.config.take_profit_multiplier)
            elif position_size < 0:  # Short position
                stop_loss = current_price + (atr * self.config.stop_loss_multiplier)
                take_profit = current_price - (atr * self.config.take_profit_multiplier)
            else:  # No position
                stop_loss = current_price
                take_profit = current_price

            return stop_loss, take_profit

        except Exception as e:
            self.logger.warning(f"⚠️ Risk level calculation failed: {e}")
            return 0.0, 0.0

class PortfolioManager:
    """
    Portfolio management and position sizing.

    Handles multi-asset portfolio construction and risk management.
    """

    def __init__(self, initial_capital: float = 100000.0):
        """Initialize portfolio manager.

        Args:
            initial_capital: Initial portfolio value
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> position info
        self.performance_history = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def update_position(self, symbol: str, signal: Dict[str, Any],
                       market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Update position based on trading signal.

        Args:
            symbol: Trading symbol
            signal: Trading signal from signal generator
            market_data: Current market data

        Returns:
            Position update results
        """
        current_price = market_data[symbol]['close'].iloc[-1]

        # Get current position
        current_position = self.positions.get(symbol, {
            'size': 0.0,
            'entry_price': 0.0,
            'value': 0.0
        })

        # Calculate new position
        target_size = signal['position_size']
        target_value = target_size * self.current_capital

        # Execute position change
        if target_size != current_position['size']:
            position_change = self._execute_trade(
                symbol, current_position['size'], target_size,
                current_price, signal
            )

            # Update position record
            self.positions[symbol] = {
                'size': target_size,
                'entry_price': current_price,
                'value': target_value,
                'signal': signal,
                'timestamp': signal['timestamp']
            }

        return {
            'symbol': symbol,
            'position_change': target_size - current_position['size'],
            'new_position': target_size,
            'entry_price': current_price
        }

    def _execute_trade(self, symbol: str, current_size: float,
                      target_size: float, current_price: float,
                      signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade to change position size."""
        size_change = target_size - current_size
        trade_value = size_change * current_price

        # Update capital
        self.current_capital -= trade_value  # Assuming no transaction costs for now

        return {
            'size_change': size_change,
            'trade_value': trade_value,
            'execution_price': current_price
        }

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        total_value = self.current_capital
        positions_value = 0.0

        for symbol, position in self.positions.items():
            positions_value += position['value']

        total_value += positions_value

        return {
            'total_value': total_value,
            'cash': self.current_capital,
            'positions_value': positions_value,
            'num_positions': len(self.positions),
            'return_pct': (total_value - self.initial_capital) / self.initial_capital * 100
        }

class RiskManager:
    """
    Risk management system.

    Monitors and controls portfolio risk exposure.
    """

    def __init__(self, max_risk_per_trade: float = 0.02, max_portfolio_risk: float = 0.06):
        """Initialize risk manager.

        Args:
            max_risk_per_trade: Maximum risk per trade (2%)
            max_portfolio_risk: Maximum portfolio risk (6%)
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.logger = logging.getLogger(self.__class__.__name__)

        self.risk_metrics = {
            'var_95': 0.0,  # 95% Value at Risk
            'var_99': 0.0,  # 99% Value at Risk
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

    def check_risk_limits(self, signal: Dict[str, Any], current_portfolio: Dict[str, Any]) -> bool:
        """Check if signal violates risk limits."""
        # Check individual trade risk
        trade_risk = abs(signal['position_change']) * self.max_risk_per_trade

        if trade_risk > self.max_risk_per_trade:
            self.logger.warning(f"⚠️ Trade risk too high: {trade_risk:.4f} > {self.max_risk_per_trade}")
            return False

        # Check portfolio risk
        current_risk = self._calculate_portfolio_risk(current_portfolio)
        new_risk = current_risk + trade_risk

        if new_risk > self.max_portfolio_risk:
            self.logger.warning(f"⚠️ Portfolio risk too high: {new_risk:.4f} > {self.max_portfolio_risk}")
            return False

        return True

    def _calculate_portfolio_risk(self, portfolio: Dict[str, Any]) -> float:
        """Calculate current portfolio risk."""
        # Simplified risk calculation
        # In practice, this would use more sophisticated methods
        positions_risk = 0.0

        for symbol, position in portfolio.items():
            if position['size'] != 0:
                # Risk based on position size and volatility
                positions_risk += abs(position['size']) * 0.01  # 1% base risk per position

        return positions_risk

class Backtester:
    """
    Backtesting framework for trading strategies.

    Evaluates trading performance on historical data.
    """

    def __init__(self, initial_capital: float = 100000.0):
        """Initialize backtester.

        Args:
            initial_capital: Initial portfolio value
        """
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(self.__class__.__name__)

        self.trades = []
        self.portfolio_history = []
        self.performance_metrics = {}

    def run_backtest(self, signal_generator: TradingSignalGenerator,
                    regime_model: nn.Module,
                    market_data: Dict[str, pd.DataFrame],
                    start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            signal_generator: Trading signal generator
            regime_model: Trained regime detection model
            market_data: Historical market data
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Backtest results
        """
        logger.info(f"📊 Running backtest from {start_date} to {end_date}")

        portfolio_manager = PortfolioManager(self.initial_capital)
        risk_manager = RiskManager()

        # Filter data by date range
        filtered_data = {}
        for symbol, data in market_data.items():
            mask = (data.index >= start_date) & (data.index <= end_date)
            filtered_data[symbol] = data[mask]

        # Run backtest day by day
        for date in pd.date_range(start_date, end_date, freq='D'):
            if date not in filtered_data[list(filtered_data.keys())[0]].index:
                continue

            # Get market data for this date
            current_data = {}
            for symbol, data in filtered_data.items():
                current_data[symbol] = data[data.index <= date].tail(100)  # Last 100 days

            # Get regime prediction
            regime_input = self._prepare_regime_input(current_data)
            with torch.no_grad():
                regime_prediction = regime_model(regime_input).numpy()

            # Get current position
            current_position = 0.0  # Simplified
            for symbol in current_data.keys():
                if symbol in portfolio_manager.positions:
                    current_position += portfolio_manager.positions[symbol]['size']

            # Generate signal
            signal = signal_generator.generate_signal(
                regime_prediction, current_data[list(current_data.keys())[0]], current_position
            )

            # Check risk limits
            if not risk_manager.check_risk_limits(signal, portfolio_manager.positions):
                continue

            # Update positions
            for symbol in current_data.keys():
                portfolio_manager.update_position(symbol, signal, current_data)

            # Log portfolio status
            portfolio_status = portfolio_manager.get_portfolio_status()
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_status['total_value'],
                'cash': portfolio_status['cash'],
                'positions_value': portfolio_status['positions_value']
            })

        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics()

        results = {
            'initial_capital': self.initial_capital,
            'final_capital': portfolio_status['total_value'],
            'total_return': portfolio_status['return_pct'],
            'performance_metrics': self.performance_metrics,
            'num_trades': len(self.trades),
            'portfolio_history': self.portfolio_history
        }

        self.logger.info(f"✅ Backtest completed")
        self.logger.info(f"📈 Total return: {portfolio_status['return_pct']:.2f}%")
        self.logger.info(f"📊 Sharpe ratio: {self.performance_metrics.get('sharpe_ratio', 0):.3f}")

        return results

    def _prepare_regime_input(self, market_data: Dict[str, pd.DataFrame]) -> torch.Tensor:
        """Prepare input for regime model."""
        # Simplified - take data from first symbol
        symbol = list(market_data.keys())[0]
        data = market_data[symbol]

        # Convert to tensor
        features = ['open', 'high', 'low', 'close', 'volume']
        feature_data = data[features].values[-100:]  # Last 100 time steps

        return torch.FloatTensor(feature_data).unsqueeze(0)

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_history:
            return {}

        portfolio_values = [entry['portfolio_value'] for entry in self.portfolio_history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Basic metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Win rate and other metrics
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        }

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values[0]
        max_dd = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

# Utility functions
def create_trading_system(regime_model: nn.Module,
                         trading_config: TradingConfig = None) -> Dict[str, Any]:
    """Create complete trading system."""
    if trading_config is None:
        trading_config = TradingConfig()

    signal_generator = TradingSignalGenerator(trading_config)
    portfolio_manager = PortfolioManager()
    risk_manager = RiskManager()
    backtester = Backtester()

    return {
        'regime_model': regime_model,
        'signal_generator': signal_generator,
        'portfolio_manager': portfolio_manager,
        'risk_manager': risk_manager,
        'backtester': backtester,
        'config': trading_config
    }

def run_live_trading(signal_generator: TradingSignalGenerator,
                    portfolio_manager: PortfolioManager,
                    market_data_stream: Any) -> None:
    """Run live trading system."""
    # Implementation for live trading
    pass

def generate_trading_report(signal_history: List[Dict],
                          portfolio_history: List[Dict]) -> Dict[str, Any]:
    """Generate comprehensive trading report."""
    # Implementation for trading report generation
    pass